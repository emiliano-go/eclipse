"""Stream Ornith logits into 260-symbol EOT target shards.

The teacher stream is [BOS, doc0 tokens, BOS, doc1 tokens, ...] and the
dumper resets memory at every document start. For each document token t_i
the logits at the preceding position predict t_i; conversion produces
len(bytes(t_i)) + 1 rows whose input symbols are [BOS|EOT] + bytes(t_i).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .shards import BOS, EOT, VOCAB, ShardWriter
from .teacher import TeacherDumper
from .trie import TokenTable

CONVERSION = "eot-dfs-v1"
BOS_TOKEN = 248044  # <|endoftext|> used as a document separator


def load_documents(corpus_dir: Path, split: str) -> list[str]:
    index = json.loads((corpus_dir / "index.json").read_text())
    meta = [d for d in index if d["split"] == split]
    data = (corpus_dir / f"{split}.bin").read_bytes()
    docs: list[str] = []
    off = 0
    for d in meta:
        n = d["bytes"]
        docs.append(data[off : off + n].decode("utf-8"))
        off += n
        if off < len(data):
            if data[off : off + 2] != b"\n\n":
                raise ValueError(f"document separator mismatch at byte {off}")
            off += 2
    if off != len(data):
        raise ValueError(f"index covers {off} of {len(data)} bytes")
    return docs


def convert_batch(
    logits: np.ndarray,
    observed: np.ndarray,
    first: np.ndarray,
    doc_ids: np.ndarray,
    table: TokenTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, lens = table.convert_batch(logits, observed)
    n = int(lens.sum())
    symbols = np.empty(n, dtype=np.uint16)
    nexts = np.empty(n, dtype=np.uint16)
    kinds = np.empty(n, dtype=np.uint8)
    out_docs = np.empty(n, dtype=np.int32)
    targets = np.empty((n, VOCAB), dtype=np.float16)

    pos = 0
    for i, tid in enumerate(observed):
        nrows = int(lens[i])
        nbytes = nrows - 1
        row = rows[i]
        is_first = bool(first[i])
        symbols[pos] = BOS if is_first else EOT
        kinds[pos] = 2 if is_first else 1
        out_docs[pos] = doc_ids[i]
        targets[pos] = row[0]
        pos += 1
        if nbytes:
            raw = table.token_bytes_padded[tid][:nbytes]
            symbols[pos : pos + nbytes] = raw
            nexts[pos - 1 : pos - 1 + nbytes] = raw
            kinds[pos : pos + nbytes] = 0
            out_docs[pos : pos + nbytes] = doc_ids[i]
            targets[pos : pos + nbytes] = row[1:nrows]
            pos += nbytes
        nexts[pos - 1] = EOT
    if pos != n:
        raise RuntimeError(f"alignment mismatch: {pos} != {n}")
    return symbols, nexts, kinds, out_docs, targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="/data/ornith-byte/corpus")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="/data/ornith-byte/models/ornith/tokenizer.json")
    ap.add_argument("--model", default="/data/ornith-byte/models/gguf/Ornith-1.5-9B-Q5_K_M.gguf")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--table", default="/data/ornith-byte/models/token_table.npz")
    ap.add_argument("--limit-tokens", type=int, default=0, help="0 = all documents")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = no limit")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-positions", type=int, default=250_000)
    ap.add_argument("--teacher-sha", default="")
    ap.add_argument("--cpp-targets", action=argparse.BooleanOptionalAction, default=True,
                    help="C++ converter (default); use --no-cpp-targets for the deprecated Python path")
    ap.add_argument("--trie", default="/data/ornith-byte/models/trie.bin")
    ap.add_argument("--converter", default="/data/ornith-byte/tools/convert_targets")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-checksum", action="store_true")
    args = ap.parse_args()
    if not args.cpp_targets:
        print("WARNING: using the deprecated Python converter; C++ is the production path.", flush=True)

    t_start = time.time()
    corpus_dir = Path(args.corpus_dir)
    docs = load_documents(corpus_dir, args.split)
    tok = Tokenizer.from_file(args.tokenizer)

    selected: list[np.ndarray] = []
    selected_texts: list[str] = []
    total_tokens = 0
    for text in docs:
        ids = np.asarray(tok.encode(text).ids, dtype=np.int32)
        if len(ids) == 0:
            continue
        selected.append(ids)
        selected_texts.append(text)
        total_tokens += len(ids)
        if args.limit_tokens and total_tokens >= args.limit_tokens:
            break
        if args.max_docs and len(selected) >= args.max_docs:
            break

    stream: list[int] = []
    doc_starts: list[int] = []
    off = 0
    for ids in selected:
        doc_starts.append(off)
        stream.append(BOS_TOKEN)
        stream.extend(int(x) for x in ids)
        off += 1 + len(ids)
    stream_arr = np.asarray(stream, dtype=np.int32)
    starts_arr = np.asarray(doc_starts, dtype=np.int32)

    expected_bytes = sum(len(t.encode("utf-8")) for t in selected_texts)
    n_tokens = sum(len(ids) for ids in selected)
    n_docs = len(selected)

    print(
        json.dumps(
            {
                "docs": n_docs,
                "tokens": n_tokens,
                "stream_positions": int(len(stream_arr)),
                "content_bytes": expected_bytes,
                "expected_symbols": expected_bytes + n_tokens,
                "pipeline": "cpp" if args.cpp_targets else "python",
            },
            indent=2,
        )
    )

    writer = ShardWriter(
        args.out,
        shard_positions=args.shard_positions,
        meta={
            "teacher_model": Path(args.model).name,
            "teacher_sha256": args.teacher_sha,
            "corpus": str(corpus_dir),
            "split": args.split,
            "conversion": "eot-dfs-v1-cpp" if args.cpp_targets else CONVERSION,
            "bos_token": BOS_TOKEN,
            "ctx": args.ctx,
            "batch": args.batch,
            "ngl": args.ngl,
            "limit_tokens": args.limit_tokens,
            "docs": n_docs,
            "tokens": n_tokens,
        },
        checksum=not args.no_checksum,
    )

    dumper = TeacherDumper(
        args.model,
        binary=args.binary,
        ctx=args.ctx,
        batch=args.batch,
        ngl=args.ngl,
    )

    rows_written = 0
    last_report = time.time()

    if args.cpp_targets:
        last_doc = -1
        for tb in dumper.stream_targets(
            stream_arr, starts_arr, args.trie, args.converter, args.threads
        ):
            writer.append(
                tb.symbols, tb.kinds, tb.doc_ids, tb.targets.astype(np.float16), tb.nexts
            )
            rows_written += len(tb.symbols)
            if tb.doc_id != last_doc and time.time() - last_report > 20:
                elapsed = time.time() - t_start
                print(f"\rdocs={tb.doc_id + 1}/{n_docs} rows={rows_written} elapsed={elapsed:.0f}s",
                      end="", flush=True)
                last_report = time.time()
                last_doc = tb.doc_id
    else:
        table = TokenTable.load(args.table)
        pool = ThreadPoolExecutor(max_workers=args.workers)
        pending: deque = deque()
        max_pending = args.workers * 2

        def flush_one() -> None:
            nonlocal rows_written
            symbols, nexts, kinds, doc_ids, targets = pending.popleft().result()
            writer.append(symbols, kinds, doc_ids, targets, nexts)
            rows_written += len(symbols)

        for batch in dumper.stream(stream_arr, starts_arr):
            d = batch.doc_id
            m = len(selected[d])
            p = batch.first_pos
            end = min(p + len(batch.token_ids), m)
            if p >= end:
                continue
            sub = end - p
            first = np.zeros(sub, dtype=bool)
            first[0] = p == 0
            doc_ids = np.full(sub, d, dtype=np.int32)
            fut = pool.submit(
                convert_batch,
                batch.logits[:sub],
                selected[d][p:end],
                first,
                doc_ids,
                table,
            )
            pending.append(fut)
            while len(pending) >= max_pending:
                flush_one()
            if time.time() - last_report > 20:
                elapsed = time.time() - t_start
                print(f"\rdocs={d + 1}/{n_docs} rows={rows_written} elapsed={elapsed:.0f}s",
                      end="", flush=True)
                last_report = time.time()

        while pending:
            flush_one()
        pool.shutdown(wait=True)

    manifest = writer.close()
    elapsed = time.time() - t_start
    counts = manifest["counts"]
    checks = {
        "positions": manifest["n_positions"] == expected_bytes + n_tokens,
        "bytes": counts["byte"] == expected_bytes,
        "eot": counts["eot"] == n_tokens - n_docs,
        "bos": counts["bos"] == n_docs,
        "next_byte": counts["next_byte"] == expected_bytes,
        "next_eot": counts["next_eot"] == n_tokens,
    }
    summary = {
        "docs": n_docs,
        "tokens": n_tokens,
        "content_bytes": expected_bytes,
        "positions": manifest["n_positions"],
        "counts": counts,
        "checks": checks,
        "elapsed_s": round(elapsed, 1),
        "teacher_tokens_per_s": round(n_tokens / elapsed, 1),
        "shards": manifest["n_shards"],
        "target_gb": round(manifest["n_positions"] * VOCAB * 2 / 1e9, 3),
        "pipeline": "cpp" if args.cpp_targets else "python",
    }
    (Path(args.out) / "extract_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
