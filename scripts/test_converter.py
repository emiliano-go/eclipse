#!/usr/bin/env python3
"""Compare the C++ converter against a float64 Python reference."""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from byte_agent.shards import BOS, EOT
from byte_agent.trie import TokenTable

RAW_MAGIC = 0x474F4C42
TGT_MAGIC = 0x54415247
VOCAB = 248320


def softmax32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    x -= x.max()
    np.maximum(x, -60.0, out=x)
    e = np.exp(x)
    e /= e.sum()
    return e


def reference(table: TokenTable, logits: np.ndarray, tokens: np.ndarray):
    """Same algorithm as the C++ converter, but with a float64 cumulative sum."""
    syms, nexts, kinds, targets = [], [], [], []
    for i, tid in enumerate(tokens):
        p = softmax32(logits[i])
        p_token = float(p[tid])
        c = np.empty(len(table.order_ids) + 1, dtype=np.float64)
        c[0] = 0.0
        np.cumsum(p[table.order_ids], dtype=np.float64, out=c[1:])
        root_mass = c[table.node_hi[0]] - c[table.node_lo[0]]
        L = int(table.token_len[tid])
        raw = table.token_bytes_padded[tid][:L].astype(np.uint16)
        node = 0
        rows = []
        for d in range(L + 1):
            mass = c[table.node_hi[node]] - c[table.node_lo[node]]
            denom = 1.0 if d == 0 else max(mass, p_token)
            row = np.zeros(260, dtype=np.float32)
            kids = table.child[node]
            for b in range(256):
                u = int(kids[b])
                if u >= 0:
                    row[b] = (c[table.node_hi[u]] - c[table.node_lo[u]]) / denom
            if d < L:
                b = int(raw[d])
                row[b] = max(row[b], p_token / denom)
            exact = p_token if d == L else (c[table.exact_hi[node]] - c[table.exact_lo[node]])
            eot = exact / denom
            if d == 0:
                eot += 1.0 - root_mass
            row[EOT] = eot
            np.maximum(row, 0, out=row)
            row /= max(float(row.sum()), 1e-30)
            rows.append(row)
            if d < L:
                node = int(kids[raw[d]])
        rows = np.stack(rows)
        first = i == 0
        syms.extend([BOS if first else EOT] + raw.tolist())
        nexts.extend(raw.tolist() + [EOT])
        kinds.extend([2 if first else 1] + [0] * L)
        targets.append(rows)
    return (np.asarray(syms, dtype=np.uint16), np.asarray(nexts, dtype=np.uint16),
            np.asarray(kinds, dtype=np.uint8), np.concatenate(targets, axis=0))


def run(converter: str, trie: str, ids: np.ndarray, observed: np.ndarray,
        logits: np.ndarray, flags: int):
    n = len(observed)
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / "raw.bin"
        out = Path(d) / "out.bin"
        with raw.open("wb") as f:
            f.write(struct.pack("<6I", RAW_MAGIC, n, VOCAB, 0, flags, 0))
            f.write(np.ascontiguousarray(ids, dtype=np.int32).tobytes())
            f.write(np.ascontiguousarray(observed, dtype=np.int32).tobytes())
            if flags & 1:
                f.write(np.ascontiguousarray(logits, dtype=np.float32).tobytes())
            else:
                f.write(logits.astype(np.float16).tobytes())
        proc = subprocess.run([converter, "--trie", trie, "--threads", "8"],
                              stdin=raw.open("rb"), stdout=out.open("wb"),
                              stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
        b = out.read_bytes()
    h = struct.unpack_from("<8I", b, 0)
    assert h[0] == TGT_MAGIC and h[1] == 1 and h[6] == 260, h
    n_rows = h[5]
    off = 32
    syms = np.frombuffer(b, dtype=np.uint16, count=n_rows, offset=off).copy(); off += 2 * n_rows
    nexts = np.frombuffer(b, dtype=np.uint16, count=n_rows, offset=off).copy(); off += 2 * n_rows
    kinds = np.frombuffer(b, dtype=np.uint8, count=n_rows, offset=off).copy(); off += n_rows
    dids = np.frombuffer(b, dtype=np.int32, count=n_rows, offset=off).copy(); off += 4 * n_rows
    targets = np.frombuffer(b, dtype=np.float32, count=n_rows * 260, offset=off).reshape(n_rows, 260).copy()
    return syms, nexts, kinds, dids, targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="/data/ornith-byte/models/token_table.npz")
    ap.add_argument("--trie", default="/data/ornith-byte/models/trie.bin")
    ap.add_argument("--converter", default="/data/ornith-byte/tools/convert_targets")
    ap.add_argument("--ids", default="/tmp/opencode/audit32k.i32")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    table = TokenTable.load(args.table)
    corpus = np.fromfile(args.ids, dtype=np.int32)
    tokens = corpus[1000 : 1000 + args.tokens].astype(np.int32)
    assert not table.is_special[tokens].any()
    rng = np.random.default_rng(args.seed)
    logits = rng.normal(0.0, 3.0, size=(len(tokens), VOCAB)).astype(np.float32)
    ids = np.concatenate([[BOS], tokens[:-1]]).astype(np.int32)
    observed = tokens.astype(np.int32)

    for flags, name in ((1, "fp32"), (0, "fp16")):
        lg = logits if flags else logits.astype(np.float16).astype(np.float32)
        exp_syms, exp_nexts, exp_kinds, exp_targets = reference(table, lg, tokens)
        syms, nexts, kinds, dids, targets = run(args.converter, args.trie, ids, observed, logits, flags)
        ok = (np.array_equal(syms, exp_syms) and np.array_equal(nexts, exp_nexts)
              and np.array_equal(kinds, exp_kinds) and np.all(dids == 0))
        diff = float(np.abs(targets - exp_targets).max())
        sums = targets.sum(axis=1)
        # Production Python path, for the precision delta it still carries.
        prod_rows, _ = table.convert_batch(lg, tokens)
        prod = np.concatenate(prod_rows, axis=0)
        prod_diff = float(np.abs(targets - prod).max())
        print(f"{name}: arrays_ok={ok} rows={len(syms)} max_diff_vs_f64={diff:.3e} "
              f"max_diff_vs_py32={prod_diff:.3e} row_sum=[{sums.min():.6f},{sums.max():.6f}] "
              f"neg={int((targets<0).sum())}")
        if not ok or diff > 2e-4 or abs(sums - 1).max() > 1e-3 or (targets < 0).any():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
