#!/usr/bin/env python3
"""Evaluate a byte student on document-held-out targets and generations."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from byte_agent.extract import load_documents
from byte_agent.model import ByteLM, ModelConfig
from byte_agent.shards import BOS, EOT, TargetDataset
from byte_agent.token_bytes import load_tokenizer_spec, prompt_symbols
from tokenizers import Tokenizer


def source_category(source: str) -> str:
    if "/python3.14/" in source:
        return "python"
    if "/share/doc/" in source:
        return "docs"
    if "/share/licenses/" in source:
        return "licenses"
    return "other"


def doc_sources(corpus_dir: Path) -> dict[int, str]:
    index = json.loads((corpus_dir / "index.json").read_text())
    vals = [d for d in index if d["split"] == "val"]
    return {i: source_category(d["source"]) for i, d in enumerate(vals)}


@torch.no_grad()
def document_metrics(model: ByteLM, root: str, corpus_dir: Path, batch: int,
                     seq_len: int, device: str, max_windows: int = 0) -> dict:
    ds = TargetDataset(root, seq_len=seq_len)
    if max_windows:
        ds = Subset(ds, range(min(max_windows, len(ds))))
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    buckets: dict[int, dict] = {}
    for symbols, targets, nexts, docs in loader:
        symbols = symbols.to(device)
        targets = targets.to(device)
        nexts = nexts.to(device)
        docs = docs.to(device)
        logits = model(symbols)
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, nexts.unsqueeze(-1)).squeeze(-1)
        t = targets.float().clamp_min(0)
        kl = (torch.special.xlogy(t, t) - t * logp).sum(-1)
        byte_mask = nexts < 256
        top = (logits.argmax(-1) == nexts)
        for d in torch.unique(docs):
            m = docs == d
            key = int(d)
            b = buckets.setdefault(key, {"bits": 0.0, "eot_bits": 0.0, "n_bytes": 0,
                                         "kl": 0.0, "top1": 0, "n": 0})
            bm = m & byte_mask
            b["bits"] += float(nll[bm].sum())
            b["eot_bits"] += float(nll[m & ~byte_mask].sum())
            b["n_bytes"] += int(bm.sum())
            b["kl"] += float(kl[m].sum())
            b["top1"] += int(top[m].sum())
            b["n"] += int(m.sum())

    src = doc_sources(corpus_dir)
    per_doc = []
    domains: dict[str, dict] = {}
    for d, b in sorted(buckets.items()):
        n_bytes = max(b["n_bytes"], 1)
        rec = {
            "doc_id": d,
            "domain": src.get(d, "other"),
            "positions": b["n"],
            "bytes": b["n_bytes"],
            "bpb": b["bits"] / math.log(2) / n_bytes,
            "bpb_eot_inclusive": (b["bits"] + b["eot_bits"]) / math.log(2) / n_bytes,
            "teacher_kl": b["kl"] / max(b["n"], 1),
            "top1": b["top1"] / max(b["n"], 1),
        }
        per_doc.append(rec)
        agg = domains.setdefault(rec["domain"], {"bits": 0.0, "eot_bits": 0.0,
                                                 "n_bytes": 0, "kl": 0.0, "top1": 0, "n": 0})
        for k in ("bits", "eot_bits", "kl"):
            agg[k] += b[k]
        for k in ("n_bytes", "top1", "n"):
            agg[k] += b[k]
    per_domain = {}
    for name, b in domains.items():
        n_bytes = max(b["n_bytes"], 1)
        per_domain[name] = {
            "bytes": b["n_bytes"],
            "bpb": b["bits"] / math.log(2) / n_bytes,
            "bpb_eot_inclusive": (b["bits"] + b["eot_bits"]) / math.log(2) / n_bytes,
            "teacher_kl": b["kl"] / max(b["n"], 1),
            "top1": b["top1"] / max(b["n"], 1),
        }
    total = {k: sum(b[k] for b in buckets.values()) for k in ("bits", "eot_bits", "n_bytes", "kl", "top1", "n")}
    n_bytes = max(total["n_bytes"], 1)
    overall = {
        "positions": total["n"],
        "bytes": total["n_bytes"],
        "bpb": total["bits"] / math.log(2) / n_bytes,
        "bpb_eot_inclusive": (total["bits"] + total["eot_bits"]) / math.log(2) / n_bytes,
        "teacher_kl": total["kl"] / max(total["n"], 1),
        "top1": total["top1"] / max(total["n"], 1),
    }
    return {"overall": overall, "per_domain": per_domain, "per_doc": per_doc}


@torch.no_grad()
def generation_metrics(model: ByteLM, corpus_dir: Path, prompts: int, prompt_bytes: int,
                       max_new: int, temperature: float, top_k: int, device: str,
                       prompt_format: str, tok: Tokenizer | None,
                       spec) -> dict:
    texts = load_documents(corpus_dir, "val")
    rows = []
    for i, text in enumerate(texts[:prompts]):
        if prompt_format == "eot":
            if tok is None or spec is None:
                raise ValueError("EOT prompt format requires a tokenizer")
            chunks = [spec.token_bytes[t] for t in tok.encode(text).ids
                      if spec.token_bytes[t] is not None]
            full = b"".join(chunks)
            ids, prompt_len = prompt_symbols(chunks, prompt_bytes)
        else:
            full = text.encode("utf-8")
            prompt_len = min(prompt_bytes, len(full))
            ids = [BOS] + list(full[:prompt_len])
        target = full[prompt_len: prompt_len + max_new]
        t0 = time.time()
        sample, stats = model.generate(ids, max_new=max_new, temperature=temperature,
                                       top_k=top_k, device=device, greedy=False,
                                       return_stats=True)
        dt = max(time.time() - t0, 1e-9)
        try:
            text_out = sample.decode("utf-8")
            valid = True
        except UnicodeDecodeError:
            text_out = sample.decode("utf-8", "replace")
            valid = False
        greedy, _ = model.generate(ids, max_new=max_new, device=device, greedy=True,
                                   return_stats=True)
        n = min(len(greedy), len(target))
        matches = sum(int(a == b) for a, b in zip(greedy[:n], target[:n]))
        prefix = 0
        for a, b in zip(greedy, target):
            if a != b:
                break
            prefix += 1
        rows.append({
            "doc_id": i,
            "valid_utf8": valid,
            "eot_rate": stats["eot"] / max(stats["symbols"], 1),
            "generated_bytes": len(sample),
            "bytes_per_s": len(sample) / dt,
            "greedy_byte_accuracy": matches / max(n, 1),
            "greedy_prefix": prefix,
            "sample": text_out[:160],
        })
    return {
        "prompts": len(rows),
        "prompt_format": prompt_format,
        "valid_utf8_rate": sum(r["valid_utf8"] for r in rows) / max(len(rows), 1),
        "mean_eot_rate": sum(r["eot_rate"] for r in rows) / max(len(rows), 1),
        "mean_greedy_byte_accuracy": sum(r["greedy_byte_accuracy"] for r in rows) / max(len(rows), 1),
        "mean_greedy_prefix": sum(r["greedy_prefix"] for r in rows) / max(len(rows), 1),
        "mean_bytes_per_s": sum(r["bytes_per_s"] for r in rows) / max(len(rows), 1),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-targets", required=True)
    ap.add_argument("--corpus-dir", default="/data/ornith-byte/corpus")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--prompt-bytes", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--prompt-format", choices=["eot", "raw"], default="eot",
                    help="EOT boundaries (training format) or raw bytes")
    ap.add_argument("--tokenizer", default="/data/ornith-byte/models/ornith/tokenizer.json")
    ap.add_argument("--tokenizer-config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = args.device
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ByteLM(ModelConfig(max_seq_len=args.seq_len)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    docs = document_metrics(model, args.val_targets, Path(args.corpus_dir),
                            args.batch, args.seq_len, device, args.max_windows)
    use_tokenizer = bool(args.prompts and args.prompt_format == "eot")
    tok = Tokenizer.from_file(args.tokenizer) if use_tokenizer else None
    spec = load_tokenizer_spec(args.tokenizer, args.tokenizer_config) if use_tokenizer else None
    gen = generation_metrics(model, Path(args.corpus_dir), args.prompts, args.prompt_bytes,
                             args.max_new, args.temperature, args.top_k, device,
                             args.prompt_format, tok, spec) if args.prompts else None
    report = {"checkpoint": args.checkpoint, "documents": docs, "generation": gen}
    print(json.dumps({k: v for k, v in report.items() if k != "documents"}, indent=2))
    print(json.dumps(docs["overall"], indent=2))
    print(json.dumps(docs["per_domain"], indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
