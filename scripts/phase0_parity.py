#!/usr/bin/env python3
"""Phase 0 gate: tokenizer round-trip and HF <-> llama.cpp token-id parity."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from byte_agent.teacher import TeacherDumper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--chars", type=int, default=200_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = Path(args.text).read_text(encoding="utf-8")[: args.chars]
    text = unicodedata.normalize("NFC", raw)

    tok = Tokenizer.from_file(args.tokenizer)
    hf_ids = np.asarray(tok.encode(text).ids, dtype=np.int32)
    decoded = tok.decode(hf_ids.tolist())

    dumper = TeacherDumper(args.model, binary=args.binary)
    llama_ids = dumper.tokenize(text)

    n = min(len(hf_ids), len(llama_ids))
    diff = np.nonzero(hf_ids[:n] != llama_ids[:n])[0]
    first = int(diff[0]) if len(diff) else None

    report = {
        "chars": len(text),
        "hf_tokens": int(len(hf_ids)),
        "llama_tokens": int(len(llama_ids)),
        "same_length": len(hf_ids) == len(llama_ids),
        "mismatches": int(len(diff)) + abs(len(hf_ids) - len(llama_ids)),
        "roundtrip_exact": decoded == text,
        "nfc_idempotent": unicodedata.normalize("NFC", text) == text,
        "first_mismatch": None,
    }
    if first is not None:
        lo = max(0, first - 3)
        hi = first + 4
        report["first_mismatch"] = {
            "index": first,
            "hf_ids": hf_ids[lo:hi].tolist(),
            "llama_ids": llama_ids[lo:hi].tolist(),
            "hf_text": tok.decode(hf_ids[lo:hi].tolist()),
            "llama_text": None,
        }

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    ok = report["same_length"] and report["mismatches"] == 0 and report["roundtrip_exact"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
