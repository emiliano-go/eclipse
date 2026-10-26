#!/usr/bin/env python3
"""Structural validation of written EOT shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from byte_agent.shards import BOS, EOT, VOCAB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sample", type=int, default=200_000, help="0 = all positions")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    total = manifest["n_positions"]
    rng = np.random.default_rng(0)

    counts = {"byte": 0, "eot": 0, "bos": 0, "next_byte": 0, "next_eot": 0}
    bad_sum = 0
    bad_observed = 0
    nonfinite = 0
    negative = 0
    checked = 0
    prev_doc = -1
    doc_order = True

    for name in manifest["shards"]:
        d = root / name
        m = json.loads((d / "manifest.json").read_text())
        n = m["n_positions"]
        symbols = np.memmap(d / "symbols.u16", dtype=np.uint16, mode="r", shape=(n,))
        kinds = np.memmap(d / "kinds.u8", dtype=np.uint8, mode="r", shape=(n,))
        docs = np.memmap(d / "doc_ids.i32", dtype=np.int32, mode="r", shape=(n,))
        nexts = np.memmap(d / "next_symbols.u16", dtype=np.uint16, mode="r", shape=(n,))
        targets = np.memmap(d / "targets.f16", dtype=np.float16, mode="r", shape=(n, VOCAB))

        k = kinds[:]
        nx = nexts[:]
        counts["byte"] += int((k == 0).sum())
        counts["eot"] += int((k == 1).sum())
        counts["bos"] += int((k == 2).sum())
        counts["next_byte"] += int((nx < 256).sum())
        counts["next_eot"] += int((nx == EOT).sum())
        if np.any(docs[1:] < docs[:-1]):
            doc_order = False
        if prev_doc >= 0 and docs[0] < prev_doc:
            doc_order = False
        prev_doc = int(docs[-1])

        if args.sample and args.sample < n:
            idx = np.sort(rng.choice(n, size=args.sample, replace=False))
        else:
            idx = np.arange(n)
        for start in range(0, len(idx), 20_000):
            chunk = idx[start : start + 20_000]
            t = np.asarray(targets[chunk], dtype=np.float32)
            nonfinite += int((~np.isfinite(t)).any(axis=1).sum())
            negative += int((t < 0).sum())
            bad_sum += int((np.abs(t.sum(axis=1) - 1.0) > 1e-2).sum())
            nxt = np.asarray(nexts[chunk], dtype=np.int64)
            rows = np.arange(len(chunk))
            observed = t[rows, nxt]
            bad_observed += int((observed <= 0).sum())
            checked += len(chunk)

        del symbols, kinds, docs, nexts, targets

    ok = (
        nonfinite == 0
        and negative == 0
        and bad_sum == 0
        and bad_observed == 0
        and doc_order
        and counts == manifest["counts"]
    )
    report = {
        "positions": total,
        "checked": checked,
        "counts": counts,
        "manifest_counts": manifest["counts"],
        "nonfinite_rows": nonfinite,
        "negative_values": negative,
        "bad_sum_rows": bad_sum,
        "missing_observed_symbol_mass": bad_observed,
        "doc_ids_non_decreasing": doc_order,
        "ok": ok,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
