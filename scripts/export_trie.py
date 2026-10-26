#!/usr/bin/env python3
"""Export the TokenTable arrays to a flat binary for the C++ converter."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from byte_agent.trie import TokenTable

MAGIC = 0x54524945
VERSION = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="/data/ornith-byte/models/token_table.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t = TokenTable.load(args.table)
    header = struct.pack(
        "<8I", MAGIC, VERSION, t.n_nodes, t.n_tokens, t.n_logits,
        t.max_token_len, len(t.order_ids), int(t.node_hi[0]),
    )
    arrays = [
        ("node_lo", t.node_lo, "<i4"),
        ("node_hi", t.node_hi, "<i4"),
        ("exact_lo", t.exact_lo, "<i4"),
        ("exact_hi", t.exact_hi, "<i4"),
        ("child", t.child, "<i4"),
        ("token_len", t.token_len, "<i4"),
        ("token_bytes", t.token_bytes_padded, "u1"),
        ("order_ids", t.order_ids, "<i4"),
        ("is_special", t.is_special, "u1"),
    ]
    out = Path(args.out)
    with out.open("wb") as f:
        f.write(header)
        for name, arr, dtype in arrays:
            a = np.ascontiguousarray(arr, dtype=dtype)
            f.write(a.tobytes())
    meta = {
        "magic": hex(MAGIC),
        "version": VERSION,
        "n_nodes": t.n_nodes,
        "n_tokens": t.n_tokens,
        "n_logits": t.n_logits,
        "max_token_len": t.max_token_len,
        "n_valid": len(t.order_ids),
        "root_hi": int(t.node_hi[0]),
        "bytes": out.stat().st_size,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
