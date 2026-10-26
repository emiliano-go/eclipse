"""Build a deduplicated, NFC-normalized byte corpus.

This is deliberately simple: local text/code roots in, one train/one val
stream out, plus a manifest. Near-duplicates are caught with a 16-hash
MinHash signature over byte shingles and verified by Jaccard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEXT_SUFFIXES = {
    ".txt", ".md", ".rst", ".py", ".c", ".h", ".cc", ".cpp", ".hpp",
    ".rs", ".go", ".java", ".js", ".ts", ".sh", ".bash", ".zsh",
    ".toml", ".yaml", ".yml", ".ini", ".cfg", ".sql", ".html", ".css",
}
SPECIAL_RE = re.compile(r"<\|[^|\n]{1,64}\|>|<tool_call>|</tool_call>|<tool_response>|</tool_response>|</think>|</think>")


@dataclass
class Doc:
    text: str
    source: str
    sha: str
    shingles: frozenset[int]


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = unicodedata.normalize("NFC", text)
    return SPECIAL_RE.sub("", text).strip()


def iter_files(roots: list[Path]):
    for root in roots:
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                yield path


def make_shingles(text: str, n: int = 8, cap: int = 512) -> frozenset[int]:
    data = text.encode("utf-8", "ignore")
    if len(data) < n:
        return frozenset()
    step = max(1, (len(data) - n) // cap)
    return frozenset(zlib.crc32(data[i : i + n]) for i in range(0, len(data) - n + 1, step))


_MULT = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x85EBCA77C2B2AE63] * 4, dtype=np.uint64)


def signature(shingles: frozenset[int]) -> np.ndarray:
    if not shingles:
        return np.zeros(16, dtype=np.uint64)
    x = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
    return np.min(x[:, None] * _MULT[None, :], axis=0)


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NearDupIndex:
    def __init__(self, bands: int = 4):
        self.bands = bands
        self.buckets: dict[tuple[int, ...], list[int]] = {}

    def _keys(self, sig: np.ndarray) -> list[tuple[int, ...]]:
        w = 16 // self.bands
        return [tuple(int(v) for v in sig[i * w : (i + 1) * w]) for i in range(self.bands)]

    def candidate(self, sig: np.ndarray, shingles: frozenset[int], docs: list[Doc]) -> int | None:
        for key in self._keys(sig):
            for idx in self.buckets.get(key, ()):
                if jaccard(docs[idx].shingles, shingles) > 0.9:
                    return idx
        return None

    def add(self, sig: np.ndarray, idx: int) -> None:
        for key in self._keys(sig):
            self.buckets.setdefault(key, []).append(idx)


def build(roots: list[Path], target_bytes: int, val_fraction: float, seed: int = 1337) -> tuple[list[Doc], list[Doc], dict]:
    docs: list[Doc] = []
    seen: set[str] = set()
    index = NearDupIndex()
    rejected = {"exact": 0, "near": 0, "decode": 0, "short": 0}
    total = 0

    for path in iter_files(roots):
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            rejected["decode"] += 1
            continue
        text = normalize(raw)
        if len(text) < 200:
            rejected["short"] += 1
            continue
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sha in seen:
            rejected["exact"] += 1
            continue
        shingles = make_shingles(text)
        sig = signature(shingles)
        if index.candidate(sig, shingles, docs) is not None:
            rejected["near"] += 1
            continue
        doc = Doc(text=text, source=str(path), sha=sha, shingles=shingles)
        seen.add(sha)
        index.add(sig, len(docs))
        docs.append(doc)
        total += len(text.encode("utf-8"))
        if total >= target_bytes:
            break

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    docs = [docs[i] for i in order]
    n_val = max(1, int(len(docs) * val_fraction))
    val, train = docs[:n_val], docs[n_val:]

    stats = {
        "docs": len(docs),
        "train_docs": len(train),
        "val_docs": len(val),
        "bytes": total,
        "rejected": rejected,
    }
    return train, val, stats


def write_stream(path: Path, docs: list[Doc]) -> int:
    data = "\n\n".join(d.text for d in docs).encode("utf-8")
    path.write_bytes(data)
    return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", required=True, help="file or directory; repeatable")
    ap.add_argument("--target-bytes", type=int, default=20_000_000)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = [Path(s) for s in args.source]
    train, val, stats = build(roots, args.target_bytes, args.val_fraction, args.seed)

    stats["train_bytes"] = write_stream(out / "train.txt", train)
    stats["val_bytes"] = write_stream(out / "val.txt", val)
    (out / "train.bin").write_bytes((out / "train.txt").read_bytes())
    (out / "val.bin").write_bytes((out / "val.txt").read_bytes())
    (out / "index.json").write_text(json.dumps(
        [{"sha": d.sha, "source": d.source, "bytes": len(d.text.encode("utf-8")), "split": "val"}
         for d in val] +
        [{"sha": d.sha, "source": d.source, "bytes": len(d.text.encode("utf-8")), "split": "train"}
         for d in train], indent=2))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
