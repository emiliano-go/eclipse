"""On-disk EOT target shards.

Each position stores:
  symbols.u16  student input symbol (0..255 byte, 256 EOT, 257 BOS, ...)
  targets.f16  (positions, 260) next-symbol distribution
  kinds.u8     0 byte, 1 EOT, 2 BOS
  doc_ids.i32  source document id

Files are preallocated memmaps and truncated at finalize, so shard sizes are
exact and the reader can map them without a copy.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

VOCAB = 260
EOT = 256
BOS = 257
EOS = 258
PAD = 259


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ShardWriter:
    def __init__(
        self,
        root: str | Path,
        shard_positions: int = 250_000,
        vocab: int = VOCAB,
        meta: dict | None = None,
        checksum: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cap = shard_positions
        self.vocab = vocab
        self.meta = dict(meta or {})
        self.checksum = checksum
        self.shard_index = 0
        self.total = 0
        self.shards: list[dict] = []
        self._open()

    def _open(self) -> None:
        d = self.root / f"shard-{self.shard_index:05d}"
        d.mkdir(parents=True, exist_ok=True)
        self.dir = d
        self.n = 0
        self.targets = np.memmap(d / "targets.f16", dtype=np.float16, mode="w+", shape=(self.cap, self.vocab))
        self.symbols = np.memmap(d / "symbols.u16", dtype=np.uint16, mode="w+", shape=(self.cap,))
        self.kinds = np.memmap(d / "kinds.u8", dtype=np.uint8, mode="w+", shape=(self.cap,))
        self.doc_ids = np.memmap(d / "doc_ids.i32", dtype=np.int32, mode="w+", shape=(self.cap,))
        self.next_symbols = np.memmap(d / "next_symbols.u16", dtype=np.uint16, mode="w+", shape=(self.cap,))

    def append(
        self,
        symbols: np.ndarray,
        kinds: np.ndarray,
        doc_ids: np.ndarray,
        targets: np.ndarray,
        next_symbols: np.ndarray,
    ) -> None:
        n = len(symbols)
        if not (len(kinds) == n and len(doc_ids) == n and len(next_symbols) == n
                and targets.shape == (n, self.vocab)):
            raise ValueError("append arrays disagree")
        i = 0
        while i < n:
            room = self.cap - self.n
            if room == 0:
                self._finalize()
                self._open()
                continue
            take = min(room, n - i)
            self.targets[self.n : self.n + take] = targets[i : i + take]
            self.symbols[self.n : self.n + take] = symbols[i : i + take]
            self.kinds[self.n : self.n + take] = kinds[i : i + take]
            self.doc_ids[self.n : self.n + take] = doc_ids[i : i + take]
            self.next_symbols[self.n : self.n + take] = next_symbols[i : i + take]
            self.n += take
            self.total += take
            i += take

    def _finalize(self) -> None:
        if self.n == 0:
            for arr in (self.targets, self.symbols, self.kinds, self.doc_ids, self.next_symbols):
                del arr
            shutil.rmtree(self.dir, ignore_errors=True)
            return
        for arr in (self.targets, self.symbols, self.kinds, self.doc_ids, self.next_symbols):
            arr.flush()
        sizes = {
            "targets.f16": self.n * self.vocab * 2,
            "symbols.u16": self.n * 2,
            "kinds.u8": self.n,
            "doc_ids.i32": self.n * 4,
            "next_symbols.u16": self.n * 2,
        }
        del self.targets, self.symbols, self.kinds, self.doc_ids, self.next_symbols
        for name, size in sizes.items():
            with (self.dir / name).open("r+b") as f:
                f.truncate(size)
        kinds = np.memmap(self.dir / "kinds.u8", dtype=np.uint8, mode="r")
        nexts = np.memmap(self.dir / "next_symbols.u16", dtype=np.uint16, mode="r")
        counts = {
            "byte": int((kinds[: self.n] == 0).sum()),
            "eot": int((kinds[: self.n] == 1).sum()),
            "bos": int((kinds[: self.n] == 2).sum()),
            "next_byte": int((nexts[: self.n] < 256).sum()),
            "next_eot": int((nexts[: self.n] == EOT).sum()),
        }
        del kinds, nexts
        manifest = {
            "shard": self.dir.name,
            "n_positions": self.n,
            "vocab": self.vocab,
            "sizes": sizes,
            "counts": counts,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if self.checksum:
            manifest["sha256"] = {name: _sha256(self.dir / name) for name in sizes}
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.shards.append(manifest)
        self.shard_index += 1

    def close(self) -> dict:
        if self.n > 0:
            self._finalize()
        manifest = {
            "n_positions": self.total,
            "n_shards": len(self.shards),
            "shard_positions": self.cap,
            "vocab": self.vocab,
            "meta": self.meta,
            "shards": [s["shard"] for s in self.shards],
            "counts": {
                k: sum(s["counts"][k] for s in self.shards)
                for k in ("byte", "eot", "bos", "next_byte", "next_eot")
            },
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


class TargetDataset:
    """Read shards as fixed-length windows; windows never cross a shard."""

    def __init__(self, root: str | Path, seq_len: int = 1024) -> None:
        self.root = Path(root)
        root_manifest = json.loads((self.root / "manifest.json").read_text())
        self.seq_len = seq_len
        self.shards: list[dict] = []
        for name in root_manifest["shards"]:
            d = self.root / name
            m = json.loads((d / "manifest.json").read_text())
            m["path"] = d
            m["n"] = m["n_positions"]
            self.shards.append(m)
        self.windows: list[tuple[int, int]] = []
        for si, m in enumerate(self.shards):
            for start in range(0, m["n"] - seq_len + 1, seq_len):
                self.windows.append((si, start))
        self.n_positions = sum(m["n"] for m in self.shards)
        self._mm: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _maps(self, si: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if si not in self._mm:
            m = self.shards[si]
            d = m["path"]
            targets = np.memmap(d / "targets.f16", dtype=np.float16, mode="r", shape=(m["n"], m["vocab"]))
            symbols = np.memmap(d / "symbols.u16", dtype=np.uint16, mode="r", shape=(m["n"],))
            nexts = np.memmap(d / "next_symbols.u16", dtype=np.uint16, mode="r", shape=(m["n"],))
            docs = np.memmap(d / "doc_ids.i32", dtype=np.int32, mode="r", shape=(m["n"],))
            self._mm[si] = (symbols, targets, nexts, docs)
        return self._mm[si]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        si, start = self.windows[idx]
        symbols, targets, nexts, docs = self._maps(si)
        return (
            np.asarray(symbols[start : start + self.seq_len], dtype=np.int64),
            np.asarray(targets[start : start + self.seq_len], dtype=np.float32),
            np.asarray(nexts[start : start + self.seq_len], dtype=np.int64),
            np.asarray(docs[start : start + self.seq_len], dtype=np.int64),
        )
