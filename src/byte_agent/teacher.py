"""Python driver for the llama.cpp logits dumper.

The dumper speaks a tiny binary protocol on stdout so raw logits never touch
disk. This module owns the subprocess, drains stderr, and yields batches.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

MAGIC = 0x474F4C42
HEADER = struct.Struct("<6I")


class DumperError(RuntimeError):
    pass


@dataclass
class Batch:
    doc_id: int
    first_pos: int
    token_ids: np.ndarray  # int32
    logits: np.ndarray  # (n, vocab) float16 or float32


def default_binary() -> Path:
    return Path("/data/ornith-byte/tools/llama.cpp/build/bin/dump_logits")


class TeacherDumper:
    def __init__(
        self,
        model: str | Path,
        binary: str | Path | None = None,
        ctx: int = 2048,
        batch: int = 256,
        ngl: int = 99,
        fp32: bool = False,
        limit: int | None = None,
    ) -> None:
        self.model = str(model)
        self.binary = str(binary or default_binary())
        self.ctx = ctx
        self.batch = batch
        self.ngl = ngl
        self.fp32 = fp32
        self.limit = limit

    def tokenize(self, text: str) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
            text_path = tf.name
            tf.write(text.encode("utf-8"))
        out_path = text_path + ".i32"
        try:
            proc = subprocess.run(
                [self.binary, "-m", self.model, "--tokenize", text_path, "--out", out_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                raise DumperError(proc.stderr.decode("utf-8", "replace"))
            return np.fromfile(out_path, dtype=np.int32)
        finally:
            for p in (text_path, out_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

    def stream(self, ids: np.ndarray, doc_starts: np.ndarray | None = None) -> Iterator[Batch]:
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        temps: list[str] = []
        with tempfile.NamedTemporaryFile(suffix=".i32", delete=False) as tf:
            ids_path = tf.name
            temps.append(ids_path)
            ids.tofile(tf)
        doc_starts_path = None
        if doc_starts is not None:
            with tempfile.NamedTemporaryFile(suffix=".i32", delete=False) as tf:
                doc_starts_path = tf.name
                temps.append(doc_starts_path)
                np.ascontiguousarray(doc_starts, dtype=np.int32).tofile(tf)

        cmd = [
            self.binary,
            "-m",
            self.model,
            "--ids",
            ids_path,
            "--ctx",
            str(self.ctx),
            "--batch",
            str(self.batch),
            "--ngl",
            str(self.ngl),
        ]
        if doc_starts_path is not None:
            cmd += ["--doc-starts", doc_starts_path]
        if self.fp32:
            cmd.append("--fp32")
        if self.limit is not None:
            cmd += ["--limit", str(self.limit)]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None and proc.stderr is not None
        err_chunks: list[bytes] = []
        drain = threading.Thread(target=_drain, args=(proc.stderr.fileno(), err_chunks), daemon=True)
        drain.start()

        try:
            while True:
                header = _read_exact(proc.stdout, HEADER.size)
                if header is None:
                    break
                magic, n, vocab, first_pos, flags, doc_id = HEADER.unpack(header)
                if magic != MAGIC:
                    raise DumperError(f"bad magic {magic:#x}")
                raw_ids = _read_exact(proc.stdout, 4 * n)
                dtype = np.float32 if flags & 1 else np.float16
                raw_logits = _read_exact(proc.stdout, np.dtype(dtype).itemsize * n * vocab)
                if raw_ids is None or raw_logits is None:
                    raise DumperError("truncated record")
                yield Batch(
                    doc_id=doc_id,
                    first_pos=first_pos,
                    token_ids=np.frombuffer(raw_ids, dtype=np.int32).copy(),
                    logits=np.frombuffer(raw_logits, dtype=dtype).reshape(n, vocab),
                )
        finally:
            proc.stdout.close()
            proc.wait()
            drain.join(timeout=2)
            for path in temps:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

        if proc.returncode != 0:
            tail = b"".join(err_chunks)[-4000:].decode("utf-8", "replace")
            raise DumperError(f"dumper exited {proc.returncode}: {tail}")


def _drain(fd: int, sink: list[bytes]) -> None:
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            return
        sink.append(chunk)


def _read_exact(stream, n: int) -> bytes | None:
    parts: list[bytes] = []
    remaining = n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == n:
                return None
            raise DumperError(f"unexpected EOF ({n - remaining}/{n} bytes)")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)
