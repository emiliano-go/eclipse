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
TARGET_MAGIC = 0x54415247
TARGET_HEADER = struct.Struct("<8I")


class DumperError(RuntimeError):
    pass


@dataclass
class Batch:
    doc_id: int
    first_pos: int
    token_ids: np.ndarray  # int32
    observed: np.ndarray  # int32, -1 at document end
    logits: np.ndarray  # (n, vocab) float16 or float32


@dataclass
class TargetBatch:
    doc_id: int
    first_pos: int
    symbols: np.ndarray  # uint16
    nexts: np.ndarray  # uint16
    kinds: np.ndarray  # uint8
    doc_ids: np.ndarray  # int32
    targets: np.ndarray  # (n_rows, 260) float32


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
                raw_obs = _read_exact(proc.stdout, 4 * n)
                dtype = np.float32 if flags & 1 else np.float16
                raw_logits = _read_exact(proc.stdout, np.dtype(dtype).itemsize * n * vocab)
                if raw_ids is None or raw_obs is None or raw_logits is None:
                    raise DumperError("truncated record")
                yield Batch(
                    doc_id=doc_id,
                    first_pos=first_pos,
                    token_ids=np.frombuffer(raw_ids, dtype=np.int32).copy(),
                    observed=np.frombuffer(raw_obs, dtype=np.int32).copy(),
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


    def stream_targets(
        self,
        ids: np.ndarray,
        doc_starts: np.ndarray,
        trie: str | Path,
        converter: str | Path,
        threads: int = 8,
    ) -> Iterator[TargetBatch]:
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        temps: list[str] = []
        with tempfile.NamedTemporaryFile(suffix=".i32", delete=False) as tf:
            ids_path = tf.name
            temps.append(ids_path)
            ids.tofile(tf)
        with tempfile.NamedTemporaryFile(suffix=".i32", delete=False) as tf:
            starts_path = tf.name
            temps.append(starts_path)
            np.ascontiguousarray(doc_starts, dtype=np.int32).tofile(tf)

        dumper_cmd = [
            self.binary, "-m", self.model, "--ids", ids_path,
            "--doc-starts", starts_path, "--ctx", str(self.ctx),
            "--batch", str(self.batch), "--ngl", str(self.ngl),
        ]
        dumper = subprocess.Popen(dumper_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert dumper.stdout is not None and dumper.stderr is not None
        conv = subprocess.Popen(
            [str(converter), "--trie", str(trie), "--threads", str(threads)],
            stdin=dumper.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        dumper.stdout.close()
        assert conv.stdout is not None and conv.stderr is not None
        err_d: list[bytes] = []
        err_c: list[bytes] = []
        td = threading.Thread(target=_drain, args=(dumper.stderr.fileno(), err_d), daemon=True)
        tc = threading.Thread(target=_drain, args=(conv.stderr.fileno(), err_c), daemon=True)
        td.start()
        tc.start()
        try:
            while True:
                header = _read_exact(conv.stdout, TARGET_HEADER.size)
                if header is None:
                    break
                magic, version, doc_id, first_pos, npos, n_rows, vocab, flags = TARGET_HEADER.unpack(header)
                if magic != TARGET_MAGIC or version != 1 or vocab != 260 or flags != 0:
                    raise DumperError(f"bad target header {header!r}")
                raw_sym = _read_exact(conv.stdout, 2 * n_rows)
                raw_next = _read_exact(conv.stdout, 2 * n_rows)
                raw_kind = _read_exact(conv.stdout, n_rows)
                raw_doc = _read_exact(conv.stdout, 4 * n_rows)
                raw_t = _read_exact(conv.stdout, 4 * n_rows * 260)
                if None in (raw_sym, raw_next, raw_kind, raw_doc, raw_t):
                    raise DumperError("truncated target record")
                yield TargetBatch(
                    doc_id=doc_id,
                    first_pos=first_pos,
                    symbols=np.frombuffer(raw_sym, dtype=np.uint16).copy(),
                    nexts=np.frombuffer(raw_next, dtype=np.uint16).copy(),
                    kinds=np.frombuffer(raw_kind, dtype=np.uint8).copy(),
                    doc_ids=np.frombuffer(raw_doc, dtype=np.int32).copy(),
                    targets=np.frombuffer(raw_t, dtype=np.float32).reshape(n_rows, 260).copy(),
                )
        finally:
            conv.stdout.close()
            conv.wait()
            dumper.wait()
            td.join(timeout=2)
            tc.join(timeout=2)
            for path in temps:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        if conv.returncode != 0:
            tail = b"".join(err_c)[-4000:].decode("utf-8", "replace")
            raise DumperError(f"converter exited {conv.returncode}: {tail}")
        if dumper.returncode != 0:
            tail = b"".join(err_d)[-4000:].decode("utf-8", "replace")
            raise DumperError(f"dumper exited {dumper.returncode}: {tail}")


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
