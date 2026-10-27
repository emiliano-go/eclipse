"""Tokenizer inspection and token -> byte mapping.

Ornith uses a GPT-2 style byte-level BPE tokenizer with an NFC normalizer.
Every base token therefore has an unambiguous UTF-8 byte string. Added
special tokens do not; they are treated as zero-byte boundaries (EOT).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte <-> unicode table."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_DECODER = {ch: b for b, ch in bytes_to_unicode().items()}


def decode_token(token: str) -> bytes:
    """Map a ByteLevel BPE token string back to raw bytes."""
    try:
        return bytes(BYTE_DECODER[c] for c in token)
    except KeyError as exc:
        raise ValueError(f"token {token!r} is not ByteLevel-encoded") from exc


def prompt_symbols(chunks: list[bytes], prompt_bytes: int) -> tuple[list[int], int]:
    """Build the training-style prompt: BOS, bytes, EOT, bytes, EOT, ... .

    The model is trained on teacher-token boundaries, so a raw-byte prompt is
    out of distribution. Complete tokens get an EOT; a token cut by the
    prompt budget is included only up to the cut and gets no EOT. Returns the
    symbol list and the number of raw bytes consumed.
    """
    from .shards import BOS, EOT

    symbols = [BOS]
    consumed = 0
    for chunk in chunks:
        if consumed >= prompt_bytes:
            break
        if not chunk:
            continue
        take = min(len(chunk), prompt_bytes - consumed)
        symbols.extend(chunk[:take])
        consumed += take
        if take == len(chunk):
            symbols.append(EOT)
        else:
            break
    return symbols, consumed


@dataclass
class TokenizerSpec:
    token_bytes: list[bytes | None]  # indexed by id; None for specials
    token_text: list[str | None]
    is_special: list[bool]

    @property
    def n_tokens(self) -> int:
        return len(self.token_bytes)

    @property
    def n_special(self) -> int:
        return sum(self.is_special)


def load_tokenizer_spec(
    tokenizer_json: str | Path,
    tokenizer_config: str | Path | None = None,
) -> TokenizerSpec:
    """Read tokenizer.json and produce the id -> bytes table.

    Base vocabulary entries are ByteLevel-decoded. Added tokens marked
    special get no byte string. Non-special added tokens are encoded as
    literal UTF-8, which is how the tokenizer emits them.

    tokenizer_config.json carries a few extra special tokens (audio/TTS) that
    are absent from tokenizer.json but present in the model embedding. They
    are merged in so the table covers every id the teacher can emit.
    """
    data = json.loads(Path(tokenizer_json).read_text())
    vocab: dict[str, int] = data["model"]["vocab"]
    entries: dict[int, dict] = {a["id"]: a for a in data.get("added_tokens", [])}

    if tokenizer_config is not None:
        cfg = json.loads(Path(tokenizer_config).read_text())
        for sid, entry in cfg.get("added_tokens_decoder", {}).items():
            tid = int(sid)
            if tid not in entries:
                entries[tid] = {
                    "id": tid,
                    "content": entry["content"],
                    "special": bool(entry.get("special", False)),
                }

    size = max([*vocab.values(), *entries]) + 1
    token_bytes: list[bytes | None] = [None] * size
    token_text: list[str | None] = [None] * size
    is_special = [False] * size

    for text, tid in vocab.items():
        token_bytes[tid] = decode_token(text)
        token_text[tid] = text

    for tid, entry in entries.items():
        text = entry["content"]
        token_text[tid] = text
        if entry.get("special", False):
            is_special[tid] = True
            token_bytes[tid] = None
        else:
            token_bytes[tid] = text.encode("utf-8")

    missing = [i for i, b in enumerate(token_bytes) if b is None and not is_special[i]]
    if missing:
        raise ValueError(f"{len(missing)} non-special tokens have no bytes: {missing[:8]}")
    empty = [i for i, b in enumerate(token_bytes) if b is not None and len(b) == 0]
    if empty:
        raise ValueError(f"{len(empty)} non-special tokens have empty bytes: {empty[:8]}")

    return TokenizerSpec(token_bytes=token_bytes, token_text=token_text, is_special=is_special)


def audit(spec: TokenizerSpec) -> dict:
    """Structural checks and duplicate-byte groups."""
    groups: dict[bytes, list[int]] = {}
    for tid, raw in enumerate(spec.token_bytes):
        if raw is not None:
            groups.setdefault(raw, []).append(tid)

    duplicates = {raw: ids for raw, ids in groups.items() if len(ids) > 1}
    lengths = [len(raw) for raw in spec.token_bytes if raw is not None]
    return {
        "n_tokens": spec.n_tokens,
        "n_special": spec.n_special,
        "n_byte_tokens": len(lengths),
        "n_unique_bytes": len(groups),
        "duplicate_byte_groups": {raw.hex(): ids for raw, ids in duplicates.items()},
        "min_token_len": min(lengths),
        "max_token_len": max(lengths),
        "mean_token_len": sum(lengths) / len(lengths),
    }


def main() -> None:
    import argparse
    import json
    import unicodedata
    from pathlib import Path

    from tokenizers import Tokenizer

    from .trie import TokenTable

    parser = argparse.ArgumentParser(description="Audit the Ornith tokenizer and build the byte trie")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-config", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out-table", required=True)
    parser.add_argument("--out-audit", required=True)
    parser.add_argument("--n-logits", type=int, default=None)
    args = parser.parse_args()

    spec = load_tokenizer_spec(args.tokenizer, args.tokenizer_config)
    report = audit(spec)

    tok = Tokenizer.from_file(args.tokenizer)
    raw_text = Path(args.text).read_text(encoding="utf-8")
    normalized = unicodedata.normalize("NFC", raw_text)
    encoded = tok.encode(normalized)
    decoded = tok.decode(encoded.ids)
    report["roundtrip"] = {
        "input_chars": len(normalized),
        "n_tokens": len(encoded.ids),
        "decoded_chars": len(decoded),
        "exact": decoded == normalized,
        "nfc_idempotent": unicodedata.normalize("NFC", normalized) == normalized,
    }
    if not report["roundtrip"]["exact"]:
        for i, (a, b) in enumerate(zip(normalized, decoded)):
            if a != b:
                report["roundtrip"]["first_mismatch"] = {
                    "index": i,
                    "expected": repr(normalized[max(0, i - 20) : i + 20]),
                    "got": repr(decoded[max(0, i - 20) : i + 20]),
                }
                break

    Path(args.out_audit).write_text(json.dumps(report, indent=2))
    table = TokenTable.build(spec, n_logits=args.n_logits)
    table.save(args.out_table)
    print(json.dumps({k: v for k, v in report.items() if k != "duplicate_byte_groups"}, indent=2))
    print(f"nodes={table.n_nodes} dfs_tokens={len(table.order_ids)} max_len={table.max_token_len}")


if __name__ == "__main__":
    main()
