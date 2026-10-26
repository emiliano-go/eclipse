"""Byte trie and token-distribution -> byte/EOT target conversion.

The vocabulary is laid out in DFS order, so every prefix subtree is a
contiguous interval of token ids. That turns each prefix mass into two
prefix-sum lookups instead of a full vocabulary scan.

EOT targets are exact: at each node along the observed token's byte path,
the next-byte distribution is the mass of each child subtree divided by the
node's mass, and the terminal node receives p(token) / mass(prefix) on the
EOT symbol. Marginalize-It is the same row with EOT zeroed and the 256 byte
entries renormalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .token_bytes import TokenizerSpec

N_BYTES = 256
EOT = 256
BOS = 257
EOS = 258
PAD = 259
VOCAB_SIZE = 260


def _softmax_fp32(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32, copy=False)
    x = x - x.max(axis=1, keepdims=True)
    # fp16 logit dumps can have gaps > 100 to the observed token. exp() would
    # underflow that token's prefix mass to zero and make the target NaN.
    # A -60 floor keeps the tail finite (~1e-26) without affecting useful mass.
    np.maximum(x, -60.0, out=x)
    e = np.exp(x, out=x)
    e /= e.sum(axis=1, keepdims=True)
    return e


@dataclass
class TokenTable:
    n_tokens: int
    n_logits: int
    n_nodes: int
    order_ids: np.ndarray  # DFS position -> token id
    dfs_index: np.ndarray  # token id -> DFS position (-1 for specials)
    node_lo: np.ndarray
    node_hi: np.ndarray
    exact_lo: np.ndarray
    exact_hi: np.ndarray
    child: np.ndarray  # (n_nodes, 256) int32, -1 = absent
    token_node: np.ndarray  # (n_tokens,) int32, -1 = special
    token_len: np.ndarray  # (n_tokens,) int16
    token_bytes_padded: np.ndarray  # (n_tokens, max_len) int16, -1 pad
    is_special: np.ndarray
    max_token_len: int

    # -- construction -----------------------------------------------------

    @classmethod
    def build(cls, spec: TokenizerSpec, n_logits: int | None = None) -> "TokenTable":
        n_tokens = spec.n_tokens
        if n_logits is None:
            n_logits = n_tokens
        if n_logits < n_tokens:
            raise ValueError("n_logits must be >= tokenizer size")

        children: list[dict[int, int]] = [dict()]
        terminal: list[list[int]] = [[]]
        token_node = np.full(n_tokens, -1, dtype=np.int32)
        max_len = 0

        for tid, raw in enumerate(spec.token_bytes):
            if raw is None:
                continue
            node = 0
            for b in raw:
                nxt = children[node].get(b)
                if nxt is None:
                    nxt = len(children)
                    children[node][b] = nxt
                    children.append({})
                    terminal.append([])
                node = nxt
            terminal[node].append(tid)
            token_node[tid] = node
            max_len = max(max_len, len(raw))

        n_nodes = len(children)
        order: list[int] = []
        node_lo = np.zeros(n_nodes, dtype=np.int32)
        node_hi = np.zeros(n_nodes, dtype=np.int32)
        exact_lo = np.zeros(n_nodes, dtype=np.int32)
        exact_hi = np.zeros(n_nodes, dtype=np.int32)

        def dfs(v: int) -> None:
            node_lo[v] = len(order)
            exact_lo[v] = len(order)
            order.extend(terminal[v])
            exact_hi[v] = len(order)
            for b in sorted(children[v]):
                dfs(children[v][b])
            node_hi[v] = len(order)

        dfs(0)

        order_ids = np.asarray(order, dtype=np.int32)
        dfs_index = np.full(n_tokens, -1, dtype=np.int32)
        dfs_index[order_ids] = np.arange(len(order_ids), dtype=np.int32)

        child = np.full((n_nodes, N_BYTES), -1, dtype=np.int32)
        for v, kids in enumerate(children):
            for b, u in kids.items():
                child[v, b] = u

        token_len = np.zeros(n_tokens, dtype=np.int16)
        token_bytes_padded = np.full((n_tokens, max_len), -1, dtype=np.int16)
        for tid, raw in enumerate(spec.token_bytes):
            if raw is None:
                continue
            token_len[tid] = len(raw)
            token_bytes_padded[tid, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)

        return cls(
            n_tokens=n_tokens,
            n_logits=n_logits,
            n_nodes=n_nodes,
            order_ids=order_ids,
            dfs_index=dfs_index,
            node_lo=node_lo,
            node_hi=node_hi,
            exact_lo=exact_lo,
            exact_hi=exact_hi,
            child=child,
            token_node=token_node,
            token_len=token_len,
            token_bytes_padded=token_bytes_padded,
            is_special=np.asarray(spec.is_special, dtype=bool),
            max_token_len=max_len,
        )

    @classmethod
    def from_vocab_bytes(
        cls,
        vocab: dict[int, bytes],
        specials: Iterable[int] = (),
        n_tokens: int | None = None,
        n_logits: int | None = None,
    ) -> "TokenTable":
        specials = set(specials)
        size = n_tokens or (max([*vocab, *specials]) + 1)
        token_bytes: list[bytes | None] = [None] * size
        for tid, raw in vocab.items():
            token_bytes[tid] = raw
        from .token_bytes import TokenizerSpec

        spec = TokenizerSpec(
            token_bytes=token_bytes,
            token_text=[None] * size,
            is_special=[i in specials for i in range(size)],
        )
        return cls.build(spec, n_logits=n_logits or size)

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            n_tokens=self.n_tokens,
            n_logits=self.n_logits,
            n_nodes=self.n_nodes,
            order_ids=self.order_ids,
            dfs_index=self.dfs_index,
            node_lo=self.node_lo,
            node_hi=self.node_hi,
            exact_lo=self.exact_lo,
            exact_hi=self.exact_hi,
            child=self.child,
            token_node=self.token_node,
            token_len=self.token_len,
            token_bytes_padded=self.token_bytes_padded,
            is_special=self.is_special,
            max_token_len=self.max_token_len,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TokenTable":
        z = np.load(path, allow_pickle=False)
        kwargs = {k: z[k] for k in z.files}
        for k in ("n_tokens", "n_logits", "n_nodes", "max_token_len"):
            kwargs[k] = int(kwargs[k])
        kwargs["is_special"] = kwargs["is_special"].astype(bool)
        return cls(**kwargs)

    # -- conversion -------------------------------------------------------

    def convert_batch(
        self,
        logits: np.ndarray,
        token_ids: Sequence[int],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Convert teacher logits to byte/EOT rows along observed tokens.

        Args:
            logits: (N, n_logits) raw teacher logits.
            token_ids: N teacher token ids (non-special).

        Returns:
            rows: list of (L_i + 1, 260) float32 targets.
            lengths: (N,) int array of row counts.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 2 or logits.shape[0] != len(token_ids):
            raise ValueError("logits must be (len(token_ids), n_logits)")
        if logits.shape[1] < self.n_tokens:
            raise ValueError("logits narrower than tokenizer vocabulary")

        ids = np.asarray(token_ids, dtype=np.int64)
        if self.is_special[ids].any():
            raise ValueError("observed token ids must be non-special")
        lens = self.token_len[ids].astype(np.int64)
        if (lens == 0).any():
            raise ValueError("observed token has no bytes")

        n = len(ids)
        rows = np.arange(n)
        p = _softmax_fp32(logits)
        p_token = p[rows, ids]
        # Tokens beyond the tokenizer table (padding ids) are boundary mass.
        p_dfs = p[:, self.order_ids]
        c = np.empty((n, p_dfs.shape[1] + 1), dtype=np.float32)
        c[:, 0] = 0.0
        np.cumsum(p_dfs, axis=1, out=c[:, 1:])
        root_mass = c[:, self.node_hi[0]] - c[:, self.node_lo[0]]
        special_mass = 1.0 - root_mass

        max_len = int(lens.max())
        targets = np.zeros((n, max_len + 1, VOCAB_SIZE), dtype=np.float32)
        node = np.zeros(n, dtype=np.int32)
        bytes_padded = self.token_bytes_padded[ids]

        for d in range(max_len + 1):
            active = lens >= d
            if not active.any():
                break
            sel = rows[active]
            nodes = node[active]
            pt = p_token[sel]
            mass = c[sel, self.node_hi[nodes]] - c[sel, self.node_lo[nodes]]
            # A rare observed token's prefix mass can vanish in the float32
            # cumulative difference. Its own probability is a valid floor.
            denom = np.ones(len(sel), dtype=np.float32) if d == 0 else np.maximum(mass, pt)

            kids = self.child[nodes]
            safe = np.maximum(kids, 0)
            cmass = c[sel[:, None], self.node_hi[safe]] - c[sel[:, None], self.node_lo[safe]]
            cmass[kids < 0] = 0.0
            np.maximum(cmass, 0.0, out=cmass)

            cont_mask = lens[sel] > d
            if cont_mask.any():
                ci = np.where(cont_mask)[0]
                b = bytes_padded[sel[ci], d].astype(np.int64)
                cmass[ci, b] = np.maximum(cmass[ci, b], pt[ci])

            exact = c[sel, self.exact_hi[nodes]] - c[sel, self.exact_lo[nodes]]
            eot = exact / denom
            term = lens[sel] == d
            if term.any():
                # At the terminal node the exact mass is the observed token.
                eot[term] = pt[term] / denom[term]
            if d == 0:
                eot += special_mass[sel]
            np.maximum(eot, 0.0, out=eot)

            row = np.zeros((len(sel), VOCAB_SIZE), dtype=np.float32)
            row[:, :N_BYTES] = cmass / denom[:, None]
            row[:, EOT] = eot
            np.maximum(row, 0.0, out=row)
            row /= np.maximum(row.sum(axis=1, keepdims=True), 1e-30)
            targets[active, d] = row

            if cont_mask.any():
                ci = np.where(cont_mask)[0]
                b = bytes_padded[sel[ci], d].astype(np.int64)
                nxt = self.child[node[sel[ci]], b]
                if (nxt < 0).any():
                    raise RuntimeError("observed token path missing from trie")
                node[sel[ci]] = nxt

        out = [targets[i, : lens[i] + 1] for i in range(n)]
        return out, lens + 1

    def convert_row(self, logits: np.ndarray, token_id: int) -> np.ndarray:
        rows, _ = self.convert_batch(logits[None, :], [token_id])
        return rows[0]


def marginalize_it(target: np.ndarray) -> np.ndarray:
    """Derive Marginalize-It from a stored EOT row: drop EOT, renormalize."""
    byte_mass = target[:N_BYTES].astype(np.float32)
    total = byte_mass.sum()
    out = np.zeros_like(target, dtype=np.float32)
    if total > 0:
        out[:N_BYTES] = byte_mass / total
    return out
