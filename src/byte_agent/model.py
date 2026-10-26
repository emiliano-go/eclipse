"""Dense byte-level Transformer with RoPE, QK-norm and a KV cache."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shards import BOS, EOS, EOT, VOCAB


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB
    d_model: int = 576
    n_layers: int = 12
    n_heads: int = 9
    ffn_hidden: int = 1536
    max_seq_len: int = 1024
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[positions].to(x.dtype)[None, None, :, :]
        sin = self.sin[positions].to(x.dtype)[None, None, :, :]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps) if cfg.qk_norm else nn.Identity()
        self.rope = RoPE(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, seq, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rope(self.q_norm(q), positions)
        k = self.rope(self.k_norm(k), positions)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        causal = past is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        y = y.transpose(1, 2).reshape(b, seq, -1)
        return self.out(y), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, positions, past=None):
        h, kv = self.attn(self.n1(x), positions, past)
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x, kv


class ByteLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        # Residual output projections are scaled down for depth stability.
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(block.mlp.w2.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ):
        b, seq = tokens.shape
        past_len = past[0][0].shape[2] if past is not None else 0
        positions = torch.arange(past_len, past_len + seq, device=tokens.device)
        x = self.emb(tokens)
        new_past = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            p = past[i] if past is not None else None
            x, kv = block(x, positions, p)
            if use_cache:
                new_past.append(kv)
        x = self.norm(x)
        logits = F.linear(x, self.emb.weight if self.head is None else self.head.weight)
        if use_cache:
            return logits, new_past
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: list[int],
        max_new: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        device: str | torch.device = "cuda",
        greedy: bool = False,
        return_stats: bool = False,
    ):
        self.eval()
        tokens = torch.tensor([prompt], dtype=torch.long, device=device)
        logits, past = self.forward(tokens, use_cache=True)
        out = bytearray()
        stats = {"eot": 0, "symbols": 0}
        for _ in range(max_new):
            if greedy:
                nxt = int(torch.argmax(logits[0, -1]))
            else:
                probs = torch.softmax(logits[0, -1].float() / temperature, dim=-1)
                if top_k:
                    k = min(top_k, probs.numel())
                    vals, idx = torch.topk(probs, k)
                    probs = torch.zeros_like(probs).scatter_(0, idx, vals)
                    probs /= probs.sum()
                nxt = int(torch.multinomial(probs, 1))
            if nxt == EOS:
                break
            stats["symbols"] += 1
            if nxt == EOT:
                stats["eot"] += 1
            elif nxt < 256:
                out.append(nxt)
            tokens = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, past = self.forward(tokens, past=past, use_cache=True)
        stats["bytes"] = len(out)
        if return_stats:
            return bytes(out), stats
        return bytes(out)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
