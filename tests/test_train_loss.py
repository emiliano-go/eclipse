import math

import torch

from byte_agent.shards import EOT
from byte_agent.train import objective_loss


def test_eot_zero_for_identical_distribution():
    p = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float32)
    logits = p.log()
    nexts = torch.tensor([[1]])
    assert objective_loss(logits, p, nexts, "eot").item() == 0.0


def test_ce_matches_negative_log_probability():
    p = torch.tensor([[[0.2, 0.8]]], dtype=torch.float32)
    logits = p.log()
    nexts = torch.tensor([[1]])
    assert abs(objective_loss(logits, p, nexts, "ce").item() - (-math.log(0.8))) < 1e-6


def test_mi_shifts_at_token_boundary():
    t = torch.zeros((1, 2, 260), dtype=torch.float32)
    t[0, 0, EOT] = 1.0
    t[0, 1, ord("a")] = 1.0
    nexts = torch.tensor([[EOT, EOT]])
    p = torch.full((1, 2, 260), 0.2 / 259)
    p[0, 0, ord("a")] = 0.8
    p[0, 1, ord("a")] = 0.8
    logits = p.log()
    loss = objective_loss(logits, t, nexts, "mi").item()
    assert abs(loss - (-math.log(0.8))) < 1e-4


def test_mix_eot_ce_is_weighted_sum():
    p = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float32)
    logits = p.log()
    nexts = torch.tensor([[1]])
    eot = objective_loss(logits, p, nexts, "eot").item()
    ce = objective_loss(logits, p, nexts, "ce").item()
    mix = objective_loss(logits, p, nexts, "mix", alpha_kl=0.9, alpha_ce=0.1).item()
    assert abs(mix - (0.9 * eot + 0.1 * ce)) < 1e-6


def test_temperature_is_finite():
    p = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float32)
    logits = p.log()
    nexts = torch.tensor([[1]])
    for obj in ("eot", "mi", "mix"):
        loss = objective_loss(logits, p, nexts, obj, temperature=2.0, alpha_ce=0.1)
        assert torch.isfinite(loss).item()
