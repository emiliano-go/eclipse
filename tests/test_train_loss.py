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
