import torch

from byte_agent.model import ByteLM, ModelConfig, parameter_count


def small() -> ByteLM:
    return ByteLM(ModelConfig(d_model=64, n_layers=2, n_heads=4, ffn_hidden=128, max_seq_len=64))


def test_parameter_count():
    n = parameter_count(ByteLM(ModelConfig()))
    assert 47_000_000 < n < 49_000_000, n


def test_causality():
    torch.manual_seed(0)
    model = small().eval()
    tokens = torch.randint(0, 256, (1, 16))
    a = model(tokens)
    b_tokens = tokens.clone()
    b_tokens[0, -1] = (b_tokens[0, -1] + 1) % 256
    b = model(b_tokens)
    torch.testing.assert_close(a[0, :-1], b[0, :-1], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(a[0, -1], b[0, -1])


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = small().eval()
    tokens = torch.randint(0, 256, (1, 20))
    full = model(tokens)
    prefix, past = model(tokens[:, :12], use_cache=True)
    torch.testing.assert_close(prefix, full[:, :12], atol=1e-5, rtol=1e-5)
    for i in range(12, 20):
        step, past = model(tokens[:, i : i + 1], past=past, use_cache=True)
        torch.testing.assert_close(step[:, 0], full[:, i], atol=1e-4, rtol=1e-4)


def test_save_load_roundtrip(tmp_path):
    model = small()
    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)
    clone = small()
    clone.load_state_dict(torch.load(path, weights_only=True))
    tokens = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(model(tokens), clone(tokens))


def test_generate_greedy_stats():
    torch.manual_seed(0)
    model = small().eval()
    out, stats = model.generate([257], max_new=4, greedy=True, return_stats=True, device="cpu")
    assert isinstance(out, bytes)
    assert stats["symbols"] <= 4
    assert stats["eot"] >= 0
    assert stats["bytes"] == len(out)
