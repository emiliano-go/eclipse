import json

import numpy as np

from byte_agent.shards import BOS, EOT, VOCAB, ShardWriter, TargetDataset


def batch(n, offset, seed=0):
    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, 256, size=n).astype(np.uint16)
    symbols[0] = BOS
    kinds = np.zeros(n, dtype=np.uint8)
    kinds[0] = 2
    targets = rng.random((n, VOCAB), dtype=np.float32)
    targets /= targets.sum(axis=1, keepdims=True)
    doc_ids = np.full(n, offset, dtype=np.int32)
    nexts = rng.integers(0, 257, size=n, dtype=np.uint16)
    return symbols, kinds, doc_ids, targets, nexts


def test_writer_roundtrip_and_splitting(tmp_path):
    w = ShardWriter(tmp_path, shard_positions=4, checksum=True)
    s0, k0, d0, t0, n0 = batch(3, 0, 0)
    s1, k1, d1, t1, n1 = batch(3, 1, 1)
    w.append(s0, k0, d0, t0, n0)
    w.append(s1, k1, d1, t1, n1)
    manifest = w.close()

    assert manifest["n_positions"] == 6
    assert manifest["n_shards"] == 2
    assert manifest["counts"]["bos"] == 2
    assert manifest["counts"]["byte"] == 4
    assert manifest["counts"]["next_eot"] == int((np.concatenate([n0, n1]) == EOT).sum())
    assert (tmp_path / "shard-00000" / "targets.f16").stat().st_size == 4 * VOCAB * 2
    assert (tmp_path / "shard-00001" / "targets.f16").stat().st_size == 2 * VOCAB * 2
    assert json.loads((tmp_path / "manifest.json").read_text())["n_positions"] == 6

    ds = TargetDataset(tmp_path, seq_len=2)
    assert ds.n_positions == 6
    assert len(ds) == 3  # shard 0: starts 0,2; shard 1: start 0
    symbols, targets, nexts = ds[0]
    np.testing.assert_array_equal(symbols, s0[:2])
    np.testing.assert_array_equal(nexts, n0[:2])
    assert targets.shape == (2, VOCAB)
    np.testing.assert_allclose(targets.sum(axis=1), 1.0, atol=1e-4)


def test_append_rejects_mismatch(tmp_path):
    w = ShardWriter(tmp_path, shard_positions=4)
    s, k, d, t, n = batch(2, 0)
    try:
        w.append(s, k[:-1], d, t, n)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    w.close()
