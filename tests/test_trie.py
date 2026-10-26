import numpy as np
import pytest

from byte_agent.token_bytes import TokenizerSpec
from byte_agent.trie import EOT, N_BYTES, TokenTable, marginalize_it


def paper_vocab() -> dict[int, bytes]:
    return {
        0: b"T",
        1: b"iram",
        2: b"isu",
        3: b"isk",
        4: b"is",
        5: b"i",
        6: b"z",
    }


def paper_probs() -> np.ndarray:
    # p(isu)=0.5, p(isk)=0.125, p(is)=0.125, rest=0.25
    return np.array([0.0625, 0.0625, 0.5, 0.125, 0.125, 0.0625, 0.0625])


def make_table(vocab=None, specials=(), n_tokens=None) -> TokenTable:
    vocab = paper_vocab() if vocab is None else vocab
    return TokenTable.from_vocab_bytes(vocab, specials=specials, n_tokens=n_tokens)


def test_paper_tiramisu_example():
    table = make_table()
    p = paper_probs()
    logits = np.log(p).astype(np.float32)
    # Observed token is "isu"; row 2 is the distribution at prefix "is".
    rows = table.convert_row(logits, token_id=2)
    at_is = rows[2]
    assert rows.shape == (4, 260)
    assert at_is[N_BYTES] == pytest.approx(0.125 / 0.75, abs=1e-6)
    assert at_is[ord("u")] == pytest.approx(0.5 / 0.75, abs=1e-6)
    assert at_is[ord("k")] == pytest.approx(0.125 / 0.75, abs=1e-6)
    assert at_is.sum() == pytest.approx(1.0, abs=1e-5)

    # Terminal EOT for token "isu" is p(isu) / mass(isu) = 1.
    assert rows[3][EOT] == pytest.approx(1.0, abs=1e-6)

    mi = marginalize_it(at_is)
    assert mi[ord("u")] == pytest.approx(0.5 / 0.625, abs=1e-6)
    assert mi[ord("k")] == pytest.approx(0.125 / 0.625, abs=1e-6)
    assert mi[EOT] == 0.0
    assert mi[:N_BYTES].sum() == pytest.approx(1.0, abs=1e-6)


def test_rows_sum_to_one_and_align():
    table = make_table()
    logits = np.log(paper_probs()).astype(np.float32)
    for tid, raw in paper_vocab().items():
        rows = table.convert_row(logits, tid)
        assert rows.shape[0] == len(raw) + 1
        np.testing.assert_allclose(rows.sum(axis=1), 1.0, atol=1e-5)
        # The observed byte is always on the support at every content position.
        for d in range(len(raw)):
            assert rows[d][raw[d]] > 0


def test_special_and_padding_mass_go_to_eot():
    table = make_table(specials={7}, n_tokens=8)
    p = np.array([0.05, 0.05, 0.45, 0.1, 0.1, 0.05, 0.05, 0.15])
    logits = np.log(p).astype(np.float32)
    rows = table.convert_row(logits, token_id=2)
    # root: special token mass 0.15 is boundary mass.
    assert rows[0][EOT] == pytest.approx(0.15, abs=1e-6)
    assert rows[0].sum() == pytest.approx(1.0, abs=1e-5)


def test_duplicate_byte_strings_merge_eot_mass():
    vocab = {0: b"ab", 1: b"ab", 2: b"a", 3: b"z"}
    table = make_table(vocab=vocab, n_tokens=4)
    p = np.array([0.3, 0.2, 0.3, 0.2])
    logits = np.log(p).astype(np.float32)
    rows = table.convert_row(logits, token_id=0)
    # At prefix "a": mass = 0.3+0.2+0.3 = 0.8. EOT is the exact token "a"
    # (0.3); the "b" child holds both "ab" ids (0.5).
    at_a = rows[1]
    assert at_a[EOT] == pytest.approx(0.3 / 0.8, abs=1e-6)
    assert at_a[ord("b")] == pytest.approx(0.5 / 0.8, abs=1e-6)


def test_observed_path_requires_bytes():
    table = make_table(specials={7}, n_tokens=8)
    with pytest.raises(ValueError):
        table.convert_row(np.zeros(8, dtype=np.float32), token_id=7)


def test_n_logits_larger_than_tokenizer():
    table = make_table(n_tokens=8)
    # ids 7 is padding (no bytes); its mass is boundary mass at the root.
    logits = np.log(np.array([0.05, 0.05, 0.45, 0.1, 0.1, 0.05, 0.05, 0.15])).astype(np.float32)
    rows = table.convert_row(logits, token_id=2)
    assert rows[0][EOT] == pytest.approx(0.15, abs=1e-6)


def test_dfs_intervals_are_contiguous():
    table = make_table()
    for node in range(table.n_nodes):
        lo, hi = table.node_lo[node], table.node_hi[node]
        assert 0 <= lo <= hi <= len(table.order_ids)
        assert table.exact_lo[node] <= table.exact_hi[node]
        assert lo <= table.exact_lo[node] <= table.exact_hi[node] <= hi
