import numpy as np

from byte_agent.extract import convert_batch
from byte_agent.shards import BOS, EOT, VOCAB
from byte_agent.trie import TokenTable

VOCAB_BYTES = {0: b"T", 1: b"iram", 2: b"isu", 3: b"isk", 4: b"is", 5: b"i", 6: b"z"}
PROBS = np.array([0.0625, 0.0625, 0.5, 0.125, 0.125, 0.0625, 0.0625], dtype=np.float32)


def test_extract_alignment_symbols_and_targets():
    table = TokenTable.from_vocab_bytes(VOCAB_BYTES)
    logits = np.stack([np.log(PROBS), np.log(PROBS)]).astype(np.float32)
    observed = np.array([2, 3], dtype=np.int64)  # isu, isk
    first = np.array([True, False])
    doc_ids = np.array([0, 0], dtype=np.int32)

    symbols, nexts, kinds, docs, targets = convert_batch(logits, observed, first, doc_ids, table)

    # BOS + i s u, then EOT + i s k
    np.testing.assert_array_equal(symbols, [BOS, ord("i"), ord("s"), ord("u"),
                                            EOT, ord("i"), ord("s"), ord("k")])
    np.testing.assert_array_equal(kinds, [2, 0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(
        nexts,
        [ord("i"), ord("s"), ord("u"), EOT, ord("i"), ord("s"), ord("k"), EOT],
    )
    np.testing.assert_array_equal(docs, [0] * 8)
    assert targets.shape == (8, VOCAB)
    np.testing.assert_allclose(targets.sum(axis=1), 1.0, atol=1e-5)

    # The first row of each token's block is the root distribution, identical
    # to the trie conversion of that token.
    np.testing.assert_allclose(targets[0], table.convert_row(logits[0], 2)[0], atol=1e-6)
    np.testing.assert_allclose(targets[4], table.convert_row(logits[1], 3)[0], atol=1e-6)
    # Last row of each token is the EOT row.
    assert targets[3][EOT] > 0
    assert targets[7][EOT] > 0
