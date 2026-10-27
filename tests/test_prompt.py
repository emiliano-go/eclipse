from byte_agent.shards import BOS, EOT
from byte_agent.token_bytes import prompt_symbols


def test_complete_tokens_get_eot():
    symbols, consumed = prompt_symbols([b"ab", b"cd"], 4)
    assert consumed == 4
    assert symbols == [BOS, ord("a"), ord("b"), EOT, ord("c"), ord("d"), EOT]


def test_cut_token_has_no_eot():
    symbols, consumed = prompt_symbols([b"ab", b"cd"], 3)
    assert consumed == 3
    assert symbols == [BOS, ord("a"), ord("b"), EOT, ord("c")]


def test_zero_budget_is_bos_only():
    symbols, consumed = prompt_symbols([b"ab"], 0)
    assert consumed == 0
    assert symbols == [BOS]
