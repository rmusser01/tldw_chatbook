"""The UI suite's token accounting is the no-tokenizer tier, deterministically.

``Tests/UI/conftest.py``'s autouse ``_no_tiktoken_bpe_download`` fixture exists
to keep mounted Console tests off tiktoken's BPE download seam (TASK-21590).
Stubbing ``get_tiktoken_encoding`` alone does that, but it also puts the whole
suite on an accounting no install runs: ``estimate_tokens`` still enters
``count_tokens_tiktoken``, whose no-encoding fallback is a bare
``int(len(text) * 0.25)`` -- no CJK weighting, no headroom, no non-empty floor.

These tests pin the two properties that distinguish the intended tier from that
fallback, so the fixture cannot quietly drift back to it.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Utils import token_counter

from Tests.UI import conftest as ui_conftest


def test_the_ui_suite_runs_the_no_tokenizer_tier() -> None:
    """The tier flag itself, not just the encoding, must be off."""
    assert token_counter.TIKTOKEN_AVAILABLE is False
    assert token_counter.get_tiktoken_encoding("gpt-4") is None


def test_non_empty_text_never_estimates_to_zero_tokens() -> None:
    """``int(len(text) * 0.25)`` returns 0 for "hi"; the real tier returns 1.

    ``_chars_estimate`` carries an explicit ``max(1, ...)`` for exactly this
    case -- a UI suite budgeting a short message at zero tokens is measuring
    something no user has.
    """
    assert token_counter.estimate_tokens("hi", "gpt-4", "openai") == 1


def test_cjk_is_weighted_rather_than_counted_as_quarter_tokens() -> None:
    """The conservative floor must stay above the flat 0.25/char fallback.

    Repeated CJK measured 84 on the chars tier against 17 on the fallback and
    50 on the real tokenizer, so the fallback undercounts the tokenizer ~3x
    while the chars tier stays above it -- which is the direction a budget
    guard has to err in.
    """
    text = "这是一个测试。" * 10
    flat_fallback = int(len(text) * 0.25)

    estimate = token_counter.estimate_tokens(text, "gpt-4", "openai")

    assert estimate > flat_fallback


def test_the_fixture_clears_a_foreign_tier_estimate_on_both_sides(
    monkeypatch,
) -> None:
    """A value cached under a different tokenizer must not be served here.

    ``_ESTIMATE_CACHE`` is process-global and its key is ``(model, provider,
    len, hash(text))`` -- no tokenizer identity -- so an entry computed under
    the real tiktoken elsewhere in the session would be returned verbatim
    here, and one computed here would leak the other way. pytest-randomly
    shuffles the run order, so "Tests/Chat happens to run first" is not a
    defence.

    The fixture's own body is driven directly rather than relying on it having
    run for this test: an assertion about "the cache is empty at test start"
    cannot tell the setup clear from the teardown clear of the *previous*
    test, and would survive removing either.
    """
    fixture_body = ui_conftest._no_tiktoken_bpe_download._get_wrapped_function()
    text = "cache-tier-witness " * 4
    key = ("gpt-4", "openai", len(text), hash(text))

    token_counter._ESTIMATE_CACHE[key] = 999_999
    generator = fixture_body(monkeypatch)
    next(generator)
    after_setup = dict(token_counter._ESTIMATE_CACHE)
    served = token_counter.estimate_tokens(text, "gpt-4", "openai")

    token_counter._ESTIMATE_CACHE[key] = 999_999
    with pytest.raises(StopIteration):
        next(generator)
    after_teardown = dict(token_counter._ESTIMATE_CACHE)

    assert key not in after_setup
    assert served != 999_999
    assert key not in after_teardown
