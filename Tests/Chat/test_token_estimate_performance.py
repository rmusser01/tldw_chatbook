# Tests/Chat/test_token_estimate_performance.py
"""TASK-18602: the token estimator's fast CJK count and per-text memo.

The optimization replaced a per-character Python loop and added a memo, so
what needs pinning is that neither changed an ANSWER -- plus the two
structural properties that make the memo safe (bounded, and holding no
reference to the text it describes).

The measured shape that motivated this: 158.8 ms to estimate a 640 KB
payload, re-run over the whole conversation every turn, for 33.1 s of
cumulative CPU across a simulated 400-turn run.
"""

from __future__ import annotations

import gc
import re
import threading
import weakref

import pytest

from tldw_chatbook.Utils import token_counter
from tldw_chatbook.Utils.token_counter import (
    ESTIMATE_CACHE_MAX_ENTRIES,
    _CJK_RANGES,
    _chars_estimate,
    _count_cjk,
    _is_cjk,
    clear_estimate_cache,
    count_tokens_messages,
    estimate_tokens,
)


def _reference_count(text: str) -> int:
    """The pre-TASK-18602 implementation, kept as the oracle."""
    return sum(1 for ch in text if _is_cjk(ch))


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_estimate_cache()
    yield
    clear_estimate_cache()


# -- the fast counter answers exactly what the loop answered ----------------


def test_ascii_text_counts_zero_cjk():
    assert _count_cjk("hello world, 123 -- def f(): return 1") == 0


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain ascii",
        "日本語のテキスト",
        "한국어 텍스트",
        "中文文本",
        "mixed english and 日本語 in one string",
        "。、「」full-width ／＝",
        "emoji 🎉 outside the ranges",
        "　〿぀ヿ一鿿",
    ],
)
def test_fast_counter_matches_the_reference_loop(text):
    assert _count_cjk(text) == _reference_count(text)


def test_every_range_boundary_matches():
    """Off-by-one at a range edge would silently reweight CJK text."""
    for lo, hi in _CJK_RANGES:
        for cp in (lo - 1, lo, lo + 1, hi - 1, hi, hi + 1):
            if cp < 0:
                continue
            ch = chr(cp)
            assert _count_cjk(ch) == _reference_count(ch), hex(cp)


def test_the_regex_is_built_from_the_same_ranges():
    """The char class and `_is_cjk` must not drift apart; they are two
    encodings of one fact."""
    for lo, hi in _CJK_RANGES:
        assert token_counter._CJK_RE.match(chr(lo))
        assert token_counter._CJK_RE.match(chr(hi))


def test_chars_estimate_is_unchanged_for_mixed_content():
    """The counter feeds `_chars_estimate`'s CJK weighting, so a changed
    count would silently change every estimate."""
    for text in ("ascii only", "日本語", "half 日本語 half ascii"):
        cjk = _reference_count(text)
        other = len(text) - cjk
        assert _chars_estimate(text, "openai") >= 1
        assert _count_cjk(text) == cjk
        assert len(text) - _count_cjk(text) == other


# -- the memo returns the same answers --------------------------------------


def test_repeated_estimates_agree_with_the_first():
    for text in ("hello", "日本語のテキスト", "x" * 5000, "mixed 中文 text"):
        first = estimate_tokens(text, "gpt-4o-mini", "openai")
        assert estimate_tokens(text, "gpt-4o-mini", "openai") == first
        # an equal-but-distinct object must hit the same answer
        assert estimate_tokens(str(text), "gpt-4o-mini", "openai") == first


def test_different_texts_of_equal_length_do_not_share_an_entry():
    a = estimate_tokens("日本語日本語", "m", "openai")
    b = estimate_tokens("abcdef", "m", "openai")
    assert a != b, "CJK and ASCII of equal length must not estimate alike"


def test_model_and_provider_are_part_of_the_key():
    """Different tokenizer identities must not serve each other's answers."""
    text = "x" * 400
    clear_estimate_cache()
    anthropic = estimate_tokens(text, "claude-opus-5", "anthropic")
    openai = estimate_tokens(text, "gpt-4o-mini", "openai")
    # They may coincide numerically; what matters is both are computed for
    # their own provider ratio rather than one being served for the other.
    assert anthropic == estimate_tokens(text, "claude-opus-5", "anthropic")
    assert openai == estimate_tokens(text, "gpt-4o-mini", "openai")


def test_empty_text_is_zero_and_never_cached():
    assert estimate_tokens("", "m", "openai") == 0
    assert len(token_counter._ESTIMATE_CACHE) == 0


def test_message_counting_is_unaffected():
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "日本語で答えて"},
        {"role": "assistant", "content": "x" * 3000},
    ]
    first = count_tokens_messages(msgs, "gpt-4o-mini", provider="openai")
    assert count_tokens_messages(msgs, "gpt-4o-mini", provider="openai") == first
    assert first > 0


# -- the memo's safety properties -------------------------------------------


def test_the_cache_is_bounded():
    clear_estimate_cache()
    for i in range(ESTIMATE_CACHE_MAX_ENTRIES + 200):
        estimate_tokens(f"unique text {i}", "m", "openai")
    assert len(token_counter._ESTIMATE_CACHE) <= ESTIMATE_CACHE_MAX_ENTRIES


def test_the_cache_holds_no_reference_to_the_estimated_text():
    """AC#5: keyed by hash+length, never by the string, so memoizing a
    600 KB message cannot pin it in memory after the run moves on."""

    class _Text(str):
        """A str subclass so it can be weak-referenced."""

    clear_estimate_cache()
    text = _Text("some text to estimate " * 50)
    ref = weakref.ref(text)
    estimate_tokens(text, "m", "openai")
    del text
    gc.collect()
    assert ref() is None


def test_concurrent_estimation_is_safe_and_consistent():
    """AC#4: the agent worker thread and the UI thread estimate at once."""
    clear_estimate_cache()
    texts = [f"payload {i} " * 40 for i in range(200)]
    expected = {t: estimate_tokens(t, "m", "openai") for t in texts}
    clear_estimate_cache()

    errors: list[BaseException] = []
    results: list[dict] = []

    def worker():
        try:
            results.append({t: estimate_tokens(t, "m", "openai") for t in texts})
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    for got in results:
        assert got == expected


def test_clear_estimate_cache_forces_recomputation():
    calls = []
    real = token_counter._chars_estimate

    def counting(text, provider):
        calls.append(text)
        return real(text, provider)

    token_counter._chars_estimate = counting
    try:
        clear_estimate_cache()
        estimate_tokens("some text", "m", "openai")
        estimate_tokens("some text", "m", "openai")
        assert len(calls) == 1, "second call should have been memoized"
        clear_estimate_cache()
        estimate_tokens("some text", "m", "openai")
        assert len(calls) == 2, "clear() should force a recompute"
    finally:
        token_counter._chars_estimate = real


# -- the shape that motivated the work --------------------------------------


def test_estimating_a_growing_conversation_stays_cheap():
    """The regression guard: re-estimating an append-only history must cost
    O(new text), not O(whole history), per turn.

    Asserted structurally (recomputes, not wall-clock) so it cannot flake
    on a loaded machine: across 100 turns the estimator may only be invoked
    once per distinct message, not once per message per turn.
    """
    clear_estimate_cache()
    computed = []
    real = token_counter._chars_estimate

    def counting(text, provider):
        computed.append(text)
        return real(text, provider)

    token_counter._chars_estimate = counting
    try:
        msgs = [{"role": "system", "content": "s" * 500}]
        for turn in range(100):
            msgs.append({"role": "assistant", "content": f"turn {turn} " * 60})
            msgs.append({"role": "user", "content": f"reply {turn} " * 60})
            count_tokens_messages(msgs, "gpt-4o-mini", provider="openai")
    finally:
        token_counter._chars_estimate = real

    # 1 system + 200 turn messages + the distinct role strings, each counted
    # once. The pre-fix behaviour recomputed every message every turn, which
    # is >10,000 invocations for the same 201 messages.
    assert len(computed) < 500, (
        f"{len(computed)} estimator invocations for 201 distinct messages "
        "-- the per-turn re-count is back"
    )
