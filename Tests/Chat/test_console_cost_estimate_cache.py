"""``TokenEstimateCache`` + ``build_cost_snapshot(estimate_cache=...)`` (task-15451).

The cache exists so the Console cost chip stops re-tokenizing an unchanged
transcript on every sync tick. Its whole safety argument is that a HIT is
verified against the estimate's full signature -- ``(model, provider, role,
content)`` -- so the cache key can only affect the hit rate, never the answer.
These tests pin both halves: the misses that must still happen (edited row,
model switch, key collision), and the hits that must (unchanged rows, and a
rebuilt-but-equal string such as the staged-evidence pseudo-row).
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.Chat import console_cost_tracker as tracker
from tldw_chatbook.Chat.console_cost_tracker import (
    TokenEstimateCache,
    build_cost_snapshot,
    token_estimate_signature,
)

PROVIDER = "anthropic"
MODEL = "claude-sonnet-4-6"


def _msg(message_id, content, role="user"):
    return SimpleNamespace(id=message_id, content=content, usage=None, role=role)


def _transcript(rows=6):
    return [
        _msg(f"m{index}", f"row {index}: " + ("lorem ipsum dolor sit amet. " * 20))
        for index in range(rows)
    ]


@pytest.fixture()
def estimator_spy(monkeypatch):
    spy = Mock(wraps=tracker._estimate_tokens_locally)
    monkeypatch.setattr(tracker, "_estimate_tokens_locally", spy)
    return spy


def _snapshot(messages, cache, model=MODEL):
    return build_cost_snapshot(
        messages, provider=PROVIDER, model=model, estimate_cache=cache
    )


# --- the cache must not change the answer ----------------------------------


def test_cached_and_uncached_snapshots_agree():
    messages = _transcript()

    uncached = build_cost_snapshot(messages, provider=PROVIDER, model=MODEL)
    cached = _snapshot(messages, TokenEstimateCache())

    assert cached == uncached


def test_default_call_still_estimates_every_row(estimator_spy):
    """No cache passed -> byte-identical to the pre-task behavior."""
    messages = _transcript(rows=4)

    build_cost_snapshot(messages, provider=PROVIDER, model=MODEL)
    build_cost_snapshot(messages, provider=PROVIDER, model=MODEL)

    assert estimator_spy.call_count == 8


# --- the hits ---------------------------------------------------------------


def test_repeat_pass_over_unchanged_rows_estimates_nothing(estimator_spy):
    messages = _transcript()
    cache = TokenEstimateCache()

    first = _snapshot(messages, cache)
    assert estimator_spy.call_count == len(messages)

    second = _snapshot(messages, cache)

    assert estimator_spy.call_count == len(messages)
    assert second == first


def test_a_fresh_snapshot_of_the_same_rows_still_hits(estimator_spy):
    """The store hands back new dataclass copies every pass
    (``messages_for_session`` -> ``dataclasses.replace``), so a hit must not
    depend on the ROW objects being the same objects."""
    messages = _transcript()
    cache = TokenEstimateCache()
    _snapshot(messages, cache)
    baseline = estimator_spy.call_count

    copies = [_msg(row.id, row.content, row.role) for row in messages]
    _snapshot(copies, cache)

    assert estimator_spy.call_count == baseline


def test_equal_but_rebuilt_content_still_hits(estimator_spy):
    """The staged-evidence pseudo-row is joined afresh on every pass, so its
    content is an equal-but-distinct string object each time."""
    cache = TokenEstimateCache()
    text = "corpus text " * 500

    _snapshot([_msg("staged", text)], cache)
    baseline = estimator_spy.call_count
    assert baseline == 1

    rebuilt = "".join(["corpus text "] * 500)
    assert rebuilt is not text and rebuilt == text
    _snapshot([_msg("staged", rebuilt)], cache)

    assert estimator_spy.call_count == baseline


# --- the misses that must still happen --------------------------------------


def test_edited_row_is_re_estimated_and_repriced(estimator_spy):
    messages = _transcript()
    cache = TokenEstimateCache()
    before = _snapshot(messages, cache)
    baseline = estimator_spy.call_count

    messages[2] = _msg(messages[2].id, "short")
    after = _snapshot(messages, cache)

    assert estimator_spy.call_count == baseline + 1
    assert after.total_tokens < before.total_tokens


def test_model_change_re_estimates_every_row(estimator_spy):
    messages = _transcript()
    cache = TokenEstimateCache()
    _snapshot(messages, cache)
    baseline = estimator_spy.call_count

    _snapshot(messages, cache, model="gpt-4")

    assert estimator_spy.call_count == baseline + len(messages)


def test_role_change_re_estimates_the_row(estimator_spy):
    """Role is part of what the estimator counts (chat-format framing), and
    it also decides input vs output pricing, so it must be part of the hit
    test rather than assumed constant per message id."""
    cache = TokenEstimateCache()
    _snapshot([_msg("m0", "text " * 50, role="user")], cache)
    baseline = estimator_spy.call_count

    _snapshot([_msg("m0", "text " * 50, role="assistant")], cache)

    assert estimator_spy.call_count == baseline + 1


def test_colliding_keys_cannot_serve_a_wrong_estimate():
    """Correctness must not rest on key uniqueness: two different rows
    forced onto ONE key must each get their own answer."""
    cache = TokenEstimateCache()
    short = token_estimate_signature((("user", "hi"),), MODEL, PROVIDER)
    long = token_estimate_signature((("user", "hi " * 500),), MODEL, PROVIDER)

    short_tokens = cache.estimate(
        "same-key",
        short,
        lambda: tracker._estimate_row_tokens("user", "hi", MODEL, PROVIDER),
    )
    long_tokens = cache.estimate(
        "same-key",
        long,
        lambda: tracker._estimate_row_tokens("user", "hi " * 500, MODEL, PROVIDER),
    )
    short_again = cache.estimate(
        "same-key",
        short,
        lambda: tracker._estimate_row_tokens("user", "hi", MODEL, PROVIDER),
    )

    assert long_tokens > short_tokens
    assert short_again == short_tokens


# --- bounded memory ---------------------------------------------------------


def test_cache_is_bounded_and_evicts_least_recently_used():
    cache = TokenEstimateCache(max_entries=2)
    signature = token_estimate_signature((("user", "hi"),), MODEL, PROVIDER)

    for key in ("a", "b"):
        cache.estimate(key, signature, lambda: 1)
    cache.estimate("a", signature, lambda: 1)  # refreshes "a", so "b" is oldest
    cache.estimate("c", signature, lambda: 1)

    assert len(cache) == 2
    # Probing mutates recency, so check the two survivors one at a time:
    # "a" (refreshed) is still there, "b" (least recently used) is gone.
    misses: list[str] = []
    cache.estimate("a", signature, lambda: misses.append("a") or 1)
    assert misses == []
    cache.estimate("b", signature, lambda: misses.append("b") or 1)
    assert misses == ["b"]


def test_clear_drops_every_entry():
    cache = TokenEstimateCache()
    _snapshot(_transcript(rows=3), cache)
    assert len(cache) == 3

    cache.clear()

    assert len(cache) == 0
