"""Cost math + chip-state formatting for the Console cost chip (PR3).

House pattern: the state dataclass owns ALL label formatting; the widget
only renders (see ConsoleControlState). Never store dollars — compute at
display time from usage rows via the pricing catalog.
"""

from types import SimpleNamespace

from tldw_chatbook.Chat.console_cost_tracker import (
    ConsoleCacheState,
    ConsoleCostSnapshot,
    build_cost_snapshot,
    build_cost_state,
)
from tldw_chatbook.Chat.console_session_settings import _estimate_tokens_locally
from tldw_chatbook.Chat.provider_usage import ProviderUsage


def _msg(content="hi", usage=None, role="assistant"):
    return SimpleNamespace(content=content, usage=usage, role=role)


def test_snapshot_sums_priced_usage_rows():
    usage = ProviderUsage(
        uncached_input=1_000_000, output=1_000_000,
        provider="anthropic", model="claude-sonnet-4-6",
    )
    snap = build_cost_snapshot(
        [_msg(usage=usage)], provider="anthropic", model="claude-sonnet-4-6"
    )
    assert snap.pricing_known is True
    assert snap.has_estimated_entries is False
    assert snap.total_usd == 18.0  # $3 in + $15 out per MTok
    assert snap.total_tokens == 2_000_000


def test_rows_without_usage_fall_back_to_estimates():
    snap = build_cost_snapshot(
        [_msg(content="x" * 400, usage=None)],
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    assert snap.has_estimated_entries is True
    assert snap.total_tokens > 0


def test_unknown_model_yields_tokens_only():
    usage = ProviderUsage(
        uncached_input=100, provider="anthropic", model="mystery-9000"
    )
    snap = build_cost_snapshot(
        [_msg(usage=usage)], provider="anthropic", model="mystery-9000"
    )
    assert snap.pricing_known is False
    assert snap.total_usd is None
    assert snap.total_tokens == 100


def test_state_normal_warm():
    snap = ConsoleCostSnapshot(0.4821, 12000, True, False, 3)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.WARM, break_reason=None,
        projected_delta_usd=None, ttl_remaining_s=240.0, pricing_as_of="2026-08-02",
    )
    assert state.label == "$0.4821 ●"
    assert state.alert is False and state.cold is False
    assert "2026-08-02" in state.tooltip and "4:00" in state.tooltip


def test_state_alert_carries_delta_and_reason():
    snap = ConsoleCostSnapshot(0.48, 12000, True, False, 3)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.WARM, break_reason="system prompt changed",
        projected_delta_usd=0.13, ttl_remaining_s=120.0, pricing_as_of="2026-08-02",
    )
    assert state.label == "$0.48 ⚠ ~+$0.13"
    assert state.compact_label == "$0.48 ⚠"
    assert state.alert is True
    assert "system prompt changed" in state.tooltip


def test_state_alert_requires_warm_cache():
    snap = ConsoleCostSnapshot(0.48, 12000, True, False, 3)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.NONE, break_reason="system prompt changed",
        projected_delta_usd=0.13, ttl_remaining_s=None, pricing_as_of=None,
    )
    assert state.alert is False  # no warm cache -> nothing to break


def test_state_expired_is_cold_not_alert():
    snap = ConsoleCostSnapshot(0.48, 12000, True, False, 3)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.EXPIRED, break_reason=None,
        projected_delta_usd=0.13, ttl_remaining_s=None, pricing_as_of=None,
    )
    assert state.label == "$0.48 ○"
    assert state.cold is True and state.alert is False
    assert "expired" in state.tooltip.lower()


def test_state_no_pricing_shows_tokens():
    snap = ConsoleCostSnapshot(None, 12_345, False, False, 2)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.NONE, break_reason=None,
        projected_delta_usd=None, ttl_remaining_s=None, pricing_as_of=None,
    )
    assert state.label == "12.3k tok"
    assert "[pricing]" in state.tooltip


def test_estimated_entries_marked_in_tooltip_and_label():
    snap = ConsoleCostSnapshot(0.10, 5000, True, True, 2)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.NONE, break_reason=None,
        projected_delta_usd=None, ttl_remaining_s=None, pricing_as_of=None,
    )
    assert state.label.startswith("~$0.10")
    assert "estimated" in state.tooltip.lower()


# --- Fix round 1: reviewer findings -----------------------------------------


def test_estimated_row_pricing_depends_on_role():
    """F1: an estimated assistant row (usage=None is real, see
    console_chat_models.py:439) must price at output_per_mtok, not
    input_per_mtok -- output rates run 4-5x input, so collapsing both onto
    the input rate badly understates an all-assistant-estimated transcript.
    """
    content = "x" * 400  # identical length for both rows

    user_snap = build_cost_snapshot(
        [_msg(content=content, usage=None, role="user")],
        provider="anthropic", model="claude-sonnet-4-6",
    )
    assistant_snap = build_cost_snapshot(
        [_msg(content=content, usage=None, role="assistant")],
        provider="anthropic", model="claude-sonnet-4-6",
    )
    assert user_snap.pricing_known is True
    assert assistant_snap.pricing_known is True
    # Same content length -> comparable token counts, but the assistant row
    # must cost strictly more since it is priced at the output rate (15.00)
    # instead of the input rate (3.00).
    assert assistant_snap.total_usd > user_snap.total_usd

    # Exact expected total for a transcript containing both rows together.
    user_tokens = _estimate_tokens_locally(
        [{"role": "user", "content": content}], "claude-sonnet-4-6", "anthropic"
    )
    assistant_tokens = _estimate_tokens_locally(
        [{"role": "assistant", "content": content}], "claude-sonnet-4-6", "anthropic"
    )
    expected_total = round(
        user_tokens * 3.00 / 1_000_000 + assistant_tokens * 15.00 / 1_000_000, 6
    )
    combined_snap = build_cost_snapshot(
        [
            _msg(content=content, usage=None, role="user"),
            _msg(content=content, usage=None, role="assistant"),
        ],
        provider="anthropic", model="claude-sonnet-4-6",
    )
    assert combined_snap.total_usd == expected_total


def test_estimate_provider_is_normalized_before_ratio_lookup():
    """F2: provider must go through provider_config_key before it reaches
    the estimator's char-ratio table (console_session_settings.py:742
    convention) -- a raw display-cased "Google" must resolve the same
    ratio entry as its normalized "google" spelling.
    """
    content = "some context content for token estimation " * 5

    snap_lower = build_cost_snapshot(
        [_msg(content=content, usage=None, role="user")],
        provider="google", model="gemini-2.5-flash",
    )
    snap_mixed = build_cost_snapshot(
        [_msg(content=content, usage=None, role="user")],
        provider="Google", model="gemini-2.5-flash",
    )
    assert snap_mixed.total_tokens == snap_lower.total_tokens
    assert snap_mixed.total_usd == snap_lower.total_usd


def test_tokens_only_tooltip_narrates_cache_state():
    """F3: an unpriced model's tooltip must still explain a warm/alerting
    cache -- the cache-state line renders between the token line and the
    [pricing] hint, not just for priced snapshots.
    """
    snap = ConsoleCostSnapshot(None, 12_345, False, False, 2)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.WARM, break_reason="system prompt changed",
        projected_delta_usd=0.13, ttl_remaining_s=90.0, pricing_as_of=None,
    )
    assert state.alert is True
    assert "system prompt changed" in state.tooltip
    assert "[pricing]" in state.tooltip

    tokens_idx = state.tooltip.index("Tokens:")
    cache_idx = state.tooltip.index("system prompt changed")
    pricing_idx = state.tooltip.index("[pricing]")
    assert tokens_idx < cache_idx < pricing_idx
