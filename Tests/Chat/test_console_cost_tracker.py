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
