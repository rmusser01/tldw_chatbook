"""Cost math + chip-state formatting for the Console cost chip (PR3).

House pattern: the state dataclass owns ALL label formatting; the widget
only renders (see ConsoleControlState). Never store dollars — compute at
display time from usage rows via the pricing catalog.
"""

from types import SimpleNamespace

from tldw_chatbook.Chat.console_cost_tracker import (
    ConsoleCacheState,
    ConsoleCostRow,
    ConsoleCostRowTotals,
    ConsoleCostSnapshot,
    PayloadFingerprint,
    build_cost_rows,
    build_cost_rows_totals,
    build_cost_snapshot,
    build_cost_state,
    fingerprint_break_reason,
    fingerprint_payload,
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


def test_fleet_tokens_fold_into_total_tokens_but_never_price():
    """PR2b Task 5 (cost rollup): a sub-agent fleet's measured token spend
    reaches the snapshot's ``total_tokens`` -- but never ``total_usd``,
    since a fleet child's combined figure has no input/output split to
    price accurately (see ``ConsoleCostSnapshot.fleet_tokens``'s
    docstring)."""
    usage = ProviderUsage(
        uncached_input=1_000_000, output=1_000_000,
        provider="anthropic", model="claude-sonnet-4-6",
    )
    snap = build_cost_snapshot(
        [_msg(usage=usage)],
        provider="anthropic",
        model="claude-sonnet-4-6",
        fleet_tokens=500,
    )
    assert snap.fleet_tokens == 500
    assert snap.total_tokens == 2_000_000 + 500
    # Unaffected: the primary transcript's own pricing stays exactly what
    # it was without any fleet spend.
    assert snap.total_usd == 18.0
    assert snap.pricing_known is True


def test_fleet_tokens_default_to_zero_and_are_backward_compatible():
    """Every pre-Task-5 caller omits `fleet_tokens` -- byte-identical
    totals to before this parameter existed."""
    snap = build_cost_snapshot(
        [_msg(content="x" * 400, usage=None)],
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    assert snap.fleet_tokens == 0


def test_fleet_tokens_tooltip_line_appears_in_tokens_only_mode():
    snap = ConsoleCostSnapshot(None, 12_345, False, False, 2, fleet_tokens=750)
    state = build_cost_state(
        snap,
        cache_state=ConsoleCacheState.NONE,
        break_reason=None,
        projected_delta_usd=None,
        ttl_remaining_s=None,
        pricing_as_of=None,
    )
    assert "Sub-agents: 750 tok (not priced)" in state.tooltip


def test_fleet_tokens_tooltip_line_appears_in_priced_mode():
    snap = ConsoleCostSnapshot(0.48, 12000, True, False, 3, fleet_tokens=1200)
    state = build_cost_state(
        snap,
        cache_state=ConsoleCacheState.NONE,
        break_reason=None,
        projected_delta_usd=None,
        ttl_remaining_s=None,
        pricing_as_of=None,
    )
    assert "Sub-agents: 1.2k tok (not priced)" in state.tooltip


def test_no_fleet_tokens_line_when_fleet_tokens_is_zero():
    snap = ConsoleCostSnapshot(0.48, 12000, True, False, 3)
    state = build_cost_state(
        snap,
        cache_state=ConsoleCacheState.NONE,
        break_reason=None,
        projected_delta_usd=None,
        ttl_remaining_s=None,
        pricing_as_of=None,
    )
    assert "Sub-agents:" not in state.tooltip


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


# --- Payload fingerprinting (cache-break detection) -------------------------


def _fp(messages, provider="anthropic", model="m"):
    return fingerprint_payload(provider, model, messages)


BASE = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
]


def test_appended_turn_is_not_a_break():
    baseline = _fp(BASE)
    current = _fp(BASE + [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}])
    assert fingerprint_break_reason(baseline, current) is None


def test_each_component_yields_its_reason_with_priority():
    baseline = _fp(BASE)
    assert fingerprint_break_reason(baseline, _fp(BASE, model="other")) == "model or provider changed"
    changed_system = [{"role": "system", "content": "be verbose"}] + BASE[1:]
    assert fingerprint_break_reason(baseline, _fp(changed_system)) == "system prompt changed"
    edited = [BASE[0], {"role": "user", "content": "EDITED"}, BASE[2]]
    assert fingerprint_break_reason(baseline, _fp(edited)) == "earlier history changed"
    # model beats system when both changed
    assert (
        fingerprint_break_reason(baseline, _fp(changed_system, model="other"))
        == "model or provider changed"
    )


def test_truncated_history_is_a_break():
    baseline = _fp(BASE)
    assert fingerprint_break_reason(baseline, _fp(BASE[:2])) == "earlier history changed"


def test_list_content_rows_hash_stably():
    rows = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert _fp(rows) == _fp([dict(r) for r in rows])


# --- build_cost_rows (task-5: per-message breakdown for the cost modal) -----


def test_build_cost_rows_prices_a_usage_row():
    usage = ProviderUsage(
        uncached_input=1000, cache_read=200, cache_write=50, output=500,
        provider="anthropic", model="claude-sonnet-4-6",
    )
    rows = build_cost_rows(
        [_msg(usage=usage)], provider="anthropic", model="claude-sonnet-4-6"
    )
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, ConsoleCostRow)
    assert row.index == 0
    assert row.model == "claude-sonnet-4-6"
    assert row.uncached_input == 1000
    assert row.cache_read == 200
    assert row.cache_write == 50
    assert row.output == 500
    assert row.estimated is False
    assert row.cost_usd is not None and row.cost_usd > 0


def test_build_cost_rows_estimates_row_without_usage_role_aware():
    content = "x" * 400
    rows = build_cost_rows(
        [
            _msg(content=content, usage=None, role="user"),
            _msg(content=content, usage=None, role="assistant"),
        ],
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    assert len(rows) == 2
    user_row, assistant_row = rows
    assert user_row.estimated is True and assistant_row.estimated is True
    # user tokens land in uncached_input, assistant tokens land in output --
    # never double-counted across buckets.
    assert user_row.uncached_input > 0 and user_row.output == 0
    assert assistant_row.output > 0 and assistant_row.uncached_input == 0
    # Output rate (15.00) > input rate (3.00) for equal token counts.
    assert assistant_row.cost_usd > user_row.cost_usd
    assert user_row.index == 0 and assistant_row.index == 1


def test_build_cost_rows_unknown_model_yields_unpriced_row():
    usage = ProviderUsage(uncached_input=100, provider="anthropic", model="mystery-9000")
    rows = build_cost_rows(
        [_msg(usage=usage)], provider="anthropic", model="mystery-9000"
    )
    assert len(rows) == 1
    assert rows[0].cost_usd is None


def test_build_cost_rows_skips_blank_content_rows():
    rows = build_cost_rows(
        [_msg(content="", usage=None), _msg(content="   ", usage=None)],
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    assert rows == []


def test_build_cost_rows_indices_skip_dropped_rows():
    """A blank row is skipped, not counted -- the next real row keeps the
    transcript-order index of its own position, not a compacted 0/1/2..."""
    rows = build_cost_rows(
        [
            _msg(content="hello", usage=None, role="user"),
            _msg(content="", usage=None),
            _msg(content="world", usage=None, role="user"),
        ],
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    assert [row.index for row in rows] == [0, 2]


def test_build_cost_rows_totals_sums_priced_rows():
    rows = [
        ConsoleCostRow(0, "user", "m", 100, 0, 0, 0, 0.10, False),
        ConsoleCostRow(1, "assistant", "m", 0, 0, 0, 50, 0.05, False),
    ]
    totals = build_cost_rows_totals(rows)
    assert isinstance(totals, ConsoleCostRowTotals)
    assert totals.total_tokens == 150
    assert totals.total_cost_usd == 0.15
    assert totals.has_estimated_entries is False
    assert totals.row_count == 2


def test_build_cost_rows_totals_none_when_any_row_unpriced():
    rows = [
        ConsoleCostRow(0, "user", "m", 100, 0, 0, 0, 0.10, False),
        ConsoleCostRow(1, "assistant", "m", 0, 0, 0, 50, None, False),
    ]
    totals = build_cost_rows_totals(rows)
    assert totals.total_cost_usd is None
    assert totals.total_tokens == 150


def test_build_cost_rows_totals_marks_estimated_flag():
    rows = [ConsoleCostRow(0, "user", "m", 100, 0, 0, 0, 0.10, True)]
    totals = build_cost_rows_totals(rows)
    assert totals.has_estimated_entries is True


def test_build_cost_rows_totals_empty_is_unpriced_zero():
    totals = build_cost_rows_totals([])
    assert totals.total_tokens == 0
    assert totals.total_cost_usd is None
    assert totals.row_count == 0


# --- task-2390: realtime audio/transcription fields on ConsoleCostRow ------


def test_build_cost_rows_carries_audio_and_transcription_fields():
    usage = ProviderUsage(
        uncached_input=33, output=118, audio_input=18, audio_output=90,
        transcription_seconds=2.5, provider="openai", model="gpt-realtime",
    )
    rows = build_cost_rows(
        [_msg(usage=usage)], provider="openai", model="gpt-realtime"
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.audio_input == 18
    assert row.audio_output == 90
    assert row.transcription_seconds == 2.5
    # cost_usd is the row's full total -- audio/transcription costs are
    # already folded in (see console_conversation_inspector.py's Costs tab
    # for the breakdown that keeps them visible rather than an
    # undecomposable single figure).
    assert row.cost_usd is not None and row.cost_usd > 0


def test_build_cost_rows_estimated_row_has_zero_audio_fields():
    rows = build_cost_rows(
        [_msg(content="hi there", usage=None, role="user")],
        provider="anthropic", model="claude-sonnet-4-6",
    )
    assert rows[0].audio_input == 0
    assert rows[0].audio_output == 0
    assert rows[0].transcription_seconds == 0.0


def test_console_cost_row_existing_positional_construction_still_works():
    # AC3-style pin for ConsoleCostRow's shape: the new fields must be
    # trailing-optional so every EXISTING positional construction site in
    # this file (the totals tests above) keeps working unmodified.
    row = ConsoleCostRow(0, "user", "m", 100, 0, 0, 0, 0.10, False)
    assert row.audio_input == 0
    assert row.audio_output == 0
    assert row.transcription_seconds == 0.0
