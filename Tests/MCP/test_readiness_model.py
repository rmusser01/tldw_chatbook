from __future__ import annotations

from tldw_chatbook.MCP.readiness import (
    READY_ACTIONS,
    REASON_PRIORITY,
    REASON_TO_ACTIONS,
    REASON_TO_STATE,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
    aggregate_summary,
    resolve_state,
)


def _snap(state: ReadinessState, reasons: tuple[ReasonCode, ...] = ()) -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key="local:demo",
        label="demo",
        source="local",
        state=state,
        reasons=reasons,
        message="",
    )


def test_every_reason_code_has_state_actions_and_priority():
    for code in ReasonCode:
        assert code in REASON_TO_STATE, f"{code} missing display state"
        assert code in REASON_TO_ACTIONS, f"{code} missing action set"
        assert code in REASON_PRIORITY, f"{code} missing from priority order"
    assert len(REASON_PRIORITY) == len(set(REASON_PRIORITY)) == len(list(ReasonCode))


def test_resolve_state_uses_priority_order_not_input_order():
    # discovery_not_run alone -> needs_setup
    assert resolve_state((ReasonCode.DISCOVERY_NOT_RUN,)) is ReadinessState.NEEDS_SETUP
    # auth_missing outranks discovery_not_run regardless of input order
    assert (
        resolve_state((ReasonCode.DISCOVERY_NOT_RUN, ReasonCode.AUTH_MISSING))
        is ReadinessState.NEEDS_SETUP
    )
    assert (
        resolve_state((ReasonCode.NO_TOOLS_RETURNED, ReasonCode.UNREACHABLE))
        is ReadinessState.NEEDS_ATTENTION
    )
    assert resolve_state(()) is ReadinessState.READY


def test_primary_reason_and_allowed_actions_follow_priority():
    snap = _snap(
        ReadinessState.NEEDS_SETUP,
        (ReasonCode.DISCOVERY_NOT_RUN, ReasonCode.AUTH_MISSING),
    )
    assert snap.primary_reason is ReasonCode.AUTH_MISSING
    assert snap.allowed_actions == REASON_TO_ACTIONS[ReasonCode.AUTH_MISSING]


def test_ready_snapshot_gets_ready_actions_and_badge():
    snap = _snap(ReadinessState.READY)
    assert snap.primary_reason is None
    assert snap.allowed_actions == READY_ACTIONS
    assert HubAction.REFRESH_DISCOVERY in snap.allowed_actions
    assert "Ready" in snap.badge_text()


def test_aggregate_summary_counts_states():
    snaps = [
        _snap(ReadinessState.READY),
        _snap(ReadinessState.READY),
        _snap(ReadinessState.NEEDS_SETUP, (ReasonCode.AUTH_MISSING,)),
        _snap(ReadinessState.STALE, (ReasonCode.RUNTIME_UNAVAILABLE,)),
    ]
    summary = aggregate_summary(snaps)
    assert "2 of 4" in summary
    assert "needs setup" in summary
    assert aggregate_summary([]) == "No MCP servers configured yet."
