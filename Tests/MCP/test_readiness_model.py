from __future__ import annotations

from tldw_chatbook.MCP.readiness import (
    READY_ACTIONS,
    REASON_LABELS,
    REASON_PRIORITY,
    REASON_TO_ACTIONS,
    REASON_TO_STATE,
    STATE_CSS_CLASSES,
    STATE_GLYPHS,
    STATE_LABELS,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
    aggregate_summary,
    as_checking,
    builtin_readiness,
    resolve_state,
)


def _snap(
    state: ReadinessState, reasons: tuple[ReasonCode, ...] = ()
) -> ReadinessSnapshot:
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


def test_every_reason_code_has_a_non_empty_human_label():
    """A3b: user-facing copy must never fall back to the raw enum value.

    The inspector renders `REASON_LABELS[primary_reason]` instead of the
    internal reason code (see mcp_inspector.update_readiness) -- every
    ReasonCode must resolve to a short, non-empty phrase, and that phrase
    must not just be the internal `.value` string leaking through.
    """
    for code in ReasonCode:
        assert code in REASON_LABELS, f"{code} missing a human label"
        label = REASON_LABELS[code]
        assert isinstance(label, str) and label.strip(), f"{code} has an empty label"
        assert label != code.value, f"{code} label is just the raw reason code"


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
    # F-059: the per-state breakdown is gone -- per-server states are
    # already itemized in the table/rail/callouts; the summary keeps only
    # the aggregate ready count (genuinely different information).
    assert "needs setup" not in summary
    assert "stale" not in summary
    assert aggregate_summary([]) == "No MCP servers configured yet."


def test_aggregate_summary_excludes_off_builtin_from_setup_math():
    """F-051: the built-in server ships disabled as an opt-in, so it is not
    part of the ready/needs-setup math -- a pristine install must not read
    '0 of 1 servers ready — 1 needs setup'. The built-in is reported
    separately as off, and genuine problems still count normally."""
    off = builtin_readiness(enabled=False)
    pristine = aggregate_summary([off])
    assert "needs setup" not in pristine
    assert "0 of 1" not in pristine
    assert "off" in pristine.lower()

    mixed = aggregate_summary([_snap(ReadinessState.READY), off])
    assert "1 of 1" in mixed
    assert "needs setup" not in mixed
    assert "off" in mixed.lower()

    problem = aggregate_summary(
        [_snap(ReadinessState.NEEDS_SETUP, (ReasonCode.AUTH_MISSING,)), off]
    )
    assert "0 of 1" in problem
    # F-059: no per-state breakdown in the summary anymore -- the problem
    # callout/table row carries the state; the aggregate count and the
    # off/opt-in note stay.
    assert "needs setup" not in problem
    assert "off" in problem.lower()


def test_state_css_classes_complete():
    assert set(STATE_CSS_CLASSES) == set(ReadinessState)
    assert all(v.startswith("mcp-status-") for v in STATE_CSS_CLASSES.values())


def test_off_opt_in_has_its_own_muted_display_state():
    """task-2239: the off-by-choice built-in gets its own muted display
    state -- no alarm glyph, no 'Needs setup' vocabulary -- while genuine
    problems keep flowing through the existing states. Every display map
    covers the new state so no surface can fall over rendering it."""
    assert ReadinessState.OFF_OPT_IN in STATE_GLYPHS
    assert ReadinessState.OFF_OPT_IN in STATE_LABELS
    # Muted, and distinct from the ready/alarm vocabulary it replaces.
    assert STATE_CSS_CLASSES[ReadinessState.OFF_OPT_IN] == "mcp-status-muted"
    assert STATE_GLYPHS[ReadinessState.OFF_OPT_IN] not in (
        STATE_GLYPHS[ReadinessState.READY],
        STATE_GLYPHS[ReadinessState.NEEDS_ATTENTION],
    )
    off = builtin_readiness(enabled=False)
    assert off.state is ReadinessState.OFF_OPT_IN
    badge = off.badge_text()
    assert badge == f"{STATE_GLYPHS[ReadinessState.OFF_OPT_IN]} Off (opt-in)"
    assert "Needs setup" not in badge
    # Genuine states are untouched.
    assert builtin_readiness(enabled=True).state is ReadinessState.READY
    assert STATE_LABELS[ReadinessState.NEEDS_SETUP] == "Needs setup"


def test_as_checking_replaces_state_and_message():
    snap = _snap(ReadinessState.READY)
    checking = as_checking(snap, "connect")
    assert checking.state is ReadinessState.CHECKING
    assert checking.reasons == ()
    assert "connect" in checking.message
    assert checking.server_key == snap.server_key
