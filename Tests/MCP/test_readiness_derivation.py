from __future__ import annotations

from tldw_chatbook.MCP.readiness import (
    BUILTIN_SERVER_KEY,
    STATE_GLYPHS,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    ReasonCode,
    builtin_readiness,
    env_placeholder_names,
    local_profile_readiness,
    server_external_record_readiness,
    server_target_readiness,
    worst_state,
)


def _local_record(**overrides):
    record = {
        "profile_id": "docs",
        "command": "python",
        "args": ["-m", "demo.server"],
        "env_placeholders": {},
        "env_literals": {},
        "discovery_snapshot": None,
        "is_connected": False,
    }
    record.update(overrides)
    return record


def test_env_placeholder_names_strips_dollar_forms():
    assert env_placeholder_names({"API_KEY": "$MY_KEY", "TOKEN": "${OTHER}"}) == [
        "MY_KEY",
        "OTHER",
    ]


def test_local_profile_never_validated_is_needs_setup_discovery_not_run():
    snap = local_profile_readiness(_local_record(), environ={})
    assert snap.server_key == "local:docs"
    assert snap.state is ReadinessState.NEEDS_SETUP
    assert snap.primary_reason is ReasonCode.DISCOVERY_NOT_RUN
    assert snap.tool_count is None


def test_local_profile_missing_env_var_is_auth_missing():
    record = _local_record(env_placeholders={"API_KEY": "$MISSING_VAR"})
    snap = local_profile_readiness(record, environ={})
    assert snap.primary_reason is ReasonCode.AUTH_MISSING
    assert "MISSING_VAR" in snap.message
    present = local_profile_readiness(record, environ={"MISSING_VAR": "x"})
    assert ReasonCode.AUTH_MISSING not in present.reasons


def test_local_profile_discovered_but_disconnected_is_stale_runtime_unavailable():
    record = _local_record(
        discovery_snapshot={
            "tools": [{"name": "a"}, {"name": "b"}],
            "resources": [],
            "prompts": [],
        },
        is_connected=False,
    )
    snap = local_profile_readiness(record, environ={})
    assert snap.state is ReadinessState.STALE
    assert snap.primary_reason is ReasonCode.RUNTIME_UNAVAILABLE
    assert snap.tool_count == 2
    assert HubAction.CONNECT in snap.allowed_actions


def test_local_profile_connected_with_snapshot_is_ready():
    record = _local_record(
        discovery_snapshot={"tools": [{"name": "a"}], "resources": [], "prompts": []},
        is_connected=True,
    )
    snap = local_profile_readiness(record, environ={})
    assert snap.state is ReadinessState.READY
    assert snap.reasons == ()
    assert snap.auth_display == "—"


class _Target:
    def __init__(self, reachability, auth_state):
        self.server_id = "main"
        self.label = "Main Server"
        self.auth_mode = "api_key"
        self.last_known_reachability = reachability
        self.last_known_auth_state = auth_state


def test_server_target_states():
    assert (
        server_target_readiness(_Target("reachable", "authenticated")).state
        is ReadinessState.READY
    )
    assert (
        server_target_readiness(_Target("unreachable", "unknown")).primary_reason
        is ReasonCode.UNREACHABLE
    )
    assert (
        server_target_readiness(_Target("reachable", "auth_required")).primary_reason
        is ReasonCode.AUTH_MISSING
    )
    never_probed = server_target_readiness(_Target(None, None))
    assert never_probed.primary_reason is ReasonCode.DISCOVERY_NOT_RUN
    assert never_probed.server_key == "server:main"


def test_server_external_record_passthrough_and_fallback():
    reported = server_external_record_readiness(
        {
            "server_id": "web-search",
            "name": "Web Search",
            "display_state": "needs_attention",
            "reason_codes": ["auth_missing"],
            "tool_count": 3,
            "transport": "http",
        },
        server_id="main",
    )
    assert (
        reported.state is ReadinessState.NEEDS_SETUP
    )  # auth_missing outranks via table
    assert reported.primary_reason is ReasonCode.AUTH_MISSING
    assert reported.tool_count == 3
    assert reported.server_key == "server:main/web-search"

    bare = server_external_record_readiness({"name": "Mystery"}, server_id="main")
    assert bare.primary_reason is ReasonCode.DISCOVERY_NOT_RUN
    assert "not reported" in bare.message.lower()


def test_server_external_record_resource_prompt_counts_follow_tool_count_rules():
    """Task 5 (MCP Hub Phase 6): resource_count/prompt_count are derived the
    same way as tool_count -- reported count wins, else the length of a raw
    list, else None (unreported, rendered as "—" by the servers-mode detail
    body -- never a fake zero)."""
    reported = server_external_record_readiness(
        {"server_id": "s1", "name": "S1", "resource_count": 3, "prompt_count": 0},
        server_id="main",
    )
    assert reported.resource_count == 3
    assert reported.prompt_count == 0

    derived = server_external_record_readiness(
        {
            "server_id": "s2",
            "name": "S2",
            "resources": [{"uri": "a"}, {"uri": "b"}],
            "prompts": [{"name": "p"}],
        },
        server_id="main",
    )
    assert derived.resource_count == 2
    assert derived.prompt_count == 1

    unreported = server_external_record_readiness(
        {"server_id": "s3", "name": "S3"}, server_id="main"
    )
    assert unreported.resource_count is None
    assert unreported.prompt_count is None

    malformed = server_external_record_readiness(
        {"server_id": "s4", "name": "S4", "resource_count": "many", "prompt_count": 2.5},
        server_id="main",
    )
    assert malformed.resource_count is None
    assert malformed.prompt_count is None


def test_server_external_record_display_state_without_reason_codes_is_trusted():
    """F1 regression: a `display_state` the backend reports with no (or
    unrecognized) `reason_codes` must not silently resolve to READY via
    `resolve_state(())` -- the explicit display_state wins.
    """
    snap = server_external_record_readiness(
        {
            "server_id": "s1",
            "name": "S1",
            "display_state": "needs_attention",
            "reason_codes": [],
        },
        server_id="main",
    )
    assert snap.state is ReadinessState.NEEDS_ATTENTION
    assert snap.reasons == ()


def test_server_external_record_display_state_message_uses_status_message_or_default():
    with_status = server_external_record_readiness(
        {
            "server_id": "s1",
            "name": "S1",
            "display_state": "stale",
            "reason_codes": [],
            "status_message": "Catalog is 3 days old.",
        },
        server_id="main",
    )
    assert with_status.state is ReadinessState.STALE
    assert with_status.message == "Catalog is 3 days old."

    without_status = server_external_record_readiness(
        {"server_id": "s1", "name": "S1", "display_state": "stale", "reason_codes": []},
        server_id="main",
    )
    assert without_status.message == "Reported by server without reason codes."


def test_server_external_record_unrecognized_display_state_is_needs_attention():
    """F1 regression: an unrecognized `display_state` string (not a valid
    ReadinessState) must degrade to NEEDS_ATTENTION with an honest message,
    not silently resolve to READY.
    """
    snap = server_external_record_readiness(
        {"server_id": "s1", "name": "S1", "display_state": "bogus", "reason_codes": []},
        server_id="main",
    )
    assert snap.state is ReadinessState.NEEDS_ATTENTION
    assert snap.message == "Server reported an unrecognized state."


def test_server_external_record_non_list_reason_codes_does_not_crash():
    """F2 regression: `reason_codes` arriving as a non-list/tuple (e.g. an
    int from a malformed backend payload) must not raise when iterated --
    it should be treated as if no reason codes were supplied.
    """
    fallback = server_external_record_readiness(
        {"server_id": "s1", "name": "S1", "reason_codes": 42},
        server_id="main",
    )
    assert fallback.primary_reason is ReasonCode.DISCOVERY_NOT_RUN

    with_display_state = server_external_record_readiness(
        {"server_id": "s1", "name": "S1", "reason_codes": 42, "display_state": "ready"},
        server_id="main",
    )
    assert with_display_state.state is ReadinessState.READY
    assert with_display_state.reasons == ()


def test_builtin_readiness():
    on = builtin_readiness(enabled=True)
    assert on.server_key == BUILTIN_SERVER_KEY
    assert on.state is ReadinessState.READY
    assert on.transport == "stdio"
    # F-059: one empty-cell placeholder everywhere -- "—", not "none".
    assert on.auth_display == "—"
    off = builtin_readiness(enabled=False)
    # task-2239: off-by-choice is its own muted display state, not the
    # NEEDS_SETUP alarm vocabulary -- the reason tuple is unchanged so the
    # is_off_opt_in() fallback and allowed-action derivation still work.
    assert off.state is ReadinessState.OFF_OPT_IN
    assert off.primary_reason is ReasonCode.NOT_CONFIGURED


def test_builtin_disabled_message_is_plain_and_keeps_technical_detail():
    """F-050: the disabled built-in's one-line message is short, plain
    language -- no config-file syntax -- so the Servers-mode callout
    ("{glyph} {label}: {message}") renders fully at 100 cols. The
    config-syntax detail stays available under `detail["technical_detail"]`
    for the callout's tooltip."""
    off = builtin_readiness(enabled=False)
    assert "[mcp]" not in off.message
    assert "=" not in off.message
    assert off.message == "Turned off — open to enable."
    callout_line = f"{STATE_GLYPHS[off.state]} {off.label}: {off.message}"
    assert len(callout_line) <= 98
    assert "[mcp].enabled = false" in str(off.detail.get("technical_detail"))


def test_runtime_error_drives_needs_attention_with_stored_message():
    record = _local_record(
        discovery_snapshot={"tools": [{"name": "a"}], "resources": [], "prompts": []},
        is_connected=False,
    )
    record["runtime_state"] = {
        "ok": False,
        "last_error": "Timed out after 45s",
        "last_action": "connect",
        "last_attempt_at": "t",
        "last_ok_at": None,
    }
    snap = local_profile_readiness(record, environ={})
    assert snap.state is ReadinessState.NEEDS_ATTENTION
    assert snap.primary_reason is ReasonCode.DISCOVERY_FAILED
    assert "Timed out" in snap.message


def test_runtime_ok_keeps_normal_derivation():
    record = _local_record(
        discovery_snapshot={"tools": [{"name": "a"}], "resources": [], "prompts": []},
        is_connected=True,
    )
    record["runtime_state"] = {
        "ok": True,
        "last_error": None,
        "last_ok_at": "2026-07-14T00:00:00Z",
    }
    snap = local_profile_readiness(record, environ={})
    assert snap.state is ReadinessState.READY
    assert snap.detail["last_ok_at"] == "2026-07-14T00:00:00Z"


def test_auth_missing_outranks_runtime_error():
    record = _local_record(env_placeholders={"API_KEY": "$MISSING"})
    record["runtime_state"] = {"ok": False, "last_error": "boom"}
    snap = local_profile_readiness(record, environ={})
    assert snap.primary_reason is ReasonCode.AUTH_MISSING


# -- Task 11: per-source Auth column copy ------------------------------------


def test_local_profile_auth_display_singular_for_one_env_var():
    record = _local_record(env_placeholders={"API_KEY": "$X"})
    snap = local_profile_readiness(record, environ={"X": "1"})
    assert snap.auth_display == "1 env var"


def test_local_profile_auth_display_plural_for_multiple_env_vars():
    record = _local_record(env_placeholders={"API_KEY": "$X", "OTHER": "$Y"})
    snap = local_profile_readiness(record, environ={"X": "1", "Y": "2"})
    assert snap.auth_display == "2 env vars"


def test_local_profile_auth_display_dash_for_no_env_vars():
    """F-059: the empty Auth cell uses the same calm "—" placeholder the
    Tools/Scope columns and `_count_display` already use -- not a second
    "none" spelling for the same nothing."""
    snap = local_profile_readiness(_local_record(), environ={})
    assert snap.auth_display == "—"


# -- Task 11: worst_state() for the aggregate status badge -------------------


def _raw_snap(state: ReadinessState) -> ReadinessSnapshot:
    return ReadinessSnapshot(
        server_key="k", label="k", source="local", state=state, reasons=(), message=""
    )


def test_worst_state_empty_or_all_ready_is_ready():
    assert worst_state([]) is ReadinessState.READY
    assert worst_state([_raw_snap(ReadinessState.READY)]) is ReadinessState.READY


def test_worst_state_prioritizes_needs_attention_over_everything_else():
    snaps = [
        _raw_snap(ReadinessState.READY),
        _raw_snap(ReadinessState.STALE),
        _raw_snap(ReadinessState.CHECKING),
        _raw_snap(ReadinessState.NEEDS_ATTENTION),
    ]
    assert worst_state(snaps) is ReadinessState.NEEDS_ATTENTION


def test_worst_state_checking_outranks_ready_only():
    snaps = [_raw_snap(ReadinessState.READY), _raw_snap(ReadinessState.CHECKING)]
    assert worst_state(snaps) is ReadinessState.CHECKING


def test_worst_state_ignores_off_builtin_opt_in():
    """F-051: the disabled built-in is an OFF/opt-in state, not a defect --
    it must not pull the aggregate badge into a warning color on a pristine
    install. Genuine problems still set the worst state.

    task-2239: the pristine all-off aggregate resolves to the muted
    OFF_OPT_IN display state rather than READY -- a ready ● glyph in front
    of the "Built-in server is off …" sentence read as a contradiction."""
    assert worst_state([builtin_readiness(enabled=False)]) is ReadinessState.OFF_OPT_IN
    snaps = [builtin_readiness(enabled=False), _raw_snap(ReadinessState.NEEDS_SETUP)]
    assert worst_state(snaps) is ReadinessState.NEEDS_SETUP
    # A ready server alongside the off built-in still reads ready overall.
    ready_mix = [builtin_readiness(enabled=False), _raw_snap(ReadinessState.READY)]
    assert worst_state(ready_mix) is ReadinessState.READY
