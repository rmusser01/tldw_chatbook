from __future__ import annotations

from tldw_chatbook.MCP.readiness import (
    BUILTIN_SERVER_KEY,
    HubAction,
    ReadinessState,
    ReasonCode,
    builtin_readiness,
    env_placeholder_names,
    local_profile_readiness,
    server_external_record_readiness,
    server_target_readiness,
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
        discovery_snapshot={"tools": [{"name": "a"}, {"name": "b"}], "resources": [], "prompts": []},
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
    assert snap.auth_display == "none"


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
    assert reported.state is ReadinessState.NEEDS_SETUP  # auth_missing outranks via table
    assert reported.primary_reason is ReasonCode.AUTH_MISSING
    assert reported.tool_count == 3
    assert reported.server_key == "server:main/web-search"

    bare = server_external_record_readiness({"name": "Mystery"}, server_id="main")
    assert bare.primary_reason is ReasonCode.DISCOVERY_NOT_RUN
    assert "not reported" in bare.message.lower()


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
    off = builtin_readiness(enabled=False)
    assert off.state is ReadinessState.NEEDS_SETUP
    assert off.primary_reason is ReasonCode.NOT_CONFIGURED


def test_runtime_error_drives_needs_attention_with_stored_message():
    record = _local_record(
        discovery_snapshot={"tools": [{"name": "a"}], "resources": [], "prompts": []},
        is_connected=False,
    )
    record["runtime_state"] = {"ok": False, "last_error": "Timed out after 45s",
                               "last_action": "connect", "last_attempt_at": "t", "last_ok_at": None}
    snap = local_profile_readiness(record, environ={})
    assert snap.state is ReadinessState.NEEDS_ATTENTION
    assert snap.primary_reason is ReasonCode.DISCOVERY_FAILED
    assert "Timed out" in snap.message


def test_runtime_ok_keeps_normal_derivation():
    record = _local_record(
        discovery_snapshot={"tools": [{"name": "a"}], "resources": [], "prompts": []},
        is_connected=True,
    )
    record["runtime_state"] = {"ok": True, "last_error": None, "last_ok_at": "2026-07-14T00:00:00Z"}
    snap = local_profile_readiness(record, environ={})
    assert snap.state is ReadinessState.READY
    assert snap.detail["last_ok_at"] == "2026-07-14T00:00:00Z"


def test_auth_missing_outranks_runtime_error():
    record = _local_record(env_placeholders={"API_KEY": "$MISSING"})
    record["runtime_state"] = {"ok": False, "last_error": "boom"}
    snap = local_profile_readiness(record, environ={})
    assert snap.primary_reason is ReasonCode.AUTH_MISSING
