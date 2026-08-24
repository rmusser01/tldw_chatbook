from __future__ import annotations

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_prompt_reducer import (
    PermissionPromptReport,
    build_permission_prompt_report,
    format_permission_prompt_report,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState


def _tool(
    server_key: str,
    name: str,
    *,
    executable: bool = True,
    stale: bool = False,
) -> HubTool:
    return HubTool(
        server_key=server_key,
        server_label=server_key.removeprefix("local:"),
        source="local",
        name=name,
        description=f"{name} description",
        input_schema={"type": "object"},
        tags=(),
        stale=stale,
        executable=executable,
    )


def _record(
    server_key: str,
    tool_name: str,
    decision: str,
    *,
    initiator: str = "agent",
    ts: str = "2026-08-01T20:00:00+00:00",
) -> dict:
    return {
        "ts": ts,
        "server_key": server_key,
        "tool_name": tool_name,
        "initiator": initiator,
        "decision": decision,
        "ok": True,
        "duration_ms": 10,
    }


def _state(
    state: str,
    *,
    config_changed: bool = False,
    risk_floored: bool = False,
) -> EffectiveToolState:
    return EffectiveToolState(
        state=state,
        origin="global_default",
        config_changed=config_changed,
        risk_floored=risk_floored,
    )


def test_recommends_repeated_agent_approved_ask_gated_tool():
    """Catches dropping repeated approved ask-gated tools from the report."""
    tool = _tool("local:docs", "search")
    records = [
        _record("local:docs", "search", "approved", ts="2026-08-01T20:00:00+00:00"),
        _record("local:docs", "search", "approved", ts="2026-08-01T20:03:00+00:00"),
    ]
    states = {("local:docs", "search"): _state("ask")}

    report = build_permission_prompt_report(records, [tool], states)

    assert [(r.server_key, r.tool_name, r.approved_count) for r in report.recommendations] == [
        ("local:docs", "search", 2)
    ]
    assert report.recommendations[0].last_seen == "2026-08-01T20:03:00+00:00"
    assert report.total_records == 2


def test_excludes_tools_below_repeated_approval_threshold():
    """Catches recommending one-off approvals as prompt-reduction candidates."""
    tool = _tool("local:docs", "search")
    records = [_record("local:docs", "search", "approved")]
    states = {("local:docs", "search"): _state("ask")}

    report = build_permission_prompt_report(records, [tool], states)

    assert report.recommendations == []
    assert report.excluded[("local:docs", "search")] == "below-threshold"


def test_excludes_already_allowed_or_denied_tools():
    """Catches suggesting changes that conflict with current permission state."""
    tools = [_tool("local:docs", "search"), _tool("local:docs", "delete")]
    records = [
        _record("local:docs", "search", "approved"),
        _record("local:docs", "search", "approved"),
        _record("local:docs", "delete", "approved"),
        _record("local:docs", "delete", "approved"),
    ]
    states = {
        ("local:docs", "search"): _state("allow"),
        ("local:docs", "delete"): _state("deny"),
    }

    report = build_permission_prompt_report(records, tools, states)

    assert report.recommendations == []
    assert report.excluded[("local:docs", "search")] == "already-allowed"
    assert report.excluded[("local:docs", "delete")] == "denied"


def test_excludes_safety_downgraded_ask_states():
    """Catches bypassing rug-pull and high-risk-floor safety downgrades."""
    tools = [_tool("local:docs", "changed"), _tool("local:docs", "process")]
    records = [
        _record("local:docs", "changed", "approved"),
        _record("local:docs", "changed", "approved"),
        _record("local:docs", "process", "approved"),
        _record("local:docs", "process", "approved"),
    ]
    states = {
        ("local:docs", "changed"): _state("ask", config_changed=True),
        ("local:docs", "process"): _state("ask", risk_floored=True),
    }

    report = build_permission_prompt_report(records, tools, states)

    assert report.recommendations == []
    assert report.excluded[("local:docs", "changed")] == "definition-changed"
    assert report.excluded[("local:docs", "process")] == "high-risk-floor"


def test_excludes_non_agent_and_non_approved_decisions():
    """Catches counting audit rows that were not user approvals in agent flow."""
    tool = _tool("local:docs", "search")
    records = [
        _record("local:docs", "search", "approved", initiator="test"),
        _record("local:docs", "search", "allowed"),
        _record("local:docs", "search", "approved-session"),
        _record("local:docs", "search", "approved-session"),
        _record("local:docs", "search", "denied"),
        _record("local:docs", "search", "denied-timeout"),
    ]
    states = {("local:docs", "search"): _state("ask")}

    report = build_permission_prompt_report(records, [tool], states)

    assert report.recommendations == []
    assert report.approval_records == 0
    assert ("local:docs", "search") not in report.excluded


def test_excludes_missing_stale_or_unexecutable_live_tools():
    """Catches recommending permissions for tools the catalog cannot execute."""
    stale = _tool("local:docs", "stale", stale=True)
    disabled = _tool("local:docs", "disabled", executable=False)
    records = [
        _record("local:docs", "missing", "approved"),
        _record("local:docs", "missing", "approved"),
        _record("local:docs", "stale", "approved"),
        _record("local:docs", "stale", "approved"),
        _record("local:docs", "disabled", "approved"),
        _record("local:docs", "disabled", "approved"),
    ]
    states = {
        ("local:docs", "stale"): _state("ask"),
        ("local:docs", "disabled"): _state("ask"),
    }

    report = build_permission_prompt_report(records, [stale, disabled], states)

    assert report.recommendations == []
    assert report.excluded[("local:docs", "missing")] == "tool-not-found"
    assert report.excluded[("local:docs", "stale")] == "tool-unavailable"
    assert report.excluded[("local:docs", "disabled")] == "tool-unavailable"


def test_recommendations_sort_by_count_then_last_seen():
    """Catches unstable report ordering for the highest-impact recommendations."""
    tools = [
        _tool("local:docs", "most"),
        _tool("local:docs", "newer"),
        _tool("local:docs", "older"),
    ]
    records = [
        _record("local:docs", "older", "approved", ts="2026-08-01T20:00:00+00:00"),
        _record("local:docs", "older", "approved", ts="2026-08-01T20:01:00+00:00"),
        _record("local:docs", "newer", "approved", ts="2026-08-01T20:02:00+00:00"),
        _record("local:docs", "newer", "approved", ts="2026-08-01T20:03:00+00:00"),
        _record("local:docs", "most", "approved", ts="2026-08-01T20:04:00+00:00"),
        _record("local:docs", "most", "approved", ts="2026-08-01T20:05:00+00:00"),
        _record("local:docs", "most", "approved", ts="2026-08-01T20:06:00+00:00"),
    ]
    states = {(tool.server_key, tool.name): _state("ask") for tool in tools}

    report = build_permission_prompt_report(records, tools, states)

    assert [r.tool_name for r in report.recommendations] == ["most", "newer", "older"]


def test_recommendations_sort_missing_last_seen_after_known_timestamps():
    """Catches unknown timestamps ranking ahead of known recent approvals."""
    tools = [
        _tool("local:docs", "unknown"),
        _tool("local:docs", "known"),
    ]
    records = [
        _record("local:docs", "unknown", "approved", ts=""),
        _record("local:docs", "unknown", "approved", ts=""),
        _record("local:docs", "known", "approved", ts="2026-08-01T20:00:00+00:00"),
        _record("local:docs", "known", "approved", ts="2026-08-01T20:01:00+00:00"),
    ]
    states = {(tool.server_key, tool.name): _state("ask") for tool in tools}

    report = build_permission_prompt_report(records, tools, states)

    assert [r.tool_name for r in report.recommendations] == ["known", "unknown"]


def test_format_report_distinguishes_an_empty_local_log():
    report = PermissionPromptReport(
        recommendations=[],
        excluded={},
        total_records=0,
        approval_records=0,
        min_approved_count=2,
    )

    rendered = format_permission_prompt_report(report)

    assert "No local MCP execution records were found." in rendered


def test_format_report_distinguishes_activity_without_permission_prompts():
    report = PermissionPromptReport(
        recommendations=[],
        excluded={},
        total_records=4,
        approval_records=0,
        min_approved_count=2,
    )

    rendered = format_permission_prompt_report(report)

    assert "No prompted agent approvals were found in those records." in rendered


def test_format_report_summarizes_all_exclusion_reasons():
    reasons = {
        ("local:docs", "few"): "below-threshold",
        ("local:docs", "allowed"): "already-allowed",
        ("local:docs", "denied"): "denied",
        ("local:docs", "changed"): "definition-changed",
        ("local:docs", "risky"): "high-risk-floor",
        ("local:docs", "missing"): "tool-not-found",
        ("local:docs", "stale"): "tool-unavailable",
        ("local:docs", "unknown"): "state-unavailable",
    }
    report = PermissionPromptReport(
        recommendations=[],
        excluded=reasons,
        total_records=16,
        approval_records=16,
        min_approved_count=2,
    )

    rendered = format_permission_prompt_report(report)

    assert "Not recommended:" in rendered
    for explanation in (
        "below threshold (1)",
        "already allowed (1)",
        "explicitly denied (1)",
        "definition changed (1)",
        "high-risk safety floor (1)",
        "missing from the current catalog (1)",
        "currently unavailable (1)",
        "permission state unavailable (1)",
    ):
        assert explanation in rendered
