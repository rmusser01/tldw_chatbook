"""Pure Agent rail values retain original source precedence and row builders."""

import json
from dataclasses import replace

import pytest

from tldw_chatbook.Agents.fleet_coordinator import FleetHandle
from tldw_chatbook.Chat.console_agent_bridge import (
    AgentLiveSnapshot,
    AgentLiveStep,
    AgentLiveTurnUsage,
    SubAgentSummary,
)
from tldw_chatbook.Chat.console_execution_agent_rail_values import (
    ConsoleAgentRailInputs,
    ConsoleAgentRailProjection,
    build_agent_rail_projection,
    decode_agent_rail_projection,
    encode_agent_rail_projection,
)
from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError
from tldw_chatbook.Chat.console_glyphs import STATUS_GLYPHS
from tldw_chatbook.Chat.console_inspector_state import (
    ConsoleInspectorSectionState,
    InspectorSectionRow,
)
from tldw_chatbook.UI.Console_Modules.agent import (
    _automatic_work_pause_notice,
    _fleet_row_from_handle,
    _fleet_row_from_summary,
)


def _inputs():
    return ConsoleAgentRailInputs(
        conversation_id="conversation",
        live_snapshot=AgentLiveSnapshot(status="running", step=2),
        historical_snapshot=AgentLiveSnapshot(
            status="done",
            subagents=(SubAgentSummary("prior child", status="done", run_id="old"),),
        ),
        handles=(),
        live_run_snapshots=(),
        setup_established=False,
        primary_run_bound=True,
        fleet_running=0,
        fleet_pending=0,
        pause_reason=None,
        wake_has_pending=False,
        has_history_picker=True,
        full_log_available=False,
        drill_run_id=None,
        observed_at=108.0,
    )


def test_actual_live_rows_keep_original_format_and_precedence():
    handle = FleetHandle(
        "handle",
        "child",
        "reader",
        "read [docs]",
        "running",
        started_at=100.0,
        queued_steering=2,
        resolved_provider="provider",
        resolved_model="model",
    )
    finished = FleetHandle(
        "finished",
        "ended",
        None,
        "old task",
        "done",
        total_tokens=17,
        undelivered_steering=1,
        can_resume=True,
    )
    child = AgentLiveSnapshot(
        status="running",
        turn_usage=AgentLiveTurnUsage(0, "provider", 101.0, 1),
    )
    primary = AgentLiveSnapshot(
        status="running",
        step=2,
        steps=(AgentLiveStep("tool_call", "read [docs]", "primary"),),
        subagents=(SubAgentSummary("child [details]"),),
    )
    inputs = replace(
        _inputs(),
        live_snapshot=primary,
        handles=(handle, finished),
        live_run_snapshots=(child, None),
        fleet_running=1,
        fleet_pending=2,
        full_log_available=True,
    )
    result = build_agent_rail_projection(inputs)
    expected_rows = (
        _fleet_row_from_handle(handle, now=108.0, live_snapshot=child),
        _fleet_row_from_handle(finished, now=108.0, live_snapshot=None),
    )
    assert result.overview_lines == (
        "Agent: running · step 2",
        "· read [docs]",
        f"{STATUS_GLYPHS['running']} child [details]",
    )
    assert result.fleet_section == ConsoleInspectorSectionState(
        expected_rows,
        f"{STATUS_GLYPHS['running']}{STATUS_GLYPHS['done']} 1 working, 1 done",
    )
    assert result.fleet_summary == "1 other agents running, 2 waiting for approval."
    assert result.cancel_all_visible
    assert result.fleet_token_total == 17
    assert result.full_log_available
    handle.task = "changed after capture"
    assert result.fleet_section.rows == expected_rows


def test_established_empty_and_historical_fallback_remain_distinct():
    inputs = _inputs()
    for current in (
        inputs,
        replace(
            inputs,
            live_snapshot=AgentLiveSnapshot(status="setup"),
            primary_run_bound=False,
            setup_established=True,
        ),
    ):
        result = build_agent_rail_projection(replace(current, historical_snapshot=None))
        assert result.fleet_section == ConsoleInspectorSectionState((), "Run history")
        assert not result.cancel_all_visible
        assert result.fleet_token_total == 0
    resumed = replace(
        inputs, live_snapshot=AgentLiveSnapshot(), primary_run_bound=False
    )
    result = build_agent_rail_projection(resumed)
    assert result.overview_lines[0] == "Agent: done"
    assert result.fleet_section.rows == (
        _fleet_row_from_summary(inputs.historical_snapshot.subagents[0], 0),
    )
    assert result.fleet_token_total == 0
    with pytest.raises(ValueError, match="historical snapshot"):
        build_agent_rail_projection(replace(resumed, historical_snapshot=None))
    with pytest.raises(ValueError, match="historical snapshot"):
        build_agent_rail_projection(
            replace(inputs, primary_run_bound=False, historical_snapshot=None)
        )
    with pytest.raises(ValueError, match="drill"):
        build_agent_rail_projection(replace(inputs, drill_run_id="child"))
    absent = build_agent_rail_projection(
        replace(
            inputs,
            live_snapshot=None,
            historical_snapshot=None,
            has_history_picker=False,
        )
    )
    assert absent.overview_lines == ("Agent: unavailable", "", "")
    assert absent.fleet_section == ConsoleInspectorSectionState((), "")


def test_actual_pause_preserves_saved_results_notice_and_row_suppression():
    inputs = replace(_inputs(), pause_reason="tokens_budget")
    result = build_agent_rail_projection(inputs)
    assert result.fleet_section == ConsoleInspectorSectionState(
        (),
        "Paused",
        _automatic_work_pause_notice("tokens_budget", results_saved=True),
    )
    # History failure suppresses the row read, including for an unbound run.
    failed = replace(
        inputs,
        pause_reason="history_unavailable",
        primary_run_bound=False,
        historical_snapshot=None,
    )
    result = build_agent_rail_projection(failed)
    assert result.fleet_section.notice == _automatic_work_pause_notice(
        "history_unavailable",
        results_saved=False,
    )
    assert build_agent_rail_projection(
        replace(failed, wake_has_pending=True)
    ).fleet_section.notice == (
        _automatic_work_pause_notice("history_unavailable", results_saved=True)
    )


def test_finite_codec_roundtrips_original_records_without_authority_or_clocks():
    zero = _fleet_row_from_summary(SubAgentSummary("zero", budget_tokens=0), 0)
    unknown = _fleet_row_from_summary(SubAgentSummary("unknown", budget_tokens=None), 1)
    assert "0 budget tok" in zero.secondary_text
    assert "Budget unavailable" in unknown.secondary_text
    original = ConsoleAgentRailProjection(
        ("Agent: idle", "", ""),
        "",
        ConsoleInspectorSectionState((zero, unknown), "", ""),
        False,
        0,
        False,
    )
    wire = encode_agent_rail_projection(original)
    decoded = decode_agent_rail_projection(
        json.loads(json.dumps(wire, allow_nan=False))
    )
    assert decoded == original
    assert type(decoded.fleet_section) is ConsoleInspectorSectionState
    assert type(decoded.fleet_section.rows[0]) is InspectorSectionRow
    assert set(wire["fleet_section"]["rows"][0]) == {
        "row_id",
        "primary_text",
        "secondary_text",
        "status",
        "clickable",
        "cancellable",
        "indent",
        "wrap_secondary",
    }
    wire["fleet_section"]["rows"][0]["primary_text"] = "changed wire"
    assert decoded.fleet_section.rows[0] == zero
    with pytest.raises(ConsoleProtocolError):
        decode_agent_rail_projection({**wire, "bridge": object()})
    with pytest.raises(ConsoleProtocolError):
        decode_agent_rail_projection({**wire, "fleet_token_total": float("nan")})
    with pytest.raises(ConsoleProtocolError):
        decode_agent_rail_projection({**wire, "overview_lines": ["too short"]})
