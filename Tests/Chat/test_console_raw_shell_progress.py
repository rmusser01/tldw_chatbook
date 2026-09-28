from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    STEP_TOOL_CALL,
    STEP_TOOL_RESULT,
    AgentStep,
    ToolCall,
)
from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.raw_cli_executor import (
    RawCliRequest,
    RawCliResult,
    RawCliStreamEvent,
)


def _bridge(tmp_path: Path) -> tuple[ConsoleAgentBridge, ConsoleChatStore, str]:
    store = ConsoleChatStore()
    session = store.create_session(title="raw progress")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="run")
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "runs.db", client_id="test"),
        store=store,
        provider_gateway=object(),
    )
    return bridge, store, session.id


def _call_step(tmp_path: Path, call_id: str, command: str) -> AgentStep:
    return AgentStep(
        index=0,
        kind=STEP_TOOL_CALL,
        tool_name="shell_exec",
        args={
            "command": command,
            "shell": "bash",
            "initial_directory": str(tmp_path),
        },
        call_id=call_id,
    )


def _raw_result(tmp_path: Path, call_id: str, output: str) -> RawCliResult:
    return RawCliResult(
        invocation_id=call_id,
        caller="model",
        resolved_shell="/bin/bash",
        initial_directory=tmp_path,
        elapsed_seconds=0.25,
        stdout_preview=output,
        stderr_preview="",
        record_output=output,
        exit_code=0,
        terminal_state="exited",
        truncated=False,
        cleanup_proven=True,
    )


def _tool_markers(store: ConsoleChatStore, session_id: str):
    return [
        message
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.TOOL
    ]


def test_two_model_calls_update_only_their_correlated_marker_across_navigation(
    tmp_path: Path,
) -> None:
    bridge, store, session_id = _bridge(tmp_path)
    other = store.create_session(title="other")
    assert bridge._project_raw_shell_step(  # noqa: SLF001 - focused seam contract
        session_id,
        "run-1",
        _call_step(tmp_path, "call-a", "printf alpha"),
        AGENT_KIND_PRIMARY,
    )
    assert bridge._project_raw_shell_step(  # noqa: SLF001 - focused seam contract
        session_id,
        "run-1",
        _call_step(tmp_path, "call-b", "printf beta"),
        AGENT_KIND_PRIMARY,
    )
    original = {
        marker.raw_cli_presentation.command: marker.id
        for marker in _tool_markers(store, session_id)
    }
    assert set(original) == {"printf alpha", "printf beta"}

    store.switch_session(other.id)
    barrier = threading.Barrier(3)

    def emit(call_id: str, text: str) -> None:
        barrier.wait()
        bridge.raw_shell_progress_sink(
            "run-1",
            call_id,
            RawCliStreamEvent(
                stream="stdout",
                text=text,
                total_bytes=len(text),
                truncated=False,
            ),
        )

    first = threading.Thread(target=emit, args=("call-a", "alpha-only"))
    second = threading.Thread(target=emit, args=("call-b", "beta-only"))
    first.start()
    second.start()
    barrier.wait()
    first.join(2.0)
    second.join(2.0)

    store.switch_session(session_id)
    by_command = {
        marker.raw_cli_presentation.command: marker
        for marker in _tool_markers(store, session_id)
    }
    assert "alpha-only" in by_command["printf alpha"].tool_output_full
    assert "beta-only" not in by_command["printf alpha"].tool_output_full
    assert "beta-only" in by_command["printf beta"].tool_output_full
    assert "alpha-only" not in by_command["printf beta"].tool_output_full
    assert {command: marker.id for command, marker in by_command.items()} == original


def test_matching_result_finalizes_the_same_marker_and_late_progress_is_ignored(
    tmp_path: Path,
) -> None:
    bridge, store, session_id = _bridge(tmp_path)
    assert bridge._project_raw_shell_step(  # noqa: SLF001 - focused seam contract
        session_id,
        "run-1",
        _call_step(tmp_path, "call-a", "printf alpha"),
        AGENT_KIND_PRIMARY,
    )
    marker_id = _tool_markers(store, session_id)[0].id
    bridge.raw_shell_progress_sink(
        "run-1", "call-a", _raw_result(tmp_path, "call-a", "final-alpha")
    )

    assert bridge._project_raw_shell_step(  # noqa: SLF001 - focused seam contract
        session_id,
        "run-1",
        AgentStep(
            index=1,
            kind=STEP_TOOL_RESULT,
            tool_name="shell_exec",
            result="terminal_state: exited\nstdout:\nfinal-alpha",
            tool_outcome="success",
            call_id="call-a",
        ),
        AGENT_KIND_PRIMARY,
    )
    markers = _tool_markers(store, session_id)
    assert [marker.id for marker in markers] == [marker_id]
    assert markers[0].raw_cli_presentation.lifecycle_state == "exited"
    assert "final-alpha" in markers[0].tool_output_full
    settled_content = markers[0].content

    bridge.raw_shell_progress_sink(
        "run-1",
        "call-a",
        RawCliStreamEvent(
            stream="stderr",
            text="too late",
            total_bytes=8,
            truncated=False,
        ),
    )
    assert _tool_markers(store, session_id)[0].content == settled_content


def test_non_shell_steps_are_left_for_the_existing_marker_path(tmp_path: Path) -> None:
    bridge, store, session_id = _bridge(tmp_path)

    assert (
        bridge._project_raw_shell_step(  # noqa: SLF001 - focused seam contract
            session_id,
            "run-1",
            AgentStep(
                index=0,
                kind=STEP_TOOL_CALL,
                tool_name="calculator",
                args={"expression": "1+1"},
                call_id="calc-1",
            ),
            AGENT_KIND_PRIMARY,
        )
        is False
    )
    assert _tool_markers(store, session_id) == []


class _StreamingRuntime:
    permitted = True
    armed = True

    def __init__(self, result: RawCliResult) -> None:
        self.result = result

    def model_session_granted(self, _session_id: str) -> bool:
        return False

    def execute(self, _request: RawCliRequest, on_event) -> RawCliResult:
        on_event(
            RawCliStreamEvent(
                stream="stdout",
                text="streamed",
                total_bytes=8,
                truncated=False,
            )
        )
        return self.result


def test_provider_forwards_stream_and_settlement_with_run_and_call_identity(
    tmp_path: Path,
) -> None:
    forwarded: list[tuple[str, str, object]] = []
    runtime = _StreamingRuntime(_raw_result(tmp_path, "call-a", "streamed"))
    provider = RawShellToolProvider(
        runtime=runtime,
        console_session_id="console-session",
        initial_directory=lambda: tmp_path,
        resolve_state=lambda _hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
        local_tools_enabled=lambda: True,
        kill_switch=lambda: False,
        progress_sink=lambda run_id, call_id, event: forwarded.append(
            (run_id, call_id, event)
        ),
    )
    call = ToolCall("shell_exec", {"command": "printf streamed"}, "call-a")
    pending = provider.pending_gate_for(call)
    assert pending is not None
    provider.apply_batch_decisions("run-1", {"call-a": "approve_once"}, [pending])

    with use_run_id("run-1"), use_tool_call_id("call-a"):
        result = provider.invoke("shell_exec", call.args)

    assert result.ok is True
    assert [(run_id, call_id) for run_id, call_id, _event in forwarded] == [
        ("run-1", "call-a"),
        ("run-1", "call-a"),
    ]
    assert isinstance(forwarded[0][2], RawCliStreamEvent)
    assert isinstance(forwarded[1][2], RawCliResult)


def test_model_shell_adopts_its_queued_marker_before_execution(tmp_path):
    """A shell approval/proposal row must not duplicate the raw execution row."""
    from dataclasses import replace

    from tldw_chatbook.Chat.console_tool_activity import ConsoleToolActivity

    bridge, store, session_id = _bridge(tmp_path)
    activity = ConsoleToolActivity(store, session_id)
    bridge._tool_activity_runs["run-1"] = activity
    step = _call_step(tmp_path, "call-a", "printf alpha")
    activity.observe(replace(step, kind="tool_proposed"), None)
    rows = _tool_markers(store, session_id)
    assert len(rows) == 1
    original_id = rows[0].id
    activity.approval(["call-a"], True)
    assert (
        _tool_markers(store, session_id)[0].activity_presentation.status
        == "awaiting_approval"
    )
    activity.approval(["call-a"], False)
    assert bridge._project_raw_shell_step(session_id, "run-1", step, AGENT_KIND_PRIMARY)
    assert [m.id for m in _tool_markers(store, session_id)] == [original_id]
    assert (
        _tool_markers(store, session_id)[0].raw_cli_presentation.command
        == "printf alpha"
    )
    bridge.raw_shell_progress_sink(
        "run-1", "call-a", _raw_result(tmp_path, "call-a", "final-alpha")
    )
    assert bridge._project_raw_shell_step(
        session_id,
        "run-1",
        replace(
            step, kind=STEP_TOOL_RESULT, result="final-alpha", tool_outcome="success"
        ),
        AGENT_KIND_PRIMARY,
    )
    activity.finish(False)
    marker = _tool_markers(store, session_id)[0]
    assert marker.id == original_id
    assert marker.raw_cli_presentation.lifecycle_state == "exited"
    assert "final-alpha" in marker.activity_presentation.result_preview


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("has_result", [False, True])
def test_teardown_settles_adopted_shell_and_preserves_executor_result(
    tmp_path, cancelled, has_result
):
    """A missing tool-result callback must not strand an adopted running row."""
    from dataclasses import replace

    from tldw_chatbook.Chat.console_tool_activity import ConsoleToolActivity
    from tldw_chatbook.Widgets.Console.console_assistant_turn import raw_cli_status_copy

    bridge, store, session_id = _bridge(tmp_path)
    clock = [10.0]
    bridge._clock = lambda: clock[0]
    activity = ConsoleToolActivity(store, session_id)
    bridge._tool_activity_runs["run"] = activity
    step = _call_step(tmp_path, "call", "printf alpha")
    activity.observe(replace(step, kind="tool_proposed"), 1)
    marker_id = _tool_markers(store, session_id)[0].id
    bridge._project_raw_shell_step(session_id, "run", step, AGENT_KIND_PRIMARY)
    event = RawCliStreamEvent(
        stream="stderr", text="partial diagnostic", total_bytes=18, truncated=False
    )
    bridge.raw_shell_progress_sink("run", "call", event)
    if has_result:
        bridge.raw_shell_progress_sink(
            "run", "call", _raw_result(tmp_path, "call", "final-alpha")
        )
    other_step = _call_step(tmp_path, "other", "printf beta")
    bridge._project_raw_shell_step(
        session_id, "other-run", other_step, AGENT_KIND_PRIMARY
    )
    clock[0] = 13.0
    bridge._clear_raw_shell_progress(
        {"run"}, **({"cancelled": True} if cancelled else {})
    )
    activity.finish(cancelled)
    marker = _tool_markers(store, session_id)[0]
    assert marker.id == marker_id
    if has_result:
        assert marker.raw_cli_presentation.lifecycle_state == "exited"
        assert marker.raw_cli_presentation.cleanup_proven is True
        assert marker.raw_cli_presentation.elapsed_seconds == 0.25
        assert "final-alpha" in marker.tool_output_full
        assert "before a result" not in marker.content
    else:
        assert marker.raw_cli_presentation.lifecycle_state == (
            "cancelled" if cancelled else "failed"
        )
        assert marker.raw_cli_presentation.cleanup_proven is None
        assert marker.raw_cli_presentation.exit_code is None
        assert marker.raw_cli_presentation.elapsed_seconds == 3.0
        assert "partial diagnostic" in marker.tool_output_full
        assert "Cleanup: Unknown" in marker.content
        assert "Exit code: Unknown" in marker.content
        assert "before a result" in marker.activity_presentation.result_preview
        assert "cleanup is unknown" in marker.activity_presentation.result_preview
        assert raw_cli_status_copy(marker.raw_cli_presentation, now=99.0).endswith(
            "3.0s"
        )
    bridge.raw_shell_progress_sink("run", "call", event)
    assert _tool_markers(store, session_id)[0] == marker
    assert ("run", "call") not in bridge._raw_shell_markers
    assert ("other-run", "other") in bridge._raw_shell_markers


def test_teardown_detaches_all_shells_before_best_effort_display_updates(
    tmp_path, monkeypatch
):
    from loguru import logger

    bridge, store, session_id = _bridge(tmp_path)
    for call_id in ("first", "second"):
        bridge._project_raw_shell_step(
            session_id,
            "run",
            _call_step(tmp_path, call_id, "printf alpha"),
            AGENT_KIND_PRIMARY,
        )
    update = bridge._update_raw_shell_marker

    def broken_update(state):
        assert not bridge._raw_shell_markers
        if state.presentation.invocation_id == "first":
            raise RuntimeError("secret output must never enter diagnostics")
        update(state)

    monkeypatch.setattr(bridge, "_update_raw_shell_marker", broken_update)
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        bridge._clear_raw_shell_progress({"run"})
    finally:
        logger.remove(sink)
    assert (
        _tool_markers(store, session_id)[1].raw_cli_presentation.lifecycle_state
        == "failed"
    )
    assert any(
        "display could not be settled (RuntimeError)" in text for text in messages
    )
    assert all("secret output" not in text for text in messages)
