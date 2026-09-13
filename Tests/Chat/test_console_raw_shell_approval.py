from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
from tldw_chatbook.Agents.run_context import use_tool_call_id
from tldw_chatbook.Chat.console_chat_controller import build_raw_shell_review_hook
from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
from tldw_chatbook.MCP.permission_store import EffectiveToolState


ASK = EffectiveToolState(state="ask", origin="global_default")


def _runtime() -> RawCliRuntime:
    runtime = RawCliRuntime(lambda: True, executor=Mock())
    assert runtime.arm().armed is True
    return runtime


def _provider(
    tmp_path: Path, runtime: RawCliRuntime | None = None
) -> RawShellToolProvider:
    return RawShellToolProvider(
        runtime=runtime or _runtime(),
        console_session_id="console-session",
        initial_directory=lambda: tmp_path,
        resolve_state=lambda _hub: ASK,
    )


def test_pending_row_discloses_the_complete_validated_raw_request(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    command = "printf 'first line\\n'\nprintf 'second line\\n'"
    pending = provider.pending_gate_for(
        ToolCall(
            "shell_exec",
            {"command": command, "shell": "bash", "timeout_seconds": 17},
            "call-raw-1",
        )
    )

    assert pending is not None
    assert pending.call_id == "call-raw-1"
    assert pending.full_command == command
    assert pending.arguments == {
        "command": command,
        "shell": "bash",
        "initial_directory": str(tmp_path),
        "timeout_seconds": 17.0,
    }
    assert pending.options == ("approve_once", "approve_session", "deny")
    assert "full authority of the OS user" in pending.warning
    assert "local log" in pending.warning
    assert "future raw shell commands" in pending.scope_notice


def test_runtime_model_session_grants_are_memory_only_and_clear_on_disarm() -> None:
    runtime = _runtime()
    assert runtime.model_session_granted("console-a") is False

    runtime.grant_model_session("console-a")
    runtime.grant_model_session("console-b")

    assert runtime.model_session_granted("console-a") is True
    assert runtime.revoke_model_sessions() == ("console-a", "console-b")
    assert runtime.model_session_granted("console-a") is False

    runtime.grant_model_session("console-a")
    runtime.disarm()
    assert runtime.model_session_granted("console-a") is False
    assert _runtime().model_session_granted("console-a") is False

    shutdown_runtime = _runtime()
    shutdown_runtime.grant_model_session("console-a")
    shutdown_runtime.shutdown()
    assert shutdown_runtime.model_session_granted("console-a") is False


def test_review_hook_keeps_repeated_raw_calls_independent(tmp_path: Path) -> None:
    runtime = _runtime()
    provider = _provider(tmp_path, runtime)
    calls = [
        ToolCall("shell_exec", {"command": "printf allowed"}, "call-a"),
        ToolCall("shell_exec", {"command": "printf denied"}, "call-b"),
    ]

    def request(rows):
        assert [(row.call_id, row.full_command) for row in rows] == [
            ("call-a", "printf allowed"),
            ("call-b", "printf denied"),
        ]
        return {"call-a": "approve_once", "call-b": "deny"}

    hook = build_raw_shell_review_hook(provider, request)
    verdicts = hook(calls, "run-1")

    assert verdicts["call-a"] == "proceed"
    assert "denied by the user" in verdicts["call-b"]
    with use_tool_call_id("call-a"):
        assert provider._pop_stamp("run-1", "shell_exec") == "approve_once"
    with use_tool_call_id("call-b"):
        assert provider._pop_stamp("run-1", "shell_exec") == "deny"

    denied = provider.invoke("shell_exec", calls[1].args)
    assert denied.ok is False
    assert runtime._executor.execute.call_count == 0


def test_review_hook_clears_stale_stamps_before_a_raising_round(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    call = ToolCall("shell_exec", {"command": "printf stale"}, "call-a")
    pending = provider.pending_gate_for(call)
    assert pending is not None
    provider.apply_batch_decisions("run-1", {"call-a": "approve_once"}, [pending])

    def raise_during_review(_rows):
        raise RuntimeError("approval bridge unavailable")

    hook = build_raw_shell_review_hook(provider, raise_during_review)
    with pytest.raises(RuntimeError, match="approval bridge unavailable"):
        hook([call], "run-1")

    with use_tool_call_id("call-a"):
        assert provider._pop_stamp("run-1", "shell_exec") is None


def test_session_decision_grants_only_this_console_session(tmp_path: Path) -> None:
    runtime = _runtime()
    provider = _provider(tmp_path, runtime)
    call = ToolCall("shell_exec", {"command": "printf session"}, "call-a")

    hook = build_raw_shell_review_hook(
        provider,
        lambda _rows: {"call-a": "approve_session"},
    )
    assert hook([call], "run-1") == {"call-a": "proceed"}

    assert runtime.model_session_granted("console-session") is True
    assert runtime.model_session_granted("different-session") is False
    assert provider.pending_gate_for(call) is None
