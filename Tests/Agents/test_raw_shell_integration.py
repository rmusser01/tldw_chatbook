from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import RUN_DONE, AgentConfig, ToolCall
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    build_raw_shell_review_hook,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.raw_cli_executor import RawCliRequest, RawCliResult


ASK = EffectiveToolState(state="ask", origin="global_default")
OFF = EffectiveToolState(state="deny", origin="tool_override")


class _Runtime:
    def __init__(self) -> None:
        self.permitted = True
        self.armed = True
        self.grants: set[str] = set()
        self.execute_calls: list[RawCliRequest] = []
        self.terminal_state = "exited"
        self.exit_code: int | None = 0
        self.stdout = "hello from raw shell"
        self.stderr = ""
        self.truncated = False
        self.cleanup_proven = True
        self.model_authority_revoker = None

    def grant_model_session(self, console_session_id: str) -> None:
        self.grants.add(console_session_id)

    def model_session_granted(self, console_session_id: str) -> bool:
        return console_session_id in self.grants

    def set_model_authority_revoker(self, callback) -> None:
        self.model_authority_revoker = callback

    def execute(self, request: RawCliRequest, on_event, **_kwargs) -> RawCliResult:
        self.execute_calls.append(request)
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell="/bin/bash",
            initial_directory=request.initial_directory,
            elapsed_seconds=0.25,
            stdout_preview=self.stdout,
            stderr_preview=self.stderr,
            record_output=self.stdout + self.stderr,
            exit_code=self.exit_code,
            terminal_state=self.terminal_state,
            truncated=self.truncated,
            cleanup_proven=self.cleanup_proven,
        )


def _provider(
    tmp_path: Path,
    *,
    runtime: _Runtime | None = None,
    gates: dict[str, object] | None = None,
) -> tuple[RawShellToolProvider, _Runtime, dict[str, object]]:
    runtime = runtime or _Runtime()
    gates = gates or {"local": True, "blocked": False, "state": ASK}
    provider = RawShellToolProvider(
        runtime=runtime,
        console_session_id="console-session",
        initial_directory=lambda: tmp_path,
        resolve_state=lambda _hub: gates["state"],
        local_tools_enabled=lambda: gates["local"],
        kill_switch=lambda: gates["blocked"],
    )
    return provider, runtime, gates


def _approve_once(provider: RawShellToolProvider, command: str, call_id: str) -> None:
    pending = provider.pending_gate_for(
        ToolCall("shell_exec", {"command": command}, call_id)
    )
    assert pending is not None
    provider.apply_batch_decisions(
        "run-1", {call_id: "approve_once"}, [pending]
    )


@pytest.mark.parametrize("closed_gate", ["permitted", "armed", "local", "blocked", "off"])
def test_stale_provider_rechecks_every_gate_before_executor_dispatch(
    tmp_path: Path, closed_gate: str
) -> None:
    provider, runtime, gates = _provider(tmp_path)
    _approve_once(provider, "printf stale", "call-1")

    if closed_gate == "permitted":
        runtime.permitted = False
    elif closed_gate == "armed":
        runtime.armed = False
    elif closed_gate == "local":
        gates["local"] = False
    elif closed_gate == "blocked":
        gates["blocked"] = True
    else:
        gates["state"] = OFF

    with use_run_id("run-1"), use_tool_call_id("call-1"):
        result = provider.invoke("shell_exec", {"command": "printf stale"})

    assert result.ok is False
    assert result.outcome == "blocked"
    assert runtime.execute_calls == []


def test_approved_call_reuses_runtime_and_returns_bounded_success(tmp_path: Path) -> None:
    provider, runtime, _gates = _provider(tmp_path)
    runtime.stdout = "x" * 10_000
    runtime.truncated = True
    _approve_once(provider, "printf hello", "call-1")

    with use_run_id("run-1"), use_tool_call_id("call-1"):
        result = provider.invoke("shell_exec", {"command": "printf hello"})

    assert result.ok is True
    assert len(runtime.execute_calls) == 1
    request = runtime.execute_calls[0]
    assert request.caller == "model"
    assert request.console_session_id == "console-session"
    assert request.command == "printf hello"
    assert len(result.content) <= 4000
    assert "terminal_state: exited" in result.content
    assert "truncated: true" in result.content
    assert "cleanup_proven: true" in result.content


@pytest.mark.parametrize(
    ("terminal_state", "exit_code", "outcome", "detail"),
    (
        ("exited", 9, "failed", "exit_code: 9"),
        ("timed_out", None, "timeout", "terminal_state: timed_out"),
        ("cancelled", None, "cancelled", "terminal_state: cancelled"),
        ("refused", None, "blocked", "terminal_state: refused"),
        ("spawn_failed", None, "failed", "terminal_state: spawn_failed"),
        (
            "containment_unavailable",
            None,
            "failed",
            "terminal_state: containment_unavailable",
        ),
    ),
)
def test_terminal_results_map_to_stable_tool_outcomes(
    tmp_path: Path,
    terminal_state: str,
    exit_code: int | None,
    outcome: str,
    detail: str,
) -> None:
    provider, runtime, _gates = _provider(tmp_path)
    runtime.terminal_state = terminal_state
    runtime.exit_code = exit_code
    runtime.stderr = "diagnostic"
    _approve_once(provider, "printf result", "call-1")

    with use_run_id("run-1"), use_tool_call_id("call-1"):
        result = provider.invoke("shell_exec", {"command": "printf result"})

    assert result.ok is False
    assert result.outcome == outcome
    assert detail in result.error
    assert "cleanup_proven: true" in result.error


class _PermissionService:
    def __init__(self, gates: dict[str, object]) -> None:
        self.gates = gates

    def get_kill_switch(self) -> bool:
        return bool(self.gates["blocked"])

    def gate_tool_test(self, _hub) -> EffectiveToolState:
        return self.gates["state"]


@pytest.mark.parametrize(
    ("permitted", "armed", "local", "blocked", "expected"),
    (
        (False, True, True, False, False),
        (True, False, True, False, False),
        (True, True, False, False, False),
        (True, True, True, True, False),
        (True, True, True, False, True),
    ),
)
def test_controller_composes_raw_provider_only_while_all_live_gates_are_open(
    tmp_path: Path,
    permitted: bool,
    armed: bool,
    local: bool,
    blocked: bool,
    expected: bool,
) -> None:
    runtime = _Runtime()
    runtime.permitted = permitted
    runtime.armed = armed
    gates: dict[str, object] = {"blocked": blocked, "state": ASK}
    controller = object.__new__(ConsoleChatController)
    controller.app = SimpleNamespace(
        unified_mcp_service=_PermissionService(gates),
        raw_cli_runtime=runtime,
    )
    turn_context = SimpleNamespace(
        tool_configuration={"local_tools_enabled": local},
        scratch_space=SimpleNamespace(root=tmp_path),
    )

    provider, hook = controller._compose_raw_shell_provider(
        session_id="console-session",
        turn_context=turn_context,
    )

    assert (provider is not None) is expected
    assert (hook is not None) is expected


class _ScriptedChat:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


def _fence(name: str, arguments: dict[str, object]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"```tool_call\n{payload}\n```"


def test_agent_service_executes_raw_shell_as_an_ordinary_tool_result(
    tmp_path: Path,
) -> None:
    provider, runtime, _gates = _provider(tmp_path)
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    approval_rows = []

    def approve_once(rows):
        approval_rows.extend(rows)
        return {(row.call_id or row.llm_name): "approve_once" for row in rows}

    chat = _ScriptedChat(
        [
            _fence("shell_exec", {"command": "printf integration"}),
            "The raw command completed.",
        ]
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=build_raw_shell_review_hook(provider, approve_once),
        review_state_scope=provider.stamp_scope,
    )

    _run_id, outcome = service.run_turn(
        conversation_id="raw-shell-e2e",
        messages=[{"role": "user", "content": "run the command"}],
        config=AgentConfig(
            model="test-model",
            system_prompt="You are helpful.",
            allowed_tools=("shell_exec",),
        ),
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "The raw command completed."
    assert len(runtime.execute_calls) == 1
    assert len(approval_rows) == 1
    tool_results = [step for step in outcome.steps if step.kind == "tool_result"]
    assert [step.tool_name for step in tool_results] == ["shell_exec"]
    assert "hello from raw shell" in tool_results[0].result
    assert any(
        message["role"] == "user"
        and message["content"].startswith("Tool result for shell_exec: ")
        and "hello from raw shell" in message["content"]
        for message in chat.calls[1]["messages_payload"]
    )
    rows = db.list_runs("raw-shell-e2e", include_superseded=True)
    assert rows
    assert all(row["agent_kind"] != "local_command" for row in rows)
