from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.builtin_tool_gate import user_denial_refusal
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.raw_cli_executor import RawCliRequest, RawCliResult
from Tests.console_provider_doubles import persisted_console_store


ASK = EffectiveToolState(state="ask", origin="global_default")


def _result(request: RawCliRequest, terminal_state: str) -> RawCliResult:
    return RawCliResult(
        invocation_id=request.invocation_id,
        caller=request.caller,
        resolved_shell=request.shell,
        initial_directory=request.initial_directory,
        elapsed_seconds=0.1,
        stdout_preview="",
        stderr_preview="",
        record_output="",
        exit_code=0 if terminal_state == "exited" else None,
        terminal_state=terminal_state,
        truncated=False,
        cleanup_proven=True,
    )


def _request(tmp_path: Path, invocation_id: str = "raw-1") -> RawCliRequest:
    return RawCliRequest(
        invocation_id=invocation_id,
        caller="model",
        command="printf revocation",
        shell="auto",
        initial_directory=tmp_path,
        timeout_seconds=30.0,
        console_session_id="console-session",
    )


class _UnexpectedExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, *_args: Any, **_kwargs: Any) -> RawCliResult:
        self.calls += 1
        raise AssertionError("revoked raw command reached the executor")


class _CancelableExecutor:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancel_event: threading.Event | None = None

    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: threading.Event,
        on_event: Any,
        admit_worker: Any,
    ) -> RawCliResult:
        del on_event

        class Tree:
            def admit(self) -> None:
                return None

        assert admit_worker(Tree(), lambda: 1.0) is True
        self.cancel_event = cancel_event
        self.started.set()
        assert cancel_event.wait(2.0), "disarm did not cancel the active command"
        return _result(request, "cancelled")


class _ImmediateExecutor:
    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: threading.Event,
        on_event: Any,
        admit_worker: Any,
    ) -> RawCliResult:
        del cancel_event, on_event

        class Tree:
            def admit(self) -> None:
                return None

        assert admit_worker(Tree(), lambda: 1.0) is True
        return _result(request, "exited")


class _PermissionService:
    def get_kill_switch(self) -> bool:
        return False

    def gate_tool_test(self, _hub) -> EffectiveToolState:
        return ASK


class _FakeApp:
    def __init__(self, runtime: RawCliRuntime) -> None:
        self.raw_cli_runtime = runtime
        self.unified_mcp_service = _PermissionService()

    def call_from_thread(self, callback, *args, **kwargs):
        return callback(*args, **kwargs)


class _RoundRegistry(dict):
    def __init__(self) -> None:
        super().__init__()
        self.raw_registered = threading.Event()
        self.other_registered = threading.Event()

    def __setitem__(self, key, state) -> None:
        super().__setitem__(key, state)
        names = tuple(state.get("names") or ())
        if names == ("raw-call",):
            self.raw_registered.set()
        if names == ("other-call",):
            self.other_registered.set()


def _controller_provider_hook(
    tmp_path: Path, runtime: RawCliRuntime
) -> tuple[ConsoleChatController, Any, Any]:
    store = persisted_console_store()
    session = store.ensure_session(title="Raw shell")
    controller = ConsoleChatController(store=store, provider_gateway=object())
    controller.app = _FakeApp(runtime)
    controller.set_pending_approval = lambda _payload: None
    provider, hook = controller._compose_raw_shell_provider(
        session_id=session.id,
        turn_context=SimpleNamespace(
            tool_configuration={"local_tools_enabled": True},
            scratch_space=SimpleNamespace(root=tmp_path),
        ),
    )
    assert provider is not None
    assert hook is not None
    return controller, provider, hook


def test_disarm_denies_only_pending_raw_shell_approval_rounds(
    tmp_path: Path,
) -> None:
    executor = _UnexpectedExecutor()
    runtime = RawCliRuntime(lambda: True, executor=executor)
    assert runtime.arm().armed is True
    controller, _provider, raw_hook = _controller_provider_hook(tmp_path, runtime)
    rounds = _RoundRegistry()
    controller._pending_approval_rounds = rounds
    decisions: dict[str, dict[str, str]] = {}

    other = MCPPendingCall(
        llm_name="other_tool",
        server_key="local:__local__",
        tool_name="other_tool",
        server_label="Local tools",
        arguments={},
        reason="ask",
        options=("approve_once", "deny"),
        call_id="other-call",
    )

    def wait_for_other() -> None:
        with use_run_id("other-run"):
            decisions["other"] = controller.request_mcp_approvals(
                [other], session_id=controller.store.active_session_id
            )

    def wait_for_raw() -> None:
        with use_run_id("raw-run"):
            decisions["raw"] = raw_hook(
                [
                    ToolCall(
                        "shell_exec",
                        {"command": "printf pending"},
                        "raw-call",
                    )
                ],
                "raw-run",
            )

    other_thread = threading.Thread(target=wait_for_other)
    raw_thread = threading.Thread(target=wait_for_raw)
    other_thread.start()
    assert rounds.other_registered.wait(2.0)
    raw_thread.start()
    assert rounds.raw_registered.wait(2.0)

    runtime.disarm()
    raw_thread.join(1.0)
    raw_released_by_disarm = not raw_thread.is_alive()
    other_preserved_by_disarm = other_thread.is_alive()

    # Always release any still-pending test rounds before asserting.
    for round_id, state in list(rounds.items()):
        names = tuple(state.get("names") or ())
        controller.resolve_pending_approval(
            {name: "deny" for name in names}, round_id=round_id
        )
    raw_thread.join(2.0)
    other_thread.join(2.0)

    assert raw_released_by_disarm is True
    assert other_preserved_by_disarm is True
    # Derived from the shared constant, not hardcoded: three modules keep this
    # wording in sync (TASK-26011) and a literal here would silently drift.
    assert decisions["raw"] == {"raw-call": user_denial_refusal("shell_exec")}
    assert decisions["other"] == {"other-call": "deny"}
    assert executor.calls == 0


def test_disarm_clears_returned_approval_stamp_before_provider_invoke(
    tmp_path: Path,
) -> None:
    runtime = RawCliRuntime(lambda: True, executor=_UnexpectedExecutor())
    assert runtime.arm().armed is True
    _controller, provider, _hook = _controller_provider_hook(tmp_path, runtime)
    call = ToolCall("shell_exec", {"command": "printf approved"}, "raw-call")
    pending = provider.pending_gate_for(call)
    assert pending is not None
    provider.apply_batch_decisions(
        "raw-run", {"raw-call": "approve_once"}, [pending]
    )

    runtime.disarm()

    with use_tool_call_id("raw-call"):
        assert provider._pop_stamp("raw-run", "shell_exec") is None
    with use_run_id("raw-run"), use_tool_call_id("raw-call"):
        result = provider.invoke("shell_exec", call.args)
    assert result.ok is False
    assert result.outcome == "blocked"


def test_quick_rearm_cannot_restore_a_pre_disarm_approval(
    tmp_path: Path,
) -> None:
    runtime = RawCliRuntime(lambda: True, executor=_UnexpectedExecutor())
    assert runtime.arm().armed is True
    _controller, provider, _hook = _controller_provider_hook(tmp_path, runtime)
    call = ToolCall("shell_exec", {"command": "printf old"}, "raw-call")
    pending = provider.pending_gate_for(call)
    assert pending is not None
    old_generation = provider.authority_generation

    runtime.disarm()
    assert runtime.arm().armed is True
    provider.apply_batch_decisions(
        "raw-run",
        {"raw-call": "approve_session"},
        [pending],
        authority_generation=old_generation,
    )

    assert runtime.model_session_granted("console-session") is False
    with use_tool_call_id("raw-call"):
        assert provider._pop_stamp("raw-run", "shell_exec") is None


def test_disarm_invalidates_an_approval_hidden_by_nested_run_scope(
    tmp_path: Path,
) -> None:
    runtime = RawCliRuntime(lambda: True, executor=_UnexpectedExecutor())
    assert runtime.arm().armed is True
    _controller, provider, _hook = _controller_provider_hook(tmp_path, runtime)
    call = ToolCall("shell_exec", {"command": "printf old"}, "raw-call")
    pending = provider.pending_gate_for(call)
    assert pending is not None
    provider.apply_batch_decisions(
        "raw-run", {"raw-call": "approve_once"}, [pending]
    )

    with provider.stamp_scope("raw-run"):
        runtime.disarm()

    assert runtime.arm().armed is True
    with use_tool_call_id("raw-call"):
        assert provider._pop_stamp("raw-run", "shell_exec") is None


def test_disarm_orders_authority_revocation_before_active_cancellation(
    tmp_path: Path,
) -> None:
    executor = _CancelableExecutor()
    runtime = RawCliRuntime(lambda: True, executor=executor)
    assert runtime.arm().armed is True
    runtime.grant_model_session("console-session")
    observed: list[tuple[bool, bool, bool]] = []
    runtime.set_model_authority_revoker(
        lambda: observed.append(
            (
                runtime.armed,
                runtime.model_session_granted("console-session"),
                bool(executor.cancel_event and executor.cancel_event.is_set()),
            )
        )
    )
    results: list[RawCliResult] = []
    thread = threading.Thread(
        target=lambda: results.append(
            runtime.execute(_request(tmp_path), lambda _event: None)
        )
    )
    thread.start()
    assert executor.started.wait(2.0)

    assert runtime.disarm() == ("raw-1",)
    thread.join(2.0)

    assert observed == [(False, False, False)]
    assert results[0].terminal_state == "cancelled"
    assert results[0].cleanup_proven is True


def test_late_disarm_cannot_replace_a_settled_exit(tmp_path: Path) -> None:
    runtime = RawCliRuntime(lambda: True, executor=_ImmediateExecutor())
    assert runtime.arm().armed is True

    result = runtime.execute(_request(tmp_path), lambda _event: None)
    assert runtime.disarm() == ()

    assert result.terminal_state == "exited"
    assert result.exit_code == 0
    assert result.cleanup_proven is True


def test_saved_unlock_off_disarms_and_removes_later_schema(tmp_path: Path) -> None:
    saved = {"permitted": True}
    runtime = RawCliRuntime(lambda: saved["permitted"], executor=_UnexpectedExecutor())
    assert runtime.arm().armed is True
    controller, _provider, _hook = _controller_provider_hook(tmp_path, runtime)

    saved["permitted"] = False
    runtime.disarm()
    provider, hook = controller._compose_raw_shell_provider(
        session_id=controller.store.active_session_id,
        turn_context=SimpleNamespace(
            tool_configuration={"local_tools_enabled": True},
            scratch_space=SimpleNamespace(root=tmp_path),
        ),
    )

    assert runtime.armed is False
    assert provider is None
    assert hook is None
