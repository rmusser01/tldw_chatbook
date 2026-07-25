"""HITL confirm + bridge closure for run_skill_script.

Covers two halves of the trust-gated skill-script-execution flow (see
``.superpowers/sdd/task-5-brief.md``):

1. The controller-side worker-thread <-> UI-thread bridge
   (``ConsoleChatController.request_skill_script_confirm`` /
   ``resolve_pending_skill_script`` / ``_deny_pending_skill_script_on_
   context_change``) -- mirrors ``request_skill_install_confirm`` (see
   ``Tests/UI/test_console_skill_install_confirm.py``) but carries a
   two-part ``{"allow", "remember"}`` decision instead of a plain bool.
2. The ``run_skill_script`` closure ``console_agent_bridge.run_reply``
   builds and hands to ``AgentService`` -- exercised through the REAL
   closure (captured off a real ``run_reply`` call by intercepting the
   ``AgentService(...)`` construction site), never a reimplementation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import pytest

import tldw_chatbook.Chat.console_agent_bridge as console_agent_bridge_module
from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.agent_service import AgentService as _RealAgentService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Skills_Interop.local_skills_service import ScriptPlan
from tldw_chatbook.Skills_Interop.skill_script_runner import ScriptRunResult


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    """Poll ``predicate`` until it is truthy or ``timeout`` elapses.

    Args:
        predicate: Zero-arg callable checked every 10ms.
        timeout: Max seconds to wait before failing the assertion.

    Raises:
        AssertionError: ``predicate`` never became truthy in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"condition not met within {timeout}s")


class _FakeApp:
    """`call_from_thread` stand-in: invokes the callback immediately.

    Mirrors the identical fake in ``Tests/UI/test_console_mcp_approval.py``
    and ``Tests/UI/test_console_skill_install_confirm.py``.
    """

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


@pytest.fixture
def make_controller() -> Callable[[], ConsoleChatController]:
    """Factory fixture: a fresh ``ConsoleChatController`` with a fake UI wired.

    Each call returns a new controller with ``app``/``set_pending_skill_
    script`` already wired to a ``_FakeApp`` + a list-collecting sink, so
    round-trip tests only need to append a decision. Tests that want the
    "no UI" fail-closed path override ``controller.app``/``controller.
    set_pending_skill_script`` back to ``None`` themselves.
    """

    def _make() -> ConsoleChatController:
        store = ConsoleChatStore()
        controller = ConsoleChatController(store=store, provider_gateway=object())
        controller.app = _FakeApp()
        controller.set_pending_skill_script = lambda payload: None
        return controller

    return _make


# -- bridge closure fixtures ------------------------------------------------


@dataclass
class _ClosureEnv:
    """Handle for driving and inspecting the REAL ``run_skill_script`` closure.

    Attributes:
        closure: The exact closure ``console_agent_bridge.run_reply`` built
            and handed to ``AgentService(run_skill_script_tool=...)``.
        confirm_calls: Every payload dict passed to the confirm callback,
            in call order.
        run_calls: Every ``(skill_name, script_path, args)`` the fake scope
            service's ``run_skill_script`` was invoked with.
        granted_names: Every skill name the fake trust service's
            ``grant_script_execution`` was called with.
    """

    closure: Callable[[str, str, list], ToolResult]
    confirm_calls: list[dict[str, Any]]
    run_calls: list[tuple[str, str, list[str]]]
    granted_names: list[str]


class _FakeTrustService:
    """Controllable stand-in for ``SkillTrustService``'s two script-grant methods."""

    def __init__(self, *, granted: bool) -> None:
        self._granted = granted
        self.granted_names: list[str] = []

    def script_execution_granted(self, skill_name: str) -> bool:
        return self._granted

    def grant_script_execution(self, skill_name: str) -> None:
        self.granted_names.append(skill_name)


class _FakeLocalService:
    """Stand-in for ``LocalSkillsService`` exposing only ``trust_service``."""

    def __init__(self, trust_service: _FakeTrustService) -> None:
        self.trust_service = trust_service


class _FakeScopeService:
    """Controllable stand-in for ``SkillsScopeService``'s script-run seams."""

    def __init__(
        self,
        *,
        trust_service: _FakeTrustService,
        enforce_side_effect: Exception | None,
        describe_side_effect: Exception | None,
        run_result: ScriptRunResult,
        run_calls: list[tuple[str, str, list[str]]],
    ) -> None:
        self.local_service = _FakeLocalService(trust_service)
        self._enforce_side_effect = enforce_side_effect
        self._describe_side_effect = describe_side_effect
        self._run_result = run_result
        self._run_calls = run_calls

    def enforce_run_script(self) -> None:
        if self._enforce_side_effect is not None:
            raise self._enforce_side_effect

    async def get_context(self, *, mode: str = "local") -> dict[str, Any]:
        return {"available_skills": [], "blocked_skills": []}

    async def describe_skill_script(
        self, skill_name: str, script_path: str, *, mode: str | None = None
    ) -> ScriptPlan:
        if self._describe_side_effect is not None:
            raise self._describe_side_effect
        return ScriptPlan(
            skill_name=skill_name,
            script_path=script_path,
            mechanism="interpreter",
            interpreter_display="python3",
            is_binary=False,
        )

    async def run_skill_script(
        self,
        skill_name: str,
        script_path: str,
        args,
        *,
        mode: str | None = None,
    ) -> ScriptRunResult:
        self._run_calls.append((skill_name, script_path, list(args)))
        return self._run_result


def _capture_run_skill_script_tool(
    tmp_path,
    monkeypatch,
    *,
    scope: Any,
    request_skill_script_confirm: Callable[[dict], dict] | None,
) -> Callable[[str, str, list], ToolResult] | None:
    """Build a real ``ConsoleAgentBridge`` and run one plain-text turn to
    capture the exact ``run_skill_script_tool`` closure ``run_reply``
    constructs, by intercepting the ``AgentService(...)`` call site.

    This deliberately never invokes the closure itself (the scripted
    provider turn returns plain text, no tool call) -- it only harvests the
    closure object so callers can drive it directly afterwards, exercising
    the REAL implementation rather than a reimplementation of it.

    Args:
        tmp_path: Pytest tmp dir, used for a throwaway AgentRunsDB.
        monkeypatch: Pytest monkeypatch fixture.
        scope: The (possibly fake) skills scope service to wire in.
        request_skill_script_confirm: Confirm callback to forward to
            ``run_reply``, or None to exercise the "tool absent" path.

    Returns:
        The captured ``run_skill_script_tool`` kwarg (a callable, or None
        when no closure was built).
    """

    class _ChunkGateway:
        """Minimal provider gateway: yields one plain-text reply, no tool calls."""

        async def stream_chat(self, resolution, messages, tools=None):
            yield "ok"

    captured: dict[str, Any] = {}
    real_agent_service = console_agent_bridge_module.AgentService

    class _CapturingAgentService(real_agent_service):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        console_agent_bridge_module, "AgentService", _CapturingAgentService
    )

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(),
        skills_service=scope,
    )
    kwargs: dict[str, Any] = dict(
        conversation_id="conv-script-closure",
        session_id=session.id,
        resolution=object(),
        assistant_message_id=assistant.id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
    )
    if request_skill_script_confirm is not None:
        kwargs["request_skill_script_confirm"] = request_skill_script_confirm
    bridge.run_reply(**kwargs)

    assert real_agent_service is _RealAgentService  # sanity: patched the real class
    return captured.get("run_skill_script_tool")


@pytest.fixture
def bridge_closure_env(
    tmp_path, monkeypatch
) -> Callable[..., _ClosureEnv]:
    """Factory fixture: builds a ``_ClosureEnv`` around the REAL closure.

    Every keyword configures one seam:
      - ``enforce_side_effect``: exception raised by ``enforce_run_script``.
      - ``describe_side_effect``: exception raised by ``describe_skill_script``.
      - ``granted``: whether the fake trust service reports a standing grant.
      - ``confirm_result``: the dict the confirm callback returns.
      - ``confirm_side_effect``: exception raised by the confirm callback.
      - ``run_result_exit_code``/``run_result_stdout``/``run_result_stderr``:
        fields of the ``ScriptRunResult`` the fake ``run_skill_script`` returns.
    """

    def _make(
        *,
        enforce_side_effect: Exception | None = None,
        describe_side_effect: Exception | None = None,
        granted: bool = False,
        confirm_result: dict[str, bool] | None = None,
        confirm_side_effect: Exception | None = None,
        run_result_exit_code: int = 0,
        run_result_stdout: str = "",
        run_result_stderr: str = "",
    ) -> _ClosureEnv:
        if confirm_result is None:
            confirm_result = {"allow": True, "remember": False}

        confirm_calls: list[dict[str, Any]] = []
        run_calls: list[tuple[str, str, list[str]]] = []
        trust_service = _FakeTrustService(granted=granted)
        run_result = ScriptRunResult(
            exit_code=run_result_exit_code,
            stdout=run_result_stdout,
            stderr=run_result_stderr,
            timed_out=False,
            output_capped=False,
            duration_seconds=0.01,
            truncated_stdout=False,
            truncated_stderr=False,
            sandbox_warnings=(),
        )
        scope = _FakeScopeService(
            trust_service=trust_service,
            enforce_side_effect=enforce_side_effect,
            describe_side_effect=describe_side_effect,
            run_result=run_result,
            run_calls=run_calls,
        )

        def confirm(payload: dict[str, Any]) -> dict[str, bool]:
            confirm_calls.append(payload)
            if confirm_side_effect is not None:
                raise confirm_side_effect
            return confirm_result

        closure = _capture_run_skill_script_tool(
            tmp_path,
            monkeypatch,
            scope=scope,
            request_skill_script_confirm=confirm,
        )
        assert closure is not None, "expected run_reply to build the closure"
        return _ClosureEnv(
            closure=closure,
            confirm_calls=confirm_calls,
            run_calls=run_calls,
            granted_names=trust_service.granted_names,
        )

    return _make


@dataclass
class _BridgeWithoutConfirm:
    """Handle exposing the captured (absent) ``run_skill_script_tool``."""

    run_skill_script_tool: Callable[[str, str, list], ToolResult] | None


@pytest.fixture
def bridge_without_confirm(tmp_path, monkeypatch) -> _BridgeWithoutConfirm:
    """A skills service is wired but NO confirm callback is passed to
    ``run_reply`` -- the tool must be entirely absent (never built), matching
    the sibling ``install_skill`` "advertised must equal usable" lesson."""
    trust_service = _FakeTrustService(granted=False)
    scope = _FakeScopeService(
        trust_service=trust_service,
        enforce_side_effect=None,
        describe_side_effect=None,
        run_result=ScriptRunResult(
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            output_capped=False,
            duration_seconds=0.0,
            truncated_stdout=False,
            truncated_stderr=False,
            sandbox_warnings=(),
        ),
        run_calls=[],
    )
    tool = _capture_run_skill_script_tool(
        tmp_path, monkeypatch, scope=scope, request_skill_script_confirm=None
    )
    return _BridgeWithoutConfirm(run_skill_script_tool=tool)


# -- Step 3a: controller HITL bridge ----------------------------------------


def test_no_ui_bridge_denies_immediately(make_controller):
    """Headless must fail closed at once, not block for the full timeout."""
    controller = make_controller()
    controller.app = None
    controller.set_pending_skill_script = None
    decision = controller.request_skill_script_confirm({"skill_name": "demo"})
    assert decision == {"allow": False, "remember": False}


def test_allow_round_trip(make_controller):
    controller = make_controller()
    result = {}

    def worker():
        result["decision"] = controller.request_skill_script_confirm(
            {"skill_name": "demo", "script_path": "scripts/hello.py"}
        )

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)
    controller.resolve_pending_skill_script(True, False)
    thread.join(timeout=5)
    assert result["decision"] == {"allow": True, "remember": False}


def test_always_allow_round_trip(make_controller):
    controller = make_controller()
    result = {}

    def worker():
        result["decision"] = controller.request_skill_script_confirm({"skill_name": "demo"})

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)
    controller.resolve_pending_skill_script(True, True)
    thread.join(timeout=5)
    assert result["decision"] == {"allow": True, "remember": True}


def test_context_change_denies_a_pending_confirm(make_controller):
    controller = make_controller()
    result = {}

    def worker():
        result["decision"] = controller.request_skill_script_confirm({"skill_name": "demo"})

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)
    controller._deny_pending_skill_script_on_context_change()
    thread.join(timeout=5)
    assert result["decision"]["allow"] is False


def test_switch_session_denies_a_pending_skill_script_confirm(make_controller):
    """`switch_session` must deny any pending script confirm, mirroring the
    identical wiring for `_deny_pending_skill_install_on_context_change`."""
    controller = make_controller()
    controller.store.ensure_session()
    other = controller.store.ensure_session()
    result = {}

    def worker():
        result["decision"] = controller.request_skill_script_confirm({"skill_name": "demo"})

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)
    controller.switch_session(other.id)
    thread.join(timeout=5)
    assert result["decision"]["allow"] is False


# -- Step 3b: bridge closure --------------------------------------------


def test_closure_denies_on_policy_without_prompting(bridge_closure_env):
    """Policy denial must not show a card."""
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError

    env = bridge_closure_env(
        enforce_side_effect=PolicyDeniedError(
            action_id="skills.run_script.launch.local",
            reason_code="authority_denied",
            user_message="Script execution is disabled by policy.",
            effective_source="local",
            authority_owner="local",
        )
    )
    result = env.closure("demo", "scripts/hello.py", [])
    assert result.ok is False
    assert "policy" in result.error.lower() or "disabled" in result.error.lower()
    assert env.confirm_calls == []


def test_closure_denies_on_bad_path_without_prompting(bridge_closure_env):
    env = bridge_closure_env(
        describe_side_effect=ValueError("local_skill_script_not_found:../x.py")
    )
    result = env.closure("demo", "../x.py", [])
    assert result.ok is False
    assert env.confirm_calls == []


def test_closure_skips_the_prompt_when_the_skill_is_granted(bridge_closure_env):
    env = bridge_closure_env(granted=True)
    result = env.closure("demo", "scripts/hello.py", [])
    assert result.ok is True
    assert env.confirm_calls == [], "a standing grant must not re-prompt"
    assert env.run_calls, "the script must still actually run"


def test_closure_records_the_grant_on_always_allow(bridge_closure_env):
    env = bridge_closure_env(confirm_result={"allow": True, "remember": True})
    env.closure("demo", "scripts/hello.py", [])
    assert env.granted_names == ["demo"]


def test_closure_denies_when_the_user_declines(bridge_closure_env):
    env = bridge_closure_env(confirm_result={"allow": False, "remember": False})
    result = env.closure("demo", "scripts/hello.py", [])
    assert result.ok is False
    assert "declined" in result.error.lower()
    assert env.run_calls == []


def test_closure_fails_closed_when_confirm_raises(bridge_closure_env):
    env = bridge_closure_env(confirm_side_effect=RuntimeError("ui exploded"))
    result = env.closure("demo", "scripts/hello.py", [])
    assert result.ok is False
    assert env.run_calls == []


def test_nonzero_exit_is_ok_true_with_the_failure_described(bridge_closure_env):
    """A failed SCRIPT is a successful TOOL CALL -- the agent must see it."""
    env = bridge_closure_env(run_result_exit_code=3, run_result_stderr="boom")
    result = env.closure("demo", "scripts/hello.py", [])
    assert result.ok is True
    assert "3" in result.content
    assert "boom" in result.content


def test_tool_is_absent_without_a_confirm_callback(bridge_without_confirm):
    """Advertised must equal usable (the #847 lesson)."""
    assert bridge_without_confirm.run_skill_script_tool is None
