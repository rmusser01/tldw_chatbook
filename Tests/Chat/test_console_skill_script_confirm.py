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

    Every payload handed to ``set_pending_skill_script`` (the show-card
    call AND the later clear-to-``None`` call) is recorded, in order, onto
    ``controller.pending_skill_script_payloads`` -- tests that need to
    inspect what was actually marshaled to the UI (e.g. that it carries
    ``timeout_seconds``/``request_id``) read that list instead of the
    payloads being silently discarded.
    """

    def _make() -> ConsoleChatController:
        store = ConsoleChatStore()
        controller = ConsoleChatController(store=store, provider_gateway=object())
        controller.app = _FakeApp()
        controller.pending_skill_script_payloads = []
        controller.set_pending_skill_script = (
            controller.pending_skill_script_payloads.append
        )
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
        # The real LocalSkillsService normalizes the name it acts on before
        # putting it in the plan (`_canonical_skill_name`), so this fake does
        # too -- otherwise a test could not tell whether the confirm payload
        # carries the agent's raw spelling or the value that will be used.
        from tldw_chatbook.tldw_api.skills_schemas import _normalize_skill_name

        return ScriptPlan(
            skill_name=_normalize_skill_name(skill_name),
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
    controller.resolve_pending_skill_script(
        True, False, request_id=controller._pending_skill_script_request_id
    )
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
    controller.resolve_pending_skill_script(
        True, True, request_id=controller._pending_skill_script_request_id
    )
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


def test_confirm_payload_carries_timeout_and_request_id(make_controller):
    """The payload actually marshaled to the UI sink must carry both the
    timeout and a per-round request id (not just the caller's own keys) --
    a card built from an under-described payload is a security defect,
    since that payload is exactly what the human approves on."""
    controller = make_controller()
    result = {}

    def worker():
        result["decision"] = controller.request_skill_script_confirm(
            {"skill_name": "demo", "script_path": "scripts/hello.py"}
        )

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)
    shown = controller.pending_skill_script_payloads[0]
    assert shown is not None
    assert shown["skill_name"] == "demo"
    assert shown["script_path"] == "scripts/hello.py"
    assert isinstance(shown["timeout_seconds"], float)
    assert shown["timeout_seconds"] > 0
    assert shown["request_id"] == controller._pending_skill_script_request_id
    assert shown["request_id"]  # non-empty
    controller.resolve_pending_skill_script(True, False, request_id=shown["request_id"])
    thread.join(timeout=5)
    assert result["decision"] == {"allow": True, "remember": False}
    # The clearing call at teardown hands `None`, not a second payload.
    assert controller.pending_skill_script_payloads[-1] is None


def test_stale_request_id_is_dropped_then_matching_id_resolves(make_controller):
    """Security-critical: a resolve carrying a PRIOR round's id must not
    authorize the CURRENT round -- see `resolve_pending_skill_script`'s
    docstring for the exact late-button-press scenario this closes."""
    controller = make_controller()

    # Round 1: arm, capture its id, then deny it via context-change so it
    # tears down (clearing _pending_skill_script_request_id) without ever
    # being resolved by a matching id.
    round_one_result = {}

    def round_one():
        round_one_result["decision"] = controller.request_skill_script_confirm(
            {"skill_name": "demo"}
        )

    t1 = threading.Thread(target=round_one)
    t1.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)
    stale_id = controller._pending_skill_script_request_id
    assert stale_id
    controller._deny_pending_skill_script_on_context_change()
    t1.join(timeout=5)
    assert round_one_result["decision"]["allow"] is False
    assert controller._pending_skill_script_request_id is None  # torn down

    # Round 2: arms a fresh id. A resolve carrying round 1's stale id must
    # be dropped -- the round stays armed, unresolved.
    round_two_result = {}

    def round_two():
        round_two_result["decision"] = controller.request_skill_script_confirm(
            {"skill_name": "demo"}
        )

    t2 = threading.Thread(target=round_two)
    t2.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)
    fresh_id = controller._pending_skill_script_request_id
    assert fresh_id and fresh_id != stale_id

    controller.resolve_pending_skill_script(True, False, request_id=stale_id)
    time.sleep(0.1)
    assert t2.is_alive(), "a stale request_id must not resolve the armed round"
    assert controller._pending_skill_script_event is not None  # still armed

    # The matching (current) id resolves it correctly.
    controller.resolve_pending_skill_script(True, False, request_id=fresh_id)
    t2.join(timeout=5)
    assert round_two_result["decision"] == {"allow": True, "remember": False}


def test_resolve_with_no_request_id_is_dropped(make_controller):
    """A resolve carrying no id at all (e.g. a not-yet-migrated caller)
    must be dropped by design, same as a stale one."""
    controller = make_controller()
    result = {}

    def worker():
        result["decision"] = controller.request_skill_script_confirm({"skill_name": "demo"})

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: controller._pending_skill_script_event is not None)

    controller.resolve_pending_skill_script(True, False)  # request_id omitted
    time.sleep(0.1)
    assert thread.is_alive(), "an id-less resolve must not resolve the armed round"

    controller.resolve_pending_skill_script(
        True, False, request_id=controller._pending_skill_script_request_id
    )
    thread.join(timeout=5)
    assert result["decision"] == {"allow": True, "remember": False}


def test_confirm_timeout_denies(make_controller):
    """`skill_script_confirm_timeout_seconds` overrides the 120s default so
    the deadline path can be exercised quickly, mirroring the identical
    seam on the sibling install-confirm flow
    (`test_console_skill_install_confirm.test_confirm_timeout_denies`)."""
    controller = make_controller()
    controller.skill_script_confirm_timeout_seconds = lambda: 0.05
    started = time.monotonic()
    decision = controller.request_skill_script_confirm({"skill_name": "demo"})
    elapsed = time.monotonic() - started
    assert decision == {"allow": False, "remember": False}
    assert elapsed < 2.5


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


def test_closure_confirm_payload_describes_the_run_being_approved(bridge_closure_env):
    """The confirm payload is exactly what the human approves on -- dropping
    a field here (e.g. the script path or args) would be a real security
    defect even though every other closure test would still pass."""
    env = bridge_closure_env()
    env.closure("demo", "scripts/hello.py", ["--flag", "value"])
    assert len(env.confirm_calls) == 1
    payload = env.confirm_calls[0]
    assert payload["skill_name"] == "demo"
    assert payload["script_path"] == "scripts/hello.py"
    assert payload["mechanism"] == "interpreter"
    assert payload["args"] == ["--flag", "value"]


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


def test_tool_is_absent_on_an_unsupported_platform(tmp_path, monkeypatch):
    """Advertised must equal usable, applied to the Windows gap (Qodo #871
    finding 2): even with a skills service AND a confirm callback wired,
    the tool must not be built when the platform's sandbox is unusable.

    Simulates Windows purely through the platform predicate
    ``skill_script_runner.sandbox_supported`` -- never by actually running
    any Windows-only code path on this (POSIX) test box.
    """
    import tldw_chatbook.Skills_Interop.skill_script_runner as skill_script_runner_module

    monkeypatch.setattr(
        skill_script_runner_module, "sandbox_supported", lambda: False
    )

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
        tmp_path,
        monkeypatch,
        scope=scope,
        request_skill_script_confirm=lambda payload: {
            "allow": True,
            "remember": False,
        },
    )
    assert tool is None


# -- Step 3c: advertised must equal usable (controller-level wiring) --------


class _StubResolution:
    ready = True
    provider = "llama_cpp"
    visible_copy = ""


class _StubGateway:
    """Minimal provider gateway: `resolve_for_send` is all `submit_draft`
    needs before `run_reply` (monkeypatched to a capturing stub below) is
    invoked directly -- no real streaming ever happens in these tests."""

    async def resolve_for_send(self, selection):
        return _StubResolution()

    async def stream_chat(self, resolution, messages):  # pragma: no cover
        yield "unused"


def _capturing_run_reply(captured: list[dict[str, Any]]):
    """`run_reply` stand-in: records its kwargs instead of running the
    agent loop, so these tests can inspect exactly what the CONTROLLER
    decided to forward to the bridge."""
    from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome

    def run_reply(**kwargs):
        captured.append(kwargs)
        return "run-test", RunOutcome(status=RUN_DONE, steps=[], final_text="ok.")

    return run_reply


def _bridged_controller(tmp_path) -> tuple[ConsoleChatController, list[dict[str, Any]]]:
    gateway = _StubGateway()
    store = ConsoleChatStore()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    captured: list[dict[str, Any]] = []
    controller._agent_bridge.run_reply = _capturing_run_reply(captured)
    return controller, captured


@pytest.mark.asyncio
async def test_confirm_callback_absent_from_bridge_when_no_ui_sink_wired(tmp_path):
    """The #847 lesson, applied to run_skill_script: with no
    `set_pending_skill_script` wired, the controller must NOT forward
    `request_skill_script_confirm` to the bridge at all -- passing the
    (always fail-closed) bound method anyway would advertise a tool the
    model can never successfully use."""
    controller, captured = _bridged_controller(tmp_path)
    assert controller.set_pending_skill_script is None  # not wired (default)

    result = await controller.submit_draft("hi")

    assert result.accepted is True
    assert captured[0]["request_skill_script_confirm"] is None


@pytest.mark.asyncio
async def test_confirm_callback_present_when_ui_sink_wired(tmp_path):
    controller, captured = _bridged_controller(tmp_path)
    controller.set_pending_skill_script = lambda payload: None

    result = await controller.submit_draft("hi")

    assert result.accepted is True
    confirm = captured[0]["request_skill_script_confirm"]
    assert confirm is not None
    assert confirm.__self__ is controller
    assert confirm.__func__ is ConsoleChatController.request_skill_script_confirm


def test_confirm_payload_shows_the_canonical_skill_name_not_the_raw_one(
    bridge_closure_env,
):
    """The card is the consent surface, so it must name what will actually run.

    The agent supplies the skill name; the service normalizes it before
    acting. Echoing the agent's raw spelling onto the card would let a run
    be approved under a name that differs from the skill it targets.
    """
    env = bridge_closure_env(granted=False)
    result = env.closure("  DEMO-Skill ", "scripts/hello.py", [])
    assert result.ok is True
    assert env.confirm_calls[0]["skill_name"] == "demo-skill"
