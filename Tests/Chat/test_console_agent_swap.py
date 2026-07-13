"""The controller send path runs the agent loop when the bridge is wired."""
import json
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_models import (
    AgentStep, RunOutcome, RUN_DONE, RUN_ERROR, RUN_STUCK, STEP_ERROR,
)


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


class _Gateway:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = 0

    async def resolve_for_send(self, selection):
        class _R:
            ready = True
            provider = "llama_cpp"
            visible_copy = ""
        return _R()

    async def stream_chat(self, resolution, messages):
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


def _controller(tmp_path, scripts, *, enabled=True):
    gateway = _Gateway(scripts)
    store = ConsoleChatStore()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, provider="llama_cpp", model="test-model",
        agent_bridge=bridge, agent_runtime_enabled=enabled)
    return controller, store, db


def _all_runs(db):
    """Read every persisted run record directly (AgentRunsDB has no list-all)."""
    with db.connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM agent_runs").fetchall()]


@pytest.mark.asyncio
async def test_agent_send_no_tools_streams_like_today(tmp_path):
    controller, store, _db = _controller(tmp_path, [["Tok", "yo."]])
    result = await controller.submit_draft("capital of Japan?")
    assert result.accepted is True
    messages = store.messages_for_session(store.active_session_id)
    assert messages[-1].role is ConsoleMessageRole.ASSISTANT
    assert messages[-1].content == "Tokyo."


@pytest.mark.asyncio
async def test_agent_tool_turn_renders_marker_not_prose(tmp_path):
    controller, store, _db = _controller(
        tmp_path, [[_fence("calculator", {"expression": "6*7"})], ["It is ", "42."]])
    await controller.submit_draft("what is 6*7?")
    messages = store.messages_for_session(store.active_session_id)
    assert any(m.role is ConsoleMessageRole.TOOL for m in messages)
    assert all(FENCE_OPEN not in m.content for m in messages
               if m.role is ConsoleMessageRole.ASSISTANT)


@pytest.mark.asyncio
async def test_stop_cancels_tree_and_persists_cancelled(tmp_path):
    controller, store, db = _controller(
        tmp_path, [[_fence("calculator", {"expression": "1"})], ["late"]])

    original = controller._agent_bridge.run_reply

    def cancel_after_first(*args, **kwargs):
        controller._stop_requested = True         # simulate Stop during the run
        return original(*args, **kwargs)

    controller._agent_bridge.run_reply = cancel_after_first
    await controller.submit_draft("loop please")

    primary = [r for r in _all_runs(db) if r["agent_kind"] == "primary"]
    assert primary and primary[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_config_gate_off_uses_legacy_path(tmp_path):
    controller, store, db = _controller(tmp_path, [["legacy answer."]], enabled=False)
    await controller.submit_draft("hi")
    messages = store.messages_for_session(store.active_session_id)
    assert messages[-1].content == "legacy answer."
    # Legacy path never touches AgentRunsDB.
    assert _all_runs(db) == []


# -- Critical 1: an uncaught bridge exception must not wedge the controller --


@pytest.mark.asyncio
async def test_bridge_exception_fails_message_and_unwedges_controller(tmp_path):
    """A raw exception from bridge.run_reply (e.g. from db.create_run/persist,
    outside AgentService's own narrow loop guard) must be caught and turned
    into a failed message + FAILED run state -- not left to escape and wedge
    run_state at STREAMING forever (every future send on every session would
    then be rejected as "A Console run is already running.")."""
    controller, store, _db = _controller(tmp_path, [["ok answer."]])
    original_run_reply = controller._agent_bridge.run_reply

    def boom(**_kwargs):
        raise RuntimeError("db locked")

    controller._agent_bridge.run_reply = boom
    result = await controller.submit_draft("hello")
    assert result.accepted is True

    first_session_id = store.active_session_id
    messages = store.messages_for_session(first_session_id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.status == "failed"
    assert any(m.role is ConsoleMessageRole.SYSTEM for m in messages)
    assert controller.run_state.status is ConsoleRunStatus.FAILED

    # A brand-new session's send must succeed -- the controller must not stay
    # permanently wedged in STREAMING from the earlier uncaught exception.
    controller.new_session()
    assert store.active_session_id != first_session_id
    controller._agent_bridge.run_reply = original_run_reply
    second = await controller.submit_draft("second session hello")
    assert second.accepted is True
    second_messages = store.messages_for_session(store.active_session_id)
    assert second_messages[-1].content == "ok answer."


# -- Critical 2: non-DONE outcomes must fail honestly, never silently complete --


@pytest.mark.asyncio
async def test_run_error_via_submit_is_failed_retryable_and_excluded_from_context(tmp_path):
    controller, store, _db = _controller(tmp_path, [["unused"]])

    def erroring_run_reply(*, assistant_message_id, **_kwargs):
        store.append_stream_chunk(assistant_message_id, "partial answer before erroring")
        return RunOutcome(
            status=RUN_ERROR,
            steps=[AgentStep(index=0, kind=STEP_ERROR, summary="tool exploded")],
        )

    controller._agent_bridge.run_reply = erroring_run_reply
    result = await controller.submit_draft("hi")
    assert result.accepted is True

    session_id = store.active_session_id
    assistant = next(
        m for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert assistant.status == "failed"
    assert assistant.content == "partial answer before erroring"
    assert "[agent error]" not in assistant.content
    assert controller.run_state.status is ConsoleRunStatus.FAILED

    # Excluded from the model context built for the next turn.
    provider_messages = controller._provider_messages_for_session(session_id)
    assert not any(m.get("content") == assistant.content for m in provider_messages)

    # Retryable: status "failed" is exactly what retry_message() requires.
    def fixed_run_reply(*, assistant_message_id, **_kwargs):
        store.append_stream_chunk(assistant_message_id, "fixed.")
        return RunOutcome(status=RUN_DONE, steps=[], final_text="fixed.")

    controller._agent_bridge.run_reply = fixed_run_reply
    retry_result = await controller.retry_message(assistant.id)
    assert retry_result.accepted is True
    completed = store.get_message(assistant.id)
    assert completed.status == "complete"
    assert completed.content == "fixed."


@pytest.mark.asyncio
async def test_run_stuck_outcome_is_visibly_failed_not_silent_complete(tmp_path):
    controller, store, _db = _controller(tmp_path, [["unused"]])

    def stuck_run_reply(*, assistant_message_id, **_kwargs):
        store.append_stream_chunk(assistant_message_id, "still thinking about it")
        return RunOutcome(
            status=RUN_STUCK,
            steps=[AgentStep(index=0, kind=STEP_ERROR, summary="step budget exhausted")],
        )

    controller._agent_bridge.run_reply = stuck_run_reply
    result = await controller.submit_draft("hi")
    assert result.accepted is True

    session_id = store.active_session_id
    messages = store.messages_for_session(session_id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.status == "failed"
    assert assistant.content == "still thinking about it"
    system_rows = [m for m in messages if m.role is ConsoleMessageRole.SYSTEM]
    assert system_rows
    assert "budget" in system_rows[-1].content.lower()
    assert controller.run_state.status is ConsoleRunStatus.FAILED
    assert "stuck" in controller.run_state.visible_copy.lower()


@pytest.mark.asyncio
async def test_run_error_via_regenerate_preserves_original_answer_and_status(tmp_path):
    controller, store, _db = _controller(tmp_path, [["unused"]])

    def good_run_reply(*, assistant_message_id, **_kwargs):
        store.append_stream_chunk(assistant_message_id, "good original answer.")
        return RunOutcome(status=RUN_DONE, steps=[], final_text="good original answer.")

    controller._agent_bridge.run_reply = good_run_reply
    first = await controller.submit_draft("hi")
    assert first.accepted is True
    session_id = store.active_session_id
    assistant = next(
        m for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert assistant.content == "good original answer."
    assert assistant.status == "complete"

    def erroring_run_reply(*, assistant_message_id, **_kwargs):
        store.append_stream_chunk(assistant_message_id, "a bad partial regenerate")
        return RunOutcome(
            status=RUN_ERROR,
            steps=[AgentStep(index=0, kind=STEP_ERROR, summary="regenerate exploded")],
        )

    controller._agent_bridge.run_reply = erroring_run_reply
    regen = await controller.regenerate_message(assistant.id)
    assert regen.accepted is True

    restored = store.get_message(assistant.id)
    assert restored.content == "good original answer."
    assert restored.status == "complete"
    # No error variant was ever created/selected -- the failing regenerate
    # never touched the persisted message at all.
    assert restored.variants is None
    system_rows = [
        m for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.SYSTEM
    ]
    assert system_rows and "regenerate exploded" in system_rows[-1].content


# -- Blind spot: retry/continue/regenerate must also run through the agent path --


@pytest.mark.asyncio
async def test_retry_through_agent_path_uses_bridge_and_completes(tmp_path):
    # Only one gateway script: the initial send's failure is injected before
    # the bridge ever reaches the gateway, so the retry is the sole real call.
    controller, store, _db = _controller(tmp_path, [["fixed answer."]])
    real_run_reply = controller._agent_bridge.run_reply

    def boom(**_kwargs):
        raise RuntimeError("first attempt exploded")

    controller._agent_bridge.run_reply = boom
    first = await controller.submit_draft("please answer")
    assert first.accepted is True
    session_id = store.active_session_id
    assistant = next(
        m for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert assistant.status == "failed"

    calls = {"n": 0}

    def counting_run_reply(**kwargs):
        calls["n"] += 1
        return real_run_reply(**kwargs)

    controller._agent_bridge.run_reply = counting_run_reply
    retry_result = await controller.retry_message(assistant.id)
    assert retry_result.accepted is True
    assert calls["n"] == 1  # confirms retry actually drove the agent bridge
    completed = store.get_message(assistant.id)
    assert completed.status == "complete"
    assert completed.content == "fixed answer."


@pytest.mark.asyncio
async def test_continue_through_agent_path_uses_bridge_and_appends_new_message(tmp_path):
    controller, store, _db = _controller(
        tmp_path, [["Paris is the capital."], [" It has the Eiffel Tower."]]
    )
    await controller.submit_draft("tell me about France")
    session_id = store.active_session_id
    first_assistant = next(
        m for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert first_assistant.content == "Paris is the capital."

    real_run_reply = controller._agent_bridge.run_reply
    calls = {"n": 0}

    def counting_run_reply(**kwargs):
        calls["n"] += 1
        return real_run_reply(**kwargs)

    controller._agent_bridge.run_reply = counting_run_reply
    result = await controller.continue_from_message(first_assistant.id)
    assert result.accepted is True
    assert calls["n"] == 1

    messages = store.messages_for_session(session_id)
    assert messages[-1].role is ConsoleMessageRole.ASSISTANT
    assert messages[-1].id != first_assistant.id
    assert messages[-1].content == " It has the Eiffel Tower."
    assert messages[-1].status == "complete"


@pytest.mark.asyncio
async def test_regenerate_through_agent_path_uses_bridge_and_selects_new_variant(tmp_path):
    controller, store, _db = _controller(
        tmp_path, [["first answer."], ["second answer."]]
    )
    await controller.submit_draft("hi")
    session_id = store.active_session_id
    assistant = next(
        m for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert assistant.content == "first answer."

    real_run_reply = controller._agent_bridge.run_reply
    calls = {"n": 0}

    def counting_run_reply(**kwargs):
        calls["n"] += 1
        return real_run_reply(**kwargs)

    controller._agent_bridge.run_reply = counting_run_reply
    result = await controller.regenerate_message(assistant.id)
    assert result.accepted is True
    assert calls["n"] == 1

    regenerated = store.get_message(assistant.id)
    assert regenerated.content == "second answer."
    assert regenerated.status == "complete"
    assert regenerated.variants is not None
    assert regenerated.variants.selected_index == 1
    assert [v.content for v in regenerated.variants.variants] == [
        "first answer.",
        "second answer.",
    ]


# -- Important 3: the agent-runtime gate + bridge must not be a boot snapshot --


@pytest.mark.asyncio
async def test_agent_runtime_gate_refreshes_without_screen_teardown():
    """Flipping ``[console] agent_runtime`` after controller construction must
    change the next send's path. Previously only controller construction
    (``_ensure_console_chat_controller``) read the gate/bridge --
    ``_sync_console_chat_core_state`` refreshed provider selection on every
    access but never the gate, so toggling the kill-switch had no effect
    until the whole screen was torn down and rebuilt."""
    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "test-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "test-model"}
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.app_config["console"] = {"agent_runtime": True}

    class _FakeBridge:
        def __init__(self):
            self.calls = 0

        def run_reply(self, **_kwargs):
            self.calls += 1
            return RunOutcome(status=RUN_DONE, steps=[], final_text="agent answer.")

    fake_bridge = _FakeBridge()
    screen = ChatScreen(app)
    screen._ensure_console_agent_bridge = lambda: fake_bridge

    class _FakeGateway:
        async def resolve_for_send(self, _selection):
            return SimpleNamespace(ready=True, provider="llama_cpp", visible_copy="")

        async def stream_chat(self, _resolution, _messages):
            for chunk in ["legacy answer."]:
                yield chunk

    app.console_provider_gateway_factory = lambda: _FakeGateway()

    controller = screen._ensure_console_chat_controller()
    assert controller._agent_bridge is fake_bridge
    assert controller._agent_runtime_enabled is True

    # Flip the kill-switch AFTER construction -- no screen teardown.
    app.app_config["console"]["agent_runtime"] = False
    screen._sync_console_chat_core_state()

    result = await controller.submit_draft("hello")
    assert result.accepted is True
    assert fake_bridge.calls == 0  # legacy path used, not the agent bridge
    store = screen._ensure_console_chat_store()
    messages = store.messages_for_session(store.active_session_id)
    assert messages[-1].content == "legacy answer."
