"""The controller send path runs the agent loop when the bridge is wired."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_models import (
    AgentStep, RunOutcome, RUN_DONE, RUN_ERROR, RUN_STUCK, SPAWN_TOOL_NAME,
    STEP_ERROR, STEP_TOOL_RESULT,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.MCP.permission_store import EffectiveToolState

from Tests.Agents.test_mcp_tool_provider import FakeMCPService, _catalog_record, _tool_dict


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


class _ParkingGateway:
    """A ``provider_gateway`` whose ``stream_chat`` blocks the calling OS
    thread mid-turn until released -- lets a test park the real bridge's
    background thread (``asyncio.to_thread`` inside ``_run_agent_reply``)
    at a well-known point and then drive a genuine cross-task
    ``stop_active_run()`` while it is still executing."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    async def resolve_for_send(self, _selection):
        return SimpleNamespace(ready=True, provider="llama_cpp", visible_copy="")

    async def stream_chat(self, _resolution, _messages):
        self.started.set()
        # Blocking (not async) wait: this runs inside the bridge's own
        # private per-run event loop (``run_loop.run_until_complete`` in
        # ``ConsoleAgentBridge.run_reply``), on a real worker thread from
        # ``asyncio.to_thread`` -- blocking it here does not touch the
        # test's own asyncio loop on the main thread at all.
        # timeout is an anti-hang safety only — the test always releases
        # explicitly; generous so CI contention can never unpark early
        # and break the choreography (PR #644 review).
        self.release.wait(timeout=60)
        yield "answered anyway."


@pytest.mark.asyncio
async def test_stop_during_parked_bridge_thread_persists_cancelled_not_done(tmp_path):
    """task-227 AC1: stop_active_run's task-cancel must not race
    _run_agent_reply's finally-reset of _stop_requested.

    asyncio.to_thread survives Task cancellation -- cancelling the Task
    only detaches the *coroutine* from the still-running background OS
    thread; the thread itself keeps executing. Pre-fix, should_cancel read
    the shared, mutable _stop_requested flag, which _run_agent_reply's
    finally block resets to False the moment the coroutine handles the
    CancelledError raised by stop_active_run's task.cancel() -- so if the
    surviving thread polls should_cancel() *after* that reset (exactly
    what happens here: the gateway is released only after awaiting the
    coroutine's own return), it incorrectly sees "not cancelled" and
    finishes the turn as RUN_DONE, persisting agent_runs.status == "done"
    even though the user stopped. The fix threads a per-run
    threading.Event through the should_cancel closure, set once by
    stop_active_run and never reset by the finally block, so the
    surviving thread still observes the cancellation correctly.
    """
    gateway = _ParkingGateway()
    store = ConsoleChatStore()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, provider="llama_cpp", model="test-model",
        agent_bridge=bridge, agent_runtime_enabled=True)

    send_task = asyncio.ensure_future(controller.submit_draft("hello"))

    for _ in range(3000):  # 30s deadline — CI-contention headroom (PR #644 review)
        if gateway.started.is_set():
            break
        await asyncio.sleep(0.01)
    assert gateway.started.is_set(), "bridge thread never reached the parked gateway call"

    # Stop while the bridge's worker thread is genuinely mid-turn -- a
    # real cross-task cancellation, not a manually-flipped test flag.
    assert controller.stop_active_run() is True
    result = await send_task
    assert result.accepted is True

    session_id = store.active_session_id
    assistant = next(
        m for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert assistant.status == "stopped"
    assert assistant.content == ""
    # The controller-side bookkeeping has already been reset by
    # _run_agent_reply's finally -- exactly the moment the pre-fix bug
    # discarded the signal the still-running bridge thread depends on.
    assert controller._stop_requested is False
    assert controller._active_stream_task is None

    # Release the parked thread now -- it is STILL RUNNING on its own OS
    # thread, oblivious to the coroutine having already returned.
    gateway.release.set()

    primary: list[dict] = []
    for _ in range(3000):  # 30s deadline — CI-contention headroom (PR #644 review)
        primary = [r for r in _all_runs(db) if r["agent_kind"] == "primary"]
        if primary and primary[0]["status"] != "running":
            break
        await asyncio.sleep(0.01)

    assert primary, "primary run was never created"
    assert primary[0]["status"] == "cancelled", (
        "the surviving bridge thread's should_cancel() must still observe "
        f"the Stop after the finally reset -- got {primary[0]['status']!r}"
    )


@pytest.mark.asyncio
async def test_finalize_after_already_stopped_is_a_benign_noop(tmp_path):
    """task-227 AC3 (LOW-2): a Stop landing in the ultra-narrow window
    after the bridge returns RUN_DONE but before _finalize_agent_reply
    runs leaves the message already "stopped" by the time finalize is
    reached. mark_message_complete (and mark_message_failed, for a
    non-done outcome) reject an already-terminal message via
    _validate_can_mark_terminal, and finalize_variant_stream has no such
    guard at all -- either raises an unhandled ValueError or silently
    resurrects the stopped message back to "complete". Finalize must
    instead treat an already-stopped target as a benign no-op."""
    controller, store, _db = _controller(tmp_path, [["unused"]])

    def parked_run_reply(*, assistant_message_id, **_kwargs):
        # Simulate the race directly at the store level: by the time the
        # bridge "returns" RUN_DONE, a concurrent Stop has already
        # finalized the message to "stopped".
        store.mark_message_stopped(assistant_message_id)
        return RunOutcome(status=RUN_DONE, steps=[], final_text="late done text")

    controller._agent_bridge.run_reply = parked_run_reply
    result = await controller.submit_draft("hi")
    assert result.accepted is True

    session_id = store.active_session_id
    assistant = next(
        m for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert assistant.status == "stopped"
    assert assistant.content == ""
    assert "late done text" not in assistant.content
    assert controller.run_state.status is ConsoleRunStatus.STOPPED


@pytest.mark.asyncio
async def test_finalize_after_already_stopped_regenerate_no_phantom_variant(tmp_path):
    """task-227 AC3 follow-up: the same post-outcome-window race as
    ``test_finalize_after_already_stopped_is_a_benign_noop`` above, but
    during a REGENERATE (``variant_mode=True``). ``mark_message_stopped``
    (console_chat_store.py) restores a mid-regenerate message to its
    *prior* status (e.g. "complete"), not "stopped" -- so
    ``_finalize_agent_reply``'s original ``current.status == "stopped"``
    guard never fires here, and RUN_DONE falls through to
    ``finalize_variant_stream``, which has no stopped-guard of its own and
    fabricates a phantom variant from the (already-cleared) variant base,
    silently resurrecting the message to "complete" with a bogus variant
    entry. The fix instead trusts the run's own per-run ``cancel_event``
    (set by ``_signal_stop`` the instant Stop is requested, never cleared
    for this run) as the authority on whether the run was stopped,
    independent of what status ``mark_message_stopped`` happened to leave
    the message at.
    """
    controller, store, _db = _controller(tmp_path, [["original answer."], ["unused"]])
    first = await controller.submit_draft("hi")
    assert first.accepted is True
    session_id = store.active_session_id
    assistant = next(
        m for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert assistant.status == "complete"
    assert assistant.content == "original answer."
    assert assistant.variants is None

    def parked_regenerate_reply(*, assistant_message_id, **_kwargs):
        # Mirror stop_active_run's own sequence -- _signal_stop() (sets
        # the per-run cancel_event) immediately followed by
        # mark_message_stopped (restores the pre-regenerate base content
        # AND status) -- simulating a real Stop landing in the window
        # after the bridge has already produced RUN_DONE but before
        # _finalize_agent_reply runs.
        controller._active_cancel_event.set()
        store.mark_message_stopped(assistant_message_id)
        return RunOutcome(status=RUN_DONE, steps=[], final_text="late regenerate text")

    controller._agent_bridge.run_reply = parked_regenerate_reply
    regen = await controller.regenerate_message(assistant.id)
    assert regen.accepted is True

    restored = store.get_message(assistant.id)
    assert restored.status == "complete"
    assert restored.content == "original answer."
    assert "late regenerate text" not in restored.content
    # No phantom variant was fabricated from the already-popped base.
    assert restored.variants is None
    assert controller.run_state.status is ConsoleRunStatus.STOPPED


@pytest.mark.asyncio
async def test_finalize_after_already_stopped_regenerate_error_no_wedge(tmp_path):
    """task-227 AC3 follow-up, RUN_ERROR/RUN_STUCK side: the same race as
    the RUN_DONE test above, but the bridge reports a non-done outcome.
    Pre-fix, ``_finalize_agent_reply`` falls through past the (never
    firing) ``current.status == "stopped"`` guard into
    ``mark_message_failed``, which raises ``ValueError`` via
    ``_validate_can_mark_terminal`` because the message is already
    "complete" (its restored prior status), not "pending"/"streaming".
    That exception is raised *outside* ``_run_agent_reply``'s own
    try/except (the call happens after the try/finally block returns),
    so it propagates uncaught all the way out of ``regenerate_message``,
    leaving ``run_state`` wedged at STREAMING forever -- every subsequent
    send is then rejected as "a Console run is already running."
    """
    controller, store, _db = _controller(tmp_path, [["original answer."], ["unused"]])
    first = await controller.submit_draft("hi")
    assert first.accepted is True
    session_id = store.active_session_id
    assistant = next(
        m for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert assistant.status == "complete"

    def parked_regenerate_error_reply(*, assistant_message_id, **_kwargs):
        controller._active_cancel_event.set()
        store.mark_message_stopped(assistant_message_id)
        return RunOutcome(
            status=RUN_ERROR,
            steps=[AgentStep(index=0, kind=STEP_ERROR, summary="late regenerate error")],
        )

    controller._agent_bridge.run_reply = parked_regenerate_error_reply
    regen = await controller.regenerate_message(assistant.id)
    assert regen.accepted is True

    restored = store.get_message(assistant.id)
    assert restored.status == "complete"
    assert restored.content == "original answer."
    assert controller.run_state.status is ConsoleRunStatus.STOPPED

    # The run must not be wedged: a fresh send is accepted immediately.
    def next_send_reply(*, assistant_message_id, **_kwargs):
        store.append_stream_chunk(assistant_message_id, "next turn works.")
        return RunOutcome(status=RUN_DONE, steps=[], final_text="next turn works.")

    controller._agent_bridge.run_reply = next_send_reply
    second = await controller.submit_draft("still there?")
    assert second.accepted is True
    second_messages = store.messages_for_session(store.active_session_id)
    assert second_messages[-1].content == "next turn works."


@pytest.mark.asyncio
async def test_stop_before_first_token_persists_cancelled_no_agent_run_failed(tmp_path):
    """Plan-B agent-runtime gate Finding 1, full-chain reproduction: the real
    live gate found that clicking Stop before a slow provider's *first*
    chunk arrives settled the run as ``error`` (with a step-log entry
    "Cannot append stream chunks to a stopped message.") and dropped the
    assistant message entirely -- because ``ConsoleChatController.
    stop_active_run`` finalizes the assistant message to "stopped" (via
    ``store.mark_message_stopped``) *before* a slow provider's first chunk
    ever streams -- but *after* the run has already genuinely started (the
    first ``should_cancel()`` poll, at the very top of the agent loop
    before the model is ever called, must still see ``False``, exactly as
    it would in production: the loop had already committed to this turn
    before Stop was clicked). This reproduces that exact ordering
    deterministically (via a gateway that flips the stop state right
    before it streams anything) rather than via real background
    threading/task-cancellation timing, which -- as
    ``test_stop_cancels_tree_and_persists_cancelled`` above already
    exercises the should_cancel-only half of -- is covered by other tests;
    this one isolates the store-level race that was Finding 1's actual
    root cause."""
    controller, store, db = _controller(tmp_path, [["late", " answer."]])
    gateway = controller.provider_gateway
    real_stream_chat = gateway.stream_chat

    async def stop_before_first_chunk(resolution, messages):
        # Mirror ConsoleChatController.stop_active_run(): mark the message
        # stopped and flip should_cancel *before* the gateway ever streams
        # a chunk into the store -- simulating Stop landing while the
        # (slow) provider is still silent.
        session_id = store.active_session_id
        assistant_message_id = next(
            m.id for m in reversed(store.messages_for_session(session_id))
            if m.role is ConsoleMessageRole.ASSISTANT
        )
        store.mark_message_stopped(assistant_message_id)
        controller._stop_requested = True
        async for chunk in real_stream_chat(resolution, messages):
            yield chunk

    gateway.stream_chat = stop_before_first_chunk
    result = await controller.submit_draft("slow question")
    assert result.accepted is True

    session_id = store.active_session_id
    messages = store.messages_for_session(session_id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    # The late "late answer." chunks were dropped, not leaked into content.
    assert assistant.status == "stopped"
    assert assistant.content == ""
    assert not any(
        m.role is ConsoleMessageRole.SYSTEM and "Agent run failed" in m.content
        for m in messages
    )

    primary = [r for r in _all_runs(db) if r["agent_kind"] == "primary"]
    assert primary and primary[0]["status"] == "cancelled"
    assert controller.run_state.status is ConsoleRunStatus.STOPPED


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


# -- P5-T6: per-run MCPToolProvider registration + review-hook wiring --


def _fake_app(service=None):
    """`controller.app`-shaped stand-in: `call_from_thread` (needed by
    `request_mcp_approvals`) plus, when given, `unified_mcp_service`."""
    kwargs = {} if service is None else {"unified_mcp_service": service}
    return SimpleNamespace(call_from_thread=lambda fn, *a, **kw: fn(*a, **kw), **kwargs)


def _capturing_run_reply(captured, *, final_text="ok."):
    def run_reply(**kwargs):
        captured.append(kwargs)
        return RunOutcome(status=RUN_DONE, steps=[], final_text=final_text)
    return run_reply


@pytest.mark.asyncio
async def test_mcp_provider_not_wired_when_no_unified_mcp_service(tmp_path):
    controller, store, _db = _controller(tmp_path, [["ok."]])
    captured = []
    controller._agent_bridge.run_reply = _capturing_run_reply(captured)
    controller.app = _fake_app()  # no unified_mcp_service attribute at all

    result = await controller.submit_draft("hi")

    assert result.accepted is True
    assert captured[0]["mcp_provider"] is None
    assert captured[0]["review_tool_calls"] is None


@pytest.mark.asyncio
async def test_mcp_provider_not_wired_when_kill_switch_on(tmp_path):
    controller, store, _db = _controller(tmp_path, [["ok."]])
    captured = []
    controller._agent_bridge.run_reply = _capturing_run_reply(captured)
    service = FakeMCPService(
        kill_switch=True,
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
    )
    controller.app = _fake_app(service)

    result = await controller.submit_draft("hi")

    assert result.accepted is True
    assert captured[0]["mcp_provider"] is None
    assert captured[0]["review_tool_calls"] is None


@pytest.mark.asyncio
async def test_mcp_provider_not_wired_when_catalog_empty(tmp_path):
    controller, store, _db = _controller(tmp_path, [["ok."]])
    captured = []
    controller._agent_bridge.run_reply = _capturing_run_reply(captured)
    service = FakeMCPService()  # no catalog records, no builtin inventory
    controller.app = _fake_app(service)

    result = await controller.submit_draft("hi")

    assert result.accepted is True
    assert captured[0]["mcp_provider"] is None
    assert captured[0]["review_tool_calls"] is None


@pytest.mark.asyncio
async def test_mcp_provider_wired_when_eligible(tmp_path):
    controller, store, _db = _controller(tmp_path, [["ok."]])
    captured = []
    controller._agent_bridge.run_reply = _capturing_run_reply(captured)
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    controller.app = _fake_app(service)

    result = await controller.submit_draft("hi")

    assert result.accepted is True
    provider = captured[0]["mcp_provider"]
    assert isinstance(provider, MCPToolProvider)
    assert len(provider.list_catalog()) == 1
    assert callable(captured[0]["review_tool_calls"])


@pytest.mark.asyncio
async def test_mcp_tool_call_executes_end_to_end_when_state_allows(tmp_path):
    """Full plumbing, real bridge: a fence call to an MCP tool name is
    registered, dispatched, and actually invoked via the real provider --
    no approval needed since the fake service's default state is "allow"."""
    scripts = [
        [_fence("mcp__srv__run", {"x": 1})],
        ["done with mcp."],
    ]
    controller, store, _db = _controller(tmp_path, scripts)
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        default_state=EffectiveToolState(state="allow", origin="tool_override"),
    )
    controller.app = _fake_app(service)

    result = await controller.submit_draft("please run it")

    assert result.accepted is True
    messages = store.messages_for_session(store.active_session_id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.content == "done with mcp."
    assert service.execute_calls == [("local:srv", "run", {"x": 1}, "agent", "allowed")]


@pytest.mark.asyncio
async def test_mcp_tool_call_ask_state_routes_through_review_hook_and_approves(tmp_path):
    """The T4/T6 review hook collects the pending call, makes ONE
    `request_mcp_approvals` round trip, and a UI-thread approval lets the
    call actually execute -- full worker-thread <-> UI-thread round trip
    through the real bridge, controller, and provider."""
    scripts = [
        [_fence("mcp__srv__run", {"x": 1})],
        ["approved and done."],
    ]
    controller, store, _db = _controller(tmp_path, scripts)
    received: list[dict | None] = []
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    controller.app = _fake_app(service)
    controller.set_pending_approval = received.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    send_task = asyncio.ensure_future(controller.submit_draft("please run it"))
    for _ in range(3000):  # 30s deadline, CI-contention headroom
        if received and received[-1] is not None:
            break
        await asyncio.sleep(0.01)
    assert received and received[-1] is not None, "approval card was never surfaced"
    llm_name = received[-1]["calls"][0]["llm_name"]
    assert llm_name == "mcp__srv__run"
    controller.resolve_pending_approval({llm_name: "approve_once"})

    result = await send_task

    assert result.accepted is True
    messages = store.messages_for_session(store.active_session_id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.content == "approved and done."
    assert service.execute_calls == [("local:srv", "run", {"x": 1}, "agent", "approved")]
    assert received[-1] is None  # the card is always cleared afterwards


@pytest.mark.asyncio
async def test_mcp_tool_call_ask_state_times_out_denies(tmp_path):
    """An undecided pending call fails closed to the exact TIMEOUT_REFUSAL
    copy -- single-sourced through `invoke()`'s own gate (never invoked)."""
    scripts = [
        [_fence("mcp__srv__run", {"x": 1})],
        ["it was refused."],
    ]
    controller, store, _db = _controller(tmp_path, scripts)
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    controller.app = _fake_app(service)
    controller.mcp_approval_timeout_seconds = lambda: 0.05

    result = await controller.submit_draft("please run it")

    assert result.accepted is True
    messages = store.messages_for_session(store.active_session_id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.content == "it was refused."
    tool_rows = [m for m in messages if m.role is ConsoleMessageRole.TOOL]
    assert any(
        "user did not approve within the time limit" in row.content for row in tool_rows
    )
    assert service.execute_calls == []  # never invoked -- refused before dispatch


# -- P5 review fix: `_compose_mcp_provider` must publish the inspector's
# "MCP" row counts -- previously composed but never written anywhere the
# `ChatScreen._console_mcp_tool_count`/`_console_mcp_not_connected_count`
# accessors could read, so the row was permanently dead scaffolding. --


@pytest.mark.asyncio
async def test_compose_mcp_provider_publishes_none_counts_when_no_service(tmp_path):
    controller, _store, _db = _controller(tmp_path, [["ok."]])
    controller.app = _fake_app()  # no unified_mcp_service attribute at all

    provider, hook = await controller._compose_mcp_provider()

    assert provider is None and hook is None
    assert controller.app.console_mcp_tool_count is None
    assert controller.app.console_mcp_not_connected_count is None


@pytest.mark.asyncio
async def test_compose_mcp_provider_publishes_none_counts_when_kill_switch_on(tmp_path):
    controller, _store, _db = _controller(tmp_path, [["ok."]])
    service = FakeMCPService(
        kill_switch=True,
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
    )
    controller.app = _fake_app(service)

    provider, hook = await controller._compose_mcp_provider()

    assert provider is None and hook is None
    assert controller.app.console_mcp_tool_count is None
    assert controller.app.console_mcp_not_connected_count is None


@pytest.mark.asyncio
async def test_compose_mcp_provider_publishes_none_counts_when_catalog_empty(tmp_path):
    controller, _store, _db = _controller(tmp_path, [["ok."]])
    service = FakeMCPService()  # no catalog records, no builtin inventory
    controller.app = _fake_app(service)

    provider, hook = await controller._compose_mcp_provider()

    assert provider is None and hook is None
    assert controller.app.console_mcp_tool_count is None
    assert controller.app.console_mcp_not_connected_count is None


@pytest.mark.asyncio
async def test_compose_mcp_provider_publishes_none_counts_when_get_kill_switch_raises(tmp_path):
    controller, _store, _db = _controller(tmp_path, [["ok."]])

    class _RaisingService:
        def get_kill_switch(self):
            raise RuntimeError("boom")

    controller.app = _fake_app()
    controller.app.unified_mcp_service = _RaisingService()

    provider, hook = await controller._compose_mcp_provider()

    assert provider is None and hook is None
    assert controller.app.console_mcp_tool_count is None
    assert controller.app.console_mcp_not_connected_count is None


@pytest.mark.asyncio
async def test_compose_mcp_provider_publishes_counts_when_eligible(tmp_path):
    """Both counts flow through simultaneously: server "a" is stale but
    still contributes an eligible tool (mirrors `Tests.Agents.
    test_mcp_tool_provider.test_not_connected_count_counts_distinct_
    eligible_stale_servers`), so `not_connected_count == 1` alongside a
    real `tool_count >= 1` -- the composed provider's own catalog and
    `not_connected_count` land on the app object verbatim, not just a
    truthy/falsy summary."""
    controller, _store, _db = _controller(tmp_path, [["ok."]])
    service = FakeMCPService(
        catalog_records=[
            _catalog_record("a", [_tool_dict("t1")], is_connected=False),
            _catalog_record("c", [_tool_dict("t3")], is_connected=True),
        ],
    )
    controller.app = _fake_app(service)

    provider, hook = await controller._compose_mcp_provider()

    assert provider is not None
    assert callable(hook)
    assert controller.app.console_mcp_tool_count == len(provider.list_catalog())
    assert controller.app.console_mcp_tool_count == 2
    assert controller.app.console_mcp_not_connected_count == provider.not_connected_count
    assert controller.app.console_mcp_not_connected_count == 1


# -- P5 review fix: the property that a sub-agent's MCP tool calls flow
# through the SAME review closure + invoke() gate as the primary's holds
# via shared `_run_one` + `AgentService`'s ctor-level `review_tool_calls`
# hook (see `agent_service.py`'s docstring), but nothing pinned it with a
# test -- easy to regress silently outside this diff. --


@pytest.mark.asyncio
async def test_mcp_tool_call_gates_subagent_call_same_as_primary(tmp_path):
    """A spawned child's own MCP tool call is gated exactly like the
    primary's: an undecided "ask"-state approval times out and denies --
    the call is refused before dispatch (`execute_calls == []`) -- because
    `AgentService._run_one`'s `spawn` closure threads the SAME ctor-level
    `review_tool_calls` hook (and the same `self.registry`) into the
    child's own `LoopDeps`/`invoke_tool`, not a bespoke unwired path."""
    scripts = [
        [_fence(SPAWN_TOOL_NAME, {"task": "please run it"})],  # primary: spawn a child
        [_fence("mcp__srv__run", {"x": 1})],                    # child: call the MCP tool
        ["child refused."],                                      # child: final answer
        ["primary done."],                                        # primary: final answer
    ]
    controller, store, db = _controller(tmp_path, scripts)
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    controller.app = _fake_app(service)
    controller.mcp_approval_timeout_seconds = lambda: 0.05

    result = await controller.submit_draft("please delegate it")

    assert result.accepted is True
    assert service.execute_calls == []  # never invoked -- refused before dispatch

    subagent_runs = [r for r in _all_runs(db) if r["agent_kind"] == "subagent"]
    assert len(subagent_runs) == 1
    child_steps = json.loads(subagent_runs[0]["steps"])
    tool_result_steps = [s for s in child_steps if s["kind"] == STEP_TOOL_RESULT]
    assert tool_result_steps, "the child never attempted the gated MCP tool call"
    assert "user did not approve within the time limit" in tool_result_steps[0]["result"]

    messages = store.messages_for_session(store.active_session_id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.content == "primary done."


@pytest.mark.asyncio
async def test_mcp_review_hook_raise_fails_open_but_invoke_gate_still_refuses(tmp_path):
    """`request_mcp_approvals` raising inside the T4/T6 review closure must
    never abort the run: `run_agent_loop`'s own call to `review_tool_calls`
    fails OPEN (an empty verdicts map, so every call in the batch defaults
    to "proceed" -- the runtime's own documented fail-open policy for this
    hook). The call still never executes, though: `invoke()` re-resolves
    the SAME "ask" gate itself on dispatch and its own `self.
    _approval_callback` fallback is the identical monkeypatched-raising
    method, so it refuses there instead -- `execute_calls` stays empty."""
    scripts = [
        [_fence("mcp__srv__run", {"x": 1})],
        ["it was refused too."],
    ]
    controller, store, _db = _controller(tmp_path, scripts)
    service = FakeMCPService(catalog_records=[_catalog_record("srv", [_tool_dict("run")])])
    controller.app = _fake_app(service)

    def _raise(_pending):
        raise RuntimeError("approval channel unavailable")

    controller.request_mcp_approvals = _raise

    result = await controller.submit_draft("please run it")

    assert result.accepted is True
    messages = store.messages_for_session(store.active_session_id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.content == "it was refused too."
    assert service.execute_calls == []  # never invoked -- refused by invoke()'s own gate
