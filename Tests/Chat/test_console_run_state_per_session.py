"""Per-session Console run state (parallel-agents spec §2)."""

from __future__ import annotations

import asyncio
import threading

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderStreamSignals

from Tests.Chat.conftest import StreamingGateway


# `controller_with_two_sessions` (and its `StreamingGateway` provider stub)
# moved to `Tests/Chat/conftest.py` (Task 7 brief) so
# `test_console_run_markers.py` can share it without a cross-module import.
# A handful of tests below (needing a custom session-creation ORDER the
# fixed 2-session fixture doesn't offer) still construct their own
# controller directly and import `StreamingGateway` straight from
# conftest -- a same-package import from the dedicated shared-fixtures
# module, not the cross-TEST-MODULE import the move above was about.


def test_run_states_are_isolated_per_session(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run A"),
        session_id=session_a,
    )

    assert controller.run_state_for(session_a).status is ConsoleRunStatus.STREAMING
    assert controller.run_state_for(session_b).is_send_allowed
    assert controller.in_flight_run_count() == 1


def test_facade_property_tracks_active_session(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions
    # `ConsoleChatStore` has no `set_active_session` method (verified by
    # grep) -- activation is `store.switch_session(session_id)`, which sets
    # `active_session_id` directly (see console_chat_store.py:490-494).
    controller.store.switch_session(session_a)
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run A"),
        session_id=session_a,
    )

    assert controller.run_state.status is ConsoleRunStatus.STREAMING
    controller.store.switch_session(session_b)
    assert controller.run_state.is_send_allowed  # B is idle

    with pytest.raises(AttributeError):
        controller.run_state = ConsoleRunState()  # facade is read-only


def test_terminal_clear_is_session_scoped(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED, "done A"), session_id=session_a
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run B"), session_id=session_b
    )

    controller._clear_terminal_run_state(session_id=session_a)

    assert controller.run_state_for(session_a).status is ConsoleRunStatus.IDLE
    assert controller.run_state_for(session_b).status is ConsoleRunStatus.STREAMING


def test_run_state_history_is_per_session(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.VALIDATING, "v"), session_id=session_a
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "s"), session_id=session_a
    )

    history_a = controller.run_state_history_for(session_a)
    history_b = controller.run_state_history_for(session_b)
    assert history_a == [
        ConsoleRunStatus.IDLE,
        ConsoleRunStatus.VALIDATING,
        ConsoleRunStatus.STREAMING,
    ]
    assert history_b == [ConsoleRunStatus.IDLE]

    # Legacy `run_state_history` property mirrors the ACTIVE session's history.
    controller.store.switch_session(session_a)
    assert controller.run_state_history == history_a


def test_in_flight_run_count_and_run_states_snapshot(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions

    assert controller.in_flight_run_count() == 0
    assert controller.run_states() == {}

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run A"), session_id=session_a
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.VALIDATING, "run B"), session_id=session_b
    )

    assert controller.in_flight_run_count() == 2
    snapshot = controller.run_states()
    assert snapshot[session_a].status is ConsoleRunStatus.STREAMING
    assert snapshot[session_b].status is ConsoleRunStatus.VALIDATING

    # Snapshot is a copy: mutating it must not affect the controller's map.
    snapshot[session_a] = ConsoleRunState()
    assert controller.run_state_for(session_a).status is ConsoleRunStatus.STREAMING


def test_send_refusal_is_per_session_and_capped(controller_with_two_sessions, monkeypatch):
    controller, session_a, session_b = controller_with_two_sessions
    monkeypatch.setattr(
        type(controller), "max_parallel_runs", property(lambda self: 1)
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run A"), session_id=session_a
    )

    assert controller.send_refusal_copy(session_a) == (
        "A run is already running in this tab."
    )
    refusal = controller.send_refusal_copy(session_b)
    assert refusal is not None and "1 agents already running" in refusal
    assert "Wait for one to finish or interrupt it." in refusal


def test_cap_default_and_floor(controller_with_two_sessions, monkeypatch):
    controller, _, _ = controller_with_two_sessions
    import tldw_chatbook.Chat.console_chat_controller as ccc
    monkeypatch.setattr(
        ccc, "get_cli_setting", lambda *a, **k: 0, raising=False
    )
    assert controller.max_parallel_runs == 1  # floor
    monkeypatch.setattr(
        ccc, "get_cli_setting", lambda *a, **k: None, raising=False
    )
    assert controller.max_parallel_runs == 3  # default


def test_lowering_cap_never_kills_running(controller_with_two_sessions, monkeypatch):
    controller, session_a, session_b = controller_with_two_sessions
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "A"), session_id=session_a
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "B"), session_id=session_b
    )
    monkeypatch.setattr(
        type(controller), "max_parallel_runs", property(lambda self: 1)
    )
    # Both stay streaming; only NEW sends are refused.
    assert controller.run_state_for(session_a).status is ConsoleRunStatus.STREAMING
    assert controller.run_state_for(session_b).status is ConsoleRunStatus.STREAMING
    assert controller.in_flight_run_count() == 2


def test_orphaned_closed_session_does_not_consume_cap_slot(
    controller_with_two_sessions, monkeypatch
):
    """Carried finding from Task 1's review: closing a session mid-VALIDATING
    leaves an orphaned entry in the per-session run-state map (``close_session``
    never touches ``controller._run_states``). Neither cap/fleet math
    (``in_flight_run_count``) nor the refusal copy (``send_refusal_copy``) may
    count or name a session that no longer exists in the store -- both share
    the ``_live_busy_session_ids`` filter. ``run_states()`` stays the RAW map
    on purpose (contract split, review round 2): it still surfaces the
    orphaned entry for callers that want the full recorded history, while
    every cap/fleet consumer must go through the live-filtered accessors.
    """
    controller, session_a, session_b = controller_with_two_sessions
    monkeypatch.setattr(
        type(controller), "max_parallel_runs", property(lambda self: 1)
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.VALIDATING, "orphan"), session_id=session_a
    )
    controller.store.close_session(session_a)

    assert session_a not in {session.id for session in controller.store.sessions()}
    # Live-filtered cap/fleet math excludes the orphan entirely...
    assert controller.in_flight_run_count() == 0
    # ...but the RAW map snapshot still holds it (contract split: run_states()
    # is not cap/fleet math and is never filtered).
    assert session_a in controller.run_states()
    # It must not occupy the cap's single slot for the surviving session.
    assert controller.send_refusal_copy(session_b) is None


def test_cap_refusal_truncates_and_k_more_suffix():
    """Test that send_refusal_copy shows exactly first N titles + 'and K more'.

    With CONSOLE_CAP_REFUSAL_TITLE_LIMIT=3 and max_parallel_runs=3, when 4+
    sessions are busy, the refusal message names the first 3 busy sessions
    and includes the literal "and K more" suffix where K = total_busy - 3.
    """
    from tldw_chatbook.Chat.console_chat_models import (
        CONSOLE_CAP_REFUSAL_TITLE_LIMIT,
    )
    from Tests.Chat.conftest import StreamingGateway

    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    # Create 5 sessions: 4 to make busy, 1 to test from
    # Note: use controller.new_session() to create new sessions each time
    # (store.ensure_session deduplicates by title, which interferes with this test)
    session_1_id = controller.new_session(title="Alpha").id
    session_2_id = controller.new_session(title="Bravo").id
    session_3_id = controller.new_session(title="Charlie").id
    session_4_id = controller.new_session(title="Delta").id
    session_5_id = controller.new_session(title="Echo").id

    # Mark first 4 sessions as busy (STREAMING status = not send_allowed)
    for session_id in [session_1_id, session_2_id, session_3_id, session_4_id]:
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "run"),
            session_id=session_id,
        )

    # With default max_parallel_runs=3 and 4 busy sessions, send from session_5 is refused
    refusal = controller.send_refusal_copy(session_5_id)
    assert refusal is not None, "Should refuse send when 4 sessions are busy"

    # Extract the busy count and verify the titles and suffix
    assert "4 agents already running" in refusal

    # Verify first 3 titles are present in order (they should be Alpha, Bravo, Charlie)
    titles_section = refusal[refusal.index("(") + 1 : refusal.index(")")]
    assert "Alpha, Bravo, Charlie" in titles_section, f"Expected first 3 titles in '{titles_section}'"

    # Verify the "and K more" suffix is exact where K = 4 - 3
    expected_suffix = f" and {4 - CONSOLE_CAP_REFUSAL_TITLE_LIMIT} more"
    assert expected_suffix in refusal, f"Should contain exact suffix '{expected_suffix}', got: {refusal}"
    assert " and 1 more" in refusal


# -- Task 3b: per-session stream/cancel state + scoped Stop/shutdown --------


def _seed_streaming_assistant(store: ConsoleChatStore, session_id: str) -> str:
    """Append a real user+assistant pair so `_mark_stream_stopped` (which
    calls `store.mark_message_stopped`) has a real row to act on -- a bare
    string id would raise KeyError deep inside `stop_active_run`."""
    store.append_message(session_id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(assistant.id, "partial")
    return assistant.id


@pytest.mark.asyncio
async def test_stop_active_run_cancels_only_viewed_sessions_task(
    controller_with_two_sessions,
):
    """Requirement 5a: two fake tasks registered for two sessions --
    `stop_active_run()` with A viewed cancels only A's task; B's task
    survives untouched and completes on its own."""
    controller, session_a, session_b = controller_with_two_sessions
    store = controller.store
    assistant_a = _seed_streaming_assistant(store, session_a)
    assistant_b = _seed_streaming_assistant(store, session_b)

    started_a = asyncio.Event()
    started_b = asyncio.Event()
    cancelled_a = asyncio.Event()
    completed_b = asyncio.Event()
    release_b = asyncio.Event()
    never_release_a = asyncio.Event()  # never set -- A is only ever cancelled

    async def never_ending():
        started_a.set()
        try:
            await never_release_a.wait()
        except asyncio.CancelledError:
            cancelled_a.set()
            raise

    async def finishes_on_release():
        started_b.set()
        await release_b.wait()
        completed_b.set()

    task_a = asyncio.create_task(never_ending())
    task_b = asyncio.create_task(finishes_on_release())
    # A task cancelled before its first scheduled step never runs its body
    # at all (asyncio discards it outright) -- wait for both to actually
    # reach their own `await` so cancellation exercises the SAME suspended-
    # mid-run state a real streaming task would be in.
    await started_a.wait()
    await started_b.wait()

    controller._active_stream_tasks[session_a] = task_a
    controller._active_assistant_message_ids[session_a] = assistant_a
    controller._active_stream_tasks[session_b] = task_b
    controller._active_assistant_message_ids[session_b] = assistant_b
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run A"), session_id=session_a
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run B"), session_id=session_b
    )

    store.switch_session(session_a)  # A is VIEWED
    assert controller.stop_active_run() is True

    with pytest.raises(asyncio.CancelledError):
        await task_a
    assert cancelled_a.is_set()

    # B is completely untouched: still registered, still running.
    assert controller._active_stream_tasks.get(session_b) is task_b
    assert not task_b.done()
    assert controller.run_state_for(session_b).status is ConsoleRunStatus.STREAMING

    release_b.set()
    await task_b
    assert completed_b.is_set()


@pytest.mark.asyncio
async def test_shutdown_cancels_and_awaits_every_sessions_task(
    controller_with_two_sessions,
):
    """Requirement 5b/3: shutdown's teardown path is GLOBAL -- it cancels
    and awaits every session's task, not just the viewed one, and leaves
    no stale entries behind for either session."""
    controller, session_a, session_b = controller_with_two_sessions
    store = controller.store
    assistant_a = _seed_streaming_assistant(store, session_a)
    assistant_b = _seed_streaming_assistant(store, session_b)

    cancelled = {"a": False, "b": False}
    started = {"a": asyncio.Event(), "b": asyncio.Event()}
    never_release = asyncio.Event()  # never set -- both are only ever cancelled

    async def never_ending(key: str):
        started[key].set()
        try:
            await never_release.wait()
        except asyncio.CancelledError:
            cancelled[key] = True
            raise

    task_a = asyncio.create_task(never_ending("a"))
    task_b = asyncio.create_task(never_ending("b"))
    # See test_stop_active_run_cancels_only_viewed_sessions_task: a task
    # cancelled before its first scheduled step never runs its body at all.
    await started["a"].wait()
    await started["b"].wait()

    controller._active_stream_tasks[session_a] = task_a
    controller._active_assistant_message_ids[session_a] = assistant_a
    controller._active_stream_tasks[session_b] = task_b
    controller._active_assistant_message_ids[session_b] = assistant_b
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run A"), session_id=session_a
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run B"), session_id=session_b
    )
    # Viewed session is irrelevant to shutdown's scope -- only A is viewed,
    # yet B must be cancelled too.
    store.switch_session(session_a)

    await controller.shutdown()

    assert cancelled == {"a": True, "b": True}
    assert controller._active_stream_tasks == {}
    assert controller._active_assistant_message_ids == {}
    assert controller._active_cancel_events == {}


@pytest.mark.asyncio
async def test_completing_run_pops_only_its_own_session_entries(
    controller_with_two_sessions,
):
    """Requirement 4: a session's own terminal path pops ONLY its own
    entries from the per-session stream/cancel maps -- a different
    session's still-registered (fake) entries are left untouched."""
    controller, session_a, session_b = controller_with_two_sessions
    store = controller.store
    store.switch_session(session_a)

    # Session B has its own, still in-flight (fake, never-completing) run.
    task_b = asyncio.create_task(asyncio.Event().wait())
    controller._active_stream_tasks[session_b] = task_b
    controller._active_assistant_message_ids[session_b] = "assistant-b"
    controller._active_cancel_events[session_b] = threading.Event()
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run B"), session_id=session_b
    )

    result = await controller.submit_draft("hello")
    assert result.accepted is True

    # Session A's own run completed and cleaned up entirely after itself.
    assert session_a not in controller._active_stream_tasks
    assert session_a not in controller._active_assistant_message_ids
    assert session_a not in controller._active_cancel_events

    # Session B's unrelated registered entries are untouched.
    assert controller._active_stream_tasks.get(session_b) is task_b
    assert controller._active_assistant_message_ids.get(session_b) == "assistant-b"
    assert not task_b.done()

    task_b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_b


# -- Fix round 1 (Critical 1): the shared `_stop_requested` flag must    --
# -- never be read by a specific run's OWN cancellation-check loop.      --


class TwoStreamGateway:
    """Two independently blockable, genuinely concurrent streams,
    distinguished by the draft text embedded in each call's own
    ``provider_messages`` (each session sends a differently-worded
    prompt) -- not by call order, which under real concurrency is not
    guaranteed to match dispatch order."""

    def __init__(self) -> None:
        self.started = {"a": asyncio.Event(), "b": asyncio.Event()}
        self.release = {"a": asyncio.Event(), "b": asyncio.Event()}

    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": True,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "visible_copy": "",
            },
        )()

    @staticmethod
    def _key_for(messages: list[dict]) -> str:
        for row in reversed(messages):
            content = row.get("content")
            if row.get("role") == "user" and isinstance(content, str):
                if "prompt-a" in content:
                    return "a"
                if "prompt-b" in content:
                    return "b"
        raise AssertionError(f"unrecognized draft in {messages!r}")

    async def stream_chat(self, resolution, messages):
        key = self._key_for(messages)
        yield f"{key}-chunk-1-"
        self.started[key].set()
        await self.release[key].wait()
        yield f"{key}-chunk-2-"
        yield f"{key}-chunk-3"


@pytest.mark.asyncio
async def test_stopping_one_session_does_not_truncate_a_concurrent_untouched_session():
    """Critical 1 regression (Fix round 1): the direct/legacy stream loop's
    cancellation check used to read the SHARED ``_stop_requested`` flag,
    which ``_signal_stop`` sets unconditionally on ANY session's Stop/
    Close -- so stopping session B silently truncated session A's still-
    streaming, completely untouched reply (live-reproduced by the
    reviewer). This drives two REAL, genuinely concurrent
    ``submit_draft()`` calls through the ACTUAL cancellation-check code
    path (``_stream_assistant_response``'s per-chunk check), not a
    registered fake.

    Deliberately signals B's stop via ``controller._signal_stop`` (the
    exact call ``stop_active_run``/``close_session`` make) WITHOUT
    task-cancelling B: B's own ``finally`` resetting the shared flag back
    to ``False`` the moment its task unwinds would otherwise mask the bug
    -- awaiting B to completion before touching A (as a full ``stop_
    active_run()`` + ``await task_b`` sequence would) closes that exact
    race window this test needs open. Isolating the bare flag WRITE from
    task-cancellation timing is also more precise: it is the write alone,
    independent of when/whether the writer's own task later unwinds, that
    the fix (Critical 1) had to stop being read by an unrelated run's
    loop.
    """
    store = ConsoleChatStore()
    gateway = TwoStreamGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    session_a = store.ensure_session(title="A")
    task_a = asyncio.create_task(controller.submit_draft("prompt-a"))
    await asyncio.wait_for(gateway.started["a"].wait(), timeout=1)

    session_b = controller.new_session(title="B")  # also activates B
    task_b = asyncio.create_task(controller.submit_draft("prompt-b"))
    await asyncio.wait_for(gateway.started["b"].wait(), timeout=1)

    # Both genuinely mid-stream and suspended at the same time.
    assert controller.in_flight_run_count() == 2
    assert not task_a.done()
    assert not task_b.done()

    # Signal B's stop via the same internal path stop_active_run uses --
    # WITHOUT cancelling B's task (still parked on its own release, its
    # own `finally` never runs during this test, so the shared flag stays
    # poisoned for the whole window A's own check runs in).
    assert store.active_session_id == session_b.id
    controller._signal_stop(session_id=session_b.id)

    # A is still blocked mid-stream -- release it now, while the shared
    # flag is poisoned by B's Stop, and let it run to completion.
    gateway.release["a"].set()
    result_a = await task_a
    assert result_a.accepted is True

    messages_a = store.messages_for_session(session_a.id)
    assistant_a = next(m for m in messages_a if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant_a.status == "complete"
    assert assistant_a.content == "a-chunk-1-a-chunk-2-a-chunk-3"

    # Clean up B (still parked on its own release) so nothing leaks.
    gateway.release["b"].set()
    result_b = await task_b
    assert result_b.accepted is True


# -- F4 fix (Qodo wave): `submit_draft` must target the session it was  --
# -- DISPATCHED for, not whichever session is active once its own body  --
# -- actually starts running.                                          --


@pytest.mark.asyncio
async def test_submit_draft_targets_dispatched_session_not_active_session_at_execution():
    """F4(b) regression (Qodo wave): before this fix, ``submit_draft``
    always resolved "the session to submit into" via ``store.
    ensure_session()``/``store.active_session_id`` -- i.e. whichever
    session is active AT THE MOMENT THIS COROUTINE BODY RUNS, not whatever
    session ``chat_screen._dispatch_console_draft_send`` captured at
    dispatch time. ``run_worker`` schedules the coroutine as a Task rather
    than running it inline, so a tab switch racing that scheduling gap
    (plausible: at least one event-loop iteration, often several Textual
    message-pump ticks) used to submit the draft into whichever session
    the user switched TO, not the one showing when Send was pressed.

    This drives the exact shape the real dispatch path
    (``ChatScreen._submit_console_native_draft``) now produces: session A
    dispatched, active session already moved to B by the time
    ``submit_draft`` actually runs -- passing A's id explicitly, exactly
    as the fixed ``_submit_console_native_draft`` does. Session A must get
    the write; session B (the one merely being *viewed*) must stay
    untouched.
    """
    store = ConsoleChatStore()
    gateway = StreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    session_a = store.ensure_session(title="A")
    # Simulate the scheduling gap: the user switches tabs to a brand new
    # session B AFTER Send was dispatched for A but BEFORE `submit_draft`'s
    # body actually runs (in production this happens between `run_worker`
    # scheduling the task and the event loop actually running it).
    session_b = controller.new_session(title="B")
    assert store.active_session_id == session_b.id

    result = await controller.submit_draft("hello-from-a", session_id=session_a.id)

    assert result.accepted is True
    messages_a = store.messages_for_session(session_a.id)
    assert any(m.content == "hello-from-a" for m in messages_a)
    messages_b = store.messages_for_session(session_b.id)
    assert messages_b == []


@pytest.mark.asyncio
async def test_submit_draft_session_id_none_preserves_active_session_bootstrap():
    """``session_id=None`` (the default) must keep resolving/creating the
    ACTIVE session exactly as before this fix -- direct-call test idioms
    and the very first send of a fresh app (no session yet) both rely on
    this. Guards against a regression where threading `session_id` through
    accidentally broke the "no session exists yet" bootstrap path (which
    ``store.ensure_session()`` -- not a session lookup -- must still
    handle)."""
    store = ConsoleChatStore()
    gateway = StreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    assert store.active_session_id is None

    result = await controller.submit_draft("hello")

    assert result.accepted is True
    assert store.active_session_id is not None
    messages = store.messages_for_session(store.active_session_id)
    assert any(m.content == "hello" for m in messages)


@pytest.mark.asyncio
async def test_submit_draft_closed_session_id_fails_closed_without_touching_active():
    """F4(b) edge case: if the dispatched session was closed during the
    scheduling gap (not just switched away from), there is nothing left
    to submit into. The fix must fail closed (``_session_closed_result``)
    rather than silently falling back to ``ensure_session()`` and
    submitting into whatever is active now."""
    store = ConsoleChatStore()
    gateway = StreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    session_a = store.ensure_session(title="A")
    closed_session_id = session_a.id
    session_b = controller.new_session(title="B")
    controller.close_session(closed_session_id)
    assert store.active_session_id == session_b.id

    result = await controller.submit_draft("hello", session_id=closed_session_id)

    assert result.accepted is True
    assert result.visible_copy == "Session closed."
    messages_b = store.messages_for_session(session_b.id)
    assert messages_b == []


@pytest.mark.asyncio
async def test_finalize_agent_success_citation_repair_keyerror_stamps_owning_session_not_active():
    """F4(a) regression (Qodo wave): `_finalize_agent_success`'s two
    citation-repair-selection ``except KeyError`` branches used to call
    ``_session_closed_result()`` with NO session id -- even though
    ``session_id`` is a REQUIRED parameter of this method (always known,
    never re-derived from anything that could go stale). The bare no-arg
    call defaulted to whichever session is ACTIVE right now, wrongly
    stamping ITS run state STOPPED even though it has nothing to do with
    this run.

    Drives ``_finalize_agent_success`` directly (mirrors this codebase's
    existing pattern of unit-testing controller internals directly, e.g.
    ``_stream_assistant_response`` in test_console_chat_controller.py),
    with a monkeypatched ``_select_post_generation_body`` that raises
    ``KeyError`` (the message's session vanished mid-run), while a
    DIFFERENT, untouched session B is the one currently active.
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    session_a = store.ensure_session(title="A")
    assistant = store.append_message(
        session_a.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="partial reply",
        persist=False,
    )

    session_b = controller.new_session(title="B")  # also activates B
    assert store.active_session_id == session_b.id

    async def _raise_key_error(**_kwargs):
        raise KeyError("message gone")

    controller._select_post_generation_body = _raise_key_error

    outcome = type("Outcome", (), {"final_text": "final text"})()
    result = await controller._finalize_agent_success(
        assistant.id,
        session_a.id,
        outcome,
        variant_mode=False,
        citation_repair_session=object(),
        stream_signals=ConsoleProviderStreamSignals(),
    )

    assert result.visible_copy == "Session closed."
    # Session A (the run's own owning session) got the STOPPED stamp...
    assert controller.run_state_for(session_a.id).status is ConsoleRunStatus.STOPPED
    # ...session B (merely the one being VIEWED) is untouched.
    assert controller.run_state_for(session_b.id).status is ConsoleRunStatus.IDLE
