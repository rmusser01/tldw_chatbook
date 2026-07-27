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


class StreamingGateway:
    """Minimal provider gateway stub -- copied from test_console_chat_controller.py's
    idiom (no network I/O, `ready=True` resolution) since this file's tests never
    actually run a send/stream, only drive run-state bookkeeping directly."""

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

    async def stream_chat(self, resolution, messages):
        for chunk in ("hel", "lo"):
            yield chunk


@pytest.fixture
def controller_with_two_sessions():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    # `store.new_session` does not exist (verified by grep) -- the real
    # session-creation surface is `store.ensure_session`/`store.create_session`
    # and `controller.new_session`. `controller.new_session()` also activates
    # the session it creates (`ConsoleChatStore.create_session` sets
    # `active_session_id`), matching how `test_controller_creates_and_
    # switches_sessions` in test_console_chat_controller.py builds two
    # sessions.
    session_a = store.ensure_session(title="Session A")
    session_b = controller.new_session(title="Session B")
    return controller, session_a.id, session_b.id


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
