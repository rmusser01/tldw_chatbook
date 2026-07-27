"""Per-session Console run state (parallel-agents spec §2)."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
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
    never touches ``controller._run_states``). The cap must not count it --
    ``send_refusal_copy`` intersects its busy list with ``store.sessions()``
    so a session that no longer exists can't consume a cap slot or appear in
    the refusal copy's session list.
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
    # The orphaned entry is still in the map...
    assert controller.in_flight_run_count() == 1
    # ...but it must not occupy the cap's single slot for the surviving session.
    assert controller.send_refusal_copy(session_b) is None
