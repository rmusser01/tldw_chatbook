"""Real controller rollback preserves both native and saved fleet fences."""

import pytest

from Tests.Chat.test_console_chat_controller import StreamingGateway
from Tests.Chat.test_console_close_session_fleet import _RecordingFleetBridge
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_chat_controller import (
    CONSOLE_SESSION_CLOSE_RECOVERY_REFUSAL,
    ConsoleChatController,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class CloseBridge(_RecordingFleetBridge):
    def __init__(
        self, store, *, fail_second=False, abort_failure="none", fail_progress=False
    ):
        super().__init__(store)
        self.fail_second = fail_second
        self.abort_failure = abort_failure
        self.fail_progress = fail_progress
        self.fences = []
        self.live_fences = set()
        self.aborts = []
        self.progress = []
        self.controller = None

    def fence_fleet(self, conversation_id, *, generation):
        self.fences.append((conversation_id, generation))
        if self.fail_second and len(self.fences) == 2:
            raise RuntimeError("second fence failed")
        self.live_fences.add(conversation_id)
        return True

    def abort_fleet_fence(self, conversation_id, *, generation):
        self.aborts.append((conversation_id, generation))
        if self.abort_failure != "none" and len(self.aborts) == 1:
            if self.abort_failure == "raise":
                raise RuntimeError("rollback failed")
            return False
        self.live_fences.remove(conversation_id)
        return True

    def begin_close_progress(self, session_id, *, conversation_id):
        self.progress.append((session_id, conversation_id))
        assert not self.controller._session_close_generations
        if self.fail_progress:
            raise RuntimeError("progress preparation failed")

    def close_progress(self, *args, **kwargs):
        raise AssertionError("preferred begin_close_progress must be used once")


def rig(**options):
    store = ConsoleChatStore()
    session = store.ensure_session()
    session.persisted_conversation_id = "saved-chat"
    bridge = CloseBridge(store, **options)
    controller = ConsoleChatController(
        store=store, provider_gateway=StreamingGateway(), agent_bridge=bridge
    )
    bridge.controller = controller
    return store, session, controller, bridge


def observe_teardown(controller, monkeypatch):
    calls = {"wake": [], "scratch": [], "queue": []}
    for component, method, label in (
        (controller._fleet_wake, "fence_conversation", "wake"),
        (controller._scratch_spaces, "close", "scratch"),
        (controller.prompt_queue_coordinator, "mark_closing", "queue"),
    ):
        original = getattr(component, method)

        def observe(*args, _original=original, _calls=calls[label], **kwargs):
            _calls.append((args, kwargs))
            return _original(*args, **kwargs)

        monkeypatch.setattr(component, method, observe)
    return calls


@pytest.mark.parametrize("abort_failure", ["none", "false", "raise"])
@private_profile_test
def test_partial_fence_acquisition_uses_existing_rollback_boundary(
    tmp_path, request, monkeypatch, abort_failure
):
    store, session, controller, bridge = rig(
        fail_second=True, abort_failure=abort_failure
    )
    teardown = observe_teardown(controller, monkeypatch)
    revision = controller.lifecycle_impact(session_id=session.id).revision
    expected = (
        "second fence failed"
        if abort_failure == "none"
        else CONSOLE_SESSION_CLOSE_RECOVERY_REFUSAL
    )
    with pytest.raises(RuntimeError, match=expected):
        controller.begin_session_close(session.id, expected_revision=revision)
    generation = bridge.fences[0][1]
    assert bridge.aborts == [(session.id, generation)]
    assert bridge.live_fences == (set() if abort_failure == "none" else {session.id})
    assert controller._failed_session_close_generations == (
        {} if abort_failure == "none" else {session.id: generation}
    )
    assert bridge.progress == []
    assert bridge.cancel_all_calls == []
    assert teardown == {"wake": [], "scratch": [], "queue": []}
    assert not controller._session_close_generations
    assert store.sessions() == [session]


@pytest.mark.parametrize("abort_failure", ["none", "false", "raise"])
@private_profile_test
def test_progress_failure_rolls_back_all_fences_before_teardown(
    tmp_path, request, monkeypatch, abort_failure
):
    store, session, controller, bridge = rig(
        fail_progress=True, abort_failure=abort_failure
    )
    teardown = observe_teardown(controller, monkeypatch)
    revision = controller.lifecycle_impact(session_id=session.id).revision
    expected = (
        "progress preparation failed"
        if abort_failure == "none"
        else CONSOLE_SESSION_CLOSE_RECOVERY_REFUSAL
    )
    with pytest.raises(RuntimeError, match=expected):
        controller.begin_session_close(session.id, expected_revision=revision)
    generation = bridge.fences[0][1]
    assert bridge.aborts == [("saved-chat", generation), (session.id, generation)]
    assert bridge.progress == [(session.id, "saved-chat")]
    assert bridge.cancel_all_calls == []
    assert teardown == {"wake": [], "scratch": [], "queue": []}
    assert not controller._session_close_generations
    assert store.sessions() == [session]
    if abort_failure != "none":
        assert controller._failed_session_close_generations == {session.id: generation}
        assert bridge.live_fences == {"saved-chat"}
        before = list(bridge.fences)
        with pytest.raises(RuntimeError, match=CONSOLE_SESSION_CLOSE_RECOVERY_REFUSAL):
            controller.begin_session_close(session.id, expected_revision=revision)
        assert bridge.fences == before
    else:
        assert not bridge.live_fences
        assert not controller._failed_session_close_generations


@private_profile_test
def test_preferred_progress_boundary_runs_once_before_committed_close(
    tmp_path, request, monkeypatch
):
    store, session, controller, bridge = rig()
    teardown = observe_teardown(controller, monkeypatch)
    revision = controller.lifecycle_impact(session_id=session.id).revision
    ticket = controller.begin_session_close(session.id, expected_revision=revision)
    assert ticket.session_id == session.id
    assert bridge.progress == [(session.id, "saved-chat")]
    assert bridge.cancel_all_calls == [session.id, "saved-chat"]
    generation = controller._session_close_generations[session.id]
    assert teardown == {
        "wake": [
            ((session.id,), {"generation": generation}),
            (("saved-chat",), {"generation": generation}),
        ],
        "scratch": [((session.id,), {})],
        "queue": [((session.id,), {})],
    }
    assert session.id in controller._session_close_generations
    assert store.sessions() == [session]
