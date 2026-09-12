"""Lifecycle hooks must preserve Console send custody and turn ownership."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_chat_controller import (
    ConsoleChatController,
    ConsoleChatStore,
    FakePersistence,
    RecordingStreamingGateway,
    _pending_image,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus


class Hooks:
    def __init__(self, *, blocked=False, context=""):
        self.blocked = blocked
        self.context = context
        self.notifications = []

    async def fire_async(self, *args, **kwargs):
        return SimpleNamespace(
            blocked=self.blocked, reason="policy", context=self.context
        )

    def notify(self, event, **kwargs):
        self.notifications.append((event, kwargs))


@pytest.mark.asyncio
async def test_hook_refusal_releases_preparation_for_retry():
    hooks = Hooks(blocked=True)
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=RecordingStreamingGateway(),
        ensure_run_hooks=lambda: hooks,
    )
    first = await controller.submit_draft("hello")
    assert not first.accepted
    assert not controller._prepared_send_continuations
    hooks.blocked = False
    second = await controller.submit_draft("hello")
    assert second.accepted


@pytest.mark.asyncio
async def test_cancelled_prompt_hook_releases_echo_and_preparation():
    entered = asyncio.Event()
    hooks = Hooks()

    async def wait(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    hooks.fire_async = wait
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=RecordingStreamingGateway(),
        ensure_run_hooks=lambda: hooks,
    )
    task = asyncio.create_task(controller.submit_draft("hello"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not controller._prepared_send_continuations
    assert all(
        row.status == "failed"
        for row in store.messages_for_session(store.active_session_id)
    )
    hooks.fire_async = Hooks().fire_async
    assert (await controller.submit_draft("retry")).accepted


@pytest.mark.asyncio
async def test_durable_hook_context_reaches_provider_and_audit():
    hooks = Hooks(context="hook context")
    store = ConsoleChatStore(persistence=FakePersistence())
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        ensure_run_hooks=lambda: hooks,
    )
    result = await controller.submit_draft("hello")
    assert result.accepted
    assert any(row.get("content") == "hook context" for row in gateway.messages_seen)
    audit = [
        row
        for row in store.messages_for_session(store.active_session_id)
        if row.metadata and row.metadata.origin == "hook"
    ]
    assert len(audit) == 1
    assert audit[0].persisted_message_id is not None


def test_stop_is_per_accepted_turn_even_while_queue_owns_notifications():
    hooks = Hooks()
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=object(),
        ensure_run_hooks=lambda: hooks,
    )
    controller.activity_for = lambda _session: SimpleNamespace(
        terminal_notification_eligible=False
    )
    for status in (ConsoleRunStatus.COMPLETED, ConsoleRunStatus.STOPPED):
        controller._arm_run_hooks_stop(session.id)
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, ""), session_id=session.id
        )
        controller._set_run_state(ConsoleRunState(status, ""), session_id=session.id)
        controller._set_run_state(ConsoleRunState(status, ""), session_id=session.id)
    assert [data["data"]["status"] for event, data in hooks.notifications] == [
        "completed",
        "cancelled",
    ]


@pytest.mark.asyncio
async def test_hook_refusal_preserves_pending_attachment(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.is_vision_capable", lambda *_: True
    )
    hooks = Hooks(blocked=True)
    store = ConsoleChatStore()
    session = store.ensure_session()
    pending = _pending_image()
    store.add_pending_attachment(session.id, pending)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=RecordingStreamingGateway(),
        ensure_run_hooks=lambda: hooks,
        model="vision-model",
    )
    refused = await controller.submit_draft("look", session_id=session.id)
    assert not refused.accepted
    assert refused.visible_copy == "Blocked by hook: policy"
    assert [row.attachment_id for row in store.pending_attachments(session.id)] == [
        pending.attachment_id
    ]
    hooks.blocked = False
    assert (await controller.submit_draft("look", session_id=session.id)).accepted
    assert not store.pending_attachments(session.id)


@pytest.mark.asyncio
async def test_queue_runs_skip_prompt_hook_and_each_emit_stop():
    from Tests.Chat.test_console_prompt_queue_coordinator import (
        SequencedGateway,
        _arm_controller,
        _queue,
    )

    hooks = Hooks()
    prompt_fires = []
    original_fire = hooks.fire_async

    async def fire(*args, **kwargs):
        prompt_fires.append(kwargs["data"]["prompt"])
        return await original_fire(*args, **kwargs)

    hooks.fire_async = fire
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)
    controller._ensure_run_hooks = lambda: hooks
    task = asyncio.create_task(
        controller.run_prompt_chain("manual", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "queued one")
    _queue(controller, session_id, "queued two")
    for index in range(3):
        await gateway.started[index].wait()
        gateway.release[index].set()
    await task
    assert prompt_fires == ["manual"]
    assert [event for event, _data in hooks.notifications] == ["Stop"] * 3


@pytest.mark.asyncio
async def test_queued_turn_cancellation_emits_one_stop():
    from Tests.Chat.test_console_prompt_queue_coordinator import (
        SequencedGateway,
        _arm_controller,
        _queue,
    )

    hooks = Hooks()
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)
    controller._ensure_run_hooks = lambda: hooks
    task = asyncio.create_task(
        controller.run_prompt_chain("manual", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "queued")
    gateway.release[0].set()
    await gateway.started[1].wait()
    assert controller.stop_active_run()
    await task
    assert [envelope["data"]["status"] for _, envelope in hooks.notifications] == [
        "completed",
        "cancelled",
    ]


@pytest.mark.asyncio
async def test_durable_postcommit_retry_keeps_hook_context_and_single_audit():
    hooks = Hooks(context="frozen hook context")
    store = ConsoleChatStore(persistence=FakePersistence())
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        ensure_run_hooks=lambda: hooks,
    )

    def fail_acceptance_projection():
        raise RuntimeError("projection temporarily unavailable")

    controller.on_submission_accepted = fail_acceptance_projection
    first = await controller.submit_draft("hello")
    assert first.accepted and first.preparation_id
    assert gateway.messages_seen is None
    hooks.context = "different next-turn context"
    controller.on_submission_accepted = None
    resumed = await controller.resume_durable_postcommit(first.preparation_id)
    assert resumed.accepted
    assert any(
        row.get("content") == "frozen hook context" for row in gateway.messages_seen
    )
    assert not any(row.get("content") == hooks.context for row in gateway.messages_seen)
    assert (
        len(
            [
                row
                for row in store.messages_for_session(store.active_session_id)
                if row.metadata and row.metadata.origin == "hook"
            ]
        )
        == 1
    )


@pytest.mark.parametrize("kind", ["approval", "skill_install", "skill_script"])
@pytest.mark.parametrize(
    "visible,background", [(True, False), (False, False), (True, True)]
)
def test_all_permission_rounds_notify_once_with_truthful_view_state(
    kind, visible, background
):
    hooks = Hooks()
    store = ConsoleChatStore()
    session = store.ensure_session()
    if background:
        session = store.create_session(activate=False)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=object(),
        ensure_run_hooks=lambda: hooks,
    )
    controller.app = SimpleNamespace(call_from_thread=lambda fn, *args: fn(*args))
    controller.set_pending_approval = lambda _payload: None
    controller.set_pending_skill_install = lambda _payload: None
    controller.set_pending_skill_script = lambda _payload: None
    host = controller._interrupt_host
    host.view_visible = visible
    state = {
        "event": threading.Event(),
        "decided": True,
        "session_id": session.id,
        "run_id": "run-1",
    }
    state["event"].set()
    payload = {
        "session_id": session.id,
        "run_id": "run-1",
        "round_id": "round-1",
        "request_id": "round-1",
        "url": "https://example.invalid/skill",
        "skill": "example",
        "mechanism": "python",
        "args": ["check"],
        "calls": [{"llm_name": "fs_write", "arguments": {"path": "notes"}}],
    }
    # Early admission and the host both register the same state. Re-entry
    # must not manufacture a second lifecycle notification.
    assert host.register_round(kind, "round-1", state)
    for _ in range(2):
        host.run_round(
            kind,
            "round-1",
            payload,
            state,
            session_id=session.id,
            owning_session_id=session.id,
            deadline=None,
            is_parked=background,
        )
    assert len(hooks.notifications) == 1
    event, envelope = hooks.notifications[0]
    assert event == "ApprovalRequested"
    assert envelope["run_id"] == "run-1"
    assert envelope["data"]["session_active"] is (visible and not background)
    assert envelope["data"]["calls"][0]["args_summary"]


@pytest.mark.parametrize("failure_site", ["accessor", "notify"])
def test_stop_observer_failure_does_not_prevent_terminal_publication(failure_site):
    store = ConsoleChatStore()
    session = store.ensure_session()
    failures = []
    hooks = Hooks()

    def accessor():
        if failure_site == "accessor":
            raise RuntimeError("observer unavailable")
        return hooks

    def broken_notify(*args, **kwargs):
        raise RuntimeError("observer unavailable")

    hooks.notify = broken_notify
    controller = ConsoleChatController(
        store=store,
        provider_gateway=RecordingStreamingGateway(),
        ensure_run_hooks=accessor,
    )
    controller.notify_run_failure = failures.append
    controller._arm_run_hooks_stop(session.id)
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING), session_id=session.id
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.FAILED, "provider failed"),
        session_id=session.id,
    )
    assert failures == ["provider failed"]
    assert session.id not in controller._run_hooks_stop_pending


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["close", "retry"])
async def test_durable_acceptance_emits_stop_after_failed_audit(monkeypatch, recovery):
    from Tests.Chat.console_close_helpers import close_controller_session

    hooks = Hooks(context="retained hook context")
    store = ConsoleChatStore(persistence=FakePersistence())
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        ensure_run_hooks=lambda: hooks,
    )
    original = store.persist_message_if_needed

    def unavailable_audit(message_id):
        if message_id.startswith("hook:"):
            raise OSError("audit publication temporarily unavailable")
        return original(message_id)

    monkeypatch.setattr(store, "persist_message_if_needed", unavailable_audit)
    result = await controller.submit_draft("hello")
    session_id = result.session_id
    assert result.accepted and result.preparation_id
    assert gateway.messages_seen is None
    assert not hooks.notifications
    if recovery == "close":
        close_controller_session(controller, session_id)
        terminal_status = ConsoleRunStatus.STOPPED
        expected_status = "cancelled"
    else:
        monkeypatch.setattr(store, "persist_message_if_needed", original)
        resumed = await controller.resume_durable_postcommit(result.preparation_id)
        assert resumed.accepted
        terminal_status = ConsoleRunStatus.COMPLETED
        expected_status = "completed"
    controller._set_run_state(ConsoleRunState(terminal_status), session_id=session_id)
    assert [
        (event, envelope["data"]["status"]) for event, envelope in hooks.notifications
    ] == [("Stop", expected_status)]


@pytest.mark.parametrize("kind", ["skill_install", "skill_script"])
def test_production_skill_approvals_keep_run_and_target_identity(kind):
    import json

    from tldw_chatbook.Agents.run_context import use_run_id

    hooks = Hooks()
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=object(),
        ensure_run_hooks=lambda: hooks,
    )
    controller.app = SimpleNamespace(call_from_thread=lambda fn, *args: fn(*args))
    controller._interrupt_host.view_visible = True

    def accept_install(payload):
        if payload:
            controller.resolve_pending_skill_install(
                True, request_id=payload["request_id"]
            )

    def accept_script(payload):
        if payload:
            controller.resolve_pending_skill_script(True, False, payload["request_id"])

    controller.set_pending_skill_install = accept_install
    controller.set_pending_skill_script = accept_script
    with use_run_id("owning-run"):
        if kind == "skill_install":
            assert controller.request_skill_install_confirm(
                "https://example.invalid/skill",
                session_id=session.id,
            )
        else:
            assert controller.request_skill_script_confirm(
                {
                    "skill_name": "real-skill",
                    "script_path": "scripts/check.py",
                    "mechanism": "python",
                    "args": ["verify"],
                },
                session_id=session.id,
            )["allow"]
    assert len(hooks.notifications) == 1
    event, envelope = hooks.notifications[0]
    assert event == "ApprovalRequested"
    assert envelope["run_id"] == "owning-run"
    assert envelope["session_id"] == session.id
    summary = json.loads(envelope["data"]["calls"][0]["args_summary"])
    if kind == "skill_install":
        assert summary["url"] == "https://example.invalid/skill"
    else:
        assert summary["skill_name"] == "real-skill"
        assert summary["script_path"] == "scripts/check.py"
        assert summary["args"] == ["verify"]
