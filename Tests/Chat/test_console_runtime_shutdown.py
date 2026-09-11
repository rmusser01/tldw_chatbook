"""Deterministic shutdown-scope tests for the app-owned Console runtime."""

from __future__ import annotations

import asyncio
import weakref

import pytest

from Tests.Chat.test_console_chat_controller import StreamingGateway
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleLifecycleRevisionChanged,
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleDispatchSettlementError,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


class _FleetDrainBridge:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.events: list[str] = []
        self.await_started = asyncio.Event()
        self.release = asyncio.Event()
        self.waiter_cancelled = False
        self.released: list[str] = []

    def on_fleet_drained(self, _name, _consumer) -> None:
        return None

    def on_fleet_activity(self, _name, _consumer) -> None:
        return None

    def fleet_snapshot(self, _conversation_id):
        return []

    def cancel_all_subagents(self, conversation_id: str) -> int:
        self.cancelled.append(conversation_id)
        self.events.append("cancel")
        return 1

    def fence_fleet(self, conversation_id: str, *, generation: int) -> bool:
        assert generation >= 1
        self.events.append(f"fleet-fence:{conversation_id}")
        return True

    def release_fleet_fence(self, conversation_id: str, *, generation: int) -> bool:
        assert generation >= 1
        self.released.append(conversation_id)
        return True

    def abort_fleet_fence(self, conversation_id: str, *, generation: int) -> bool:
        assert generation >= 1
        self.events.append(f"fleet-abort:{conversation_id}")
        return True

    async def await_fleet_terminal(self, _conversation_id: str) -> bool:
        self.await_started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.waiter_cancelled = True
            raise
        return True


def _runtime_with_fleet():
    store = ConsoleChatStore()
    session = store.ensure_session()
    bridge = _FleetDrainBridge()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        agent_bridge=bridge,
    )
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_agent_bridge(bridge)
    runtime.set_chat_controller(controller)
    return runtime, controller, bridge, store, session


def _recovery_record(runtime, session_id, turn_id):
    from Tests.Chat.test_console_turn_execution_context import _custody_configuration
    from tldw_chatbook.Chat.attachment_core import PendingAttachment
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

    attachment = PendingAttachment(
        "/recovery.png", "recovery.png", "image", "attachment", data=b"private bytes"
    )
    request = ConsoleTurnCustodyRequest(
        turn_id=turn_id,
        session_id=session_id,
        draft=f"private draft {turn_id}",
        configuration=_custody_configuration(session_id),
        attachment_ids=(attachment.attachment_id,),
    )
    record = runtime._register_custody(request, (attachment,))
    return record, weakref.ref(attachment)


def _seed_recovery(runtime, session_id, turn_id):
    record, attachment_ref = _recovery_record(runtime, session_id, turn_id)
    runtime._record_turn_recovery(record)
    runtime._release_custody(turn_id)
    assert (
        runtime.recoveries_for_session(session_id)[-1].attachments[0]
        is attachment_ref()
    )
    return attachment_ref


@pytest.mark.asyncio
async def test_successful_close_releases_only_owning_recovery_inputs():
    runtime, controller, bridge, store, session = _runtime_with_fleet()
    other = store.create_session(ephemeral=True)
    closed_attachment = _seed_recovery(runtime, session.id, "closed-turn")
    kept_attachment = _seed_recovery(runtime, other.id, "kept-turn")
    bridge.release.set()
    try:
        await runtime.close_session(
            session.id,
            expected_revision=controller.lifecycle_impact(
                session_id=session.id
            ).revision,
        )
        assert session.id not in {item.id for item in store.sessions()}
        assert runtime.recoveries_for_session(session.id) == ()
        assert session.id not in runtime._recovery_turns_by_session
        assert "closed-turn" not in runtime._turn_recoveries
        assert closed_attachment() is None
        kept = runtime.recoveries_for_session(other.id)[0]
        assert kept.draft == "private draft kept-turn"
        assert kept.attachments[0] is kept_attachment()
        runtime.restore_turn_recovery("kept-turn")
        assert store.session_draft(other.id) == "private draft kept-turn"
        assert store.pending_attachments(other.id)[0] is kept_attachment()
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ("stale_close", "failed_close", "stale_dispose"))
async def test_refused_terminal_fence_preserves_recovery_inputs(monkeypatch, refusal):
    runtime, controller, bridge, store, session = _runtime_with_fleet()
    attachment_ref = _seed_recovery(runtime, session.id, "retained-turn")
    revision = controller.lifecycle_impact(session_id=session.id).revision
    whole_revision = controller.lifecycle_impact().revision
    if refusal == "failed_close":

        def refuse(*args, **kwargs):
            raise RuntimeError("reversible close refusal")

        monkeypatch.setattr(controller, "begin_session_close", refuse)
    else:
        controller._advance_lifecycle_revision(session.id)
    bridge.release.set()
    try:
        with pytest.raises(RuntimeError):
            if refusal == "stale_dispose":
                runtime.begin_dispose(expected_revision=whole_revision)
            else:
                await runtime.close_session(session.id, expected_revision=revision)
        runtime._raise_if_disposed_or_session_fenced(session.id)
        entry = runtime.recoveries_for_session(session.id)[0]
        assert entry.draft == "private draft retained-turn"
        assert entry.attachments[0] is attachment_ref()
        runtime.restore_turn_recovery("retained-turn")
        assert store.pending_attachments(session.id)[0] is attachment_ref()
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("begin_first", (False, True))
async def test_dispose_releases_all_retained_recovery_inputs(begin_first):
    runtime, controller, bridge, store, session = _runtime_with_fleet()
    other = store.create_session(ephemeral=True)
    first = _seed_recovery(runtime, session.id, "first-turn")
    second = _seed_recovery(runtime, other.id, "second-turn")
    bridge.release.set()
    if begin_first:
        runtime.begin_dispose(expected_revision=controller.lifecycle_impact().revision)
    await runtime.dispose()
    assert runtime._turn_recoveries == {}
    assert runtime._recovery_turns_by_session == {}
    assert first() is None and second() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ("close", "dispose"))
async def test_late_predurable_failure_cannot_recreate_recovery_after_fence(terminal):
    runtime, controller, bridge, store, session = _runtime_with_fleet()
    record, attachment_ref = _recovery_record(runtime, session.id, "late-turn")
    release_failure = asyncio.Event()
    callback_done = asyncio.Event()

    async def fail_late():
        try:
            await release_failure.wait()
        except asyncio.CancelledError:
            # Model a provider boundary that reports its already-in-flight
            # failure after cancellation. The outer finally always releases it.
            await release_failure.wait()
        raise RuntimeError("late pre-durable failure")

    def finish(task):
        try:
            runtime._finish_custodied_turn(
                task,
                turn_id="late-turn",
                recover_before_acceptance=True,
                terminal_callback=None,
            )
        finally:
            callback_done.set()

    task = asyncio.create_task(fail_late())
    record.task = task
    task.add_done_callback(finish)
    operation = asyncio.create_task(
        runtime.close_session(
            session.id,
            expected_revision=controller.lifecycle_impact(
                session_id=session.id
            ).revision,
        )
        if terminal == "close"
        else runtime.dispose()
    )
    try:
        await asyncio.wait_for(bridge.await_started.wait(), 1)
        release_failure.set()
        await asyncio.wait_for(callback_done.wait(), 1)
        assert not task.cancelled()
        assert isinstance(task.exception(), RuntimeError)
        assert runtime.recoveries_for_session(session.id) == ()
        assert record.request is None and record.inputs.attachments == ()
        assert attachment_ref() is None
        bridge.release.set()
        await asyncio.wait_for(operation, 2)
        assert runtime._turn_recoveries == {}
        assert runtime._recovery_turns_by_session == {}
    finally:
        release_failure.set()
        bridge.release.set()
        await asyncio.gather(task, operation, return_exceptions=True)
        await runtime.dispose()


def test_runtime_dispose_fence_rejects_a_stale_whole_runtime_revision():
    runtime, controller, _bridge, store, session = _runtime_with_fleet()
    revision = controller.lifecycle_impact().revision
    controller._advance_lifecycle_revision(session.id)

    with pytest.raises(ConsoleLifecycleRevisionChanged):
        runtime.begin_dispose(expected_revision=revision)

    runtime._raise_if_disposed_or_session_fenced(session.id)
    assert [item.id for item in store.sessions()] == [session.id]


def test_runtime_dispose_rechecks_revision_after_fleet_admission_is_fenced():
    class _RacingFleetBridge(_FleetDrainBridge):
        def on_fleet_activity(self, _name, consumer) -> None:
            self.activity = consumer

        def fence_fleet(self, conversation_id: str, *, generation: int) -> bool:
            self.events.append(f"fleet-fence:{conversation_id}")
            self.activity(conversation_id)
            return True

    store = ConsoleChatStore()
    session = store.ensure_session()
    bridge = _RacingFleetBridge()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        agent_bridge=bridge,
    )
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_agent_bridge(bridge)
    runtime.set_chat_controller(controller)
    revision = controller.lifecycle_impact().revision

    with pytest.raises(ConsoleLifecycleRevisionChanged):
        runtime.begin_dispose(expected_revision=revision)

    assert bridge.events == [
        f"fleet-fence:{session.id}",
        f"fleet-abort:{session.id}",
    ]
    assert runtime._disposed is False
    runtime._raise_if_disposed_or_session_fenced(session.id)


def test_runtime_begin_dispose_fences_fleet_before_controller_shutdown():
    runtime, controller, bridge, _store, session = _runtime_with_fleet()
    original_begin_shutdown = controller.begin_shutdown

    def ordered_begin_shutdown() -> None:
        assert runtime._disposed is True
        with pytest.raises(RuntimeError, match="disposed"):
            runtime._raise_if_disposed_or_session_fenced(session.id)
        bridge.events.append("controller-shutdown")
        original_begin_shutdown()

    controller.begin_shutdown = ordered_begin_shutdown
    revision = controller.lifecycle_impact().revision

    runtime.begin_dispose(expected_revision=revision)

    assert bridge.events == [
        f"fleet-fence:{session.id}",
        "controller-shutdown",
    ]


def test_runtime_begin_dispose_stays_terminal_if_controller_teardown_raises():
    runtime, controller, bridge, _store, session = _runtime_with_fleet()

    def fail_after_terminal_fence() -> None:
        raise RuntimeError("controller teardown failed")

    controller.begin_shutdown = fail_after_terminal_fence
    revision = controller.lifecycle_impact().revision

    runtime.begin_dispose(expected_revision=revision)

    assert runtime._disposed is True
    with pytest.raises(RuntimeError, match="disposed"):
        runtime._raise_if_disposed_or_session_fenced(session.id)
    assert bridge.events == [f"fleet-fence:{session.id}"]


def test_terminal_completion_wins_before_a_late_close_stop_gate():
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.append_stream_chunk(assistant.id, "committed answer")
    store.mark_message_complete(assistant.id)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."),
        session_id=session.id,
    )

    terminal = controller._mark_stream_stopped(
        assistant.id,
        visible_copy="Session closed.",
    )

    assert terminal.status == "complete"
    assert terminal.content == "committed answer"
    assert controller.run_state_for(session.id).status is ConsoleRunStatus.COMPLETED


def test_terminal_cancellation_wins_and_drops_a_late_provider_chunk():
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.append_stream_chunk(assistant.id, "before cancellation")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
    )

    controller._mark_stream_stopped(
        assistant.id,
        visible_copy="Session closed.",
    )
    late = store.append_stream_chunk(assistant.id, "late provider output")

    assert late.status == "stopped"
    assert late.content == "before cancellation"
    with pytest.raises(ValueError, match="Cannot mark a stopped message terminal"):
        store.mark_message_complete(assistant.id)
    assert controller.run_state_for(session.id).status is ConsoleRunStatus.STOPPED


def test_session_close_fences_late_wakes_before_cancelling_children():
    runtime, controller, bridge, store, session = _runtime_with_fleet()

    class _OrderedWakeFence:
        def fence_conversation(self, conversation_id: str, *, generation: int) -> None:
            assert conversation_id == session.id
            assert generation == 1
            bridge.events.append("fence")

    controller._fleet_wake = _OrderedWakeFence()
    revision = controller.lifecycle_impact(session_id=session.id).revision

    ticket = controller.begin_session_close(
        session.id,
        expected_revision=revision,
    )

    assert bridge.events == [f"fleet-fence:{session.id}", "fence", "cancel"]
    assert controller.finalize_session_close(ticket) is None
    assert store.sessions() == []
    assert runtime.chat_controller is controller


def test_failed_stop_settlement_aborts_the_provisional_fleet_fence():
    _runtime, controller, bridge, store, session = _runtime_with_fleet()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="partial",
    )
    controller._active_assistant_message_ids[session.id] = assistant.id

    def refuse_stop(*_args, **_kwargs):
        raise ConsoleDispatchSettlementError("durable stop refused")

    controller._mark_stream_stopped = refuse_stop
    controller._restore_dispatch_recovery_after_settlement_failure = (
        lambda *_args, **_kwargs: None
    )
    revision = controller.lifecycle_impact(session_id=session.id).revision

    with pytest.raises(ConsoleDispatchSettlementError, match="durable stop refused"):
        controller.begin_session_close(
            session.id,
            expected_revision=revision,
        )

    assert bridge.events == [
        f"fleet-fence:{session.id}",
        f"fleet-abort:{session.id}",
    ]
    assert session.id not in controller._session_close_generations


@pytest.mark.asyncio
async def test_session_close_fences_admission_and_deletes_only_after_fleet_drain():
    runtime, controller, bridge, store, session = _runtime_with_fleet()
    revision = controller.lifecycle_impact(session_id=session.id).revision

    closing = asyncio.create_task(
        runtime.close_session(session.id, expected_revision=revision)
    )
    await bridge.await_started.wait()

    assert [item.id for item in store.sessions()] == [session.id]
    with pytest.raises(RuntimeError, match="closed"):
        runtime._raise_if_disposed_or_session_fenced(session.id)
    assert bridge.cancelled == [session.id]

    bridge.release.set()
    await closing

    assert store.sessions() == []
    assert bridge.released == [session.id]


@pytest.mark.asyncio
async def test_session_close_revision_mismatch_reopens_runtime_admission():
    runtime, controller, _bridge, store, session = _runtime_with_fleet()
    stale_revision = controller.lifecycle_impact(session_id=session.id).revision
    controller._advance_lifecycle_revision(session.id)

    with pytest.raises(RuntimeError, match="activity changed"):
        await runtime.close_session(
            session.id,
            expected_revision=stale_revision,
        )

    runtime._raise_if_disposed_or_session_fenced(session.id)
    assert [item.id for item in store.sessions()] == [session.id]


@pytest.mark.asyncio
async def test_session_close_rechecks_revision_after_fleet_admission_is_fenced():
    class _RacingFleetBridge(_FleetDrainBridge):
        def on_fleet_activity(self, _name, consumer) -> None:
            self.activity = consumer

        def fence_fleet(self, conversation_id: str, *, generation: int) -> bool:
            self.events.append(f"fleet-fence:{conversation_id}")
            self.activity(conversation_id)
            return True

    store = ConsoleChatStore()
    session = store.ensure_session()
    bridge = _RacingFleetBridge()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        agent_bridge=bridge,
    )
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_agent_bridge(bridge)
    runtime.set_chat_controller(controller)
    revision = controller.lifecycle_impact(session_id=session.id).revision

    with pytest.raises(ConsoleLifecycleRevisionChanged):
        await runtime.close_session(session.id, expected_revision=revision)

    assert bridge.events == [
        f"fleet-fence:{session.id}",
        f"fleet-abort:{session.id}",
    ]
    runtime._raise_if_disposed_or_session_fenced(session.id)
    assert [item.id for item in store.sessions()] == [session.id]


@pytest.mark.asyncio
async def test_session_close_timeout_is_bounded_and_still_finalizes_deletion():
    runtime, controller, bridge, store, session = _runtime_with_fleet()
    revision = controller.lifecycle_impact(session_id=session.id).revision

    await asyncio.wait_for(
        runtime.close_session(
            session.id,
            expected_revision=revision,
            timeout_seconds=0.01,
        ),
        timeout=0.5,
    )
    await asyncio.sleep(0)

    assert store.sessions() == []
    assert bridge.waiter_cancelled is True
    assert bridge.released == []


@pytest.mark.asyncio
async def test_session_close_keeps_fences_when_fleet_drain_fails():
    class _FailedFleetDrainBridge(_FleetDrainBridge):
        async def await_fleet_terminal(self, _conversation_id: str) -> bool:
            raise RuntimeError("fleet drain unavailable")

    store = ConsoleChatStore()
    session = store.ensure_session()
    bridge = _FailedFleetDrainBridge()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        agent_bridge=bridge,
    )
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_agent_bridge(bridge)
    runtime.set_chat_controller(controller)
    revision = controller.lifecycle_impact(session_id=session.id).revision

    await runtime.close_session(
        session.id,
        expected_revision=revision,
    )

    assert store.sessions() == []
    assert bridge.released == []


def test_close_generations_do_not_repeat_across_saved_conversation_incarnations():
    runtime, controller, _bridge, store, first = _runtime_with_fleet()
    first.persisted_conversation_id = "saved-conversation"
    first_ticket = controller.begin_session_close(
        first.id,
        expected_revision=controller.lifecycle_impact(session_id=first.id).revision,
    )
    controller.finalize_session_close(first_ticket)
    assert controller.release_session_close_fences(first_ticket) is True

    second = store.create_session(title="Reopened")
    second.persisted_conversation_id = "saved-conversation"
    second_ticket = controller.begin_session_close(
        second.id,
        expected_revision=controller.lifecycle_impact(session_id=second.id).revision,
    )

    assert second_ticket.generation > first_ticket.generation


def test_wake_release_failure_keeps_the_stronger_fleet_fence_latched():
    _runtime, controller, bridge, _store, session = _runtime_with_fleet()

    class _WakeReleaseRefusal:
        def fence_conversation(
            self, _conversation_id: str, *, generation: int
        ) -> None:
            assert generation >= 1

        def release_conversation_fence(
            self, _conversation_id: str, *, generation: int
        ) -> bool:
            assert generation >= 1
            return False

    controller._fleet_wake = _WakeReleaseRefusal()
    ticket = controller.begin_session_close(
        session.id,
        expected_revision=controller.lifecycle_impact(session_id=session.id).revision,
    )
    controller.finalize_session_close(ticket)

    assert controller.release_session_close_fences(ticket) is False
    assert bridge.released == []


def test_app_shutdown_prevents_an_inflight_session_close_from_releasing_fences():
    _runtime, controller, bridge, _store, session = _runtime_with_fleet()
    ticket = controller.begin_session_close(
        session.id,
        expected_revision=controller.lifecycle_impact(session_id=session.id).revision,
    )
    controller.finalize_session_close(ticket)

    controller.begin_shutdown()

    assert controller.release_session_close_fences(ticket) is False
    assert bridge.released == []


@pytest.mark.asyncio
async def test_cancelled_close_finishes_its_bounded_drain_and_deletion():
    runtime, controller, bridge, store, session = _runtime_with_fleet()
    revision = controller.lifecycle_impact(session_id=session.id).revision
    closing = asyncio.create_task(
        runtime.close_session(
            session.id,
            expected_revision=revision,
            timeout_seconds=0.5,
        )
    )
    await bridge.await_started.wait()

    closing.cancel()
    await asyncio.sleep(0)

    assert closing.done() is False
    assert [item.id for item in store.sessions()] == [session.id]

    bridge.release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert store.sessions() == []


@pytest.mark.asyncio
async def test_runtime_dispose_bounds_controller_shutdown_and_closes_gateway():
    shutdown_started = asyncio.Event()
    shutdown_cancelled = asyncio.Event()
    gateway_closed = asyncio.Event()

    class _Controller:
        def begin_shutdown(self) -> None:
            return None

        async def shutdown(self) -> None:
            shutdown_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                shutdown_cancelled.set()
                raise

    class _Gateway:
        async def aclose(self) -> None:
            gateway_closed.set()

    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_controller(_Controller())
    runtime.set_provider_gateway(_Gateway())

    await asyncio.wait_for(runtime.dispose(timeout_seconds=0.01), timeout=0.5)
    await asyncio.sleep(0)

    assert shutdown_started.is_set()
    assert shutdown_cancelled.is_set()
    assert gateway_closed.is_set()
    assert runtime.generation == 1


@pytest.mark.asyncio
async def test_runtime_dispose_bounds_an_uncooperative_gateway_close():
    close_started = asyncio.Event()
    close_cancelled = asyncio.Event()

    class _Gateway:
        async def aclose(self) -> None:
            close_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                close_cancelled.set()
                raise

    runtime = ConsoleRuntime(app=None)
    runtime.set_provider_gateway(_Gateway())

    await asyncio.wait_for(runtime.dispose(timeout_seconds=0.01), timeout=0.5)
    await asyncio.sleep(0)

    assert close_started.is_set()
    assert close_cancelled.is_set()
