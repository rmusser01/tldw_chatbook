"""Joined controller/coordinator tests for sequential Console prompt queues."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunMarker,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore as _ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpoint,
    ConsoleDispatchCheckpointState,
    ConsoleDispatchResultStatus,
    ConsoleDispatchWriteResult,
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicySnapshot
from tldw_chatbook.Chat.console_prompt_queue import (
    PromptQueueMode,
    PromptQueuePauseReason,
    PromptQueueReservation,
)
from tldw_chatbook.Chat.console_prompt_queue_coordinator import (
    QueueGenerationAuthorization,
)


class ConsoleChatStore(_ConsoleChatStore):
    """Test store whose intentionally db-less sessions are explicitly ephemeral."""

    def create_session(self, **kwargs):
        kwargs.setdefault("ephemeral", self.persistence is None)
        return super().create_session(**kwargs)


class SequencedGateway:
    def __init__(self, *, fail_call: int | None = None) -> None:
        self.fail_call = fail_call
        self.started = [asyncio.Event() for _ in range(5)]
        self.release = [asyncio.Event() for _ in range(5)]
        self.user_turns: list[str] = []

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
                "resolved_destination": ConsoleResolvedDestination(
                    provider="llama_cpp",
                    model="test-model",
                    endpoint_identity="http://127.0.0.1:9099",
                    egress_class=ConsoleEgressClass.ON_DEVICE,
                ),
            },
        )()

    async def stream_chat(self, resolution, messages, **kwargs):
        call = len(self.user_turns)
        user_text = next(
            message["content"]
            for message in reversed(messages)
            if message.get("role") == "user"
        )
        self.user_turns.append(user_text)
        self.started[call].set()
        await self.release[call].wait()
        yield f"reply-{call + 1}"
        if self.fail_call == call:
            raise RuntimeError("planned stream failure")


class RecordingPromptHistory:
    def __init__(self) -> None:
        self.items: list[str] = []

    async def append(self, text: str) -> None:
        self.items.append(text)


class RecordingPersistence:
    def __init__(self) -> None:
        self.created_messages: list[dict] = []
        self._policy_snapshot = None
        self.console_library_policy_repository = SimpleNamespace(read=self._read_policy)
        self.console_dispatch_repository = self
        self._checkpoint = None

    def _read_policy(self, conversation_id):
        del conversation_id
        return SimpleNamespace(durable_policy=object(), snapshot=self._policy_snapshot)

    def _cas_state(self, transition):
        checkpoint = self._checkpoint
        if checkpoint is None:
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.NOT_FOUND, None, None, None
            )
        checkpoint = replace(
            checkpoint,
            state=transition.new_state,
            checkpoint_revision=checkpoint.checkpoint_revision + 1,
            assistant_message_version=checkpoint.assistant_message_version + 1,
            attempt_id=transition.new_attempt_id,
        )
        self._checkpoint = checkpoint
        return ConsoleDispatchWriteResult(
            ConsoleDispatchResultStatus.COMMITTED,
            checkpoint,
            checkpoint.assistant_message_version,
            "fake-payload-hash",
        )

    cas_state = _cas_state

    def settle_with_assistant(self, settlement):
        checkpoint = self._checkpoint
        if checkpoint is None:
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.NOT_FOUND, None, None, None
            )
        self._checkpoint = None
        return ConsoleDispatchWriteResult(
            ConsoleDispatchResultStatus.COMMITTED,
            None,
            checkpoint.assistant_message_version + 1,
            "fake-terminal-hash",
        )

    def commit_durable_turn(self, *, acceptance, policy_candidate, conversation_kwargs):
        del conversation_kwargs
        self._policy_snapshot = ConsoleLibraryPolicySnapshot(
            auto_retrieve=policy_candidate.auto_retrieve,
            assistant_access=policy_candidate.assistant_access,
            policy_revision=1,
            source="durable",
        )
        self.created_messages.extend(
            (
                {
                    "sender": "user",
                    "content": acceptance.user_content,
                    "message_id": acceptance.user_message_id,
                },
                {
                    "sender": "assistant",
                    "content": "",
                    "message_id": acceptance.assistant_message_id,
                },
            )
        )
        checkpoint = ConsoleDispatchCheckpoint(
            assistant_message_id=acceptance.assistant_message_id,
            user_message_id=acceptance.user_message_id,
            conversation_id=acceptance.conversation_id,
            preparation_id=acceptance.preparation_id,
            attempt_id=acceptance.attempt_id,
            state=ConsoleDispatchCheckpointState.ACCEPTED,
            checkpoint_revision=1,
            user_message_version=1,
            assistant_message_version=1,
            origin=acceptance.origin,
            queue_entry_id=acceptance.queue_entry_id,
            frozen_authority=acceptance.frozen_authority,
            resolved_destination=acceptance.resolved_destination,
            reconstructability=acceptance.reconstructability,
        )
        self._checkpoint = checkpoint
        return checkpoint

    def create_conversation(self, **kwargs):
        return "conversation-1"

    def create_message(
        self,
        *,
        conversation_id,
        sender,
        content,
        image_data,
        image_mime_type,
        message_id=None,
        parent_message_id=None,
        feedback=None,
    ):
        self.created_messages.append(
            {
                "sender": sender,
                "content": content,
                "message_id": message_id,
            }
        )
        return f"persisted-{len(self.created_messages)}"

    def update_message_content(self, **kwargs):
        return True


class RefuseSecondGateway(SequencedGateway):
    def __init__(self) -> None:
        super().__init__()
        self.resolve_calls = 0

    async def resolve_for_send(self, selection):
        self.resolve_calls += 1
        if self.resolve_calls == 2:
            return type(
                "Resolution",
                (),
                {"ready": False, "visible_copy": "Provider blocked: unavailable"},
            )()
        return await super().resolve_for_send(selection)


class BlockSecondReadinessGateway(SequencedGateway):
    def __init__(self) -> None:
        super().__init__()
        self.resolve_calls = 0
        self.second_resolve_started = asyncio.Event()
        self.release_second_resolve = asyncio.Event()

    async def resolve_for_send(self, selection):
        self.resolve_calls += 1
        if self.resolve_calls == 2:
            self.second_resolve_started.set()
            await self.release_second_resolve.wait()
        return await super().resolve_for_send(selection)


def _arm_controller(gateway: SequencedGateway):
    store = ConsoleChatStore()
    session = store.ensure_session(title="Queue owner")
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    return controller, store, session.id


def _queue(controller: ConsoleChatController, session_id: str, text: str) -> str:
    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    result = controller.queue_prompt(
        session_id,
        text=text,
        expected_revision=snapshot.revision,
    )
    assert result.applied
    assert result.entry_id is not None
    return result.entry_id


def test_controller_refuses_unsafe_queue_text_before_admission() -> None:
    controller, store, session_id = _arm_controller(SequencedGateway())
    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    snapshot = controller.prompt_queue_registry.begin_chain(
        session_id,
        context_epoch=store.conversation_context_epoch(session_id),
        expected_revision=snapshot.revision,
    ).snapshot

    result = controller.queue_prompt(
        session_id,
        text="<script>alert('queued')</script>",
        expected_revision=snapshot.revision,
    )

    assert not result.applied
    assert result.status.value == "invalid"
    assert controller.prompt_queue_registry.snapshot(session_id).total_count == 0


@pytest.mark.asyncio
async def test_lifecycle_impact_counts_claimed_entries_without_prompt_content():
    gateway = BlockSecondReadinessGateway()
    controller, _store, session_id = _arm_controller(gateway)

    initial = controller.lifecycle_impact()
    assert initial.live_run_count == 0
    assert initial.queued_session_count == 0
    assert initial.unsent_prompt_count == 0

    task = asyncio.create_task(
        controller.run_prompt_chain("manual", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "private first follow-up")
    _queue(controller, session_id, "private second follow-up")

    gateway.release[0].set()
    await gateway.second_resolve_started.wait()

    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    impact = controller.lifecycle_impact()
    assert snapshot.claimed_count == 1
    assert impact.revision > initial.revision
    assert impact.live_run_count == 1
    assert impact.queued_session_count == 1
    assert impact.unsent_prompt_count == 2
    assert "private first follow-up" not in repr(impact)
    assert "private second follow-up" not in repr(impact)

    gateway.release_second_resolve.set()
    await gateway.started[1].wait()
    gateway.release[1].set()
    await gateway.started[2].wait()
    gateway.release[2].set()
    await task


@pytest.mark.asyncio
async def test_lifecycle_impact_does_not_describe_paused_queue_as_live_run():
    gateway = SequencedGateway(fail_call=0)
    controller, _store, session_id = _arm_controller(gateway)

    task = asyncio.create_task(
        controller.run_prompt_chain("manual", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "wait until recovery")
    gateway.release[0].set()
    await task

    activity = controller.activity_for(session_id)
    impact = controller.lifecycle_impact()
    assert activity.queue_paused is True
    assert impact.live_run_count == 0
    assert impact.queued_session_count == 1
    assert impact.unsent_prompt_count == 1


def test_session_lifecycle_impact_is_revisioned_independently():
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)
    other = controller.new_session(title="Other", ephemeral=True)

    session_before = controller.lifecycle_impact(session_id=session_id)
    fleet_before = controller.lifecycle_impact()
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING),
        session_id=other.id,
    )

    session_after = controller.lifecycle_impact(session_id=session_id)
    fleet_after = controller.lifecycle_impact()
    assert session_after == session_before
    assert fleet_after.revision > fleet_before.revision
    assert fleet_after.live_run_count == 1


@pytest.mark.asyncio
async def test_three_turn_chain_drains_fifo_with_one_slot_and_explicit_origins():
    gateway = SequencedGateway()
    controller, store, session_id = _arm_controller(gateway)
    manual_accepts = 0
    queued_accepts = []
    history = RecordingPromptHistory()

    def accepted_manual() -> None:
        nonlocal manual_accepts
        manual_accepts += 1

    controller.on_submission_accepted = accepted_manual
    controller.on_queued_submission_accepted = queued_accepts.append
    controller.prompt_history = history

    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    second_id = _queue(controller, session_id, "two")
    third_id = _queue(controller, session_id, "three")

    activity = controller.activity_for(session_id)
    assert activity.occupies_slot
    assert activity.accepted_live_turn
    assert activity.queued_count == 2
    assert controller.in_flight_run_count() == 1

    gateway.release[0].set()
    await gateway.started[1].wait()
    assert controller.in_flight_run_count() == 1
    assert controller.run_marker_for(session_id) is ConsoleRunMarker.RUNNING

    gateway.release[1].set()
    await gateway.started[2].wait()
    assert controller.in_flight_run_count() == 1

    gateway.release[2].set()
    result = await task

    assert result.accepted
    assert result.session_id == session_id
    assert result.user_message_id
    assert result.assistant_message_id
    assert result.terminal_status is ConsoleRunStatus.COMPLETED
    assert result.origin is ConsoleSubmissionOrigin.MANUAL
    assert result.committed_context_epoch == store.conversation_context_epoch(
        session_id
    )
    assert gateway.user_turns == ["one", "two", "three"]
    assert history.items == ["one", "two", "three"]
    assert manual_accepts == 1
    assert [(event.session_id, event.entry_id) for event in queued_accepts] == [
        (session_id, second_id),
        (session_id, third_id),
    ]
    assert controller.in_flight_run_count() == 0
    assert controller.prompt_queue_registry.snapshot(session_id).total_count == 0
    assert [
        message.content
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.USER
    ] == ["one", "two", "three"]


@pytest.mark.asyncio
async def test_intermediate_completions_emit_only_one_final_background_outcome():
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)
    outcomes: list[tuple[str, ConsoleRunStatus]] = []
    controller.notify_run_outcome = lambda sid, status: outcomes.append((sid, status))

    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "two")
    controller.new_session(title="Viewed elsewhere", ephemeral=True)

    gateway.release[0].set()
    await gateway.started[1].wait()
    assert outcomes == []
    assert controller.run_marker_for(session_id) is ConsoleRunMarker.RUNNING

    gateway.release[1].set()
    await task

    assert outcomes == [(session_id, ConsoleRunStatus.COMPLETED)]
    assert controller.run_marker_for(session_id) is ConsoleRunMarker.FINISHED_OK


@pytest.mark.asyncio
async def test_failed_accepted_queued_turn_pauses_remaining_without_requeueing_it():
    gateway = SequencedGateway(fail_call=1)
    controller, store, session_id = _arm_controller(gateway)
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "two")
    third_id = _queue(controller, session_id, "three")

    gateway.release[0].set()
    await gateway.started[1].wait()
    gateway.release[1].set()
    await task

    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.pause_reason is PromptQueuePauseReason.FAILED
    assert snapshot.reservation is PromptQueueReservation.RELEASED
    assert [entry.entry_id for entry in snapshot.entries] == [third_id]
    assert gateway.user_turns == ["one", "two"]
    assert [
        message.content
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.USER
    ] == ["one", "two"]


@pytest.mark.asyncio
async def test_unexpected_exception_after_queued_acceptance_keeps_only_future_work():
    gateway = SequencedGateway()
    controller, store, session_id = _arm_controller(gateway)

    async def raise_after_acceptance(**kwargs):
        raise RuntimeError("planned post-acceptance exception")

    def arm_exception(_event) -> None:
        controller._stream_assistant_response = raise_after_acceptance

    controller.on_queued_submission_accepted = arm_exception
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "two")
    third_id = _queue(controller, session_id, "three")
    gateway.release[0].set()

    with pytest.raises(RuntimeError, match="post-acceptance"):
        await task

    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.pause_reason is PromptQueuePauseReason.FAILED
    assert [entry.entry_id for entry in snapshot.entries] == [third_id]
    assert [
        message.content
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.USER
    ] == ["one", "two"]


@pytest.mark.asyncio
async def test_context_change_before_first_admission_pauses_for_explicit_review():
    gateway = SequencedGateway()
    controller, store, session_id = _arm_controller(gateway)
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    user = next(
        message
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.USER
    )
    store.set_session_context_summary(session_id, "summary changed", user.id)
    queued_id = _queue(controller, session_id, "two")

    gateway.release[0].set()
    gateway.release[1].set()  # lets an illicit mutated dispatch fail, not hang
    await task

    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.pause_reason is PromptQueuePauseReason.CONTEXT_CHANGED
    assert [entry.entry_id for entry in snapshot.entries] == [queued_id]
    assert gateway.user_turns == ["one"]


@pytest.mark.asyncio
async def test_queued_origin_requires_coordinator_authority():
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)

    with pytest.raises(PermissionError):
        await controller.submit_draft(
            "forged",
            session_id=session_id,
            origin=ConsoleSubmissionOrigin.QUEUED,
            queue_entry_id="forged-entry",
        )


@pytest.mark.asyncio
async def test_stop_pauses_immediately_and_resume_next_dispatches_once():
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "two")

    assert controller.stop_active_run()
    stopped_snapshot = controller.prompt_queue_registry.snapshot(session_id)
    assert stopped_snapshot.mode is PromptQueueMode.PAUSED
    assert stopped_snapshot.pause_reason is PromptQueuePauseReason.STOPPED
    assert stopped_snapshot.reservation is PromptQueueReservation.RELEASED
    await task

    resume_task = asyncio.create_task(controller.resume_prompt_queue(session_id))
    await gateway.started[1].wait()
    starting = controller.prompt_queue_registry.snapshot(session_id)
    assert starting.total_count == 0  # accepted boundary settled the claim
    gateway.release[1].set()
    await resume_task

    assert gateway.user_turns == ["one", "two"]
    assert controller.prompt_queue_registry.snapshot(session_id).total_count == 0


@pytest.mark.asyncio
async def test_failed_retry_adopts_authorized_epoch_then_drains_next_prompt():
    gateway = SequencedGateway(fail_call=0)
    controller, store, session_id = _arm_controller(gateway)
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "two")
    gateway.release[0].set()
    await task
    failed = next(
        message
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.ASSISTANT and message.status == "failed"
    )

    recovery = asyncio.create_task(controller.retry_failed_queue_turn(failed.id))
    await gateway.started[1].wait()
    gateway.release[1].set()
    await gateway.started[2].wait()
    gateway.release[2].set()
    await recovery

    assert gateway.user_turns == ["one", "one", "two"]
    assert controller.prompt_queue_registry.snapshot(session_id).total_count == 0


@pytest.mark.asyncio
async def test_failed_retry_stays_on_queue_owner_after_viewed_session_switch():
    gateway = SequencedGateway(fail_call=0)
    controller, store, session_id = _arm_controller(gateway)
    task = asyncio.create_task(
        controller.run_prompt_chain("owner turn", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "owner follow-up")
    gateway.release[0].set()
    await task
    failed = next(
        message
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.ASSISTANT and message.status == "failed"
    )

    viewed = controller.new_session(title="Viewed elsewhere", ephemeral=True)
    assert store.active_session_id == viewed.id
    recovery = asyncio.create_task(controller.retry_failed_queue_turn(failed.id))
    await gateway.started[1].wait()
    gateway.release[1].set()
    await gateway.started[2].wait()
    gateway.release[2].set()
    await recovery

    assert gateway.user_turns == [
        "owner turn",
        "owner turn",
        "owner follow-up",
    ]
    assert not store.messages_for_session(viewed.id)


@pytest.mark.asyncio
async def test_preaccept_refusal_returns_claim_to_head_and_writes_no_history():
    gateway = RefuseSecondGateway()
    controller, _store, session_id = _arm_controller(gateway)
    history = RecordingPromptHistory()
    controller.prompt_history = history
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    queued_id = _queue(controller, session_id, "two")
    gateway.release[0].set()
    await task

    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.pause_reason is PromptQueuePauseReason.DISPATCH_REFUSED
    assert [entry.entry_id for entry in snapshot.entries] == [queued_id]
    assert history.items == ["one"]
    assert gateway.user_turns == ["one"]


@pytest.mark.asyncio
async def test_shutdown_tombstones_before_cancel_and_never_starts_next_prompt():
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)
    chain_task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "two")

    await controller.shutdown()
    await chain_task

    assert gateway.user_turns == ["one"]
    assert controller.prompt_queue_registry.shutting_down


@pytest.mark.asyncio
async def test_shutdown_during_claimed_readiness_cannot_accept_or_dispatch_it():
    gateway = BlockSecondReadinessGateway()
    controller, store, session_id = _arm_controller(gateway)
    accepted_entries = []
    controller.on_queued_submission_accepted = accepted_entries.append
    chain_task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "two")
    gateway.release[0].set()
    await gateway.second_resolve_started.wait()

    await controller.shutdown()
    gateway.release_second_resolve.set()
    gateway.release[1].set()  # lets a mutated illicit dispatch fail visibly, not hang
    await chain_task

    assert accepted_entries == []
    assert gateway.user_turns == ["one"]
    assert not any(
        message.role is ConsoleMessageRole.USER and message.content == "two"
        for message in store.messages_for_session(session_id)
    )


@pytest.mark.asyncio
async def test_close_tombstones_before_cancel_and_never_starts_next_prompt():
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)
    chain_task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "two")

    controller.close_session(session_id)
    await asyncio.gather(chain_task, return_exceptions=True)

    assert gateway.user_turns == ["one"]
    assert controller.prompt_queue_registry.snapshot(session_id).total_count == 0


@pytest.mark.asyncio
async def test_paused_queue_gates_unrelated_generation_and_cap_refuses_reacquire(
    monkeypatch,
):
    gateway = SequencedGateway(fail_call=0)
    controller, store, session_id = _arm_controller(gateway)
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    _queue(controller, session_id, "two")
    gateway.release[0].set()
    await task

    before_users = [
        message.id
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.USER
    ]
    refused = await controller.submit_draft("bypass", session_id=session_id)
    assert not refused.accepted
    assert "Queued messages control" in refused.visible_copy
    assert [
        message.id
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.USER
    ] == before_users

    other = controller.new_session(title="Occupies only slot", ephemeral=True)
    controller._set_run_state(
        controller.run_state_for(other.id).__class__(
            ConsoleRunStatus.STREAMING, "Streaming response."
        ),
        session_id=other.id,
    )
    monkeypatch.setattr(
        type(controller), "max_parallel_runs", property(lambda _self: 1)
    )
    reacquire = await controller.resume_prompt_queue(session_id)
    assert not reacquire.applied
    paused = controller.prompt_queue_registry.snapshot(session_id)
    assert paused.mode is PromptQueueMode.PAUSED
    assert paused.reservation is PromptQueueReservation.RELEASED


@pytest.mark.asyncio
async def test_rag_capture_receives_manual_then_queued_origin_for_owner_session():
    gateway = SequencedGateway()
    seen: list[tuple[str, ConsoleSubmissionOrigin, str]] = []

    async def capture(draft, turn_context, origin):
        seen.append((draft, origin, turn_context.session_id))
        return None

    store = ConsoleChatStore()
    session = store.ensure_session(title="RAG owner")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        rag_capture_provider=capture,
    )
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session.id)
    )
    await gateway.started[0].wait()
    _queue(controller, session.id, "two")
    gateway.release[0].set()
    await gateway.started[1].wait()
    gateway.release[1].set()
    await task

    assert seen == [
        ("one", ConsoleSubmissionOrigin.MANUAL, session.id),
        ("two", ConsoleSubmissionOrigin.QUEUED, session.id),
    ]


@pytest.mark.asyncio
async def test_accepted_queued_prompts_use_normal_persistence_exactly_once():
    gateway = SequencedGateway()
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Persistent queue")
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session.id)
    )
    await gateway.started[0].wait()
    _queue(controller, session.id, "two")
    gateway.release[0].set()
    await gateway.started[1].wait()
    gateway.release[1].set()
    await task

    persisted_users = [
        item["content"]
        for item in persistence.created_messages
        if item["sender"] == "user"
    ]
    assert persisted_users == ["one", "two"]


@pytest.mark.asyncio
async def test_two_sessions_keep_independent_chains_and_each_occupies_one_slot():
    gateway = SequencedGateway()
    store = ConsoleChatStore()
    first = store.ensure_session(title="First")
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    second = controller.new_session(title="Second", ephemeral=True)

    first_task = asyncio.create_task(
        controller.run_prompt_chain("a1", session_id=first.id)
    )
    await gateway.started[0].wait()
    second_task = asyncio.create_task(
        controller.run_prompt_chain("b1", session_id=second.id)
    )
    await gateway.started[1].wait()
    _queue(controller, first.id, "a2")
    _queue(controller, second.id, "b2")
    assert controller.in_flight_run_count() == 2

    for release in gateway.release:
        release.set()
    await asyncio.gather(first_task, second_task)

    assert [
        message.content
        for message in store.messages_for_session(first.id)
        if message.role is ConsoleMessageRole.USER
    ] == ["a1", "a2"]
    assert [
        message.content
        for message in store.messages_for_session(second.id)
        if message.role is ConsoleMessageRole.USER
    ] == ["b1", "b2"]
    assert controller.in_flight_run_count() == 0


@pytest.mark.asyncio
async def test_approval_wait_uses_same_activity_projection_and_keeps_queue_editable():
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    entry_id = _queue(controller, session_id, "two original")
    controller.add_pending_round(session_id, "approval-round")

    activity = controller.activity_for(session_id)
    assert activity.needs_approval
    assert activity.occupies_slot
    assert controller.run_marker_for(session_id) is ConsoleRunMarker.NEEDS_APPROVAL
    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    edited = controller.prompt_queue_registry.edit(
        session_id,
        entry_id=entry_id,
        text="two edited",
        expected_revision=snapshot.revision,
    )
    assert edited.applied

    controller.discard_pending_round(session_id, "approval-round")
    gateway.release[0].set()
    await gateway.started[1].wait()
    gateway.release[1].set()
    await task
    assert gateway.user_turns == ["one", "two edited"]


@pytest.mark.asyncio
async def test_rider_added_after_admission_returns_claim_without_consuming_it():
    gateway = SequencedGateway()
    rider_present = False
    store = ConsoleChatStore()
    session = store.ensure_session(title="Rider owner")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        queued_staged_rider_provider=lambda _session_id: rider_present,
    )
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session.id)
    )
    await gateway.started[0].wait()
    queued_id = _queue(controller, session.id, "two")
    rider_present = True
    gateway.release[0].set()
    await task

    snapshot = controller.prompt_queue_registry.snapshot(session.id)
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.pause_reason is PromptQueuePauseReason.DISPATCH_REFUSED
    assert [entry.entry_id for entry in snapshot.entries] == [queued_id]
    assert gateway.user_turns == ["one"]


def test_queue_generation_authorization_cannot_be_constructed_externally():
    with pytest.raises(PermissionError):
        QueueGenerationAuthorization(object(), "session", _key=object())
