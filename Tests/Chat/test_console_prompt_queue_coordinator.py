"""Joined controller/coordinator tests for sequential Console prompt queues."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.console_close_helpers import close_controller_session
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_activity_receipts import (
    ConsoleActivityReceiptService,
)
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
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from Tests.console_provider_doubles import provider_resolution
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


pytestmark = pytest.mark.bootstrap_profile


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
            return provider_resolution(
                ready=False, visible_copy="Provider blocked: unavailable"
            )
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


async def _queue(controller: ConsoleChatController, session_id: str, text: str) -> str:
    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    result = await controller.queue_prompt(
        session_id,
        text=text,
        expected_revision=snapshot.revision,
    )
    assert result.applied
    assert result.entry_id is not None
    return result.entry_id


async def test_controller_refuses_unsafe_queue_text_before_admission() -> None:
    controller, store, session_id = _arm_controller(SequencedGateway())
    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    snapshot = controller.prompt_queue_registry.begin_chain(
        session_id,
        context_epoch=store.conversation_context_epoch(session_id),
        expected_revision=snapshot.revision,
    ).snapshot

    result = await controller.queue_prompt(
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
    await _queue(controller, session_id, "private first follow-up")
    await _queue(controller, session_id, "private second follow-up")

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
    await _queue(controller, session_id, "wait until recovery")
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
    second_id = await _queue(controller, session_id, "two")
    third_id = await _queue(controller, session_id, "three")

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
async def test_queued_drain_enters_runtime_custody_before_controller_with_frozen_config():
    gateway = SequencedGateway()
    store = ConsoleChatStore()
    session = store.ensure_session(title="Queue custody owner")
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    accepted_requests = []
    original_accept_turn = runtime.accept_turn

    def accept_turn_once(request, **kwargs):
        accepted_requests.append((request, kwargs))
        return original_accept_turn(request, **kwargs)

    runtime.accept_turn = accept_turn_once  # type: ignore[method-assign]
    controller.temperature = 0.15

    chain_task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session.id)
    )
    await gateway.started[0].wait()
    queued_id = await _queue(controller, session.id, "two frozen")
    controller.temperature = 1.75

    queued_submit_started = asyncio.Event()
    release_queued_submit = asyncio.Event()
    observations: list[tuple[str, float | None, bool]] = []
    original_submit = controller.submit_draft

    async def submit_with_barrier(draft: str, **kwargs):
        if kwargs.get("origin") is ConsoleSubmissionOrigin.QUEUED:
            record = next(iter(runtime._turn_custody.values()), None)
            configuration = kwargs.get("configuration")
            observations.append(
                (
                    draft,
                    configuration.provider_payload_settings.get("temperature"),
                    bool(
                        record is not None
                        and record.request is not None
                        and record.request.draft == draft
                    ),
                )
            )
            queued_submit_started.set()
            await release_queued_submit.wait()
        return await original_submit(draft, **kwargs)

    controller.submit_draft = submit_with_barrier  # type: ignore[method-assign]
    gateway.release[0].set()
    await queued_submit_started.wait()
    release_queued_submit.set()
    await gateway.started[1].wait()
    gateway.release[1].set()
    await chain_task

    assert observations == [("two frozen", 0.15, True)]
    assert len(accepted_requests) == 1
    accepted_request, accepted_kwargs = accepted_requests[0]
    assert accepted_request.draft == "two frozen"
    assert accepted_request.attachment_ids == ()
    assert accepted_request.staged_evidence_launch is None
    assert accepted_kwargs["origin"] is ConsoleSubmissionOrigin.QUEUED
    assert accepted_kwargs["queue_entry_id"] == queued_id
    assert controller.prompt_queue_registry.snapshot(session.id).total_count == 0
    assert not runtime._turn_custody


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
    await _queue(controller, session_id, "two")
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
    await _queue(controller, session_id, "two")
    third_id = await _queue(controller, session_id, "three")

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
    await _queue(controller, session_id, "two")
    third_id = await _queue(controller, session_id, "three")
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
    queued_id = await _queue(controller, session_id, "two")

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


def test_agent_wake_acceptance_does_not_require_a_prompt_queue_chain():
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)

    controller.prompt_queue_coordinator.turn_accepted(
        session_id,
        origin=ConsoleSubmissionOrigin.AGENT_WAKE,
        context_epoch=0,
    )

    assert controller.prompt_queue_registry.snapshot(session_id).entries == ()


class HoldValidationGateway(SequencedGateway):
    """Holds ``resolve_for_send`` (the run's VALIDATING window) on demand."""

    def __init__(self) -> None:
        super().__init__()
        self.hold = False
        self.validation_started = asyncio.Event()
        self.validation_release = asyncio.Event()

    async def resolve_for_send(self, selection):
        if self.hold:
            self.validation_started.set()
            await self.validation_release.wait()
        return await super().resolve_for_send(selection)


@pytest.mark.asyncio
async def test_only_a_prompt_chain_turn_is_preparing_before_acceptance():
    """TASK-33620.4 review: ``preparing_before_acceptance`` means "a queue will
    open once this turn is accepted". A manual prompt-chain turn in validation
    is exactly that; a chainless regenerate in the SAME status is never
    queue-accepted, so it occupies the slot without promising a queue."""
    gateway = HoldValidationGateway()
    controller, store, session_id = _arm_controller(gateway)
    coordinator = controller.prompt_queue_coordinator

    # Positive control: a manual prompt-chain turn held in validation.
    gateway.hold = True
    chain = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await asyncio.wait_for(gateway.validation_started.wait(), timeout=5)
    assert controller.run_state_for(session_id).status is ConsoleRunStatus.VALIDATING
    activity = coordinator.activity(session_id)
    assert activity.occupies_slot and not activity.accepted_live_turn
    assert activity.preparing_before_acceptance
    gateway.hold = False
    gateway.validation_release.set()
    gateway.release[0].set()
    await asyncio.wait_for(chain, timeout=5)
    assistant_id = store.active_leaf(session_id)
    assert assistant_id is not None

    # The chainless regenerate, held in the same VALIDATING window.
    gateway.validation_started = asyncio.Event()
    gateway.validation_release = asyncio.Event()
    gateway.hold = True
    regenerate = asyncio.create_task(controller.regenerate_message(assistant_id))
    await asyncio.wait_for(gateway.validation_started.wait(), timeout=5)
    assert controller.run_state_for(session_id).status is ConsoleRunStatus.VALIDATING
    activity = coordinator.activity(session_id)
    assert activity.occupies_slot and not activity.accepted_live_turn
    assert not activity.preparing_before_acceptance
    gateway.hold = False
    gateway.validation_release.set()
    gateway.release[1].set()
    result = await asyncio.wait_for(regenerate, timeout=5)
    assert result.accepted
    assert gateway.user_turns == ["one", "one"]


@pytest.mark.asyncio
async def test_stop_pauses_immediately_and_resume_next_dispatches_once():
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    await _queue(controller, session_id, "two")

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
    await _queue(controller, session_id, "two")
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
    await _queue(controller, session_id, "owner follow-up")
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
    controller, store, session_id = _arm_controller(gateway)
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    history = RecordingPromptHistory()
    controller.prompt_history = history
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    queued_id = await _queue(controller, session_id, "two")
    gateway.release[0].set()
    await task

    snapshot = controller.prompt_queue_registry.snapshot(session_id)
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.pause_reason is PromptQueuePauseReason.DISPATCH_REFUSED
    assert [entry.entry_id for entry in snapshot.entries] == [queued_id]
    assert history.items == ["one"]
    assert gateway.user_turns == ["one"]
    assert runtime.recoveries_for_session(session_id) == ()
    assert not runtime._turn_custody


@pytest.mark.asyncio
async def test_shutdown_tombstones_before_cancel_and_never_starts_next_prompt():
    gateway = SequencedGateway()
    controller, _store, session_id = _arm_controller(gateway)
    chain_task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    await gateway.started[0].wait()
    await _queue(controller, session_id, "two")

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
    await _queue(controller, session_id, "two")
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
    await _queue(controller, session_id, "two")

    close_controller_session(controller, session_id)
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
    await _queue(controller, session_id, "two")
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
    await _queue(controller, session.id, "two")
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
    await _queue(controller, session.id, "two")
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
    await _queue(controller, first.id, "a2")
    await _queue(controller, second.id, "b2")
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
    entry_id = await _queue(controller, session_id, "two original")
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
    queued_id = await _queue(controller, session.id, "two")
    later_attachment = PendingAttachment(
        "/later.png",
        "later.png",
        "image",
        "attachment",
        data=b"later",
    )
    assert store.add_pending_attachment(session.id, later_attachment)
    rider_present = True
    gateway.release[0].set()
    await task

    snapshot = controller.prompt_queue_registry.snapshot(session.id)
    assert snapshot.mode is PromptQueueMode.PAUSED
    assert snapshot.pause_reason is PromptQueuePauseReason.DISPATCH_REFUSED
    assert [entry.entry_id for entry in snapshot.entries] == [queued_id]
    assert gateway.user_turns == ["one"]
    assert store.pending_attachments(session.id) == [later_attachment]


def test_queue_generation_authorization_cannot_be_constructed_externally():
    with pytest.raises(PermissionError):
        QueueGenerationAuthorization(object(), "session", _key=object())


@pytest.mark.asyncio
async def test_multi_entry_queue_chain_publishes_one_final_durable_outcome(tmp_path):
    gateway = SequencedGateway()
    store = ConsoleChatStore()
    queued_session = store.ensure_session(title="Queued")
    active_session = store.create_session(title="Active", ephemeral=True)
    store.switch_session(queued_session.id)
    service = ConsoleActivityReceiptService(
        AgentRunsDB(tmp_path / "queue-activity.db"), None
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        activity_receipts=service,
    )

    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=queued_session.id)
    )
    await gateway.started[0].wait()
    await _queue(controller, queued_session.id, "two")
    store.switch_session(active_session.id)
    gateway.release[0].set()
    await gateway.started[1].wait()
    gateway.release[1].set()
    await task

    receipts = service.unseen_snapshot()
    assert len(receipts) == 1
    assert receipts[0].logical_outcome_id.startswith("queue-chain:")
    assert receipts[0].status == "done"
    assert receipts[0].session_id == queued_session.id


# ---------------------------------------------------------------------------
# TASK-33621.19 (review findings GAP1-04 / GAP5-09): a persisted Console chat
# takes the durable path, where every queued turn is acknowledged by the
# post-commit ``queue_acknowledgement`` effect. That acknowledgement used to
# pause the queue as FAILED whenever later prompts were still waiting -- even
# while the live chain that claimed the entry was draining -- so a queue of
# three halted after one successful turn under a false "Turn failed". The
# joined tests above all ran db-less (ephemeral) sessions, which never reach
# the durable acknowledgement; these drive a real SQLite-backed store.
# ---------------------------------------------------------------------------


def _durable_controller(tmp_path, gateway: SequencedGateway):
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "durable-queue.sqlite", client_id="task-33621-19")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(session_id="durable-queue", title="Durable queue")
    assert session.ephemeral is False
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
    )
    return db, store, controller, session.id


def _observe_durable_acknowledgements(
    monkeypatch, controller
) -> list[tuple[str, bool]]:
    """Record each REAL durable acknowledgement and whether a live chain owned it."""

    coordinator = controller.prompt_queue_coordinator
    original = coordinator.acknowledge_durable_acceptance
    observed: list[tuple[str, bool]] = []

    def observe(session_id: str, **kwargs):
        chain = coordinator._chains.get(session_id)
        observed.append(
            (
                kwargs["entry_id"],
                chain is not None and chain.current_entry_id == kwargs["entry_id"],
            )
        )
        return original(session_id, **kwargs)

    monkeypatch.setattr(coordinator, "acknowledge_durable_acceptance", observe)
    return observed


async def _release_all(gateway: SequencedGateway, task: asyncio.Task) -> None:
    for release in gateway.release:
        release.set()
    if not task.done():
        task.cancel()
    try:
        await task
    except BaseException:  # noqa: BLE001 - teardown of a failed assertion path
        pass


@pytest.mark.asyncio
async def test_durable_queue_drains_three_entries_through_live_postcommit_ack(
    tmp_path, monkeypatch
):
    """AC#1/#4/#6: every queued entry is durably acknowledged and all drain."""

    gateway = SequencedGateway()
    db, store, controller, session_id = _durable_controller(tmp_path, gateway)
    registry = controller.prompt_queue_registry
    coordinator = controller.prompt_queue_coordinator
    acknowledgements = _observe_durable_acknowledgements(monkeypatch, controller)
    terminals: list[ConsoleRunStatus] = []
    publish_terminal = coordinator.on_chain_terminal

    def observe_terminal(sid, status, outcome_id):
        terminals.append(status)
        publish_terminal(sid, status, outcome_id)

    coordinator.on_chain_terminal = observe_terminal

    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    try:
        await asyncio.wait_for(gateway.started[0].wait(), timeout=10)
        queued_ids = [
            await _queue(controller, session_id, text)
            for text in ("two", "three", "four")
        ]
        # The owner is viewed elsewhere, so a chain terminal lands on the
        # rail as an unvisited outcome marker -- the surface that showed a
        # failure glyph after every turn had succeeded.
        viewed = store.create_session(title="Viewed elsewhere", ephemeral=True)
        store.switch_session(viewed.id)

        for index in (1, 2, 3):
            gateway.release[index - 1].set()
            await asyncio.wait_for(gateway.started[index].wait(), timeout=10)
            # Provider entry runs AFTER the post-commit acknowledgement of
            # this queued entry, so the queue state here is the one the
            # shelf painted while the reply was generating.
            snapshot = registry.snapshot(session_id)
            assert snapshot.mode is PromptQueueMode.DRAINING, (
                index,
                snapshot.mode,
                snapshot.pause_reason,
            )
            assert snapshot.pause_reason is None
            assert snapshot.reservation is PromptQueueReservation.HELD
            assert snapshot.claimed_count == 0
            assert snapshot.waiting_count == 3 - index
            assert controller.activity_for(session_id).queue_paused is False
        gateway.release[3].set()
        result = await asyncio.wait_for(task, timeout=10)
    finally:
        await _release_all(gateway, task)

    assert result.terminal_status is ConsoleRunStatus.COMPLETED
    assert gateway.user_turns == ["one", "two", "three", "four"]
    assert acknowledgements == [(entry_id, True) for entry_id in queued_ids]
    final = registry.snapshot(session_id)
    assert final.total_count == 0
    assert final.mode is PromptQueueMode.DRAINING
    assert final.pause_reason is None
    assert final.reservation is PromptQueueReservation.RELEASED
    assert terminals == [ConsoleRunStatus.COMPLETED]
    assert controller.run_marker_for(session_id) is ConsoleRunMarker.FINISHED_OK
    attention = {
        fact.kind for fact in controller.conversation_attention_for(session_id)
    }
    assert "failed" not in attention
    assert "paused" not in attention
    messages = store.messages_for_session(session_id)
    assert [m.content for m in messages if m.role is ConsoleMessageRole.USER] == [
        "one",
        "two",
        "three",
        "four",
    ]
    assert [m.status for m in messages if m.role is ConsoleMessageRole.ASSISTANT] == [
        "complete"
    ] * 4
    persisted_users = (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM messages WHERE sender = 'user'")
        .fetchone()[0]
    )
    assert persisted_users == 4


def test_live_chain_durable_ack_settles_its_claim_without_pausing_later_work():
    """The exact live owner keeps draining; the detached case still pauses."""

    from tldw_chatbook.Chat.console_prompt_queue import ConsolePromptQueueRegistry
    from tldw_chatbook.Chat.console_prompt_queue_coordinator import (
        ConsolePromptQueueCoordinator,
        _PromptChain,
    )

    registry = ConsolePromptQueueRegistry(
        id_factory=iter(("accepted-entry", "later-entry", "last-entry")).__next__
    )
    coordinator = ConsolePromptQueueCoordinator(
        registry=registry,
        context_epoch=lambda _session_id: 0,
        run_status=lambda _session_id: ConsoleRunStatus.STREAMING,
        submit_queued=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
    )
    snapshot = registry.begin_chain(
        "session-1", context_epoch=0, expected_revision=0
    ).snapshot
    for text in ("accepted body", "later body", "last body"):
        snapshot = registry.admit(
            "session-1", text=text, expected_revision=snapshot.revision
        ).snapshot
    claimed = registry.claim_next("session-1", expected_revision=snapshot.revision)
    assert claimed.claim is not None
    assert registry.bind_claimed_preparation(
        "session-1", entry_id="accepted-entry", preparation_id="preparation-1"
    ).applied
    chain = _PromptChain(current_entry_id="accepted-entry")
    coordinator._chains["session-1"] = chain

    assert coordinator.acknowledge_durable_acceptance(
        "session-1",
        entry_id="accepted-entry",
        preparation_id="preparation-1",
        context_epoch=0,
    )

    settled = registry.snapshot("session-1")
    assert settled.claimed_count == 0
    assert [entry.entry_id for entry in settled.entries] == [
        "later-entry",
        "last-entry",
    ]
    assert settled.mode is PromptQueueMode.DRAINING
    assert settled.pause_reason is None
    assert settled.reservation is PromptQueueReservation.HELD
    # The live owner keeps its exact entry (as the ephemeral turn_accepted
    # path does) so its own post-turn step -- the drain loop, or
    # finish_recovered_entry for a reclaimed preparation -- still owns the
    # advance; clearing it stranded reclaimed durable chains.
    assert chain.current_entry_id == "accepted-entry"
    assert chain.accepted_live_turn is True
    # Re-delivery of the same committed acknowledgement stays idempotent.
    assert coordinator.acknowledge_durable_acceptance(
        "session-1",
        entry_id="accepted-entry",
        preparation_id="preparation-1",
        context_epoch=0,
    )
    assert registry.snapshot("session-1") == settled
    # And the live chain claims the next entry normally.
    assert (
        registry.claim_next("session-1", expected_revision=settled.revision).entry_id
        == "later-entry"
    )


def _queue_ui(controller, notices: list[tuple[str, str]]):
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptQueueUIController,
    )

    async def sync_ui() -> None:
        return None

    async def append_system(_text: str) -> None:
        return None

    return ConsolePromptQueueUIController(
        chat_controller_accessor=lambda: controller,
        capture_configuration=lambda _session_id: None,
        ensure_active_session=lambda: None,
        blocked_reason_accessor=lambda: "",
        setup_blocked_reason_accessor=lambda: "",
        append_system_message=append_system,
        notify=lambda text, severity: notices.append((text, severity)),
        focus_composer=lambda: None,
        note_follow_intent=lambda: None,
        launch_chain=lambda _draft, _session_id: "unused",
        commit_captured_draft=lambda _session_id, _stash: None,
        commit_queued_draft=lambda _session_id, _stash: None,
        turn_recovery_ids=lambda _session_id: (),
        restore_turn_recovery=lambda _turn_id: None,
        discard_turn_recovery=lambda _turn_id: False,
        load_recovered_turn=lambda _session_id: None,
        edit_refusal=lambda _text: "",
        sync_ui=sync_ui,
    )


@pytest.mark.asyncio
async def test_real_failed_queued_turn_is_named_and_retry_reruns_it_then_drains(
    tmp_path,
):
    """AC#2: 'Turn failed' names the failed queued turn; Retry re-runs it."""

    gateway = SequencedGateway(fail_call=1)
    _db, store, controller, session_id = _durable_controller(tmp_path, gateway)
    registry = controller.prompt_queue_registry
    notices: list[tuple[str, str]] = []
    ui = _queue_ui(controller, notices)

    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    try:
        await asyncio.wait_for(gateway.started[0].wait(), timeout=10)
        for text in ("two", "three", "four"):
            await _queue(controller, session_id, text)
        gateway.release[0].set()
        await asyncio.wait_for(gateway.started[1].wait(), timeout=10)
        # While the first queued turn is generating nothing has failed yet.
        assert registry.snapshot(session_id).mode is PromptQueueMode.DRAINING
        gateway.release[1].set()
        await asyncio.wait_for(task, timeout=10)
    finally:
        await _release_all(gateway, task)

    paused = registry.snapshot(session_id)
    assert paused.mode is PromptQueueMode.PAUSED
    assert paused.pause_reason is PromptQueuePauseReason.FAILED
    failed = next(
        m
        for m in reversed(store.messages_for_session(session_id))
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert failed.status == "failed"
    presentation = ui.presentation_for(session_id)
    assert presentation.state_label == 'Turn failed: "two"'
    assert presentation.pause_label == "Retry"
    assert presentation.next_preview == "three"

    gateway2 = SequencedGateway()
    controller.provider_gateway = gateway2
    retry = asyncio.create_task(
        ui.handle_primary_intent(
            session_id,
            action=presentation.primary_action,
            expected_revision=presentation.revision,
        )
    )
    try:
        for index in range(3):
            await asyncio.wait_for(gateway2.started[index].wait(), timeout=10)
            gateway2.release[index].set()
        await asyncio.wait_for(retry, timeout=10)
    finally:
        await _release_all(gateway2, retry)

    assert gateway2.user_turns == ["two", "three", "four"]
    assert notices == []
    assert store.get_message(failed.id).status == "complete"
    final = registry.snapshot(session_id)
    assert final.total_count == 0
    assert final.mode is PromptQueueMode.DRAINING
    assert ui.presentation_for(session_id).shelf_visible is False


@pytest.mark.asyncio
async def test_real_paused_queue_without_failed_turn_offers_resume_that_drains(
    tmp_path,
):
    """AC#3: a FAILED pause with no failed message offers a working Resume."""

    gateway = SequencedGateway()
    _db, store, controller, session_id = _durable_controller(tmp_path, gateway)
    registry = controller.prompt_queue_registry
    notices: list[tuple[str, str]] = []
    ui = _queue_ui(controller, notices)

    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    try:
        await asyncio.wait_for(gateway.started[0].wait(), timeout=10)
        for text in ("two", "three"):
            await _queue(controller, session_id, text)
        snapshot = registry.snapshot(session_id)
        assert controller.pause_prompt_queue_after_turn(
            session_id, expected_revision=snapshot.revision
        ).applied
        gateway.release[0].set()
        await asyncio.wait_for(task, timeout=10)
    finally:
        await _release_all(gateway, task)
    # The shape a detached durable acknowledgement leaves behind: paused as
    # FAILED although every turn in the transcript succeeded.
    snapshot = registry.snapshot(session_id)
    assert registry.pause(
        session_id,
        reason=PromptQueuePauseReason.FAILED,
        expected_revision=snapshot.revision,
    ).applied
    assert all(
        m.status == "complete"
        for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )

    presentation = ui.presentation_for(session_id)
    assert presentation.state_label == "Paused"
    assert presentation.pause_label == "Resume"

    gateway2 = SequencedGateway()
    controller.provider_gateway = gateway2
    resume = asyncio.create_task(
        ui.handle_primary_intent(
            session_id,
            action=presentation.primary_action,
            expected_revision=presentation.revision,
        )
    )
    try:
        for index in range(2):
            await asyncio.wait_for(gateway2.started[index].wait(), timeout=10)
            gateway2.release[index].set()
        await asyncio.wait_for(resume, timeout=10)
    finally:
        await _release_all(gateway2, resume)

    assert gateway2.user_turns == ["two", "three"]
    assert notices == []
    assert registry.snapshot(session_id).total_count == 0


# ---------------------------------------------------------------------------
# TASK-33621.19 review (PR #2943): the shelf's "Paused | Resume" crashed in
# the two states where this PR makes it the primary action. A failed queue
# retry-stopped regeneration and a deleted failed turn both advance the
# conversation context epoch past the queue's baseline. resume() then
# re-pauses the queue as CONTEXT_CHANGED -- a real change, so the registry
# reports it APPLIED -- and resume_and_drain read that as "resumed" and
# drained a chain resume() never created: KeyError in _drain_waiting. The
# shelf runs that press in an app worker, so the error would exit the app.
# These drive the real controller, coordinator and store: no stubbed resume.
# ---------------------------------------------------------------------------


class FailBeforeReplyGateway(SequencedGateway):
    """A provider that refuses call ``fail_call`` before streaming anything."""

    async def stream_chat(self, resolution, messages, **kwargs):
        call = len(self.user_turns)
        self.user_turns.append(
            next(
                message["content"]
                for message in reversed(messages)
                if message.get("role") == "user"
            )
        )
        self.started[call].set()
        await self.release[call].wait()
        if call == self.fail_call:
            raise RuntimeError("planned provider refusal")
        yield f"reply-{call + 1}"


async def _press_resume_then_use_current_context(
    controller, ui, notices, gateway, session_id: str, *, next_call: int
) -> None:
    """Press the shelf's Resume, then its follow-up Review, both for real."""

    registry = controller.prompt_queue_registry
    presentation = ui.presentation_for(session_id)
    assert (presentation.state_label, presentation.pause_label) == (
        "Paused",
        "Resume",
    )
    paused = registry.snapshot(session_id)
    assert paused.pause_reason is PromptQueuePauseReason.FAILED
    assert controller.store.conversation_context_epoch(session_id) != (
        paused.expected_context_epoch
    )
    turns_before = list(gateway.user_turns)

    # The press must not raise (it did: KeyError in _drain_waiting).
    await asyncio.wait_for(
        ui.handle_primary_intent(
            session_id,
            action=presentation.primary_action,
            expected_revision=presentation.revision,
        ),
        timeout=10,
    )

    # Nothing was dispatched under the changed context; the queue now asks
    # for an explicit review instead of staying on a dead Resume.
    assert gateway.user_turns == turns_before
    repaused = registry.snapshot(session_id)
    assert repaused.mode is PromptQueueMode.PAUSED
    assert repaused.pause_reason is PromptQueuePauseReason.CONTEXT_CHANGED
    assert [entry.preview for entry in repaused.entries] == ["two"]
    review = ui.presentation_for(session_id)
    assert (review.state_label, review.pause_label, review.primary_action) == (
        "Context changed",
        "Review",
        "review",
    )

    # And that state's action works: the Manage modal's "Use current
    # context" adopts the reviewed epoch and drains the waiting prompt.
    _baseline, current_epoch = ui.context_review(session_id)
    drain = asyncio.create_task(
        ui.recover(
            session_id,
            action="use-current-context",
            expected_revision=review.revision,
            reviewed_context_epoch=current_epoch,
        )
    )
    try:
        await asyncio.wait_for(gateway.started[next_call].wait(), timeout=10)
        gateway.release[next_call].set()
        result = await asyncio.wait_for(drain, timeout=10)
    finally:
        await _release_all(gateway, drain)
    assert result.applied
    assert gateway.user_turns == [*turns_before, "two"]
    assert registry.snapshot(session_id).total_count == 0
    assert notices == []


@pytest.mark.asyncio
async def test_resume_after_a_failed_stopped_turn_regeneration_does_not_raise():
    """Shape 1: Retry stopped regenerates a stopped queued turn and it fails.

    regenerate_message keeps the failed sibling off-path and restores the
    stopped original as the active leaf, so the shelf has no failed turn to
    name and offers Resume.
    """

    gateway = FailBeforeReplyGateway(fail_call=1)
    controller, store, session_id = _arm_controller(gateway)
    notices: list[tuple[str, str]] = []
    ui = _queue_ui(controller, notices)
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    try:
        await asyncio.wait_for(gateway.started[0].wait(), timeout=10)
        await _queue(controller, session_id, "two")
        assert controller.stop_active_run()
        await asyncio.wait_for(task, timeout=10)
    finally:
        await _release_all(gateway, task)
    stopped = ui.presentation_for(session_id)
    assert stopped.state_label == "Turn stopped"

    regenerate = asyncio.create_task(
        ui.recover(
            session_id,
            action="retry-stopped",
            expected_revision=stopped.revision,
        )
    )
    try:
        await asyncio.wait_for(gateway.started[1].wait(), timeout=10)
        gateway.release[1].set()
        await asyncio.wait_for(regenerate, timeout=10)
    finally:
        await _release_all(gateway, regenerate)
    assert gateway.user_turns == ["one", "one"]
    newest = next(
        m
        for m in store.iter_messages_newest_first(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert newest.status == "stopped"  # the failed sibling is off-path
    assert ui.recovery_turn(session_id, action="retry-failed") is None

    await _press_resume_then_use_current_context(
        controller, ui, notices, gateway, session_id, next_call=2
    )


@pytest.mark.asyncio
async def test_resume_after_deleting_the_failed_queued_turn_does_not_raise():
    """Shape 2: the PR's documented fallback -- delete the failed turn."""

    gateway = SequencedGateway(fail_call=0)
    controller, store, session_id = _arm_controller(gateway)
    notices: list[tuple[str, str]] = []
    ui = _queue_ui(controller, notices)
    task = asyncio.create_task(
        controller.run_prompt_chain("one", session_id=session_id)
    )
    try:
        await asyncio.wait_for(gateway.started[0].wait(), timeout=10)
        await _queue(controller, session_id, "two")
        gateway.release[0].set()
        await asyncio.wait_for(task, timeout=10)
    finally:
        await _release_all(gateway, task)
    assert ui.presentation_for(session_id).state_label == 'Turn failed: "one"'
    failed = next(
        m
        for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT and m.status == "failed"
    )

    store.delete_message(failed.id)

    await _press_resume_then_use_current_context(
        controller, ui, notices, gateway, session_id, next_call=1
    )
