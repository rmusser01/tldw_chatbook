"""Early received input custody without premature clearing or transfer.

These controls verify admission and scheduling. Actual native producer retirement
and natural Preparing/input frames are covered by separate integration controls.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from Tests.Chat.test_console_received_custody import (
    _attachment,
    _retire,
)
from Tests.Chat.test_console_received_custody import (
    _case as complete_case,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_configuration_preparation import (
    ConsoleTurnCaptureSelection,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest_asyncio.fixture
async def intent_case():
    runtime, store, session, request = complete_case()
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    runtime.set_chat_controller(controller)
    store.set_session_draft(session.id, request.draft)
    try:
        yield runtime, store, session, request
    finally:
        await _retire(runtime)


def _intent(case, *, turn_id="received-unconfigured", queue_revision=None):
    from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent

    runtime, store, session, request = case
    launch, revision, _notice = runtime.snapshot_console_staged_evidence()
    return ConsoleReceivedTurnIntent(
        turn_id=turn_id,
        session_id=session.id,
        inputs=store.session_input_snapshot(session.id),
        selection=ConsoleTurnCaptureSelection(
            provider_selection=request.configuration.provider_selection,
            presentation_context=request.configuration.presentation_context,
            rag_defaults={},
            tool_configuration={"agent_runtime_enabled": False},
            skill_workspace_id=None,
            project_bindings_eligible=False,
            agent_runtime_enabled=False,
        ),
        staged_evidence_launch=launch,
        staged_evidence_revision=revision,
        view_attachment_generation=runtime._attached_generation or 0,
        queue_revision=queue_revision,
    )


async def test_normal_receipt_precedes_lazy_driver_without_moving_staged_inputs(
    intent_case,
):
    runtime, store, session, request = intent_case
    attachment = _attachment("received-original-attachment")
    assert store.add_pending_attachment(session.id, attachment)
    intent = _intent(intent_case)
    before_prefill = store.session_one_shot_prefill_snapshot(session.id)

    turn_id = runtime.accept_received_intent(intent)

    claim = store.received_turn_for_session(session.id)
    record = runtime._turn_custody[turn_id]
    assert turn_id == intent.turn_id and claim.request_id == turn_id
    assert record.received_claim is claim and record.store is store
    assert record.request is None
    assert record.received_intent is intent
    assert record.task is not None and not record.task.done()
    assert runtime._chat_controller.activity_for(session.id).preparing_before_acceptance
    assert store.preparation_for_session(session.id) is None
    assert store.session_draft(session.id) == request.draft
    assert store.pending_attachments(session.id) == [attachment]
    assert store.pending_attachments(session.id)[0] is attachment
    assert record.inputs.attachments == ()
    assert store.session_one_shot_prefill_snapshot(session.id) == before_prefill

    # No task step occurs before cancellation. Receipt cancellation must retain
    # all still-live input and retire the exact occupied admission slot.
    record.task.cancel()
    await asyncio.gather(record.task, return_exceptions=True)
    await asyncio.sleep(0)
    assert store.received_turn_for_session(session.id) is None
    assert not runtime.has_custodied_turns(session.id)
    assert store.session_draft(session.id) == request.draft
    assert store.pending_attachments(session.id)[0] is attachment


async def test_duplicate_receipt_preserves_later_draft_and_attachment(intent_case):
    runtime, store, session, _request = intent_case
    original = _attachment("received-first-attachment")
    later = _attachment("received-later-attachment")
    assert store.add_pending_attachment(session.id, original)
    first = _intent(intent_case)
    first_id = runtime.accept_received_intent(first)
    claim = store.received_turn_for_session(session.id)
    first_task = runtime._turn_custody[first_id].task
    assert store.add_pending_attachment(session.id, later)
    store.set_session_draft(session.id, "newer authored draft")
    second = _intent(intent_case, turn_id="received-loser")

    with pytest.raises(RuntimeError):
        runtime.accept_received_intent(second)

    assert tuple(runtime._turn_custody) == (first_id,)
    assert store.received_turn_for_session(session.id) is claim
    assert store.session_draft(session.id) == second.inputs.draft
    assert store.pending_attachments(session.id) == [original, later]
    assert store.pending_attachments(session.id)[0] is original
    assert store.pending_attachments(session.id)[1] is later
    first_task.cancel()
    await asyncio.gather(first_task, return_exceptions=True)
    await asyncio.sleep(0)
    assert store.received_turn_for_session(session.id) is None
    assert store.session_draft(session.id) == second.inputs.draft
    assert store.pending_attachments(session.id) == [original, later]


async def test_received_scheduling_failure_releases_claim_and_keeps_staging(
    intent_case,
    monkeypatch,
):
    runtime, store, session, request = intent_case
    attachment = _attachment("received-scheduling-attachment")
    assert store.add_pending_attachment(session.id, attachment)
    intent = _intent(intent_case)
    before_prefill = store.session_one_shot_prefill_snapshot(session.id)

    def fail_to_schedule(_coroutine):
        raise RuntimeError("received injected scheduling failure")

    monkeypatch.setattr(runtime, "_create_custody_task", fail_to_schedule)
    with pytest.raises(RuntimeError, match="received injected scheduling failure"):
        runtime.accept_received_intent(intent)

    assert store.received_turn_for_session(session.id) is None
    assert not runtime.has_custodied_turns(session.id)
    assert store.session_draft(session.id) == request.draft
    assert store.pending_attachments(session.id) == [attachment]
    assert store.pending_attachments(session.id)[0] is attachment
    assert store.session_one_shot_prefill_snapshot(session.id) == before_prefill


@pytest.mark.parametrize("queued", [False, True], ids=["ordinary", "queued"])
async def test_purge_lease_refuses_received_intent_before_preparation(
    intent_case, monkeypatch, queued
):
    from tldw_chatbook.Chat import console_received_dispatch

    runtime, store, session, request = intent_case
    attachment = _attachment("capture-purge-kept-attachment")
    assert store.add_pending_attachment(session.id, attachment)
    queue = runtime._chat_controller.prompt_queue_registry
    if queued:
        assert queue.begin_chain(
            session.id,
            context_epoch=store.conversation_context_epoch(session.id),
            expected_revision=queue.snapshot(session.id).revision,
        ).applied
    before_queue = queue.snapshot(session.id)
    before_prefill = store.session_one_shot_prefill_snapshot(session.id)
    before_capture = store.capture_revision(session.id)
    intent = _intent(
        intent_case, queue_revision=before_queue.revision if queued else None
    )
    source_calls = []
    original_source = console_received_dispatch.received_preparation_source

    def observe_source(*args, **kwargs):
        source_calls.append(True)
        return original_source(*args, **kwargs)

    monkeypatch.setattr(
        console_received_dispatch, "received_preparation_source", observe_source
    )
    assert store.begin_capture_quiescence(session.id)
    try:
        with pytest.raises(RuntimeError, match="retry"):
            runtime.accept_received_intent(intent)

        assert not source_calls
        assert store.received_turn_for_session(session.id) is None
        assert not runtime.has_custodied_turns(session.id)
        assert store.session_draft(session.id) == request.draft
        assert store.pending_attachments(session.id) == [attachment]
        assert store.pending_attachments(session.id)[0] is attachment
        assert store.session_one_shot_prefill_snapshot(session.id) == before_prefill
        assert queue.snapshot(session.id) == before_queue
        assert store.capture_revision(session.id) == before_capture
    finally:
        store.end_capture_quiescence(session.id)
        record = runtime._turn_custody.get(intent.turn_id)
        if record is not None and record.task is not None:
            task = record.task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
