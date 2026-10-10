"""Pressed input custody relaxes only draft identity, including queue rerouting."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from Tests.Chat import test_console_received_intent_inputs as input_controls
from Tests.Chat import test_console_received_intent_transitions as controls
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Chat.console_prompt_queue import (
    PromptQueueReservation,
    QueueMutationStatus,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
catalog_store = controls.catalog_store
local_root = controls.local_root
mcp_sources = controls.mcp_sources
snapshot_case = controls.snapshot_case
runtime_case = controls.runtime_case
configuration_workspace = controls.configuration_workspace
received_workspace = controls.received_workspace


def _pressed_intent(case, *, queue_revision=None):
    intent = controls._stock_intent(case, queue_revision=queue_revision)
    return replace(intent, _pressed_inputs=intent.inputs)


@pytest.mark.parametrize("change", ["attachment", "settings", "source"])
async def test_pressed_capture_still_refuses_non_draft_change(
    received_workspace, monkeypatch, change
):
    case = received_workspace
    runtime, store, session, controller = (
        case.runtime,
        case.store,
        case.session,
        case.controller,
    )
    store.set_session_draft(session.id, "pressed original draft")
    original_capture = controller.capture_turn_configuration_snapshot
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_capture(*args, **kwargs):
        snapshot = await original_capture(*args, **kwargs)
        entered.set()
        await release.wait()
        return snapshot

    monkeypatch.setattr(controller, "capture_turn_configuration_snapshot", held_capture)
    intent = _pressed_intent(case)
    assert intent._pressed_inputs is intent.inputs
    runtime.accept_received_intent(intent)
    record = runtime._turn_custody[intent.turn_id]
    task = record.task
    assert task is not None
    try:
        await controls._wait_for_seam(record, entered)
        assert record.request is None
        assert record.received_claim is not None
        store.set_session_draft(session.id, "newer draft", authored_token=(21, 2))
        assert not store.session_inputs_are_current(intent.inputs)
        assert store.session_inputs_are_current(intent.inputs, include_draft=False)

        if change == "attachment":
            store.add_pending_attachment(
                session.id, input_controls._attachment("later")
            )
        elif change == "settings":
            store.replace_session_settings(
                session.id,
                ConsoleSessionSettings(provider="llama_cpp", model="changed-model"),
            )
        else:
            # This leaves session inputs current: the separate source fence must
            # reject the replacement application configuration owner itself.
            monkeypatch.setattr(case.app, "app_config", dict(case.app.app_config))
        assert store.session_inputs_are_current(intent.inputs, include_draft=False) is (
            change == "source"
        )
        release.set()
        result = await controls._outcome(record)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert isinstance(result, RecoveryRequired)
    assert "console_snapshot_owner_changed" in str(result)
    assert record.request is None
    assert not store.messages_for_session(session.id)
    assert store.session_draft(session.id) == "newer draft"
    assert store.received_turn_for_session(session.id) is None
    assert intent.turn_id not in runtime._turn_custody
    if change == "attachment":
        assert (
            store.session_input_snapshot(session.id).attachments[0].attachment_id
            == "later"
        )


@pytest.mark.parametrize("outcome", ["queued", "rerouted"])
async def test_pressed_queue_retains_newer_draft_through_original_admission(
    received_workspace, monkeypatch, outcome
):
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case = received_workspace
    runtime, store, session, controller = (
        case.runtime,
        case.store,
        case.session,
        case.controller,
    )
    draft = "pressed queue draft"
    store.set_session_draft(session.id, draft)
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="llama_cpp", model="test-model")
    )
    controller.prompt_history = PromptHistory()
    gateway = controls.SequencedGateway()
    gateway.release[0].set()
    controller.provider_gateway = gateway
    store.replace_session_trace_privacy_override(
        session.id,
        capture_enabled=False,
        pii_redaction_enabled=None,
        expected_policy_revision=store.capture_policy_state(session.id).policy_revision,
    )
    registry = controller.prompt_queue_registry
    begun = registry.begin_chain(
        session.id,
        context_epoch=store.conversation_context_epoch(session.id),
        expected_revision=registry.snapshot(session.id).revision,
    )
    assert begun.applied
    entered, release = asyncio.Event(), asyncio.Event()
    original_admission = controller.hook_admission_reason

    async def held_admission():
        reason = await original_admission()
        assert reason is None
        entered.set()
        await release.wait()
        return reason

    monkeypatch.setattr(controller, "hook_admission_reason", held_admission)
    intent = _pressed_intent(case, queue_revision=begun.snapshot.revision)
    with controls._observe_original_promotion(store) as promoted:
        runtime.accept_received_intent(intent)
        record = runtime._turn_custody[intent.turn_id]
        task = record.task
        assert task is not None
        try:
            await controls._wait_for_seam(record, entered)
            assert record.request is None
            assert record.received_claim is None
            store.set_session_draft(
                session.id, "newer queue draft", authored_token=(22, 2)
            )
            assert not store.session_inputs_are_current(intent.inputs)
            assert store.session_inputs_are_current(intent.inputs, include_draft=False)
            if outcome == "rerouted":
                # The real empty-chain transition makes the held original
                # queue admission return REROUTE_NORMAL_SEND at its old revision.
                finished = registry.finalize_empty_chain(
                    session.id, expected_revision=begun.snapshot.revision
                )
                assert finished.applied
                assert finished.snapshot.reservation is PromptQueueReservation.RELEASED
                assert finished.snapshot.revision > intent.queue_revision
            release.set()
            result = await controls._outcome(record)
        finally:
            release.set()
            for event in gateway.release:
                event.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert store.session_draft(session.id) == "newer queue draft"
    assert store.received_turn_for_session(session.id) is None
    assert intent.turn_id not in runtime._turn_custody
    if outcome == "queued":
        assert result.status is QueueMutationStatus.APPLIED
        assert result.snapshot.total_count == 1
        text = registry.read_waiting_text(
            session.id,
            entry_id=result.entry_id,
            expected_revision=result.snapshot.revision,
        )
        assert text.status is QueueMutationStatus.APPLIED
        assert text.text == draft
        assert record.request is None
        assert not promoted
        assert gateway.user_turns == []
        assert not store.messages_for_session(session.id)
    else:
        assert result.accepted
        assert gateway.user_turns == [draft]
        assert record.request is None
        assert registry.snapshot(session.id).total_count == 0
        assert [revision for _identity, revision in promoted] == [
            intent.inputs.draft_revision
        ], "rerouting must promote the original pressed input revision"
