"""Receipt interleavings at real review, capture, submission and queue owners."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import sys
from types import SimpleNamespace

import pytest
import pytest_asyncio

from Tests.Chat import test_console_configuration_worker_lifetime as worker_controls
from Tests.Chat import test_console_configuration_preparation as selection_controls
from Tests.Chat import test_console_initial_hook_review as review_controls
from Tests.Chat import test_console_received_intent_custody as intake_controls
from Tests.Chat.test_console_prompt_queue_coordinator import SequencedGateway
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
catalog_store = worker_controls.catalog_store
local_root = worker_controls.local_root
mcp_sources = worker_controls.mcp_sources
snapshot_case = worker_controls.snapshot_case
runtime_case = worker_controls.runtime_case
configuration_workspace = worker_controls.configuration_workspace


@pytest_asyncio.fixture
async def received_workspace(configuration_workspace, monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Agents import hook_permissions

    # MCP's fixture installs a fresh real config module. Align this consuming
    # module alias too; the shared snapshot fixture only aligns function aliases.
    monkeypatch.setattr(hook_permissions, "config", config)
    assert hook_permissions.config is config
    yield configuration_workspace


review_runtime = review_controls.review_runtime
hook_file = review_controls.hook_file


async def _outcome(record):
    outcomes = await asyncio.gather(record.task, return_exceptions=True)
    await asyncio.sleep(0)
    return outcomes[0]


async def _wait_for_seam(record, entered):
    task = record.task
    assert task is not None
    assert await review_controls._until(lambda: entered.is_set() or task.done())
    if task.done() and not entered.is_set():
        task.result()
    assert entered.is_set(), "the original requested await seam was not reached"


@contextmanager
def _observe_original_promotion(store):
    from tldw_chatbook.Chat.console_received_turn import (
        ConsoleReceivedTurnAdmissionMixin,
    )

    code = ConsoleReceivedTurnAdmissionMixin.promote_received_turn.__code__
    observations = []

    def returned(current_code, _offset, result):
        if current_code is not code or result is None:
            return
        frame = sys._getframe(1)
        if frame.f_locals.get("self") is store:
            observations.append((result.preparation_id, result.input_draft_revision))

    monitoring = sys.monitoring
    tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, "received-original-promotion")
    try:
        monitoring.register_callback(tool, monitoring.events.PY_RETURN, returned)
        monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
        yield observations
    finally:
        monitoring.set_local_events(tool, code, 0)
        monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
        monitoring.free_tool_id(tool)


def _stock_intent(case, *, turn_id="stock-received-turn", queue_revision=None):
    from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent

    launch, revision, _notice = case.runtime.snapshot_console_staged_evidence()
    return ConsoleReceivedTurnIntent(
        turn_id=turn_id,
        session_id=case.session.id,
        inputs=case.store.session_input_snapshot(case.session.id),
        selection=selection_controls._selection(case),
        staged_evidence_launch=launch,
        staged_evidence_revision=revision,
        view_attachment_generation=case.runtime._attached_generation or 0,
        queue_revision=queue_revision,
    )


async def test_draft_changed_during_real_initial_review_refuses_before_capture(
    review_runtime, monkeypatch
):
    from tldw_chatbook.Chat.console_hook_review import HookReviewResult

    state = review_runtime
    state.store.set_session_draft(state.session.id, "original reviewed draft")
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id=state.session.id,
        provider_selection=ConsoleProviderSelection(
            provider="llama_cpp", explicit_model="test-model"
        ),
    )
    case = (
        state.runtime,
        state.store,
        state.session,
        SimpleNamespace(configuration=configuration),
    )
    intent = intake_controls._intent(case, turn_id="receipt-real-review")
    captured = []
    original_capture = state.controller.capture_turn_configuration_snapshot

    async def observed_capture(*args, **kwargs):
        captured.append(args)
        return await original_capture(*args, **kwargs)

    monkeypatch.setattr(
        state.controller, "capture_turn_configuration_snapshot", observed_capture
    )
    state.runtime.accept_received_intent(intent)
    record = state.runtime._turn_custody[intent.turn_id]
    claim = record.received_claim
    assert claim is not None
    assert await review_controls._until(
        lambda: intent.turn_id in state.host.registries.get("hook_review", {})
    )
    _view, _attachment, projection = review_controls._present(
        state, intent.turn_id, claim.generation
    )
    state.store.set_session_draft(
        state.session.id, "new authored draft", authored_token=(9, 2)
    )
    approved = await state.runtime.apply_hook_review_action(
        intent.turn_id,
        claim.generation,
        "approve",
        projection.snapshot,
        review_controls._keys(projection.snapshot),
        presentation_token=projection.presentation_token,
    )
    assert approved.ready
    assert state.runtime.resolve_initial_hook_review(
        intent.turn_id,
        claim.generation,
        HookReviewResult("ready", approved),
        presentation_token=projection.presentation_token,
    )
    result = await _outcome(record)

    assert isinstance(result, RecoveryRequired)
    assert not captured
    assert not state.store.messages_for_session(state.session.id)
    assert state.store.session_draft(state.session.id) == "new authored draft"
    assert state.store.received_turn_for_session(state.session.id) is None


async def test_staged_evidence_change_during_capture_refuses_before_promotion(
    received_workspace, monkeypatch
):
    case = received_workspace
    runtime, store, session, controller = (
        case.runtime,
        case.store,
        case.session,
        case.controller,
    )
    draft = "captured original draft"
    store.set_session_draft(session.id, draft)
    original_capture = controller.capture_turn_configuration_snapshot
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_capture(*args, **kwargs):
        result = await original_capture(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(controller, "capture_turn_configuration_snapshot", held_capture)
    intent = _stock_intent(case)
    runtime.accept_received_intent(intent)
    record = runtime._turn_custody[intent.turn_id]
    try:
        await _wait_for_seam(record, entered)
        assert record.request is None
        replacement = object()
        runtime.stage_console_staged_evidence(replacement)
        release.set()
        result = await _outcome(record)
    finally:
        release.set()

    assert isinstance(result, RecoveryRequired)
    assert record.request is None
    assert runtime.snapshot_console_staged_evidence()[0] is replacement
    assert store.session_draft(session.id) == draft
    assert not store.messages_for_session(session.id)


@pytest.mark.parametrize(
    "newer_draft",
    ["captured original draft", "next independent draft"],
    ids=["identical_retype", "distinct_next_draft"],
)
async def test_new_draft_after_full_request_continues_original_controller_preparation(
    received_workspace, monkeypatch, newer_draft
):
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case = received_workspace
    runtime, store, session, controller = (
        case.runtime,
        case.store,
        case.session,
        case.controller,
    )
    draft = "captured original draft"
    store.set_session_draft(session.id, draft)
    store.replace_session_settings(
        session.id, ConsoleSessionSettings(provider="llama_cpp", model="test-model")
    )
    # The capture fixture omits history operations; this submission uses the
    # original real history owner rather than its capture-only placeholder.
    controller.prompt_history = PromptHistory()
    gateway = SequencedGateway()
    gateway.release[0].set()
    controller.provider_gateway = gateway
    store.replace_session_trace_privacy_override(
        session.id,
        capture_enabled=False,
        pii_redaction_enabled=None,
        expected_policy_revision=store.capture_policy_state(session.id).policy_revision,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original_run = runtime._run_custodied_turn

    async def held_run(record, **kwargs):
        assert record.request is not None
        entered.set()
        await release.wait()
        return await original_run(record, **kwargs)

    monkeypatch.setattr(runtime, "_run_custodied_turn", held_run)
    intent = _stock_intent(case)
    with _observe_original_promotion(store) as promoted:
        runtime.accept_received_intent(intent)
        record = runtime._turn_custody[intent.turn_id]
        try:
            await _wait_for_seam(record, entered)
            assert record.request.draft == draft
            store.set_session_draft(session.id, newer_draft, authored_token=(12, 4))
            release.set()
            result = await _outcome(record)
        finally:
            release.set()
            for event in gateway.release:
                event.set()

    assert result.accepted
    assert gateway.user_turns == [draft]
    assert store.session_draft(session.id) == newer_draft
    assert [revision for _identity, revision in promoted] == [
        intent.inputs.draft_revision
    ], "the original controller never promoted the exact receipt revision"


async def test_queue_input_edit_during_original_hook_admission_refuses_before_mutation(
    received_workspace, monkeypatch
):
    case = received_workspace
    runtime, store, session, controller = (
        case.runtime,
        case.store,
        case.session,
        case.controller,
    )
    store.set_session_draft(session.id, "captured queue draft")
    registry = controller.prompt_queue_registry
    previous = registry.snapshot(session.id)
    begun = registry.begin_chain(
        session.id,
        context_epoch=store.conversation_context_epoch(session.id),
        expected_revision=previous.revision,
    )
    assert begun.applied
    before = begun.snapshot
    entered, release = asyncio.Event(), asyncio.Event()
    original_admission = controller.hook_admission_reason

    async def held_admission():
        reason = await original_admission()
        assert reason is None
        entered.set()
        await release.wait()
        return reason

    monkeypatch.setattr(controller, "hook_admission_reason", held_admission)
    intent = _stock_intent(case, queue_revision=before.revision)
    runtime.accept_received_intent(intent)
    record = runtime._turn_custody[intent.turn_id]
    try:
        await _wait_for_seam(record, entered)
        store.set_session_draft(session.id, "new queue draft", authored_token=(15, 2))
        release.set()
        result = await _outcome(record)
    finally:
        release.set()

    assert isinstance(result, RecoveryRequired)
    after = registry.snapshot(session.id)
    assert after.revision == before.revision
    assert after.entries == before.entries
    assert store.session_draft(session.id) == "new queue draft"
    assert store.received_turn_for_session(session.id) is None
    assert not store.messages_for_session(session.id)


async def test_late_custom_queue_wrapper_is_refused_before_call_or_admission(
    received_workspace, monkeypatch
):
    case = received_workspace
    runtime, store, session, controller = (
        case.runtime,
        case.store,
        case.session,
        case.controller,
    )
    store.set_session_draft(session.id, "captured queue input")
    registry = controller.prompt_queue_registry
    before = registry.begin_chain(
        session.id,
        context_epoch=store.conversation_context_epoch(session.id),
        expected_revision=registry.snapshot(session.id).revision,
    ).snapshot
    entered, release = asyncio.Event(), asyncio.Event()
    original_capture = controller.capture_turn_configuration_snapshot
    original_queue = controller.queue_prompt
    calls = []

    async def held_capture(*args, **kwargs):
        snapshot = await original_capture(*args, **kwargs)
        entered.set()
        await release.wait()
        return snapshot

    async def custom_queue(session_id, *, text, expected_revision, configuration=None):
        calls.append(session_id)
        await asyncio.sleep(0)
        return await original_queue(
            session_id,
            text=text,
            expected_revision=expected_revision,
            configuration=configuration,
        )

    monkeypatch.setattr(controller, "capture_turn_configuration_snapshot", held_capture)
    intent = _stock_intent(case, queue_revision=before.revision)
    runtime.accept_received_intent(intent)
    record = runtime._turn_custody[intent.turn_id]
    try:
        await _wait_for_seam(record, entered)
        monkeypatch.setattr(controller, "queue_prompt", custom_queue)
        release.set()
        result = await _outcome(record)
    finally:
        release.set()

    assert (
        not calls
    ), "a late custom queue callback ran before the received source refusal"
    assert isinstance(result, RecoveryRequired)
    after = registry.snapshot(session.id)
    assert after.revision == before.revision
    assert after.entries == before.entries
    assert store.session_draft(session.id) == intent.inputs.draft
    assert not store.messages_for_session(session.id)


async def test_legacy_custom_queue_old_signature_still_uses_original_admission(
    received_workspace, monkeypatch
):
    case = received_workspace
    controller, store, session = case.controller, case.store, case.session
    store.set_session_draft(session.id, "legacy captured queue input")
    registry = controller.prompt_queue_registry
    before = registry.begin_chain(
        session.id,
        context_epoch=store.conversation_context_epoch(session.id),
        expected_revision=registry.snapshot(session.id).revision,
    ).snapshot
    original_queue = controller.queue_prompt
    calls = []

    async def custom_queue(session_id, *, text, expected_revision, configuration=None):
        calls.append((session_id, text, expected_revision))
        await asyncio.sleep(0)
        return await original_queue(
            session_id,
            text=text,
            expected_revision=expected_revision,
            configuration=configuration,
        )

    monkeypatch.setattr(controller, "queue_prompt", custom_queue)
    configuration = await controller.capture_turn_configuration_snapshot(
        session.id, selection=selection_controls._selection(case)
    )
    result = await controller.queue_prompt(
        session.id,
        text="legacy captured queue input",
        expected_revision=before.revision,
        configuration=configuration,
    )

    assert result.applied
    assert calls == [(session.id, "legacy captured queue input", before.revision)]
    assert result.snapshot.total_count == before.total_count + 1
    assert store.session_draft(session.id) == "legacy captured queue input"
