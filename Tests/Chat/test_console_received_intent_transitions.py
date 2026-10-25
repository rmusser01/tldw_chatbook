"""Receipt interleavings at real review, capture, submission and queue owners."""

from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import pytest_asyncio

from Tests.Chat import test_console_configuration_preparation as selection_controls
from Tests.Chat import test_console_configuration_worker_lifetime as worker_controls
from Tests.Chat import test_console_initial_hook_review as review_controls
from Tests.Chat import test_console_received_intent_custody as intake_controls
from Tests.Chat.test_console_prompt_queue_coordinator import SequencedGateway
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleProviderSelection,
    ConsoleStagedSource,
)
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
    received_workspace, monkeypatch, newer_draft, tmp_path
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
    controller.prompt_history = PromptHistory(tmp_path / "received-history.jsonl")
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


def _command_owner(case):
    from tldw_chatbook.Chat.console_remote_commands import (
        REVISION_FIELDS,
        ConsoleRemoteCommands,
    )
    from tldw_chatbook.Chat.console_remote_models import (
        ConsoleHostIdentity,
        ConsoleTarget,
    )

    host = ConsoleHostIdentity(server_epoch="epoch", console_id="console")
    target = ConsoleTarget(
        **host.model_dump(),
        session_id=case.session.id,
        incarnation_id=case.session.incarnation_id,
    )
    revisions = {key: getattr(case.session, key) for key in REVISION_FIELDS}
    return ConsoleRemoteCommands(case.runtime, host), target, revisions


def _configure_command_case(case, tmp_path):
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case.store.replace_session_settings(
        case.session.id,
        ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
    )
    case.controller.prompt_history = PromptHistory(tmp_path / "command-history.jsonl")
    case.store.replace_session_trace_privacy_override(
        case.session.id,
        capture_enabled=False,
        pii_redaction_enabled=None,
        expected_policy_revision=case.store.capture_policy_state(
            case.session.id
        ).policy_revision,
    )
    gateway = SequencedGateway()
    case.controller.provider_gateway = gateway
    return gateway


@pytest.mark.parametrize("native_change", [None, "settings", "binding"])
async def test_command_native_capture_retains_desktop_and_refuses_scope_races(
    received_workspace, monkeypatch, tmp_path, native_change
):
    from Tests.Chat.test_console_received_intent_inputs import _attachment

    case = received_workspace
    gateway = _configure_command_case(case, tmp_path)
    commands, target, revisions = _command_owner(case)
    store, session, runtime = case.store, case.session, case.runtime
    store.set_session_draft(session.id, "phone correction")
    attachment = _attachment("desktop-only")
    store.add_pending_attachment(session.id, attachment)
    store.set_session_one_shot_prefill(session.id, "desktop prefill")
    staged = object()
    runtime.stage_console_staged_evidence(staged)
    store.workspace_context = replace(
        store.workspace_context,
        staged_sources=(ConsoleStagedSource("desktop-rag", "private", "note"),),
    )
    original_capture = case.controller.capture_turn_configuration_snapshot
    entered, release = asyncio.Event(), asyncio.Event()
    captured = []

    async def held_capture(*args, **kwargs):
        entered.set()
        await release.wait()
        configuration = await original_capture(*args, **kwargs)
        captured.append(configuration)
        return configuration

    monkeypatch.setattr(
        case.controller, "capture_turn_configuration_snapshot", held_capture
    )
    result = await commands.execute(
        "send",
        target,
        {"text": "phone correction", "expected_revisions": revisions},
        "phone-turn",
    )
    assert result == {
        "state": "pending",
        "native_id": "phone-turn",
        "reason": "Preparing",
    }
    record = runtime._turn_custody["phone-turn"]
    try:
        await _wait_for_seam(record, entered)
        assert record.request is None and not record.inputs.durable_accepted
        assert not store.messages_for_session(session.id)
        store.set_session_draft(session.id, "phone correction", authored_token=(77, 1))
        desktop = store.create_session(session_id="desktop-view", ephemeral=True)
        store.switch_session(desktop.id)
        newer_stage = object()
        runtime.stage_console_staged_evidence(newer_stage)
        if native_change == "settings":
            store.replace_session_settings(
                session.id, replace(session.settings, temperature=0.3)
            )
        elif native_change == "binding":
            session.conversation_binding_revision += 1
        for event in gateway.release:
            event.set()
        release.set()
        outcome = await _outcome(record)
    finally:
        release.set()
        for event in gateway.release:
            event.set()
    if native_change is None:
        assert outcome.accepted
        assert gateway.user_turns == ["phone correction"]
        assert captured[0].provider_selection.workspace_context.staged_sources == ()
        assert (
            sum(
                m.content == "phone correction" and m.role.value == "user"
                for m in store.messages_for_session(session.id)
            )
            == 1
        )
    else:
        assert isinstance(outcome, RecoveryRequired)
        assert not gateway.user_turns
    assert store.session_draft(session.id) == "phone correction"
    assert store.pending_attachments(session.id) == [attachment]
    assert session.one_shot_prefill == "desktop prefill"
    assert runtime.snapshot_console_staged_evidence()[0] is newer_stage
    assert store.active_session_id == desktop.id


@pytest.mark.parametrize("change_settings", [False, True])
async def test_command_queue_preserves_desktop_riders_through_native_drain(
    received_workspace, tmp_path, change_settings
):
    from Tests.Chat.test_console_received_intent_inputs import _attachment

    case = received_workspace
    gateway = _configure_command_case(case, tmp_path)
    commands, target, revisions = _command_owner(case)
    store, session, runtime = case.store, case.session, case.runtime
    store.set_session_draft(session.id, "desktop draft")
    attachment = _attachment("queue-desktop-only")
    store.add_pending_attachment(session.id, attachment)
    store.set_session_one_shot_prefill(session.id, "desktop prefill")
    staged = object()
    runtime.stage_console_staged_evidence(staged)
    first = await commands.send(
        target, "phone first", "phone-first", expected_revisions=revisions
    )
    record = runtime._turn_custody[first]
    try:
        await _wait_for_seam(record, gateway.started[0])
        second = await commands.send(
            target, "phone second", "phone-second", expected_revisions=revisions
        )
        second_record = runtime._turn_custody[second]
        queued = await _outcome(second_record)
        assert queued.applied
        prompt = case.controller.prompt_queue_registry._states[session.id].waiting[0]
        assert prompt.custody_request.input_origin == "command"
        assert prompt.custody_request.one_shot_prefill is None
        assert prompt.custody_request.command_id == "phone-second"
        assert prompt.custody_request.command_inputs is not None
        if change_settings:
            store.replace_session_settings(
                session.id, replace(session.settings, temperature=0.2)
            )
        for event in gateway.release:
            event.set()
        outcome = await _outcome(record)
    finally:
        for event in gateway.release:
            event.set()
    if change_settings:
        assert isinstance(outcome, RuntimeError)
        assert str(outcome) == "Console command target configuration changed."
    else:
        assert outcome.accepted
    assert gateway.user_turns == (
        ["phone first"] if change_settings else ["phone first", "phone second"]
    )
    assert store.session_draft(session.id) == "desktop draft"
    assert store.pending_attachments(session.id) == [attachment]
    assert session.one_shot_prefill == "desktop prefill"
    assert runtime.snapshot_console_staged_evidence()[0] is staged


async def test_command_settings_change_after_capture_refuses_native_promotion(
    received_workspace, monkeypatch, tmp_path
):
    case = received_workspace
    gateway = _configure_command_case(case, tmp_path)
    commands, target, revisions = _command_owner(case)
    case.store.set_session_draft(case.session.id, "desktop next")
    entered, release = asyncio.Event(), asyncio.Event()
    original_resolve = gateway.resolve_for_send

    async def held_resolve(selection):
        result = await original_resolve(selection)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(gateway, "resolve_for_send", held_resolve)
    turn_id = await commands.send(
        target, "phone", "phone-promotion", expected_revisions=revisions
    )
    record = case.runtime._turn_custody[turn_id]
    try:
        await _wait_for_seam(record, entered)
        assert record.request is not None and not record.inputs.durable_accepted
        case.store.replace_session_settings(
            case.session.id, replace(case.session.settings, temperature=0.2)
        )
        release.set()
        outcome = await _outcome(record)
    finally:
        release.set()
    assert isinstance(outcome, RuntimeError)
    assert not gateway.user_turns
    assert not record.inputs.durable_accepted
    assert case.store.session_draft(case.session.id) == "desktop next"


@pytest.mark.parametrize("onboarding_completed", [False, True])
async def test_command_reroutes_natively_when_primary_finishes_during_queue_admission(
    received_workspace, monkeypatch, tmp_path, onboarding_completed
):
    case = received_workspace
    gateway = _configure_command_case(case, tmp_path)
    case.controller.app.app_config.setdefault("console", {}).setdefault(
        "onboarding", {}
    )["first_send_completed"] = onboarding_completed
    commands, target, revisions = _command_owner(case)
    case.store.set_session_draft(case.session.id, "desktop next")
    first = await commands.send(
        target, "phone first", "phone-first", expected_revisions=revisions
    )
    first_record = case.runtime._turn_custody[first]
    entered, release = asyncio.Event(), asyncio.Event()
    original_admission = case.controller.hook_admission_reason

    async def held_admission():
        reason = await original_admission()
        entered.set()
        await release.wait()
        return reason

    try:
        await _wait_for_seam(first_record, gateway.started[0])
        monkeypatch.setattr(case.controller, "hook_admission_reason", held_admission)
        second = await commands.send(
            target, "phone second", "phone-second", expected_revisions=revisions
        )
        second_record = case.runtime._turn_custody[second]
        assert second_record.received_intent.queue_revision is not None
        await _wait_for_seam(second_record, entered)
        assert second_record.received_claim is None
        gateway.release[0].set()
        assert (await _outcome(first_record)).accepted
        case.store.set_session_draft(case.session.id, "new desktop next")
        gateway.release[1].set()
        release.set()
        outcome = await _outcome(second_record)
        if onboarding_completed:
            assert outcome.accepted
        else:
            # Native first-send persistence changes the config source mid-admission.
            assert isinstance(outcome, RecoveryRequired)
    finally:
        release.set()
        for event in gateway.release:
            event.set()
    assert gateway.user_turns == (
        ["phone first", "phone second"] if onboarding_completed else ["phone first"]
    )
    assert case.store.session_draft(case.session.id) == "new desktop next"
