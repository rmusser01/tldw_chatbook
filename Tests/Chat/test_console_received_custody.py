"""Exact received admission across the existing complete runtime boundary."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ConsoleSubmitResult,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleProviderSelection,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnCustodyRequest,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsoleTurnPreparationState,
)

pytestmark = pytest.mark.bootstrap_profile


def _case():
    store = ConsoleChatStore()
    session = store.create_session(session_id="received-session", ephemeral=True)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    request = ConsoleTurnCustodyRequest(
        turn_id="received-request",
        session_id=session.id,
        draft="captured original draft",
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id=session.id,
            provider_selection=ConsoleProviderSelection(
                provider="llama_cpp", explicit_model="test-model"
            ),
        ),
    )
    return runtime, store, session, request


def _attachment(identifier):
    return PendingAttachment(
        file_path=f"/{identifier}",
        display_name=identifier,
        file_type="image",
        insert_mode="attachment",
        attachment_id=identifier,
    )


async def _cancel_custody(runtime, additional=()):
    tasks = tuple(
        record.task
        for record in runtime._turn_custody.values()
        if record.task is not None
    ) + tuple(additional)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)


async def _retire(runtime, additional=()):
    await _cancel_custody(runtime, additional)
    # set_chat_store also owns the Canvas watcher; custody alone is not teardown.
    await asyncio.wait_for(runtime.dispose(timeout_seconds=0.01), 5)


@pytest_asyncio.fixture(autouse=True)
async def _owned_case_runtimes(monkeypatch):
    """Retire each exact runtime even when a regression assertion fails early."""
    original = _case
    runtimes = []

    def owned_case():
        case = original()
        runtimes.append(case[0])
        return case

    monkeypatch.setitem(globals(), "_case", owned_case)
    yield
    for runtime in reversed(runtimes):
        if not runtime._disposed:
            await _retire(runtime)


def _preparation(request):
    from Tests.Chat.test_console_automatic_library_preparation import (
        _preparation as complete_preparation,
    )

    # Reuse the established checkpoint-valid authority/source-types/destination;
    # retain this request's exact configuration and ephemeral fixture posture.
    preparation = complete_preparation(
        session_id=request.session_id,
        preparation_id="received-preparation",
        state=ConsoleTurnPreparationState.READY,
        draft=request.draft,
    )
    return replace(
        preparation,
        execution_context=replace(
            preparation.execution_context,
            configuration=request.configuration,
        ),
        ephemeral=True,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_OFF,
    )


@pytest.mark.asyncio
async def test_actual_duplicate_accept_before_task_start_moves_no_loser_inputs():
    runtime, store, session, request = _case()
    session.persisted_conversation_id = "original-conversation"
    runtime._app = SimpleNamespace()
    first, later = _attachment("first-attachment"), _attachment("later-attachment")
    assert store.add_pending_attachment(session.id, first)
    request = replace(request, attachment_ids=(first.attachment_id,))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def run(_record, **_kwargs):
        entered.set()
        await release.wait()

    runtime._run_custodied_turn = run
    try:
        first_id = runtime.accept_turn(request)
        assert not entered.is_set()
        assert store.add_pending_attachment(session.id, later)
        second = replace(
            request,
            turn_id="loser-request",
            draft="later draft",
            attachment_ids=(later.attachment_id,),
        )
        with pytest.raises(RuntimeError):
            runtime.accept_turn(second)
        assert not entered.is_set()
        assert tuple(runtime._turn_custody) == (first_id,)
        assert runtime._app._conversation_send_inflight == {"original-conversation": 1}
        assert store.pending_attachments(session.id) == [later]
        assert store.pending_attachments(session.id)[0] is later
        assert runtime._turn_custody[first_id].inputs.attachments == (first,)
    finally:
        release.set()
        await _retire(runtime)
    assert runtime._app._conversation_send_inflight == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin,preparing",
    [
        (ConsoleSubmissionOrigin.MANUAL, True),
        (ConsoleSubmissionOrigin.AGENT_WAKE, False),
    ],
    ids=["manual-preparing", "wake-occupied"],
)
async def test_received_activity_occupies_slot_before_any_runtime_task_runs(
    origin, preparing
):
    runtime, store, session, request = _case()
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    runtime._chat_controller = controller
    entered = asyncio.Event()

    async def run(_record, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    runtime._run_custodied_turn = run
    try:
        runtime.accept_turn(request, origin=origin)
        assert not entered.is_set()
        activity = controller.activity_for(session.id)
        assert activity.occupies_slot
        assert activity.preparing_before_acceptance is preparing
        assert not activity.accepted_live_turn
        assert not activity.terminal_notification_eligible
        assert controller._active_run_rejection(session_id=session.id) is not None
    finally:
        await _retire(runtime)


@pytest.mark.asyncio
async def test_cancel_before_first_task_step_releases_original_claim_and_keeps_exact_recovery():
    runtime, store, session, request = _case()
    first, later = _attachment("first-attachment"), _attachment("later-attachment")
    assert store.add_pending_attachment(session.id, first)
    request = replace(request, attachment_ids=(first.attachment_id,))
    entered = asyncio.Event()

    async def run(_record, **_kwargs):
        entered.set()
        pytest.fail("Cancelled-before-start custody ran its body")

    runtime._run_custodied_turn = run
    turn_id = runtime.accept_turn(request)
    claim = store.received_turn_for_session(session.id)
    assert claim is not None
    assert store.add_pending_attachment(session.id, later)
    await _cancel_custody(runtime)
    assert not entered.is_set()
    assert store.received_turn_for_session(session.id) is None
    assert not runtime.has_custodied_turns(session.id)
    entry = runtime.recoveries_for_session(session.id)[0]
    assert entry.turn_id == turn_id and entry.draft == request.draft
    assert entry.attachments == (first,) and entry.attachments[0] is first
    assert store.pending_attachments(session.id) == [later]
    assert store.pending_attachments(session.id)[0] is later


@pytest.mark.asyncio
async def test_private_lazy_task_ignores_eager_start_then_raising_loop_factory():
    runtime, store, session, request = _case()
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    factory_tasks = []
    calls = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def run(_record, **_kwargs):
        entered.set()
        await release.wait()

    def factory(loop, coroutine, **kwargs):
        calls.append(coroutine)
        factory_tasks.append(
            asyncio.Task(coroutine, loop=loop, eager_start=True, **kwargs)
        )
        raise RuntimeError("eager factory started then failed")

    runtime._run_custodied_turn = run
    try:
        loop.set_task_factory(factory)
        try:
            turn_id = runtime.accept_turn(request)
        finally:
            loop.set_task_factory(original_factory)
        assert not calls
        assert not entered.is_set()
        assert store.received_turn_for_session(session.id) is not None
        task = runtime._turn_custody[turn_id].task
        await asyncio.wait_for(entered.wait(), 1)
        release.set()
        await asyncio.wait_for(task, 1)
        await asyncio.sleep(0)
        assert store.received_turn_for_session(session.id) is None
    finally:
        loop.set_task_factory(original_factory)
        release.set()
        await _retire(runtime, factory_tasks)


@pytest.mark.asyncio
async def test_scheduling_failure_restores_prefix_and_releases_only_its_original_claim(
    monkeypatch,
):
    runtime, store, session, request = _case()
    first, later = _attachment("first-attachment"), _attachment("later-attachment")
    assert store.add_pending_attachment(session.id, first)
    assert store.add_pending_attachment(session.id, later)
    request = replace(request, attachment_ids=(first.attachment_id,))
    observed = []

    def fail(coroutine):
        observed.append(store.received_turn_for_session(session.id))
        raise RuntimeError("private scheduler unavailable")

    monkeypatch.setattr(runtime, "_create_custody_task", fail, raising=False)
    try:
        with pytest.raises(RuntimeError, match="private scheduler unavailable"):
            runtime.accept_turn(request)
        assert len(observed) == 1 and observed[0] is not None
        assert store.received_turn_for_session(session.id) is None
        assert runtime._turn_custody == {}
        pending = store.pending_attachments(session.id)
        assert pending == [first, later]
        assert pending[0] is first and pending[1] is later
        assert not runtime.recoveries_for_session(session.id)
    finally:
        await _retire(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["promote", "seal", "release"])
async def test_existing_custom_submit_shape_observes_exact_task_claim_and_no_stale_fallback(
    retirement,
):
    from tldw_chatbook.Chat.console_received_turn import received_turn_claim_for

    runtime, store, session, request = _case()
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    runtime._chat_controller = controller
    preparation = _preparation(request)
    observed = []

    async def chain(*, session_id, initial_turn):
        return await initial_turn()

    # Exact current public invocation shape: no claim or private authorization kwargs.
    async def submit(
        draft,
        *,
        session_id,
        origin,
        queue_entry_id,
        queue_authorization,
        wake_authorization,
        configuration,
        accepted_attachments,
        captured_one_shot_prefill,
        captured_one_shot_prefill_revision,
        staged_evidence_launch,
        staged_evidence_capture,
        staged_evidence_release,
        custody_acceptance_hook,
    ):
        claim = received_turn_claim_for(store, session_id)
        assert claim is store.received_turn_for_session(session_id)
        assert claim is not None
        assert draft == request.draft and configuration.session_id == session_id
        assert queue_authorization is None and wake_authorization is None

        async def child():
            assert received_turn_claim_for(store, session_id) is None
            assert (
                controller._begin_submit_preparation(
                    asyncio.current_task(), preparation
                )
                is None
            )

        await asyncio.create_task(child())
        if retirement == "seal":
            assert store.seal_received_turn(claim)
        elif retirement == "release":
            assert store.release_received_turn(claim)
        # The binding MUST still expose a sealed/released claim, never ordinary fallback.
        assert received_turn_claim_for(store, session_id) is claim
        installed = controller._begin_submit_preparation(
            asyncio.current_task(), preparation
        )
        observed.append((claim, installed))
        assert installed is (preparation if retirement == "promote" else None)
        return ConsoleSubmitResult(
            False, False, terminal_status=ConsoleRunStatus.BLOCKED
        )

    controller.run_prompt_chain = chain
    controller.submit_draft = submit
    try:
        turn_id = runtime.accept_turn(request, recover_before_acceptance=False)
        task = runtime._turn_custody[turn_id].task
        await asyncio.wait_for(task, 1)
        await asyncio.sleep(0)
        assert len(observed) == 1
        assert store.received_turn_for_session(session.id) is None
        assert store.preparation_for_session(session.id) is (
            preparation if retirement == "promote" else None
        )
        assert store.release_received_turn(observed[0][0]) is False
    finally:
        await _retire(runtime)


@pytest.mark.asyncio
async def test_stop_before_first_task_step_seals_claim_without_controller_submission():
    runtime, store, session, request = _case()
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    runtime._chat_controller = controller
    submitted = []

    async def submit(*args, **kwargs):
        submitted.append(args)
        pytest.fail("A sealed received request entered controller submission")

    controller.submit_draft = submit
    try:
        turn_id = runtime.accept_turn(request)
        claim = store.received_turn_for_session(session.id)
        assert claim is not None
        assert controller.stop_active_run()
        assert claim.sealed
        assert store.received_turn_for_session(session.id) is claim
        task = runtime._turn_custody[turn_id].task
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
        await asyncio.sleep(0)
        assert not submitted
        assert store.received_turn_for_session(session.id) is None
        assert not runtime.has_custodied_turns(session.id)
    finally:
        await _retire(runtime)


@pytest.mark.asyncio
async def test_close_before_preparation_seals_original_claim_before_session_removal():
    runtime, store, session, request = _case()
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    runtime._chat_controller = controller
    entered = asyncio.Event()

    async def run(_record, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    runtime._run_custodied_turn = run
    try:
        runtime.accept_turn(request)
        claim = store.received_turn_for_session(session.id)
        assert claim is not None
        await asyncio.wait_for(
            runtime.close_session(
                session.id,
                expected_revision=controller.lifecycle_impact().revision,
                timeout_seconds=0.01,
            ),
            1,
        )
        assert claim.sealed
        assert store.received_turn_for_session(session.id) is None
        assert not runtime.has_custodied_turns(session.id)
        assert all(item is not session for item in store.sessions())
    finally:
        await _retire(runtime)


@pytest.mark.asyncio
async def test_replaced_runtime_store_cannot_receive_submit_or_release_successor_claim():
    runtime, store, session, request = _case()
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    runtime._chat_controller = controller
    submitted = []

    async def submit(*args, **kwargs):
        submitted.append(args)
        pytest.fail("Source-displaced request entered submission")

    controller.submit_draft = submit
    replacement = ConsoleChatStore()
    replacement.create_session(session_id=session.id, ephemeral=True)
    try:
        turn_id = runtime.accept_turn(request)
        original = store.received_turn_for_session(session.id)
        assert original is not None
        successor = replacement.claim_received_turn(session.id, "successor-request")
        assert successor is not None
        runtime.set_chat_store(replacement)
        task = runtime._turn_custody[turn_id].task
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
        await asyncio.sleep(0)
        assert not submitted
        assert store.received_turn_for_session(session.id) is None
        assert replacement.received_turn_for_session(session.id) is successor
        assert store.release_received_turn(original) is False
    finally:
        if "successor" in locals():
            replacement.release_received_turn(successor)
        await _retire(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin", [ConsoleSubmissionOrigin.QUEUED, ConsoleSubmissionOrigin.AGENT_WAKE]
)
async def test_received_busy_guard_preserves_forged_origin_permission_errors(origin):
    runtime, store, session, request = _case()
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    runtime._chat_controller = controller

    async def admission():
        return None

    controller.hook_admission_reason = admission
    claim = store.claim_received_turn(session.id, "other-request")
    assert claim is not None
    kwargs = (
        {"queue_entry_id": "forged-entry", "queue_authorization": None}
        if origin is ConsoleSubmissionOrigin.QUEUED
        else {"wake_authorization": object()}
    )
    try:
        with pytest.raises(PermissionError):
            await controller.submit_draft(
                "forged body",
                session_id=session.id,
                origin=origin,
                configuration=request.configuration,
                **kwargs,
            )
        assert store.received_turn_for_session(session.id) is claim
        assert store.messages_for_session(session.id) == []
    finally:
        store.release_received_turn(claim)
        await _retire(runtime)


@pytest.mark.asyncio
async def test_nonpromoted_attachment_turn_releases_before_actual_next_queue_custody(
    monkeypatch,
):
    from Tests.Chat.test_console_prompt_queue_coordinator import (
        SequencedGateway,
        _queue,
    )

    runtime, store, session, request = _case()
    gateway = SequencedGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime.set_chat_controller(controller)
    attachment = PendingAttachment(
        "/received-image",
        "received.png",
        "image",
        "attachment",
        data=b"processed-image",
        mime_type="image/png",
    )
    assert store.add_pending_attachment(session.id, attachment)
    request = replace(
        request,
        draft="",
        attachment_ids=(attachment.attachment_id,),
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id=session.id,
            provider_selection=ConsoleProviderSelection(
                provider="llama_cpp", explicit_model="test-model"
            ),
            capabilities={"vision": True, "max_history_images": 1},
        ),
    )
    accepted = []
    first_task = None
    original_accept = runtime.accept_turn

    def observe_accept(next_request, **kwargs):
        turn_id = original_accept(next_request, **kwargs)
        accepted.append(
            (
                next_request,
                store.received_turn_for_session(session.id),
                first_task is None or not first_task.done(),
            )
        )
        return turn_id

    monkeypatch.setattr(runtime, "accept_turn", observe_accept)
    try:
        first_id = runtime.accept_turn(request)
        first_task = runtime._turn_custody[first_id].task
        first_claim = accepted[0][1]
        assert first_claim is not None
        await asyncio.wait_for(gateway.started[0].wait(), 5)
        assert store.preparation_for_session(session.id) is None
        assert store.received_turn_for_session(session.id) is first_claim
        queued_id = await _queue(controller, session.id, "queued after attachment")
        gateway.release[0].set()
        await asyncio.wait_for(gateway.started[1].wait(), 5)
        assert not first_task.done()
        assert len(accepted) == 2
        queued_request, next_claim, first_pending = accepted[1]
        assert queued_request.draft == "queued after attachment"
        assert next_claim is not None and next_claim is not first_claim
        assert next_claim.origin is ConsoleSubmissionOrigin.QUEUED
        assert first_pending
        assert controller.prompt_queue_registry.snapshot(session.id).total_count == 0
        gateway.release[1].set()
        result = await asyncio.wait_for(first_task, 5)
        await asyncio.sleep(0)
        assert result.accepted
        assert len(gateway.user_turns) == 2
        assert not runtime.has_custodied_turns(session.id)
        assert store.received_turn_for_session(session.id) is None
        assert queued_id
    finally:
        for release in gateway.release:
            release.set()
        await _retire(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["different-store", "same-store-same-id"])
async def test_already_recorded_recovery_cannot_restore_into_a_successor_session(
    replacement,
):
    runtime, store, session, request = _case()
    attachment = _attachment("captured-attachment")
    assert store.add_pending_attachment(session.id, attachment)
    request = replace(request, attachment_ids=(attachment.attachment_id,))
    turn_id = runtime.accept_turn(request)
    await _cancel_custody(runtime)
    entry = runtime.recoveries_for_session(session.id)[0]
    assert entry.turn_id == turn_id and entry.attachments[0] is attachment
    if replacement == "different-store":
        successor_store = ConsoleChatStore()
        successor_store.create_session(session_id=session.id, ephemeral=True)
        runtime.set_chat_store(successor_store)
    else:
        store.close_session(session.id)
        store.create_session(session_id=session.id, ephemeral=True)
        successor_store = store
    successor_attachment = _attachment("successor-attachment")
    assert successor_store.add_pending_attachment(session.id, successor_attachment)
    assert runtime.recoveries_for_session(session.id) == ()
    with pytest.raises(RuntimeError, match="owner changed"):
        runtime.restore_turn_recovery(turn_id)
    assert runtime._turn_recoveries[turn_id] is entry
    assert successor_store.session_draft(session.id) == ""
    assert successor_store.pending_attachments(session.id) == [successor_attachment]
    assert successor_store.pending_attachments(session.id)[0] is successor_attachment


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["seal", "binding-change", "store-change", "foreign-receipt"]
)
async def test_nonpromoted_resolution_rechecks_receipt_before_any_acceptance(mutation):
    from Tests.Chat.test_console_prompt_queue_coordinator import SequencedGateway

    class HeldResolution(SequencedGateway):
        def __init__(self):
            super().__init__()
            self.resolving = asyncio.Event()
            self.allow_resolution = asyncio.Event()

        async def resolve_for_send(self, selection):
            self.resolving.set()
            await self.allow_resolution.wait()
            return await super().resolve_for_send(selection)

    runtime, store, session, request = _case()
    gateway = HeldResolution()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime.set_chat_controller(controller)
    accepted = []
    controller.on_submission_accepted = lambda: accepted.append(True)
    attachment = PendingAttachment(
        "/image",
        "image.png",
        "image",
        "attachment",
        data=b"image",
        mime_type="image/png",
    )
    assert store.add_pending_attachment(session.id, attachment)
    request = replace(
        request,
        draft="",
        attachment_ids=(attachment.attachment_id,),
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id=session.id,
            provider_selection=ConsoleProviderSelection(
                provider="llama_cpp", explicit_model="test-model"
            ),
            capabilities={"vision": True, "max_history_images": 1},
        ),
    )
    try:
        if mutation == "foreign-receipt":
            task = asyncio.create_task(
                controller.submit_draft(
                    request.draft,
                    session_id=session.id,
                    configuration=request.configuration,
                    accepted_attachments=(attachment,),
                )
            )
        else:
            turn_id = runtime.accept_turn(request)
            task = runtime._turn_custody[turn_id].task
        await asyncio.wait_for(gateway.resolving.wait(), 5)
        claim = store.received_turn_for_session(session.id)
        assert (claim is None) is (mutation == "foreign-receipt")
        assert store.preparation_for_session(session.id) is None
        if mutation == "foreign-receipt":

            async def hold_foreign(_record, **_kwargs):
                await asyncio.Event().wait()

            runtime._run_custodied_turn = hold_foreign
            runtime.accept_turn(replace(request, turn_id="foreign-request"))
            foreign_claim = store.received_turn_for_session(session.id)
            assert foreign_claim is not None
        elif mutation == "seal":
            assert store.seal_received_turn(claim)
        elif mutation == "binding-change":
            session.conversation_binding_revision += 1
        else:
            successor = ConsoleChatStore()
            successor.create_session(session_id=session.id, ephemeral=True)
            controller.store = successor
        gateway.allow_resolution.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
        await asyncio.sleep(0)
        assert accepted == []
        assert gateway.user_turns == []
        if mutation == "foreign-receipt":
            assert store.received_turn_for_session(session.id) is foreign_claim
        else:
            assert store.received_turn_for_session(session.id) is None
            recovery = runtime._turn_recoveries[turn_id]
            assert (
                recovery.attachments == (attachment,)
                and recovery.attachments[0] is attachment
            )
        assert not any(
            message.role.value == "assistant"
            for message in store.messages_for_session(session.id)
        )
    finally:
        gateway.allow_resolution.set()
        await _retire(runtime, (task,) if "task" in locals() else ())


@pytest.mark.asyncio
async def test_late_predecessor_callback_does_not_release_same_id_successor():
    runtime, store, session, request = _case()
    releases = []
    terminal = []

    async def run(_record, **_kwargs):
        gate = asyncio.Event()
        releases.append(gate)
        await gate.wait()

    runtime._run_custodied_turn = run
    first_id = runtime.accept_turn(request, terminal_callback=terminal.append)
    first_record = runtime._turn_custody[first_id]
    first_task = first_record.task
    old_claim = store.received_turn_for_session(session.id)
    assert old_claim is not None
    # Model a late original callback after exact old admission is retired.
    assert store.release_received_turn(old_claim)
    runtime._turn_custody.pop(first_id)
    successor_id = runtime.accept_turn(request)
    successor = runtime._turn_custody[successor_id]
    successor_claim = store.received_turn_for_session(session.id)
    assert successor_claim is not None and successor_claim is not old_claim
    try:
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)
        await asyncio.sleep(0)
        assert runtime._turn_custody[successor_id] is successor
        assert store.received_turn_for_session(session.id) is successor_claim
        assert not successor.task.done()
        assert runtime.recoveries_for_session(session.id) == ()
        assert terminal == [False]
    finally:
        for release in releases:
            release.set()
        await _retire(runtime, (first_task,))
