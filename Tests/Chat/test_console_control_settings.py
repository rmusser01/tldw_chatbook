"""Actual persisted settings writes retain control responsiveness and order."""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import sys
import threading
from contextlib import contextmanager

import pytest

from Tests.Chat.test_console_domain_boundary import domain_case as _domain_case
from Tests.Chat.test_console_settings_apply_store import _settings, _submission
from Tests.Chat.test_console_first_send_atomicity import _until
from Tests.Backup_Recovery.test_console_main_display_batch import _worker_connections
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases

domain_case = _domain_case

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@contextmanager
def _hold_full_writer(service):
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

    code = inspect.unwrap(
        ChatPersistenceService.update_conversation_console_session_settings
    ).__code__
    entered, release = threading.Event(), threading.Event()
    previous, previous_main = threading.getprofile(), sys.getprofile()
    calls = []

    def observe(frame, event, value):
        if previous is not None:
            previous(frame, event, value)
        if (
            event == "return"
            and frame.f_code is code
            and frame.f_locals.get("self") is service
        ):
            calls.append(frame.f_locals["settings"].model)
            if len(calls) == 1:
                entered.set()
                assert release.wait(10), "Held original settings writer did not release"

    threading.setprofile_all_threads(observe)
    try:
        yield entered, release, calls
    finally:
        release.set()
        threading.setprofile_all_threads(previous)
        sys.setprofile(previous_main)


async def _entered(event):
    async with asyncio.timeout(10):
        while not event.is_set():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("domain_case", ["control_settings"], indirect=True)
async def test_full_snapshot_retains_original_order_and_coherent_input_capture(
    domain_case,
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
        UpdateInputs,
    )
    from tldw_chatbook.Chat.console_control_settings import settings_pending

    case = domain_case
    case.store.replace_session_settings(
        case.session.id, _settings("old", system_prompt="owned")
    )
    conversation_id = case.store.persistence.create_conversation(
        conversation_title="Settings owner"
    )
    case.store.publish_first_persisted_conversation(case.session.id, conversation_id)
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )

    def identity(number):
        return ControlIdentity(
            "profile",
            "service",
            number,
            f"settings-{number}",
            case.session.id,
            case.session.incarnation_id,
        )

    first = _submission(
        case.store, case.session.id, submission_id="first", model="model-a"
    )
    second = _submission(
        case.store, case.session.id, submission_id="second", model="model-b"
    )
    before = case.session.settings_revision
    with (
        _worker_connections(case.database) as connections,
        _hold_full_writer(case.store.persistence) as (entered, release, calls),
    ):
        applied = controls.dispatch(ApplySettings(identity(1), first))
        await _entered(entered)
        pending = settings_pending(case.store, case.session.id)
        leases = worker_leases(case.database)
        assert leases and connections
        assert pending is not None and pending.effect in controls._effects
        assert case.session.settings.model == "model-a"
        assert case.session.settings_revision == before
        assert not case.store._voice_promotion_mutation_is_admitted_here(
            case.session.id
        )
        assert case.store._voice_promotion_mutation_admissions[case.session.id] > 0
        assert case.store._fork_source_transitions[case.session.id] > 0
        # Same-loop controls can still acquire preparation while the actual
        # original snapshot writer is held on its own native worker.
        with case.store._preparation_lock:
            assert case.session.settings.system_prompt == "owned"
        updated = controls.dispatch(
            UpdateInputs(identity(2), "captured after apply", (1, 1))
        )
        later = controls.dispatch(ApplySettings(identity(3), second))
        await asyncio.sleep(0)
        assert not updated.applied.done() and not later.applied.done()
        assert calls == ["model-a"]
        release.set()
        commit = await applied.applied
        await updated.applied
        newest = await later.applied
        assert commit.settings.model == "model-a" and newest.settings.model == "model-b"
        assert calls == ["model-a", "model-b"]
        assert await applied.retired and await later.retired
        assert (
            case.store.session_input_snapshot(case.session.id).draft
            == "captured after apply"
        )
        assert case.session.settings_revision == before + 2
        assert not case.store._fork_source_transitions
        assert not case.store._voice_promotion_mutation_admissions
        assert connections and leases
        assert (
            pending.retirement.confirmed and pending.retirement.connection is not None
        )
        import sqlite3

        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(connection)
        assert not worker_leases(case.database)
    await case.store.persist_console_settings_commit_serialized(newest)
    row = case.database.get_conversation_by_id(conversation_id)
    import json

    assert json.loads(row["metadata"])["console_session_settings"]["model"] == "model-b"


@pytest.mark.parametrize("domain_case", ["control_settings"], indirect=True)
async def test_apply_cancelled_before_first_turn_releases_admission_and_fence(
    domain_case,
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
        FenceSource,
    )

    case = domain_case
    case.store.replace_session_settings(case.session.id, _settings("old"))
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    identity = lambda number: ControlIdentity(
        "profile",
        "service",
        number,
        f"pre-entry-{number}",
        case.session.id,
        case.session.incarnation_id,
    )
    submission = _submission(
        case.store, case.session.id, submission_id="never-started", model="new"
    )
    receipt = controls.dispatch(ApplySettings(identity(1), submission))
    (effect,) = controls._effects
    operation = effect.owner
    assert operation.admission in operation.durability._leases
    assert operation.finished in operation.lifecycle.pending_live
    receipt.work.cancel()
    fence = controls.dispatch(FenceSource(identity(2), "configuration"))
    async with asyncio.timeout(10):
        with pytest.raises(asyncio.CancelledError):
            await receipt.applied
        assert await receipt.retired
        await fence.applied
    assert case.session.settings.model == "old"
    assert "never-started" not in case.session.applied_settings_submission_ids
    assert operation.retirement is None and operation.transition is None
    assert not operation.durability._leases and not operation.lifecycle.pending_live
    assert not controls._effects and not controls._session_capture_tails


@pytest.mark.parametrize("domain_case", ["control_settings"], indirect=True)
async def test_pre_entry_cancelled_settings_and_input_keep_unfinished_predecessor(
    domain_case,
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
        UpdateInputs,
    )

    case = domain_case
    case.store.replace_session_settings(case.session.id, _settings("old"))
    conversation = case.store.persistence.create_conversation(
        conversation_title="Canceled successor ordering"
    )
    case.store.publish_first_persisted_conversation(case.session.id, conversation)
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    identity = lambda number: ControlIdentity(
        "profile",
        "service",
        number,
        f"ordered-cancel-{number}",
        case.session.id,
        case.session.incarnation_id,
    )
    first = _submission(
        case.store, case.session.id, submission_id="prior", model="first"
    )
    canceled = _submission(
        case.store, case.session.id, submission_id="skipped", model="skipped"
    )
    with _hold_full_writer(case.store.persistence) as (entered, release, calls):
        prior = controls.dispatch(ApplySettings(identity(1), first))
        await _entered(entered)
        skipped = controls.dispatch(ApplySettings(identity(2), canceled))
        skipped.work.cancel()
        skipped_input = controls.dispatch(
            UpdateInputs(identity(3), "never captured", (1, 1))
        )
        skipped_input.work.cancel()
        successor = controls.dispatch(
            UpdateInputs(identity(4), "after original writer", (1, 1))
        )
        with pytest.raises(asyncio.CancelledError):
            await skipped.applied
        with pytest.raises(asyncio.CancelledError):
            await skipped_input.applied
        assert await skipped.retired
        assert not successor.applied.done() and not prior.retired.done()
        assert worker_leases(case.database) and calls == ["first"]
        release.set()
        await prior.applied
        await successor.applied
    assert "skipped" not in case.session.applied_settings_submission_ids
    assert (
        case.store.session_input_snapshot(case.session.id).draft
        == "after original writer"
    )
    assert not controls._effects and not controls._session_capture_tails


@pytest.mark.parametrize("domain_case", ["control_settings"], indirect=True)
@pytest.mark.parametrize("refused_coroutine", ["execute", "capture"])
async def test_settings_task_scheduling_refusal_releases_exact_admission(
    domain_case, monkeypatch, refused_coroutine
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
        FenceSource,
    )

    case = domain_case
    case.store.replace_session_settings(case.session.id, _settings("old"))
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    identity = lambda number: ControlIdentity(
        "profile",
        "service",
        number,
        f"schedule-refusal-{number}",
        case.session.id,
        case.session.incarnation_id,
    )
    submission = _submission(
        case.store, case.session.id, submission_id="refused", model="new"
    )
    original = controls._loop.create_task
    refused = []

    def refuse(coroutine, *args, **kwargs):
        if not refused and coroutine.cr_code.co_name == refused_coroutine:
            refused.append(coroutine)
            raise RuntimeError("injected task scheduling refusal")
        return original(coroutine, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controls._loop, "create_task", refuse)
        with pytest.raises(RuntimeError, match="injected task scheduling refusal"):
            receipt = controls.dispatch(ApplySettings(identity(1), submission))
            await receipt.applied
    assert refused and refused[0].cr_frame is None
    fence = controls.dispatch(FenceSource(identity(2), "configuration"))
    async with asyncio.timeout(10):
        await fence.applied
    lifecycle = case.store._settings_persistence_lifecycles[case.session.id]
    owner = case.sources.settings_durability_owner
    assert not owner._leases and not owner.tasks
    assert not lifecycle.pending_live and not lifecycle.tasks
    assert not controls._effects and not controls._session_capture_tails
    assert case.session.settings.model == "old"
    assert "refused" not in case.session.applied_settings_submission_ids


@pytest.mark.parametrize("domain_case", ["control_settings"], indirect=True)
async def test_cancelled_apply_keeps_post_phase_and_source_fence_behind_native_retirement(
    domain_case,
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
        FenceSource,
    )
    from tldw_chatbook.Chat.console_control_settings import settings_pending

    case = domain_case
    case.store.replace_session_settings(case.session.id, _settings("old"))
    conversation_id = case.store.persistence.create_conversation(
        conversation_title="Cancelled settings"
    )
    case.store.publish_first_persisted_conversation(case.session.id, conversation_id)
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )

    def identity(number):
        return ControlIdentity(
            "profile",
            "service",
            number,
            f"cancel-settings-{number}",
            case.session.id,
            case.session.incarnation_id,
        )

    submission = _submission(
        case.store, case.session.id, submission_id="cancelled", model="model-a"
    )
    before = case.session.settings_revision
    with _hold_full_writer(case.store.persistence) as (entered, release, calls):
        receipt = controls.dispatch(ApplySettings(identity(1), submission))
        await _entered(entered)
        pending = settings_pending(case.store, case.session.id)
        assert pending is not None and worker_leases(case.database)
        receipt.work.cancel()
        await asyncio.sleep(0)
        receipt.work.cancel()
        fence = controls.dispatch(FenceSource(identity(2), "configuration"))
        assert controls.source_revision == 1
        await asyncio.sleep(0)
        assert not fence.applied.done() and not receipt.retired.done()
        assert pending in (effect.owner for effect in controls._effects)
        assert case.store._fork_source_transitions[case.session.id] > 0
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await receipt.applied
        assert await receipt.retired
        await fence.applied
        assert calls == ["model-a"]
        assert case.session.settings_revision == before + 1
        assert case.session.applied_settings_submission_ids[-1] == "cancelled"
        assert pending.commit is not None and pending.retirement.confirmed
        assert not worker_leases(case.database)
        assert not case.store._fork_source_transitions
        assert not case.store._voice_promotion_mutation_admissions


@pytest.mark.parametrize("domain_case", ["control_settings"], indirect=True)
async def test_actual_durability_task_cancelled_before_capture_entry_releases_tail(
    domain_case,
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
        FenceSource,
    )

    case = domain_case
    case.store.replace_session_settings(case.session.id, _settings("old"))
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="inner", service_epoch="local"
    )
    identity = lambda number: ControlIdentity(
        "inner",
        "local",
        number,
        f"inner-{number}",
        case.session.id,
        case.session.incarnation_id,
    )
    receipt = controls.dispatch(
        ApplySettings(
            identity(1),
            _submission(
                case.store, case.session.id, submission_id="inner-cancel", model="new"
            ),
        )
    )
    (effect,) = controls._effects
    operation = effect.owner
    tail = controls._session_capture_tails[case.session.id]
    actual = []

    def cancel_actual_inner():
        (task,) = operation.lifecycle.tasks
        actual.append(task)
        assert operation.pre_state is None and operation.retirement is None
        task.cancel()

    controls._loop.call_soon(cancel_actual_inner)
    async with asyncio.timeout(10):
        with pytest.raises(asyncio.CancelledError):
            await receipt.applied
        assert await receipt.retired
        await controls.dispatch(FenceSource(identity(2), "configuration")).applied
    assert len(actual) == 1 and actual[0].cancelled()
    assert case.session.settings.model == "old"
    assert operation.pre_state is None and operation.retirement is None
    assert operation.admission not in operation.durability._leases
    assert not operation.lifecycle.pending_live and not operation.lifecycle.tasks
    assert tail.done() and tail not in controls._capture_predecessors
    assert not controls._effects and not controls._session_capture_tails


async def _settle_held_control_targets(
    case, body_error, apply, fence, target, *, cancel_target
):
    """Retrieve exact issued work, retaining body AND cleanup failures."""
    errors = [] if body_error is None else [body_error]
    if target is not None:
        if body_error is not None and not target.done():
            case.controller._signal_stop(session_id=case.session.id)
            if cancel_target:
                target.cancel()  # Cleanup only, never the positive CancelTurn proof.
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(target, return_exceptions=True)), 10
            )
        except BaseException as error:
            errors.append(error)
    if apply is not None:
        for future in (apply.applied, apply.retired):
            try:
                await asyncio.wait_for(asyncio.shield(future), 10)
            except BaseException as error:
                errors.append(error)
    if fence is not None:
        try:
            await asyncio.wait_for(asyncio.shield(fence.applied), 10)
        except BaseException as error:
            errors.append(error)
    if errors:
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("Held control body and original settlement", errors)


def _assert_held_settings_control_owner(case, controls, apply, pending, connections):
    assert pending is not None and pending.effect in controls._effects
    assert (
        pending.source.controls is controls and pending.source.sources is case.sources
    )
    assert pending.native_entered and pending.retirement is not None
    assert not pending.retirement.confirmed
    assert connections and worker_leases(case.database)
    assert not apply.applied.done() and not apply.retired.done()
    assert case.store._fork_source_transitions[case.session.id] > 0
    assert case.store._voice_promotion_mutation_admissions[case.session.id] > 0
    assert not case.store._voice_promotion_mutation_is_admitted_here(case.session.id)


def _assert_held_control_settings_retired(
    case, pending, connections, before, submission_id, request
):
    assert pending.commit is not None and pending.retirement.confirmed
    assert pending.retirement.connection is not None
    assert any(
        connection is pending.retirement.connection for connection in connections
    )
    assert case.session.settings_revision == before + 1
    assert case.session.applied_settings_submission_ids[-1] == submission_id
    assert not case.store._fork_source_transitions
    assert not case.store._voice_promotion_mutation_admissions
    remaining = worker_leases(case.database)
    assert pending.retirement.lease not in remaining
    request.node.user_properties.append(
        (
            "post_write_remaining_leases",
            json.dumps(
                [
                    {
                        "lease_id": id(lease),
                        "thread_id": lease.resource_thread.ident,
                        "thread_name": lease.resource_thread.name,
                        "same_write_lease": lease is pending.retirement.lease,
                        "same_write_creator": lease.resource_thread
                        is pending.retirement.creator,
                        "participant_connections": [
                            id(connection)
                            for connection, live in pending.retirement.participant.connections.items()
                            if live is lease
                        ],
                    }
                    for lease in remaining
                ]
            ),
        )
    )
    row = case.database.get_conversation_by_id(case.session.persisted_conversation_id)
    assert json.loads(row["metadata"])["console_session_settings"]["model"] == "model-a"
    with pytest.raises(sqlite3.ProgrammingError):
        sqlite3.Connection.in_transaction.__get__(pending.retirement.connection)


@pytest.mark.parametrize("domain_case", ["control_settings"], indirect=True)
async def test_typed_cancel_turn_is_live_during_saved_settings_native_return(
    domain_case, request
):
    import pytest_timeout
    from Tests.Chat.test_console_domain_boundary import _selection
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        CancelTurn,
        ControlIdentity,
        Drain,
        FenceSource,
        ReceiveTurn,
        UpdateInputs,
    )
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case = domain_case
    # Prerequisite hunk supplies this ORIGINAL concrete owner before graph creation.
    scope = case.sources.conversation_scope_service
    assert scope is not None and scope.local_service.db is case.database
    assert case.parent.chat_conversation_scope_service is scope
    case.controller.prompt_history = PromptHistory(
        case.database.db_path.parent / "history.jsonl"
    )
    case.sources.configuration["api_settings"] = {
        "openai": {"api_key": "task60-local-readiness-only"}
    }
    case.store.replace_session_settings(
        case.session.id,
        ConsoleSessionSettings(
            provider="openai",
            model="gpt-4o-mini",
            base_url="http://127.0.0.1:1/v1",
        ),
    )
    case.store.replace_session_trace_privacy_override(
        case.session.id,
        capture_enabled=False,
        pii_redaction_enabled=None,
        expected_policy_revision=case.store.capture_policy_state(
            case.session.id
        ).policy_revision,
    )
    await (
        case.runtime._execution_service.ensure_permission_operations().set_kill_switch(
            False
        )
    )
    conversation_id = case.store.persistence.create_conversation(
        conversation_title="Held settings cancel"
    )
    case.store.publish_first_persisted_conversation(case.session.id, conversation_id)
    # Same actual saved Library-policy prerequisite as original Home resume seed.
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
        ConsoleAutoRetrieve,
        ConsoleLibraryPolicyCandidate,
    )

    case.store.persistence.console_library_policy_repository.insert(
        conversation_id,
        ConsoleLibraryPolicyCandidate(
            ConsoleAutoRetrieve.NEVER, ConsoleAssistantLibraryAccess.BLOCKED
        ),
    )
    assert case.database.get_conversation_by_id(conversation_id) is not None
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    sequence = 0

    def identity():
        nonlocal sequence
        sequence += 1
        return ControlIdentity(
            "profile",
            "service",
            sequence,
            f"held-cancel-{sequence}",
            case.session.id,
            case.session.incarnation_id,
            controls.source_revision,
        )

    # Actual original pre-ACK wait: no fake claim, worker, pending plan or ACK.
    updated = controls.dispatch(
        UpdateInputs(identity(), "hold this exact plan", (1, 1))
    )
    received = controls.dispatch(
        ReceiveTurn(
            identity(), "held-cancel-turn", updated.applied.result(), _selection(case)
        )
    )
    record = case.runtime._turn_custody["held-cancel-turn"]
    target = received.work
    apply = fence = pending = None
    body_error = None
    try:
        assert target is record.task and record.received_claim is not None
        assert await _until(lambda: controls.commit_plans or target.done())
        if target.done():
            target.result()  # Preserve actual original preparation refusal/failure.
        assert len(controls.commit_plans) == 1
        plan = controls.commit_plans[0]
        assert plan.turn_id == "held-cancel-turn" and plan.ordinal == 1
        assert plan.recovery.fingerprint == plan.fingerprint
        assert not target.done() and not controls.admitted_effects
        original_owner = case.controller._ordinary_native_commit_owner(
            plan.fingerprint.preparation_id
        )
        assert original_owner is not None and original_owner.task is target
        assert (
            case.store.durable_turn_commit_for(
                plan.fingerprint.preparation_id, fingerprint=plan.fingerprint
            )
            is None
        )
        # Origin captured AFTER actual saved claim/preparation, before Apply.
        submission = _submission(
            case.store,
            case.session.id,
            submission_id="held-cancel-apply",
            model="model-a",
        )
        before = case.session.settings_revision
        with (
            _worker_connections(case.database) as connections,
            _hold_full_writer(case.store.persistence) as (entered, release, calls),
        ):
            try:
                apply = controls.dispatch(ApplySettings(identity(), submission))
                await _entered(entered)
                pending = settings_pending(case.store, case.session.id)
                assert (
                    not pytest_timeout.is_debugging()
                    and float(request.config.getini("timeout")) == 300
                )
                request.node.user_properties.append(
                    ("effective_observer_watchdog", "300/debugger=False")
                )
                _assert_held_settings_control_owner(
                    case, controls, apply, pending, connections
                )
                assert (
                    not target.done() and not release.is_set() and calls == ["model-a"]
                )
                assert case.runtime._turn_custody["held-cancel-turn"] is record
                assert controls._turns["held-cancel-turn"][1] is record
                cancelled = controls.dispatch(
                    CancelTurn(identity(), "held-cancel-turn")
                )
                # No await of settings, no direct target.cancel in the positive path.
                assert cancelled.applied.done() and cancelled.applied.result() is True
                assert cancelled.work is target and (
                    target.cancelling() or target.done()
                )
                assert (
                    not release.is_set()
                    and not apply.applied.done()
                    and not apply.retired.done()
                )
                assert not pending.retirement.confirmed
                fence = controls.dispatch(FenceSource(identity(), "configuration"))
                await asyncio.sleep(0)
                assert not fence.applied.done() and not apply.retired.done()
            except BaseException as error:
                body_error = error
            finally:
                release.set()
                await _settle_held_control_targets(
                    case, body_error, apply, fence, target, cancel_target=True
                )
            assert target.cancelled()  # Exact original typed cancellation settled.
            assert calls == ["model-a"]
            assert (
                apply.retired.result() is True
                and apply.applied.result() is pending.commit
            )
            _assert_held_control_settings_retired(
                case, pending, connections, before, submission.submission_id, request
            )
            assert (
                case.store.durable_turn_commit_for(
                    plan.fingerprint.preparation_id, fingerprint=plan.fingerprint
                )
                is None
            )
    except BaseException as error:
        body_error = error
        raise
    finally:
        if apply is None:
            # Setup failure before full writer issuance still owns its received Task.
            await _settle_held_control_targets(
                case, body_error, None, None, target, cancel_target=True
            )
    assert await controls.dispatch(Drain(identity())).applied
    assert not controls._effects and not controls._tool_threads


@pytest.mark.parametrize("domain_case", ["control_settings"], indirect=True)
async def test_typed_question_decision_is_live_during_saved_settings_native_return(
    domain_case, request
):
    import pytest_timeout
    from tldw_chatbook.Agents.ask_user_questions import validate_questions
    from tldw_chatbook.Chat.console_control_decisions import QuestionDecision
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
        Drain,
        FenceSource,
        ResolveDecision,
    )
    from tldw_chatbook.Chat.console_control_settings import settings_pending

    case = domain_case
    case.store.replace_session_settings(
        case.session.id, _settings("old", system_prompt="owned")
    )
    conversation_id = case.store.persistence.create_conversation(
        conversation_title="Held settings question"
    )
    case.store.publish_first_persisted_conversation(case.session.id, conversation_id)
    assert case.database.get_conversation_by_id(conversation_id) is not None
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    sequence = 0

    def identity():
        nonlocal sequence
        sequence += 1
        return ControlIdentity(
            "profile",
            "service",
            sequence,
            f"held-question-{sequence}",
            case.session.id,
            case.session.incarnation_id,
            controls.source_revision,
        )

    host = case.controller._interrupt_host
    host.retain_decision_target(
        case.session.id
    )  # Original explicit Buddy opener capability.
    assert host._resolve_ask_user_timeout_seconds() == 0
    questions = validate_questions(
        [
            {
                "question": "Choose a result",
                "header": "Choice",
                "options": [{"label": "One"}, {"label": "Two"}],
            }
        ]
    )
    question_threads = []

    def request_questions():
        question_threads.append(threading.current_thread())
        return case.controller.request_user_questions(
            questions, session_id=case.session.id
        )

    target = asyncio.create_task(asyncio.to_thread(request_questions))
    apply = fence = pending = None
    body_error = None
    try:
        assert await _until(lambda: host.pending_question_ids() or target.done())
        assert not target.done()
        request_id = host.pending_question_ids()[0]
        with host.lock:
            state = host.registries["question"][request_id]
            assert (
                state["session_id"] == case.session.id and not state["event"].is_set()
            )
        submission = _submission(
            case.store,
            case.session.id,
            submission_id="held-question-apply",
            model="model-a",
        )
        before = case.session.settings_revision
        with (
            _worker_connections(case.database) as connections,
            _hold_full_writer(case.store.persistence) as (entered, release, calls),
        ):
            try:
                apply = controls.dispatch(ApplySettings(identity(), submission))
                await _entered(entered)
                pending = settings_pending(case.store, case.session.id)
                assert (
                    not pytest_timeout.is_debugging()
                    and float(request.config.getini("timeout")) == 300
                )
                request.node.user_properties.append(
                    ("effective_observer_watchdog", "300/debugger=False")
                )
                _assert_held_settings_control_owner(
                    case, controls, apply, pending, connections
                )
                wrong = controls.dispatch(
                    ResolveDecision(identity(), QuestionDecision("different", []))
                )
                assert wrong.applied.done() and wrong.applied.result() is False
                assert not state["event"].is_set() and not target.done()
                answers = [{"question": "Choose a result", "selected": ["One"]}]
                resolved = controls.dispatch(
                    ResolveDecision(identity(), QuestionDecision(request_id, answers))
                )
                assert resolved.applied.done() and resolved.applied.result() is True
                assert state["event"].is_set()
                result = await asyncio.wait_for(asyncio.shield(target), 10)
                assert result["answered"] and result["answers"][0]["selected"] == [
                    "One"
                ]
                assert request_id not in host.registries["question"]
                assert not release.is_set() and calls == ["model-a"]
                assert (
                    not apply.applied.done()
                    and not apply.retired.done()
                    and not pending.retirement.confirmed
                )
                late = controls.dispatch(
                    ResolveDecision(identity(), QuestionDecision(request_id, answers))
                )
                assert late.applied.done() and late.applied.result() is False
                fence = controls.dispatch(FenceSource(identity(), "configuration"))
                await asyncio.sleep(0)
                assert not fence.applied.done() and not apply.retired.done()
            except BaseException as error:
                body_error = error
            finally:
                release.set()
                await _settle_held_control_targets(
                    case, body_error, apply, fence, target, cancel_target=False
                )
            assert calls == ["model-a"]
            assert (
                apply.retired.result() is True
                and apply.applied.result() is pending.commit
            )
            _assert_held_control_settings_retired(
                case, pending, connections, before, submission.submission_id, request
            )
    except BaseException as error:
        body_error = error
        raise
    finally:
        if apply is None:
            await _settle_held_control_targets(
                case, body_error, None, None, target, cancel_target=False
            )
    assert await controls.dispatch(Drain(identity())).applied
    assert (
        not controls._effects
        and not controls._tool_threads
        and request_id not in host.registries["question"]
    )
    assert target.done() and len(question_threads) == 1
    remaining = worker_leases(case.database)
    request.node.user_properties.append(
        (
            "post_drain_remaining_leases",
            json.dumps(
                [
                    {
                        "lease_id": id(lease),
                        "thread_id": lease.resource_thread.ident,
                        "question_test_thread": lease.resource_thread
                        is question_threads[0],
                        "same_write_lease": lease is pending.retirement.lease,
                    }
                    for lease in remaining
                ]
            ),
        )
    )
    assert all(lease.resource_thread is question_threads[0] for lease in remaining)
    # The plain test-owned Buddy request's legacy thread-local handle belongs
    # to the fixture's DB/executor teardown, not the selected settings effect.


def _saved_order_case(case, title):
    store, session = case.store, case.session
    store.replace_session_settings(session.id, _settings("old", system_prompt="owned"))
    conversation_id = store.persistence.create_conversation(conversation_title=title)
    store.publish_first_persisted_conversation(session.id, conversation_id)
    assert store.active_session_id == session.id and not session.ephemeral
    assert store.messages_for_session(session.id) == []
    return store, session, conversation_id


def _dispatch_order_apply(case, submission):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
    )

    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    identity = ControlIdentity(
        "profile",
        "service",
        1,
        "settings-order-1",
        case.session.id,
        case.session.incarnation_id,
    )
    return controls, controls.dispatch(ApplySettings(identity, submission))


def _assert_original_full_retired(case, controls, receipt, pending, connections):
    import sqlite3

    from tldw_chatbook.Chat.console_control_settings import settings_pending

    commit = receipt.applied.result()
    assert pending.commit is commit and receipt.retired.result() is True
    assert (
        pending.finished.done()
        and pending.finished not in pending.lifecycle.pending_live
    )
    assert pending.effect.completed.result() is True
    assert pending.effect not in controls._effects
    assert settings_pending(case.store, case.session.id) is None
    assert pending.retirement.confirmed
    connection = pending.retirement.connection
    assert connection is not None and any(connection is item for item in connections)
    with pytest.raises(sqlite3.ProgrammingError):
        sqlite3.Connection.in_transaction.__get__(connection)
    assert pending.retirement.lease not in worker_leases(case.database)
    assert not case.store._voice_promotion_mutation_admissions
    assert not case.store._fork_source_transitions
    return commit


@pytest.mark.parametrize("domain_case", ["control_settings"], indirect=True)
async def test_public_restore_and_valid_voice_claim_refuse_during_saved_full_writer(
    domain_case,
    request,
):
    import pytest_timeout

    assert float(request.config.getini("timeout")) == 300

    from Tests.Chat.test_console_speculative_voice_promotion_races import _voice_context
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.Chat.console_voice_promotion import (
        ConsoleVoicePromotionClaimStatus,
    )

    case = domain_case
    store, session, conversation_id = _saved_order_case(case, "Restore and voice order")
    context = _voice_context(store, session.id, promotion_id="settings-order-voice")
    assert context.origin.persisted_conversation_id == conversation_id
    assert context.expected_native_leaf_id is None
    assert context.expected_persisted_leaf_id is None

    def claim_and_abort():
        claim = store.try_claim_voice_promotion(context)
        assert claim.status is ConsoleVoicePromotionClaimStatus.CLAIMED
        assert claim.lease is not None
        try:
            assert claim.lease.destination.persisted_conversation_id == conversation_id
        finally:
            assert store.abort_voice_promotion(claim.lease) is True

    claim_and_abort()
    submission = _submission(store, session.id, submission_id="held", model="model-a")
    original_revision = session.settings_revision
    with (
        _worker_connections(case.database) as connections,
        _hold_full_writer(store.persistence) as (entered, release, calls),
    ):
        controls, receipt = _dispatch_order_apply(case, submission)
        try:
            await _entered(entered)
            assert not pytest_timeout.is_debugging()
            request.node.user_properties.append(
                ("effective_observer_watchdog", "300/debugger=False")
            )
            pending = settings_pending(store, session.id)
            assert pending is not None and pending.effect in controls._effects
            assert pending.commit is None and not pending.retirement.confirmed
            assert connections and worker_leases(case.database)
            assert not receipt.applied.done() and not receipt.retired.done()
            assert calls == ["model-a"]
            assert session.settings.model == "model-a"
            assert session.settings_revision == original_revision
            sessions = store.sessions()
            origin = store.capture_console_settings_origin(session.id)
            current_settings = session.settings

            with pytest.raises(
                RuntimeError,
                match=r"^Voice promotion state prevents store replacement\.$",
            ):
                store.restore_state(sessions=(), active_session_id=None)

            blocked = store.try_claim_voice_promotion(context)
            if blocked.lease is not None:
                # Release an unexpected exact claim before failing the assertion.
                store.abort_voice_promotion(blocked.lease)
            assert (
                blocked.status is ConsoleVoicePromotionClaimStatus.TRANSIENT_CONTENTION
            )
            assert blocked.lease is None
            assert session.id not in store._voice_promotion_leases
            assert session.id not in store._voice_promotion_contexts
            assert store.active_session_id == session.id
            assert all(
                current is prior
                for current, prior in zip(store.sessions(), sessions, strict=True)
            )
            assert store.capture_console_settings_origin(session.id) == origin
            assert session.settings is current_settings
            assert session.settings_revision == original_revision
            assert store.messages_for_session(session.id) == []
            assert not release.is_set() and not pending.finished.done()
        finally:
            release.set()
            async with asyncio.timeout(10):
                await asyncio.shield(
                    asyncio.gather(
                        receipt.applied, receipt.retired, return_exceptions=True
                    )
                )

    commit = _assert_original_full_retired(
        case, controls, receipt, pending, connections
    )
    assert commit.settings.model == "model-a"
    assert session.settings_revision == original_revision + 1
    assert session.applied_settings_submission_ids[-1] == "held"
    claim_and_abort()


@pytest.mark.parametrize("domain_case", ["control_settings"], indirect=True)
async def test_earlier_component_request_waits_for_later_saved_full_writer_post(
    domain_case,
    request,
):
    import pytest_timeout

    assert float(request.config.getini("timeout")) == 300

    import json

    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat import console_settings_persistence as drains
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleSettingsComponent
    from tldw_chatbook.Chat.console_context_policy import ContextCompactionMode
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.Chat.console_generation_settings_metadata import (
        snapshot_from_session_settings,
    )

    case = domain_case
    store, session, conversation_id = _saved_order_case(case, "Component CAS order")
    earlier_submission = _submission(
        store,
        session.id,
        submission_id="earlier",
        model="model-a",
    )
    earlier = store.commit_console_settings_live(earlier_submission)
    assert earlier.accepted_submission is earlier_submission
    assert earlier.persisted_conversation_id == conversation_id
    assert earlier.generation_revision == session.generation_settings_revision
    assert earlier.context_policy_revision == session.context_policy_revision
    later = _submission(
        store,
        session.id,
        submission_id="later",
        model="model-b",
        compaction=ContextCompactionMode.AUTOMATIC,
    )
    components = frozenset(
        {
            ConsoleSettingsComponent.GENERATION_SETTINGS,
            ConsoleSettingsComponent.CONTEXT_POLICY,
        }
    )
    original_revision = session.settings_revision
    pending = None
    component_request = None
    locked = []
    entries = []

    def observe_locked(frame):
        if (
            frame.f_locals.get("self") is store
            and frame.f_locals.get("session_id") == session.id
        ):
            locked.append(
                (
                    frame.f_locals["drain"],
                    pending.retirement.confirmed,
                    pending.finished.done(),
                    pending.effect.completed.done(),
                    pending.commit,
                )
            )

    def observe_writer(frame):
        if (
            frame.f_locals.get("self") is store.persistence
            and frame.f_locals.get("conversation_id") == conversation_id
        ):
            entries.append(
                (
                    frame.f_code,
                    pending.retirement.confirmed,
                    pending.finished.done(),
                    pending.effect.completed.done(),
                    pending.commit,
                )
            )

    generation_code = inspect.unwrap(
        ChatPersistenceService.update_conversation_generation_settings
    ).__code__
    context_code = inspect.unwrap(
        ChatPersistenceService.update_conversation_context_policy
    ).__code__
    event = sys.monitoring.events.PY_START
    with (
        _observe(
            drains._persist_console_settings_components_locked.__code__,
            event,
            observe_locked,
        ),
        _observe(generation_code, event, observe_writer),
        _observe(context_code, event, observe_writer),
    ):
        with (
            _worker_connections(case.database) as connections,
            _hold_full_writer(store.persistence) as (entered, release, calls),
        ):
            controls, receipt = _dispatch_order_apply(case, later)
            try:
                await _entered(entered)
                assert not pytest_timeout.is_debugging()
                request.node.user_properties.append(
                    ("effective_observer_watchdog", "300/debugger=False")
                )
                pending = settings_pending(store, session.id)
                assert pending is not None and pending.effect in controls._effects
                assert pending.commit is None and not pending.retirement.confirmed
                assert connections and worker_leases(case.database)
                assert calls == ["model-b"]
                assert session.settings.model == "model-b"
                assert session.settings_revision == original_revision
                assert (
                    session.context_policy_revision == earlier.context_policy_revision
                )

                component_request = asyncio.create_task(
                    store.persist_console_settings_commit_serialized(earlier)
                )
                await asyncio.sleep(
                    0
                )  # Let the public component_request register its actual drain.
                drain = pending.lifecycle.drain
                assert drain is not None and drain.task in pending.lifecycle.tasks
                assert drain.generation_revision == earlier.generation_revision
                assert drain.context_policy_revision == earlier.context_policy_revision
                await asyncio.sleep(0)  # Give the registered drain its first loop turn.
                assert pending.finished in pending.lifecycle.pending_live
                assert not component_request.done() and not drain.task.done()
                assert not receipt.applied.done() and not receipt.retired.done()
                assert not locked and not entries and not release.is_set()
            finally:
                release.set()
                targets = [receipt.applied, receipt.retired]
                if component_request is not None:
                    targets.append(component_request)
                async with asyncio.timeout(10):
                    await asyncio.shield(
                        asyncio.gather(*targets, return_exceptions=True)
                    )

        newest = _assert_original_full_retired(
            case, controls, receipt, pending, connections
        )
        assert session.settings_revision == original_revision + 1
        assert newest.generation_revision > earlier.generation_revision
        assert newest.context_policy_revision > earlier.context_policy_revision
        older_outcome = component_request.result()
        assert older_outcome.stale_components == components
        assert (
            not older_outcome.written_components and not older_outcome.failed_components
        )
        assert drain.task.done() and drain.task.result() is older_outcome
        assert locked == [(drain, True, True, True, newest)]
        assert not entries

        newest_request = asyncio.create_task(
            store.persist_console_settings_commit_serialized(newest)
        )
        try:
            async with asyncio.timeout(10):
                newest_outcome = await asyncio.shield(newest_request)
        finally:
            if not newest_request.done():
                async with asyncio.timeout(10):
                    await asyncio.shield(
                        asyncio.gather(newest_request, return_exceptions=True)
                    )
        assert newest_outcome.written_components == components
        assert (
            not newest_outcome.stale_components and not newest_outcome.failed_components
        )
        assert entries == [
            (generation_code, True, True, True, newest),
            (context_code, True, True, True, newest),
        ]

    expected_generation = snapshot_from_session_settings(newest.settings)
    durable_generation = store.persistence.get_conversation_generation_settings(
        conversation_id
    )
    durable_policy = store.persistence.get_conversation_context_policy(conversation_id)
    assert (
        durable_generation.snapshot
        == session.generation_durable_snapshot
        == expected_generation
    )
    assert durable_policy.overrides == newest.context_policy_overrides
    assert durable_policy.revision is not None
    assert durable_policy.revision == session.context_policy_durable_revision
    row = case.database.get_conversation_by_id(conversation_id)
    assert json.loads(row["metadata"])["console_session_settings"]["model"] == "model-b"
