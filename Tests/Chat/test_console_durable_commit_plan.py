"""Causal controls across the real local reserve/execute boundary."""

# ruff: noqa: F811 -- imported pytest fixture is injected by its original name.

from __future__ import annotations

import asyncio
import sys
import sqlite3
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace

import pytest

from Tests.Chat.test_console_domain_boundary import domain_case  # noqa: F401
from Tests.Chat.test_console_durable_turn_acceptance import (
    _authority,
    _context,
    _ready_store,
)
from Tests.Chat.test_console_first_send_atomicity import _until
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDurableTurnAcceptance,
    ConsoleDispatchReconstructability,
)
from tldw_chatbook.Chat.console_durable_commit_plan import (
    bind_durable_plan_source,
    capture_durable_plan_source,
)
from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicyHolder
from tldw_chatbook.Chat.console_native_commit import (
    reserve_durable_commit_plan_owned,
    commit_reserved_durable_turn_owned,
)
from tldw_chatbook.Chat.console_received_dispatch import received_preparation_source
from tldw_chatbook.Chat.console_received_turn import bind_received_turn_claim
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent
from Tests.Chat.test_console_domain_boundary import _selection
from Tests.Chat import test_console_received_intent_transitions as received_controls

catalog_store = received_controls.catalog_store
local_root = received_controls.local_root
mcp_sources = received_controls.mcp_sources
snapshot_case = received_controls.snapshot_case
runtime_case = received_controls.runtime_case
configuration_workspace = received_controls.configuration_workspace
received_workspace = received_controls.received_workspace

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _closed_corruption_statement(db, sql, params=()):
    """Close each statement inside the caller's disposable corruption scope."""
    connection = db.get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params)
    finally:
        cursor.close()


def saved_legacy_plan(store, acceptance):
    """Observe the real synchronous composition, without opening a bypass."""
    plans = []
    code = store.reserve_durable_commit_plan.__func__.__code__
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is code
            and frame.f_locals.get("self") is store
        ):
            plans.append(arg)

    sys.setprofile(observe)
    try:
        commit = store.commit_durable_turn(acceptance)
    finally:
        sys.setprofile(previous)
    assert len(plans) == 1
    return commit, plans[0]


def settle_original_pair(
    repository, checkpoint, *, state="complete", continuation=None
):
    from Tests.Chat.test_console_dispatch_recovery import TERMINAL_RECEIPT
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleAssistantSettlement,
    )
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.Chat.provider_continuation import dump_provider_continuation_json

    receipt = TERMINAL_RECEIPT if state == "complete" else None
    result = repository.settle_with_assistant(
        ConsoleAssistantSettlement(
            assistant_message_id=checkpoint.assistant_message_id,
            expected_checkpoint_state=checkpoint.state,
            expected_checkpoint_revision=checkpoint.checkpoint_revision,
            expected_user_message_version=checkpoint.user_message_version,
            expected_assistant_message_version=checkpoint.assistant_message_version,
            terminal_state=state,
            content="answer",
            metadata_json=MessageMetadata(
                engine="task58", terminal_receipt_id=receipt
            ).to_json()
            if receipt
            else None,
            terminal_receipt_id=receipt,
            provider_continuation_json=dump_provider_continuation_json(continuation)
            if continuation
            else None,
        )
    )
    assert result.status.value == "committed"
    return result


async def test_original_received_driver_binds_and_resets_selected_source(
    domain_case, monkeypatch
):
    from tldw_chatbook.Chat import console_received_dispatch as dispatch

    case = domain_case
    async with prepared(case) as (acceptance, record):
        outer = capture_durable_plan_source(case.store).binding
        source = received_preparation_source(case.runtime)

        async def prepared_driver(runtime, received, captured):
            receipt = capture_durable_plan_source(case.store)
            assert receipt.source is source and receipt.binding is not outer
            assert receipt.record is record and receipt.task is asyncio.current_task()
            plan = await reserve_durable_commit_plan_owned(case.store, acceptance)
            return await commit_reserved_durable_turn_owned(plan)

        monkeypatch.setattr(dispatch, "_run_received_intent_bound", prepared_driver)
        result = await dispatch.run_received_intent(case.runtime, record, source)
        assert (
            result.commit is not None
            and result.error is None
            and result.native_retired is True
        )
        assert capture_durable_plan_source(case.store).binding is outer


@pytest.mark.parametrize("changed", ["intent", "claim"])
async def test_reservation_refuses_replacement_of_original_bound_receipt(
    domain_case, changed
):
    case = domain_case
    async with prepared(case) as (acceptance, record):
        name = "received_intent" if changed == "intent" else "received_claim"
        original = getattr(record, name)
        setattr(record, name, replace(original))
        try:
            with pytest.raises(RuntimeError, match="original received task"):
                await reserve_durable_commit_plan_owned(case.store, acceptance)
            assert not case.store._durable_commit_in_flight
            assert not case.store._durable_commit_plans
        finally:
            setattr(record, name, original)


async def test_original_empty_queue_reroute_anchors_claim_through_promotion(
    received_workspace,
    tmp_path,
):
    from Tests.Chat.test_console_prompt_queue_coordinator import SequencedGateway
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.console_received_turn import (
        ConsoleReceivedTurnAdmissionMixin,
    )

    case = received_workspace
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case.controller.prompt_history = PromptHistory(tmp_path / "reroute-history.jsonl")
    case.store.replace_session_settings(
        case.session.id,
        ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
    )
    gateway = SequencedGateway()
    for event in gateway.release:
        event.set()
    case.controller.provider_gateway = gateway
    case.store.replace_session_trace_privacy_override(
        case.session.id,
        capture_enabled=False,
        pii_redaction_enabled=None,
        expected_policy_revision=case.store.capture_policy_state(
            case.session.id
        ).policy_revision,
    )
    case.store.set_session_draft(case.session.id, "rerouted original input")
    intent = received_controls._stock_intent(
        case,
        queue_revision=case.controller.prompt_queue_registry.snapshot(
            case.session.id
        ).revision,
    )
    claims, promoted = [], []
    claim_code = ConsoleReceivedTurnAdmissionMixin.claim_received_turn.__code__
    promotion_code = case.controller._begin_submit_preparation.__func__.__code__

    def observe(frame, event, value):
        if event != "return" or value is None:
            return
        if frame.f_code is claim_code and frame.f_locals.get("self") is case.store:
            claims.append(value)
        if (
            frame.f_code is promotion_code
            and frame.f_locals.get("self") is case.controller
        ):
            receipt = capture_durable_plan_source(case.store)
            assert case.store.preparation_for_session(case.session.id) is value
            assert (
                case.controller._active_submit_preparations[receipt.task]
                == value.preparation_id
            )
            assert frame.f_locals["claim"] is receipt.claim
            promoted.append((receipt, value))

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        case.runtime.accept_received_intent(intent)
        record = case.runtime._turn_custody[intent.turn_id]
        issuing_task = record.task
        assert record.received_claim is None
        result = await received_controls._outcome(record)
    finally:
        sys.setprofile(previous)
    if isinstance(result, BaseException):
        raise result
    assert result.accepted
    assert len(claims) == len(promoted) == 1
    receipt, preparation = promoted[0]
    assert receipt.claim is claims[0] and receipt.intent is intent
    assert receipt.record is record and receipt.task is issuing_task
    assert issuing_task.done() and record.task is None
    with pytest.raises(RuntimeError, match="original received task"):
        capture_durable_plan_source(case.store)


async def test_separated_projection_is_bounded_without_changing_legacy_save(
    domain_case, tmp_path, owned_console_databases
):
    case = domain_case
    attachments = tuple(
        {
            "position": n,
            "data": b"image",
            "mime_type": "image/png",
            "display_name": "image.png",
        }
        for n in range(6)
    )
    async with prepared(case) as (acceptance, record):
        with pytest.raises(RuntimeError, match="projection exceeds"):
            await reserve_durable_commit_plan_owned(
                case.store, replace(acceptance, attachments=attachments)
            )
        assert not case.store._durable_commit_in_flight
        assert (
            case.database.get_connection()
            .execute("SELECT count(*) FROM messages")
            .fetchone()[0]
            == 0
        )
    db, service, store, preparation, acceptance = _ready_store(tmp_path)
    owned_console_databases(db)
    acceptance = replace(acceptance, attachments=attachments)
    with pytest.raises(RuntimeError, match="preparation-read custody"):
        store.reserve_durable_commit_plan(acceptance)
    commit, plan = saved_legacy_plan(store, acceptance)
    assert commit is not None and plan.recovery is None


@pytest.mark.parametrize("change", ["stock", "custom", "fingerprint", "write"])
async def test_selected_contributions_keep_original_finite_definitions(
    domain_case, monkeypatch, change
):
    from tldw_chatbook.Chat.library_preparation import (
        LibraryPreparationContribution,
        LibraryPreparationEvent,
    )
    from tldw_chatbook.Chat.console_prompt_mixin_service import MixinAttemptContribution
    from tldw_chatbook.Chat.console_prompt_mixins import (
        MixinAttemptWrite,
        MixinAttemptSnapshot,
    )

    calls = []

    @dataclass(frozen=True)
    class Custom:
        def durable_acceptance_fingerprint(self):
            calls.append("custom")
            return "custom"

    async with prepared(domain_case) as (acceptance, record):
        contributions = (
            LibraryPreparationContribution(
                LibraryPreparationEvent(1, "zero_matches", acceptance.attempt_id, 0, ())
            ),
            MixinAttemptContribution(
                MixinAttemptWrite(
                    acceptance.attempt_id,
                    acceptance.user_message_id,
                    acceptance.assistant_message_id,
                    MixinAttemptSnapshot(),
                )
            ),
        )
        if change == "custom":
            with pytest.raises(RuntimeError, match="legacy profile"):
                await reserve_durable_commit_plan_owned(
                    domain_case.store, replace(acceptance, contributions=(Custom(),))
                )
            assert not calls and not domain_case.store._durable_commit_in_flight
            return
        plan = await reserve_durable_commit_plan_owned(
            domain_case.store, replace(acceptance, contributions=contributions)
        )
        if change == "fingerprint":
            monkeypatch.setattr(
                MixinAttemptContribution,
                "durable_acceptance_fingerprint",
                lambda self: "changed",
            )
        elif change == "write":
            monkeypatch.setattr(
                LibraryPreparationContribution,
                "write",
                lambda self, **kwargs: calls.append("write"),
            )
        result = await commit_reserved_durable_turn_owned(plan)
        if change == "stock":
            assert (
                result.commit is not None
                and result.error is None
                and result.native_retired is True
            )
        else:
            assert result.commit is None and "contribution source changed" in str(
                result.error
            )
            assert not calls
            assert (
                domain_case.database.get_connection()
                .execute("SELECT count(*) FROM messages")
                .fetchone()[0]
                == 0
            )


@asynccontextmanager
async def prepared(case):
    store, session, controller, runtime = (
        case.store,
        case.session,
        case.controller,
        case.runtime,
    )
    task = asyncio.current_task()
    session.settings = replace(session.settings, provider="openai", model="test-model")
    authority = _authority()
    session.library_policy_holder = ConsoleLibraryPolicyHolder(authority.policy)
    store.set_session_draft(session.id, "exact captured draft")
    launch, revision, _ = runtime.snapshot_console_staged_evidence(session.id)
    intent = ConsoleReceivedTurnIntent(
        turn_id="plan-turn",
        session_id=session.id,
        inputs=store.session_input_snapshot(session.id),
        selection=_selection(case),
        staged_evidence_launch=launch,
        staged_evidence_revision=revision,
        view_attachment_generation=runtime._attached_generation or 0,
    )
    claim = store.claim_received_turn(
        session.id, intent.turn_id, draft_revision=intent.inputs.draft_revision
    )
    record = runtime._register_custody(
        None, store=store, received_claim=claim, received_intent=intent
    )
    record.task = task
    controller._register_submit_task(task, session.id)
    source = received_preparation_source(runtime)
    echo = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content=intent.inputs.draft,
        persist=False,
    )
    preparation = ConsoleTurnPreparation(
        preparation_id="plan-preparation",
        attempt_id="attempt-1",
        session_id=session.id,
        origin="manual",
        queue_entry_id=None,
        executed_draft=intent.inputs.draft,
        execution_context=_context(session.id, authority),
        transient_user_message_id=echo.id,
        attachment_ids=(),
        evidence_ids=(),
        prefill_id=None,
        queue_generation=None,
        pre_send_title=session.title,
        pre_send_conversation_id=None,
        state=ConsoleTurnPreparationState.COMMITTING,
        pause_kind=None,
        one_shot_bypass=False,
        ephemeral=False,
    )
    with (
        bind_received_turn_claim(store, claim),
        bind_durable_plan_source(runtime, record, source),
    ):
        preparation = controller._begin_submit_preparation(task, preparation)
        identity = store.stage_durable_turn_identity(
            session.id, preparation.preparation_id, title="Sealed title"
        )
        acceptance = ConsoleDurableTurnAcceptance(
            conversation_id=identity.conversation_id,
            user_message_id=echo.id,
            assistant_message_id="plan-assistant",
            parent_message_id=None,
            user_content=intent.inputs.draft,
            attachments=(
                {
                    "position": 0,
                    "data": b"image",
                    "mime_type": "image/png",
                    "display_name": "image.png",
                },
            ),
            preparation_id=preparation.preparation_id,
            attempt_id=preparation.attempt_id,
            origin="manual",
            queue_entry_id=None,
            frozen_authority=authority,
            resolved_destination=_context(session.id, authority).resolved_destination,
            reconstructability=ConsoleDispatchReconstructability(
                True, True, True, "opaque:plan"
            ),
            contributions=(),
        )
        try:
            yield acceptance, record
        finally:
            controller._unregister_submit_task(task)
            runtime._release_custody(record.turn_id)


async def test_plan_seals_inputs_without_native_admission_then_commits_once(
    domain_case, monkeypatch
):
    case = domain_case
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_preparation_reads import (
        preparation_read_for,
        drain_preparation_reads,
    )

    await asyncio.wait_for(drain_preparation_reads(case.runtime._preparation_reads), 8)

    async with prepared(case) as (acceptance, record):
        admissions = []
        acquire = storage.acquire_storage

        def observe(*args, **kwargs):
            admissions.append(threading.get_ident())
            return acquire(*args, **kwargs)

        monkeypatch.setattr(storage, "acquire_storage", observe)
        reads = []
        code = case.store.reserve_durable_commit_plan.__func__.__code__

        def profile(frame, event, arg):
            if (
                event == "call"
                and frame.f_code is code
                and frame.f_locals["self"] is case.store
            ):
                read = preparation_read_for(case.runtime)
                assert read in case.runtime._preparation_reads
                assert read in case.controller._preparation_reads
                reads.append(read)

        previous, threads = sys.getprofile(), threading.getprofile()
        threading.setprofile_all_threads(profile)
        try:
            plan = await reserve_durable_commit_plan_owned(case.store, acceptance)
        finally:
            threading.setprofile_all_threads(threads)
            sys.setprofile(previous)
        assert admissions == []
        assert (
            plan.received_acceptance is acceptance and plan.acceptance is not acceptance
        )
        assert plan.reservation.owner_thread_id != threading.get_ident()
        assert len(reads) == 1
        read = reads[0]
        assert read.callback_returned and read.result is plan
        assert read.retired.done() and read._producer.done()
        assert read not in case.runtime._preparation_reads
        assert read not in case.controller._preparation_reads
        with pytest.raises(TypeError):
            plan.conversation_kwargs["conversation_title"] = "mutated"
        acceptance.attachments[0]["data"] = b"changed after reserve"
        assert plan.acceptance.attachments[0]["data"] == b"image"
        completion = await commit_reserved_durable_turn_owned(plan)
        assert completion.error is None, repr(completion.error)
        assert completion.commit.identity.title == "Sealed title"
        assert admissions
        assert not case.store._native_commit_owners_by_preparation
        assert not case.store._durable_plan_executing
        result = case.store.persistence.console_dispatch_repository.read_exact_commit_outcome(
            plan
        )
        assert result.status == "matching_durable_acceptance_observed"
        assert result.full_fingerprint_verified
        assert (
            case.database.get_connection()
            .execute("SELECT count(*) FROM messages")
            .fetchone()[0]
            == 2
        )


@pytest.mark.parametrize(
    "change",
    [
        "configuration",
        "settings",
        "session",
        "preparation",
        "task",
        "intent",
        "source_receipt",
        "callback",
        "settings_owner",
    ],
)
async def test_waited_plan_refuses_changed_original_source(domain_case, change):
    case = domain_case
    async with prepared(case) as (acceptance, record):
        plan = await reserve_durable_commit_plan_owned(case.store, acceptance)
        restore = lambda: None
        if change == "configuration":
            original = case.domain.sources
            case.domain.sources = replace(original, configuration={})
            restore = lambda: setattr(case.domain, "sources", original)
        elif change == "settings":
            case.session.settings_revision += 1
        elif change == "settings_owner":
            case.session.settings = replace(case.session.settings)
        elif change == "session":
            case.store._sessions[case.session.id] = replace(case.session)
            restore = lambda: case.store._sessions.__setitem__(
                case.session.id, case.session
            )
        elif change == "preparation":
            replacement = replace(
                plan.preparation,
                attempt_id="different-attempt",
                execution_context=replace(
                    plan.preparation.execution_context,
                    library_authority=replace(
                        plan.preparation.execution_context.library_authority,
                        attempt_id="different-attempt",
                    ),
                ),
            )
            case.store._preparations_by_id[acceptance.preparation_id] = replacement
            case.store._preparations_by_session[case.session.id] = replacement
        elif change == "task":
            case.controller._active_submit_preparations[asyncio.current_task()] = (
                "other-preparation"
            )
        elif change == "source_receipt":
            plan.source.source.config_identity = object()
        elif change == "callback":
            case.store.persistence.commit_durable_turn = lambda **kwargs: pytest.fail(
                "changed callback entered"
            )
            restore = lambda: delattr(case.store.persistence, "commit_durable_turn")
        else:
            record.received_intent = replace(
                record.received_intent, turn_id="other-turn"
            )
        try:
            result = await commit_reserved_durable_turn_owned(plan)
            assert result.commit is None and result.error is not None
            assert (
                case.database.get_connection()
                .execute("SELECT count(*) FROM messages")
                .fetchone()[0]
                == 0
            )
        finally:
            restore()


async def test_copied_child_context_cannot_reserve_and_abandon_is_exact(domain_case):
    case = domain_case
    async with prepared(case) as (acceptance, record):

        async def copied():
            with pytest.raises(RuntimeError, match="original received task"):
                capture_durable_plan_source(case.store)

        await asyncio.create_task(copied())
        plan = await reserve_durable_commit_plan_owned(case.store, acceptance)
        with pytest.raises(RuntimeError, match="already in flight"):
            await reserve_durable_commit_plan_owned(case.store, acceptance)
        assert not case.store.abandon_reserved_durable_commit(replace(plan), "foreign")
        assert case.store.abandon_reserved_durable_commit(plan, "cancelled")
        assert not case.store.abandon_reserved_durable_commit(plan, "again")
        assert (
            case.store.staged_durable_turn_identity_for(acceptance.preparation_id)
            is plan.identity
        )
        assert (
            case.store.durable_acceptance_fingerprint_for(acceptance.preparation_id)
            == plan.fingerprint
        )
        completion = await commit_reserved_durable_turn_owned(plan)
        assert completion.commit is None and completion.error is not None


async def test_executing_plan_cancellation_joins_exact_native_owner(
    domain_case, monkeypatch
):
    case = domain_case
    from tldw_chatbook.Chat.console_preparation_reads import drain_preparation_reads

    await asyncio.wait_for(drain_preparation_reads(case.runtime._preparation_reads), 8)
    async with prepared(case) as (acceptance, record):
        plan = await reserve_durable_commit_plan_owned(case.store, acceptance)
        entered, release, exited = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original = case.store.persistence.commit_durable_turn
        connections = []
        statements = []

        def held(frame, event, arg):
            if (
                frame.f_code is original.__func__.__code__
                and frame.f_locals.get("self") is case.store.persistence
            ):
                if event == "call":
                    connections.append(case.database.get_connection())
                    connections[-1].set_trace_callback(statements.append)
                    entered.set()
                    assert release.wait(10)
                elif event == "return":
                    exited.set()

        previous, threads = sys.getprofile(), threading.getprofile()
        threading.setprofile_all_threads(held)
        task = asyncio.create_task(commit_reserved_durable_turn_owned(plan))
        try:
            assert await _until(entered.is_set, timeout=5)
            owner = case.controller._ordinary_native_commit_owner(
                acceptance.preparation_id
            )
            assert owner.task is task and not owner.retired.done()
            assert len(connections) == 1
            assert case.database._connection_quiescence.is_registered(connections[0])
            assert not case.store.abandon_reserved_durable_commit(plan, "too_late")
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                assert (
                    not task.done() and not exited.is_set() and not owner.retired.done()
                )
            release.set()
            result = await asyncio.wait_for(asyncio.shield(task), 5)
            assert result.caller_cancelled and result.error is None
            assert result.native_retired is True
            assert exited.is_set() and owner.retired.done()
            assert not case.database._connection_quiescence.is_registered(
                connections[0]
            )
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                sqlite3.Connection.in_transaction.__get__(connections[0])
            assert (
                sum(sql.strip().upper() == "BEGIN IMMEDIATE" for sql in statements) == 1
            )
            assert sum(sql.strip().upper() == "COMMIT" for sql in statements) == 1
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            threading.setprofile_all_threads(threads)
            sys.setprofile(previous)


@pytest.mark.parametrize(
    "state",
    [
        "checkpoint",
        "terminal",
        "absent",
        "partial",
        "deleted",
        "digest",
        "parent",
        "inactive",
        "attachment",
        "discarded",
        "role",
        "duplicate",
        "receipt",
        "read_error",
    ],
)
async def test_exact_outcome_reads_original_sqlite_pair(
    tmp_path, owned_console_databases, monkeypatch, state
):
    _raw_semantic_corruption = _closed_corruption_statement

    db, service, store, preparation, acceptance = _ready_store(tmp_path)
    owned_console_databases(db)
    commit, plan = saved_legacy_plan(store, acceptance)
    with pytest.raises(RuntimeError, match="original synchronous call"):
        store.commit_reserved_durable_turn(plan)
    if state in {"terminal", "discarded"}:
        settle_original_pair(
            service.console_dispatch_repository,
            commit.checkpoint,
            state="discarded" if state == "discarded" else "complete",
        )
    connection = db.get_connection()
    connection.create_function(
        "console_semantic_mutation_authorized", 2, lambda *_args: 1
    )
    try:
        if state in {"absent", "partial"}:
            assert not connection.in_transaction
            connection.execute("PRAGMA foreign_keys=OFF")
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        if state == "absent":
            with db.transaction() as cursor:
                cursor.execute("DELETE FROM console_dispatch_checkpoints")
                _raw_semantic_corruption(
                    db,
                    "DELETE FROM messages WHERE conversation_id=?",
                    (acceptance.conversation_id,),
                )
        else:
            assert store.commit_durable_turn(acceptance) is commit
            with db.transaction() as cursor:
                if state == "partial":
                    cursor.execute("DELETE FROM console_dispatch_checkpoints")
                    _raw_semantic_corruption(
                        db,
                        "UPDATE messages SET assistant_generation_state=? WHERE id=?",
                        (
                            "discarded" if state == "discarded" else "complete",
                            acceptance.assistant_message_id,
                        ),
                    )
                if state == "partial":
                    _raw_semantic_corruption(
                        db,
                        "DELETE FROM messages WHERE id=?",
                        (acceptance.assistant_message_id,),
                    )
                if state == "deleted":
                    _raw_semantic_corruption(
                        db,
                        "UPDATE messages SET deleted=1 WHERE id=?",
                        (acceptance.user_message_id,),
                    )
                if state == "digest":
                    _raw_semantic_corruption(
                        db,
                        "UPDATE messages SET content='changed' WHERE id=?",
                        (acceptance.user_message_id,),
                    )
                if state == "parent":
                    _raw_semantic_corruption(
                        db,
                        "UPDATE messages SET parent_message_id=NULL WHERE id=?",
                        (acceptance.assistant_message_id,),
                    )
                if state == "inactive":
                    cursor.execute(
                        "UPDATE conversations SET active_leaf_message_id=NULL"
                    )
                if state == "attachment":
                    _raw_semantic_corruption(
                        db, "UPDATE message_attachments SET data=?", (b"changed",)
                    )
                if state == "role":
                    _raw_semantic_corruption(
                        db,
                        "UPDATE messages SET role='assistant' WHERE id=?",
                        (acceptance.user_message_id,),
                    )
                if state == "duplicate":
                    duplicate = dict(
                        cursor.execute(
                            "SELECT * FROM console_dispatch_checkpoints"
                        ).fetchone()
                    )
                    duplicate.update(
                        assistant_message_id=acceptance.user_message_id,
                        preparation_id="conflicting-preparation",
                    )
                    cursor.execute(
                        "INSERT INTO console_dispatch_checkpoints ("
                        + ",".join(duplicate)
                        + ") VALUES ("
                        + ",".join("?" for _ in duplicate)
                        + ")",
                        tuple(duplicate.values()),
                    )
    finally:
        db.close_connection()
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    reader_connection = db.get_connection()
    assert reader_connection is not connection
    assert reader_connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    expected = plan.recovery
    if state == "receipt":
        expected = replace(
            expected, checkpoint=replace(commit.checkpoint, checkpoint_revision=99)
        )
    if state == "read_error":

        def failed_read(**kwargs):
            raise OSError("injected admitted read failure")

        monkeypatch.setattr(db, "transaction", failed_read)
    result = service.console_dispatch_repository.read_exact_commit_outcome(expected)
    assert not result.full_fingerprint_verified
    assert result.status == (
        "matching_durable_acceptance_observed"
        if state in {"checkpoint", "inactive"}
        else "matching_pair_terminal_observed"
        if state in {"terminal", "discarded"}
        else "no_matching_durable_rows_observed"
        if state == "absent"
        else "conflict_or_unknown"
    )


@pytest.mark.parametrize(
    "state", ["active", "inactive", "invalid", "destination", "version"]
)
async def test_exact_outcome_uses_original_continuation_handoff(
    tmp_path, owned_console_databases, state
):
    from Tests.Chat.test_console_dispatch_continuation_handoff import (
        _continuation,
        _deepseek_acceptance,
    )
    from Tests.Chat.test_console_dispatch_recovery import (
        _start,
        _raw_semantic_corruption,
    )
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleContinuationHandoff,
        ConsoleDispatchResultStatus,
    )
    from tldw_chatbook.Chat.provider_continuation import dump_provider_continuation_json

    db, service, store, preparation, acceptance = _ready_store(tmp_path)
    owned_console_databases(db)
    template = _deepseek_acceptance(acceptance.conversation_id)
    acceptance = replace(
        acceptance,
        resolved_destination=template.resolved_destination,
        frozen_authority=replace(
            acceptance.frozen_authority,
            provider_intent=template.frozen_authority.provider_intent,
        ),
    )
    committed, plan = saved_legacy_plan(store, acceptance)
    repository = service.console_dispatch_repository
    started = _start(repository, committed.checkpoint)
    result = repository.handoff_to_provider_continuation(
        ConsoleContinuationHandoff(
            assistant_message_id=started.assistant_message_id,
            expected_checkpoint_revision=started.checkpoint_revision,
            expected_user_message_version=started.user_message_version,
            expected_assistant_message_version=started.assistant_message_version,
            provider_continuation_json=dump_provider_continuation_json(_continuation()),
        )
    )
    assert result.status is ConsoleDispatchResultStatus.COMMITTED
    expected = replace(plan.recovery, checkpoint=committed.checkpoint)
    if state == "inactive":
        with db.transaction() as cursor:
            cursor.execute("UPDATE conversations SET active_leaf_message_id=NULL")
    elif state == "invalid":
        _raw_semantic_corruption(
            db,
            "UPDATE messages SET provider_continuation_json='{}' WHERE id=?",
            (acceptance.assistant_message_id,),
        )
    elif state == "destination":
        expected = replace(
            expected,
            resolved_destination_json=expected.resolved_destination_json.replace(
                "deepseek-v4-flash", "different-model"
            ),
        )
    elif state == "version":
        _raw_semantic_corruption(
            db,
            "UPDATE messages SET version=0 WHERE id=?",
            (acceptance.assistant_message_id,),
        )
    result = repository.read_exact_commit_outcome(expected)
    assert result.status == (
        "matching_pair_continuation_observed"
        if state == "active"
        else "conflict_or_unknown"
    )
    assert not result.full_fingerprint_verified


@pytest.mark.parametrize("kind", ["terminal", "continuation"])
@pytest.mark.parametrize("retained_receipt", [False, True])
async def test_checkpoint_cleared_pair_requires_original_version_advancement(
    tmp_path, owned_console_databases, kind, retained_receipt
):
    from Tests.Chat.test_console_dispatch_continuation_handoff import (
        _continuation,
        _deepseek_acceptance,
    )
    from Tests.Chat.test_console_dispatch_recovery import (
        _start,
        _raw_semantic_corruption,
    )
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleContinuationHandoff,
    )
    from tldw_chatbook.Chat.provider_continuation import dump_provider_continuation_json

    db, service, store, preparation, acceptance = _ready_store(tmp_path)
    owned_console_databases(db)
    template = _deepseek_acceptance(acceptance.conversation_id)
    acceptance = replace(
        acceptance,
        resolved_destination=template.resolved_destination,
        frozen_authority=replace(
            acceptance.frozen_authority,
            provider_intent=template.frozen_authority.provider_intent,
        ),
    )
    commit, plan = saved_legacy_plan(store, acceptance)
    repository = service.console_dispatch_repository
    started = _start(repository, commit.checkpoint)
    if kind == "terminal":
        settle_original_pair(repository, started)
    else:
        result = repository.handoff_to_provider_continuation(
            ConsoleContinuationHandoff(
                assistant_message_id=started.assistant_message_id,
                expected_checkpoint_revision=started.checkpoint_revision,
                expected_user_message_version=started.user_message_version,
                expected_assistant_message_version=started.assistant_message_version,
                provider_continuation_json=dump_provider_continuation_json(
                    _continuation()
                ),
            )
        )
        assert result.status.value == "committed"
    expected = (
        replace(plan.recovery, checkpoint=started)
        if retained_receipt
        else plan.recovery
    )
    assert repository.read_exact_commit_outcome(expected).status == (
        "matching_pair_terminal_observed"
        if kind == "terminal"
        else "matching_pair_continuation_observed"
    )
    _raw_semantic_corruption(
        db,
        "UPDATE messages SET version=? WHERE id=?",
        (
            started.assistant_message_version
            if retained_receipt
            else expected.initial_assistant_version,
            acceptance.assistant_message_id,
        ),
    )
    result = repository.read_exact_commit_outcome(expected)
    assert result.status == "conflict_or_unknown"
    assert result.detail == "checkpoint_not_advanced"


@pytest.mark.parametrize("mismatched", [False, True])
async def test_complete_continuation_obeys_original_terminal_content_contract(
    tmp_path, owned_console_databases, mismatched
):
    from Tests.Chat.test_console_dispatch_continuation_handoff import (
        _continuation,
        _deepseek_acceptance,
    )
    from Tests.Chat.test_console_dispatch_recovery import (
        _start,
        _raw_semantic_corruption,
    )

    db, service, store, preparation, acceptance = _ready_store(tmp_path)
    owned_console_databases(db)
    template = _deepseek_acceptance(acceptance.conversation_id)
    acceptance = replace(
        acceptance,
        resolved_destination=template.resolved_destination,
        frozen_authority=replace(
            acceptance.frozen_authority,
            provider_intent=template.frozen_authority.provider_intent,
        ),
    )
    commit, plan = saved_legacy_plan(store, acceptance)
    repository = service.console_dispatch_repository
    started = _start(repository, commit.checkpoint)
    complete = _continuation(state="complete", call_state="completed", content="answer")
    settle_original_pair(repository, started, continuation=complete)
    if mismatched:
        _raw_semantic_corruption(
            db,
            "UPDATE messages SET content='different answer' WHERE id=?",
            (acceptance.assistant_message_id,),
        )
    result = repository.read_exact_commit_outcome(
        replace(plan.recovery, checkpoint=started)
    )
    assert result.status == (
        "conflict_or_unknown" if mismatched else "matching_pair_terminal_observed"
    )
    assert not result.full_fingerprint_verified


async def test_cancelled_reservation_retires_original_read_and_abandons_without_native_entry(
    domain_case,
):
    case = domain_case
    from tldw_chatbook.Chat.console_preparation_reads import (
        preparation_read_for,
        drain_preparation_reads,
    )

    await asyncio.wait_for(drain_preparation_reads(case.runtime._preparation_reads), 8)
    entered, release = threading.Event(), threading.Event()
    reads, plans = [], []
    code = case.store.reserve_durable_commit_plan.__func__.__code__

    def profile(frame, event, arg):
        if frame.f_code is code and frame.f_locals.get("self") is case.store:
            if event == "call":
                reads.append(preparation_read_for(case.runtime))
                entered.set()
                assert release.wait(10)
            elif event == "return" and arg is not None:
                plans.append(arg)

    async def issuing():
        async with prepared(case) as (acceptance, record):
            await reserve_durable_commit_plan_owned(case.store, acceptance)

    previous, threads = sys.getprofile(), threading.getprofile()
    threading.setprofile_all_threads(profile)
    task = asyncio.create_task(issuing())
    try:
        assert await _until(entered.is_set, timeout=5)
        read = reads[0]
        assert (
            read in case.runtime._preparation_reads
            and read in case.controller._preparation_reads
        )
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not read.retired.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), 5)
        assert len(plans) == 1 and read.result is plans[0]
        assert read.callback_returned and read.retired.done() and read._producer.done()
        assert (
            read not in case.runtime._preparation_reads
            and read not in case.controller._preparation_reads
        )
        assert (
            plans[0].acceptance.preparation_id not in case.store._durable_commit_plans
        )
        assert (
            case.database.get_connection()
            .execute("SELECT count(*) FROM messages")
            .fetchone()[0]
            == 0
        )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        threading.setprofile_all_threads(threads)
        sys.setprofile(previous)


@pytest.mark.parametrize("mode", ["new", "borrowed", "swallowed", "raised"])
async def test_selected_native_retirement_tracks_exact_existing_owners(
    domain_case, monkeypatch, mode
):
    from concurrent.futures import ThreadPoolExecutor
    import time
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import repository_participant
    from tldw_chatbook.Chat.console_native_commit import (
        ConsoleNativeRetirementUncertain,
    )
    from tldw_chatbook.Chat.console_preparation_reads import drain_preparation_reads

    case = domain_case
    await asyncio.wait_for(drain_preparation_reads(case.runtime._preparation_reads), 8)
    executor = ThreadPoolExecutor(max_workers=1)
    observed = []
    owner_records = []
    fail = (
        sqlite3.OperationalError("injected native close")
        if mode == "swallowed"
        else OSError("injected native close")
    )
    close_type, close_method = None, None
    previous, threads = sys.getprofile(), threading.getprofile()
    loop = asyncio.get_running_loop()
    run_in_executor = loop.run_in_executor
    original = case.store.persistence.commit_durable_turn

    def profile(frame, event, arg):
        nonlocal close_type, close_method
        if (
            event == "return"
            and frame.f_code is original.__func__.__code__
            and frame.f_locals.get("self") is case.store.persistence
        ):
            connection = case.database._local.conn
            participant = repository_participant(case.database)
            with storage._lock:
                lease = participant.connections[connection]
            observed.append(
                (connection, participant, lease, threading.current_thread())
            )
            owner_records.append(
                case.controller._ordinary_native_commit_owner("plan-preparation")
            )
            if mode in {"swallowed", "raised"}:
                close_type, close_method = type(connection), type(connection).close

                def refused(target):
                    if target is connection:
                        raise fail
                    return close_method(target)

                monkeypatch.setattr(close_type, "close", refused)

    try:
        if mode == "borrowed":
            borrowed = executor.submit(case.database.get_connection).result(timeout=5)
        async with prepared(case) as (acceptance, record):
            plan = await reserve_durable_commit_plan_owned(case.store, acceptance)
            monkeypatch.setattr(
                loop,
                "run_in_executor",
                lambda selected, function, *args: run_in_executor(
                    executor if selected is None else selected, function, *args
                ),
            )
            threading.setprofile_all_threads(profile)
            try:
                completion = await commit_reserved_durable_turn_owned(plan)
            finally:
                threading.setprofile_all_threads(threads)
                sys.setprofile(previous)
            assert len(observed) == 1 and len(owner_records) == 1
            connection, participant, lease, creator = observed[0]
            assert creator is not threading.current_thread()
            assert owner_records[0].retired.done()  # Wrapper completion only.
            assert completion.commit is not None
            if mode in {"swallowed", "raised"}:
                assert completion.native_retired is False
                assert isinstance(completion.error, ConsoleNativeRetirementUncertain)
                if mode == "raised":
                    assert completion.error.__cause__ is fail
                assert case.database._connection_quiescence.is_registered(connection)
                with storage._lock:
                    assert (
                        participant.connections[connection] is lease
                        and lease in storage._live_leases
                    )
                participant.close_admission()
                try:
                    assert not participant.drain(time.monotonic())
                finally:
                    participant.resume()
            elif mode == "borrowed":
                assert completion.native_retired is True and completion.error is None
                assert (
                    connection is borrowed
                    and case.database._connection_quiescence.is_registered(connection)
                )
                assert participant.connections[connection] is lease
            else:
                assert completion.native_retired is True and completion.error is None
                assert not case.database._connection_quiescence.is_registered(
                    connection
                )
                with storage._lock:
                    assert (
                        connection not in participant.connections
                        and lease not in storage._live_leases
                    )
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    sqlite3.Connection.in_transaction.__get__(connection)
    finally:
        threading.setprofile_all_threads(threads)
        sys.setprofile(previous)
        if close_type is not None:
            monkeypatch.setattr(close_type, "close", close_method)
        monkeypatch.setattr(loop, "run_in_executor", run_in_executor)
        executor.submit(case.database.close_connection).result(timeout=5)
        executor.shutdown(wait=True)
        if observed:
            connection, participant, lease, _ = observed[0]
            assert not case.database._connection_quiescence.is_registered(connection)
            with storage._lock:
                assert (
                    connection not in participant.connections
                    and lease not in storage._live_leases
                )
