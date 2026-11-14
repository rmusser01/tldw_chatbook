"""Real admission, collector and provider reservation lifetime controls."""

from dataclasses import replace

import asyncio

import pytest

from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import (
    _acceptance,
    _insert,
)
from Tests.Chat.response_rules_fixtures import source
from Tests.Chat.test_console_trace_runtime import _semantic_request
from Tests.Chat.test_response_rule_corrections import acceptance as repair_acceptance
from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleAssistantSettlement,
    ConsoleDispatchCheckpointState,
    ConsoleDispatchResultStatus,
)
from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
from tldw_chatbook.Chat.console_prepared_request import (
    prepare_provider_request,
    resolve_request_capacity,
)
from tldw_chatbook.Chat.console_trace_maintenance import (
    LegacyTraceMaintenance,
    TraceGarbageCollector,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    SavedRevisionTraceProvenance,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.fixture
def admitted(tmp_path, request):
    db = CharactersRAGDB(tmp_path / "admission.sqlite", "admission-gc")
    try:
        conversation = db.add_conversation({"title": "admission"})
        assert LegacyTraceMaintenance(db).run_batch().logical_complete
        mode = getattr(request, "param", "ordinary")
        admission = _acceptance(conversation)
        if mode == "repair":
            user = db.add_message(
                {"conversation_id": conversation, "sender": "user", "content": "task"}
            )
            parent = db.add_message(
                {
                    "conversation_id": conversation,
                    "sender": "assistant",
                    "content": "violation",
                    "parent_message_id": user,
                }
            )
            admission = repair_acceptance(
                source(conversation_id=conversation, message_id=parent)
            )
            admission = replace(
                admission,
                continuation_receipt=ContinuationReceipt(
                    "turn", "stop", parent, "chain", 1
                ),
                machine_followup_receipt=replace(
                    admission.machine_followup_receipt, contributors=("native", "hook")
                ),
            )
        repository = ConsoleDispatchRepository(db)
        before = TraceGarbageCollector(db).current_graph_epoch()
        checkpoint = _insert(db, repository, admission)
        assert TraceGarbageCollector(db).current_graph_epoch() > before
        yield db, repository, checkpoint
    finally:
        db.close()


def prepared(db, checkpoint):
    row = (
        db.get_connection()
        .execute(
            "SELECT revision_id FROM console_trace_semantic_revisions WHERE live_message_id=? ORDER BY revision_sequence DESC LIMIT 1",
            (checkpoint.user_message_id,),
        )
        .fetchone()
    )
    assert row is not None
    revision = row[0]
    request = _semantic_request(
        [{"role": "user", "content": "hello"}],
        [SavedRevisionTraceProvenance(revision)],
        FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None),
    )
    return revision, prepare_provider_request(
        request,
        wire_style="distinct_roles",
        provider="openai",
        model="gpt-test",
        capacity=resolve_request_capacity(context_window_tokens=None),
        apply_safety_window=False,
    )


def assert_payload_root(db, request_id, revision, *, retained):
    mark = (
        db.get_connection()
        .execute(
            "SELECT 1 FROM console_trace_gc_marks WHERE request_id=? "
            "AND entity_kind='console_trace_payload_revisions' AND entity_id=?",
            (request_id, revision),
        )
        .fetchone()
    )
    assert (mark is not None) is retained
    # A still-live canonical message keeps metadata independently of admission.
    assert (
        db.get_connection()
        .execute(
            "SELECT 1 FROM console_trace_semantic_revisions WHERE revision_id=?",
            (revision,),
        )
        .fetchone()
    )


def settle(repository, checkpoint):
    result = repository.settle_with_assistant(
        ConsoleAssistantSettlement(
            assistant_message_id=checkpoint.assistant_message_id,
            expected_checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
            expected_checkpoint_revision=1,
            expected_user_message_version=1,
            expected_assistant_message_version=1,
            terminal_state="stopped",
            metadata_json=None,
            content="",
        )
    )
    assert result.status is ConsoleDispatchResultStatus.COMMITTED


@pytest.mark.parametrize("admitted", ["ordinary", "repair"], indirect=True)
def test_collect_between_admission_and_reservation_retains_exact_live_revision(
    admitted,
):
    db, repository, checkpoint = admitted
    revision, request = prepared(db, checkpoint)
    collector = TraceGarbageCollector(db)
    collector.mark(request_id="admission-gap")
    assert_payload_root(db, "admission-gap", revision, retained=True)
    result = collector.sweep(request_id="admission-gap")
    assert result.status == "completed"
    # Real factory must find the originally admitted revision, not recreate a replacement.
    boundary = ConsoleTraceBoundaryFactory(db)(request, None, ConsoleRequestRoute.FRESH)
    assert boundary is not None
    assert (
        db.get_connection()
        .execute(
            "SELECT 1 FROM console_trace_semantic_revisions WHERE revision_id=?",
            (revision,),
        )
        .fetchone()
    )
    settle(repository, checkpoint)
    collector.collect(request_id="call-takes-over")
    assert (
        db.get_connection()
        .execute(
            "SELECT 1 FROM console_trace_semantic_revisions WHERE revision_id=?",
            (revision,),
        )
        .fetchone()
    )


def test_settlement_revokes_unreserved_root_and_invalidates_old_mark(admitted):
    db, repository, checkpoint = admitted
    revision, _ = prepared(db, checkpoint)
    collector = TraceGarbageCollector(db)
    marked = collector.mark(request_id="before-settlement")
    settle(repository, checkpoint)
    assert collector.current_graph_epoch() > marked.marked_epoch
    assert collector.sweep(request_id="before-settlement").status == "stale_epoch"
    assert (
        db.get_connection()
        .execute(
            "SELECT 1 FROM console_trace_semantic_revisions WHERE revision_id=?",
            (revision,),
        )
        .fetchone()
    )
    collector.mark(request_id="revoked-root")
    assert_payload_root(db, "revoked-root", revision, retained=False)
    assert collector.sweep(request_id="revoked-root").status == "completed"


@pytest.mark.parametrize(
    "invalid", ["version", "payload", "parent", "off_path", "orphan"]
)
def test_invalid_or_unowned_admissions_do_not_retain_payload_roots(admitted, invalid):
    db, repository, checkpoint = admitted
    revision, _ = prepared(db, checkpoint)
    if invalid == "orphan":
        settle(repository, checkpoint)
    elif invalid == "off_path":
        from tldw_chatbook.DB.ChaChaNotes_DB import InputError

        with pytest.raises(InputError, match="pending dispatch"):
            db.set_conversation_active_leaf(checkpoint.conversation_id, None)
        # The public owner refuses this state; construct imported corruption
        # solely to prove it cannot become a GC retention root.
        with db.transaction(immediate=True) as cursor:
            cursor.execute(
                "UPDATE conversations SET active_leaf_message_id=NULL WHERE id=?",
                (checkpoint.conversation_id,),
            )
    else:
        with db.transaction(immediate=True) as cursor:
            if invalid == "version":
                cursor.execute(
                    "UPDATE console_dispatch_checkpoints SET user_message_version=2"
                )
            elif invalid == "payload":
                cursor.execute(
                    "UPDATE console_dispatch_checkpoints SET frozen_authority_json='{}'"
                )
            else:
                cursor.execute(
                    "UPDATE console_dispatch_checkpoints SET user_message_id=assistant_message_id"
                )
    collector = TraceGarbageCollector(db)
    collector.mark(request_id="invalid-owner")
    assert_payload_root(db, "invalid-owner", revision, retained=False)
    assert collector.sweep(request_id="invalid-owner").status == "completed"


def test_admission_after_mark_invalidates_sweep_and_new_mark_retains_it(tmp_path):
    db = CharactersRAGDB(tmp_path / "mark-before-admission.sqlite", "gc-interleave")
    try:
        conversation = db.add_conversation({"title": "interleaving"})
        assert LegacyTraceMaintenance(db).run_batch().logical_complete
        collector = TraceGarbageCollector(db)
        collector.mark(request_id="before-admission")
        checkpoint = _insert(
            db, ConsoleDispatchRepository(db), _acceptance(conversation)
        )
        _revision, request = prepared(db, checkpoint)
        assert collector.sweep(request_id="before-admission").status == "stale_epoch"
        collector.collect(request_id="after-admission")
        assert ConsoleTraceBoundaryFactory(db)(request, None, ConsoleRequestRoute.FRESH)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_automatic_collector_revisits_settled_checkpoint_epoch(
    admitted, monkeypatch
):
    from Tests.Chat.test_console_trace_maintenance_parking import _fast_runtime
    from tldw_chatbook.Chat import console_runtime as runtime_module
    from tldw_chatbook.Chat.console_trace_legacy import LegacyTraceNormalizer

    db, repository, checkpoint = admitted
    revision, _ = prepared(db, checkpoint)
    runtime = _fast_runtime(monkeypatch)
    # Accelerate the existing periodic check; keep the actual worker, collector,
    # epoch deduplication, compactor eligibility, and connection ownership.
    monkeypatch.setattr(
        runtime_module, "TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS", 0.0
    )
    from tldw_chatbook.Chat.console_trace_maintenance import PhysicalTraceCompactor

    physical = []
    original_physical = PhysicalTraceCompactor.run_after_gc

    def compact(self, result):
        outcome = original_physical(self, result)
        physical.append(outcome)
        return outcome

    monkeypatch.setattr(PhysicalTraceCompactor, "run_after_gc", compact)
    marked_roots = {}
    original_mark = TraceGarbageCollector.mark

    def mark(self, **kwargs):
        result = original_mark(self, **kwargs)
        marked_roots[result.request_id] = (
            self.db.get_connection()
            .execute(
                "SELECT 1 FROM console_trace_gc_marks WHERE request_id=? "
                "AND entity_kind='console_trace_payload_revisions' AND entity_id=?",
                (result.request_id, revision),
            )
            .fetchone()
            is not None
        )
        return result

    monkeypatch.setattr(TraceGarbageCollector, "mark", mark)
    collected = []
    original = TraceGarbageCollector.collect

    def collect(self, **kwargs):
        result = original(self, **kwargs)
        collected.append(result)
        return result

    monkeypatch.setattr(TraceGarbageCollector, "collect", collect)
    runtime._schedule_legacy_trace_maintenance(db, lambda: LegacyTraceNormalizer(db))
    try:
        # Poll a real deadline: short asyncio sleeps may resume early on Windows.
        async with asyncio.timeout(5):
            while len(physical) < 3:
                await asyncio.sleep(0.01)
        assert len(collected) == 1
        assert all(row.reason_code == "database_threshold" for row in physical)
        assert (
            db.get_connection()
            .execute(
                "SELECT 1 FROM console_trace_semantic_revisions WHERE revision_id=?",
                (revision,),
            )
            .fetchone()
        )
        assert marked_roots[collected[-1].request_id]
        first_epoch = collected[-1].marked_epoch
        settle(repository, checkpoint)
        async with asyncio.timeout(5):
            while not any(row.marked_epoch > first_epoch for row in collected):
                await asyncio.sleep(0.01)
        assert not marked_roots[collected[-1].request_id]
    finally:
        await runtime.dispose()
