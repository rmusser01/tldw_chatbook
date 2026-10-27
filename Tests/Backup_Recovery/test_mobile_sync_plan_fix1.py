"""Reached native controls for S's first scoped review correction."""

from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import content_reads
from Tests.Backup_Recovery.test_mobile_sync_plan import _version_default


def test_latest_native_proof_version_is_not_silently_repaired(tmp_path, monkeypatch):
    originals = {}

    def native(db, purpose):
        _version_default(db, purpose)
        with db.transaction() as cursor:
            for table in ("chat_dictionaries", "world_books"):
                for identity in (61, 62):
                    cursor.execute(
                        "INSERT INTO " + table + "(id,name) VALUES (?,?)",
                        (identity, purpose + str(identity)),
                    )
            for identity in (61, 62):
                cursor.execute(
                    "INSERT INTO world_book_entries(id,world_book_id,keys,content) VALUES (?,61,'[]','native body')",
                    (identity,),
                )
            if purpose == "legacy":
                for entity in (
                    "chat_dictionaries",
                    "world_books",
                    "world_book_entries",
                ):
                    row = dict(
                        cursor.execute(
                            "SELECT change_id,entity,entity_id,operation,CAST(timestamp AS TEXT) AS timestamp,version,client_id,payload FROM sync_log WHERE entity=? AND entity_id='62'",
                            (entity,),
                        ).fetchone()
                    )
                    originals[entity] = row
                    cursor.execute(
                        "UPDATE sync_log SET version=version+1 WHERE change_id=?",
                        (row["change_id"],),
                    )

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=native
    ) as reads:
        plan = reads.sync_plan()
        columns, values = reads.page("legacy", "chat", "sync_log", limit=1000)
        retained = {row[0]: dict(zip(columns, row, strict=True)) for row in values}
        for entity, source in originals.items():
            assert any(
                f.entity == entity and f.original_id == 61 for f in plan.frontiers
            )
            assert any(
                b.table == entity
                and b.original == (62,)
                and b.reason == "current_proof_version_mismatch"
                for b in plan.blockers
            )
            assert not any(
                f.entity == entity and f.original_id == 62 for f in plan.frontiers
            )
            assert retained[source["change_id"]] == source | {
                "version": source["version"] + 1
            }
            proof = next(
                p for p in plan.provenance if p.change_id == source["change_id"]
            )
            assert proof.version == source["version"] + 1
            assert (
                proof.entity == source["entity"]
                and proof.operation == source["operation"]
            )
    with pytest.raises(ValueError, match="inactive"):
        reads.sync_plan()


def test_integrated_valid_workflows_keep_exact_pending_native_quarantines(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import (
        _authority,
        _destination,
        _reconstructability,
    )
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        dump_console_dispatch_reconstructability_json,
        dump_console_resolved_destination_json,
        dump_console_turn_library_authority_json,
    )

    original_seed = mobile_consolidation_fixture._seed

    def linked(db, legacy, identities):
        if legacy:
            with db.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO fleet_progress_messages VALUES (7,'fleet-one','conversation','handle',?,?,NULL,'helper','native body','2026-10-25')",
                    (identities["child_run_id"], identities["parent_run_id"]),
                )

    def seeded(stores, **kwargs):
        return original_seed(stores, **kwargs, linked_chat_seed=linked)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seeded)

    def native(db, purpose):
        _version_default(db, purpose)
        if purpose != "legacy":
            return
        for identity, role, parent, state in (
            ("workflow-user", "user", None, None),
            ("workflow-answer", "assistant", "workflow-user", "accepted"),
            ("workflow-child", "assistant", "workflow-user", None),
        ):
            db.add_message(
                {
                    "id": identity,
                    "conversation_id": "conversation",
                    "sender": role,
                    "role": role,
                    "parent_message_id": parent,
                    "content": "native",
                    "assistant_generation_state": state,
                }
            )
        db.set_conversation_active_leaf("conversation", "workflow-answer")
        authority = _authority()
        authority = replace(
            authority,
            scope_snapshot=replace(authority.scope_snapshot, note_ids=(), media_ids=()),
        )
        with db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO console_dispatch_checkpoints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "workflow-answer",
                    "workflow-user",
                    "conversation",
                    1,
                    "prep-1",
                    "attempt-1",
                    "accepted",
                    1,
                    1,
                    1,
                    "manual",
                    None,
                    None,
                    dump_console_turn_library_authority_json(authority),
                    dump_console_resolved_destination_json(_destination()),
                    dump_console_dispatch_reconstructability_json(
                        _reconstructability()
                    ),
                    "2026-10-25",
                    "2026-10-25",
                ),
            )
            cursor.execute(
                "INSERT INTO console_hook_continuation_receipts VALUES ('turn','stop','conversation','workflow-answer','workflow-child','chain',1,'hook_continuation')"
            )
            cursor.execute(
                "INSERT INTO console_machine_followup_receipts VALUES ('operation','turn','settlement','conversation','workflow-answer','workflow-child','chain',1,1,'[\"native\",\"hook\"]')"
            )

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=native
    ) as reads:
        plan = reads.integrated_plan()
        expected = {
            "console_dispatch_checkpoints": "S:imported_dispatch_quarantine",
            "console_hook_continuation_receipts": "S:imported_followup_quarantine",
            "console_machine_followup_receipts": "S:imported_followup_quarantine",
            "fleet_progress_messages": "S:imported_delivery_quarantine",
        }
        for table, requirement in expected.items():
            qualified = [r for r in plan.sync.trace.content.rows if r.table == table]
            assert len(qualified) == 1, plan.sync.trace.content.blockers
            row = qualified[0]
            assert any(
                r.table == table and r.original == row.original for r in plan.rows
            ), plan.blockers
            assert any(
                d.table == table
                and d.original == row.original
                and d.owner == "owner-write"
                and d.requirement == requirement
                for d in plan.pending_duties
            )
        for value in (plan, replace(plan, pending_handlers=())):
            with pytest.raises(ValueError, match="pending"):
                value.require_complete()
    with pytest.raises(ValueError, match="inactive"):
        reads.integrated_plan()


def test_integrated_repeated_native_metadata_spends_same_remaining_allowance(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_mobile_trace_plan import _seed_response_trace
    from tldw_chatbook.Backup_Recovery import (
        mobile_integrated_semantics,
        mobile_trace_plan,
    )
    from tldw_chatbook.Chat.console_chat_fork import (
        encode_console_fork_message_metadata,
    )
    from tldw_chatbook.Chat.console_trace_models import (
        SemanticRevisionRef,
        TraceCallState,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadLimit
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository

    def native(db, purpose):
        _version_default(db, purpose)
        call_id, user, answer, revision = _seed_response_trace(db, purpose, "repeat")
        assert db.update_message_metadata_local(
            answer,
            encode_console_fork_message_metadata("complete", "x" * (64 * 1024), user),
        )
        repository = ConsoleTraceRepository()
        with db.transaction() as cursor:
            first = dict(
                cursor.execute(
                    "SELECT * FROM console_trace_calls WHERE call_id=?", (call_id,)
                ).fetchone()
            )
            for index in range(1, 24):
                call = repository.reserve_call(
                    cursor,
                    owner_id=first["owner_id"],
                    segment_id=first["segment_id"],
                    turn_id=user,
                    run_id="repeat",
                    call_sequence=index,
                    idempotency_key=purpose + str(index),
                    policy_id=first["policy_id"],
                )
                repository.bind_call(
                    cursor,
                    call_id=call.call_id,
                    surface_node_id=first["surface_node_id"],
                    request_header_id=first["request_header_id"],
                    provider_name="openai",
                    model_name="trace-test-" + purpose,
                    route_identity="fresh",
                )
                for state in (
                    TraceCallState.DISPATCH_STARTED,
                    TraceCallState.RESPONSE_STARTED,
                    TraceCallState.COMPLETE,
                ):
                    repository.advance_call_state(
                        cursor,
                        call_id=call.call_id,
                        target=state,
                        occurred_at="2026-10-27T00:00:00Z",
                        integrity_state="complete",
                    )
                repository.store_response_link(
                    cursor, call_id=call.call_id, response=SemanticRevisionRef(revision)
                )

    from tldw_chatbook.Backup_Recovery import mobile_integrated_closure
    from tldw_chatbook.Chat import console_chat_fork
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    budgets, reached, decoded, refused_before_decode = [], [], [], []
    credits = {}
    native_json = _TraceReadBudget.json
    native_parse = console_chat_fork.parse_console_fork_message_metadata
    native_projection = mobile_integrated_closure._same_projection

    def reserve_json(self, raw):
        native_json(self, raw)
        if reached and isinstance(raw, str) and len(raw) > 64 * 1024:
            assert self is budgets[0]
            credits[raw] = credits.get(raw, 0) + 1

    def parse(raw):
        if reached and raw and len(raw) > 64 * 1024:
            assert credits.get(raw, 0) > 0, "reserve each repeated native decode first"
            credits[raw] -= 1
            decoded.append(raw)
        return native_parse(raw)

    def projection(*args):
        before = len(decoded)
        assert args[-1] is budgets[0]
        try:
            return native_projection(*args)
        except _TraceReadLimit:
            refused_before_decode.append(len(decoded) == before)
            raise

    monkeypatch.setattr(_TraceReadBudget, "json", reserve_json)
    monkeypatch.setattr(console_chat_fork, "parse_console_fork_message_metadata", parse)
    monkeypatch.setattr(mobile_integrated_closure, "_same_projection", projection)
    native_trace = mobile_trace_plan._plan_traces
    native_semantics = mobile_integrated_semantics._close_semantics

    def trace(reads, **kwargs):
        budget = kwargs.get("_read_budget")
        result = native_trace(reads, **kwargs)
        budgets.append(budget)
        return result

    def semantics(*args, **kwargs):
        budget = kwargs.get("_read_budget")
        reached.append(budget)
        assert budget is budgets[0], "integration must retain the actual T/S allowance"
        return native_semantics(*args, **kwargs)

    monkeypatch.setattr(mobile_trace_plan, "_plan_traces", trace)
    monkeypatch.setattr(mobile_integrated_semantics, "_close_semantics", semantics)
    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=native
    ) as reads:
        with pytest.raises(_TraceReadLimit):
            reads.integrated_plan()
        assert reached and reached[0] is budgets[0]
        assert len(decoded) > 2
        assert refused_before_decode == [True]
    with pytest.raises(ValueError, match="inactive"):
        reads.integrated_plan()
