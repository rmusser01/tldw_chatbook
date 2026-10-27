"""Native captured workflow ledgers preserve meaning without replay."""

import json
from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import _seed_core, content_reads


def _seed_workflows(connection, purpose):
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

    _seed_core(connection, purpose)
    connection.execute(
        "UPDATE messages SET assistant_generation_state='accepted' WHERE id='graph-answer'"
    )
    connection.execute(
        "UPDATE conversations SET active_leaf_message_id='graph-answer' WHERE id='conversation'"
    )
    authority = _authority()
    authority = replace(
        authority,
        scope_snapshot=replace(
            authority.scope_snapshot, note_ids=("note-a",), media_ids=()
        ),
    )
    connection.execute(
        "INSERT INTO console_dispatch_checkpoints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "graph-answer",
            "graph-user",
            "conversation",
            1,
            "prep-1",
            "attempt-1",
            "accepted",
            1,
            3,
            3,
            "manual",
            None,
            None,
            dump_console_turn_library_authority_json(authority),
            dump_console_resolved_destination_json(_destination()),
            dump_console_dispatch_reconstructability_json(_reconstructability()),
            "2026-10-25",
            "2026-10-25",
        ),
    )
    connection.execute(
        "INSERT INTO console_hook_continuation_receipts VALUES ('turn-opaque','stop-opaque','conversation','graph-answer','graph-variant','chain-opaque',1,'hook_continuation')"
    )
    connection.execute(
        "INSERT INTO console_machine_followup_receipts VALUES ('operation-opaque','turn-opaque','settled-opaque','conversation','graph-answer','graph-variant','chain-opaque',1,1,'[\"native\",\"hook\"]')"
    )
    connection.execute(
        "INSERT INTO fleet_progress_messages VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            7,
            "fleet-one",
            "conversation",
            "handle-opaque",
            "run",
            "run",
            None,
            "helper",
            "graph-user is body text",
            "2026-10-25",
        ),
    )
    connection.execute(
        "INSERT INTO message_generation_metadata VALUES ('graph-user',1,'graph-answer prompt','','custom',NULL,42,NULL,?, '2026-10-25')",
        (json.dumps({"opaque_provider_parameter": "graph-user"}),),
    )
    connection.execute(
        "INSERT INTO message_trajectory_metadata VALUES ('graph-user','conversation','graph-user',1,'user',1.0,NULL,NULL,NULL,NULL,NULL)"
    )
    connection.execute(
        "INSERT INTO message_trajectory_metadata VALUES ('graph-answer','conversation','graph-user',2,'user_feedback',2.0,NULL,NULL,NULL,NULL,?)",
        (
            json.dumps(
                {"action": "comment", "quote": "graph-user", "comment": "exact prose"}
            ),
        ),
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import TrajectoryRowWrite

    diagnostics = []
    ConsoleChatStore()._write_capture_failed_diagnostic(
        diagnostics.extend,
        [
            TrajectoryRowWrite(
                message_id="graph-answer",
                conversation_id="conversation",
                turn_id="graph-user",
                seq=None,
                event_kind="model_error",
                step_started_at=3.0,
                payload_json=json.dumps({"event_id": "native-original-source-event"}),
            )
        ],
    )
    assert len(diagnostics) == 1
    connection.execute(
        "INSERT INTO message_trajectory_metadata VALUES ('graph-answer','conversation','graph-user',3,'capture_failed',3.0,NULL,NULL,NULL,NULL,?)",
        (diagnostics[0].payload_json,),
    )
    connection.execute(
        "INSERT INTO transcript_annotations VALUES ('annotation-one','conversation','message:graph-answer','graph-answer','graph-user','exact comment','2026-10-25','2026-10-25',0)"
    )


def test_captured_workflow_ledgers_map_native_references_and_keep_authority_pending(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture

    original_seed = mobile_consolidation_fixture._seed

    def linked(db, legacy, identities):
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE fleet_progress_messages SET run_id=?,parent_run_id=?",
                (identities["child_run_id"], identities["parent_run_id"]),
            )
            if not legacy:
                cursor.execute("DELETE FROM console_dispatch_checkpoints")
                cursor.execute("DELETE FROM console_hook_continuation_receipts")
                cursor.execute("DELETE FROM console_machine_followup_receipts")

    def seed(stores, **kwargs):
        return original_seed(stores, **kwargs, linked_chat_seed=linked)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with content_reads(tmp_path, monkeypatch, _seed_workflows) as reads:
        build = getattr(reads, "content_workflow_plan", None)
        assert callable(build), "captured native reader needs C2b workflow planning"
        plan = build()
        assert not plan.blockers
        _native_consumers(plan)
        rows = {
            r.table: dict(r.values)
            for r in plan.rows
            if r.table != "message_trajectory_metadata"
        }
        ids = plan.identities
        checkpoint = rows["console_dispatch_checkpoints"]
        assert checkpoint["assistant_message_id"] == ids.mapped(
            "message", "graph-answer"
        )
        assert checkpoint["attempt_id"] == "attempt-1"
        authority = json.loads(checkpoint["frozen_authority_json"])
        assert authority["scope_snapshot"]["note_ids"] == [ids.mapped("note", "note-a")]
        assert (
            rows["console_hook_continuation_receipts"]["parent_turn_id"]
            == "turn-opaque"
        )
        assert (
            rows["console_machine_followup_receipts"]["settlement_id"]
            == "settled-opaque"
        )
        assert (
            rows["fleet_progress_messages"]["sequence"]
            == ids.mapped("fleet_sequence", 7)
            > 7
        )
        assert rows["fleet_progress_messages"]["body"] == "graph-user is body text"
        assert rows["message_generation_metadata"]["params_json"] == json.dumps(
            {"opaque_provider_parameter": "graph-user"}
        )
        assert rows["transcript_annotations"]["row_key"] == "message:" + ids.mapped(
            "message", "graph-answer"
        )
        assert all(
            r.required_proofs for r in plan.rows if r.table == "fleet_progress_messages"
        )
        diagnostic = next(
            r
            for r in plan.rows
            if r.table == "message_trajectory_metadata"
            and dict(r.values)["event_kind"] == "capture_failed"
        )
        original = next(
            r
            for r in plan.source_rows
            if r.table == diagnostic.table and r.original == diagnostic.original
        )
        assert (
            dict(diagnostic.values)["payload_json"]
            == dict(original.values)["payload_json"]
        )
        assert any(
            p.original == diagnostic.original
            and p.meaning == "historical_source_event"
            and p.source_identity == p.projected_identity
            for p in plan.provenance
        )
        with pytest.raises(ValueError, match="handlers_pending"):
            replace(
                plan, rows=(), blockers=(), successors=(), pending_handlers=()
            ).require_complete()
        assert not any(s.handler == "C2b" for s in plan.successors)
        assert {"C2c", "C3", "T", "S", "A-runs"} <= set(plan.pending_handlers)
        with pytest.raises(ValueError, match="handlers_pending"):
            plan.require_complete()
        assert any(
            r.table == "transcript_annotations"
            and dict(r.values)["row_key"] == "message:graph-answer"
            for r in plan.source_rows
        )


def test_native_retrieval_trajectory_references_map_without_rewriting_summary():
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.mobile_content_workflows import (
        _trajectory_payload,
    )

    builder = SimpleNamespace(
        ref=lambda domain, value: {"graph-user": "mapped-user"}[value]
    )
    source = {
        "summary": "graph-user is quoted text",
        "status": "completed",
        "event_id": "console-trace:graph-user:retrieval:context_attached",
        "parent_event_id": "console-trace:graph-user:retrieval:retrieval_completed",
        "source_event_id": "console-trace:graph-user:retrieval:retrieval_completed",
        "replacement_event_id": None,
        "field_states": {"payload": "omitted"},
        "sensitivity": "system_context",
    }
    value = json.loads(
        _trajectory_payload(
            builder,
            {"event_kind": "context_attached", "payload_json": json.dumps(source)},
            lambda *args: None,
        )
    )
    assert value["summary"] == source["summary"]
    assert value["event_id"] == "console-trace:mapped-user:retrieval:context_attached"
    assert (
        value["parent_event_id"]
        == "console-trace:mapped-user:retrieval:retrieval_completed"
    )


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("future_authority", "{}"),
        ("assistant", '{"trace_version":true,"model_status":"completed"}'),
        ("user_feedback", '{"action":"comment","quote":"x","authority":true}'),
        ("user_feedback", '{"action":"comment","quote":"x","quote":"y"}'),
        ("tool_result", '{"name":"tool","args":{"message_id":"x"},"result":"ok"}'),
        (
            "library_preparation",
            '{"version":1,"outcome":"zero_matches","attempt_id":"a","result_count":0,"source_types":[],"authority":true}',
        ),
    ],
)
def test_trajectory_unknown_or_lossy_payload_refuses(kind, payload):
    from tldw_chatbook.Backup_Recovery.mobile_content_workflows import (
        _trajectory_payload,
    )

    with pytest.raises(ValueError):
        _trajectory_payload(
            None, {"event_kind": kind, "payload_json": payload}, lambda *args: None
        )


def test_captured_workflow_collision_and_dependency_refusals_retain_originals(
    tmp_path, monkeypatch
):
    def seed(connection, purpose):
        _seed_workflows(connection, purpose)
        if purpose == "legacy":
            connection.execute(
                "UPDATE transcript_annotations SET row_key='unknown:graph-answer'"
            )
            connection.execute(
                "UPDATE message_trajectory_metadata SET event_kind='unknown-authority' WHERE seq=2"
            )
            other = connection.execute(
                "SELECT id FROM conversations WHERE id != 'conversation' LIMIT 1"
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO transcript_annotations VALUES ('wrong-scope',?,'message:graph-user',NULL,'quote','comment','2026','2026',0)",
                (other,),
            )
            connection.execute(
                "INSERT INTO console_hook_continuation_receipts VALUES ('invalid-hook-turn',?,'conversation','graph-answer','graph-variant','invalid-chain',1,'hook_continuation')",
                ("x" * 257,),
            )
            connection.execute(
                "INSERT INTO console_machine_followup_receipts VALUES ('orphan-hook-op','invalid-hook-turn','settlement','conversation','graph-answer','graph-variant','invalid-chain',1,0,'[\"hook\"]')"
            )

    with content_reads(tmp_path, monkeypatch, seed) as reads:
        plan = reads.content_workflow_plan()
        keys = {(b.table, b.reason, b.dependency_domain) for b in plan.blockers}
        assert (
            "console_hook_continuation_receipts",
            "native_unique_conflict",
            None,
        ) in keys
        assert (
            "console_machine_followup_receipts",
            "native_unique_conflict",
            None,
        ) in keys
        assert ("console_dispatch_checkpoints", "native_unique_conflict", None) in keys
        assert ("fleet_progress_messages", "dependency_unproven", "run") in keys
        assert (
            "transcript_annotations",
            "dependency_unproven",
            "trajectory_event",
        ) in keys
        assert (
            "message_trajectory_metadata",
            "dependency_unproven",
            "trajectory_event_kind",
        ) in keys
        refusals = {
            "annotation_scope": any(
                b.table == "transcript_annotations" and b.original == ("wrong-scope",)
                for b in plan.blockers
            ),
            "invalid_hook_parent": any(
                b.table == "console_machine_followup_receipts"
                and b.original[0] == "orphan-hook-op"
                for b in plan.blockers
            ),
        }
        assert refusals == {"annotation_scope": True, "invalid_hook_parent": True}
        blocked = {b.table for b in plan.blockers}
        assert blocked <= {
            r.table
            for r in (
                *plan.source_rows,
                *plan.settings.source_rows,
                *plan.settings.graph.source_rows,
            )
        }
        assert not any(
            r.table in blocked
            for r in plan.rows
            if r.table in {row.table for row in plan.source_rows}
            and r.table != "message_trajectory_metadata"
        )
        with pytest.raises(ValueError, match="handlers_pending"):
            plan.require_complete()


def _native_consumers(plan):
    """Disposable native sink proves read grammar, never migration authorization."""
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery.mobile_content_workflows import _TABLES
    from tldw_chatbook.Chat.console_chat_models import GenerationVariantMeta
    from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
    from tldw_chatbook.Chat.response_rules.repository import ResponseRuleRepository
    from tldw_chatbook.Chat.trajectory import derive_trajectory
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.fleet_progress_repository import FleetProgressRepository

    ids = plan.identities
    cid = ids.mapped("conversation", "conversation")
    with closing(CharactersRAGDB(":memory:", "workflow-consumer")) as db:
        db.add_conversation({"id": cid})
        messages = [
            dict(r.values)
            for r in plan.rows
            if r.table == "messages" and dict(r.values)["conversation_id"] == cid
        ]
        with db.transaction() as cursor:
            for row in messages:
                cursor.execute(
                    "INSERT INTO messages ("
                    + ",".join(row)
                    + ") VALUES ("
                    + ",".join("?" for _ in row)
                    + ")",
                    tuple(row.values()),
                )
            for record in plan.rows:
                if record.table not in _TABLES:
                    continue
                row = dict(record.values)
                cursor.execute(
                    "INSERT INTO "
                    + record.table
                    + " ("
                    + ",".join(row)
                    + ") VALUES ("
                    + ",".join("?" for _ in row)
                    + ")",
                    tuple(row.values()),
                )
            cursor.execute(
                "UPDATE conversations SET active_leaf_message_id=? WHERE id=?",
                (ids.mapped("message", "graph-answer"), cid),
            )
        result = ConsoleDispatchRepository(db).read_for_session(cid)
        assert result.checkpoint is not None
        assert result.checkpoint.frozen_authority.scope_snapshot.note_ids == (
            ids.mapped("note", "note-a"),
        )
        assert ResponseRuleRepository(db).historical_machine_parent(
            cid, ids.mapped("message", "graph-variant")
        ) == ids.mapped("message", "graph-answer")
        reports = FleetProgressRepository(db).load(cid)
        assert reports[0].identity.handle_id == "handle-opaque"
        assert reports[0].body == "graph-user is body text"
        generation = db.get_generation_metadata_for_messages(
            [ids.mapped("message", "graph-user")]
        )
        assert GenerationVariantMeta.from_row(
            generation[ids.mapped("message", "graph-user")][0]
        ).params == {"opaque_provider_parameter": "graph-user"}
        annotations = db.get_transcript_annotations(cid)
        assert annotations[0]["row_key"] == "message:" + ids.mapped(
            "message", "graph-answer"
        )
        trajectory = derive_trajectory(
            messages,
            {},
            db.get_trajectory_rows(cid),
            (),
            (),
            active_leaf_message_id=ids.mapped("message", "graph-answer"),
        )
        assert any(
            r.kind == "user_feedback" and r.payload["quote"] == "graph-user"
            for t in trajectory.turns
            for r in t.records
        )


def test_captured_checkpoint_conflicts_and_checkpoint_free_history(
    tmp_path, monkeypatch
):
    from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import (
        _active_continuation_json,
        _authority,
        _destination,
        _reconstructability,
    )
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        dump_console_dispatch_reconstructability_json,
        dump_console_resolved_destination_json,
        dump_console_turn_library_authority_json,
    )
    from tldw_chatbook.Chat.console_dispatch_repository import (
        _OWNER_SELECT,
        _RECONCILE_WRITE_NEEDED,
        ConsoleDispatchRepository,
    )

    native_results = {}
    cases = (
        "valid",
        "stranded",
        "write-needed",
        "terminal",
        "stale",
        "accepted-free",
        "dispatched-free",
        "continuation-free",
    )

    def populate(db, purpose):
        for case in cases:
            cid, user, assistant = "case-" + case, "user-" + case, "assistant-" + case
            db.add_conversation({"id": cid})
            db.add_message(
                {
                    "id": user,
                    "conversation_id": cid,
                    "sender": "User",
                    "role": "user",
                    "content": "historical user",
                }
            )
            state = (
                "complete"
                if case == "terminal"
                else "dispatch_started"
                if case == "dispatched-free"
                else None
                if case == "continuation-free"
                else "accepted"
            )
            if case == "continuation-free":
                # Seed the legacy persisted grammar at INSERT. The modern writer
                # eagerly normalizes this state; no mutation guard is overridden.
                with db.transaction() as cursor:
                    cursor.execute(
                        "INSERT INTO messages(id,conversation_id,parent_message_id,sender,role,content,provider_continuation_json,assistant_generation_state,client_id) VALUES (?,?,?,'Assistant','assistant','historical answer',?,NULL,?)",
                        (
                            assistant,
                            cid,
                            user,
                            _active_continuation_json(),
                            db.client_id,
                        ),
                    )
            else:
                db.add_message(
                    {
                        "id": assistant,
                        "conversation_id": cid,
                        "parent_message_id": user,
                        "sender": "Assistant",
                        "role": "assistant",
                        "content": "historical answer",
                        "assistant_generation_state": state,
                        "provider_continuation_json": _active_continuation_json()
                        if case in {"write-needed", "continuation-free"}
                        else None,
                    }
                )
            authority = _authority()
            authority = replace(
                authority,
                attempt_id="attempt-" + case,
                scope_snapshot=replace(
                    authority.scope_snapshot, note_ids=(), media_ids=()
                ),
            )
            with db.transaction() as cursor:
                cursor.execute(
                    "UPDATE conversations SET active_leaf_message_id=? WHERE id=?",
                    (user if case == "stranded" else assistant, cid),
                )
                if not case.endswith("-free"):
                    cursor.execute(
                        "INSERT INTO console_dispatch_checkpoints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            assistant,
                            user,
                            cid,
                            1,
                            "prep-" + case + "-" + purpose,
                            "attempt-" + case,
                            "accepted",
                            1,
                            2 if case == "stale" else 1,
                            1,
                            "manual",
                            None,
                            None,
                            dump_console_turn_library_authority_json(authority),
                            dump_console_resolved_destination_json(_destination()),
                            dump_console_dispatch_reconstructability_json(
                                _reconstructability()
                            ),
                            "2026-10-26T00:00:00Z",
                            "2026-10-26T00:00:00Z",
                        ),
                    )
                    source = cursor.execute(
                        _OWNER_SELECT + " WHERE checkpoint.conversation_id=?", (cid,)
                    ).fetchone()
                    repository = ConsoleDispatchRepository(db)
                    if case in {"write-needed", "terminal", "stranded"}:
                        assert (
                            repository._reconcile_checkpoint_row_uncoordinated(
                                cursor, cid, source, allow_writes=False
                            )
                            is _RECONCILE_WRITE_NEEDED
                        )
                    elif case == "valid":
                        assert repository._checkpoint_from_row(source)[0] is not None
                    else:
                        assert repository._checkpoint_from_row(source)[0] is None
                else:
                    state = ConsoleDispatchRepository(
                        db
                    )._reconcile_checkpoint_free_owner(cursor, cid)
                    native_results[purpose, case] = (state.kind.value, state.actions)

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=populate
    ) as reads:
        plan = reads.content_workflow_plan()
        originals = {
            r.original[0]: dict(r.values)
            for r in plan.source_rows
            if r.table == "console_dispatch_checkpoints"
        }
        blockers = {
            b.original[0]: b.reason
            for b in plan.blockers
            if b.table == "console_dispatch_checkpoints"
        }
        assert not plan.settings.blockers, plan.settings.blockers
        assert blockers == {
            "assistant-write-needed": "native_checkpoint_reconciliation_required",
            "assistant-terminal": "native_checkpoint_reconciliation_required",
            "assistant-stale": "native_checkpoint_owner_conflict",
            "assistant-stranded": "native_checkpoint_branch_conflict",
        }
        assert originals["assistant-stale"]["user_message_version"] == 2
        assert originals["assistant-terminal"]["state"] == "accepted"
        accepted = {
            r.original[0]: dict(r.values) for r in plan.rows if r.table == "messages"
        }
        for case in ("accepted-free", "dispatched-free", "continuation-free"):
            assert "assistant-" + case in accepted
            assert native_results["legacy", case][1] == ()
        assert (
            accepted["assistant-continuation-free"]["assistant_generation_state"]
            is None
        )
        assert (
            accepted["assistant-continuation-free"]["provider_continuation_json"]
            == _active_continuation_json()
        )
        assert any(
            r.table == "console_dispatch_checkpoints"
            and r.original == ("assistant-valid",)
            for r in plan.rows
        )
        assert not any(
            r.table == "messages" and r.original[0] in blockers for r in plan.rows
        )
        with pytest.raises(ValueError, match="handlers_pending"):
            plan.require_complete()


def test_captured_checkpoint_free_owner_cardinality_matches_native_path(
    tmp_path, monkeypatch
):
    from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import (
        _active_continuation_json,
    )
    from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository

    native_states = {}
    source_ids = {}

    def populate(db, purpose):
        for case in ("duplicate", "single", "off-path"):
            cid = "owner-" + case
            user, first, second = cid + "-user", cid + "-first", cid + "-second"
            db.add_conversation({"id": cid})
            db.add_message(
                {
                    "id": user,
                    "conversation_id": cid,
                    "sender": "User",
                    "role": "user",
                    "content": "historical user",
                }
            )
            db.add_message(
                {
                    "id": first,
                    "conversation_id": cid,
                    "parent_message_id": user,
                    "sender": "Assistant",
                    "role": "assistant",
                    "content": "historical first",
                    "provider_continuation_json": _active_continuation_json(),
                }
            )
            source_ids[case] = [first]
            leaf = first
            if case != "single":
                parent = user
                if case == "duplicate":
                    parent = cid + "-next-user"
                    db.add_message(
                        {
                            "id": parent,
                            "conversation_id": cid,
                            "parent_message_id": first,
                            "sender": "User",
                            "role": "user",
                            "content": "historical next",
                        }
                    )
                    leaf = second
                db.add_message(
                    {
                        "id": second,
                        "conversation_id": cid,
                        "parent_message_id": parent,
                        "sender": "Assistant",
                        "role": "assistant",
                        "content": "historical second",
                        "provider_continuation_json": _active_continuation_json(),
                    }
                )
                source_ids[case].append(second)
            with db.transaction() as cursor:
                cursor.execute(
                    "UPDATE conversations SET active_leaf_message_id=? WHERE id=?",
                    (leaf, cid),
                )
                state = ConsoleDispatchRepository(db)._reconcile_checkpoint_free_owner(
                    cursor, cid
                )
                native_states[purpose, case] = (
                    state.kind.value,
                    state.error_code,
                    state.actions,
                )
                assert state.actions == ()
                if case == "duplicate":
                    assert state.error_code == "duplicate_active_path_owner"
                else:
                    assert state.error_code is None
                    assert state.assistant_message_id == first

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=populate
    ) as reads:
        plan = reads.content_workflow_plan()
        assert not plan.settings.blockers
        assert {
            (b.original, b.reason) for b in plan.blockers if b.table == "conversations"
        } == {(("owner-duplicate",), "duplicate_active_path_owner")}
        retained = {
            r.original[0]: dict(r.values) for r in plan.rows if r.table == "messages"
        }
        original = {
            r.original[0]: dict(r.values)
            for r in plan.settings.source_rows
            if r.table == "messages"
        }
        for identity in source_ids["duplicate"]:
            assert identity not in retained
            assert (
                original[identity]["provider_continuation_json"]
                == _active_continuation_json()
            )
            assert plan.identities.mapped("message", identity)
            assert not any(
                s.original == (identity,) for s in (*plan.discharged, *plan.successors)
            )
        for case in ("single", "off-path"):
            for identity in source_ids[case]:
                assert (
                    retained[identity]["provider_continuation_json"]
                    == original[identity]["provider_continuation_json"]
                )
                assert any(
                    s.original == (identity,) and s.handler == "S"
                    for s in plan.successors
                )
            assert native_states["legacy", case][1:] == (None, ())
        with pytest.raises(ValueError, match="handlers_pending"):
            plan.require_complete()
