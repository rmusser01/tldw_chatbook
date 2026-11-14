"""C1 Canvas and response-rule projections reach actual native consumers."""

import json
from contextlib import closing
from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import (
    _install_rows,
    _seed_core,
    content_reads,
)

CANVAS = "11111111-1111-4111-8111-111111111111"
REVISION = "22222222-2222-4222-8222-222222222222"
CHILD = "33333333-3333-4333-8333-333333333333"
DELETED = "44444444-4444-4444-8444-444444444444"
DELETED_REVISION = "55555555-5555-4555-8555-555555555555"
INVALID_REVISION = "66666666-6666-4666-8666-666666666666"
SOURCE = "<p>graph-answer is literal text Ω</p>"


def _native_backing(db, purpose):
    from Tests.Chat.response_rules_fixtures import source
    from Tests.Chat.response_rules_store_fixtures import learning
    from tldw_chatbook.Canvas.repository import CanvasRepository
    from tldw_chatbook.Chat.response_rules.models import (
        RuleAssessment,
        RuleBinding,
        RuleCheck,
        RuleScope,
    )
    from tldw_chatbook.Chat.response_rules.repository import ResponseRuleRepository
    from tldw_chatbook.Chat.response_rules.store import ResponseRuleStore

    canvas = CanvasRepository(db)
    common = {
        "title": "Exact canvas",
        "source": SOURCE,
        "runtime_profile": "canvas-v1",
        "actor_kind": "assistant",
        "origin_message_id": "graph-answer",
        "origin_turn_id": "historical-turn",
        "created_at": "2026-10-25T00:00:00Z",
    }
    canvas.create_canvas(
        "conversation", canvas_id=CANVAS, revision_id=REVISION, **common
    )
    canvas.append_revision(
        "conversation",
        CANVAS,
        parent_revision_id=REVISION,
        revision_id=CHILD,
        **(common | {"source": SOURCE + "<b>second</b>"}),
    )
    canvas.create_canvas(
        "conversation", canvas_id=DELETED, revision_id=DELETED_REVISION, **common
    )
    canvas.soft_delete_canvas("conversation", DELETED)
    canvas.set_reopen_hint("conversation", CANVAS)
    origin = source(
        message_id="graph-answer",
        branch_id="graph-variant",
        message_version=3,
        effective_digest="original-pinned-history",
    )
    result = learning(origin)
    repository = ResponseRuleRepository(db)
    scope = RuleScope("chat", "conversation")
    with db.transaction() as cursor:
        repository._save_draft(
            cursor, scope, origin, result, "preserve exact complaint"
        )
        repository._set_binding(cursor, RuleBinding(scope, "rule", 1, "enabled", 1), 0)
        repository._set_binding(
            cursor,
            RuleBinding(
                RuleScope("workspace", "conflicting-workspace"),
                "rule",
                1,
                "disabled",
                1,
            ),
            0,
        )
        repository._set_binding(
            cursor,
            RuleBinding(
                RuleScope("workspace", "workspace-default"), "rule", None, "excluded", 1
            ),
            0,
        )
        cursor.execute("UPDATE console_response_rule_drafts SET draft_id='draft'")
    assessment = RuleAssessment(
        "assessment",
        origin,
        (
            RuleCheck(
                "rule", 1, "applicable", "pass", "exact reason", ("opaque-evidence",)
            ),
        ),
        True,
        "historical-input-digest",
        "pass",
        "completed",
    )
    ResponseRuleStore(repository).save_assessment(assessment)
    ResponseRuleStore(repository).save_assessment(
        replace(assessment, assessment_id="pending-assessment", state="pending")
    )


def test_canvas_and_response_rules_survive_native_import_and_readers(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Canvas.repository import CanvasNotFoundError, CanvasRepository
    from tldw_chatbook.Chat.response_rules.models import RuleScope
    from tldw_chatbook.Chat.response_rules.repository import (
        ResponseRuleRepository,
        validate_activation,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    with content_reads(
        tmp_path, monkeypatch, _seed_core, native_seed=_native_backing
    ) as reads:
        plan = reads.content_graph_plan()
        assert any(row.table == "canvas_revisions" for row in plan.rows), (
            "C1 must retain native Canvas revision graphs"
        )
        assert any(
            row.table == "console_response_rule_validations" for row in plan.rows
        ), "C1 must transform native rule support records"
        assert not plan.blockers
        assert "exact message body" not in repr(plan)
        assert "preserve exact complaint" not in repr(plan)
        ids = plan.identities
        conversation = ids.mapped("conversation", "conversation")
        with closing(CharactersRAGDB(":memory:", "backing-proof")) as db:
            db.add_conversation(
                {"id": conversation, "title": "private native consumer"}
            )
            _install_rows(
                db,
                tuple(row for row in plan.rows if not row.table.startswith("canvas_")),
            )
            canvas = CanvasRepository(db)
            assert len(plan.canvas_batches) == 1
            canvas.import_batch(plan.canvas_batches[0])
            revision = canvas.read_revision(
                conversation, ids.mapped("canvas_revision", CHILD)
            )
            assert revision.source == SOURCE + "<b>second</b>"
            assert revision.parent_revision_id == ids.mapped(
                "canvas_revision", REVISION
            )
            assert revision.origin_message_id == ids.mapped("message", "graph-answer")
            assert revision.origin_turn_id == "historical-turn"
            with pytest.raises(CanvasNotFoundError):
                canvas.read_revision(
                    conversation, ids.mapped("canvas_revision", DELETED_REVISION)
                )
            assert (
                canvas.read_revision(
                    conversation,
                    ids.mapped("canvas_revision", DELETED_REVISION),
                    include_deleted=True,
                ).source
                == SOURCE
            )
            assert canvas.get_reopen_hint(conversation) == ids.mapped("canvas", CANVAS)
            repository = ResponseRuleRepository(db)
            rule = repository.get_revision(ids.mapped("response_rule", "rule"), 1)
            validation = repository.get_validation(rule.rule_id, 1)
            validate_activation(rule, validation)
            assert rule.origin.message_id == ids.mapped("message", "graph-answer")
            assert rule.origin.branch_id == ids.mapped("message", "graph-variant")
            assert rule.origin.effective_digest == "original-pinned-history"
            assert rule.origin.profile_id == "profile"
            assert (
                repository.list_bindings(RuleScope("chat", conversation))[0].state
                == "disabled"
            )
            assert (
                repository.list_bindings(RuleScope("chat", conversation))[
                    0
                ].binding_revision
                == 2
            )
            assert (
                repository.list_bindings(
                    RuleScope(
                        "workspace", ids.mapped("workspace", "conflicting-workspace")
                    )
                )[0].state
                == "disabled"
            )
            draft = repository.list_drafts(RuleScope("chat", conversation))[0]
            assert draft.rule == rule
            assert draft.fixtures["case-0"].response_text == "exact message body"
            assert draft.fixtures["case-1"].response_text == "Missing proof"
        # Compare to the installed restriction operation in a disposable native
        # owner populated with authenticated original rows, never either owner.
        from tldw_chatbook.DB.recovery_response_rules import prepare_imported_rules

        restriction_tables = {
            "conversations",
            "messages",
            "console_response_rule_revisions",
            "console_response_rule_bindings",
            "console_response_rule_assessments",
        }
        with closing(
            CharactersRAGDB(":memory:", "native-restriction-proof")
        ) as original_db:
            _install_rows(
                original_db,
                tuple(
                    row for row in plan.source_rows if row.table in restriction_tables
                ),
            )
            with original_db.transaction() as cursor:
                prepare_imported_rules(cursor)
                for projection in plan.rows:
                    values = dict(projection.values)
                    if projection.table == "console_response_rule_bindings":
                        native = cursor.execute(
                            "SELECT state,binding_revision FROM console_response_rule_bindings WHERE scope_kind=? AND scope_id=? AND rule_id=?",
                            projection.original,
                        ).fetchone()
                        assert tuple(native) == (
                            values["state"],
                            values["binding_revision"],
                        )
                    if projection.table == "console_response_rule_assessments":
                        native = cursor.execute(
                            "SELECT state,assessment_json FROM console_response_rule_assessments WHERE assessment_id=?",
                            projection.original,
                        ).fetchone()
                        document = json.loads(values["assessment_json"])
                        assert native["state"] == values["state"] == document["state"]
                        assert (
                            json.loads(native["assessment_json"])["evidence_digest"]
                            == document["evidence_digest"]
                        )
        pending_source = next(
            dict(row.values)
            for row in plan.source_rows
            if row.table == "console_response_rule_assessments"
            and row.original == ("pending-assessment",)
        )
        assert pending_source["state"] == "pending"
        original = next(
            dict(row.values)
            for row in plan.source_rows
            if row.table == "console_response_rule_revisions"
        )
        assert json.loads(original["origin_json"])["message_id"] == "graph-answer"
        with pytest.raises(ValueError, match="pending"):
            plan.require_complete()


def test_exact_canvas_sequence_and_global_rule_scope_refuse_without_losing_history(
    tmp_path, monkeypatch
):
    def backing(db, purpose):
        _native_backing(db, purpose)
        if purpose == "legacy":
            with db.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO canvas_revisions SELECT ?,canvas_id,?,4,title,runtime_profile,html,content_sha256,html_bytes,actor_kind,origin_message_id,origin_turn_id,created_at,deleted_at FROM canvas_revisions WHERE id=?",
                    (INVALID_REVISION, CHILD, CHILD),
                )
                cursor.execute(
                    "UPDATE console_response_rule_bindings SET scope_kind='global',scope_id='profile',conversation_id=NULL WHERE scope_kind='workspace' AND scope_id='conflicting-workspace'"
                )
                cursor.execute(
                    "UPDATE console_response_rule_validations SET message_version=message_version+1"
                )
                cursor.execute(
                    "UPDATE console_response_rule_assessments SET state='stale' WHERE assessment_id='assessment'"
                )

    with content_reads(tmp_path, monkeypatch, _seed_core, native_seed=backing) as reads:
        plan = reads.content_graph_plan()
        assert any(
            b.table == "canvas_revisions" and b.original == (INVALID_REVISION,)
            for b in plan.blockers
        )
        assert any(
            b.table == "console_response_rule_bindings"
            and b.dependency_domain == "response_rule_global_namespace"
            and b.dependency_id == "profile"
            for b in plan.blockers
        )
        assert any(
            b.table == "console_response_rule_assessments"
            and b.original == ("assessment",)
            for b in plan.blockers
        )
        assert any(row.table == "console_response_rule_revisions" for row in plan.rows)
        assert any(
            b.table == "console_response_rule_validations"
            and b.reason == "native_content_invalid"
            for b in plan.blockers
        )
        assert not any(
            row.table == "console_response_rule_bindings"
            and dict(row.values)["scope_kind"] == "chat"
            for row in plan.rows
        )
        original = next(
            dict(row.values)
            for row in plan.source_rows
            if row.table == "canvas_revisions" and row.original == (INVALID_REVISION,)
        )
        assert original["html"] == SOURCE + "<b>second</b>"
