"""Captured regressions for native C1 occupancy and closed rule JSON boundaries."""

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, replace

import pytest

from Tests.Backup_Recovery.test_mobile_content_backing import (
    CANVAS,
    REVISION,
    _native_backing,
)
from Tests.Backup_Recovery.test_mobile_content_graph import _seed_core, content_reads


def test_canvas_opposite_canonical_tables_block_both_source_identities(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Canvas.repository import CanvasConflictError, CanvasRepository

    def backing(db, purpose):
        if purpose == "legacy":
            _native_backing(db, purpose)
            return
        repository = CanvasRepository(db)
        repository.create_canvas(
            "conversation",
            canvas_id=REVISION,
            revision_id=CANVAS,
            title="Canonical opposite occupancy",
            source="<p>canonical bytes</p>",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id="graph-answer",
            origin_turn_id="canonical-turn",
            created_at="2026-10-25T00:00:00Z",
        )
        # The actual native import predicate rejects each direction separately.
        with db.transaction() as cursor:
            for documents, revisions in (((CANVAS,), ()), ((), (REVISION,))):
                with pytest.raises(CanvasConflictError, match="identity_collision"):
                    repository._require_ids_available(cursor, documents, revisions)

    with content_reads(tmp_path, monkeypatch, _seed_core, native_seed=backing) as reads:
        identities = reads.identity_plan()
        assert identities.mapped("canvas", CANVAS) == CANVAS
        assert identities.mapped("canvas_revision", REVISION) == REVISION
        plan = reads.content_graph_plan()
        blocked = {(b.table, b.original) for b in plan.blockers}
        assert {
            ("canvas_documents", (CANVAS,)),
            ("canvas_revisions", (REVISION,)),
        } <= blocked
        assert not plan.canvas_batches
        assert not any(
            (row.table, row.original)
            in {("canvas_documents", (CANVAS,)), ("canvas_revisions", (REVISION,))}
            for row in plan.rows
        )
        assert (
            next(
                dict(r.values)["html"]
                for r in plan.source_rows
                if r.table == "canvas_revisions" and r.original == (REVISION,)
            )
            == "<p>graph-answer is literal text Ω</p>"
        )


def test_final_rule_json_limit_blocks_colliding_near_limit_assessment(
    tmp_path, monkeypatch
):
    from Tests.Chat.response_rules_fixtures import source
    from tldw_chatbook.Chat.response_rules.models import (
        RuleAssessment,
        RuleCheck,
        RuleSource,
        canonical_json,
    )
    from tldw_chatbook.Chat.response_rules.repository import ResponseRuleRepository
    from tldw_chatbook.Chat.response_rules.store import ResponseRuleStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    originals = {}

    def backing(db, purpose):
        _native_backing(db, purpose)
        assessment = RuleAssessment(
            "near-limit",
            source(
                message_id="graph-answer", branch_id="graph-variant", message_version=3
            ),
            (RuleCheck("rule", 1, "applicable", "pass", "", ("evidence",)),),
            True,
            "historical",
            "pass",
            "pending",
        )
        padding = 65536 - len(canonical_json(asdict(assessment)).encode("utf-8"))
        assessment = replace(
            assessment,
            checks=(
                replace(
                    assessment.checks[0],
                    reason="Ω" * (padding // 2) + "x" * (padding % 2),
                ),
            ),
        )
        originals[purpose] = canonical_json(asdict(assessment))
        assert len(originals[purpose].encode("utf-8")) == 65536
        ResponseRuleStore(ResponseRuleRepository(db)).save_assessment(assessment)

    with content_reads(tmp_path, monkeypatch, _seed_core, native_seed=backing) as reads:
        plan = reads.content_graph_plan()
        ids = plan.identities
        assert ids.mapped("response_assessment", "near-limit") != "near-limit"
        assert ids.mapped("message", "graph-answer") != "graph-answer"
        original = json.loads(originals["legacy"])
        # Independently construct the declared mapping/state projection and reach
        # the actual installed CHECK, rather than trusting a planner limit helper.
        mapped_source = RuleSource(
            **(
                original["source"]
                | {
                    "conversation_id": ids.mapped("conversation", "conversation"),
                    "message_id": ids.mapped("message", "graph-answer"),
                    "branch_id": ids.mapped("message", "graph-variant"),
                }
            )
        )
        projected = RuleAssessment(
            **(
                original
                | {
                    "assessment_id": ids.mapped("response_assessment", "near-limit"),
                    "source": mapped_source,
                    "checks": tuple(
                        RuleCheck(
                            **(check | {"rule_id": ids.mapped("response_rule", "rule")})
                        )
                        for check in original["checks"]
                    ),
                    "state": "cancelled",
                }
            )
        )
        assert len(canonical_json(asdict(projected)).encode("utf-8")) > 65536
        with closing(CharactersRAGDB(":memory:", "rule-limit-proof")) as db:
            db.add_conversation(
                {"id": mapped_source.conversation_id, "title": "Private limit proof"}
            )
            db.add_message(
                {
                    "id": mapped_source.message_id,
                    "conversation_id": mapped_source.conversation_id,
                    "role": "assistant",
                    "sender": "assistant",
                    "content": "private",
                }
            )
            with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
                ResponseRuleStore(ResponseRuleRepository(db)).save_assessment(projected)
        assert any(
            b.table == "console_response_rule_assessments"
            and b.original == ("near-limit",)
            and b.reason == "native_content_invalid"
            for b in plan.blockers
        )
        assert not any(
            r.table == "console_response_rule_assessments"
            and r.original == ("near-limit",)
            for r in plan.rows
        )
        assert (
            next(
                dict(r.values)["assessment_json"]
                for r in plan.source_rows
                if r.table == "console_response_rule_assessments"
                and r.original == ("near-limit",)
            )
            == originals["legacy"]
        )


def test_malformed_native_rule_json_refuses_before_collection_coercion(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.response_rules.models import canonical_json

    originals = {}

    def backing(db, purpose):
        _native_backing(db, purpose)
        if purpose != "legacy":
            return
        with db.transaction() as cursor:
            original = json.loads(
                cursor.execute(
                    "SELECT assessment_json FROM console_response_rule_assessments WHERE assessment_id='assessment'"
                ).fetchone()[0]
            )
            mutations = {
                "bad-evidence-refs": lambda d: d["checks"][0].update(
                    evidence_refs="opaque-evidence"
                ),
                "bad-checks": lambda d: d.update(checks={}),
                "bad-bool": lambda d: d.update(evidence_complete="true"),
                "bad-reason": lambda d: d["checks"][0].update(reason=["body"]),
            }
            for identity, mutate in mutations.items():
                document = json.loads(canonical_json(original))
                document["assessment_id"] = identity
                mutate(document)
                encoded = canonical_json(document)
                originals[("console_response_rule_assessments", (identity,))] = (
                    "assessment_json",
                    encoded,
                )
                cursor.execute(
                    "INSERT INTO console_response_rule_assessments VALUES (?,?,?,?,?)",
                    (identity, "conversation", "graph-answer", "completed", encoded),
                )
            validation = json.loads(
                cursor.execute(
                    "SELECT validation_json FROM console_response_rule_validations"
                ).fetchone()[0]
            )
            validation["fixture_ids"] = "case-0"
            encoded = canonical_json(validation)
            originals[("console_response_rule_validations", ("rule", 1))] = (
                "validation_json",
                encoded,
            )
            cursor.execute(
                "UPDATE console_response_rule_validations SET validation_json=?",
                (encoded,),
            )
            draft = json.loads(
                cursor.execute(
                    "SELECT result_json FROM console_response_rule_drafts"
                ).fetchone()[0]
            )
            draft["validation"]["case_results"][0]["check"]["evidence_refs"] = (
                "nested-evidence"
            )
            encoded = canonical_json(draft)
            originals[("console_response_rule_drafts", ("draft",))] = (
                "result_json",
                encoded,
            )
            cursor.execute(
                "UPDATE console_response_rule_drafts SET result_json=?", (encoded,)
            )
            fixture = json.loads(
                cursor.execute(
                    "SELECT input_json FROM console_response_rule_fixtures WHERE case_id='case-1'"
                ).fetchone()[0]
            )
            fixture["evidence_complete"] = "true"
            encoded = canonical_json(fixture)
            originals[("console_response_rule_fixtures", ("rule", 1, "case-1"))] = (
                "input_json",
                encoded,
            )
            cursor.execute(
                "UPDATE console_response_rule_fixtures SET input_json=? WHERE case_id='case-1'",
                (encoded,),
            )

    with content_reads(tmp_path, monkeypatch, _seed_core, native_seed=backing) as reads:
        plan = reads.content_graph_plan()
        blocked = {
            (b.table, b.original)
            for b in plan.blockers
            if b.reason == "native_content_invalid"
        }
        assert set(originals) <= blocked
        assert not any((r.table, r.original) in originals for r in plan.rows)
        source = {(r.table, r.original): dict(r.values) for r in plan.source_rows}
        for key, (column, encoded) in originals.items():
            assert source[key][column] == encoded
        assert "opaque-evidence" not in repr(plan.blockers)
        assert any(r.table == "console_response_rule_revisions" for r in plan.rows)
