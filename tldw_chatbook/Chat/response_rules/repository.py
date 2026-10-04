"""Private main-SQLite sidecars; no provider calls occur in transactions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, replace
from typing import TYPE_CHECKING
from uuid import uuid4

from .models import (
    EVALUATOR_PROTOCOL_VERSION,
    RuleBinding,
    RuleCandidate,
    RuleCaseResult,
    RuleCheck,
    RuleEvidence,
    RuleInput,
    RuleLearningResult,
    RuleRevision,
    RuleScope,
    RuleSource,
    RuleValidation,
    canonical_json,
)

if TYPE_CHECKING:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class RuleBindingConflict(RuntimeError):
    """The reviewed scope binding changed before this write."""


def revision_document(rule: RuleRevision) -> dict:
    document = asdict(rule)
    document["candidate"] = rule.candidate.model_dump(mode="json")
    return document


def read_revision(document: dict) -> RuleRevision:
    return RuleRevision(
        **(
            document
            | {
                "candidate": RuleCandidate.model_validate(document["candidate"]),
                "origin": RuleSource(**document["origin"]),
            }
        )
    )


def read_validation(document: dict) -> RuleValidation:
    return RuleValidation(
        **(
            document
            | {
                "source": RuleSource(**document["source"]),
                "case_results": tuple(
                    RuleCaseResult(**(case | {"check": RuleCheck(**case["check"])}))
                    for case in document["case_results"]
                ),
            }
        )
    )


def read_input(document: dict) -> RuleInput:
    return RuleInput(
        **(
            document
            | {"evidence": tuple(RuleEvidence(**e) for e in document["evidence"])}
        )
    )


def validate_activation(rule: RuleRevision, validation: RuleValidation) -> None:
    """Require an exact, successful host-labelled discrimination record."""
    if (
        validation.candidate_digest != rule.candidate.candidate_digest()
        or validation.detector_digest != rule.candidate.detector_digest()
        or validation.protocol_version != EVALUATOR_PROTOCOL_VERSION
        or validation.source != rule.origin
    ):
        raise ValueError("validation_identity_mismatch")
    expected = {
        "recorded_violation",
        "synthetic_violation",
        "synthetic_correction",
        "synthetic_acceptable",
    }
    if {c.case_type for c in validation.case_results} != expected or len(
        validation.case_results
    ) != 4:
        raise ValueError("validation_cases_incomplete")
    for case in validation.case_results:
        check = case.check
        if check.rule_id != rule.rule_id or check.revision != rule.revision:
            raise ValueError("validation_revision_mismatch")
        violation = case.case_type in {"recorded_violation", "synthetic_violation"}
        if violation:
            valid = check.applicability == "applicable" and check.verdict == "violation"
        else:
            valid = (
                check.applicability == "applicable" and check.verdict == "pass"
            ) or (check.applicability == "inapplicable" and check.verdict is None)
        if not valid:
            raise ValueError("validation_not_discriminating")


def body_free_validation(validation: RuleValidation) -> RuleValidation:
    """Retain host-labelled outcomes without generated explanation or examples."""
    return replace(
        validation,
        case_results=tuple(
            replace(
                case,
                check=replace(case.check, reason="example_tested", evidence_refs=()),
            )
            for case in validation.case_results
        ),
    )


class ResponseRuleRepository:
    """Persist immutable definitions and CAS bindings under the existing DB owner."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def _put_revision(self, cursor: sqlite3.Cursor, rule: RuleRevision) -> None:
        row = cursor.execute(
            "SELECT definition_json,schema_version,origin_json,created_at FROM console_response_rule_revisions WHERE rule_id=? AND revision=?",
            (rule.rule_id, rule.revision),
        ).fetchone()
        definition = canonical_json(rule.candidate.model_dump(mode="json"))
        if row is not None:
            if (row[0], row[1], row[2], row[3]) != (
                definition,
                rule.schema_version,
                canonical_json(asdict(rule.origin)),
                rule.created_at,
            ):
                raise ValueError("immutable_rule_revision")
            return
        cursor.execute(
            "INSERT INTO console_response_rule_revisions VALUES (?,?,?,?,?,?)",
            (
                rule.rule_id,
                rule.revision,
                definition,
                rule.schema_version,
                canonical_json(asdict(rule.origin)),
                rule.created_at,
            ),
        )

    def _assert_source(self, cursor: sqlite3.Cursor, source: RuleSource) -> None:
        if source.conversation_id is None:
            return
        row = cursor.execute(
            "SELECT conversation_id,version,sender,deleted FROM messages WHERE id=?",
            (source.message_id,),
        ).fetchone()
        if (
            row is None
            or row[0] != source.conversation_id
            or row[1] != source.message_version
            or row[2] != "assistant"
            or row[3]
        ):
            raise ValueError("stale_rule_source")

    def _put_validation(
        self,
        cursor: sqlite3.Cursor,
        rule: RuleRevision,
        validation: RuleValidation,
        *,
        detached: bool = False,
    ) -> None:
        detached = (
            detached
            or cursor.execute(
                "SELECT 1 FROM console_response_rule_bindings WHERE rule_id=? AND revision=? AND scope_kind!='chat' LIMIT 1",
                (rule.rule_id, rule.revision),
            ).fetchone()
            is not None
        )
        if detached:
            validation = body_free_validation(validation)
        source = validation.source
        cursor.execute(
            "INSERT INTO console_response_rule_validations VALUES (?,?,?,?,?,?) ON CONFLICT(rule_id,revision) DO UPDATE SET validation_json=excluded.validation_json,conversation_id=excluded.conversation_id,message_id=excluded.message_id,message_version=excluded.message_version",
            (
                rule.rule_id,
                rule.revision,
                canonical_json(asdict(validation)),
                source.conversation_id if not detached else None,
                source.message_id if source.conversation_id and not detached else None,
                source.message_version,
            ),
        )

    def _set_binding(
        self, cursor: sqlite3.Cursor, binding: RuleBinding, expected: int
    ) -> RuleBinding:
        row = cursor.execute(
            "SELECT binding_revision FROM console_response_rule_bindings WHERE scope_kind=? AND scope_id=? AND rule_id=?",
            (binding.scope.kind, binding.scope.owner_id, binding.rule_id),
        ).fetchone()
        if (row[0] if row else 0) != expected:
            raise RuleBindingConflict("rule_binding_changed")
        updated = RuleBinding(
            binding.scope,
            binding.rule_id,
            binding.revision,
            binding.state,
            expected + 1,
        )
        conversation = binding.scope.owner_id if binding.scope.kind == "chat" else None
        cursor.execute(
            "INSERT INTO console_response_rule_bindings VALUES (?,?,?,?,?,?,?) ON CONFLICT(scope_kind,scope_id,rule_id) DO UPDATE SET revision=excluded.revision,state=excluded.state,binding_revision=excluded.binding_revision",
            (
                binding.scope.kind,
                binding.scope.owner_id,
                binding.rule_id,
                binding.revision,
                binding.state,
                expected + 1,
                conversation,
            ),
        )
        return updated

    def _retain_validation_provenance(
        self, cursor: sqlite3.Cursor, rule: RuleRevision, validation: RuleValidation
    ) -> None:
        """Keep independently promoted calibration after private source cleanup."""
        self._put_validation(cursor, rule, validation, detached=True)

    def historical_machine_parent(
        self, conversation_id: str, assistant_id: str
    ) -> str | None:
        """Read inert host-recorded ancestry; this never recreates execution authority."""
        with self.db.transaction() as cursor:
            rows = cursor.execute(
                "SELECT parent_assistant_message_id FROM console_machine_followup_receipts WHERE conversation_id=? AND assistant_message_id=? LIMIT 2",
                (conversation_id, assistant_id),
            ).fetchall()
        return rows[0][0] if len(rows) == 1 else None

    def list_bindings(self, scope: RuleScope) -> tuple[RuleBinding, ...]:
        with self.db.transaction() as cursor:
            return tuple(
                RuleBinding(scope, r[0], r[1], r[2], r[3])
                for r in cursor.execute(
                    "SELECT rule_id,revision,state,binding_revision FROM console_response_rule_bindings WHERE scope_kind=? AND scope_id=? ORDER BY rule_id",
                    (scope.kind, scope.owner_id),
                )
            )

    def get_revision(self, rule_id: str, revision: int) -> RuleRevision:
        with self.db.transaction() as cursor:
            row = cursor.execute(
                "SELECT definition_json,schema_version,origin_json,created_at FROM console_response_rule_revisions WHERE rule_id=? AND revision=?",
                (rule_id, revision),
            ).fetchone()
        if row is None:
            raise KeyError((rule_id, revision))
        return RuleRevision(
            rule_id,
            revision,
            RuleCandidate.model_validate_json(row[0]),
            row[1],
            RuleSource(**json.loads(row[2])),
            row[3],
        )

    def get_validation(self, rule_id: str, revision: int) -> RuleValidation | None:
        with self.db.transaction() as cursor:
            row = cursor.execute(
                "SELECT validation_json FROM console_response_rule_validations WHERE rule_id=? AND revision=?",
                (rule_id, revision),
            ).fetchone()
        return read_validation(json.loads(row[0])) if row else None

    def _save_draft(
        self,
        cursor: sqlite3.Cursor,
        scope: RuleScope,
        source: RuleSource,
        result: RuleLearningResult,
        complaint: str,
    ) -> None:
        if len(complaint.encode("utf-8")) > 8192:
            raise ValueError("complaint_too_large")
        rule = result.rule
        if rule is not None:
            self._put_revision(cursor, rule)
        if rule is not None and result.validation is not None:
            self._put_validation(cursor, rule, result.validation)
            types = {
                case.case_id: case.case_type for case in result.validation.case_results
            }
            for case_id, inputs in result.fixtures.items():
                case_type = types[case_id]
                if (
                    case_type != "recorded_violation"
                    and len(inputs.response_text.encode("utf-8")) > 8192
                ):
                    raise ValueError("fixture_too_large")
                cursor.execute(
                    "INSERT INTO console_response_rule_fixtures VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(rule_id,revision,case_id) DO UPDATE SET input_json=excluded.input_json",
                    (
                        rule.rule_id,
                        rule.revision,
                        case_id,
                        case_type,
                        source.conversation_id,
                        source.message_id if source.conversation_id else None,
                        source.message_version,
                        (
                            None
                            if case_type == "recorded_violation"
                            else canonical_json(asdict(inputs))
                        ),
                    ),
                )
        document = {
            "state": "inactive" if result.state == "active" else result.state,
            "rule": revision_document(rule) if rule else None,
            "validation": asdict(result.validation) if result.validation else None,
            "reason": result.reason,
        }
        cursor.execute(
            "INSERT INTO console_response_rule_drafts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                uuid4().hex,
                scope.kind,
                scope.owner_id,
                rule.rule_id if rule else None,
                rule.revision if rule else None,
                canonical_json(asdict(source)),
                canonical_json(document),
                complaint,
                source.conversation_id,
                source.message_id if source.conversation_id else None,
            ),
        )

    def list_drafts(self, scope: RuleScope) -> tuple[RuleLearningResult, ...]:
        with self.db.transaction() as cursor:
            rows = cursor.execute(
                "SELECT result_json FROM console_response_rule_drafts WHERE scope_kind=? AND scope_id=? ORDER BY rowid",
                (scope.kind, scope.owner_id),
            ).fetchall()
            results = []
            for row in rows:
                document = json.loads(row[0])
                rule = read_revision(document["rule"]) if document["rule"] else None
                validation = (
                    read_validation(document["validation"])
                    if document["validation"]
                    else None
                )
                fixtures = {}
                if rule:
                    for case in cursor.execute(
                        "SELECT case_id,input_json,message_id,message_version FROM console_response_rule_fixtures WHERE rule_id=? AND revision=?",
                        (rule.rule_id, rule.revision),
                    ).fetchall():
                        if case[1] is not None:
                            fixtures[case[0]] = read_input(json.loads(case[1]))
                        elif case[2]:
                            message = cursor.execute(
                                "SELECT content,version,deleted FROM messages WHERE id=?",
                                (case[2],),
                            ).fetchone()
                            if message and message[1] == case[3] and not message[2]:
                                fixtures[case[0]] = RuleInput(
                                    "", message[0], (), False, None
                                )
                results.append(
                    RuleLearningResult(
                        document["state"],
                        rule,
                        validation,
                        fixtures,
                        document["reason"],
                    )
                )
        return tuple(results)

    def _garbage_collect(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "DELETE FROM console_response_rule_revisions AS r WHERE NOT EXISTS (SELECT 1 FROM console_response_rule_bindings b WHERE b.rule_id=r.rule_id AND b.revision=r.revision) AND NOT EXISTS (SELECT 1 FROM console_response_rule_drafts d WHERE d.rule_id=r.rule_id AND d.revision=r.revision)"
        )

    def remove_source(
        self,
        conversation_id: str,
        message_id: str | None,
        *,
        permanent: bool,
        cursor: sqlite3.Cursor,
    ) -> None:
        if not permanent:
            return
        for statement in (
            "DELETE FROM console_response_rule_drafts WHERE conversation_id=? AND (? IS NULL OR message_id=?)",
            "DELETE FROM console_response_rule_validations WHERE conversation_id=? AND (? IS NULL OR message_id=?)",
            "DELETE FROM console_response_rule_fixtures WHERE conversation_id=? AND (? IS NULL OR message_id=?)",
            "DELETE FROM console_response_rule_assessments WHERE conversation_id=? AND (? IS NULL OR message_id=?)",
        ):
            cursor.execute(statement, (conversation_id, message_id, message_id))
        if message_id is None:
            cursor.execute(
                "DELETE FROM console_response_rule_bindings WHERE scope_kind='chat' AND scope_id=?",
                (conversation_id,),
            )
        self._garbage_collect(cursor)
