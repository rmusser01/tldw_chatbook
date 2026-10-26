"""Native response-rule history with explicit historical provenance mapping."""

import json
from dataclasses import asdict, replace
from typing import Literal, TypedDict

from pydantic import TypeAdapter

from tldw_chatbook.Chat.response_rules.models import (
    RULE_SCHEMA_VERSION,
    RuleAssessment,
    RuleBinding,
    RuleInput,
    RuleRevision,
    RuleScope,
    RuleSource,
    RuleValidation,
    canonical_json,
)
from tldw_chatbook.Chat.response_rules.repository import (
    revision_document,
    validate_activation,
)

from .mobile_content_graph import _UnprovedReference

_RULE_TABLES = (
    "console_response_rule_revisions",
    "console_response_rule_validations",
    "console_response_rule_fixtures",
    "console_response_rule_bindings",
    "console_response_rule_drafts",
    "console_response_rule_assessments",
)


class _DraftDocument(TypedDict):
    state: Literal["active", "inactive", "cancelled", "stale"]
    rule: RuleRevision | None
    validation: RuleValidation | None
    reason: str


# JSON mode validates original array/scalar shapes before native dataclass
# __post_init__ can coerce them. Only these fixed installed types are admitted.
_SOURCE_JSON = TypeAdapter(RuleSource)
_REVISION_JSON = TypeAdapter(RuleRevision)
_VALIDATION_JSON = TypeAdapter(RuleValidation)
_INPUT_JSON = TypeAdapter(RuleInput)
_ASSESSMENT_JSON = TypeAdapter(RuleAssessment)
_DRAFT_JSON = TypeAdapter(_DraftDocument)

# Installed response-rule CHECK limits (DB/recovery_core_schema.py), UTF-8 bytes.
_SOURCE_JSON_BYTES = 8192
_VALIDATION_JSON_BYTES = 32768
_RESULT_JSON_BYTES = 65536


def _encoded_rule_json(document, max_bytes):
    encoded = canonical_json(document)
    if len(encoded.encode("utf-8")) > max_bytes:
        raise ValueError("native_rule_json_too_large")
    return encoded


def _source(builder, source):
    if (
        type(source) is not RuleSource
        or type(source.message_version) is not int
        or source.message_version < 1
        or type(source.authority_epoch) is not int
        or source.authority_epoch < 0
    ):
        raise ValueError("invalid_rule_source")
    for name in (
        "profile_id",
        "session_id",
        "branch_id",
        "message_id",
        "parent_turn_id",
        "operation_id",
        "settlement_id",
        "effective_digest",
    ):
        if type(getattr(source, name)) is not str:
            raise ValueError("invalid_rule_source_field")
    if source.conversation_id is None:
        # A detached temporary source is historical, not a durable namespace.
        return source
    message = builder.by_id["messages"].get(source.message_id)
    if message is None or message["conversation_id"] != source.conversation_id:
        raise _UnprovedReference("response_rule_source_message", source.message_id)
    branch = source.branch_id
    if branch in builder.by_id["messages"]:
        if (
            builder.by_id["messages"][branch]["conversation_id"]
            != source.conversation_id
        ):
            raise _UnprovedReference("response_rule_branch", branch)
        branch = builder.ref("message", branch)
    # Match native adopt_temporary's known-branch grammar. Unknown branch text
    # and profile/session/turn/operation/epoch/digests remain historical.
    return replace(
        source,
        conversation_id=builder.ref("conversation", source.conversation_id),
        message_id=builder.ref("message", source.message_id),
        branch_id=branch,
    )


def _rule_key(builder, rule_id, revision):
    if type(revision) is not int or revision < 1:
        raise ValueError("invalid_rule_revision")
    if not any(
        row["rule_id"] == rule_id and row["revision"] == revision
        for row in builder.source["console_response_rule_revisions"]
    ):
        raise _UnprovedReference("response_rule_revision", (rule_id, revision))
    return builder.ref("response_rule_revision", (rule_id, revision))


def _revision(builder, rule):
    rule_id, revision = _rule_key(builder, rule.rule_id, rule.revision)
    if rule.schema_version != RULE_SCHEMA_VERSION:
        raise ValueError("invalid_rule_schema")
    return replace(
        rule, rule_id=rule_id, revision=revision, origin=_source(builder, rule.origin)
    )


def _check(builder, check):
    rule_id, revision = _rule_key(builder, check.rule_id, check.revision)
    if check.applicability not in {
        "applicable",
        "inapplicable",
        "unknown",
    } or check.verdict not in {None, "pass", "violation", "couldnt_verify"}:
        raise ValueError("invalid_rule_check")
    return replace(check, rule_id=rule_id, revision=revision)


def _validation(builder, validation):
    return replace(
        validation,
        source=_source(builder, validation.source),
        case_results=tuple(
            replace(case, check=_check(builder, case.check))
            for case in validation.case_results
        ),
    )


def _scope(builder, kind, owner):
    scope = RuleScope(kind, owner)
    if scope.kind == "global":
        raise _UnprovedReference("response_rule_global_namespace", owner)
    return RuleScope(
        kind, builder.ref("conversation" if kind == "chat" else "workspace", owner)
    )


def _optional_source_columns(builder, row):
    return {
        "conversation_id": builder.ref(
            "conversation", row["conversation_id"], optional=True
        ),
        "message_id": builder.ref("message", row["message_id"], optional=True),
    }


def _rules(builder):
    native_revisions, native_validations = {}, {}

    def revision(row):
        original = _REVISION_JSON.validate_json(
            canonical_json(
                {
                    "rule_id": row["rule_id"],
                    "revision": row["revision"],
                    "candidate": json.loads(row["definition_json"]),
                    "schema_version": row["schema_version"],
                    "origin": json.loads(row["origin_json"]),
                    "created_at": row["created_at"],
                }
            ),
            strict=True,
            extra="forbid",
        )
        mapped = _revision(builder, original)
        origin_json = _encoded_rule_json(asdict(mapped.origin), _SOURCE_JSON_BYTES)
        native_revisions[(row["rule_id"], row["revision"])] = mapped
        builder.emit(
            "console_response_rule_revisions",
            row,
            row
            | {
                "rule_id": mapped.rule_id,
                "origin_json": origin_json,
            },
        )

    builder.handle("console_response_rule_revisions", revision)

    def validation(row):
        rule_id, _ = _rule_key(builder, row["rule_id"], row["revision"])
        original = _VALIDATION_JSON.validate_json(
            row["validation_json"], strict=True, extra="forbid"
        )
        association = row["conversation_id"], row["message_id"]
        expected = (
            original.source.conversation_id,
            original.source.message_id if original.source.conversation_id else None,
        )
        # Native detached validation intentionally clears both relational IDs.
        if (
            association not in {(None, None), expected}
            or row["message_version"] != original.source.message_version
        ):
            raise ValueError("invalid_validation_association")
        mapped = _validation(builder, original)
        validation_json = _encoded_rule_json(asdict(mapped), _VALIDATION_JSON_BYTES)
        native_validations[(row["rule_id"], row["revision"])] = mapped
        builder.emit(
            "console_response_rule_validations",
            row,
            row
            | _optional_source_columns(builder, row)
            | {"rule_id": rule_id, "validation_json": validation_json},
        )

    builder.handle("console_response_rule_validations", validation)

    def fixture(row):
        rule_id, _ = _rule_key(builder, row["rule_id"], row["revision"])
        if row["case_type"] not in {
            "recorded_violation",
            "synthetic_violation",
            "synthetic_correction",
            "synthetic_acceptable",
        }:
            raise ValueError("invalid_rule_fixture_kind")
        if row["input_json"] is not None:
            _INPUT_JSON.validate_json(row["input_json"], strict=True, extra="forbid")
        builder.emit(
            "console_response_rule_fixtures",
            row,
            row | _optional_source_columns(builder, row) | {"rule_id": rule_id},
        )

    builder.handle("console_response_rule_fixtures", fixture)

    def draft(row):
        scope = _scope(builder, row["scope_kind"], row["scope_id"])
        original_source = _SOURCE_JSON.validate_json(
            row["source_json"], strict=True, extra="forbid"
        )
        if (row["conversation_id"], row["message_id"]) != (
            original_source.conversation_id,
            original_source.message_id if original_source.conversation_id else None,
        ):
            raise ValueError("invalid_draft_association")
        source = _source(builder, original_source)
        document = _DRAFT_JSON.validate_json(
            row["result_json"], strict=True, extra="forbid"
        )
        rule = (
            _revision(builder, document["rule"])
            if document["rule"] is not None
            else None
        )
        if (
            rule is not None
            and native_revisions.get((row["rule_id"], row["revision"])) != rule
        ):
            raise ValueError("invalid_draft_revision")
        validation = (
            _validation(builder, document["validation"])
            if document["validation"] is not None
            else None
        )
        if (row["rule_id"], row["revision"]) != (
            (document["rule"].rule_id, document["rule"].revision)
            if rule
            else (None, None)
        ):
            raise ValueError("invalid_rule_draft_identity")
        mapped_document = document | {
            "rule": revision_document(rule) if rule else None,
            "validation": asdict(validation) if validation else None,
        }
        builder.emit(
            "console_response_rule_drafts",
            row,
            row
            | _optional_source_columns(builder, row)
            | {
                "draft_id": builder.ref("response_draft", row["draft_id"]),
                "scope_id": scope.owner_id,
                "rule_id": rule.rule_id if rule else None,
                "source_json": _encoded_rule_json(asdict(source), _SOURCE_JSON_BYTES),
                "result_json": _encoded_rule_json(mapped_document, _RESULT_JSON_BYTES),
            },
        )

    builder.handle("console_response_rule_drafts", draft)

    def binding(row):
        scope = _scope(builder, row["scope_kind"], row["scope_id"])
        rule_id = builder.ref("response_rule", row["rule_id"])
        if row["revision"] is not None:
            _rule_key(builder, row["rule_id"], row["revision"])
        original = RuleBinding(
            RuleScope(row["scope_kind"], row["scope_id"]),
            row["rule_id"],
            row["revision"],
            row["state"],
            row["binding_revision"],
        )
        if row["conversation_id"] != (
            row["scope_id"] if row["scope_kind"] == "chat" else None
        ):
            raise ValueError("invalid_rule_scope_association")
        if original.state == "enabled" and original.binding_revision >= 2**63 - 1:
            raise ValueError("unrepresentable_imported_binding_revision")
        if original.state == "enabled":
            key = (row["rule_id"], row["revision"])
            if key not in native_revisions or key not in native_validations:
                raise _UnprovedReference("response_rule_validation", key)
            builder.ref("response_rule_validation", key)
            validate_activation(native_revisions[key], native_validations[key])
        # Match native imported preparation, never alter original/canonical rows.
        mapped = RuleBinding(
            scope,
            rule_id,
            original.revision,
            "disabled" if original.state == "enabled" else original.state,
            original.binding_revision + (original.state == "enabled"),
        )
        builder.emit(
            "console_response_rule_bindings",
            row,
            row
            | {
                "scope_id": scope.owner_id,
                "rule_id": rule_id,
                "state": mapped.state,
                "binding_revision": mapped.binding_revision,
                "conversation_id": builder.ref(
                    "conversation", row["conversation_id"], optional=True
                ),
            },
        )

    builder.handle("console_response_rule_bindings", binding)

    def assessment(row):
        original = _ASSESSMENT_JSON.validate_json(
            row["assessment_json"], strict=True, extra="forbid"
        )
        if (
            original.assessment_id,
            original.state,
            original.source.conversation_id,
            original.source.message_id,
        ) != (
            row["assessment_id"],
            row["state"],
            row["conversation_id"],
            row["message_id"],
        ):
            raise ValueError("invalid_assessment_association")
        if original.state not in {
            "pending",
            "completed",
            "cancelled",
            "stale",
        } or original.outcome not in {
            "pass",
            "violation",
            "couldnt_verify",
            "no_applicable_rules",
        }:
            raise ValueError("invalid_assessment_state")
        mapped = replace(
            original,
            assessment_id=builder.ref("response_assessment", original.assessment_id),
            source=_source(builder, original.source),
            checks=tuple(_check(builder, check) for check in original.checks),
            state="cancelled" if original.state == "pending" else original.state,
        )
        builder.emit(
            "console_response_rule_assessments",
            row,
            row
            | _optional_source_columns(builder, row)
            | {
                "assessment_id": mapped.assessment_id,
                "state": mapped.state,
                "assessment_json": _encoded_rule_json(
                    asdict(mapped), _RESULT_JSON_BYTES
                ),
            },
        )

    builder.handle("console_response_rule_assessments", assessment)
