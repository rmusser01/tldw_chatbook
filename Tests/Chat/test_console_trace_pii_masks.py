"""Immutable built-in PII masks for semantic trace projections."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_custom_pii import (
    CUSTOM_PII_RULESET_UNAVAILABLE,
    redact_pii_value_for_ruleset_revision,
    redact_pii_value_with_custom_rules,
    register_custom_pii_ruleset,
    validate_custom_pii_rules_config,
)
from tldw_chatbook.Chat.console_trace_redaction import (
    BUILTIN_PII_RULESET_REVISION_ID,
    BUILTIN_PII_RULESET_VERSION,
    CREDENTIAL_FILTER_VERSION,
    PII_DETECTOR_UNAVAILABLE,
    BuiltInPIIDetector,
    PIIRedactionSpan,
    apply_frozen_pii_masks,
    apply_pii_mask,
    merge_pii_spans,
    redact_pii_value,
)
from tldw_chatbook.Chat.console_trace_regex_worker import (
    CUSTOM_PII_WORKER_TIMEOUT,
    CustomPIIWorkerLimits,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService
from tldw_chatbook.Chat.console_semantic_revision import (
    SemanticRevisionCoordinator,
    project_semantic_revision_trace_message,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_unicode_codepoint_spans_are_deterministic_and_mask_without_mutation() -> None:
    source = "👩🏽‍💻 Reach Élise at elise@example.test or +1 (415) 555-2671."
    detector = BuiltInPIIDetector()

    first = detector.detect(source)
    second = detector.detect(source)

    assert first == second
    assert first.available is True
    assert [(span.start_codepoint, span.end_codepoint) for span in first.spans] == [
        (source.index("elise@"), source.index("elise@") + len("elise@example.test")),
        (source.index("+1"), source.index("+1") + len("+1 (415) 555-2671")),
    ]
    assert apply_pii_mask(source, first.spans) == (
        "👩🏽‍💻 Reach Élise at [PII omitted] or [PII omitted]."
    )
    assert "elise@example.test" in source
    with pytest.raises(FrozenInstanceError):
        first.spans[0].category = "changed"  # type: ignore[misc]


def test_overlapping_candidates_union_with_mixed_category_and_stable_rule_ids() -> None:
    merged = merge_pii_spans(
        (
            PIIRedactionSpan(4, 12, "email", "builtin-email", "v1"),
            PIIRedactionSpan(1, 7, "phone", "builtin-phone", "v1"),
            PIIRedactionSpan(20, 24, "email", "builtin-email", "v1"),
        )
    )

    assert merged == (
        PIIRedactionSpan(
            1,
            12,
            "mixed",
            "builtin-email+builtin-phone",
            "v1",
        ),
        PIIRedactionSpan(20, 24, "email", "builtin-email", "v1"),
    )


def test_pii_detector_bounds_dense_work_and_fails_content_free() -> None:
    too_long = BuiltInPIIDetector(max_text_codepoints=16).detect(
        "secret-person@example.test"
    )
    too_many = BuiltInPIIDetector(max_matches=2).detect(
        "a@example.test b@example.test c@example.test"
    )

    for result in (too_long, too_many):
        assert result.available is False
        assert result.spans == ()
        assert result.omission_reason_code == PII_DETECTOR_UNAVAILABLE
        assert "example" not in repr(result)


def test_pii_in_mapping_keys_is_masked_with_content_free_ordinal_path() -> None:
    source = {"person@example.test": {"value": "keep"}}

    redacted = redact_pii_value(source)

    assert redacted.available is True
    assert redacted.value == {"[PII omitted]": {"value": "keep"}}
    assert [item.field_path for item in redacted.field_redactions] == ["$/@0#key"]
    assert "person@example.test" not in repr(redacted.field_redactions)
    masks = {item.field_path: (item.span,) for item in redacted.field_redactions}
    assert apply_frozen_pii_masks(source, masks) == redacted.value


def test_custom_spans_merge_with_builtins_without_mutating_source() -> None:
    source = {"content": "Email elise@example.test; customer-ABCDWXYZ"}
    ruleset = validate_custom_pii_rules_config(
        {
            "version": 1,
            "revision_id": "11111111-1111-4111-8111-111111111111",
            "rules": [
                {
                    "id": "customer-id",
                    "label": "Customer ID",
                    "category": "customer_id",
                    "pattern": r"customer-[A-Z]{8}",
                    "flags": [],
                    "enabled": True,
                    "priority": 10,
                }
            ],
        }
    ).ruleset
    assert ruleset is not None

    redacted = redact_pii_value_with_custom_rules(source, ruleset)

    assert redacted.available is True
    assert redacted.value == {"content": "Email [PII omitted]; [PII omitted]"}
    assert [item.span.rule_id for item in redacted.field_redactions] == [
        "builtin-email",
        "customer-id",
    ]
    assert source == {"content": "Email elise@example.test; customer-ABCDWXYZ"}


def test_custom_worker_timeout_makes_the_component_content_free_unavailable() -> None:
    source = {"content": "a" * 30_000 + "!"}
    ruleset = validate_custom_pii_rules_config(
        {
            "version": 1,
            "revision_id": "11111111-1111-4111-8111-111111111111",
            "rules": [
                {
                    "id": "pathological",
                    "label": "Pathological",
                    "category": "custom",
                    "pattern": r"(a+)+$",
                    "flags": [],
                    "enabled": True,
                    "priority": 10,
                }
            ],
        }
    ).ruleset
    assert ruleset is not None

    redacted = redact_pii_value_with_custom_rules(
        source,
        ruleset,
        worker_limits=CustomPIIWorkerLimits(deadline_ms=50),
    )

    assert redacted.available is False
    assert redacted.value is None
    assert redacted.field_redactions == ()
    assert redacted.omission_reason_code == CUSTOM_PII_WORKER_TIMEOUT
    assert "a" * 100 not in repr(redacted)


def test_missing_custom_ruleset_revision_fails_closed_without_source_content() -> None:
    source = {"content": "customer-ABCDWXYZ"}

    redacted = redact_pii_value_for_ruleset_revision(
        source,
        "33333333-3333-4333-8333-333333333333",
    )

    assert redacted.available is False
    assert redacted.value is None
    assert redacted.field_redactions == ()
    assert redacted.omission_reason_code == CUSTOM_PII_RULESET_UNAVAILABLE
    assert "customer" not in repr(redacted)


def test_repository_reuses_content_free_masks_per_frozen_policy() -> None:
    database = CharactersRAGDB(":memory:", "trace-pii-mask-test")
    repository = ConsoleTraceRepository()
    source = "Contact elise@example.test"
    detection = BuiltInPIIDetector().detect(source)
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=True,
        pii_ruleset_revision_id=new_opaque_id(),
    )
    conversation_id = database.add_conversation({"title": "PII masks"})
    assert conversation_id is not None
    try:
        with database.transaction(immediate=True) as cursor:
            repository.ensure_policy(cursor, policy)
            revision = repository.ensure_semantic_revision(
                cursor,
                source_conversation_id=conversation_id,
                source_message_id=new_opaque_id(),
                revision_sequence=0,
                normalized_role="user",
                content_kind="text",
                creation_reason="message_create",
                live_locator_retired_at="2026-08-31T00:00:00Z",
            )
            before_spans = repository.get_graph_epoch(cursor)
            first = repository.ensure_redaction_spans(
                cursor,
                policy_id=policy.policy_id,
                semantic_revision_id=revision.revision_id,
                artifact_id=None,
                field_path="$.content",
                spans=detection.spans,
            )
            assert repository.get_graph_epoch(cursor) == before_spans + 1
            second = repository.ensure_redaction_spans(
                cursor,
                policy_id=policy.policy_id,
                semantic_revision_id=revision.revision_id,
                artifact_id=None,
                field_path="$.content",
                spans=detection.spans,
            )
            assert repository.get_graph_epoch(cursor) == before_spans + 1

        assert second == first
        assert len(first) == 1
        durable = (
            database.get_connection()
            .execute("SELECT * FROM console_trace_redaction_spans")
            .fetchone()
        )
        assert "elise" not in " ".join(str(item) for item in durable)
        assert "example.test" not in " ".join(str(item) for item in durable)
        assert BUILTIN_PII_RULESET_VERSION not in {policy.pii_ruleset_revision_id}
    finally:
        database.close_connection()


def test_canonical_message_stays_unchanged_while_trace_projection_is_masked() -> None:
    database = CharactersRAGDB(":memory:", "trace-pii-canonical-test")
    repository = ConsoleTraceRepository()
    conversation_id = database.add_conversation({"title": "PII canonical"})
    assert conversation_id is not None
    source = "Contact elise@example.test"
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": source,
        }
    )
    assert message_id is not None
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=True,
        pii_ruleset_revision_id=new_opaque_id(),
    )
    try:
        with database.transaction(immediate=True) as cursor:
            revision = SemanticRevisionCoordinator(database).ensure_current_revision(
                cursor,
                message_id=message_id,
            )
            repository.ensure_policy(cursor, policy)
            detection = BuiltInPIIDetector().detect(source)
            repository.ensure_redaction_spans(
                cursor,
                policy_id=policy.policy_id,
                semantic_revision_id=revision.revision_id,
                artifact_id=None,
                field_path="$/@0",
                spans=detection.spans,
            )
            safe = project_semantic_revision_trace_message(
                cursor,
                revision_id=revision.revision_id,
                expected_conversation_id=conversation_id,
                policy_id=policy.policy_id,
            )
            full = project_semantic_revision_trace_message(
                cursor,
                revision_id=revision.revision_id,
                expected_conversation_id=conversation_id,
                policy_id=policy.policy_id,
            )
            canonical = cursor.execute(
                "SELECT content FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()[0]

        assert safe == full == {"role": "user", "content": "Contact [PII omitted]"}
        assert canonical == source
    finally:
        database.close_connection()


def test_retired_revision_remains_available_through_masked_trace_artifact() -> None:
    database = CharactersRAGDB(":memory:", "trace-pii-retired-test")
    repository = ConsoleTraceRepository()
    coordinator = SemanticRevisionCoordinator(database, repository=repository)
    conversation_id = database.add_conversation({"title": "Retired PII trace"})
    assert conversation_id is not None
    source = "Contact elise@example.test"
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": source,
        }
    )
    assert message_id is not None
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version=CREDENTIAL_FILTER_VERSION,
        pii_redaction_enabled=True,
        pii_ruleset_revision_id=BUILTIN_PII_RULESET_REVISION_ID,
    )
    try:
        with database.transaction(immediate=True) as cursor:
            revision = coordinator.ensure_current_revision(
                cursor,
                message_id=message_id,
            )
            repository.ensure_policy(cursor, policy)
            coordinator._materialize_policies(
                cursor,
                revision_id=revision.revision_id,
                policy_ids=(policy.policy_id,),
                envelope=coordinator._message_envelope(cursor, message_id),
            )
            coordinator.mutate_message(
                cursor,
                message_id=message_id,
                creation_reason="message_edit",
                mutate=lambda mutation_cursor: mutation_cursor.execute(
                    "UPDATE messages SET content = ? WHERE id = ?",
                    ("Replacement without PII", message_id),
                ),
            )

            projected = project_semantic_revision_trace_message(
                cursor,
                revision_id=revision.revision_id,
                expected_conversation_id=conversation_id,
                policy_id=policy.policy_id,
            )
            canonical = cursor.execute(
                "SELECT content FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()[0]

        assert projected == {"role": "user", "content": "Contact [PII omitted]"}
        assert canonical == "Replacement without PII"
    finally:
        database.close_connection()


def test_retired_revision_uses_registered_custom_rules_then_needs_no_pattern() -> None:
    database = CharactersRAGDB(":memory:", "trace-custom-pii-retired-test")
    repository = ConsoleTraceRepository()
    coordinator = SemanticRevisionCoordinator(database, repository=repository)
    conversation_id = database.add_conversation({"title": "Retired custom PII"})
    assert conversation_id is not None
    source = "Account customer-ABCDWXYZ"
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": source,
        }
    )
    assert message_id is not None
    revision_id = "44444444-4444-4444-8444-444444444444"
    ruleset = validate_custom_pii_rules_config(
        {
            "version": 1,
            "revision_id": revision_id,
            "rules": [
                {
                    "id": "customer-id",
                    "label": "Customer ID",
                    "category": "customer_id",
                    "pattern": r"customer-[A-Z]{8}",
                    "flags": [],
                    "enabled": True,
                    "priority": 10,
                }
            ],
        }
    ).ruleset
    assert ruleset is not None
    assert register_custom_pii_ruleset(ruleset) is True
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version=CREDENTIAL_FILTER_VERSION,
        pii_redaction_enabled=True,
        pii_ruleset_revision_id=revision_id,
    )
    try:
        with database.transaction(immediate=True) as cursor:
            revision = coordinator.ensure_current_revision(
                cursor, message_id=message_id
            )
            repository.ensure_policy(cursor, policy)
            coordinator._materialize_policies(
                cursor,
                revision_id=revision.revision_id,
                policy_ids=(policy.policy_id,),
                envelope=coordinator._message_envelope(cursor, message_id),
            )
            coordinator.mutate_message(
                cursor,
                message_id=message_id,
                creation_reason="message_edit",
                mutate=lambda mutation_cursor: mutation_cursor.execute(
                    "UPDATE messages SET content = ? WHERE id = ?",
                    ("Replacement", message_id),
                ),
            )
            projected = project_semantic_revision_trace_message(
                cursor,
                revision_id=revision.revision_id,
                expected_conversation_id=conversation_id,
                policy_id=policy.policy_id,
            )

        assert projected == {"role": "user", "content": "Account [PII omitted]"}
        assert source not in repr(projected)
    finally:
        database.close_connection()


def test_trace_artifact_runs_custom_worker_once_and_persists_only_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    database = CharactersRAGDB(":memory:", "trace-custom-pii-artifact-test")
    repository = ConsoleTraceRepository()
    service = ConsoleTraceService(repository)
    revision_id = "55555555-5555-4555-8555-555555555555"
    ruleset = validate_custom_pii_rules_config(
        {
            "version": 1,
            "revision_id": revision_id,
            "rules": [
                {
                    "id": "customer-id",
                    "label": "Customer ID",
                    "category": "customer_id",
                    "pattern": r"customer-[A-Z]{8}",
                    "flags": [],
                    "enabled": True,
                    "priority": 10,
                }
            ],
        }
    ).ruleset
    assert ruleset is not None
    assert register_custom_pii_ruleset(ruleset) is True
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version=CREDENTIAL_FILTER_VERSION,
        pii_redaction_enabled=True,
        pii_ruleset_revision_id=revision_id,
    )
    calls = 0
    from tldw_chatbook.Chat import console_trace_regex_worker as regex_worker

    real_run = regex_worker.run_custom_pii_batch

    def counted_run(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(regex_worker, "run_custom_pii_batch", counted_run)
    try:
        with database.transaction(immediate=True) as cursor:
            repository.ensure_policy(cursor, policy)
            reference = service._store_artifact(
                cursor,
                {"argument": "customer-ABCDWXYZ"},
                policy=policy,
            )
            artifact = repository.get_artifact(cursor, reference.content_id)
            spans = cursor.execute(
                "SELECT field_path, rule_id FROM console_trace_redaction_spans"
            ).fetchall()

        assert calls == 1
        assert artifact is not None
        assert b"customer-ABCDWXYZ" not in artifact.sanitized_bytes
        assert b"[PII omitted]" in artifact.sanitized_bytes
        assert [tuple(row) for row in spans] == [("$/@0", "customer-id")]
    finally:
        database.close_connection()
