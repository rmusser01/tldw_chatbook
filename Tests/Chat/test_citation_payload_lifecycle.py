from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import logging
import sqlite3

import pytest
from pydantic import ValidationError

from Tests.Chat.test_citation_trace_repository import (
    TEST_FINGERPRINT_CODEC,
    _authorization,
    _identity,
    _persist,
    _repository,
    _sealed_write,
)
from tldw_chatbook.Chat.citation_payload_lifecycle import (
    CitationCollectionBarriers,
    CitationPayloadLifecycle,
    PayloadRetentionPolicy,
    PayloadTombstone,
    SnapshotDedupeScope,
)
from tldw_chatbook.Chat.citation_trace_identity import local_trace_namespace
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationHydrationState,
    CitationPersistenceUnavailable,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(
        tmp_path / "citation-payload-lifecycle.sqlite",
        client_id="citation-payload-lifecycle-test",
    )
    yield database
    database.close_connection()


def _policy(*, soft_delete_seconds: int = 0) -> PayloadRetentionPolicy:
    return PayloadRetentionPolicy(
        policy_version="policy-1",
        soft_deleted_owner_retention_seconds=soft_delete_seconds,
        max_collection_batch_size=32,
    )


def _tombstone(
    db: CharactersRAGDB,
    *,
    retain_until: datetime | None = None,
) -> PayloadTombstone:
    return PayloadTombstone(
        profile_id=_identity(db).profile_id,
        origin_namespace="local_payload_v1",
        origin_payload_id="snapshot-1",
        revocation_scope_id="snapshot-1",
        reason_code="source_revoked",
        policy_version="policy-1",
        revoked_at=NOW,
        retain_until=retain_until or NOW + timedelta(days=30),
    )


def _lifecycle(
    db: CharactersRAGDB,
    *,
    soft_delete_seconds: int = 0,
) -> CitationPayloadLifecycle:
    return CitationPayloadLifecycle(
        _repository(db),
        retention_policy=_policy(soft_delete_seconds=soft_delete_seconds),
    )


def _mark_owner_deleted(
    db: CharactersRAGDB,
    *,
    trace_id: str = "trace-1",
    updated_at: datetime = NOW - timedelta(days=1),
) -> None:
    with db.transaction() as cursor:
        cursor.execute(
            """
            UPDATE rag_message_trace_owners
            SET state = 'deleted', updated_at = ?
            WHERE trace_id = ?
            """,
            (updated_at.isoformat(), trace_id),
        )


def _insert_same_origin_snapshot(
    db: CharactersRAGDB,
    *,
    payload_id: str = "snapshot-2",
    changed_field: str | None = None,
    changed_value: str | None = None,
) -> None:
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_evidence_snapshots(
                profile_id, payload_id, governance_scope_id, authority_id,
                confidentiality_policy_id, revocation_scope_id,
                origin_namespace, origin_payload_id, storage_mode,
                redaction_state, retention_class, snapshot_text, title,
                source_identity_json, locator_json, lineage_json,
                transformations_json, content_hash, comparison_fingerprint,
                created_at, retain_until, purged_at
            )
            SELECT
                profile_id, ?, governance_scope_id, authority_id,
                confidentiality_policy_id, revocation_scope_id,
                origin_namespace, origin_payload_id, storage_mode,
                redaction_state, retention_class, snapshot_text, title,
                source_identity_json, locator_json, lineage_json,
                transformations_json, ?, ?, created_at, retain_until, purged_at
            FROM rag_evidence_snapshots
            WHERE payload_id = 'snapshot-1'
            """,
            (payload_id, f"content-hmac-{payload_id}", f"comparison-{payload_id}"),
        )
        if changed_field is not None:
            assert changed_field in {
                "governance_scope_id",
                "authority_id",
                "confidentiality_policy_id",
                "revocation_scope_id",
            }
            cursor.execute(
                f"""
                UPDATE rag_evidence_snapshots
                SET {changed_field} = ?
                WHERE payload_id = ?
                """,
                (changed_value, payload_id),
            )


def test_lifecycle_contracts_are_strict_frozen_bounded_and_utc() -> None:
    scope = SnapshotDedupeScope(
        governance_scope_id="profile-a",
        authority_id="authority-a",
        confidentiality_policy_id="policy-a",
        revocation_scope_id="scope-a",
        exact_content_identity="secret-content-hmac",
    )
    with pytest.raises(ValidationError, match="frozen"):
        scope.authority_id = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SnapshotDedupeScope.model_validate(
            {
                **scope.model_dump(),
                "governance_scope_id": None,
            },
            strict=True,
        )
    with pytest.raises(ValidationError):
        PayloadRetentionPolicy(
            policy_version="policy-1",
            soft_deleted_owner_retention_seconds=-1,
            max_collection_batch_size=1,
        )
    with pytest.raises(ValidationError, match="timezone-aware"):
        PayloadTombstone(
            profile_id="profile-a",
            origin_namespace="local",
            origin_payload_id="payload-a",
            revocation_scope_id="scope-a",
            reason_code="revoked",
            policy_version="policy-1",
            revoked_at=datetime(2026, 7, 24),
            retain_until=NOW,
        )


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("governance_scope_id", "other-profile"),
        ("authority_id", "other-authority"),
        ("confidentiality_policy_id", "other-policy"),
        ("revocation_scope_id", "other-revocation"),
        ("exact_content_identity", "other-content-hmac"),
    ],
)
def test_snapshot_dedupe_requires_the_complete_non_null_secret_scoped_key(
    db: CharactersRAGDB,
    changed_field: str,
    changed_value: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    exact = SnapshotDedupeScope(
        governance_scope_id=identity.profile_id,
        authority_id=identity.local_authority_id,
        confidentiality_policy_id="policy-1",
        revocation_scope_id="snapshot-1",
        exact_content_identity="content-hmac",
    )

    assert repository.find_reusable_snapshot_payload_id(exact) == "snapshot-1"
    changed = exact.model_copy(update={changed_field: changed_value})
    assert repository.find_reusable_snapshot_payload_id(changed) is None


def test_revoke_atomically_purges_all_governed_fields_and_preserves_structure(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    namespace = local_trace_namespace(identity, trace_id="trace-1")
    active = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )
    connection = db.get_connection()
    trace_before = tuple(
        connection.execute(
            """
            SELECT aggregate_json, completeness_at_seal, selected_attempt_id
            FROM rag_citation_traces
            """
        ).fetchone()
    )
    run_identity_before = tuple(
        connection.execute(
            "SELECT trace_id, run_id, run_ordinal, stage FROM rag_evidence_runs"
        ).fetchone()
    )
    snapshot_identity_before = tuple(
        connection.execute(
            """
            SELECT profile_id, payload_id, governance_scope_id, authority_id,
                   confidentiality_policy_id, revocation_scope_id,
                   origin_namespace, origin_payload_id, storage_mode,
                   retention_class, created_at
            FROM rag_evidence_snapshots
            """
        ).fetchone()
    )
    answer_identity_before = tuple(
        connection.execute(
            """
            SELECT profile_id, payload_id, trace_id, attempt_id,
                   retention_class, created_at
            FROM rag_answer_attempt_payloads
            """
        ).fetchone()
    )
    refs_before = [
        tuple(row)
        for row in connection.execute(
            """
            SELECT prompt_set_id, evidence_ordinal, run_id,
                   snapshot_payload_id, marker_ordinal, storage_mode
            FROM rag_trace_evidence_refs
            """
        ).fetchall()
    ]

    result = CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
    ).revoke(
        namespace,
        snapshot_payload_id="snapshot-1",
        tombstone=_tombstone(db),
    )

    assert result == _tombstone(db)
    run = connection.execute("SELECT * FROM rag_evidence_runs").fetchone()
    assert tuple(
        run[key] for key in ("trace_id", "run_id", "run_ordinal", "stage")
    ) == (run_identity_before)
    assert run["redaction_state"] == "purged"
    assert run["run_payload_json"] is None
    assert run["purged_at"] == NOW.isoformat()
    snapshot = connection.execute("SELECT * FROM rag_evidence_snapshots").fetchone()
    assert (
        tuple(
            snapshot[key]
            for key in (
                "profile_id",
                "payload_id",
                "governance_scope_id",
                "authority_id",
                "confidentiality_policy_id",
                "revocation_scope_id",
                "origin_namespace",
                "origin_payload_id",
                "storage_mode",
                "retention_class",
                "created_at",
            )
        )
        == snapshot_identity_before
    )
    for field in (
        "snapshot_text",
        "title",
        "source_identity_json",
        "locator_json",
        "lineage_json",
        "transformations_json",
        "content_hash",
        "comparison_fingerprint",
    ):
        assert snapshot[field] is None
    assert snapshot["redaction_state"] == "purged"
    assert snapshot["purged_at"] == NOW.isoformat()
    answer = connection.execute("SELECT * FROM rag_answer_attempt_payloads").fetchone()
    assert (
        tuple(
            answer[key]
            for key in (
                "profile_id",
                "payload_id",
                "trace_id",
                "attempt_id",
                "retention_class",
                "created_at",
            )
        )
        == answer_identity_before
    )
    assert answer["redaction_state"] == "purged"
    assert answer["answer_body"] is None
    assert answer["body_integrity_hmac"] is None
    assert (
        tuple(
            connection.execute(
                """
            SELECT aggregate_json, completeness_at_seal, selected_attempt_id
            FROM rag_citation_traces
            """
            ).fetchone()
        )
        == trace_before
    )
    assert [
        tuple(row)
        for row in connection.execute(
            """
            SELECT prompt_set_id, evidence_ordinal, run_id,
                   snapshot_payload_id, marker_ordinal, storage_mode
            FROM rag_trace_evidence_refs
            """
        ).fetchall()
    ] == refs_before
    tombstone = connection.execute("SELECT * FROM rag_payload_tombstones").fetchone()
    assert set(tombstone.keys()) == {
        "profile_id",
        "origin_namespace",
        "origin_payload_id",
        "revocation_scope_id",
        "reason_code",
        "policy_version",
        "revoked_at",
        "retain_until",
    }
    assert repository.verify_active_trace_result(active) is False
    hydration = repository.hydrate_trace(
        namespace,
        authorization=_authorization(identity),
    )
    assert hydration.state is CitationHydrationState.REVOKED
    assert hydration.governed_payloads is None


def test_revoke_invalidates_capabilities_issued_by_every_repository_instance(
    db: CharactersRAGDB,
) -> None:
    revoking_repository = _repository(db)
    reading_repository = _repository(db)
    _persist(db, revoking_repository)
    active = reading_repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )
    assert reading_repository.verify_active_trace_result(active) is True

    CitationPayloadLifecycle(
        revoking_repository,
        retention_policy=_policy(),
    ).revoke(
        local_trace_namespace(_identity(db), trace_id="trace-1"),
        snapshot_payload_id="snapshot-1",
        tombstone=_tombstone(db),
    )

    assert reading_repository.verify_active_trace_result(active) is False
    refreshed = reading_repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )
    assert refreshed.state.value == "active"
    assert refreshed.availability_warning.value == "evidence_revoked"
    assert reading_repository.verify_active_trace_result(refreshed) is True
    assert refreshed.summary is not None
    safe_summary = json.dumps(refreshed.summary.model_dump(mode="json"))
    for secret in (
        "private query",
        "private exact submitted evidence",
        "private source title",
        "private-document",
        "Answer [S1].",
        "content-hmac",
    ):
        assert secret not in safe_summary
    hydration = reading_repository.hydrate_trace(
        local_trace_namespace(_identity(db), trace_id="trace-1"),
        authorization=_authorization(_identity(db)),
    )
    assert hydration.state is CitationHydrationState.REVOKED
    assert hydration.governed_payloads is None
    mismatch = reading_repository.get_active_trace_for_message(
        "message-1",
        1,
        "Different answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )
    assert mismatch.state.value != "active"
    object.__setattr__(refreshed, "availability_warning", None)
    assert reading_repository.verify_active_trace_result(refreshed) is False


@pytest.mark.parametrize("completeness", ["partial", "redacted", "unavailable"])
def test_revoke_never_upgrades_noncomplete_seals_to_grounded_warning_capabilities(
    db: CharactersRAGDB,
    completeness: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    with db.transaction() as cursor:
        row = cursor.execute(
            """
            SELECT aggregate_json
            FROM rag_citation_traces
            WHERE trace_id = 'trace-1'
            """
        ).fetchone()
        aggregate = json.loads(row["aggregate_json"])
        aggregate["completeness_at_seal"] = completeness
        cursor.execute(
            """
            UPDATE rag_citation_traces
            SET completeness_at_seal = ?, aggregate_json = ?
            WHERE trace_id = 'trace-1'
            """,
            (
                completeness,
                json.dumps(aggregate, separators=(",", ":"), sort_keys=True),
            ),
        )
    before_revoke = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )
    assert before_revoke.state.value != "active"
    assert before_revoke.summary is None
    CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
    ).revoke(
        local_trace_namespace(_identity(db), trace_id="trace-1"),
        snapshot_payload_id="snapshot-1",
        tombstone=_tombstone(db),
    )

    refreshed = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )

    assert refreshed.state.value != "active"
    assert refreshed.summary is None


@pytest.mark.parametrize(
    "isolated_purge",
    [
        "run",
        "run_redacted",
        "run_corrupt",
        "run_missing",
        "selected_attempt",
        "diagnostic_attempt",
        "snapshot",
    ],
)
def test_untrusted_payload_state_without_tombstone_never_issues_revocation_warning(
    db: CharactersRAGDB,
    isolated_purge: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    with db.transaction() as cursor:
        if isolated_purge == "run":
            cursor.execute(
                """
                UPDATE rag_evidence_runs
                SET redaction_state = 'purged', run_payload_json = NULL,
                    purged_at = ?
                WHERE trace_id = 'trace-1'
                """,
                (NOW.isoformat(),),
            )
        elif isolated_purge == "run_redacted":
            cursor.execute(
                """
                UPDATE rag_evidence_runs
                SET redaction_state = 'redacted', run_payload_json = NULL
                WHERE trace_id = 'trace-1'
                """
            )
        elif isolated_purge == "run_corrupt":
            cursor.execute(
                """
                UPDATE rag_evidence_runs
                SET run_payload_json = 'not-json'
                WHERE trace_id = 'trace-1'
                """
            )
        elif isolated_purge == "run_missing":
            cursor.execute(
                "DELETE FROM rag_trace_evidence_refs WHERE trace_id = 'trace-1'"
            )
            cursor.execute("DELETE FROM rag_evidence_runs WHERE trace_id = 'trace-1'")
        elif isolated_purge == "selected_attempt":
            cursor.execute(
                """
                UPDATE rag_answer_attempt_payloads
                SET redaction_state = 'purged', answer_body = NULL,
                    body_integrity_hmac = NULL, purged_at = ?
                WHERE trace_id = 'trace-1' AND attempt_id = 'attempt-1'
                """,
                (NOW.isoformat(),),
            )
        elif isolated_purge == "diagnostic_attempt":
            cursor.execute(
                """
                INSERT INTO rag_answer_attempt_payloads VALUES (
                    ?, 'diagnostic-payload', 'trace-1', 'diagnostic-attempt',
                    'purged', 'default', NULL, NULL, ?, NULL, ?
                )
                """,
                (identity.profile_id, NOW.isoformat(), NOW.isoformat()),
            )
        else:
            cursor.execute(
                """
                UPDATE rag_evidence_snapshots
                SET redaction_state = 'purged', snapshot_text = NULL,
                    title = NULL, source_identity_json = NULL,
                    locator_json = NULL, lineage_json = NULL,
                    transformations_json = NULL, content_hash = NULL,
                    comparison_fingerprint = NULL, purged_at = ?
                WHERE payload_id = 'snapshot-1'
                """,
                (NOW.isoformat(),),
            )

    result = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )

    assert result.state.value != "active"
    assert result.summary is None
    hydration = repository.hydrate_trace(
        local_trace_namespace(identity, trace_id="trace-1"),
        authorization=_authorization(identity),
    )
    assert hydration.state is not CitationHydrationState.AUTHORIZED
    assert hydration.state is not CitationHydrationState.REVOKED


def test_revoke_purges_run_and_attempt_payloads_for_every_shared_snapshot_reference(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    connection = db.get_connection()
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_citation_traces(
                profile_id, trace_id, schema_version, request_id, generation_id,
                origin_scope_id, origin, lifecycle, completeness_at_seal,
                selected_attempt_id, policy_version, aggregate_json,
                visibility_state, created_at, sealed_at
            ) VALUES (?, 'trace-2', 1, 'request-2', 'generation-2', ?,
                      'local', 'sealed', 'complete', 'attempt-2', 'policy-1',
                      '{}', 'active', ?, ?)
            """,
            (
                identity.profile_id,
                identity.profile_id,
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        cursor.execute(
            """
            INSERT INTO rag_evidence_runs VALUES (
                ?, 'trace-2', 'run-2', 1, 'retrieval', 'available',
                '{"payload_id":"run-payload-2"}', ?, ?, NULL
            )
            """,
            (identity.profile_id, NOW.isoformat(), NOW.isoformat()),
        )
        cursor.execute(
            """
            INSERT INTO rag_answer_attempt_payloads VALUES (
                ?, 'answer-payload-2', 'trace-2', 'attempt-2',
                'available', 'default', 'Other answer', 'integrity',
                ?, NULL, NULL
            )
            """,
            (identity.profile_id, NOW.isoformat()),
        )
        cursor.execute(
            """
            INSERT INTO rag_answer_attempt_payloads VALUES (
                ?, 'answer-payload-2-diagnostic', 'trace-2',
                'attempt-2-diagnostic', 'available', 'default',
                'Diagnostic answer', 'diagnostic-integrity', ?, NULL, NULL
            )
            """,
            (identity.profile_id, NOW.isoformat()),
        )
        cursor.execute(
            """
            INSERT INTO rag_trace_evidence_refs VALUES (
                ?, 'trace-2', 'prompt-2', 1, 'run-2',
                'snapshot-1', 1, 'embedded'
            )
            """,
            (identity.profile_id,),
        )

    CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
    ).revoke(
        local_trace_namespace(identity, trace_id="trace-1"),
        snapshot_payload_id="snapshot-1",
        tombstone=_tombstone(db),
    )

    assert {
        tuple(row)
        for row in connection.execute(
            """
            SELECT trace_id, redaction_state, run_payload_json
            FROM rag_evidence_runs
            """
        ).fetchall()
    } == {
        ("trace-1", "purged", None),
        ("trace-2", "purged", None),
    }
    assert {
        tuple(row)
        for row in connection.execute(
            """
            SELECT payload_id, trace_id, redaction_state,
                   answer_body, body_integrity_hmac
            FROM rag_answer_attempt_payloads
            """
        ).fetchall()
    } == {
        ("answer-payload-1", "trace-1", "purged", None, None),
        ("answer-payload-2", "trace-2", "purged", None, None),
        (
            "answer-payload-2-diagnostic",
            "trace-2",
            "purged",
            None,
            None,
        ),
    }
    assert (
        connection.execute(
            """
        SELECT completeness_at_seal
        FROM rag_citation_traces
        WHERE trace_id = 'trace-2'
        """
        ).fetchone()[0]
        == "complete"
    )


def test_revoke_purges_every_payload_id_and_trace_for_one_coherent_origin(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    _insert_same_origin_snapshot(db)
    connection = db.get_connection()
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_citation_traces(
                profile_id, trace_id, schema_version, request_id, generation_id,
                origin_scope_id, origin, lifecycle, completeness_at_seal,
                selected_attempt_id, policy_version, aggregate_json,
                visibility_state, created_at, sealed_at
            ) VALUES (?, 'trace-2', 1, 'request-2', 'generation-2', ?,
                      'local', 'sealed', 'complete', 'attempt-2', 'policy-1',
                      '{}', 'active', ?, ?)
            """,
            (
                identity.profile_id,
                identity.profile_id,
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        cursor.execute(
            """
            INSERT INTO rag_evidence_runs VALUES (
                ?, 'trace-2', 'run-2', 1, 'retrieval', 'available',
                '{"payload_id":"run-payload-2"}', ?, ?, NULL
            )
            """,
            (identity.profile_id, NOW.isoformat(), NOW.isoformat()),
        )
        cursor.execute(
            """
            INSERT INTO rag_answer_attempt_payloads VALUES (
                ?, 'answer-payload-2', 'trace-2', 'attempt-2',
                'available', 'default', 'Other answer', 'integrity',
                ?, NULL, NULL
            )
            """,
            (identity.profile_id, NOW.isoformat()),
        )
        cursor.execute(
            """
            INSERT INTO rag_trace_evidence_refs VALUES (
                ?, 'trace-2', 'prompt-2', 1, 'run-2',
                'snapshot-2', 1, 'embedded'
            )
            """,
            (identity.profile_id,),
        )
        cursor.execute(
            """
            INSERT INTO rag_answer_attempt_payloads VALUES (
                ?, 'answer-payload-2-diagnostic', 'trace-2',
                'attempt-2-diagnostic', 'available', 'default',
                'Diagnostic answer', 'diagnostic-integrity', ?, NULL, NULL
            )
            """,
            (identity.profile_id, NOW.isoformat()),
        )

    CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
    ).revoke(
        local_trace_namespace(identity, trace_id="trace-1"),
        snapshot_payload_id="snapshot-1",
        tombstone=_tombstone(db),
    )

    assert {
        tuple(row)
        for row in connection.execute(
            """
            SELECT payload_id, redaction_state, snapshot_text
            FROM rag_evidence_snapshots
            ORDER BY payload_id
            """
        ).fetchall()
    } == {
        ("snapshot-1", "purged", None),
        ("snapshot-2", "purged", None),
    }
    assert {
        tuple(row)
        for row in connection.execute(
            """
            SELECT trace_id, redaction_state, run_payload_json
            FROM rag_evidence_runs
            """
        ).fetchall()
    } == {
        ("trace-1", "purged", None),
        ("trace-2", "purged", None),
    }
    assert {
        tuple(row)
        for row in connection.execute(
            """
            SELECT payload_id, trace_id, redaction_state, answer_body
            FROM rag_answer_attempt_payloads
            """
        ).fetchall()
    } == {
        ("answer-payload-1", "trace-1", "purged", None),
        ("answer-payload-2", "trace-2", "purged", None),
        ("answer-payload-2-diagnostic", "trace-2", "purged", None),
    }
    origin_rows = [
        (row["origin_namespace"], row["origin_payload_id"])
        for row in connection.execute(
            """
            SELECT origin_namespace, origin_payload_id
            FROM rag_evidence_snapshots
            ORDER BY payload_id
            """
        ).fetchall()
    ]
    assert origin_rows == [
        ("local_payload_v1", "snapshot-1"),
        ("local_payload_v1", "snapshot-1"),
    ]
    for origin_namespace, origin_payload_id in origin_rows:
        with pytest.raises(
            CitationPersistenceUnavailable,
            match="payload_origin_revoked",
        ):
            with db.transaction() as cursor:
                repository.assert_payload_origin_writable(
                    cursor,
                    profile_id=identity.profile_id,
                    origin_namespace=origin_namespace,
                    origin_payload_id=origin_payload_id,
                    seam="sync",
                )


@pytest.mark.parametrize(
    "changed_field",
    [
        "governance_scope_id",
        "authority_id",
        "confidentiality_policy_id",
        "revocation_scope_id",
    ],
)
def test_revoke_rejects_origin_key_policy_collisions_without_writes(
    db: CharactersRAGDB,
    changed_field: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    _insert_same_origin_snapshot(
        db,
        changed_field=changed_field,
        changed_value=f"collision-{changed_field}",
    )

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="revoke_origin_policy_collision",
    ):
        CitationPayloadLifecycle(
            repository,
            retention_policy=_policy(),
        ).revoke(
            local_trace_namespace(_identity(db), trace_id="trace-1"),
            snapshot_payload_id="snapshot-1",
            tombstone=_tombstone(db),
        )

    connection = db.get_connection()
    assert (
        connection.execute("SELECT count(*) FROM rag_payload_tombstones").fetchone()[0]
        == 0
    )
    assert {
        tuple(row)
        for row in connection.execute(
            """
            SELECT payload_id, redaction_state, snapshot_text, purged_at
            FROM rag_evidence_snapshots
            ORDER BY payload_id
            """
        ).fetchall()
    } == {
        (
            "snapshot-1",
            "available",
            "private exact submitted evidence",
            None,
        ),
        (
            "snapshot-2",
            "available",
            "private exact submitted evidence",
            None,
        ),
    }
    assert {
        tuple(row)
        for row in connection.execute(
            """
            SELECT redaction_state, run_payload_json IS NOT NULL
            FROM rag_evidence_runs
            """
        ).fetchall()
    } == {("available", 1)}
    assert {
        tuple(row)
        for row in connection.execute(
            """
            SELECT redaction_state, answer_body IS NOT NULL,
                   body_integrity_hmac IS NOT NULL
            FROM rag_answer_attempt_payloads
            """
        ).fetchall()
    } == {("available", 1, 1)}


def test_revoke_rejects_mismatched_trace_origin_scope_policy_and_database(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    lifecycle = CitationPayloadLifecycle(repository, retention_policy=_policy())
    namespace = local_trace_namespace(identity, trace_id="trace-1")
    mismatches = (
        (
            "trace",
            local_trace_namespace(identity, trace_id="missing-trace"),
            _tombstone(db),
        ),
        (
            "origin",
            namespace,
            _tombstone(db).model_copy(update={"origin_payload_id": "other"}),
        ),
        (
            "scope",
            namespace,
            _tombstone(db).model_copy(update={"revocation_scope_id": "other"}),
        ),
        (
            "policy",
            namespace,
            _tombstone(db).model_copy(update={"policy_version": "other"}),
        ),
    )
    for reason, target, tombstone in mismatches:
        with pytest.raises(CitationPersistenceUnavailable, match=reason):
            lifecycle.revoke(
                target,
                snapshot_payload_id="snapshot-1",
                tombstone=tombstone,
            )
    foreign_db = CharactersRAGDB(
        tmp_path / "foreign.sqlite",
        client_id="foreign-lifecycle-test",
    )
    try:
        with foreign_db.transaction() as cursor:
            with pytest.raises(RuntimeError, match="repository database"):
                lifecycle.revoke(
                    namespace,
                    snapshot_payload_id="snapshot-1",
                    tombstone=_tombstone(db),
                    cursor=cursor,
                )
    finally:
        foreign_db.close_connection()


def test_revoke_rolls_back_tombstone_and_every_purge_on_failure(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    lifecycle = CitationPayloadLifecycle(repository, retention_policy=_policy())

    def fail_after_snapshot() -> None:
        raise RuntimeError("forced lifecycle failure")

    monkeypatch.setattr(lifecycle, "_after_snapshot_purge", fail_after_snapshot)
    with pytest.raises(RuntimeError, match="forced lifecycle failure"):
        lifecycle.revoke(
            local_trace_namespace(_identity(db), trace_id="trace-1"),
            snapshot_payload_id="snapshot-1",
            tombstone=_tombstone(db),
        )

    connection = db.get_connection()
    assert (
        connection.execute("SELECT count(*) FROM rag_payload_tombstones").fetchone()[0]
        == 0
    )
    assert (
        connection.execute("SELECT run_payload_json FROM rag_evidence_runs").fetchone()[
            0
        ]
        is not None
    )
    assert (
        connection.execute(
            "SELECT snapshot_text FROM rag_evidence_snapshots"
        ).fetchone()[0]
        is not None
    )
    assert (
        connection.execute(
            "SELECT answer_body FROM rag_answer_attempt_payloads"
        ).fetchone()[0]
        is not None
    )


def test_revoke_is_idempotent_and_never_logs_governed_values(
    db: CharactersRAGDB,
    caplog: pytest.LogCaptureFixture,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    lifecycle = CitationPayloadLifecycle(repository, retention_policy=_policy())
    namespace = local_trace_namespace(_identity(db), trace_id="trace-1")

    with caplog.at_level(logging.DEBUG):
        first = lifecycle.revoke(
            namespace,
            snapshot_payload_id="snapshot-1",
            tombstone=_tombstone(db),
        )
        second = lifecycle.revoke(
            namespace,
            snapshot_payload_id="snapshot-1",
            tombstone=_tombstone(db),
        )

    assert first == second
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_payload_tombstones")
        .fetchone()[0]
        == 1
    )
    log_text = caplog.text
    for secret in (
        "private query",
        "private exact submitted evidence",
        "private source title",
        "private-document",
        "Answer [S1].",
        "content-hmac",
    ):
        assert secret not in log_text


def test_revoke_retry_can_extend_retention_without_rewriting_original_purge_time(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    lifecycle = CitationPayloadLifecycle(repository, retention_policy=_policy())
    namespace = local_trace_namespace(_identity(db), trace_id="trace-1")
    lifecycle.revoke(
        namespace,
        snapshot_payload_id="snapshot-1",
        tombstone=_tombstone(db, retain_until=NOW + timedelta(days=30)),
    )
    retry = _tombstone(
        db,
        retain_until=NOW + timedelta(days=60),
    ).model_copy(update={"revoked_at": NOW + timedelta(days=1)})

    stored = lifecycle.revoke(
        namespace,
        snapshot_payload_id="snapshot-1",
        tombstone=retry,
    )

    assert stored.revoked_at == NOW
    assert stored.retain_until == NOW + timedelta(days=60)
    connection = db.get_connection()
    assert (
        connection.execute("SELECT revoked_at FROM rag_payload_tombstones").fetchone()[
            0
        ]
        == NOW.isoformat()
    )
    assert (
        connection.execute("SELECT purged_at FROM rag_evidence_snapshots").fetchone()[0]
        == NOW.isoformat()
    )


def test_all_repository_payload_seams_reject_tombstoned_origins(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())
    _persist(db, repository)
    CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
    ).revoke(
        local_trace_namespace(_identity(db), trace_id="trace-1"),
        snapshot_payload_id="snapshot-1",
        tombstone=_tombstone(db),
    )
    conversation_id = db.add_conversation(
        {"title": "Cache replay", "character_id": None}
    )
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="payload_origin_revoked",
    ):
        with db.transaction() as cursor:
            repository.write_prepared(
                cursor,
                prepared,
                message_id="message-1",
                message_revision=1,
                message_body="Answer [S1].",
            )
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="payload_origin_revoked",
    ):
        with db.transaction() as cursor:
            db.add_message(
                {
                    "id": "cache-message",
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "Answer [S1].",
                    "client_id": db.client_id,
                }
            )
            repository.link_cache_message_owner(
                cursor,
                local_trace_namespace(_identity(db), trace_id="trace-1"),
                message_id="cache-message",
                message_revision=1,
                message_body="Answer [S1].",
            )
    for seam in ("import", "sync"):
        with pytest.raises(
            CitationPersistenceUnavailable,
            match="payload_origin_revoked",
        ):
            with db.transaction() as cursor:
                repository.assert_payload_origin_writable(
                    cursor,
                    profile_id=_identity(db).profile_id,
                    origin_namespace="local_payload_v1",
                    origin_payload_id="snapshot-1",
                    seam=seam,
                )


def test_collect_preserves_live_and_soft_deleted_owners_until_policy_expiry(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    lifecycle = CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(soft_delete_seconds=3600),
    )
    assert lifecycle.collect(now=NOW).traces_collected == 0

    _mark_owner_deleted(db, updated_at=NOW)
    assert lifecycle.collect(now=NOW + timedelta(minutes=59)).traces_collected == 0
    result = lifecycle.collect(now=NOW + timedelta(hours=1))
    assert result.traces_collected == 1
    assert result.snapshots_collected == 1
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_citation_traces")
        .fetchone()[0]
        == 0
    )


def test_collect_retains_expired_tombstone_until_every_shared_origin_row_is_gone(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    _insert_same_origin_snapshot(db)
    identity = _identity(db)
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_trace_evidence_refs VALUES (
                ?, 'trace-1', 'prompt-1', 2, 'run-1',
                'snapshot-2', 2, 'embedded'
            )
            """,
            (identity.profile_id,),
        )
    lifecycle = CitationPayloadLifecycle(repository, retention_policy=_policy())
    lifecycle.revoke(
        local_trace_namespace(identity, trace_id="trace-1"),
        snapshot_payload_id="snapshot-1",
        tombstone=_tombstone(db, retain_until=NOW),
    )

    blocked = lifecycle.collect(now=NOW, limit=1)

    assert blocked.traces_collected == 0
    assert blocked.tombstones_collected == 0
    connection = db.get_connection()
    assert (
        connection.execute("SELECT count(*) FROM rag_payload_tombstones").fetchone()[0]
        == 1
    )
    assert (
        connection.execute("SELECT count(*) FROM rag_evidence_snapshots").fetchone()[0]
        == 2
    )
    for seam in ("write", "import", "sync"):
        with pytest.raises(
            CitationPersistenceUnavailable,
            match="payload_origin_revoked",
        ):
            with db.transaction() as cursor:
                repository.assert_payload_origin_writable(
                    cursor,
                    profile_id=identity.profile_id,
                    origin_namespace="local_payload_v1",
                    origin_payload_id="snapshot-1",
                    seam=seam,
                )

    _mark_owner_deleted(db)
    collected = lifecycle.collect(
        now=NOW,
        limit=1,
        continuation_cursor=blocked.continuation_cursor,
    )

    assert collected.traces_collected == 1
    assert collected.snapshots_collected == 2
    assert collected.tombstones_collected == 1
    assert (
        connection.execute("SELECT count(*) FROM rag_payload_tombstones").fetchone()[0]
        == 0
    )
    with db.transaction() as cursor:
        repository.assert_payload_origin_writable(
            cursor,
            profile_id=identity.profile_id,
            origin_namespace="local_payload_v1",
            origin_payload_id="snapshot-1",
            seam="write",
        )


def test_collect_acquires_writer_lock_through_singleton_identity_row(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    connection = db.get_connection()
    updated_tables: list[tuple[str | None, str | None]] = []

    def authorize(
        action: int,
        table: str | None,
        column: str | None,
        database: str | None,
        trigger: str | None,
    ) -> int:
        del database, trigger
        if action == sqlite3.SQLITE_UPDATE:
            updated_tables.append((table, column))
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorize)
    try:
        result = CitationPayloadLifecycle(
            repository,
            retention_policy=_policy(),
        ).collect(now=NOW)
    finally:
        connection.set_authorizer(None)

    assert result.traces_collected == 0
    assert ("rag_identity_context", "profile_id") in updated_tables
    assert not any(table == "rag_citation_traces" for table, _ in updated_tables)


def test_collect_scans_past_oldest_barriered_traces_across_repeated_batches(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    with db.transaction() as cursor:
        for ordinal in (2, 3):
            cursor.execute(
                """
                INSERT INTO rag_citation_traces(
                    profile_id, trace_id, schema_version, request_id,
                    generation_id, origin_scope_id, origin, lifecycle,
                    completeness_at_seal, selected_attempt_id, policy_version,
                    aggregate_json, visibility_state, created_at, sealed_at
                ) VALUES (?, ?, 1, ?, ?, ?, 'local', 'sealed', 'complete',
                          'attempt-empty', 'policy-1', '{}', 'active', ?, ?)
                """,
                (
                    identity.profile_id,
                    f"trace-{ordinal}",
                    f"request-{ordinal}",
                    f"generation-{ordinal}",
                    identity.profile_id,
                    (NOW + timedelta(seconds=ordinal)).isoformat(),
                    (NOW + timedelta(seconds=ordinal)).isoformat(),
                ),
            )
    lifecycle = CitationPayloadLifecycle(repository, retention_policy=_policy())

    first = lifecycle.collect(now=NOW + timedelta(days=1), limit=1)
    assert first.traces_collected == 1
    assert first.traces_examined == 2
    assert {
        row["trace_id"]
        for row in db.get_connection()
        .execute("SELECT trace_id FROM rag_citation_traces ORDER BY trace_id")
        .fetchall()
    } == {"trace-1", "trace-3"}

    second = lifecycle.collect(
        now=NOW + timedelta(days=1),
        limit=1,
        continuation_cursor=first.continuation_cursor,
    )
    assert second.traces_collected == 1
    assert second.traces_examined == 1
    assert {
        row["trace_id"]
        for row in db.get_connection()
        .execute("SELECT trace_id FROM rag_citation_traces ORDER BY trace_id")
        .fetchall()
    } == {"trace-1"}


def test_collect_scans_past_oldest_barriered_tombstones_across_repeated_batches(
    db: CharactersRAGDB,
) -> None:
    identity = _identity(db)
    with db.transaction() as cursor:
        for ordinal in (1, 2, 3):
            cursor.execute(
                """
                INSERT INTO rag_payload_tombstones VALUES (
                    ?, 'local_payload_v1', ?, ?, 'source_revoked',
                    'policy-1', ?, ?
                )
                """,
                (
                    identity.profile_id,
                    f"expired-{ordinal}",
                    f"scope-{ordinal}",
                    (NOW - timedelta(days=ordinal + 1)).isoformat(),
                    (NOW - timedelta(days=4 - ordinal)).isoformat(),
                ),
            )
    lifecycle = _lifecycle(db)
    barriers = CitationCollectionBarriers(
        payload_origins=(("local_payload_v1", "expired-1"),)
    )

    first = lifecycle.collect(now=NOW, barriers=barriers, limit=1)
    assert first.tombstones_collected == 1
    assert {
        row["origin_payload_id"]
        for row in db.get_connection()
        .execute(
            """
            SELECT origin_payload_id
            FROM rag_payload_tombstones
            ORDER BY retain_until, origin_payload_id
            """
        )
        .fetchall()
    } == {"expired-1", "expired-3"}

    second = lifecycle.collect(
        now=NOW,
        barriers=barriers,
        limit=1,
        continuation_cursor=first.continuation_cursor,
    )
    assert second.tombstones_collected == 1
    assert {
        row["origin_payload_id"]
        for row in db.get_connection()
        .execute(
            """
            SELECT origin_payload_id
            FROM rag_payload_tombstones
            ORDER BY retain_until, origin_payload_id
            """
        )
        .fetchall()
    } == {"expired-1"}


def test_collect_cursor_progresses_beyond_scan_page_and_wraps_to_newly_eligible_rows(
    db: CharactersRAGDB,
) -> None:
    identity = _identity(db)
    with db.transaction() as cursor:
        for ordinal in range(1, 35):
            cursor.execute(
                """
                INSERT INTO rag_citation_traces(
                    profile_id, trace_id, schema_version, request_id,
                    generation_id, origin_scope_id, origin, lifecycle,
                    completeness_at_seal, selected_attempt_id, policy_version,
                    aggregate_json, visibility_state, created_at, sealed_at
                ) VALUES (?, ?, 1, ?, ?, ?, 'local', 'sealed', 'complete',
                          'attempt-empty', 'policy-1', '{}', 'active', ?, ?)
                """,
                (
                    identity.profile_id,
                    f"trace-{ordinal:02}",
                    f"request-{ordinal:02}",
                    f"generation-{ordinal:02}",
                    identity.profile_id,
                    (NOW + timedelta(seconds=ordinal)).isoformat(),
                    (NOW + timedelta(seconds=ordinal)).isoformat(),
                ),
            )
            cursor.execute(
                """
                INSERT INTO rag_payload_tombstones VALUES (
                    ?, 'local_payload_v1', ?, ?, 'source_revoked',
                    'policy-1', ?, ?
                )
                """,
                (
                    identity.profile_id,
                    f"expired-{ordinal:02}",
                    f"scope-{ordinal:02}",
                    (NOW - timedelta(days=2)).isoformat(),
                    (NOW - timedelta(days=1) + timedelta(seconds=ordinal)).isoformat(),
                ),
            )
    trace_barriers = tuple(f"trace-{ordinal:02}" for ordinal in range(1, 34))
    origin_barriers = tuple(
        ("local_payload_v1", f"expired-{ordinal:02}") for ordinal in range(1, 34)
    )
    lifecycle = _lifecycle(db)

    first = lifecycle.collect(
        now=NOW,
        barriers=CitationCollectionBarriers(
            trace_ids=trace_barriers,
            payload_origins=origin_barriers,
        ),
        limit=1,
    )
    assert first.traces_examined == 32
    assert first.traces_collected == 0
    assert first.tombstones_collected == 0
    assert isinstance(first.continuation_cursor, str)

    second = lifecycle.collect(
        now=NOW,
        barriers=CitationCollectionBarriers(
            trace_ids=trace_barriers,
            payload_origins=origin_barriers,
        ),
        limit=1,
        continuation_cursor=first.continuation_cursor,
    )
    assert second.traces_examined == 2
    assert second.traces_collected == 1
    assert second.tombstones_collected == 1
    assert isinstance(second.continuation_cursor, str)
    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT count(*) FROM rag_citation_traces WHERE trace_id = 'trace-34'"
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            """
        SELECT count(*) FROM rag_payload_tombstones
        WHERE origin_payload_id = 'expired-34'
        """
        ).fetchone()[0]
        == 0
    )
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_citation_traces(
                profile_id, trace_id, schema_version, request_id,
                generation_id, origin_scope_id, origin, lifecycle,
                completeness_at_seal, selected_attempt_id, policy_version,
                aggregate_json, visibility_state, created_at, sealed_at
            ) VALUES (?, 'trace-35', 1, 'request-35', 'generation-35', ?,
                      'local', 'sealed', 'complete', 'attempt-empty',
                      'policy-1', '{}', 'active', ?, ?)
            """,
            (
                identity.profile_id,
                identity.profile_id,
                (NOW + timedelta(seconds=35)).isoformat(),
                (NOW + timedelta(seconds=35)).isoformat(),
            ),
        )
        cursor.execute(
            """
            INSERT INTO rag_payload_tombstones VALUES (
                ?, 'local_payload_v1', 'expired-35', 'scope-35',
                'source_revoked', 'policy-1', ?, ?
            )
            """,
            (
                identity.profile_id,
                (NOW - timedelta(days=2)).isoformat(),
                (NOW - timedelta(days=1) + timedelta(seconds=35)).isoformat(),
            ),
        )

    wrapped = lifecycle.collect(
        now=NOW,
        barriers=CitationCollectionBarriers(
            trace_ids=trace_barriers[1:],
            payload_origins=origin_barriers[1:],
        ),
        limit=1,
        continuation_cursor=second.continuation_cursor,
    )
    assert wrapped.traces_examined == 1
    assert wrapped.traces_collected == 1
    assert wrapped.tombstones_collected == 1
    assert (
        connection.execute(
            "SELECT count(*) FROM rag_citation_traces WHERE trace_id = 'trace-01'"
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            """
        SELECT count(*) FROM rag_payload_tombstones
        WHERE origin_payload_id = 'expired-01'
        """
        ).fetchone()[0]
        == 0
    )


def test_collect_fails_closed_on_malformed_owner_retention_timestamp(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    with db.transaction() as cursor:
        cursor.execute(
            """
            UPDATE rag_message_trace_owners
            SET state = 'deleted', updated_at = 'not-a-timestamp'
            """
        )

    result = CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(soft_delete_seconds=3600),
    ).collect(now=NOW)

    assert result.traces_collected == 0
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_citation_traces")
        .fetchone()[0]
        == 1
    )


@pytest.mark.parametrize(
    "barrier",
    [
        "snapshot_retention",
        "answer_retention",
        "artifact_link_pending",
        "artifact_live",
        "artifact_unlink_pending",
        "artifact_pending_operation",
        "artifact_applied_operation",
        "sync_trace",
        "sync_origin",
    ],
)
def test_collect_respects_payload_artifact_operation_and_sync_barriers(
    db: CharactersRAGDB,
    barrier: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    _mark_owner_deleted(db)
    connection = db.get_connection()
    barriers = CitationCollectionBarriers()
    if barrier == "snapshot_retention":
        connection.execute(
            "UPDATE rag_evidence_snapshots SET retain_until = ?",
            ((NOW + timedelta(days=1)).isoformat(),),
        )
    elif barrier == "answer_retention":
        connection.execute(
            "UPDATE rag_answer_attempt_payloads SET retain_until = ?",
            ((NOW + timedelta(days=1)).isoformat(),),
        )
    elif barrier.startswith("artifact_"):
        lease_state = {
            "artifact_link_pending": "link_pending",
            "artifact_live": "live",
            "artifact_unlink_pending": "unlink_pending",
            "artifact_pending_operation": "released",
            "artifact_applied_operation": "released",
        }[barrier]
        connection.execute(
            """
            INSERT INTO rag_artifact_owner_leases VALUES (
                ?, 'chatbook', 'artifact-1', 1, 'trace-1',
                'lease-1', ?, ?, ?, NULL
            )
            """,
            (
                _identity(db).profile_id,
                lease_state,
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        if "operation" in barrier:
            operation_state = barrier.removeprefix("artifact_").removesuffix(
                "_operation"
            )
            connection.execute(
                """
                INSERT INTO rag_artifact_owner_operations VALUES (
                    ?, 'operation-1', 'chatbook', 'artifact-1', 1,
                    'trace-1', 'unlink', ?, ?, ?
                )
                """,
                (
                    _identity(db).profile_id,
                    operation_state,
                    NOW.isoformat(),
                    NOW.isoformat(),
                ),
            )
    elif barrier == "sync_trace":
        barriers = CitationCollectionBarriers(trace_ids=("trace-1",))
    else:
        barriers = CitationCollectionBarriers(
            payload_origins=(("local_payload_v1", "snapshot-1"),)
        )
    connection.commit()

    result = CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
    ).collect(now=NOW, barriers=barriers)

    assert result.traces_collected == 0
    assert (
        connection.execute("SELECT count(*) FROM rag_citation_traces").fetchone()[0]
        == 1
    )


def test_collect_deletes_released_acknowledged_artifact_receipts_before_trace(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    _mark_owner_deleted(db)
    connection = db.get_connection()
    connection.execute(
        """
        INSERT INTO rag_artifact_owner_leases VALUES (
            ?, 'chatbook', 'artifact-1', 1, 'trace-1',
            'lease-1', 'released', ?, ?, NULL
        )
        """,
        (_identity(db).profile_id, NOW.isoformat(), NOW.isoformat()),
    )
    connection.execute(
        """
        INSERT INTO rag_artifact_owner_operations VALUES (
            ?, 'operation-1', 'chatbook', 'artifact-1', 1,
            'trace-1', 'unlink', 'acknowledged', ?, ?
        )
        """,
        (_identity(db).profile_id, NOW.isoformat(), NOW.isoformat()),
    )
    connection.commit()

    result = CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
    ).collect(now=NOW)

    assert result.traces_collected == 1
    assert (
        connection.execute(
            "SELECT count(*) FROM rag_artifact_owner_operations"
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute("SELECT count(*) FROM rag_artifact_owner_leases").fetchone()[
            0
        ]
        == 0
    )


def test_collect_preserves_a_snapshot_referenced_by_a_surviving_trace(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    _mark_owner_deleted(db)
    connection = db.get_connection()
    trace = connection.execute(
        "SELECT * FROM rag_citation_traces WHERE trace_id = 'trace-1'"
    ).fetchone()
    aggregate = json.loads(trace["aggregate_json"])
    aggregate["trace_id"] = "trace-2"
    aggregate["request_id"] = "request-2"
    aggregate["generation_id"] = "generation-2"
    aggregate["evidence_runs"][0]["run_id"] = "run-2"
    aggregate["evidence_runs"][0]["request_id"] = "request-2"
    aggregate["evidence_runs"][0]["payload_ref"] = "run-payload-2"
    aggregate["prompt_evidence_sets"][0]["entries"][0]["run_id"] = "run-2"
    aggregate["answer_attempts"][0]["attempt_id"] = "attempt-2"
    aggregate["answer_attempts"][0]["answer_payload_ref"] = "answer-payload-2"
    aggregate["selected_attempt_id"] = "attempt-2"
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_citation_traces(
                profile_id, trace_id, schema_version, request_id, generation_id,
                origin_scope_id, origin, lifecycle, completeness_at_seal,
                selected_attempt_id, policy_version, aggregate_json,
                visibility_state, created_at, sealed_at
            ) VALUES (?, 'trace-2', 1, 'request-2', 'generation-2', ?,
                      'local', 'sealed', 'complete', 'attempt-2', 'policy-1',
                      ?, 'active', ?, ?)
            """,
            (
                _identity(db).profile_id,
                _identity(db).profile_id,
                json.dumps(aggregate, separators=(",", ":"), sort_keys=True),
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        cursor.execute(
            """
            INSERT INTO rag_evidence_runs VALUES (
                ?, 'trace-2', 'run-2', 1, 'retrieval', 'available',
                '{"schema_version":1,"payload_id":"run-payload-2","run_id":"run-2","raw_query":null,"query_fingerprint":null,"authority_id":null,"retrieval_metadata":{},"candidates":[]}',
                ?, ?, NULL
            )
            """,
            (_identity(db).profile_id, NOW.isoformat(), NOW.isoformat()),
        )
        cursor.execute(
            """
            INSERT INTO rag_answer_attempt_payloads VALUES (
                ?, 'answer-payload-2', 'trace-2', 'attempt-2',
                'available', 'default', 'Other [S1].', 'integrity',
                ?, NULL, NULL
            )
            """,
            (_identity(db).profile_id, NOW.isoformat()),
        )
        cursor.execute(
            """
            INSERT INTO rag_trace_evidence_refs VALUES (
                ?, 'trace-2', 'prompt-1', 1, 'run-2',
                'snapshot-1', 1, 'embedded'
            )
            """,
            (_identity(db).profile_id,),
        )
        cursor.execute(
            """
            INSERT INTO rag_message_trace_owners VALUES (
                ?, 'message-1', 1, 'trace-2', 'body_mismatch',
                'body-fingerprint-2', 'owner-key-2', ?, ?
            )
            """,
            (_identity(db).profile_id, NOW.isoformat(), NOW.isoformat()),
        )

    result = CitationPayloadLifecycle(
        repository,
        retention_policy=_policy(),
    ).collect(now=NOW)

    assert result.traces_collected == 1
    assert result.snapshots_collected == 0
    assert (
        connection.execute(
            "SELECT count(*) FROM rag_citation_traces WHERE trace_id = 'trace-2'"
        ).fetchone()[0]
        == 1
    )
    assert (
        connection.execute(
            "SELECT count(*) FROM rag_evidence_snapshots WHERE payload_id = 'snapshot-1'"
        ).fetchone()[0]
        == 1
    )


def test_collect_rechecks_barriers_before_deleting_and_rolls_back_failures(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    _mark_owner_deleted(db)
    lifecycle = CitationPayloadLifecycle(repository, retention_policy=_policy())

    def restore_owner(cursor, profile_id: str, trace_id: str) -> None:
        cursor.execute(
            """
            UPDATE rag_message_trace_owners
            SET state = 'active', updated_at = ?
            WHERE profile_id = ? AND trace_id = ?
            """,
            (NOW.isoformat(), profile_id, trace_id),
        )

    monkeypatch.setattr(lifecycle, "_before_collect_delete", restore_owner)
    assert lifecycle.collect(now=NOW).traces_collected == 0
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_citation_traces")
        .fetchone()[0]
        == 1
    )

    _mark_owner_deleted(db)

    def fail_after_refs() -> None:
        raise RuntimeError("forced GC failure")

    monkeypatch.setattr(lifecycle, "_before_collect_delete", lambda *_: None)
    monkeypatch.setattr(lifecycle, "_after_collect_refs", fail_after_refs)
    with pytest.raises(RuntimeError, match="forced GC failure"):
        lifecycle.collect(now=NOW)
    connection = db.get_connection()
    assert (
        connection.execute("SELECT count(*) FROM rag_citation_traces").fetchone()[0]
        == 1
    )
    assert (
        connection.execute("SELECT count(*) FROM rag_trace_evidence_refs").fetchone()[0]
        == 1
    )


def test_collect_is_bounded_idempotent_and_retains_tombstones_until_expiry(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    lifecycle = CitationPayloadLifecycle(repository, retention_policy=_policy())
    lifecycle.revoke(
        local_trace_namespace(_identity(db), trace_id="trace-1"),
        snapshot_payload_id="snapshot-1",
        tombstone=_tombstone(db, retain_until=NOW + timedelta(hours=1)),
    )
    _mark_owner_deleted(db)

    first = lifecycle.collect(now=NOW, limit=1)
    second = lifecycle.collect(now=NOW, limit=1)
    assert first.traces_examined <= 1
    assert first.traces_collected == 1
    assert second.traces_collected == 0
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_payload_tombstones")
        .fetchone()[0]
        == 1
    )

    expired = lifecycle.collect(now=NOW + timedelta(hours=1), limit=1)
    assert expired.tombstones_collected == 1
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_payload_tombstones")
        .fetchone()[0]
        == 0
    )
