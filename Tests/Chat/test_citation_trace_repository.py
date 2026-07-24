from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import gc
import json
import sqlite3
import weakref

import pytest
from pydantic import ValidationError

from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_source_locators import (
    AuthorityScope,
    CitationReadAuthorization,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    CitationFingerprintCodec,
    CitationFingerprintKeyUnavailable,
    LocalCitationIdentityContext,
    local_trace_namespace,
)
from tldw_chatbook.Chat.citation_trace_models import (
    AnswerAttempt,
    AnswerAttemptKind,
    AnswerAttemptPayload,
    CitationCompleteness,
    CitationOccurrence,
    CitationTrace,
    EvidenceRun,
    EvidenceRunPayload,
    EvidenceSnapshotPayload,
    EvidenceStorageMode,
    GOVERNED_PAYLOAD_UTF8_BYTES_MAX,
    MarkerNamespace,
    PolicyCapability,
    PromptEvidenceEntry,
    PromptEvidenceSet,
    SealedCitationWrite,
    StructuralValidationState,
    TraceLifecycle,
    TraceOrigin,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationHydrationState,
    CitationPersistenceUnavailable,
    CitationTraceRepository,
    load_local_citation_identity_context,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


NOW = datetime(2026, 7, 24, 7, 0, tzinfo=UTC)
ROW_FAMILIES = ("trace", "runs", "snapshots", "attempts", "refs", "owner")


@pytest.fixture
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(
        tmp_path / "citation-repository.sqlite",
        client_id="citation-repository-test",
    )
    yield database
    database.close_connection()


def _identity(db: CharactersRAGDB) -> LocalCitationIdentityContext:
    context = load_local_citation_identity_context(db)
    assert context is not None
    return context


def _sealed_write(*, authority_id: str | None = None) -> SealedCitationWrite:
    answer = "Answer [S1]."
    run = EvidenceRun(
        run_id="run-1",
        request_id="request-1",
        run_ordinal=1,
        stage="retrieval",
        payload_ref="run-payload-1",
        started_at=NOW,
        ended_at=NOW,
    )
    prompt = PromptEvidenceSet(
        prompt_set_id="prompt-1",
        prompt_set_ordinal=1,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        entries=(
            PromptEvidenceEntry(
                evidence_ordinal=1,
                marker_ordinal=1,
                run_id=run.run_id,
                snapshot_payload_ref="snapshot-1",
                storage_mode=EvidenceStorageMode.EMBEDDED,
            ),
        ),
        created_at=NOW,
    )
    attempt = AnswerAttempt(
        attempt_id="attempt-1",
        attempt_ordinal=1,
        kind=AnswerAttemptKind.INITIAL,
        prompt_evidence_set_id=prompt.prompt_set_id,
        answer_payload_ref="answer-payload-1",
        occurrences=(
            CitationOccurrence(
                occurrence_id="occurrence-1",
                occurrence_ordinal=1,
                raw_marker="[S1]",
                marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
                evidence_ordinal=1,
                marker_start=7,
                marker_end=11,
                claim_start=0,
                claim_end=6,
                structural_state=StructuralValidationState.VALID,
            ),
        ),
        created_at=NOW,
    )
    trace = CitationTrace(
        trace_id="trace-1",
        request_id="request-1",
        generation_id="generation-1",
        origin=TraceOrigin.LOCAL,
        lifecycle=TraceLifecycle.SEALED,
        completeness_at_seal=CitationCompleteness.COMPLETE,
        evidence_runs=(run,),
        prompt_evidence_sets=(prompt,),
        answer_attempts=(attempt,),
        selected_attempt_id=attempt.attempt_id,
        policy_capabilities=(PolicyCapability.VIEW_SNAPSHOT,),
        policy_version="policy-1",
        created_at=NOW,
        sealed_at=NOW,
    )
    return SealedCitationWrite(
        trace=trace,
        evidence_run_payloads=(
            EvidenceRunPayload(
                payload_id="run-payload-1",
                run_id=run.run_id,
                raw_query="private query",
                query_fingerprint="query-hmac",
                authority_id=authority_id,
                retrieval_metadata={"retriever": "synthetic"},
            ),
        ),
        evidence_snapshot_payloads=(
            EvidenceSnapshotPayload(
                payload_id="snapshot-1",
                storage_mode=EvidenceStorageMode.EMBEDDED,
                snapshot_text="private exact submitted evidence",
                title="private source title",
                source_identity={"document_id": "private-document"},
                locator={"source_kind": "media_db", "item_id": "private-item"},
                lineage={"chunk_id": "private-chunk"},
                transformations=("truncate",),
                content_hash="content-hmac",
                comparison_hash="comparison-hmac",
            ),
        ),
        answer_attempt_payloads=(
            AnswerAttemptPayload(
                payload_id="answer-payload-1",
                attempt_id=attempt.attempt_id,
                answer_body=answer,
                body_integrity_hmac="answer-hmac",
            ),
        ),
    )


def _repository(
    db: CharactersRAGDB,
    *,
    enabled: bool = True,
    codec: CitationFingerprintCodec | None = None,
    identity: LocalCitationIdentityContext | None = None,
    failure_after: str | None = None,
) -> CitationTraceRepository:
    return CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=enabled),
        identity_context=identity if identity is not None else _identity(db),
        fingerprint_codec=(
            codec if codec is not None else CitationFingerprintCodec(b"k" * 32)
        ),
        failure_after_row_family=failure_after,
    )


def _exact_governed_payload_write() -> SealedCitationWrite:
    base = _sealed_write()
    snapshot = base.evidence_snapshot_payloads[0]
    remaining = GOVERNED_PAYLOAD_UTF8_BYTES_MAX - base.governed_payload_bytes
    assert snapshot.title is not None
    exact_snapshot = snapshot.model_copy(
        update={"title": snapshot.title + ("x" * remaining)}
    )
    exact = SealedCitationWrite(
        trace=base.trace,
        evidence_run_payloads=base.evidence_run_payloads,
        evidence_snapshot_payloads=(exact_snapshot,),
        answer_attempt_payloads=base.answer_attempt_payloads,
    )
    assert exact.governed_payload_bytes == GOVERNED_PAYLOAD_UTF8_BYTES_MAX
    return exact


def _conversation(db: CharactersRAGDB) -> str:
    return db.add_conversation({"title": "Citation test", "character_id": None})


def _persist(
    db: CharactersRAGDB,
    repository: CitationTraceRepository,
    *,
    message_id: str = "message-1",
    sealed_write: SealedCitationWrite | None = None,
) -> None:
    prepared = repository.prepare_write(sealed_write or _sealed_write())
    with db.transaction() as cursor:
        db.add_message(
            {
                "id": message_id,
                "conversation_id": _conversation(db),
                "sender": "assistant",
                "content": "Answer [S1].",
                "client_id": db.client_id,
            }
        )
        repository.write_prepared(
            cursor,
            prepared,
            message_id=message_id,
            message_revision=1,
            message_body="Answer [S1].",
        )


def _authorization(
    identity: LocalCitationIdentityContext,
    *,
    profile_id: str | None = None,
    authority_id: str | None = None,
    view_snapshot: bool = True,
) -> CitationReadAuthorization:
    profile = profile_id or identity.profile_id
    return CitationReadAuthorization(
        authority_scope=AuthorityScope.LOCAL_PROFILE,
        profile_id=profile,
        governance_scope_id=profile,
        allowlisted_authority_ids=(authority_id or identity.local_authority_id,),
        view_snapshot=view_snapshot,
    )


def test_runtime_policy_is_frozen_and_disabled_write_fails_before_validation(
    db: CharactersRAGDB,
) -> None:
    policy = CitationProvenanceRuntimePolicy(canonical_writes_enabled=False)
    with pytest.raises(ValidationError, match="frozen"):
        policy.canonical_writes_enabled = True  # type: ignore[misc]

    repository = _repository(db, enabled=False)
    hostile = SealedCitationWrite.model_construct(trace="not-a-trace")
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="canonical_citation_writes_disabled",
    ):
        repository.prepare_write(hostile)


@pytest.mark.parametrize(
    ("identity_present", "codec_present", "reason"),
    [
        (False, True, "citation_identity_context_unavailable"),
        (True, False, "fingerprint_key_unavailable"),
    ],
)
def test_write_preflight_fails_closed_without_identity_or_key(
    db: CharactersRAGDB,
    identity_present: bool,
    codec_present: bool,
    reason: str,
) -> None:
    repository = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
        identity_context=_identity(db) if identity_present else None,
        fingerprint_codec=CitationFingerprintCodec(b"k" * 32)
        if codec_present
        else None,
    )

    with pytest.raises(CitationPersistenceUnavailable, match=reason):
        repository.prepare_write(_sealed_write())


class _TrackingKeyProvider:
    def __init__(self, secret: bytes | None) -> None:
        self.secret = secret
        self.calls: list[str] = []

    def load_key(self, fingerprint_key_id: str) -> bytes:
        self.calls.append(fingerprint_key_id)
        if self.secret is None:
            raise CitationFingerprintKeyUnavailable("missing")
        return self.secret


def test_repository_composition_never_loads_a_key_while_writes_are_disabled(
    db: CharactersRAGDB,
) -> None:
    provider = _TrackingKeyProvider(b"k" * 32)

    repository = CitationTraceRepository.from_key_provider(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=False),
        identity_context=_identity(db),
        key_provider=provider,
    )

    assert provider.calls == []
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="canonical_citation_writes_disabled",
    ):
        repository.prepare_write(_sealed_write())
    assert provider.calls == []


def test_missing_existing_key_is_not_silently_provisioned_or_replaced(
    db: CharactersRAGDB,
) -> None:
    identity = _identity(db)
    provider = _TrackingKeyProvider(None)

    repository = CitationTraceRepository.from_key_provider(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
        identity_context=identity,
        key_provider=provider,
    )

    assert provider.calls == [identity.fingerprint_key_id]
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="fingerprint_key_unavailable",
    ):
        repository.prepare_write(_sealed_write())
    assert provider.calls == [identity.fingerprint_key_id]


def test_preflight_revalidates_a_hostile_model_copy(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    hostile = _sealed_write().model_copy(update={"evidence_snapshot_payloads": ()})

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="invalid_sealed_citation_write",
    ):
        repository.prepare_write(hostile)


def test_preflight_rejects_a_run_from_another_authority(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="run_authority_mismatch",
    ):
        repository.prepare_write(_sealed_write(authority_id="hostile-authority"))


def test_preflight_accepts_null_or_matching_run_authority(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)

    repository.prepare_write(_sealed_write())
    repository.prepare_write(
        _sealed_write(authority_id=_identity(db).local_authority_id)
    )


def test_prepared_write_is_an_immutable_snapshot_of_nested_governed_values(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    sealed_write = _sealed_write()
    prepared = repository.prepare_write(sealed_write)
    mutable_graph = getattr(prepared, "sealed_write", None)
    if mutable_graph is not None:
        mutable_graph.evidence_snapshot_payloads[0].source_identity["document_id"] = (
            "tampered-after-prepare"
        )
        mutable_graph.evidence_run_payloads[0].retrieval_metadata["retriever"] = (
            "tampered-after-prepare"
        )

    with db.transaction() as cursor:
        conversation_id = _conversation(db)
        db.add_message(
            {
                "id": "immutable-prepared",
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "Answer [S1].",
                "client_id": db.client_id,
            }
        )
        repository.write_prepared(
            cursor,
            prepared,
            message_id="immutable-prepared",
            message_revision=1,
            message_body="Answer [S1].",
        )

    connection = db.get_connection()
    source_identity = json.loads(
        connection.execute(
            "SELECT source_identity_json FROM rag_evidence_snapshots"
        ).fetchone()[0]
    )
    run_payload = json.loads(
        connection.execute("SELECT run_payload_json FROM rag_evidence_runs").fetchone()[
            0
        ]
    )
    assert not hasattr(prepared, "sealed_write")
    assert source_identity == {"document_id": "private-document"}
    assert run_payload["retrieval_metadata"] == {"retriever": "synthetic"}


def test_prepare_enforces_exact_total_governed_payload_boundary(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    exact = _exact_governed_payload_write()
    repository.prepare_write(exact)

    exact_snapshot = exact.evidence_snapshot_payloads[0]
    assert exact_snapshot.title is not None
    oversized = exact.model_copy(
        update={
            "evidence_snapshot_payloads": (
                exact_snapshot.model_copy(update={"title": exact_snapshot.title + "x"}),
            )
        }
    )
    with pytest.raises(
        CitationPersistenceUnavailable,
        match="invalid_sealed_citation_write",
    ):
        repository.prepare_write(oversized)


def test_prepared_write_cannot_be_forged_or_reused_across_repositories(
    db: CharactersRAGDB,
) -> None:
    first = _repository(db)
    second = _repository(db)
    prepared = first.prepare_write(_sealed_write())
    conversation_id = _conversation(db)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="prepared_citation_write_not_owned",
    ):
        with db.transaction() as cursor:
            db.add_message(
                {
                    "id": "forged-prepared",
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "Answer [S1].",
                    "client_id": db.client_id,
                }
            )
            second.write_prepared(
                cursor,
                replace(prepared, repository_token=object()),
                message_id="forged-prepared",
                message_revision=1,
                message_body="Answer [S1].",
            )

    assert db.get_message_by_id("forged-prepared") is None


@pytest.mark.parametrize(
    ("row_family", "cell_index", "replacement"),
    [
        ("trace_row", -1, "2027-01-01T00:00:00+00:00"),
        ("run_rows", 5, '{"tampered":true}'),
        ("snapshot_rows", 10, "tampered-title"),
        ("answer_rows", 5, "tampered answer"),
        ("reference_rows", 7, "redacted"),
    ],
)
def test_same_token_replaced_prepared_rows_are_rejected_before_any_insert(
    db: CharactersRAGDB,
    row_family: str,
    cell_index: int,
    replacement: str,
) -> None:
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())
    original_rows = getattr(prepared, row_family)
    if row_family == "trace_row":
        forged_rows = list(original_rows)
        forged_rows[cell_index] = replacement
        forged = replace(prepared, **{row_family: tuple(forged_rows)})
    else:
        forged_row = list(original_rows[0])
        forged_row[cell_index] = replacement
        forged = replace(
            prepared,
            **{row_family: (tuple(forged_row), *original_rows[1:])},
        )
    assert forged.repository_token is prepared.repository_token
    conversation_id = _conversation(db)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="prepared_citation_write_not_owned",
    ):
        with db.transaction() as cursor:
            db.add_message(
                {
                    "id": f"forged-{row_family}",
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "Answer [S1].",
                    "client_id": db.client_id,
                }
            )
            repository.write_prepared(
                cursor,
                forged,
                message_id=f"forged-{row_family}",
                message_revision=1,
                message_body="Answer [S1].",
            )

    connection = db.get_connection()
    assert db.get_message_by_id(f"forged-{row_family}") is None
    for table in (
        "rag_citation_traces",
        "rag_evidence_runs",
        "rag_evidence_snapshots",
        "rag_answer_attempt_payloads",
        "rag_trace_evidence_refs",
        "rag_message_trace_owners",
    ):
        assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_exact_prepared_object_can_retry_after_outer_transaction_rollback(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())
    conversation_id = _conversation(db)

    with pytest.raises(RuntimeError, match="force caller rollback"):
        with db.transaction() as cursor:
            db.add_message(
                {
                    "id": "prepared-retry",
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "Answer [S1].",
                    "client_id": db.client_id,
                }
            )
            repository.write_prepared(
                cursor,
                prepared,
                message_id="prepared-retry",
                message_revision=1,
                message_body="Answer [S1].",
            )
            raise RuntimeError("force caller rollback")

    with db.transaction() as cursor:
        db.add_message(
            {
                "id": "prepared-retry",
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "Answer [S1].",
                "client_id": db.client_id,
            }
        )
        repository.write_prepared(
            cursor,
            prepared,
            message_id="prepared-retry",
            message_revision=1,
            message_body="Answer [S1].",
        )

    assert db.get_message_by_id("prepared-retry") is not None
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_citation_traces")
        .fetchone()[0]
        == 1
    )


def test_prepared_digest_rejects_exact_object_row_tampering_before_insert(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())
    snapshot_row = list(prepared.snapshot_rows[0])
    snapshot_row[10] = "tampered-title"
    object.__setattr__(
        prepared,
        "snapshot_rows",
        (tuple(snapshot_row), *prepared.snapshot_rows[1:]),
    )
    conversation_id = _conversation(db)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="prepared_citation_write_not_owned",
    ):
        with db.transaction() as cursor:
            db.add_message(
                {
                    "id": "digest-tamper",
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "Answer [S1].",
                    "client_id": db.client_id,
                }
            )
            repository.write_prepared(
                cursor,
                prepared,
                message_id="digest-tamper",
                message_revision=1,
                message_body="Answer [S1].",
            )

    assert db.get_message_by_id("digest-tamper") is None
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_citation_traces")
        .fetchone()[0]
        == 0
    )


def test_prepared_registration_is_removed_when_the_exact_object_is_collected(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())
    prepared_id = id(prepared)
    prepared_ref = weakref.ref(prepared)
    assert prepared_id in repository._issued_prepared_writes

    del prepared
    gc.collect()

    assert prepared_ref() is None
    assert prepared_id not in repository._issued_prepared_writes


def test_prepared_write_rejects_an_active_cursor_from_another_database(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())

    with sqlite3.connect(":memory:") as foreign_connection:
        foreign_connection.execute("BEGIN")
        with pytest.raises(
            RuntimeError,
            match="repository database transaction",
        ):
            repository.write_prepared(
                foreign_connection.cursor(),
                prepared,
                message_id="foreign-cursor",
                message_revision=1,
                message_body="Answer [S1].",
            )


def test_fingerprint_row_guard_detects_persisted_owners_and_hashes(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    assert repository.fingerprint_bearing_rows_exist() is False

    _persist(db, repository)

    assert repository.fingerprint_bearing_rows_exist() is True


def test_complete_write_creates_all_aggregate_row_families_and_safe_json(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    connection = db.get_connection()

    expected_counts = {
        "rag_citation_traces": 1,
        "rag_evidence_runs": 1,
        "rag_evidence_snapshots": 1,
        "rag_answer_attempt_payloads": 1,
        "rag_trace_evidence_refs": 1,
        "rag_message_trace_owners": 1,
    }
    for table, expected in expected_counts.items():
        assert (
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            == expected
        )

    aggregate = connection.execute(
        "SELECT aggregate_json FROM rag_citation_traces"
    ).fetchone()[0]
    assert json.loads(aggregate)["trace_id"] == "trace-1"
    for governed in (
        "private query",
        "private exact submitted evidence",
        "private source title",
        "private-document",
        "private-item",
        "private-chunk",
        "content-hmac",
        "comparison-hmac",
        "Answer [S1].",
    ):
        assert governed not in aggregate


@pytest.mark.parametrize("row_family", ROW_FAMILIES)
def test_failure_after_each_row_family_rolls_back_every_aggregate_row(
    db: CharactersRAGDB,
    row_family: str,
) -> None:
    repository = _repository(db, failure_after=row_family)
    prepared = repository.prepare_write(_sealed_write())
    conversation_id = _conversation(db)

    with pytest.raises(RuntimeError, match=f"forced_failure_after_{row_family}"):
        with db.transaction() as cursor:
            db.add_message(
                {
                    "id": "rollback-message",
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "Answer [S1].",
                    "client_id": db.client_id,
                }
            )
            repository.write_prepared(
                cursor,
                prepared,
                message_id="rollback-message",
                message_revision=1,
                message_body="Answer [S1].",
            )

    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT count(*) FROM messages WHERE id = 'rollback-message'"
        ).fetchone()[0]
        == 0
    )
    for table in (
        "rag_citation_traces",
        "rag_evidence_runs",
        "rag_evidence_snapshots",
        "rag_answer_attempt_payloads",
        "rag_trace_evidence_refs",
        "rag_message_trace_owners",
    ):
        assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_summary_read_needs_no_identity_or_fingerprint_key(
    db: CharactersRAGDB,
) -> None:
    writer = _repository(db)
    _persist(db, writer)
    identity = _identity(db)
    reader = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=False),
        identity_context=None,
        fingerprint_codec=None,
    )

    summary = reader.get_trace_summary(
        local_trace_namespace(identity, trace_id="trace-1")
    )

    assert summary is not None
    assert summary.trace.trace_id == "trace-1"
    assert summary.trace.lifecycle is TraceLifecycle.SEALED


@pytest.mark.parametrize(
    ("authorization", "state"),
    [
        ("cross_profile", CitationHydrationState.PROFILE_DENIED),
        ("wrong_authority", CitationHydrationState.AUTHORITY_DENIED),
        ("no_snapshot", CitationHydrationState.SNAPSHOT_CAPABILITY_DENIED),
    ],
)
def test_hydration_denials_return_only_safe_summary(
    db: CharactersRAGDB,
    authorization: str,
    state: CitationHydrationState,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    namespace = local_trace_namespace(identity, trace_id="trace-1")
    auth = {
        "cross_profile": _authorization(identity, profile_id="another-profile"),
        "wrong_authority": _authorization(identity, authority_id="another-authority"),
        "no_snapshot": _authorization(identity, view_snapshot=False),
    }[authorization]

    result = repository.hydrate_trace(namespace, authorization=auth)

    assert result.state is state
    assert result.summary is not None
    assert result.governed_payloads is None
    assert "private" not in repr(result)


def test_hydration_requires_the_sealed_snapshot_capability(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    write = _sealed_write()
    trace_without_capability = write.trace.model_copy(
        update={"policy_capabilities": ()}
    )
    _persist(
        db,
        repository,
        sealed_write=SealedCitationWrite(
            trace=trace_without_capability,
            evidence_run_payloads=write.evidence_run_payloads,
            evidence_snapshot_payloads=write.evidence_snapshot_payloads,
            answer_attempt_payloads=write.answer_attempt_payloads,
        ),
    )
    identity = _identity(db)

    result = repository.hydrate_trace(
        local_trace_namespace(identity, trace_id="trace-1"),
        authorization=_authorization(identity, view_snapshot=True),
    )

    assert result.state is CitationHydrationState.SNAPSHOT_CAPABILITY_DENIED
    assert result.summary is not None
    assert result.governed_payloads is None
    assert "private" not in repr(result)


@pytest.mark.parametrize(
    "denial",
    ["snapshot_redacted", "run_purged", "answer_purged", "tombstoned"],
)
def test_hydration_checks_redaction_and_tombstone_before_governed_select(
    db: CharactersRAGDB,
    denial: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    connection = db.get_connection()
    if denial == "snapshot_redacted":
        connection.execute(
            "UPDATE rag_evidence_snapshots SET redaction_state = 'redacted'"
        )
    elif denial == "run_purged":
        connection.execute(
            """
            UPDATE rag_evidence_runs
            SET redaction_state = 'purged',
                run_payload_json = NULL,
                purged_at = '2026-07-24T01:00:00Z'
            """
        )
    elif denial == "answer_purged":
        connection.execute(
            """
            UPDATE rag_answer_attempt_payloads
            SET redaction_state = 'purged',
                answer_body = NULL,
                body_integrity_hmac = NULL,
                purged_at = '2026-07-24T01:00:00Z'
            """
        )
    else:
        connection.execute(
            """
            INSERT INTO rag_payload_tombstones VALUES (
                ?, 'local_payload_v1', 'snapshot-1', 'snapshot-1',
                'revoked', 'policy-1',
                '2026-07-24T00:00:00Z', '2027-07-24T00:00:00Z'
            )
            """,
            (identity.profile_id,),
        )
    connection.commit()

    result = repository.hydrate_trace(
        local_trace_namespace(identity, trace_id="trace-1"),
        authorization=_authorization(identity),
    )

    expected = (
        CitationHydrationState.REDACTED
        if denial != "tombstoned"
        else CitationHydrationState.REVOKED
    )
    assert result.state is expected
    assert result.governed_payloads is None


def test_authorized_hydration_returns_revalidated_governed_payloads(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)

    result = repository.hydrate_trace(
        local_trace_namespace(identity, trace_id="trace-1"),
        authorization=_authorization(identity),
    )

    assert result.state is CitationHydrationState.AUTHORIZED
    assert result.governed_payloads is not None
    assert (
        result.governed_payloads.evidence_run_payloads[0].raw_query == "private query"
    )
    assert result.governed_payloads.evidence_run_payloads[0].authority_id is None
    assert (
        result.governed_payloads.evidence_snapshot_payloads[0].snapshot_text
        == "private exact submitted evidence"
    )
    assert (
        result.governed_payloads.answer_attempt_payloads[0].answer_body
        == "Answer [S1]."
    )


def test_hydration_returns_bounded_unavailable_state_for_incomplete_rows(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    connection = db.get_connection()
    connection.execute("DELETE FROM rag_answer_attempt_payloads")
    connection.commit()

    result = repository.hydrate_trace(
        local_trace_namespace(identity, trace_id="trace-1"),
        authorization=_authorization(identity),
    )

    assert result.state is CitationHydrationState.PAYLOAD_UNAVAILABLE
    assert result.summary is not None
    assert result.governed_payloads is None


@pytest.mark.parametrize(
    ("hostile_authority", "expected_state"),
    [
        ("hostile-authority", CitationHydrationState.AUTHORITY_DENIED),
        (0, CitationHydrationState.PAYLOAD_UNAVAILABLE),
    ],
)
def test_hydration_denies_a_hostile_persisted_run_authority(
    db: CharactersRAGDB,
    hostile_authority: str | int,
    expected_state: CitationHydrationState,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    connection = db.get_connection()
    payload = json.loads(
        connection.execute("SELECT run_payload_json FROM rag_evidence_runs").fetchone()[
            0
        ]
    )
    payload["authority_id"] = hostile_authority
    connection.execute(
        "UPDATE rag_evidence_runs SET run_payload_json = ?",
        (json.dumps(payload),),
    )
    connection.commit()

    authorization = _authorization(identity)
    if isinstance(hostile_authority, str):
        authorization = authorization.model_copy(
            update={
                "allowlisted_authority_ids": (
                    identity.local_authority_id,
                    hostile_authority,
                )
            }
        )
    result = repository.hydrate_trace(
        local_trace_namespace(identity, trace_id="trace-1"),
        authorization=authorization,
    )

    assert result.state is expected_state
    assert result.summary is not None
    assert result.governed_payloads is None
    assert "private query" not in repr(result)


def test_injected_identity_must_match_the_persisted_singleton(
    db: CharactersRAGDB,
) -> None:
    mismatched = _identity(db).model_copy(update={"profile_id": "another-profile"})
    repository = _repository(db, identity=mismatched)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="citation_identity_context_mismatch",
    ):
        repository.prepare_write(_sealed_write())
