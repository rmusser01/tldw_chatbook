from __future__ import annotations

import copy
from dataclasses import replace
from datetime import UTC, datetime
import gc
import hmac
import json
import sqlite3
import traceback
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
    CitationFingerprintDomain,
    CitationFingerprintKeyUnavailable,
    LocalCitationIdentityContext,
    local_trace_namespace,
)
from tldw_chatbook.Chat.citation_trace_builder import CitationTraceBuilder
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
    ActiveCitationTraceResult,
    ActiveCitationTraceState,
    CitationHydrationState,
    CitationPersistenceUnavailable,
    CitationTraceRepository,
    load_local_citation_identity_context,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


NOW = datetime(2026, 7, 24, 7, 0, tzinfo=UTC)
ROW_FAMILIES = ("trace", "runs", "snapshots", "attempts", "refs", "owner")
TEST_FINGERPRINT_CODEC = CitationFingerprintCodec(b"k" * 32)


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
        policy_capabilities=(
            PolicyCapability.VIEW_SNAPSHOT,
            PolicyCapability.VIEW_SOURCE_IDENTITY,
        ),
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
                body_integrity_hmac=TEST_FINGERPRINT_CODEC.fingerprint(
                    CitationFingerprintDomain.MESSAGE_BODY,
                    answer,
                ),
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
        fingerprint_codec=(codec if codec is not None else TEST_FINGERPRINT_CODEC),
        failure_after_row_family=failure_after,
    )


def _assert_no_citation_rows(db: CharactersRAGDB) -> None:
    connection = db.get_connection()
    for table in (
        "rag_citation_traces",
        "rag_evidence_runs",
        "rag_evidence_snapshots",
        "rag_answer_attempt_payloads",
        "rag_trace_evidence_refs",
        "rag_message_trace_owners",
    ):
        assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


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
    view_source_identity: bool = True,
) -> CitationReadAuthorization:
    profile = profile_id or identity.profile_id
    return CitationReadAuthorization(
        authority_scope=AuthorityScope.LOCAL_PROFILE,
        profile_id=profile,
        governance_scope_id=profile,
        allowlisted_authority_ids=(authority_id or identity.local_authority_id,),
        view_snapshot=view_snapshot,
        view_source_identity=view_source_identity,
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


def test_local_citation_writes_ready_requires_matching_persisted_identity(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)

    assert repository.local_citation_writes_ready is True
    with pytest.raises(AttributeError):
        repository.local_citation_writes_ready = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "unready_reason",
    (
        "disabled",
        "missing_identity",
        "missing_codec",
        "missing_persisted_identity",
        "mismatched_identity",
    ),
)
def test_local_citation_writes_ready_fails_closed_for_unready_composition(
    db: CharactersRAGDB,
    unready_reason: str,
) -> None:
    identity = _identity(db)
    enabled = unready_reason != "disabled"
    injected_identity = (
        None
        if unready_reason == "missing_identity"
        else identity.model_copy(
            update={"local_authority_id": "mismatched-local-authority"}
        )
        if unready_reason == "mismatched_identity"
        else identity
    )
    codec = None if unready_reason == "missing_codec" else TEST_FINGERPRINT_CODEC
    repository = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(
            canonical_writes_enabled=enabled,
        ),
        identity_context=injected_identity,
        fingerprint_codec=codec,
    )
    if unready_reason == "missing_persisted_identity":
        with db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM rag_identity_context WHERE context_name = 'default'"
            )

    assert repository.local_citation_writes_ready is False


def test_local_citation_writes_ready_identity_read_failure_is_content_free(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(db)
    governed_identity = "governed-authority-do-not-log"
    exception_text = "identity-read-exception-do-not-log"

    def fail_identity_read(_db: CharactersRAGDB) -> None:
        raise sqlite3.DatabaseError(f"{exception_text}: {governed_identity}")

    monkeypatch.setattr(
        "tldw_chatbook.Chat.citation_trace_repository."
        "load_local_citation_identity_context",
        fail_identity_read,
    )

    assert repository.local_citation_writes_ready is False
    captured = capsys.readouterr()
    output = f"{captured.out}\n{captured.err}"
    assert governed_identity not in output
    assert exception_text not in output


def test_create_local_trace_builder_uses_local_citation_writes_ready_contract(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(db)
    readiness_reads = 0

    def unready(_repository: CitationTraceRepository) -> bool:
        nonlocal readiness_reads
        readiness_reads += 1
        return False

    monkeypatch.setattr(
        CitationTraceRepository,
        "local_citation_writes_ready",
        property(unready),
    )

    assert (
        repository.create_local_trace_builder(
            request_id="request-shared-readiness",
            generation_id="generation-shared-readiness",
        )
        is None
    )
    assert readiness_reads == 1


def test_local_trace_builder_disabled_returns_before_identity_read(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(db, enabled=False)
    identity_reads = 0

    def fail_on_identity_read(_db: CharactersRAGDB) -> None:
        nonlocal identity_reads
        identity_reads += 1
        raise AssertionError("disabled capture must not read identity")

    monkeypatch.setattr(
        "tldw_chatbook.Chat.citation_trace_repository."
        "load_local_citation_identity_context",
        fail_on_identity_read,
    )

    assert (
        repository.create_local_trace_builder(
            request_id="request-disabled",
            generation_id="generation-disabled",
        )
        is None
    )
    assert identity_reads == 0


def test_local_trace_builder_canonical_capture_returns_request_scoped_builder(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)

    builder = repository.create_local_trace_builder(
        request_id="request-capture",
        generation_id="generation-capture",
    )

    assert isinstance(builder, CitationTraceBuilder)
    assert builder.request_id == "request-capture"
    assert builder.generation_id == "generation-capture"
    assert builder.is_sealed is False
    assert builder.evidence_runs == ()
    assert builder.prompt_evidence_sets == ()


def test_repository_factory_owns_fixed_closed_retrieval_policy(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(db)
    sentinel = object()
    captured: dict[str, object] = {}

    def capture_local_builder(
        cls: type[CitationTraceBuilder],
        **kwargs: object,
    ) -> object:
        del cls
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        CitationTraceBuilder,
        "local",
        classmethod(capture_local_builder),
    )

    result = repository.create_local_trace_builder(
        request_id="request-policy",
        generation_id="generation-policy",
    )

    assert result is sentinel
    assert captured["policy_version"] == "local-prompt-provenance-v1"
    assert captured["policy_capabilities"] == (
        PolicyCapability.VIEW_SNAPSHOT,
        PolicyCapability.VIEW_SOURCE_IDENTITY,
    )
    assert PolicyCapability.RESOLVE_CURRENT_SOURCE not in captured[
        "policy_capabilities"
    ]
    assert PolicyCapability.OPEN_NATIVE not in captured["policy_capabilities"]
    assert PolicyCapability.OPEN_EXTERNAL not in captured["policy_capabilities"]
    assert set(captured) == {
        "request_id",
        "generation_id",
        "identity_context",
        "fingerprint_codec",
        "policy_version",
        "policy_capabilities",
    }


def test_local_trace_builder_canonical_capture_without_injected_identity_returns_none(
    db: CharactersRAGDB,
) -> None:
    repository = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
        identity_context=None,
        fingerprint_codec=TEST_FINGERPRINT_CODEC,
    )

    assert (
        repository.create_local_trace_builder(
            request_id="request-no-identity",
            generation_id="generation-no-identity",
        )
        is None
    )


def test_local_trace_builder_canonical_capture_without_codec_returns_none(
    db: CharactersRAGDB,
) -> None:
    repository = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
        identity_context=_identity(db),
        fingerprint_codec=None,
    )

    assert (
        repository.create_local_trace_builder(
            request_id="request-no-codec",
            generation_id="generation-no-codec",
        )
        is None
    )


def test_local_trace_builder_canonical_capture_without_persisted_identity_returns_none(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    with db.transaction() as cursor:
        cursor.execute(
            "DELETE FROM rag_identity_context WHERE context_name = 'default'"
        )

    assert (
        repository.create_local_trace_builder(
            request_id="request-no-persisted-identity",
            generation_id="generation-no-persisted-identity",
        )
        is None
    )


def test_local_trace_builder_canonical_capture_identity_mismatch_returns_none(
    db: CharactersRAGDB,
) -> None:
    mismatched = _identity(db).model_copy(
        update={"local_authority_id": "replacement-authority"}
    )
    repository = _repository(db, identity=mismatched)

    assert (
        repository.create_local_trace_builder(
            request_id="request-identity-mismatch",
            generation_id="generation-identity-mismatch",
        )
        is None
    )


def test_local_trace_builder_canonical_capture_identity_read_failure_returns_none(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(db)

    def fail_identity_read(_db: CharactersRAGDB) -> None:
        raise sqlite3.DatabaseError("identity table unreadable")

    monkeypatch.setattr(
        "tldw_chatbook.Chat.citation_trace_repository."
        "load_local_citation_identity_context",
        fail_identity_read,
    )

    assert (
        repository.create_local_trace_builder(
            request_id="request-identity-read-failure",
            generation_id="generation-identity-read-failure",
        )
        is None
    )


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


def test_preflight_validation_error_traceback_does_not_expose_governed_input(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    sentinel = "governed-error-sentinel"
    write = _sealed_write()
    hostile = write.model_copy(
        update={
            "evidence_snapshot_payloads": (
                write.evidence_snapshot_payloads[0].model_copy(
                    update={"snapshot_text": sentinel * 10_000}
                ),
            )
        }
    )

    with pytest.raises(CitationPersistenceUnavailable) as captured:
        repository.prepare_write(hostile)

    assert sentinel not in "".join(traceback.format_exception(captured.value))


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


def test_preflight_requires_the_selected_attempt_exact_answer_body(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    write = _sealed_write()
    selected = write.trace.answer_attempts[0]
    without_occurrences = selected.model_copy(update={"occurrences": ()})
    unavailable = write.model_copy(
        update={
            "trace": write.trace.model_copy(
                update={"answer_attempts": (without_occurrences,)}
            ),
            "answer_attempt_payloads": (
                write.answer_attempt_payloads[0].model_copy(
                    update={"answer_body": None, "body_integrity_hmac": None}
                ),
            ),
        }
    )

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="selected_answer_payload_unavailable",
    ):
        repository.prepare_write(unavailable)


def test_preflight_rejects_a_tampered_selected_answer_fingerprint(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    write = _sealed_write()
    tampered = write.model_copy(
        update={
            "answer_attempt_payloads": (
                write.answer_attempt_payloads[0].model_copy(
                    update={"body_integrity_hmac": "tampered-answer-hmac"}
                ),
            ),
        }
    )

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="selected_answer_integrity_mismatch",
    ):
        repository.prepare_write(tampered)


def test_preflight_does_not_bind_a_diagnostic_attempt_body_to_the_final_message(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    write = _sealed_write()
    diagnostic_attempt = AnswerAttempt(
        attempt_id="attempt-2",
        attempt_ordinal=2,
        kind=AnswerAttemptKind.PIPELINE_RERUN,
        prompt_evidence_set_id=write.trace.prompt_evidence_sets[0].prompt_set_id,
        answer_payload_ref="answer-payload-2",
        occurrences=(),
        created_at=NOW,
    )
    with_diagnostic = write.model_copy(
        update={
            "trace": write.trace.model_copy(
                update={
                    "answer_attempts": (
                        write.trace.answer_attempts[0],
                        diagnostic_attempt,
                    )
                }
            ),
            "answer_attempt_payloads": (
                write.answer_attempt_payloads[0],
                AnswerAttemptPayload(
                    payload_id="answer-payload-2",
                    attempt_id="attempt-2",
                    answer_body="Discarded diagnostic answer.",
                    body_integrity_hmac="diagnostic-hmac-is-not-the-final-body-hmac",
                ),
            ),
        }
    )

    prepared = repository.prepare_write(with_diagnostic)

    assert prepared.selected_answer_body == "Answer [S1]."
    assert len(prepared.answer_rows) == 2


def test_write_prepared_rejects_a_body_other_than_the_selected_answer_before_insert(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())
    conversation_id = _conversation(db)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="selected_answer_message_mismatch",
    ):
        with db.transaction() as cursor:
            db.add_message(
                {
                    "id": "selected-body-mismatch",
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "Different persisted answer.",
                    "client_id": db.client_id,
                }
            )
            repository.write_prepared(
                cursor,
                prepared,
                message_id="selected-body-mismatch",
                message_revision=1,
                message_body="Different persisted answer.",
            )

    assert db.get_message_by_id("selected-body-mismatch") is None
    _assert_no_citation_rows(db)


@pytest.mark.parametrize(
    ("persisted_body", "persisted_revision"),
    [
        ("Different persisted answer.", 1),
        ("Answer [S1].", 2),
    ],
)
def test_write_prepared_verifies_the_actual_message_row_before_insert(
    db: CharactersRAGDB,
    persisted_body: str,
    persisted_revision: int,
) -> None:
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())
    conversation_id = _conversation(db)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="message_row_identity_conflict",
    ):
        with db.transaction() as cursor:
            db.add_message(
                {
                    "id": "persisted-message-mismatch",
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": persisted_body,
                    "client_id": db.client_id,
                }
            )
            if persisted_revision != 1:
                cursor.execute(
                    "UPDATE messages SET version = ? WHERE id = ?",
                    (persisted_revision, "persisted-message-mismatch"),
                )
            repository.write_prepared(
                cursor,
                prepared,
                message_id="persisted-message-mismatch",
                message_revision=1,
                message_body="Answer [S1].",
            )

    assert db.get_message_by_id("persisted-message-mismatch") is None
    _assert_no_citation_rows(db)


def test_write_prepared_revalidates_execution_identity_before_insert(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())
    conversation_id = _conversation(db)
    with db.transaction() as cursor:
        cursor.execute(
            """
            UPDATE rag_identity_context
            SET local_authority_id = ?
            WHERE context_name = 'default'
            """,
            ("replacement-authority",),
        )

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="citation_identity_context_mismatch",
    ):
        with db.transaction() as cursor:
            db.add_message(
                {
                    "id": "changed-execution-identity",
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "Answer [S1].",
                    "client_id": db.client_id,
                }
            )
            repository.write_prepared(
                cursor,
                prepared,
                message_id="changed-execution-identity",
                message_revision=1,
                message_body="Answer [S1].",
            )

    assert db.get_message_by_id("changed-execution-identity") is None
    _assert_no_citation_rows(db)


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


def test_prepared_digest_rejects_selected_answer_tampering_before_insert(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())
    object.__setattr__(prepared, "selected_answer_body", "Tampered selected answer.")
    conversation_id = _conversation(db)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="prepared_citation_write_not_owned",
    ):
        with db.transaction() as cursor:
            db.add_message(
                {
                    "id": "selected-answer-tamper",
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "Tampered selected answer.",
                    "client_id": db.client_id,
                }
            )
            repository.write_prepared(
                cursor,
                prepared,
                message_id="selected-answer-tamper",
                message_revision=1,
                message_body="Tampered selected answer.",
            )

    assert db.get_message_by_id("selected-answer-tamper") is None
    _assert_no_citation_rows(db)


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


@pytest.mark.parametrize(
    (
        "sealed_capabilities",
        "request_snapshot",
        "request_identity",
        "expected_state",
    ),
    [
        (
            (PolicyCapability.VIEW_SOURCE_IDENTITY,),
            True,
            True,
            CitationHydrationState.SNAPSHOT_CAPABILITY_DENIED,
        ),
        (
            (
                PolicyCapability.VIEW_SNAPSHOT,
                PolicyCapability.VIEW_SOURCE_IDENTITY,
            ),
            False,
            True,
            CitationHydrationState.SNAPSHOT_CAPABILITY_DENIED,
        ),
        (
            (PolicyCapability.VIEW_SNAPSHOT,),
            True,
            True,
            CitationHydrationState.SOURCE_IDENTITY_CAPABILITY_DENIED,
        ),
        (
            (
                PolicyCapability.VIEW_SNAPSHOT,
                PolicyCapability.VIEW_SOURCE_IDENTITY,
            ),
            True,
            False,
            CitationHydrationState.SOURCE_IDENTITY_CAPABILITY_DENIED,
        ),
        (
            (
                PolicyCapability.VIEW_SNAPSHOT,
                PolicyCapability.VIEW_SOURCE_IDENTITY,
            ),
            True,
            True,
            CitationHydrationState.AUTHORIZED,
        ),
    ],
)
def test_hydration_requires_independent_snapshot_and_identity_capabilities(
    db: CharactersRAGDB,
    sealed_capabilities: tuple[PolicyCapability, ...],
    request_snapshot: bool,
    request_identity: bool,
    expected_state: CitationHydrationState,
) -> None:
    repository = _repository(db)
    write = _sealed_write()
    trace_with_capabilities = write.trace.model_copy(
        update={"policy_capabilities": sealed_capabilities}
    )
    _persist(
        db,
        repository,
        sealed_write=SealedCitationWrite(
            trace=trace_with_capabilities,
            evidence_run_payloads=write.evidence_run_payloads,
            evidence_snapshot_payloads=write.evidence_snapshot_payloads,
            answer_attempt_payloads=write.answer_attempt_payloads,
        ),
    )
    identity = _identity(db)

    result = repository.hydrate_trace(
        local_trace_namespace(identity, trace_id="trace-1"),
        authorization=_authorization(
            identity,
            view_snapshot=request_snapshot,
            view_source_identity=request_identity,
        ),
    )

    assert result.state is expected_state
    assert result.summary is not None
    if expected_state is CitationHydrationState.AUTHORIZED:
        assert result.governed_payloads is not None
    else:
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


def test_hydration_reports_revoked_when_tombstoned_payload_is_also_purged(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_payload_tombstones VALUES (
                ?, 'local_payload_v1', 'snapshot-1', 'snapshot-1',
                'revoked', 'policy-1',
                '2026-07-24T00:00:00+00:00', '2027-07-24T00:00:00+00:00'
            )
            """,
            (identity.profile_id,),
        )
        cursor.execute(
            """
            UPDATE rag_evidence_snapshots
            SET redaction_state = 'purged',
                snapshot_text = NULL,
                title = NULL,
                source_identity_json = NULL,
                locator_json = NULL,
                lineage_json = NULL,
                transformations_json = NULL,
                content_hash = NULL,
                comparison_fingerprint = NULL,
                purged_at = '2026-07-24T00:00:00+00:00'
            """
        )

    result = repository.hydrate_trace(
        local_trace_namespace(identity, trace_id="trace-1"),
        authorization=_authorization(identity),
    )

    assert result.state is CitationHydrationState.REVOKED
    assert result.governed_payloads is None


@pytest.mark.parametrize(
    "mutation",
    [
        "snapshot_redacted",
        "snapshot_authority_changed",
        "snapshot_tombstoned",
        "run_payload_malformed",
    ],
)
def test_hydration_final_statement_rechecks_governance_after_safe_preflight(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    hook_called = False

    with sqlite3.connect(db.db_path_str) as competing_connection:
        competing_connection.execute("PRAGMA foreign_keys = ON")

        def mutate_after_preflight() -> None:
            nonlocal hook_called
            hook_called = True
            if mutation == "snapshot_redacted":
                competing_connection.execute(
                    """
                    UPDATE rag_evidence_snapshots
                    SET redaction_state = 'redacted'
                    WHERE profile_id = ? AND payload_id = 'snapshot-1'
                    """,
                    (identity.profile_id,),
                )
            elif mutation == "snapshot_authority_changed":
                competing_connection.execute(
                    """
                    UPDATE rag_evidence_snapshots
                    SET authority_id = 'hostile-authority'
                    WHERE profile_id = ? AND payload_id = 'snapshot-1'
                    """,
                    (identity.profile_id,),
                )
            elif mutation == "snapshot_tombstoned":
                competing_connection.execute(
                    """
                    INSERT INTO rag_payload_tombstones VALUES (
                        ?, 'local_payload_v1', 'snapshot-1', 'snapshot-1',
                        'revoked', 'policy-1',
                        '2026-07-24T00:00:00Z', '2027-07-24T00:00:00Z'
                    )
                    """,
                    (identity.profile_id,),
                )
            else:
                competing_connection.execute(
                    """
                    UPDATE rag_evidence_runs
                    SET run_payload_json = '{'
                    WHERE profile_id = ? AND run_id = 'run-1'
                    """,
                    (identity.profile_id,),
                )
            competing_connection.commit()

        monkeypatch.setattr(
            repository,
            "_before_governed_select",
            mutate_after_preflight,
            raising=False,
        )
        result = repository.hydrate_trace(
            local_trace_namespace(identity, trace_id="trace-1"),
            authorization=_authorization(identity),
        )

    assert hook_called
    assert result.state is CitationHydrationState.PAYLOAD_UNAVAILABLE
    assert result.governed_payloads is None
    assert "private" not in repr(result)


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
    "mismatch",
    ["run_payload_ref", "answer_payload_ref", "evidence_reference"],
)
def test_hydration_final_statement_rechecks_exact_aggregate_references(
    db: CharactersRAGDB,
    mismatch: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    connection = db.get_connection()
    if mismatch == "run_payload_ref":
        payload = json.loads(
            connection.execute(
                "SELECT run_payload_json FROM rag_evidence_runs"
            ).fetchone()[0]
        )
        payload["payload_id"] = "substituted-run-payload"
        connection.execute(
            "UPDATE rag_evidence_runs SET run_payload_json = ?",
            (json.dumps(payload),),
        )
    elif mismatch == "answer_payload_ref":
        connection.execute(
            """
            UPDATE rag_answer_attempt_payloads
            SET payload_id = 'substituted-answer-payload'
            """
        )
    else:
        connection.execute("UPDATE rag_trace_evidence_refs SET marker_ordinal = 2")
    connection.commit()

    result = repository.hydrate_trace(
        local_trace_namespace(identity, trace_id="trace-1"),
        authorization=_authorization(identity),
    )

    assert result.state is CitationHydrationState.PAYLOAD_UNAVAILABLE
    assert result.summary is not None
    assert result.governed_payloads is None
    assert "private" not in repr(result)


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


def test_committed_aggregate_retry_reuses_every_exact_row_identity(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    prepared = repository.prepare_write(_sealed_write())

    with db.transaction() as cursor:
        repository.write_prepared(
            cursor,
            prepared,
            message_id="message-1",
            message_revision=1,
            message_body="Answer [S1].",
        )

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
        assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == (
            expected
        )


@pytest.mark.parametrize("mutation", ("trace", "governed", "body", "owner"))
def test_committed_identity_reuse_with_different_immutable_data_fails_closed(
    db: CharactersRAGDB,
    mutation: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    write = _sealed_write()
    message_id = "message-1"
    message_body = "Answer [S1]."
    if mutation == "trace":
        write = write.model_copy(
            update={
                "trace": write.trace.model_copy(
                    update={"generation_id": "different-generation"}
                )
            }
        )
    elif mutation == "governed":
        snapshot = write.evidence_snapshot_payloads[0]
        write = write.model_copy(
            update={
                "evidence_snapshot_payloads": (
                    snapshot.model_copy(update={"title": "different-title"}),
                )
            }
        )
    elif mutation == "body":
        message_body = "Changed answer [S1]."
    else:
        message_id = "different-owner"
        conversation_id = _conversation(db)
        db.add_message(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": message_body,
                "client_id": db.client_id,
            }
        )
        with db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE rag_message_trace_owners
                SET idempotency_key = 'hostile-owner-key'
                WHERE message_id = 'message-1'
                """
            )
        message_id = "message-1"

    prepared = repository.prepare_write(write)
    expected_reason = (
        "selected_answer_message_mismatch"
        if mutation == "body"
        else "identity_conflict"
    )
    with pytest.raises(CitationPersistenceUnavailable, match=expected_reason):
        with db.transaction() as cursor:
            repository.write_prepared(
                cursor,
                prepared,
                message_id=message_id,
                message_revision=1,
                message_body=message_body,
            )

    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_citation_traces")
        .fetchone()[0]
        == 1
    )
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_message_trace_owners")
        .fetchone()[0]
        == 1
    )


def test_cache_reuse_adds_an_idempotent_owner_without_cloning_trace(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    namespace = local_trace_namespace(identity, trace_id="trace-1")
    conversation_id = _conversation(db)
    db.add_message(
        {
            "id": "cache-message",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Answer [S1].",
            "client_id": db.client_id,
        }
    )

    for _ in range(2):
        with db.transaction() as cursor:
            repository.link_cache_message_owner(
                cursor,
                namespace,
                message_id="cache-message",
                message_revision=1,
                message_body="Answer [S1].",
            )

    connection = db.get_connection()
    trace = connection.execute(
        "SELECT trace_id, generation_id FROM rag_citation_traces"
    ).fetchone()
    owners = connection.execute(
        """
        SELECT message_id, trace_id, body_fingerprint, idempotency_key
        FROM rag_message_trace_owners
        ORDER BY message_id
        """
    ).fetchall()
    assert tuple(trace) == ("trace-1", "generation-1")
    assert [(row["message_id"], row["trace_id"]) for row in owners] == [
        ("cache-message", "trace-1"),
        ("message-1", "trace-1"),
    ]
    assert owners[0]["idempotency_key"] != owners[1]["idempotency_key"]
    selected_hmac = connection.execute(
        """
        SELECT payload.body_integrity_hmac
        FROM rag_citation_traces AS trace
        JOIN rag_answer_attempt_payloads AS payload
          ON payload.profile_id = trace.profile_id
         AND payload.trace_id = trace.trace_id
         AND payload.attempt_id = trace.selected_attempt_id
        WHERE trace.trace_id = 'trace-1'
        """
    ).fetchone()[0]
    assert hmac.compare_digest(owners[0]["body_fingerprint"], selected_hmac)


@pytest.mark.parametrize(
    ("selected_payload_state", "cached_body", "reason"),
    [
        ("available", "Different cached answer.", "cache_selected_answer_mismatch"),
        ("purged", "Answer [S1].", "cache_selected_answer_unavailable"),
        ("redacted", "Answer [S1].", "cache_selected_answer_unavailable"),
        ("missing", "Answer [S1].", "cache_selected_answer_unavailable"),
        ("tampered", "Answer [S1].", "cache_selected_answer_integrity_mismatch"),
    ],
)
def test_cache_reuse_requires_the_exact_available_selected_answer(
    db: CharactersRAGDB,
    selected_payload_state: str,
    cached_body: str,
    reason: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    namespace = local_trace_namespace(identity, trace_id="trace-1")
    conversation_id = _conversation(db)
    db.add_message(
        {
            "id": "untrusted-cache-message",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": cached_body,
            "client_id": db.client_id,
        }
    )
    connection = db.get_connection()
    if selected_payload_state == "missing":
        with db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM rag_answer_attempt_payloads WHERE trace_id = 'trace-1'"
            )
    elif selected_payload_state == "purged":
        with db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE rag_answer_attempt_payloads
                SET redaction_state = 'purged', answer_body = NULL,
                    body_integrity_hmac = NULL, purged_at = ?
                WHERE trace_id = 'trace-1'
                """,
                (NOW.isoformat(),),
            )
    elif selected_payload_state == "redacted":
        connection.execute("PRAGMA ignore_check_constraints = ON")
        try:
            with db.transaction() as cursor:
                cursor.execute(
                    """
                    UPDATE rag_answer_attempt_payloads
                    SET redaction_state = 'redacted', answer_body = NULL,
                        body_integrity_hmac = NULL
                    WHERE trace_id = 'trace-1'
                    """
                )
        finally:
            connection.execute("PRAGMA ignore_check_constraints = OFF")
    elif selected_payload_state == "tampered":
        with db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE rag_answer_attempt_payloads
                SET body_integrity_hmac = 'tampered-cache-answer-hmac'
                WHERE trace_id = 'trace-1'
                """
            )

    with pytest.raises(CitationPersistenceUnavailable, match=reason):
        with db.transaction() as cursor:
            repository.link_cache_message_owner(
                cursor,
                namespace,
                message_id="untrusted-cache-message",
                message_revision=1,
                message_body=cached_body,
            )

    assert (
        connection.execute(
            """
            SELECT count(*) FROM rag_message_trace_owners
            WHERE message_id = 'untrusted-cache-message'
            """
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize(
    ("message_revision", "message_body"),
    [
        (2, "Answer [S1]."),
        (1, "Different caller body."),
    ],
)
def test_cache_reuse_requires_the_exact_persisted_message_revision_and_body(
    db: CharactersRAGDB,
    message_revision: int,
    message_body: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    conversation_id = _conversation(db)
    db.add_message(
        {
            "id": "cache-message-row-mismatch",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Answer [S1].",
            "client_id": db.client_id,
        }
    )

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="cache_owner_identity_conflict",
    ):
        with db.transaction() as cursor:
            repository.link_cache_message_owner(
                cursor,
                local_trace_namespace(identity, trace_id="trace-1"),
                message_id="cache-message-row-mismatch",
                message_revision=message_revision,
                message_body=message_body,
            )

    assert (
        db.get_connection()
        .execute(
            """
            SELECT count(*) FROM rag_message_trace_owners
            WHERE message_id = 'cache-message-row-mismatch'
            """
        )
        .fetchone()[0]
        == 0
    )


def test_active_lookup_verifies_body_and_preserves_historical_summary_on_mismatch(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    codec = CitationFingerprintCodec(b"k" * 32)

    active = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        codec,
    )
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("Edited answer [S1].", "message-1"),
        )
    mismatch = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Edited answer [S1].",
        codec,
    )

    assert active.state is ActiveCitationTraceState.ACTIVE
    assert active.summary is not None
    assert active.summary.trace.trace_id == "trace-1"
    assert mismatch.state is ActiveCitationTraceState.BODY_MISMATCH
    assert mismatch.summary is None
    assert (
        db.get_connection()
        .execute(
            """
        SELECT state FROM rag_message_trace_owners
        WHERE message_id = 'message-1'
        """
        )
        .fetchone()[0]
        == "body_mismatch"
    )
    assert (
        repository.get_trace_summary(
            local_trace_namespace(_identity(db), trace_id="trace-1")
        )
        is not None
    )


def test_active_lookup_with_missing_or_wrong_codec_is_unverifiable_without_mutation(
    db: CharactersRAGDB,
) -> None:
    writer = _repository(db)
    _persist(db, writer)
    identity = _identity(db)
    no_key_reader = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=False),
        identity_context=identity,
        fingerprint_codec=None,
    )

    missing = no_key_reader.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        None,
    )
    wrong = writer.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        CitationFingerprintCodec(b"x" * 32),
    )

    assert missing.state is ActiveCitationTraceState.UNVERIFIABLE
    assert wrong.state is ActiveCitationTraceState.UNVERIFIABLE
    assert missing.summary is None
    assert wrong.summary is None
    assert (
        db.get_connection()
        .execute(
            """
        SELECT state FROM rag_message_trace_owners
        WHERE message_id = 'message-1'
        """
        )
        .fetchone()[0]
        == "active"
    )


def test_active_lookup_never_promotes_summary_without_owner_verification(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    identity = _identity(db)
    summary = repository.get_trace_summary(
        local_trace_namespace(identity, trace_id="trace-1")
    )

    result = repository.get_active_trace_for_message(
        "message-without-owner",
        1,
        "Answer [S1].",
        CitationFingerprintCodec(b"k" * 32),
    )

    assert summary is not None
    assert result.state is ActiveCitationTraceState.NOT_FOUND
    assert result.summary is None
    with pytest.raises(ValueError, match="repository-issued"):
        ActiveCitationTraceResult(
            state=ActiveCitationTraceState.ACTIVE,
            summary=None,
        )
    with pytest.raises(ValueError, match="repository-issued"):
        ActiveCitationTraceResult(
            state=ActiveCitationTraceState.ACTIVE,
            summary=summary,
        )
    with pytest.raises(ValueError, match="repository-issued"):
        ActiveCitationTraceResult(
            state=ActiveCitationTraceState.UNVERIFIABLE,
            summary=summary,
        )
    with pytest.raises(TypeError, match="ActiveCitationTraceState"):
        ActiveCitationTraceResult(state="active")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="repository-issued"):
        replace(
            result,
            state=ActiveCitationTraceState.ACTIVE,
            summary=summary,
        )
    assert not hasattr(ActiveCitationTraceResult, "model_construct")
    assert not hasattr(result, "model_copy")


def test_active_result_requires_repository_issuance_and_exact_object_verification(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    active = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )

    assert active.state is ActiveCitationTraceState.ACTIVE
    assert active.summary is not None
    assert repository.verify_active_trace_result(active) is True
    with pytest.raises((TypeError, ValueError)):
        replace(active)
    with pytest.raises(TypeError):
        copy.copy(active)

    forged = object.__new__(ActiveCitationTraceResult)
    object.__setattr__(forged, "state", ActiveCitationTraceState.ACTIVE)
    object.__setattr__(forged, "summary", active.summary)
    assert repository.verify_active_trace_result(forged) is False

    object.__setattr__(active, "state", ActiveCitationTraceState.UNVERIFIABLE)
    assert repository.verify_active_trace_result(active) is False
    assert id(active) not in repository._issued_active_results


@pytest.mark.parametrize(
    "state_change",
    (
        "message_edit",
        "message_delete",
        "owner_body_mismatch",
        "owner_delete",
        "trace_visibility",
    ),
)
def test_active_result_replay_fails_after_persisted_state_changes(
    db: CharactersRAGDB,
    state_change: str,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    active = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )
    assert repository.verify_active_trace_result(active) is True

    if state_change == "message_edit":
        message = db.get_message_by_id("message-1")
        assert message is not None
        db.update_message(
            "message-1",
            {"content": "Edited answer [S1]."},
            expected_version=message["version"],
        )
    elif state_change == "message_delete":
        db.soft_delete_message("message-1", expected_version=1)
    elif state_change == "owner_body_mismatch":
        with db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE rag_message_trace_owners
                SET state = 'body_mismatch'
                WHERE message_id = 'message-1'
                """
            )
    elif state_change == "owner_delete":
        with db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM rag_message_trace_owners WHERE message_id = 'message-1'"
            )
    else:
        with db.transaction() as cursor:
            cursor.execute(
                """
                UPDATE rag_citation_traces
                SET visibility_state = 'migrating'
                WHERE trace_id = 'trace-1'
                """
            )

    assert repository.verify_active_trace_result(active) is False
    assert id(active) not in repository._issued_active_results


def test_active_result_replay_fails_after_identity_drift(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    active = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )
    assert repository.verify_active_trace_result(active) is True
    with db.transaction() as cursor:
        cursor.execute(
            """
            UPDATE rag_identity_context
            SET local_authority_id = 'replacement-authority'
            WHERE context_name = 'default'
            """
        )

    assert repository.verify_active_trace_result(active) is False
    assert id(active) not in repository._issued_active_results


def test_active_result_is_not_transferable_and_registration_expires_with_gc(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    other_repository = _repository(db)
    _persist(db, repository)
    active = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        TEST_FINGERPRINT_CODEC,
    )
    active_id = id(active)
    active_ref = weakref.ref(active)

    assert other_repository.verify_active_trace_result(active) is False
    assert repository.verify_active_trace_result(active) is True

    del active
    gc.collect()

    assert active_ref() is None
    assert active_id not in repository._issued_active_results


def test_active_lookup_rejects_stale_caller_body_that_is_not_the_message_row(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)
    current = db.get_message_by_id("message-1")
    assert current is not None
    db.update_message(
        "message-1",
        {"content": "Edited outside citation service [S1]."},
        expected_version=current["version"],
    )

    result = repository.get_active_trace_for_message(
        "message-1",
        1,
        "Answer [S1].",
        CitationFingerprintCodec(b"k" * 32),
    )

    assert result.state is ActiveCitationTraceState.UNVERIFIABLE
    assert result.summary is None


def test_owner_transition_rejects_a_new_revision_not_bound_to_the_message_row(
    db: CharactersRAGDB,
) -> None:
    repository = _repository(db)
    _persist(db, repository)

    with pytest.raises(
        CitationPersistenceUnavailable,
        match="message_revision_identity_conflict",
    ):
        with db.transaction() as cursor:
            repository.transition_owner_for_message_update(
                cursor,
                message_id="message-1",
                previous_revision=1,
                new_revision=2,
                new_body="Answer [S1].",
            )

    assert (
        db.get_connection()
        .execute(
            """
        SELECT count(*) FROM rag_message_trace_owners
        WHERE message_id = 'message-1'
        """
        )
        .fetchone()[0]
        == 1
    )
