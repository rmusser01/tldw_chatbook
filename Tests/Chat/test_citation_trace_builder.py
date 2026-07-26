from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import warnings

import pytest
from pydantic import BaseModel, ValidationError

import tldw_chatbook.Chat.citation_trace_builder as builder_module
from tldw_chatbook.Chat.citation_source_locators import CanonicalSourceKind
from tldw_chatbook.Chat.citation_trace_builder import (
    CitationTraceBuilder,
    LocalPromptEvidenceCapture,
    LocalRetrievalCandidateCapture,
    LocalRetrievalRunMetadata,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    CitationFingerprintCodec,
    CitationFingerprintDomain,
    LocalCitationIdentityContext,
)
from tldw_chatbook.Chat.citation_trace_models import (
    ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX,
    EVIDENCE_ENTRIES_PER_PROMPT_MAX,
    PROMPT_EVIDENCE_SETS_MAX,
    RETRIEVAL_CANDIDATES_PER_RUN_MAX,
    SNAPSHOT_TEXT_UTF8_BYTES_MAX,
    AnswerAttemptKind,
    CitationCompleteness,
    EvidenceStorageMode,
    MarkerNamespace,
    PolicyCapability,
    RetrievalScoreKind,
    RetrievalScoreScale,
    TraceLifecycle,
    TraceOrigin,
    reduce_selected_attempt_completeness,
)


NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
SECRET = b"0123456789abcdef0123456789abcdef"
TEST_POLICY_VERSION = "test-local-policy-v1"
TEST_POLICY_CAPABILITIES = (
    PolicyCapability.VIEW_SNAPSHOT,
    PolicyCapability.VIEW_SOURCE_IDENTITY,
)


def _identity() -> LocalCitationIdentityContext:
    return LocalCitationIdentityContext(
        profile_id="profile-1",
        local_authority_id="local-authority-1",
        fingerprint_key_id="fingerprint-key-1",
    )


def _builder() -> CitationTraceBuilder:
    return CitationTraceBuilder.local(
        request_id="request-1",
        generation_id="generation-1",
        identity_context=_identity(),
        fingerprint_codec=CitationFingerprintCodec(SECRET),
        policy_version=TEST_POLICY_VERSION,
        policy_capabilities=TEST_POLICY_CAPABILITIES,
        created_at=NOW,
    )


def _candidate(
    *,
    rank: int = 1,
    source_id: str = "media-1",
    title: str = "Alpha",
    score: float = 0.9,
) -> LocalRetrievalCandidateCapture:
    return LocalRetrievalCandidateCapture(
        candidate_rank=rank,
        source_kind=CanonicalSourceKind.MEDIA_DB,
        source_id=source_id,
        title=title,
        score_kind=RetrievalScoreKind.VECTOR_SIMILARITY,
        score_scale=RetrievalScoreScale.ZERO_TO_ONE,
        score=score,
        chunk_index=3,
        start_char=10,
        end_char=20,
    )


def _metadata() -> LocalRetrievalRunMetadata:
    return LocalRetrievalRunMetadata(
        search_mode="hybrid",
        requested_top_k=5,
        max_context_characters=10_000,
        rerank_enabled=True,
        source_kinds=(CanonicalSourceKind.MEDIA_DB,),
        scope_state="unscoped",
    )


def _record_run(builder: CitationTraceBuilder) -> str:
    return builder.record_retrieval_run(
        stage="hybrid",
        raw_query="secret query",
        candidates=(
            _candidate(rank=1, source_id="media-1", title="Alpha"),
            _candidate(rank=2, source_id="media-2", title="Beta"),
        ),
        retrieval_metadata=_metadata(),
        started_at=NOW,
        ended_at=NOW,
    )


def _record_prompt_set(
    builder: CitationTraceBuilder,
    *,
    run_id: str | None = None,
    created_at: datetime = NOW,
) -> str:
    linked_run_id = run_id if run_id is not None else _record_run(builder)
    return builder.record_prompt_evidence_set(
        run_id=linked_run_id,
        evidence=(
            LocalPromptEvidenceCapture(
                candidate_rank=1,
                snapshot_text="[S1] MEDIA — Alpha\nExact",
            ),
        ),
        created_at=created_at,
    )


def _builder_with_initial_answer(
    *,
    answer_body: str = "Marker-free exact answer.",
    completed_at: datetime = NOW,
) -> tuple[CitationTraceBuilder, str]:
    builder = _builder()
    prompt_set_id = _record_prompt_set(builder)
    attempt_id = builder.record_initial_answer_attempt(
        prompt_evidence_set_id=prompt_set_id,
        answer_body=answer_body,
        completed_at=completed_at,
    )
    return builder, attempt_id


def _compact_model_json_bytes(model: BaseModel) -> int:
    return len(
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def test_local_builder_starts_empty_unsealed_and_redacts_private_context() -> None:
    builder = _builder()

    assert builder.request_id == "request-1"
    assert builder.generation_id == "generation-1"
    assert builder.created_at == NOW
    assert builder.evidence_runs == ()
    assert builder.evidence_run_payloads == ()
    assert builder.prompt_evidence_sets == ()
    assert builder.evidence_snapshot_payloads == ()
    assert builder.answer_attempts == ()
    assert builder.answer_attempt_payloads == ()
    assert builder.is_sealed is False
    assert hasattr(builder, "seal")
    assert not hasattr(builder, "persist")
    assert not hasattr(builder, "prepare_write")
    assert SECRET.decode() not in repr(builder)
    assert "fingerprint-key-1" not in repr(builder)


def test_local_builder_requires_a_valid_frozen_local_identity_binding() -> None:
    identity = _identity()

    with pytest.raises(ValidationError, match="frozen"):
        identity.profile_id = "changed"  # type: ignore[misc]

    missing_profile = LocalCitationIdentityContext.model_construct(
        profile_id="",
        local_authority_id="local-authority-1",
        fingerprint_key_id="fingerprint-key-1",
    )
    with pytest.raises((TypeError, ValueError, ValidationError), match="identifier"):
        CitationTraceBuilder.local(
            request_id="request-1",
            generation_id="generation-1",
            identity_context=missing_profile,
            fingerprint_codec=CitationFingerprintCodec(SECRET),
            policy_version=TEST_POLICY_VERSION,
            policy_capabilities=TEST_POLICY_CAPABILITIES,
            created_at=NOW,
        )
    with pytest.raises(TypeError, match="identity_context"):
        CitationTraceBuilder.local(
            request_id="request-1",
            generation_id="generation-1",
            identity_context=None,  # type: ignore[arg-type]
            fingerprint_codec=CitationFingerprintCodec(SECRET),
            policy_version=TEST_POLICY_VERSION,
            policy_capabilities=TEST_POLICY_CAPABILITIES,
            created_at=NOW,
        )


def test_local_builder_requires_explicit_policy_metadata() -> None:
    with pytest.raises(TypeError):
        CitationTraceBuilder.local(
            request_id="request-1",
            generation_id="generation-1",
            identity_context=_identity(),
            fingerprint_codec=CitationFingerprintCodec(SECRET),
            created_at=NOW,
        )


@pytest.mark.parametrize(
    ("request_id", "generation_id", "created_at"),
    [
        ("", "generation-1", NOW),
        ("request-1", "", NOW),
        ("request-1", "generation-1", NOW.replace(tzinfo=None)),
    ],
)
def test_local_builder_rejects_invalid_request_identity_or_timestamp(
    request_id: str,
    generation_id: str,
    created_at: datetime,
) -> None:
    with pytest.raises((ValueError, ValidationError)):
        CitationTraceBuilder.local(
            request_id=request_id,
            generation_id=generation_id,
            identity_context=_identity(),
            fingerprint_codec=CitationFingerprintCodec(SECRET),
            policy_version=TEST_POLICY_VERSION,
            policy_capabilities=TEST_POLICY_CAPABILITIES,
            created_at=created_at,
        )


def test_local_builder_rejects_falsy_non_datetime_created_at() -> None:
    with pytest.raises(ValidationError):
        CitationTraceBuilder.local(
            request_id="request-1",
            generation_id="generation-1",
            identity_context=_identity(),
            fingerprint_codec=CitationFingerprintCodec(SECRET),
            policy_version=TEST_POLICY_VERSION,
            policy_capabilities=TEST_POLICY_CAPABILITIES,
            created_at=0,  # type: ignore[arg-type]
        )


def test_strict_revalidation_never_leaks_sensitive_values_to_warnings_or_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity_sentinel = "identity-secret-warning-sentinel"
    metadata_sentinel = "metadata-secret-warning-sentinel"
    candidate_sentinel = "candidate-secret-warning-sentinel"
    snapshot_sentinel = "snapshot-secret-warning-sentinel"
    forged_identity = LocalCitationIdentityContext.model_construct(
        profile_id=[identity_sentinel],
        local_authority_id="local-authority-1",
        fingerprint_key_id="fingerprint-key-1",
    )
    forged_metadata = LocalRetrievalRunMetadata.model_construct(
        search_mode=[metadata_sentinel],
        requested_top_k=5,
        max_context_characters=10_000,
        rerank_enabled=True,
        source_kinds=(CanonicalSourceKind.MEDIA_DB,),
        scope_state="unscoped",
    )
    forged_candidate = LocalRetrievalCandidateCapture.model_construct(
        **{
            **_candidate().model_dump(mode="python"),
            "source_id": [candidate_sentinel],
        }
    )
    forged_snapshot = LocalPromptEvidenceCapture.model_construct(
        candidate_rank=1,
        snapshot_text=[snapshot_sentinel],
        transformations=(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError):
            CitationTraceBuilder.local(
                request_id="request-1",
                generation_id="generation-1",
                identity_context=forged_identity,
                fingerprint_codec=CitationFingerprintCodec(SECRET),
                policy_version=TEST_POLICY_VERSION,
                policy_capabilities=TEST_POLICY_CAPABILITIES,
                created_at=NOW,
            )
        with pytest.raises(ValidationError):
            _builder().record_retrieval_run(
                stage="hybrid",
                raw_query="secret query",
                candidates=(_candidate(),),
                retrieval_metadata=forged_metadata,
                started_at=NOW,
                ended_at=NOW,
            )
        with pytest.raises(ValidationError):
            _builder().record_retrieval_run(
                stage="hybrid",
                raw_query="secret query",
                candidates=(forged_candidate,),
                retrieval_metadata=_metadata(),
                started_at=NOW,
                ended_at=NOW,
            )
        prompt_builder = _builder()
        run_id = _record_run(prompt_builder)
        with pytest.raises(ValidationError):
            prompt_builder.record_prompt_evidence_set(
                run_id=run_id,
                evidence=(forged_snapshot,),
                created_at=NOW,
            )

    captured = capsys.readouterr()
    leak_text = "\n".join(
        [*(str(item.message) for item in caught), captured.out, captured.err]
    )
    for sentinel in (
        identity_sentinel,
        metadata_sentinel,
        candidate_sentinel,
        snapshot_sentinel,
    ):
        assert sentinel not in leak_text


def test_local_capture_types_reject_non_local_source_families() -> None:
    with pytest.raises(ValidationError, match="local source"):
        LocalRetrievalCandidateCapture(
            candidate_rank=1,
            source_kind=CanonicalSourceKind.WEB_CONTENT,
            source_id="web-1",
            title="Remote result",
        )
    with pytest.raises(ValidationError, match="local source"):
        LocalRetrievalRunMetadata(
            search_mode="hybrid",
            requested_top_k=5,
            max_context_characters=10_000,
            rerank_enabled=False,
            source_kinds=(CanonicalSourceKind.WEB_CONTENT,),
            scope_state="unscoped",
        )


def test_record_retrieval_run_preserves_order_and_governs_sensitive_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    builder = _builder()
    codec = CitationFingerprintCodec(SECRET)
    candidates = (
        _candidate(rank=2, source_id="media-2", title="Sensitive Beta"),
        _candidate(rank=1, source_id="media-1", title="Sensitive Alpha"),
    )

    run_id = builder.record_retrieval_run(
        stage="hybrid",
        raw_query="secret query",
        candidates=candidates,
        retrieval_metadata=_metadata(),
        started_at=NOW,
        ended_at=NOW,
    )

    run = builder.evidence_runs[0]
    payload = builder.evidence_run_payloads[0]
    assert run.run_id == run_id
    assert run.request_id == "request-1"
    assert run.run_ordinal == 1
    assert run.stage == "hybrid"
    assert run.payload_ref == payload.payload_id
    assert run.started_at == NOW
    assert run.ended_at == NOW
    assert payload.run_id == run_id
    assert payload.raw_query is None
    assert payload.query_fingerprint == codec.fingerprint(
        CitationFingerprintDomain.RAW_QUERY,
        "secret query",
    )
    assert payload.authority_id == "local-authority-1"
    assert payload.retrieval_metadata == {
        "search_mode": "hybrid",
        "requested_top_k": 5,
        "max_context_characters": 10_000,
        "rerank_enabled": True,
        "source_kinds": ["media_db"],
        "scope_state": "unscoped",
    }
    assert [candidate.rank for candidate in payload.candidates] == [2, 1]
    assert [candidate.title for candidate in payload.candidates] == [
        "Sensitive Beta",
        "Sensitive Alpha",
    ]
    assert payload.candidates[0].source_identity == {
        "source_kind": "media_db",
        "source_id": "media-2",
    }
    assert payload.candidates[0].locator == {}
    assert payload.candidates[0].lineage == {
        "chunk_index": 3,
        "start_char": 10,
        "end_char": 20,
    }
    assert payload.candidates[0].score_kind is RetrievalScoreKind.VECTOR_SIMILARITY
    assert payload.candidates[0].score_scale is RetrievalScoreScale.ZERO_TO_ONE
    dumped = payload.model_dump(mode="json")
    assert "content" not in repr(dumped)
    assert "secret query" not in repr(builder)
    assert "Sensitive Alpha" not in repr(builder)
    assert "media-1" not in repr(builder)
    assert payload.query_fingerprint not in repr(builder)
    assert "secret query" not in caplog.text
    assert "Sensitive Alpha" not in caplog.text
    assert payload.query_fingerprint not in caplog.text


def test_governed_payload_views_are_deep_detached_from_builder_state() -> None:
    builder = _builder()
    run_id = _record_run(builder)
    returned_run_payload = builder.evidence_run_payloads[0]
    returned_candidate = returned_run_payload.candidates[0]

    returned_run_payload.retrieval_metadata["search_mode"] = "tampered"
    returned_candidate.source_identity["source_id"] = "tampered-source"
    returned_candidate.lineage["chunk_index"] = 999
    builder.record_prompt_evidence_set(
        run_id=run_id,
        evidence=(
            LocalPromptEvidenceCapture(
                candidate_rank=1,
                snapshot_text="[S1] MEDIA — Alpha\nExact",
            ),
        ),
        created_at=NOW,
    )

    internal_run_payload = builder.evidence_run_payloads[0]
    internal_snapshot = builder.evidence_snapshot_payloads[0]
    assert internal_run_payload.retrieval_metadata["search_mode"] == "hybrid"
    assert internal_run_payload.candidates[0].source_identity["source_id"] == "media-1"
    assert internal_run_payload.candidates[0].lineage["chunk_index"] == 3
    assert internal_snapshot.source_identity["source_id"] == "media-1"
    assert internal_snapshot.lineage["chunk_index"] == 3

    internal_snapshot.source_identity["source_id"] = "tampered-snapshot"
    internal_snapshot.lineage["chunk_index"] = 777
    fresh_snapshot = builder.evidence_snapshot_payloads[0]
    assert fresh_snapshot.source_identity["source_id"] == "media-1"
    assert fresh_snapshot.lineage["chunk_index"] == 3


def test_retrieval_inputs_forbid_free_form_or_executable_metadata() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LocalRetrievalRunMetadata.model_validate(
            {
                **_metadata().model_dump(mode="python"),
                "workspace_path": "/private/workspace",
            }
        )
    with pytest.raises(ValidationError):
        LocalRetrievalRunMetadata.model_validate(
            {
                **_metadata().model_dump(mode="python"),
                "search_mode": "https://example.invalid/search",
            }
        )
    with pytest.raises(ValidationError):
        LocalRetrievalRunMetadata.model_validate(
            {
                **_metadata().model_dump(mode="python"),
                "scope_state": "/private/workspace",
            }
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LocalRetrievalCandidateCapture.model_validate(
            {
                **_candidate().model_dump(mode="python"),
                "content": "candidate content must not cross this boundary",
            }
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LocalRetrievalCandidateCapture.model_validate(
            {
                **_candidate().model_dump(mode="python"),
                "metadata": {"url": "https://example.invalid"},
            }
        )
    for field, value in (
        ("source_id", "https://example.invalid/item"),
        ("source_id", "/private/source"),
        ("chunk_id", "../other/chunk"),
    ):
        with pytest.raises(ValidationError, match="executable path or URL"):
            LocalRetrievalCandidateCapture.model_validate(
                {
                    **_candidate().model_dump(mode="python"),
                    field: value,
                }
            )


def test_retrieval_run_rejects_non_finite_scores_and_candidate_overflow_atomically() -> (
    None
):
    builder = _builder()
    non_finite = LocalRetrievalCandidateCapture.model_construct(
        **{**_candidate().model_dump(mode="python"), "score": float("nan")}
    )

    with pytest.raises(ValidationError, match="finite"):
        builder.record_retrieval_run(
            stage="semantic",
            raw_query="secret query",
            candidates=(non_finite,),
            retrieval_metadata=_metadata(),
            started_at=NOW,
            ended_at=NOW,
        )
    assert builder.evidence_runs == ()
    assert builder.evidence_run_payloads == ()

    with pytest.raises(ValueError, match="candidates"):
        builder.record_retrieval_run(
            stage="semantic",
            raw_query="secret query",
            candidates=(_candidate(),) * (RETRIEVAL_CANDIDATES_PER_RUN_MAX + 1),
            retrieval_metadata=_metadata(),
            started_at=NOW,
            ended_at=NOW,
        )
    assert builder.evidence_runs == ()
    assert builder.evidence_run_payloads == ()


@pytest.mark.parametrize(
    ("stage", "started_at", "ended_at", "message"),
    [
        ("", NOW, NOW, "at least 1 character"),
        ("hybrid", NOW, NOW.replace(tzinfo=None), "timezone-aware"),
        (
            "hybrid",
            NOW,
            datetime(2026, 7, 25, 11, 59, tzinfo=UTC),
            "must not precede",
        ),
    ],
)
def test_invalid_retrieval_run_data_does_not_partially_mutate_state(
    stage: str,
    started_at: datetime,
    ended_at: datetime,
    message: str,
) -> None:
    builder = _builder()

    with pytest.raises(ValidationError, match=message):
        builder.record_retrieval_run(
            stage=stage,
            raw_query="secret query",
            candidates=(_candidate(),),
            retrieval_metadata=_metadata(),
            started_at=started_at,
            ended_at=ended_at,
        )

    assert builder.evidence_runs == ()
    assert builder.evidence_run_payloads == ()


def test_retrieval_run_cannot_start_before_builder_creation_and_is_atomic() -> None:
    builder = _builder()

    with pytest.raises(ValueError, match="builder created_at"):
        builder.record_retrieval_run(
            stage="hybrid",
            raw_query="secret query",
            candidates=(_candidate(),),
            retrieval_metadata=_metadata(),
            started_at=NOW - timedelta(seconds=1),
            ended_at=NOW,
        )

    assert builder.evidence_runs == ()
    assert builder.evidence_run_payloads == ()


def test_duplicate_candidate_ranks_fail_without_a_partial_run_or_payload() -> None:
    builder = _builder()

    with pytest.raises(ValidationError, match="candidate rank"):
        builder.record_retrieval_run(
            stage="hybrid",
            raw_query="secret query",
            candidates=(
                _candidate(rank=1, source_id="media-1"),
                _candidate(rank=1, source_id="media-2"),
            ),
            retrieval_metadata=_metadata(),
            started_at=NOW,
            ended_at=NOW,
        )

    assert builder.evidence_runs == ()
    assert builder.evidence_run_payloads == ()


def test_candidate_rank_cannot_exceed_requested_top_k_before_id_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _builder()
    allocated_prefixes: list[str] = []
    monkeypatch.setattr(
        builder_module,
        "new_opaque_id",
        lambda prefix: allocated_prefixes.append(prefix) or f"{prefix}_unused",
    )

    with pytest.raises(ValueError, match="requested_top_k"):
        builder.record_retrieval_run(
            stage="hybrid",
            raw_query="secret query",
            candidates=(_candidate(rank=2),),
            retrieval_metadata=_metadata().model_copy(update={"requested_top_k": 1}),
            started_at=NOW,
            ended_at=NOW,
        )

    assert allocated_prefixes == []
    assert builder.evidence_runs == ()
    assert builder.evidence_run_payloads == ()


def test_candidate_source_kind_must_be_declared_before_id_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _builder()
    allocated_prefixes: list[str] = []
    monkeypatch.setattr(
        builder_module,
        "new_opaque_id",
        lambda prefix: allocated_prefixes.append(prefix) or f"{prefix}_unused",
    )
    note_candidate = _candidate().model_copy(
        update={"source_kind": CanonicalSourceKind.NOTES}
    )

    with pytest.raises(ValueError, match="metadata source_kinds"):
        builder.record_retrieval_run(
            stage="hybrid",
            raw_query="secret query",
            candidates=(note_candidate,),
            retrieval_metadata=_metadata(),
            started_at=NOW,
            ended_at=NOW,
        )

    assert allocated_prefixes == []
    assert builder.evidence_runs == ()
    assert builder.evidence_run_payloads == ()


def test_empty_scope_rejects_candidates_before_id_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _builder()
    allocated_prefixes: list[str] = []
    monkeypatch.setattr(
        builder_module,
        "new_opaque_id",
        lambda prefix: allocated_prefixes.append(prefix) or f"{prefix}_unused",
    )

    with pytest.raises(ValueError, match="empty scope"):
        builder.record_retrieval_run(
            stage="hybrid",
            raw_query="secret query",
            candidates=(_candidate(),),
            retrieval_metadata=_metadata().model_copy(update={"scope_state": "empty"}),
            started_at=NOW,
            ended_at=NOW,
        )

    assert allocated_prefixes == []
    assert builder.evidence_runs == ()
    assert builder.evidence_run_payloads == ()


def test_record_prompt_evidence_set_preserves_exact_snapshot_bytes_and_linkage() -> (
    None
):
    builder = _builder()
    codec = CitationFingerprintCodec(SECRET)
    run_id = _record_run(builder)
    first_snapshot = "[S1] MEDIA — Alpha\r\nExact transformed 🧪 e\u0301"
    second_snapshot = "[S2] MEDIA — Beta\nSecond exact block"

    prompt_set_id = builder.record_prompt_evidence_set(
        run_id=run_id,
        evidence=(
            LocalPromptEvidenceCapture(
                candidate_rank=1,
                snapshot_text=first_snapshot,
                transformations=("heading_injected", "marker_injected"),
            ),
            LocalPromptEvidenceCapture(
                candidate_rank=2,
                snapshot_text=second_snapshot,
                transformations=("heading_injected", "marker_injected"),
            ),
        ),
        created_at=NOW,
    )

    prompt_set = builder.prompt_evidence_sets[0]
    assert prompt_set.prompt_set_id == prompt_set_id
    assert prompt_set.prompt_set_ordinal == 1
    assert prompt_set.marker_namespace is MarkerNamespace.CHATBOOK_S_V1
    assert prompt_set.created_at == NOW
    assert [entry.evidence_ordinal for entry in prompt_set.entries] == [1, 2]
    assert [entry.marker_ordinal for entry in prompt_set.entries] == [1, 2]
    assert [entry.run_id for entry in prompt_set.entries] == [run_id, run_id]
    assert all(
        entry.storage_mode is EvidenceStorageMode.EMBEDDED
        for entry in prompt_set.entries
    )

    first_payload, second_payload = builder.evidence_snapshot_payloads
    assert prompt_set.entries[0].snapshot_payload_ref == first_payload.payload_id
    assert prompt_set.entries[1].snapshot_payload_ref == second_payload.payload_id
    assert first_payload.snapshot_text == first_snapshot
    assert first_payload.snapshot_text.encode("utf-8") == first_snapshot.encode("utf-8")
    assert first_payload.storage_mode is EvidenceStorageMode.EMBEDDED
    assert first_payload.title == "Alpha"
    assert first_payload.source_identity == {
        "source_kind": "media_db",
        "source_id": "media-1",
    }
    assert first_payload.locator == {}
    assert first_payload.lineage == {
        "chunk_index": 3,
        "start_char": 10,
        "end_char": 20,
    }
    assert first_payload.transformations == (
        "heading_injected",
        "marker_injected",
    )
    assert first_payload.content_hash == codec.fingerprint(
        CitationFingerprintDomain.EXACT_PAYLOAD,
        "exact-snapshot-v1",
        first_snapshot.encode("utf-8"),
    )
    assert first_payload.comparison_hash == codec.fingerprint(
        CitationFingerprintDomain.EXACT_PAYLOAD,
        "comparison-nfc-lf-v1",
        "[S1] MEDIA — Alpha\nExact transformed 🧪 é".encode("utf-8"),
    )
    assert second_payload.snapshot_text == second_snapshot
    assert first_snapshot not in repr(builder)
    assert first_payload.content_hash not in repr(builder)


def test_prompt_evidence_rejects_unknown_runs_zero_entries_and_duplicate_ranks() -> (
    None
):
    builder = _builder()
    run_id = _record_run(builder)

    with pytest.raises(ValueError, match="must not be empty"):
        builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=(),
            created_at=NOW,
        )
    with pytest.raises(ValueError, match="unknown evidence run"):
        builder.record_prompt_evidence_set(
            run_id="missing-run",
            evidence=(
                LocalPromptEvidenceCapture(
                    candidate_rank=1,
                    snapshot_text="[S1] MEDIA — Alpha\nExact",
                ),
            ),
            created_at=NOW,
        )
    with pytest.raises(ValueError, match="candidate_rank"):
        builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=(
                LocalPromptEvidenceCapture(
                    candidate_rank=1,
                    snapshot_text="[S1] MEDIA — Alpha\nExact",
                ),
                LocalPromptEvidenceCapture(
                    candidate_rank=1,
                    snapshot_text="[S2] MEDIA — Alpha\nDuplicate rank",
                ),
            ),
            created_at=NOW,
        )

    assert builder.prompt_evidence_sets == ()
    assert builder.evidence_snapshot_payloads == ()


@pytest.mark.parametrize(
    ("started_at", "ended_at", "prompt_created_at"),
    [
        (
            NOW,
            NOW + timedelta(seconds=2),
            NOW + timedelta(seconds=1),
        ),
        (
            NOW + timedelta(seconds=2),
            None,
            NOW + timedelta(seconds=1),
        ),
    ],
    ids=("ended-at-boundary", "started-at-boundary"),
)
def test_prompt_set_cannot_precede_linked_run_terminal_boundary_and_is_atomic(
    started_at: datetime,
    ended_at: datetime | None,
    prompt_created_at: datetime,
) -> None:
    builder = _builder()
    run_id = builder.record_retrieval_run(
        stage="hybrid",
        raw_query="secret query",
        candidates=(_candidate(),),
        retrieval_metadata=_metadata(),
        started_at=started_at,
        ended_at=ended_at,
    )

    with pytest.raises(ValueError, match="run terminal boundary"):
        builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=(
                LocalPromptEvidenceCapture(
                    candidate_rank=1,
                    snapshot_text="[S1] MEDIA — Alpha\nExact",
                ),
            ),
            created_at=prompt_created_at,
        )

    assert len(builder.evidence_runs) == 1
    assert len(builder.evidence_run_payloads) == 1
    assert builder.prompt_evidence_sets == ()
    assert builder.evidence_snapshot_payloads == ()


@pytest.mark.parametrize(
    ("evidence", "created_at", "message"),
    [
        (
            (
                LocalPromptEvidenceCapture(
                    candidate_rank=1,
                    snapshot_text="[S9] MEDIA — Alpha\nWrong marker",
                ),
            ),
            NOW,
            "marker ordinal",
        ),
        (
            (
                LocalPromptEvidenceCapture(
                    candidate_rank=1,
                    snapshot_text="[S1] MEDIA — Alpha\nExact",
                ),
            ),
            NOW.replace(tzinfo=None),
            "timezone-aware",
        ),
    ],
)
def test_invalid_prompt_evidence_set_is_atomic(
    evidence: tuple[LocalPromptEvidenceCapture, ...],
    created_at: datetime,
    message: str,
) -> None:
    builder = _builder()
    run_id = _record_run(builder)

    with pytest.raises((ValueError, ValidationError), match=message):
        builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=evidence,
            created_at=created_at,
        )

    assert builder.prompt_evidence_sets == ()
    assert builder.evidence_snapshot_payloads == ()


def test_prompt_evidence_set_enforces_snapshot_entry_and_set_caps_atomically() -> None:
    builder = _builder()
    run_id = _record_run(builder)
    oversized = LocalPromptEvidenceCapture.model_construct(
        candidate_rank=1,
        snapshot_text="[S1] " + ("é" * SNAPSHOT_TEXT_UTF8_BYTES_MAX),
        transformations=(),
    )

    with pytest.raises(ValidationError, match="snapshot_text exceeds"):
        builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=(oversized,),
            created_at=NOW,
        )
    with pytest.raises(ValueError, match="evidence entries"):
        builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=tuple(
                LocalPromptEvidenceCapture(
                    candidate_rank=1,
                    snapshot_text=f"[S{ordinal}] MEDIA — Alpha\nExact",
                )
                for ordinal in range(1, EVIDENCE_ENTRIES_PER_PROMPT_MAX + 2)
            ),
            created_at=NOW,
        )
    assert builder.prompt_evidence_sets == ()
    assert builder.evidence_snapshot_payloads == ()

    single = (
        LocalPromptEvidenceCapture(
            candidate_rank=1,
            snapshot_text="[S1] MEDIA — Alpha\nExact",
        ),
    )
    for _ in range(PROMPT_EVIDENCE_SETS_MAX):
        builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=single,
            created_at=NOW,
        )
    snapshot_count = len(builder.evidence_snapshot_payloads)

    with pytest.raises(ValueError, match="prompt evidence sets"):
        builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=single,
            created_at=NOW,
        )

    assert len(builder.prompt_evidence_sets) == PROMPT_EVIDENCE_SETS_MAX
    assert len(builder.evidence_snapshot_payloads) == snapshot_count


def test_cumulative_run_payload_budget_accepts_exact_limit_then_rejects_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _builder()
    _record_run(builder)
    one_run_bytes = _compact_model_json_bytes(builder.evidence_run_payloads[0])
    monkeypatch.setattr(
        builder_module,
        "GOVERNED_PAYLOAD_UTF8_BYTES_MAX",
        one_run_bytes * 2,
        raising=False,
    )

    _record_run(builder)
    assert len(builder.evidence_runs) == 2
    assert len(builder.evidence_run_payloads) == 2

    with pytest.raises(ValueError, match="governed payload exceeds"):
        _record_run(builder)

    assert len(builder.evidence_runs) == 2
    assert len(builder.evidence_run_payloads) == 2


def test_cumulative_snapshot_batch_overflow_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (
        LocalPromptEvidenceCapture(
            candidate_rank=1,
            snapshot_text="[S1] MEDIA — Alpha\nExact first block",
        ),
        LocalPromptEvidenceCapture(
            candidate_rank=2,
            snapshot_text="[S2] MEDIA — Beta\nExact second block",
        ),
    )
    measuring_builder = _builder()
    measuring_run_id = _record_run(measuring_builder)
    measuring_builder.record_prompt_evidence_set(
        run_id=measuring_run_id,
        evidence=evidence,
        created_at=NOW,
    )
    snapshot_batch_bytes = sum(
        _compact_model_json_bytes(payload)
        for payload in measuring_builder.evidence_snapshot_payloads
    )

    builder = _builder()
    run_id = _record_run(builder)
    existing_run_bytes = sum(
        _compact_model_json_bytes(payload) for payload in builder.evidence_run_payloads
    )
    monkeypatch.setattr(
        builder_module,
        "GOVERNED_PAYLOAD_UTF8_BYTES_MAX",
        existing_run_bytes + snapshot_batch_bytes - 1,
        raising=False,
    )

    with pytest.raises(ValueError, match="governed payload exceeds"):
        builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=evidence,
            created_at=NOW,
        )

    assert builder.prompt_evidence_sets == ()
    assert builder.evidence_snapshot_payloads == ()


def test_record_initial_answer_attempt_retains_governed_body_and_metadata() -> None:
    answer_body = "Marker-free exact answer 🧪."
    builder, attempt_id = _builder_with_initial_answer(answer_body=answer_body)

    assert len(builder.answer_attempts) == 1
    attempt = builder.answer_attempts[0]
    payload = builder.answer_attempt_payloads[0]
    assert attempt.attempt_id == attempt_id
    assert attempt.attempt_ordinal == 1
    assert attempt.kind is AnswerAttemptKind.INITIAL
    assert attempt.prompt_evidence_set_id == builder.prompt_evidence_sets[0].prompt_set_id
    assert attempt.answer_payload_ref == payload.payload_id
    assert attempt.occurrences == ()
    assert attempt.created_at == NOW
    assert payload.attempt_id == attempt_id
    assert payload.answer_body == answer_body
    assert payload.body_integrity_hmac == CitationFingerprintCodec(SECRET).fingerprint(
        CitationFingerprintDomain.MESSAGE_BODY,
        answer_body,
    )
    immutable_json = json.dumps(attempt.model_dump(mode="json"), sort_keys=True)
    assert answer_body not in immutable_json
    assert payload.body_integrity_hmac not in immutable_json


def test_initial_answer_governed_payload_view_is_deep_detached() -> None:
    builder, _attempt_id = _builder_with_initial_answer()

    returned = builder.answer_attempt_payloads[0]
    object.__setattr__(returned, "answer_body", "tampered")

    assert builder.answer_attempt_payloads[0].answer_body == "Marker-free exact answer."


@pytest.mark.parametrize(
    "answer_body",
    (
        "Eligible [S1] marker.",
        "Eligible [S1] and [S2] markers.",
    ),
)
def test_initial_answer_with_eligible_markers_is_unavailable_and_atomic(
    answer_body: str,
) -> None:
    builder = _builder()
    prompt_set_id = _record_prompt_set(builder)

    with pytest.raises(ValueError) as captured:
        builder.record_initial_answer_attempt(
            prompt_evidence_set_id=prompt_set_id,
            answer_body=answer_body,
            completed_at=NOW,
        )

    assert isinstance(
        captured.value,
        builder_module.CitationTraceBuildUnavailable,
    )
    assert captured.value.reason_code == "occurrence_mapping_unavailable"
    assert builder.answer_attempts == ()
    assert builder.answer_attempt_payloads == ()


@pytest.mark.parametrize(
    "answer_body",
    (
        r"Escaped marker \[S1] is literal.",
        "Inline code `[S1]` is literal.",
        "```text\n[S1]\n```\nCode fence only.",
    ),
)
def test_marker_free_eligibility_ignores_markdown_code_and_escaped_literals(
    answer_body: str,
) -> None:
    builder, attempt_id = _builder_with_initial_answer(answer_body=answer_body)

    assert builder.answer_attempts[0].attempt_id == attempt_id
    assert builder.answer_attempts[0].occurrences == ()


def test_answer_body_byte_cap_rejects_initial_attempt_atomically() -> None:
    builder = _builder()
    prompt_set_id = _record_prompt_set(builder)
    sentinel = "ANSWER_BODY_SECRET_SENTINEL"
    oversized = sentinel * (
        (ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX // len(sentinel)) + 1
    )

    with pytest.raises(ValueError, match="answer_body exceeds") as captured:
        builder.record_initial_answer_attempt(
            prompt_evidence_set_id=prompt_set_id,
            answer_body=oversized,
            completed_at=NOW,
        )

    assert "ANSWER_BODY_" not in str(captured.value)
    assert builder.answer_attempts == ()
    assert builder.answer_attempt_payloads == ()


def test_answer_body_aggregate_governed_payload_overflow_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measuring_builder, _attempt_id = _builder_with_initial_answer()
    governed_bytes = sum(
        _compact_model_json_bytes(payload)
        for payload in (
            *measuring_builder.evidence_run_payloads,
            *measuring_builder.evidence_snapshot_payloads,
            *measuring_builder.answer_attempt_payloads,
        )
    )
    builder = _builder()
    prompt_set_id = _record_prompt_set(builder)
    monkeypatch.setattr(
        builder_module,
        "GOVERNED_PAYLOAD_UTF8_BYTES_MAX",
        governed_bytes - 1,
        raising=False,
    )

    with pytest.raises(ValueError, match="governed payload exceeds"):
        builder.record_initial_answer_attempt(
            prompt_evidence_set_id=prompt_set_id,
            answer_body="Marker-free exact answer.",
            completed_at=NOW,
        )

    assert builder.answer_attempts == ()
    assert builder.answer_attempt_payloads == ()


def test_initial_answer_rejects_unknown_prompt_set_atomically() -> None:
    builder = _builder()

    with pytest.raises(ValueError, match="unknown prompt evidence set"):
        builder.record_initial_answer_attempt(
            prompt_evidence_set_id="missing-prompt",
            answer_body="Marker-free exact answer.",
            completed_at=NOW,
        )

    assert builder.answer_attempts == ()
    assert builder.answer_attempt_payloads == ()


def test_attempt_completion_cannot_precede_prompt_creation_and_is_atomic() -> None:
    builder = _builder()
    prompt_set_id = _record_prompt_set(builder, created_at=NOW + timedelta(seconds=1))

    with pytest.raises(ValueError, match="prompt evidence set"):
        builder.record_initial_answer_attempt(
            prompt_evidence_set_id=prompt_set_id,
            answer_body="Marker-free exact answer.",
            completed_at=NOW,
        )

    assert builder.answer_attempts == ()
    assert builder.answer_attempt_payloads == ()


def test_second_initial_answer_attempt_is_rejected_without_repair_or_rerun_api() -> None:
    builder, first_attempt_id = _builder_with_initial_answer()

    with pytest.raises(ValueError, match="initial answer attempt"):
        builder.record_initial_answer_attempt(
            prompt_evidence_set_id=builder.prompt_evidence_sets[0].prompt_set_id,
            answer_body="Another marker-free answer.",
            completed_at=NOW,
        )

    assert [attempt.attempt_id for attempt in builder.answer_attempts] == [
        first_attempt_id
    ]
    assert len(builder.answer_attempt_payloads) == 1
    assert not hasattr(builder, "record_repair_answer_attempt")
    assert not hasattr(builder, "record_pipeline_rerun")


def test_seal_returns_one_shot_local_immutable_write_with_fixed_linkage() -> None:
    answer_body = "Marker-free exact answer."
    builder, attempt_id = _builder_with_initial_answer(answer_body=answer_body)

    sealed_write = builder.seal(
        selected_attempt_id=attempt_id,
        sealed_at=NOW + timedelta(seconds=1),
    )

    trace = sealed_write.trace
    assert builder.is_sealed is True
    assert trace.origin is TraceOrigin.LOCAL
    assert trace.lifecycle is TraceLifecycle.SEALED
    assert trace.completeness_at_seal is CitationCompleteness.COMPLETE
    assert trace.completeness_at_seal is reduce_selected_attempt_completeness(
        trace,
        {
            payload.payload_id: payload
            for payload in sealed_write.evidence_snapshot_payloads
        },
    )
    assert trace.selected_attempt_id == attempt_id
    assert trace.answer_attempts[0].occurrences == ()
    assert trace.policy_version == TEST_POLICY_VERSION
    assert trace.policy_capabilities == TEST_POLICY_CAPABILITIES
    assert sealed_write.evidence_run_payloads == builder.evidence_run_payloads
    assert (
        sealed_write.evidence_snapshot_payloads
        == builder.evidence_snapshot_payloads
    )
    assert sealed_write.answer_attempt_payloads == builder.answer_attempt_payloads
    trace_json = json.dumps(trace.model_dump(mode="json"), sort_keys=True)
    assert answer_body not in trace_json
    assert sealed_write.answer_attempt_payloads[0].body_integrity_hmac not in trace_json
    assert sealed_write.answer_attempt_payloads[0].answer_body == answer_body
    assert (
        sealed_write.answer_attempt_payloads[0].body_integrity_hmac
        == CitationFingerprintCodec(SECRET).fingerprint(
            CitationFingerprintDomain.MESSAGE_BODY,
            answer_body,
        )
    )
    persistence_retry = sealed_write
    assert persistence_retry is sealed_write


def test_seal_requires_complete_graph_and_known_selected_attempt() -> None:
    empty = _builder()
    with pytest.raises(ValueError, match="evidence run"):
        empty.seal(selected_attempt_id="missing", sealed_at=NOW)
    assert empty.is_sealed is False

    run_only = _builder()
    _record_run(run_only)
    with pytest.raises(ValueError, match="prompt evidence set"):
        run_only.seal(selected_attempt_id="missing", sealed_at=NOW)
    assert run_only.is_sealed is False

    prompt_only = _builder()
    _record_prompt_set(prompt_only)
    with pytest.raises(ValueError, match="answer attempt"):
        prompt_only.seal(selected_attempt_id="missing", sealed_at=NOW)
    assert prompt_only.is_sealed is False

    complete, _attempt_id = _builder_with_initial_answer()
    with pytest.raises(ValueError, match="selected answer attempt"):
        complete.seal(selected_attempt_id="missing", sealed_at=NOW)
    assert complete.is_sealed is False


def test_seal_requires_every_retrieval_run_to_have_ended() -> None:
    builder = _builder()
    run_id = builder.record_retrieval_run(
        stage="hybrid",
        raw_query="secret query",
        candidates=(_candidate(),),
        retrieval_metadata=_metadata(),
        started_at=NOW,
        ended_at=None,
    )
    prompt_set_id = _record_prompt_set(builder, run_id=run_id)
    attempt_id = builder.record_initial_answer_attempt(
        prompt_evidence_set_id=prompt_set_id,
        answer_body="Marker-free exact answer.",
        completed_at=NOW,
    )

    with pytest.raises(ValueError, match="ended_at"):
        builder.seal(selected_attempt_id=attempt_id, sealed_at=NOW)

    assert builder.is_sealed is False


def test_seal_revalidates_run_prompt_and_attempt_lifecycle_order() -> None:
    builder, attempt_id = _builder_with_initial_answer()
    prompt = builder.prompt_evidence_sets[0]
    builder._evidence_runs[0] = builder.evidence_runs[0].model_copy(  # type: ignore[attr-defined]
        update={"ended_at": prompt.created_at + timedelta(seconds=1)}
    )

    with pytest.raises(ValueError, match="prompt"):
        builder.seal(
            selected_attempt_id=attempt_id,
            sealed_at=NOW + timedelta(seconds=2),
        )
    assert builder.is_sealed is False

    builder._evidence_runs[0] = builder.evidence_runs[0].model_copy(  # type: ignore[attr-defined]
        update={"ended_at": NOW}
    )
    builder._answer_attempts[0] = builder.answer_attempts[0].model_copy(  # type: ignore[attr-defined]
        update={"created_at": prompt.created_at - timedelta(seconds=1)}
    )
    with pytest.raises(ValueError, match="attempt"):
        builder.seal(
            selected_attempt_id=attempt_id,
            sealed_at=NOW + timedelta(seconds=2),
        )
    assert builder.is_sealed is False


@pytest.mark.parametrize(
    "sealed_at",
    (
        NOW - timedelta(seconds=1),
        NOW + timedelta(milliseconds=500),
        NOW + timedelta(seconds=2, milliseconds=500),
        NOW + timedelta(seconds=3, milliseconds=500),
    ),
)
def test_seal_rejects_sealed_at_before_any_lifecycle_order_boundary(
    sealed_at: datetime,
) -> None:
    builder = _builder()
    run_id = builder.record_retrieval_run(
        stage="hybrid",
        raw_query="secret query",
        candidates=(_candidate(),),
        retrieval_metadata=_metadata(),
        started_at=NOW + timedelta(seconds=1),
        ended_at=NOW + timedelta(seconds=2),
    )
    prompt_set_id = _record_prompt_set(
        builder,
        run_id=run_id,
        created_at=NOW + timedelta(seconds=3),
    )
    attempt_id = builder.record_initial_answer_attempt(
        prompt_evidence_set_id=prompt_set_id,
        answer_body="Marker-free exact answer.",
        completed_at=NOW + timedelta(seconds=4),
    )

    with pytest.raises((ValueError, ValidationError), match="sealed_at"):
        builder.seal(selected_attempt_id=attempt_id, sealed_at=sealed_at)

    assert builder.is_sealed is False


def test_every_mutation_and_second_seal_reject_after_successful_seal() -> None:
    builder, attempt_id = _builder_with_initial_answer()
    builder.seal(selected_attempt_id=attempt_id, sealed_at=NOW)
    run_id = builder.evidence_runs[0].run_id
    prompt_set_id = builder.prompt_evidence_sets[0].prompt_set_id

    mutations = (
        lambda: builder.record_retrieval_run(
            stage="hybrid",
            raw_query="secret query",
            candidates=(_candidate(),),
            retrieval_metadata=_metadata(),
            started_at=NOW,
            ended_at=NOW,
        ),
        lambda: builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=(
                LocalPromptEvidenceCapture(
                    candidate_rank=1,
                    snapshot_text="[S1] MEDIA — Alpha\nExact",
                ),
            ),
            created_at=NOW,
        ),
        lambda: builder.record_initial_answer_attempt(
            prompt_evidence_set_id=prompt_set_id,
            answer_body="Marker-free answer.",
            completed_at=NOW,
        ),
        lambda: builder.seal(selected_attempt_id=attempt_id, sealed_at=NOW),
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="sealed"):
            mutation()
