from __future__ import annotations

from datetime import UTC, datetime
from itertools import permutations

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from tldw_chatbook.Chat.citation_trace_models import (
    ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX,
    ANSWER_ATTEMPTS_MAX,
    CITATION_OCCURRENCES_MAX,
    EVIDENCE_ENTRIES_PER_PROMPT_MAX,
    EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX,
    GOVERNED_PAYLOAD_UTF8_BYTES_MAX,
    IMMUTABLE_AGGREGATE_JSON_BYTES_MAX,
    PROMPT_EVIDENCE_SETS_MAX,
    RETRIEVAL_CANDIDATES_PER_RUN_MAX,
    SNAPSHOT_TEXT_UTF8_BYTES_MAX,
    AnswerAttempt,
    AnswerAttemptKind,
    AnswerAttemptPayload,
    CitationCompleteness,
    CitationOccurrence,
    CitationTrace,
    ClaimSupport,
    EvidenceRun,
    EvidenceRunPayload,
    EvidenceSnapshotPayload,
    EvidenceStorageMode,
    MarkerNamespace,
    OffsetBasis,
    PromptEvidenceEntry,
    PromptEvidenceSet,
    RetrievalCandidatePayload,
    SealedCitationWrite,
    StructuralValidationState,
    TraceLifecycle,
    TraceOrigin,
    reduce_selected_attempt_completeness,
    validate_aggregate_json_bytes,
)


NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def _snapshot_payload(
    ordinal: int,
    mode: EvidenceStorageMode,
    *,
    text: str | None = None,
) -> EvidenceSnapshotPayload:
    payload_id = f"snapshot-payload-{ordinal}"
    if mode is EvidenceStorageMode.EMBEDDED:
        return EvidenceSnapshotPayload(
            payload_id=payload_id,
            storage_mode=mode,
            snapshot_text=text if text is not None else f"snapshot {ordinal}",
            source_identity={"source_id": f"source-{ordinal}"},
            title=f"Title {ordinal}",
            locator={"kind": "legacy", "value": f"opaque-{ordinal}"},
            lineage={"chunk_id": f"chunk-{ordinal}"},
            content_hash=f"governed-hash-{ordinal}",
            comparison_hash=f"governed-comparison-{ordinal}",
        )
    if mode is EvidenceStorageMode.SERVER_REFERENCE:
        return EvidenceSnapshotPayload(
            payload_id=payload_id,
            storage_mode=mode,
            server_reference=f"server-payload-{ordinal}",
            source_identity={"source_id": f"source-{ordinal}"},
        )
    if mode is EvidenceStorageMode.EPHEMERAL:
        return EvidenceSnapshotPayload(
            payload_id=payload_id,
            storage_mode=mode,
            snapshot_text=text if text is not None else f"ephemeral {ordinal}",
        )
    return EvidenceSnapshotPayload(
        payload_id=payload_id,
        storage_mode=mode,
    )


def _write_for_modes(
    modes: tuple[EvidenceStorageMode, ...],
    *,
    completeness: CitationCompleteness | None = None,
    answer_body: str = "",
    occurrences: tuple[CitationOccurrence, ...] = (),
    prompt_set_id: str = "prompt-1",
    attempt_id: str = "attempt-1",
) -> SealedCitationWrite:
    entries = tuple(
        PromptEvidenceEntry(
            evidence_ordinal=index,
            marker_ordinal=index,
            run_id="run-1",
            snapshot_payload_ref=f"snapshot-payload-{index}",
            storage_mode=mode,
        )
        for index, mode in enumerate(modes, start=1)
    )
    snapshots = tuple(
        _snapshot_payload(index, mode) for index, mode in enumerate(modes, start=1)
    )
    prompt_set = PromptEvidenceSet(
        prompt_set_id=prompt_set_id,
        prompt_set_ordinal=1,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        entries=entries,
        created_at=NOW,
    )
    attempt = AnswerAttempt(
        attempt_id=attempt_id,
        attempt_ordinal=1,
        kind=AnswerAttemptKind.INITIAL,
        prompt_evidence_set_id=prompt_set_id,
        answer_payload_ref="answer-payload-1",
        occurrences=occurrences,
        created_at=NOW,
    )
    provisional_trace = CitationTrace(
        trace_id="trace-1",
        request_id="request-1",
        generation_id="generation-1",
        origin=TraceOrigin.LOCAL,
        lifecycle=TraceLifecycle.SEALED,
        completeness_at_seal=completeness or CitationCompleteness.COMPLETE,
        evidence_runs=(
            EvidenceRun(
                run_id="run-1",
                request_id="request-1",
                run_ordinal=1,
                stage="initial",
                payload_ref="run-payload-1",
                started_at=NOW,
            ),
        ),
        prompt_evidence_sets=(prompt_set,),
        answer_attempts=(attempt,),
        selected_attempt_id=attempt_id,
        policy_version="citation-policy-v1",
        created_at=NOW,
        sealed_at=NOW,
    )
    payload_index = {payload.payload_id: payload for payload in snapshots}
    expected = reduce_selected_attempt_completeness(provisional_trace, payload_index)
    trace = provisional_trace.model_copy(
        update={"completeness_at_seal": completeness or expected}
    )
    return SealedCitationWrite(
        trace=trace,
        evidence_run_payloads=(
            EvidenceRunPayload(payload_id="run-payload-1", run_id="run-1"),
        ),
        evidence_snapshot_payloads=snapshots,
        answer_attempt_payloads=(
            AnswerAttemptPayload(
                payload_id="answer-payload-1",
                attempt_id=attempt_id,
                answer_body=answer_body,
            ),
        ),
    )


def test_models_are_strict_frozen_versioned_and_round_trip_deterministically() -> None:
    write = _write_for_modes((EvidenceStorageMode.EMBEDDED,))
    trace = write.trace
    encoded = trace.model_dump_json()

    assert trace.schema_version == 1
    assert CitationTrace.model_validate_json(encoded) == trace
    assert CitationTrace.model_validate_json(encoded).model_dump_json() == encoded

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CitationTrace(**{**trace.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError, match="Input should be 1"):
        CitationTrace(**{**trace.model_dump(), "schema_version": 2})
    with pytest.raises(ValidationError, match="frozen"):
        trace.trace_id = "changed"  # type: ignore[misc]


def test_trace_requires_selected_attempt_and_resolves_every_cross_reference() -> None:
    trace = _write_for_modes((EvidenceStorageMode.EMBEDDED,)).trace

    with pytest.raises(ValidationError, match="selected_attempt_id"):
        CitationTrace(**{**trace.model_dump(), "selected_attempt_id": "missing"})
    with pytest.raises(ValidationError, match="prompt evidence set"):
        CitationTrace(
            **{
                **trace.model_dump(),
                "answer_attempts": (
                    trace.answer_attempts[0].model_copy(
                        update={"prompt_evidence_set_id": "missing"}
                    ),
                ),
            }
        )
    with pytest.raises(ValidationError, match="request_id"):
        CitationTrace(
            **{
                **trace.model_dump(),
                "evidence_runs": (
                    trace.evidence_runs[0].model_copy(
                        update={"request_id": "another-request"}
                    ),
                ),
            }
        )
    with pytest.raises(ValidationError, match="evidence run"):
        CitationTrace(
            **{
                **trace.model_dump(),
                "prompt_evidence_sets": (
                    trace.prompt_evidence_sets[0].model_copy(
                        update={
                            "entries": (
                                trace.prompt_evidence_sets[0]
                                .entries[0]
                                .model_copy(update={"run_id": "missing"}),
                            )
                        }
                    ),
                ),
            }
        )


@pytest.mark.parametrize("field", ["evidence_ordinal", "marker_ordinal"])
def test_prompt_ordinals_are_positive_and_unique(field: str) -> None:
    entry = (
        _write_for_modes((EvidenceStorageMode.EMBEDDED,))
        .trace.prompt_evidence_sets[0]
        .entries[0]
    )

    with pytest.raises(ValidationError):
        PromptEvidenceEntry(**{**entry.model_dump(), field: 0})
    duplicate_update = {
        "snapshot_payload_ref": "other",
        ("marker_ordinal" if field == "evidence_ordinal" else "evidence_ordinal"): 2,
    }
    with pytest.raises(ValidationError, match=field):
        PromptEvidenceSet(
            prompt_set_id="prompt",
            prompt_set_ordinal=1,
            marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
            entries=(entry, entry.model_copy(update=duplicate_update)),
            created_at=NOW,
        )


def test_unicode_repeated_grouped_reordered_and_unknown_markers_round_trip() -> None:
    answer = "😀 Alpha [S2][S1]. Again [S2] and unknown [S9]."
    markers = ("[S2]", "[S1]", "[S2]", "[S9]")
    starts: list[int] = []
    cursor = 0
    for marker in markers:
        start = answer.index(marker, cursor)
        starts.append(start)
        cursor = start + len(marker)
    grouped_claim = (2, answer.index(".") + 1)
    occurrences = tuple(
        CitationOccurrence(
            occurrence_id=f"occurrence-{index}",
            occurrence_ordinal=index,
            raw_marker=marker,
            marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
            evidence_ordinal=evidence_ordinal,
            marker_start=marker_start,
            marker_end=marker_start + len(marker),
            claim_start=grouped_claim[0] if index in (1, 2) else None,
            claim_end=grouped_claim[1] if index in (1, 2) else None,
            offset_basis=OffsetBasis.UNICODE_CODEPOINT_V1,
            structural_state=(
                StructuralValidationState.UNKNOWN_MARKER
                if evidence_ordinal is None
                else StructuralValidationState.VALID
            ),
            claim_support=ClaimSupport.NOT_CHECKED,
        )
        for index, (marker, evidence_ordinal, marker_start) in enumerate(
            zip(markers, (2, 1, 2, None), starts, strict=True),
            start=1,
        )
    )

    write = _write_for_modes(
        (EvidenceStorageMode.EMBEDDED, EvidenceStorageMode.EMBEDDED),
        answer_body=answer,
        occurrences=occurrences,
    )
    restored = SealedCitationWrite.model_validate_json(write.model_dump_json())

    assert restored == write
    assert restored.trace.answer_attempts[0].occurrences[0].marker_start == 8
    assert restored.trace.answer_attempts[0].occurrences[0].evidence_ordinal == 2
    assert restored.trace.answer_attempts[0].occurrences[2].evidence_ordinal == 2
    assert restored.trace.answer_attempts[0].occurrences[0].claim_start == (
        restored.trace.answer_attempts[0].occurrences[1].claim_start
    )
    assert restored.trace.answer_attempts[0].occurrences[-1].evidence_ordinal is None


def test_marker_grammar_and_exact_answer_offsets_are_enforced() -> None:
    with pytest.raises(ValidationError, match="chatbook_s_v1 marker"):
        CitationOccurrence(
            occurrence_id="occurrence",
            occurrence_ordinal=1,
            raw_marker="[S01]",
            marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
            evidence_ordinal=1,
            marker_start=0,
            marker_end=5,
            structural_state=StructuralValidationState.VALID,
        )

    occurrence = CitationOccurrence(
        occurrence_id="occurrence",
        occurrence_ordinal=1,
        raw_marker="[S1]",
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        evidence_ordinal=1,
        marker_start=1,
        marker_end=5,
        structural_state=StructuralValidationState.VALID,
    )
    with pytest.raises(ValidationError, match="answer offsets"):
        _write_for_modes(
            (EvidenceStorageMode.EMBEDDED,),
            answer_body="[S1]",
            occurrences=(occurrence,),
        )

    valid_occurrence = occurrence.model_copy(
        update={"marker_start": 0, "marker_end": 4}
    )
    with pytest.raises(ValidationError, match="marker span"):
        AnswerAttempt(
            attempt_id="attempt",
            attempt_ordinal=1,
            kind=AnswerAttemptKind.INITIAL,
            prompt_evidence_set_id="prompt",
            answer_payload_ref="answer",
            occurrences=(
                valid_occurrence,
                valid_occurrence.model_copy(
                    update={
                        "occurrence_id": "occurrence-2",
                        "occurrence_ordinal": 2,
                    }
                ),
            ),
            created_at=NOW,
        )


def test_selected_attempt_only_completeness_uses_worst_state_precedence() -> None:
    cases = (
        ((EvidenceStorageMode.EMBEDDED,), CitationCompleteness.COMPLETE),
        (
            (EvidenceStorageMode.EMBEDDED, EvidenceStorageMode.EPHEMERAL),
            CitationCompleteness.PARTIAL,
        ),
        (
            (EvidenceStorageMode.EPHEMERAL, EvidenceStorageMode.REDACTED),
            CitationCompleteness.REDACTED,
        ),
    )
    for modes, expected in cases:
        write = _write_for_modes(modes)
        index = {
            payload.payload_id: payload for payload in write.evidence_snapshot_payloads
        }
        assert reduce_selected_attempt_completeness(write.trace, index) is expected

    selected = _write_for_modes((EvidenceStorageMode.EMBEDDED,))
    diagnostic = _write_for_modes(
        (EvidenceStorageMode.REDACTED,),
        prompt_set_id="prompt-diagnostic",
        attempt_id="attempt-diagnostic",
    )
    diagnostic_prompt = diagnostic.trace.prompt_evidence_sets[0].model_copy(
        update={
            "prompt_set_ordinal": 2,
            "entries": (
                diagnostic.trace.prompt_evidence_sets[0]
                .entries[0]
                .model_copy(update={"snapshot_payload_ref": "diagnostic-snapshot"}),
            ),
        }
    )
    diagnostic_attempt = diagnostic.trace.answer_attempts[0].model_copy(
        update={"attempt_ordinal": 2}
    )
    diagnostic_snapshot = diagnostic.evidence_snapshot_payloads[0].model_copy(
        update={"payload_id": "diagnostic-snapshot"}
    )
    trace = CitationTrace(
        **{
            **selected.trace.model_dump(),
            "prompt_evidence_sets": (
                diagnostic_prompt,
                selected.trace.prompt_evidence_sets[0],
            ),
            "answer_attempts": (
                diagnostic_attempt,
                selected.trace.answer_attempts[0],
            ),
        }
    )
    index = {
        payload.payload_id: payload
        for payload in (
            *selected.evidence_snapshot_payloads,
            diagnostic_snapshot,
        )
    }
    assert reduce_selected_attempt_completeness(trace, index) is (
        CitationCompleteness.COMPLETE
    )
    assert (
        reduce_selected_attempt_completeness(
            trace,
            {key: value for key, value in index.items() if key != "snapshot-payload-1"},
        )
        is CitationCompleteness.UNAVAILABLE
    )


@given(st.sampled_from(tuple(permutations(EvidenceStorageMode, 4))))
def test_completeness_reduction_is_stable_under_entry_permutations(
    modes: tuple[EvidenceStorageMode, ...],
) -> None:
    write = _write_for_modes(modes)
    index = {
        payload.payload_id: payload for payload in write.evidence_snapshot_payloads
    }
    assert reduce_selected_attempt_completeness(write.trace, index) is (
        CitationCompleteness.REDACTED
    )
    assert SealedCitationWrite.model_validate_json(write.model_dump_json()) == write


@given(st.permutations(("selected", "partial", "redacted")))
def test_non_final_attempt_and_prompt_order_cannot_change_selected_completeness(
    order: tuple[str, ...],
) -> None:
    selected = _write_for_modes((EvidenceStorageMode.EMBEDDED,))
    selected_prompt = selected.trace.prompt_evidence_sets[0]
    selected_attempt = selected.trace.answer_attempts[0]
    prompts = {
        "selected": selected_prompt,
        "partial": selected_prompt.model_copy(
            update={
                "prompt_set_id": "prompt-partial",
                "prompt_set_ordinal": 2,
                "entries": (
                    selected_prompt.entries[0].model_copy(
                        update={
                            "snapshot_payload_ref": "snapshot-partial",
                            "storage_mode": EvidenceStorageMode.EPHEMERAL,
                        }
                    ),
                ),
            }
        ),
        "redacted": selected_prompt.model_copy(
            update={
                "prompt_set_id": "prompt-redacted",
                "prompt_set_ordinal": 3,
                "entries": (
                    selected_prompt.entries[0].model_copy(
                        update={
                            "snapshot_payload_ref": "snapshot-redacted",
                            "storage_mode": EvidenceStorageMode.REDACTED,
                        }
                    ),
                ),
            }
        ),
    }
    attempts = {
        "selected": selected_attempt,
        "partial": selected_attempt.model_copy(
            update={
                "attempt_id": "attempt-partial",
                "attempt_ordinal": 2,
                "prompt_evidence_set_id": "prompt-partial",
                "answer_payload_ref": None,
            }
        ),
        "redacted": selected_attempt.model_copy(
            update={
                "attempt_id": "attempt-redacted",
                "attempt_ordinal": 3,
                "prompt_evidence_set_id": "prompt-redacted",
                "answer_payload_ref": None,
            }
        ),
    }
    trace = CitationTrace(
        **{
            **selected.trace.model_dump(),
            "prompt_evidence_sets": tuple(prompts[name] for name in order),
            "answer_attempts": tuple(attempts[name] for name in reversed(order)),
        }
    )
    payload_index = {
        selected.evidence_snapshot_payloads[0].payload_id: (
            selected.evidence_snapshot_payloads[0]
        )
    }

    assert (
        reduce_selected_attempt_completeness(trace, payload_index)
        is CitationCompleteness.COMPLETE
    )
    assert CitationTrace.model_validate_json(trace.model_dump_json()) == trace


@given(
    st.lists(
        st.one_of(st.just(1), st.none()),
        min_size=1,
        max_size=32,
    )
)
def test_repeated_and_unknown_occurrence_mappings_round_trip_stably(
    evidence_ordinals: list[int | None],
) -> None:
    markers = ["[S1]" if ordinal == 1 else "[S9]" for ordinal in evidence_ordinals]
    answer = "".join(markers)
    occurrences = tuple(
        CitationOccurrence(
            occurrence_id=f"occurrence-{index}",
            occurrence_ordinal=index,
            raw_marker=marker,
            marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
            evidence_ordinal=evidence_ordinal,
            marker_start=(index - 1) * 4,
            marker_end=index * 4,
            structural_state=(
                StructuralValidationState.VALID
                if evidence_ordinal is not None
                else StructuralValidationState.UNKNOWN_MARKER
            ),
        )
        for index, (marker, evidence_ordinal) in enumerate(
            zip(markers, evidence_ordinals, strict=True),
            start=1,
        )
    )
    write = _write_for_modes(
        (EvidenceStorageMode.EMBEDDED,),
        answer_body=answer,
        occurrences=occurrences,
    )

    assert SealedCitationWrite.model_validate_json(write.model_dump_json()) == write


@given(st.binary(min_size=1, max_size=32).map(bytes.hex))
def test_aggregate_serialization_never_contains_governed_fields_or_values(
    governed_text: str,
) -> None:
    secret_value = f"<<<governed:{governed_text}:value>>>"
    write = _write_for_modes((EvidenceStorageMode.EMBEDDED,))
    governed = write.evidence_snapshot_payloads[0].model_copy(
        update={
            "snapshot_text": secret_value,
            "title": f"title-{secret_value}",
            "source_identity": {"secret-source": secret_value},
            "locator": {"secret-locator": secret_value},
            "lineage": {"secret-lineage": secret_value},
            "content_hash": f"secret-hash-{secret_value}",
        }
    )
    write = SealedCitationWrite(
        trace=write.trace,
        evidence_run_payloads=write.evidence_run_payloads,
        evidence_snapshot_payloads=(governed,),
        answer_attempt_payloads=write.answer_attempt_payloads,
    )
    aggregate = write.trace.model_dump_json()

    for forbidden in (
        "snapshot_text",
        "source_identity",
        "title",
        "locator",
        "lineage",
        "content_hash",
        "comparison_hash",
        "transformations",
        "answer_body",
        "raw_query",
    ):
        assert forbidden not in aggregate
    assert secret_value not in aggregate


def test_exact_and_over_aggregate_snapshot_and_governed_payload_byte_bounds() -> None:
    exact_json = b'"' + (b"x" * (IMMUTABLE_AGGREGATE_JSON_BYTES_MAX - 2)) + b'"'
    assert validate_aggregate_json_bytes(exact_json) == len(exact_json)
    with pytest.raises(ValueError, match="aggregate"):
        validate_aggregate_json_bytes(exact_json + b" ")

    exact_snapshot = "é" * (SNAPSHOT_TEXT_UTF8_BYTES_MAX // 2)
    assert len(exact_snapshot.encode("utf-8")) == SNAPSHOT_TEXT_UTF8_BYTES_MAX
    EvidenceSnapshotPayload(
        payload_id="snapshot",
        storage_mode=EvidenceStorageMode.EMBEDDED,
        snapshot_text=exact_snapshot,
    )
    with pytest.raises(ValidationError, match="snapshot_text"):
        EvidenceSnapshotPayload(
            payload_id="snapshot",
            storage_mode=EvidenceStorageMode.EMBEDDED,
            snapshot_text=exact_snapshot + "x",
        )

    modes = (EvidenceStorageMode.EMBEDDED,) * EVIDENCE_ENTRIES_PER_PROMPT_MAX
    write = _write_for_modes(modes)
    empty_snapshots = tuple(
        payload.model_copy(update={"snapshot_text": ""})
        for payload in write.evidence_snapshot_payloads
    )
    base_write = SealedCitationWrite(
        trace=write.trace,
        evidence_run_payloads=write.evidence_run_payloads,
        evidence_snapshot_payloads=empty_snapshots,
        answer_attempt_payloads=write.answer_attempt_payloads,
    )
    remaining = GOVERNED_PAYLOAD_UTF8_BYTES_MAX - base_write.governed_payload_bytes
    sizes: list[int] = []
    for _ in empty_snapshots:
        size = min(remaining, SNAPSHOT_TEXT_UTF8_BYTES_MAX)
        sizes.append(size)
        remaining -= size
    assert remaining == 0
    snapshots = tuple(
        payload.model_copy(update={"snapshot_text": "x" * size})
        for payload, size in zip(empty_snapshots, sizes, strict=True)
    )
    exact_write = SealedCitationWrite(
        trace=write.trace,
        evidence_run_payloads=write.evidence_run_payloads,
        evidence_snapshot_payloads=snapshots,
        answer_attempt_payloads=write.answer_attempt_payloads,
    )
    assert exact_write.governed_payload_bytes == GOVERNED_PAYLOAD_UTF8_BYTES_MAX
    with pytest.raises(ValidationError, match="governed payload"):
        SealedCitationWrite(
            trace=write.trace,
            evidence_run_payloads=write.evidence_run_payloads,
            evidence_snapshot_payloads=snapshots,
            answer_attempt_payloads=(
                write.answer_attempt_payloads[0].model_copy(
                    update={"answer_body": "x"}
                ),
            ),
        )


def test_exact_and_over_count_bounds() -> None:
    one = _write_for_modes((EvidenceStorageMode.EMBEDDED,))
    base_prompt = one.trace.prompt_evidence_sets[0]
    base_attempt = one.trace.answer_attempts[0]

    prompts = tuple(
        base_prompt.model_copy(
            update={"prompt_set_id": f"prompt-{index}", "prompt_set_ordinal": index}
        )
        for index in range(1, PROMPT_EVIDENCE_SETS_MAX + 1)
    )
    attempts = tuple(
        base_attempt.model_copy(
            update={
                "attempt_id": f"attempt-{index}",
                "attempt_ordinal": index,
                "prompt_evidence_set_id": f"prompt-{index}",
                "answer_payload_ref": None,
            }
        )
        for index in range(1, ANSWER_ATTEMPTS_MAX + 1)
    )
    CitationTrace(
        **{
            **one.trace.model_dump(),
            "prompt_evidence_sets": prompts,
            "answer_attempts": attempts,
            "selected_attempt_id": "attempt-1",
        }
    )
    with pytest.raises(ValidationError):
        CitationTrace(
            **{
                **one.trace.model_dump(),
                "prompt_evidence_sets": prompts
                + (
                    base_prompt.model_copy(
                        update={
                            "prompt_set_id": "prompt-over",
                            "prompt_set_ordinal": PROMPT_EVIDENCE_SETS_MAX + 1,
                        }
                    ),
                ),
            }
        )
    with pytest.raises(ValidationError):
        CitationTrace(
            **{
                **one.trace.model_dump(),
                "answer_attempts": attempts
                + (
                    base_attempt.model_copy(
                        update={
                            "attempt_id": "attempt-over",
                            "attempt_ordinal": ANSWER_ATTEMPTS_MAX + 1,
                        }
                    ),
                ),
            }
        )

    entries = tuple(
        base_prompt.entries[0].model_copy(
            update={
                "evidence_ordinal": index,
                "marker_ordinal": index,
                "snapshot_payload_ref": f"snapshot-{index}",
            }
        )
        for index in range(1, EVIDENCE_ENTRIES_PER_PROMPT_MAX + 1)
    )
    PromptEvidenceSet(**{**base_prompt.model_dump(), "entries": entries})
    with pytest.raises(ValidationError):
        PromptEvidenceSet(
            **{
                **base_prompt.model_dump(),
                "entries": entries
                + (
                    entries[0].model_copy(
                        update={
                            "evidence_ordinal": EVIDENCE_ENTRIES_PER_PROMPT_MAX + 1,
                            "marker_ordinal": EVIDENCE_ENTRIES_PER_PROMPT_MAX + 1,
                            "snapshot_payload_ref": "snapshot-over",
                        }
                    ),
                ),
            }
        )


def test_exact_and_over_occurrence_candidate_external_id_and_answer_body_bounds() -> (
    None
):
    occurrences = tuple(
        CitationOccurrence(
            occurrence_id=f"occurrence-{index}",
            occurrence_ordinal=index,
            raw_marker="[S1]",
            marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
            evidence_ordinal=1,
            marker_start=(index - 1) * 4,
            marker_end=index * 4,
            structural_state=StructuralValidationState.VALID,
        )
        for index in range(1, CITATION_OCCURRENCES_MAX + 1)
    )
    AnswerAttempt(
        attempt_id="attempt",
        attempt_ordinal=1,
        kind=AnswerAttemptKind.INITIAL,
        prompt_evidence_set_id="prompt",
        answer_payload_ref="answer",
        occurrences=occurrences,
        created_at=NOW,
    )
    with pytest.raises(ValidationError):
        AnswerAttempt(
            attempt_id="attempt",
            attempt_ordinal=1,
            kind=AnswerAttemptKind.INITIAL,
            prompt_evidence_set_id="prompt",
            answer_payload_ref="answer",
            occurrences=occurrences
            + (
                occurrences[-1].model_copy(
                    update={
                        "occurrence_id": "occurrence-over",
                        "occurrence_ordinal": CITATION_OCCURRENCES_MAX + 1,
                    }
                ),
            ),
            created_at=NOW,
        )

    candidates = tuple(
        RetrievalCandidatePayload(candidate_id=f"candidate-{index}", rank=index)
        for index in range(1, RETRIEVAL_CANDIDATES_PER_RUN_MAX + 1)
    )
    EvidenceRunPayload(payload_id="run-payload", run_id="run", candidates=candidates)
    with pytest.raises(ValidationError):
        EvidenceRunPayload(
            payload_id="run-payload",
            run_id="run",
            candidates=candidates
            + (
                RetrievalCandidatePayload(
                    candidate_id="candidate-over",
                    rank=RETRIEVAL_CANDIDATES_PER_RUN_MAX + 1,
                ),
            ),
        )

    exact_external_id = "é" * (EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX // 2)
    EvidenceRun(
        run_id=exact_external_id,
        request_id="request",
        run_ordinal=1,
        stage="initial",
        payload_ref="payload",
        started_at=NOW,
    )
    with pytest.raises(ValidationError, match="UTF-8 bytes"):
        EvidenceRun(
            run_id=exact_external_id + "x",
            request_id="request",
            run_ordinal=1,
            stage="initial",
            payload_ref="payload",
            started_at=NOW,
        )

    exact_body = "x" * ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX
    AnswerAttemptPayload(
        payload_id="answer",
        attempt_id="attempt",
        answer_body=exact_body,
    )
    with pytest.raises(ValidationError, match="answer_body"):
        AnswerAttemptPayload(
            payload_id="answer",
            attempt_id="attempt",
            answer_body=exact_body + "x",
        )


def test_sealed_write_requires_complete_non_extraneous_payload_graph() -> None:
    write = _write_for_modes((EvidenceStorageMode.EMBEDDED,))

    with pytest.raises(ValidationError, match="missing governed snapshot payload"):
        SealedCitationWrite(
            trace=write.trace,
            evidence_run_payloads=write.evidence_run_payloads,
            evidence_snapshot_payloads=(),
            answer_attempt_payloads=write.answer_attempt_payloads,
        )
    with pytest.raises(ValidationError, match="extraneous governed snapshot payload"):
        SealedCitationWrite(
            trace=write.trace,
            evidence_run_payloads=write.evidence_run_payloads,
            evidence_snapshot_payloads=write.evidence_snapshot_payloads
            + (
                EvidenceSnapshotPayload(
                    payload_id="snapshot-extra",
                    storage_mode=EvidenceStorageMode.REDACTED,
                ),
            ),
            answer_attempt_payloads=write.answer_attempt_payloads,
        )
    with pytest.raises(ValidationError, match="duplicate governed snapshot payload"):
        SealedCitationWrite(
            trace=write.trace,
            evidence_run_payloads=write.evidence_run_payloads,
            evidence_snapshot_payloads=write.evidence_snapshot_payloads * 2,
            answer_attempt_payloads=write.answer_attempt_payloads,
        )
    with pytest.raises(ValidationError, match="completeness_at_seal"):
        SealedCitationWrite(
            trace=write.trace.model_copy(
                update={"completeness_at_seal": CitationCompleteness.PARTIAL}
            ),
            evidence_run_payloads=write.evidence_run_payloads,
            evidence_snapshot_payloads=write.evidence_snapshot_payloads,
            answer_attempt_payloads=write.answer_attempt_payloads,
        )
