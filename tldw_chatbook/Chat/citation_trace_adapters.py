"""Pure compatibility adapters for legacy evidence and citation contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from .citation_evidence_models import CitationRef, EvidenceBundle, EvidenceReference
from .citation_trace_identity import new_opaque_id
from .citation_trace_models import (
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
    MarkerNamespace,
    OffsetBasis,
    PromptEvidenceEntry,
    PromptEvidenceSet,
    RetrievalCandidatePayload,
    SealedCitationWrite,
    StructuralTrustSummary,
    StructuralValidationState,
    TraceLifecycle,
    TraceOrigin,
)


_LEGACY_MARKER = re.compile(r"\[(S)?([1-9][0-9]*)\]")
_UNKNOWN_AUTHORITIES = frozenset({"unknown", "unavailable", "none", "n/a"})


def synthesize_legacy_evidence_bundle(
    bundle: EvidenceBundle,
    *,
    answer_body: str,
    citations: Sequence[CitationRef] = (),
    created_at: datetime,
) -> SealedCitationWrite:
    """Synthesize honest legacy provenance from an existing evidence bundle.

    The adapter never claims complete provenance because the legacy contract
    does not prove the exact prompt boundary or authoritative storage policy.
    """

    if not isinstance(bundle, EvidenceBundle):
        raise TypeError("bundle must be EvidenceBundle")
    if any(not isinstance(citation, CitationRef) for citation in citations):
        raise TypeError("citations must contain CitationRef values")

    authority_known = bool(bundle.references) and all(
        reference.authority_label.strip().lower() not in _UNKNOWN_AUTHORITIES
        for reference in bundle.references
    )
    return _synthesize(
        bundle=bundle,
        citations=tuple(citations),
        answer_body=answer_body,
        created_at=created_at,
        prompt_boundary_available=authority_known,
    )


def synthesize_legacy_citation_refs(
    citations: Sequence[CitationRef],
    *,
    answer_body: str,
    created_at: datetime,
) -> SealedCitationWrite:
    """Synthesize unavailable provenance from citations lacking prompt evidence."""

    if any(not isinstance(citation, CitationRef) for citation in citations):
        raise TypeError("citations must contain CitationRef values")
    return _synthesize(
        bundle=None,
        citations=tuple(citations),
        answer_body=answer_body,
        created_at=created_at,
        prompt_boundary_available=False,
    )


def synthesize_legacy_payloads(
    bundle_payload: Mapping[str, Any],
    citation_payloads: Sequence[Mapping[str, Any]],
    *,
    answer_body: str,
    created_at: datetime,
) -> SealedCitationWrite:
    """Validate legacy serialized payloads before synthesizing provenance."""

    if not isinstance(bundle_payload, Mapping):
        raise TypeError("bundle_payload must be a mapping")
    references = bundle_payload.get("references", ())
    if not isinstance(references, (list, tuple)):
        raise ValueError("legacy bundle references must be a list")
    if not isinstance(citation_payloads, (list, tuple)):
        raise TypeError("citation_payloads must be a sequence")
    if any(not isinstance(payload, Mapping) for payload in citation_payloads):
        raise ValueError("legacy citation payload must be a mapping")

    bundle = EvidenceBundle.from_payload(bundle_payload)
    citations = tuple(
        CitationRef.from_payload(payload) for payload in citation_payloads
    )
    return synthesize_legacy_evidence_bundle(
        bundle,
        answer_body=answer_body,
        citations=citations,
        created_at=created_at,
    )


def _synthesize(
    *,
    bundle: EvidenceBundle | None,
    citations: tuple[CitationRef, ...],
    answer_body: str,
    created_at: datetime,
    prompt_boundary_available: bool,
) -> SealedCitationWrite:
    trace_id = new_opaque_id("legacy-trace")
    request_id = new_opaque_id("legacy-request")
    generation_id = new_opaque_id("legacy-generation")
    run_id = new_opaque_id("legacy-run")
    run_payload_id = new_opaque_id("run-payload")
    prompt_set_id = new_opaque_id("legacy-prompt")
    attempt_id = new_opaque_id("legacy-attempt")
    answer_payload_id = new_opaque_id("answer-payload")

    references = bundle.references if bundle is not None else ()
    namespace = _marker_namespace(answer_body)
    marker_ordinals = _reference_marker_ordinals(references)
    entries: tuple[PromptEvidenceEntry, ...] = ()
    snapshots: tuple[EvidenceSnapshotPayload, ...] = ()
    if prompt_boundary_available:
        entries, snapshots = _legacy_prompt_payloads(
            references,
            marker_ordinals,
            run_id,
        )

    prompt_set = PromptEvidenceSet(
        prompt_set_id=prompt_set_id,
        prompt_set_ordinal=1,
        marker_namespace=namespace,
        entries=entries,
        created_at=created_at,
    )
    evidence_by_marker = {
        entry.marker_ordinal: entry.evidence_ordinal for entry in entries
    }
    occurrences = _legacy_occurrences(
        answer_body,
        namespace,
        evidence_by_marker,
    )
    attempt = AnswerAttempt(
        attempt_id=attempt_id,
        attempt_ordinal=1,
        kind=AnswerAttemptKind.LEGACY_INFERRED,
        prompt_evidence_set_id=prompt_set_id,
        answer_payload_ref=answer_payload_id,
        occurrences=occurrences,
        structural_summary=StructuralTrustSummary(
            valid_occurrences=sum(
                occurrence.evidence_ordinal is not None for occurrence in occurrences
            ),
            unknown_occurrences=sum(
                occurrence.evidence_ordinal is None for occurrence in occurrences
            ),
        ),
        created_at=created_at,
    )
    completeness = (
        CitationCompleteness.PARTIAL
        if prompt_boundary_available and entries
        else CitationCompleteness.UNAVAILABLE
    )
    trace = CitationTrace(
        trace_id=trace_id,
        request_id=request_id,
        generation_id=generation_id,
        origin=TraceOrigin.LEGACY_INFERRED,
        lifecycle=TraceLifecycle.SEALED,
        completeness_at_seal=completeness,
        evidence_runs=(
            EvidenceRun(
                run_id=run_id,
                request_id=request_id,
                run_ordinal=1,
                stage="legacy_inferred",
                payload_ref=run_payload_id,
                started_at=created_at,
                ended_at=created_at,
            ),
        ),
        prompt_evidence_sets=(prompt_set,),
        answer_attempts=(attempt,),
        selected_attempt_id=attempt_id,
        policy_version="legacy-inference-v1",
        created_at=created_at,
        sealed_at=created_at,
    )
    run_payload = EvidenceRunPayload(
        payload_id=run_payload_id,
        run_id=run_id,
        raw_query=bundle.query if bundle is not None else None,
        retrieval_metadata={
            "legacy_bundle_id": bundle.bundle_id if bundle is not None else None,
            "legacy_citation_count": len(citations),
        },
        candidates=(
            tuple(
                _legacy_candidate(reference, rank)
                for rank, reference in enumerate(references, start=1)
            )
            if references
            else tuple(
                _legacy_citation_candidate(citation, rank)
                for rank, citation in enumerate(citations, start=1)
            )
        ),
    )
    return SealedCitationWrite(
        trace=trace,
        evidence_run_payloads=(run_payload,),
        evidence_snapshot_payloads=snapshots,
        answer_attempt_payloads=(
            AnswerAttemptPayload(
                payload_id=answer_payload_id,
                attempt_id=attempt_id,
                answer_body=answer_body,
            ),
        ),
    )


def _marker_namespace(answer_body: str) -> MarkerNamespace:
    matches = tuple(_LEGACY_MARKER.finditer(answer_body))
    namespaces = {
        (
            MarkerNamespace.CHATBOOK_S_V1
            if match.group(1)
            else MarkerNamespace.LEGACY_NUMERIC_V1
        )
        for match in matches
    }
    if len(namespaces) > 1:
        raise ValueError("mixed legacy marker namespaces are not synthesized")
    if namespaces:
        return next(iter(namespaces))
    return MarkerNamespace.LEGACY_NUMERIC_V1


def _reference_marker_ordinals(
    references: Sequence[EvidenceReference],
) -> tuple[int, ...]:
    ordinals: list[int] = []
    for fallback, reference in enumerate(references, start=1):
        raw = reference.evidence_id
        if raw.isdecimal() and int(raw) > 0 and not raw.startswith("0"):
            ordinal = int(raw)
        elif (
            raw.startswith("S")
            and raw[1:].isdecimal()
            and int(raw[1:]) > 0
            and not raw[1:].startswith("0")
        ):
            ordinal = int(raw[1:])
        else:
            ordinal = fallback
        ordinals.append(ordinal)
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("legacy marker ordinals must be unique")
    return tuple(ordinals)


def _legacy_prompt_payloads(
    references: Sequence[EvidenceReference],
    marker_ordinals: tuple[int, ...],
    run_id: str,
) -> tuple[
    tuple[PromptEvidenceEntry, ...],
    tuple[EvidenceSnapshotPayload, ...],
]:
    entries: list[PromptEvidenceEntry] = []
    snapshots: list[EvidenceSnapshotPayload] = []
    for evidence_ordinal, (reference, marker_ordinal) in enumerate(
        zip(references, marker_ordinals, strict=True),
        start=1,
    ):
        payload_id = new_opaque_id("snapshot-payload")
        entries.append(
            PromptEvidenceEntry(
                evidence_ordinal=evidence_ordinal,
                marker_ordinal=marker_ordinal,
                run_id=run_id,
                snapshot_payload_ref=payload_id,
                storage_mode=EvidenceStorageMode.EPHEMERAL,
            )
        )
        snapshots.append(
            EvidenceSnapshotPayload(
                payload_id=payload_id,
                storage_mode=EvidenceStorageMode.EPHEMERAL,
                snapshot_text=reference.snippet,
                title=reference.title,
                source_identity={
                    "source_id": reference.source_id,
                    "source_type": reference.source_type,
                    "authority_label": reference.authority_label,
                    "workspace_id": reference.workspace_id,
                    "source_owner": reference.source_owner,
                },
                locator=(
                    {"legacy_free_form": reference.content_ref}
                    if reference.content_ref is not None
                    else {}
                ),
            )
        )
    return tuple(entries), tuple(snapshots)


def _legacy_occurrences(
    answer_body: str,
    namespace: MarkerNamespace,
    evidence_by_marker: Mapping[int, int],
) -> tuple[CitationOccurrence, ...]:
    occurrences: list[CitationOccurrence] = []
    for match in _LEGACY_MARKER.finditer(answer_body):
        current_namespace = (
            MarkerNamespace.CHATBOOK_S_V1
            if match.group(1)
            else MarkerNamespace.LEGACY_NUMERIC_V1
        )
        if current_namespace is not namespace:
            raise ValueError("mixed legacy marker namespaces are not synthesized")
        marker_ordinal = int(match.group(2))
        evidence_ordinal = evidence_by_marker.get(marker_ordinal)
        occurrences.append(
            CitationOccurrence(
                occurrence_id=new_opaque_id("legacy-occurrence"),
                occurrence_ordinal=len(occurrences) + 1,
                raw_marker=match.group(0),
                marker_namespace=namespace,
                evidence_ordinal=evidence_ordinal,
                marker_start=match.start(),
                marker_end=match.end(),
                offset_basis=OffsetBasis.UNICODE_CODEPOINT_V1,
                structural_state=(
                    StructuralValidationState.VALID
                    if evidence_ordinal is not None
                    else StructuralValidationState.UNKNOWN_MARKER
                ),
            )
        )
    return tuple(occurrences)


def _legacy_candidate(
    reference: EvidenceReference,
    rank: int,
) -> RetrievalCandidatePayload:
    return RetrievalCandidatePayload(
        candidate_id=new_opaque_id("legacy-candidate"),
        rank=rank,
        source_identity={
            "source_id": reference.source_id,
            "source_type": reference.source_type,
            "authority_label": reference.authority_label,
        },
        title=reference.title,
        locator=(
            {"legacy_free_form": reference.content_ref}
            if reference.content_ref is not None
            else {}
        ),
        score_kind="legacy_score" if reference.score is not None else None,
        score=reference.score,
    )


def _legacy_citation_candidate(
    citation: CitationRef,
    rank: int,
) -> RetrievalCandidatePayload:
    return RetrievalCandidatePayload(
        candidate_id=new_opaque_id("legacy-candidate"),
        rank=rank,
        source_identity={
            "evidence_id": citation.evidence_id,
            "source_id": citation.source_id,
        },
    )


__all__ = [
    "synthesize_legacy_citation_refs",
    "synthesize_legacy_evidence_bundle",
    "synthesize_legacy_payloads",
]
