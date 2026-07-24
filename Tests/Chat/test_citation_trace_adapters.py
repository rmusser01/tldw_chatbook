from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tldw_chatbook.Chat.citation_evidence_models import (
    CitationRef,
    EvidenceBundle,
    EvidenceReference,
)
from tldw_chatbook.Chat.citation_trace_adapters import (
    synthesize_legacy_citation_refs,
    synthesize_legacy_evidence_bundle,
    synthesize_legacy_payloads,
)
from tldw_chatbook.Chat.citation_trace_models import (
    CitationCompleteness,
    MarkerNamespace,
    TraceOrigin,
)


NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def _bundle(*, authority: str = "Local Library") -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="legacy-bundle",
        query="raw legacy query",
        references=(
            EvidenceReference(
                evidence_id="1",
                source_id="note-1",
                source_type="note",
                title="Private title",
                snippet="Exact legacy snippet",
                authority_label=authority,
                content_ref="/legacy/free/form/path",
            ),
            EvidenceReference(
                evidence_id="2",
                source_id="note-2",
                source_type="note",
                title="Second title",
                snippet="Second snippet",
                authority_label=authority,
            ),
        ),
    )


def test_evidence_bundle_adapter_is_partial_legacy_and_preserves_numeric_markers() -> (
    None
):
    answer = "Legacy claim [1][2]. Repeated [1]. Unknown [9]."
    write = synthesize_legacy_evidence_bundle(
        _bundle(),
        answer_body=answer,
        citations=(
            CitationRef(evidence_id="1", source_id="note-1"),
            CitationRef(evidence_id="2", source_id="note-2"),
        ),
        created_at=NOW,
    )

    assert write.trace.origin is TraceOrigin.LEGACY_INFERRED
    assert write.trace.completeness_at_seal is CitationCompleteness.PARTIAL
    assert (
        write.trace.prompt_evidence_sets[0].marker_namespace
        is MarkerNamespace.LEGACY_NUMERIC_V1
    )
    occurrences = write.trace.answer_attempts[0].occurrences
    assert [occurrence.raw_marker for occurrence in occurrences] == [
        "[1]",
        "[2]",
        "[1]",
        "[9]",
    ]
    assert occurrences[-1].evidence_ordinal is None
    assert write.answer_attempt_payloads[0].answer_body == answer

    aggregate = write.trace.model_dump_json()
    for governed in (
        "raw legacy query",
        "Private title",
        "Exact legacy snippet",
        "/legacy/free/form/path",
        "note-1",
    ):
        assert governed not in aggregate


def test_unknown_legacy_authority_and_missing_prompt_evidence_are_unavailable() -> None:
    unknown_authority = synthesize_legacy_evidence_bundle(
        _bundle(authority="unknown"),
        answer_body="Stored answer [1].",
        created_at=NOW,
    )
    no_bundle = synthesize_legacy_citation_refs(
        (CitationRef(evidence_id="1", source_id="note-1"),),
        answer_body="Stored answer [1].",
        created_at=NOW,
    )

    assert (
        unknown_authority.trace.completeness_at_seal is CitationCompleteness.UNAVAILABLE
    )
    assert no_bundle.trace.completeness_at_seal is CitationCompleteness.UNAVAILABLE
    assert no_bundle.trace.origin is TraceOrigin.LEGACY_INFERRED
    assert no_bundle.answer_attempt_payloads[0].answer_body == "Stored answer [1]."
    assert no_bundle.evidence_run_payloads[0].candidates[0].source_identity == {
        "evidence_id": "1",
        "source_id": "note-1",
    }
    assert "note-1" not in no_bundle.trace.model_dump_json()


def test_legacy_payload_adapter_round_trips_existing_types_without_mutating_them() -> (
    None
):
    bundle = _bundle()
    bundle_payload = bundle.to_payload()
    citation_payloads = [CitationRef(evidence_id="1", source_id="note-1").to_payload()]

    write = synthesize_legacy_payloads(
        bundle_payload,
        citation_payloads,
        answer_body="Legacy [1].",
        created_at=NOW,
    )

    assert bundle.to_payload() == bundle_payload
    assert write.trace.origin is TraceOrigin.LEGACY_INFERRED
    assert write.trace.completeness_at_seal is not CitationCompleteness.COMPLETE


@pytest.mark.parametrize(
    ("bundle_payload", "citation_payloads"),
    (
        ({"bundle_id": "", "references": []}, []),
        ({"bundle_id": "bundle", "references": "not-a-list"}, []),
        ({"bundle_id": "bundle", "references": []}, [{"evidence_id": ""}]),
    ),
)
def test_malformed_legacy_payloads_fail_closed(
    bundle_payload: dict,
    citation_payloads: list[dict],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        synthesize_legacy_payloads(
            bundle_payload,
            citation_payloads,
            answer_body="answer",
            created_at=NOW,
        )
