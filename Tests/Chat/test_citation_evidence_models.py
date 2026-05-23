from __future__ import annotations

import json

import pytest

from tldw_chatbook.Chat.citation_evidence_models import (
    CitationRef,
    EvidenceBundle,
    EvidenceReference,
    EVIDENCE_SNIPPET_CHAR_LIMIT,
)


def test_evidence_bundle_round_trip_preserves_source_authority_and_json_payload() -> None:
    reference = EvidenceReference(
        evidence_id="S1",
        source_id="note-1",
        source_type="note",
        title="Release notes",
        snippet="The feature is enabled by default.",
        authority_label="Workspace: Research",
        workspace_id="workspace-a",
        source_owner="local",
        content_ref="local:note:note-1",
        score=0.87,
        metadata={"page": 2, "token": "secret-value"},
    )
    bundle = EvidenceBundle(
        bundle_id="bundle-1",
        query="Is the feature enabled?",
        references=(reference,),
    )

    payload = bundle.to_payload()
    json.dumps(payload)
    restored = EvidenceBundle.from_payload(payload)

    assert payload["references"][0]["metadata"] == {"page": 2}
    assert restored.references[0].evidence_id == "S1"
    assert restored.references[0].authority_label == "Workspace: Research"
    assert restored.references[0].workspace_id == "workspace-a"
    assert restored.references[0].source_owner == "local"
    assert restored.references[0].content_ref == "local:note:note-1"
    assert restored.available_references()[0].source_id == "note-1"


def test_evidence_reference_truncates_large_snippets_without_losing_original_count() -> None:
    long_snippet = "x" * (EVIDENCE_SNIPPET_CHAR_LIMIT + 25)

    reference = EvidenceReference(
        evidence_id="S1",
        source_id="note-1",
        source_type="note",
        title="Large source",
        snippet=long_snippet,
        authority_label="Local Library",
    )

    payload = reference.to_payload()

    assert len(payload["snippet"]) == EVIDENCE_SNIPPET_CHAR_LIMIT
    assert payload["snippet_truncated"] is True
    assert payload["original_snippet_char_count"] == len(long_snippet)


def test_evidence_contract_rejects_unsupported_metadata_payloads() -> None:
    with pytest.raises(TypeError, match="unsupported metadata value"):
        EvidenceReference(
            evidence_id="S1",
            source_id="note-1",
            source_type="note",
            title="Bad metadata",
            snippet="snippet",
            authority_label="Local Library",
            metadata={"callback": object()},
        ).to_payload()


def test_citation_ref_validates_against_bundle_and_rejects_unknown_status() -> None:
    bundle = EvidenceBundle(
        bundle_id="bundle-1",
        query="question",
        references=(
            EvidenceReference(
                evidence_id="S1",
                source_id="note-1",
                source_type="note",
                title="Release notes",
                snippet="The feature is enabled.",
                authority_label="Local Library",
            ),
        ),
    )
    citation = CitationRef(
        evidence_id="S1",
        source_id="note-1",
        quote="The feature is enabled.",
        status="validated",
    )

    assert citation.to_payload()["status"] == "validated"
    assert citation.validate_against(bundle).status == "validated"
    assert CitationRef(evidence_id="S2", source_id="note-2", status="validated").validate_against(bundle).status == "unknown"

    with pytest.raises(ValueError, match="Unsupported citation status"):
        CitationRef(evidence_id="S1", source_id="note-1", status="trusted")
