from __future__ import annotations

from types import SimpleNamespace

from tldw_chatbook.Chat.answer_citations import (
    build_answer_citation_validation,
    extract_citation_markers,
    format_evidence_for_cited_answer,
)
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.citation_evidence_models import EvidenceBundle, EvidenceReference
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import (
    attach_current_handoff_citation_validation,
)


def _bundle(*references: EvidenceReference, status: str = "available") -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="library-rag:incident",
        query="Why did the incident happen?",
        status=status,
        references=references,
    )


def _reference(
    evidence_id: str,
    *,
    source_id: str = "note-42",
    title: str = "Incident Review",
    snippet: str = "Expired credential caused the incident.",
    status: str = "available",
) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=evidence_id,
        source_id=source_id,
        source_type="note",
        title=title,
        snippet=snippet,
        authority_label="Source authority: local",
        status=status,
    )


def test_format_evidence_for_cited_answer_injects_citation_instructions() -> None:
    context = format_evidence_for_cited_answer(_bundle(_reference("S1")))

    assert "[Staged evidence]" in context
    assert "[Citation instructions]" in context
    assert "Cite each evidence-supported claim with its bracketed source label" in context
    assert "Do not invent citation labels" in context
    assert "[S1] Incident Review (note-42) - Source authority: local - available" in context
    assert "Snippet: Expired credential caused the incident." in context


def test_format_evidence_for_cited_answer_reports_insufficient_available_evidence() -> None:
    context = format_evidence_for_cited_answer(
        _bundle(
            _reference(
                "S1",
                snippet="This belongs to another workspace.",
                status="blocked",
            ),
            status="blocked",
        )
    )

    assert "No available evidence references are eligible for grounding." in context
    assert "If the available evidence is insufficient, say so explicitly" in context
    assert "[S1] Incident Review (note-42) - Source authority: local - blocked" in context
    assert "Do not cite this reference as trusted evidence." in context


def test_extract_citation_markers_preserves_first_seen_order_without_duplicates() -> None:
    markers = extract_citation_markers(
        "The outage came from credential expiry [S2]. It also affected login [S1]. "
        "Duplicate details [S2] and prose [not-a-source]."
    )

    assert markers == ("S2", "S1")


def test_answer_citation_validation_marks_valid_unknown_and_uncited_refs() -> None:
    result = build_answer_citation_validation(
        "The credential expired [S1]. A made-up source was also cited [S9].",
        _bundle(
            _reference("S1", source_id="note-1"),
            _reference("S2", source_id="note-2", title="Runbook"),
        ),
    )

    payload = result.to_payload()

    assert result.status == "unverified"
    assert result.cited_evidence_ids == ("S1", "S9")
    assert result.unknown_citation_ids == ("S9",)
    assert result.uncited_evidence_ids == ("S2",)
    assert payload["citations"][0]["evidence_id"] == "S1"
    assert payload["citations"][0]["source_id"] == "note-1"
    assert payload["citations"][0]["status"] == "validated"
    assert payload["citations"][1]["evidence_id"] == "S9"
    assert payload["citations"][1]["status"] == "unknown"
    assert payload["citations"][2]["evidence_id"] == "S2"
    assert payload["citations"][2]["status"] == "uncited"


def test_answer_citation_validation_is_insufficient_when_no_available_evidence_exists() -> None:
    result = build_answer_citation_validation(
        "The answer should not be trusted as grounded.",
        _bundle(_reference("S1", status="blocked"), status="blocked"),
    )

    assert result.status == "insufficient_evidence"
    assert result.recovery == "No available evidence can validate this answer."
    assert result.to_payload()["citations"][0]["status"] == "blocked"


def test_answer_citation_validation_does_not_duplicate_cited_blocked_reference() -> None:
    result = build_answer_citation_validation(
        "The cited source is blocked [S1].",
        _bundle(_reference("S1", status="blocked"), status="blocked"),
    )

    payload = result.to_payload()

    assert result.status == "insufficient_evidence"
    assert [citation["evidence_id"] for citation in payload["citations"]] == ["S1"]
    assert payload["citations"][0]["status"] == "blocked"


def test_answer_citation_validation_marks_cited_blocked_reference_unverified() -> None:
    result = build_answer_citation_validation(
        "The blocked source was cited [S2].",
        _bundle(
            _reference("S1", source_id="note-1"),
            _reference("S2", source_id="note-2", status="blocked"),
        ),
    )

    payload = result.to_payload()

    assert result.status == "unverified"
    assert result.uncited_evidence_ids == ("S1",)
    assert payload["citations"][0]["evidence_id"] == "S2"
    assert payload["citations"][0]["status"] == "blocked"


def test_answer_citation_validation_marks_mixed_validated_and_blocked_unverified() -> None:
    result = build_answer_citation_validation(
        "The credential expired [S1]. The blocked source was cited [S2].",
        _bundle(
            _reference("S1", source_id="note-1"),
            _reference("S2", source_id="note-2", status="blocked"),
        ),
    )

    payload = result.to_payload()

    assert result.status == "unverified"
    assert result.cited_evidence_ids == ("S1", "S2")
    assert result.uncited_evidence_ids == ()
    assert [citation["status"] for citation in payload["citations"]] == ["validated", "blocked"]


def test_answer_citation_validation_quote_uses_question_and_exclamation_boundaries() -> None:
    result = build_answer_citation_validation(
        "Can credentials expire? They did [S1]! Follow-up started.",
        _bundle(_reference("S1")),
    )

    assert result.citations[0].quote == "They did [S1]!"


def test_chat_handoff_model_context_uses_answer_citation_prompt_contract() -> None:
    payload = ChatHandoffPayload(
        source="search-rag",
        item_type="rag-result",
        title="Incident Review",
        body="Retrieved content",
        metadata={
            "evidence_bundle": _bundle(_reference("S1")).to_payload(),
        },
    )

    context = payload.model_context_block()

    assert "[Citation instructions]" in context
    assert "Cite each evidence-supported claim with its bracketed source label" in context
    assert "Content:\nRetrieved content" in context


def test_attach_current_handoff_citation_validation_sets_widget_and_app_payload() -> None:
    app = SimpleNamespace(
        _current_chat_handoff_payload=ChatHandoffPayload(
            source="search-rag",
            item_type="rag-result",
            title="Incident Review",
            body="Retrieved content",
            metadata={
                "evidence_bundle": _bundle(_reference("S1", source_id="note-1")).to_payload(),
            },
        ).to_dict()
    )
    widget = SimpleNamespace()

    validation = attach_current_handoff_citation_validation(
        app,
        widget,
        "The credential expired [S1].",
    )

    assert validation is not None
    assert validation.status == "validated"
    assert widget.citation_validation.status == "validated"
    assert widget.citation_refs[0].evidence_id == "S1"
    assert app._current_chat_answer_citation_validation["status"] == "validated"


def test_attach_current_handoff_citation_validation_clears_stale_app_payload_without_evidence() -> None:
    app = SimpleNamespace(
        _current_chat_handoff_payload=None,
        _current_chat_pending_evidence_bundle=None,
        _current_chat_answer_citation_validation={"status": "validated"},
    )
    widget = SimpleNamespace()

    validation = attach_current_handoff_citation_validation(
        app,
        widget,
        "Ungrounded answer.",
    )

    assert validation is None
    assert app._current_chat_answer_citation_validation is None
    assert not hasattr(widget, "citation_validation")
