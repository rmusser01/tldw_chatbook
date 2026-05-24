"""Answer-level citation prompt and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from tldw_chatbook.Chat.citation_evidence_models import (
    CitationRef,
    EvidenceBundle,
    EvidenceReference,
)


CITATION_MARKER_PATTERN = re.compile(r"\[(S[0-9][A-Za-z0-9_-]*)\]")
QUOTE_BOUNDARY_CHARS = (".", "?", "!", "\n")


@dataclass(frozen=True)
class AnswerCitationValidation:
    """Validated citation state extracted from one assistant answer.

    Attributes:
        status: Overall answer citation status.
        citations: Per-marker and uncited evidence citation records.
        cited_evidence_ids: Citation labels found in the answer text.
        unknown_citation_ids: Citation labels that do not map to staged evidence.
        uncited_evidence_ids: Available evidence labels not cited in the answer.
        recovery: User-facing recovery copy for unverified or insufficient evidence.
    """

    status: str
    citations: tuple[CitationRef, ...]
    cited_evidence_ids: tuple[str, ...]
    unknown_citation_ids: tuple[str, ...]
    uncited_evidence_ids: tuple[str, ...]
    recovery: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Serialize validation output for later persistence/export slices.

        Returns:
            JSON-safe citation validation payload.
        """
        return {
            "status": self.status,
            "citations": [citation.to_payload() for citation in self.citations],
            "cited_evidence_ids": list(self.cited_evidence_ids),
            "unknown_citation_ids": list(self.unknown_citation_ids),
            "uncited_evidence_ids": list(self.uncited_evidence_ids),
            "recovery": self.recovery,
        }


def evidence_bundle_from_value(value: Any) -> EvidenceBundle | None:
    """Coerce a payload value into an evidence bundle when possible.

    Args:
        value: Existing evidence bundle object or serialized evidence bundle payload.

    Returns:
        Evidence bundle when the value can be parsed, otherwise ``None``.
    """
    if isinstance(value, EvidenceBundle):
        return value
    if isinstance(value, Mapping):
        try:
            return EvidenceBundle.from_payload(value)
        except (TypeError, ValueError):
            return None
    return None


def format_evidence_for_cited_answer(value: Any) -> str:
    """Render staged evidence plus citation instructions for model prompts.

    Args:
        value: Existing evidence bundle object or serialized evidence bundle payload.

    Returns:
        Prompt-ready staged evidence block, or an empty string when no valid
        evidence bundle is available.
    """
    bundle = evidence_bundle_from_value(value)
    if bundle is None:
        return ""

    available_count = len(bundle.available_references())
    lines = [
        "[Staged evidence]",
        f"Evidence bundle: {bundle.bundle_id}",
        f"Evidence query: {bundle.query or 'none'}",
        f"Evidence status: {bundle.status}",
        "[Citation instructions]",
        "- Use only available evidence references below to ground factual claims.",
        "- Cite each evidence-supported claim with its bracketed source label, for example [S1].",
        "- If the available evidence is insufficient, say so explicitly instead of guessing.",
        "- Do not invent citation labels; unknown labels are untrusted.",
    ]
    if available_count == 0:
        lines.append("No available evidence references are eligible for grounding.")

    for reference in bundle.references:
        lines.extend(_format_reference_for_prompt(reference))
    return "\n".join(lines) + "\n\n"


def extract_citation_markers(answer_text: Any) -> tuple[str, ...]:
    """Extract first-seen evidence labels from assistant text.

    Args:
        answer_text: Assistant answer text to scan for ``[S#]`` markers.

    Returns:
        Citation labels in first-seen order with duplicates removed.
    """
    seen: set[str] = set()
    markers: list[str] = []
    for match in CITATION_MARKER_PATTERN.finditer(str(answer_text or "")):
        marker = match.group(1)
        if marker not in seen:
            markers.append(marker)
            seen.add(marker)
    return tuple(markers)


def build_answer_citation_validation(
    answer_text: Any,
    evidence_bundle: Any,
) -> AnswerCitationValidation:
    """Validate assistant citation markers against staged evidence.

    Args:
        answer_text: Assistant answer text containing optional ``[S#]`` markers.
        evidence_bundle: Existing evidence bundle object or serialized evidence bundle payload.

    Returns:
        Validation result containing overall status, marker-level citation
        states, uncited available evidence, and recovery copy when applicable.
    """
    bundle = evidence_bundle_from_value(evidence_bundle)
    if bundle is None:
        return AnswerCitationValidation(
            status="unknown",
            citations=(),
            cited_evidence_ids=(),
            unknown_citation_ids=(),
            uncited_evidence_ids=(),
            recovery="No evidence bundle is available for citation validation.",
        )

    cited_ids = extract_citation_markers(answer_text)
    citations: list[CitationRef] = []
    unknown_ids: list[str] = []
    invalid_cited_ids: list[str] = []
    validated_count = 0

    for evidence_id in cited_ids:
        reference = bundle.reference_by_id(evidence_id)
        if reference is None:
            unknown_ids.append(evidence_id)
            invalid_cited_ids.append(evidence_id)
            citations.append(
                CitationRef(
                    evidence_id=evidence_id,
                    source_id="unknown",
                    quote=_quote_for_marker(str(answer_text or ""), evidence_id),
                    status="unknown",
                )
            )
            continue

        citation = CitationRef(
            evidence_id=evidence_id,
            source_id=reference.source_id,
            quote=_quote_for_marker(str(answer_text or ""), evidence_id),
            status="validated",
        ).validate_against(bundle)
        if citation.status == "validated":
            validated_count += 1
        else:
            invalid_cited_ids.append(evidence_id)
            if citation.status == "unknown":
                unknown_ids.append(evidence_id)
        citations.append(citation)

    available_ids = tuple(reference.evidence_id for reference in bundle.available_references())
    uncited_ids = tuple(evidence_id for evidence_id in available_ids if evidence_id not in cited_ids)
    for evidence_id in uncited_ids:
        reference = bundle.reference_by_id(evidence_id)
        if reference is not None:
            citations.append(
                CitationRef(
                    evidence_id=evidence_id,
                    source_id=reference.source_id,
                    status="uncited",
                )
            )

    if available_ids:
        if invalid_cited_ids:
            status = "unverified"
            recovery = "Some citation markers do not match available staged evidence."
        elif validated_count:
            status = "validated"
            recovery = ""
        else:
            status = "uncited"
            recovery = "The answer does not cite available staged evidence."
    else:
        status = "insufficient_evidence"
        recovery = "No available evidence can validate this answer."
        citations.extend(
            CitationRef(
                evidence_id=reference.evidence_id,
                source_id=reference.source_id,
                status=reference.status,
            )
            for reference in bundle.references
            if reference.evidence_id not in cited_ids
        )

    return AnswerCitationValidation(
        status=status,
        citations=tuple(citations),
        cited_evidence_ids=cited_ids,
        unknown_citation_ids=tuple(unknown_ids),
        uncited_evidence_ids=uncited_ids,
        recovery=recovery,
    )


def _format_reference_for_prompt(reference: EvidenceReference) -> list[str]:
    lines = [
        (
            f"[{reference.evidence_id}] {reference.title} "
            f"({reference.source_id}) - {reference.authority_label} - {reference.status}"
        )
    ]
    if reference.status != "available":
        lines.append("Do not cite this reference as trusted evidence.")
    if reference.snippet:
        lines.append(f"Snippet: {reference.snippet}")
    return lines


def _quote_for_marker(answer_text: str, evidence_id: str) -> str:
    marker = f"[{evidence_id}]"
    marker_index = answer_text.find(marker)
    if marker_index < 0:
        return ""

    sentence_start = max(
        answer_text.rfind(boundary, 0, marker_index)
        for boundary in QUOTE_BOUNDARY_CHARS
    )
    start = 0 if sentence_start < 0 else sentence_start + 1
    sentence_end_candidates = [
        index
        for index in (
            answer_text.find(boundary, marker_index + len(marker))
            for boundary in QUOTE_BOUNDARY_CHARS
        )
        if index >= 0
    ]
    end = min(sentence_end_candidates) + 1 if sentence_end_candidates else len(answer_text)
    return answer_text[start:end].strip()
