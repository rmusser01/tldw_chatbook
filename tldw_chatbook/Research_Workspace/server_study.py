"""Translate qualified Studio requests and receipts for native Server Study."""

from __future__ import annotations

from datetime import datetime, timezone

from tldw_chatbook.Workspaces.output_membership import output_membership_metadata

from .studio_mapping import PRIMARY_OUTPUT_MAPPINGS
from .studio_models import ResearchOutputKind
from .studio_provenance import generation_metadata, output_reference


def study_payload(request, generated, supersedes) -> dict:
    """Send the validated generated entries and their captured source IDs."""
    kind = request.output_kind
    entries = (
        generated.cards
        if kind is ResearchOutputKind.FLASHCARDS
        else generated.questions
    )
    if not entries or len(entries) > request.generation.item_count:
        raise ValueError("Study needs a bounded set of valid generated entries")
    citation_sources = {
        citation.marker: citation.source.source_id for citation in generated.citations
    }
    items = []
    for entry in entries:
        if any(marker not in citation_sources for marker in entry.citations):
            raise ValueError(
                "An output citation belongs to a different source snapshot"
            )
        fields = entry.model_dump(mode="json", exclude_none=True)
        fields["citations"] = [citation_sources[marker] for marker in entry.citations]
        items.append(fields)
    return {
        **generation_metadata(request, supersedes)["research_generation"],
        "title": request.generation.title,
        "items": items,
    }


def study_result(ref, row, *, expected=None):
    """Project the native owner only after validating its provenance."""
    generation = output_membership_metadata(
        {"research_generation": row.generation}, "output"
    )["research_generation"]
    kind = ResearchOutputKind(generation["output_kind"])
    if generation["status"] != "complete":
        raise ValueError("The Server has no completed Study record for this output")
    if kind not in {ResearchOutputKind.FLASHCARDS, ResearchOutputKind.QUIZ}:
        raise ValueError("The Server returned a different Study output kind")
    if expected is not None and generation != expected:
        raise ValueError("The Server did not confirm the Study generation provenance")
    mapping = PRIMARY_OUTPUT_MAPPINGS[ref.data_source, kind]
    output = output_reference(
        ref, str(row.owner_id), mapping.owner_kind, row.version, generation
    )
    return output, row.model_dump(mode="json")


def output_creation_time(row) -> float:
    """Order native Study and workspace artifact records by a common timestamp."""
    try:
        timestamp = datetime.fromisoformat(row.get("created_at") or "")
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.timestamp()
    except (TypeError, ValueError, OverflowError, OSError):
        return 0.0
