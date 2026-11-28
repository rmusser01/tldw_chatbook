"""Render a Studio text result as a canonical Local Chatbook artifact."""

from .studio_generation import StudioGeneratedOutput
from .studio_models import ResearchStudioRequest


def build_research_chatbook_payload(
    request: ResearchStudioRequest, output: StudioGeneratedOutput
) -> dict:
    label = request.output_kind.value.replace("_", " ").title()
    imported = (
        {
            "review_status": "draft",
            "research_run_ref": request.research_run_ref.provenance(),
        }
        if request.research_run_ref
        else {}
    )
    return {
        "name": request.generation.title or label,
        "description": f"Research Studio {label}",
        "tags": ["research", "artifact"],
        "categories": ["Research", "Artifacts"],
        "metadata": {
            "artifact_source": "research",
            "artifact_kind": request.output_kind.value,
            "content": output.text,
            **imported,
            "citations": [
                {
                    "marker": citation.marker,
                    "source_id": citation.source.source_id,
                    "catalog_item_id": citation.source.catalog_item_id,
                }
                for citation in output.citations
            ],
        },
    }
