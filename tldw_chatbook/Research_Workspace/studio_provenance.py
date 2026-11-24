"""Encode generation fields without carrying the output or source text."""

import json
from dataclasses import asdict
from hashlib import sha256

from tldw_chatbook.Workspaces.output_membership import output_membership_metadata

from .chat_models import ResearchSourceVersionRef
from .studio_models import ResearchStudioRequest, WorkspaceOutputRef


def generation_metadata(
    request: ResearchStudioRequest, supersedes: WorkspaceOutputRef | None = None
):
    sources = []
    for source in request.source_snapshot:
        fields = asdict(source)
        del fields["ref"]
        sources.append(fields)
    route = json.dumps(asdict(request.processing_route), sort_keys=True)
    return output_membership_metadata(
        {
            "research_generation": {
                "operation_id": request.operation_id,
                "output_kind": request.output_kind.value,
                "source_snapshot": sources,
                "settings_fingerprint": request.generation.fingerprint,
                "processing_route_fingerprint": sha256(route.encode()).hexdigest(),
                "provider": request.processing_route.provider,
                "model": request.processing_route.model,
                "status": "complete",
                "supersedes_owner_id": supersedes.owner_id if supersedes else None,
                "research_run_ref": request.research_run_ref.provenance()
                if request.research_run_ref
                else None,
            }
        },
        "output",
    )


def output_reference(
    ref, owner_id, owner_kind, version, provenance
) -> WorkspaceOutputRef:
    return WorkspaceOutputRef(
        ref,
        provenance["output_kind"],
        owner_kind,
        owner_id,
        version,
        tuple(
            ResearchSourceVersionRef(ref=ref, **source)
            for source in provenance["source_snapshot"]
        ),
        provenance["settings_fingerprint"],
        provenance["status"],
    )


def citation_references(generated, markers):
    return [
        {
            "marker": citation.marker,
            "source_id": citation.source.source_id,
            "catalog_item_id": citation.source.catalog_item_id,
        }
        for citation in generated.citations
        if citation.marker in markers
    ]
