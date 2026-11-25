"""Project Studio provenance without carrying the output or source text."""

import json
from dataclasses import asdict
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from tldw_chatbook.Workspaces.output_membership import output_membership_metadata

from .chat_models import ResearchSourceVersionRef
from .studio_mapping import STUDIO_OUTPUT_MAPPINGS
from .studio_models import ResearchOutputKind, ResearchStudioRequest, WorkspaceOutputRef


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


def server_copy_artifact_id(workspace_id: str, item_key: str) -> str:
    """Derive the Server artifact identity from the existing Copy owner proof."""
    return str(uuid5(NAMESPACE_URL, f"tldw:research-copy:{workspace_id}:{item_key}"))


def copied_output_reference(
    ref, owner_id, version, artifact_type, proof
) -> WorkspaceOutputRef:
    kind = ResearchOutputKind(artifact_type)
    owner = STUDIO_OUTPUT_MAPPINGS[ref.data_source, kind]
    return WorkspaceOutputRef(
        ref, kind, owner.owner_kind, owner_id, version, (), None, "complete", proof
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
