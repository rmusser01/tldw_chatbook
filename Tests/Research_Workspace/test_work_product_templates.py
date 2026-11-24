import asyncio
from dataclasses import replace

import pytest

from Tests.Research_Workspace.test_local_studio import local_studio  # noqa: F401
from Tests.Research_Workspace.test_server_studio import _ArtifactServer, server_request
from Tests.Research_Workspace.test_studio_generation import _GenerationFixture, request
from Tests.Research_Workspace.test_studio_models import SECOND
from tldw_chatbook.Research_Workspace.server_studio import ServerStudioAdapter
from tldw_chatbook.Research_Workspace.studio_generation import StudioEvidence
from tldw_chatbook.Research_Workspace.studio_models import ResearchWorkProductKind
from tldw_chatbook.Research_Workspace.studio_service import ResearchStudioService

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

PRODUCTS = [
    (
        "executive_brief",
        ("Situation", "Key Findings", "Evidence", "Risks", "Recommended Actions"),
    ),
    (
        "evidence_bound_hypotheses",
        ("Hypotheses", "Support", "Predictions", "Methods", "Validity", "Confidence"),
    ),
    (
        "research_proposal_pack",
        (
            "Research Question",
            "Literature",
            "Gaps",
            "Hypotheses",
            "Methods",
            "Risks",
            "Source Audit",
        ),
    ),
]


def _fixture(sections):
    fixture = _GenerationFixture()
    fixture.evidence = (
        *fixture.evidence,
        StudioEvidence(SECOND, "Log", "Second observation"),
    )
    fixture.answer = "\n\n".join(
        f"## {label}\n\nObservation [S1]." for label in sections
    )
    return fixture


@pytest.mark.parametrize("kind,sections", PRODUCTS)
async def test_text_work_products_keep_their_identity_in_chatbooks(
    local_studio,  # noqa: F811
    kind,
    sections,
):
    fixture = _fixture(sections)
    operation = request(kind)
    service = ResearchStudioService(fixture.generator(), local_studio.adapter)
    saved = await service.generate(operation, lambda _: None, asyncio.Event())
    assert isinstance(saved.output_kind, ResearchWorkProductKind)
    assert saved.output_kind.value == kind and saved.owner_kind == "local_chatbook"
    assert (await local_studio.adapter.list_outputs(operation.workspace)).items == (
        saved,
    )
    record = await local_studio.artifacts.get_chatbook(saved.owner_id)
    assert record["metadata"]["content"] == fixture.answer
    instruction = fixture.prompts[0][1]["content"]
    assert all(label in instruction for label in sections)


async def test_server_report_artifact_preserves_work_product_identity():
    server = _ArtifactServer()
    owner = ServerStudioAdapter(lambda _: server)
    operation = server_request(output_kind="executive_brief")
    fixture = _fixture(PRODUCTS[0][1])
    fixture.sources = operation.source_snapshot
    fixture.evidence = (StudioEvidence(fixture.sources[0], "Log", "Observation"),)

    async def scoped(ref):
        assert ref == operation.workspace
        return fixture.sources

    async def approvals(ref):
        assert ref == operation.workspace
        return ()

    generator = fixture.generator()
    generator.current_sources = scoped
    generator.approvals = approvals
    saved = await ResearchStudioService(generator, owner).generate(
        operation, lambda _: None, asyncio.Event()
    )
    artifact = await owner.resolve_output(saved)
    assert artifact["artifact_type"] == "report"
    assert (
        artifact["producer_metadata"]["research_generation"]["output_kind"]
        == "executive_brief"
    )


async def test_work_product_rejects_missing_sections_and_unknown_citations():
    fixture = _fixture(PRODUCTS[0][1])
    operation = request("executive_brief")
    with pytest.raises(ValueError, match="require citations"):
        replace(
            operation,
            generation=replace(
                operation.generation,
                retrieval=replace(operation.generation.retrieval, citations=False),
            ),
        )
    for answer in ("## Situation\nOnly this section [S1]", fixture.answer + " [S9]"):
        fixture.answer = answer
        result = await fixture.generator().generate(
            operation, on_chunk=lambda _: None, cancel_event=asyncio.Event()
        )
        assert result.status == "failed" and "work product" in result.failure_reason

    with pytest.raises(ValueError, match="at least 2 selected ready sources"):
        replace(
            request("evidence_bound_hypotheses"),
            source_snapshot=(operation.source_snapshot[0],),
        )
    with pytest.raises(ValueError, match="available Studio output owner"):
        request("literature_matrix")
