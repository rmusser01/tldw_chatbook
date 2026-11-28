import asyncio
from dataclasses import replace

import pytest

from Tests.Research_Workspace.test_server_studio import _ArtifactServer, server_request
from Tests.Research_Workspace.test_studio_models import LOCAL, ROUTE, SERVER, SOURCE
from tldw_chatbook.Research_Workspace.server_studio import ServerStudioAdapter
from tldw_chatbook.Research_Workspace.studio_generation import (
    ResearchStudioGenerator,
    StudioEvidence,
)
from tldw_chatbook.Research_Workspace.studio_models import (
    ResearchGenerationConfig,
    ResearchStudioRequest,
)
from tldw_chatbook.Research_Workspace.studio_service import ResearchStudioService


def _generator(answer, *, after_completion=None):
    selected = (replace(SOURCE, ref=SERVER),)

    async def sources(ref):
        assert ref == SERVER
        return selected

    async def evidence(request):
        return (StudioEvidence(request.source_snapshot[0], "Log", "Opened in 2021"),)

    async def approvals(ref):
        assert ref == SERVER
        return ()

    async def stream(request, messages):
        nonlocal selected
        assert "Opened in 2021" in messages[1]["content"]
        assert "[S#]" in messages[0]["content"]
        yield answer
        if after_completion is not None:
            selected = after_completion

    return ResearchStudioGenerator(sources, evidence, stream, approvals)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,artifact_type,answer",
    [
        (
            "mind_map",
            "mindmap",
            "```mermaid\nmindmap\n  Observatory\n    Opened in 2021 [S1]\n```",
        ),
        (
            "timeline",
            "timeline",
            "- [2021] - Opened [S1]\r\n- [2022] - Reached 90% of capacity [S1]",
        ),
    ],
)
async def test_diagram_uses_exact_server_artifact_type_and_versions(
    kind, artifact_type, answer
):
    server = _ArtifactServer()
    operation = server_request(output_kind=kind)
    owner = ServerStudioAdapter(lambda ref: server)
    service = ResearchStudioService(_generator(answer), owner)
    first = await service.generate(operation, lambda _: None, asyncio.Event())
    reopened = ServerStudioAdapter(lambda ref: server)
    row = await reopened.resolve_output(first)
    assert row["artifact_type"] == artifact_type
    assert (
        row["producer_metadata"]["research_citations"][0]["source_id"]
        == SOURCE.source_id
    )
    assert (await reopened.list_outputs(SERVER)).items == (first,)

    second = await service.generate(
        replace(operation, operation_id="new-operation"),
        lambda _: None,
        asyncio.Event(),
        replace_output=first,
    )
    assert (second.owner_id, second.owner_version) == (first.owner_id, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,answer",
    [
        ("mind_map", "flowchart TD\n  Foo --> Bar [S1]"),
        ("mind_map", "mindmap\n  Root\n    %%{init: {}}%% [S1]"),
        ("mind_map", "mindmap\n  Root\n    <b>Unsafe</b> [S1]"),
        ("mind_map", "mindmap\n  Root\n    Leaf [S2]"),
        ("timeline", "- [2021] - [Link](https://example.invalid) [S1]"),
        ("timeline", "- [2021] - Missing evidence"),
        ("timeline", "A paragraph [S1]."),
    ],
)
async def test_invalid_diagram_does_not_publish(kind, answer):
    server = _ArtifactServer()
    owner = ServerStudioAdapter(lambda ref: server)
    service = ResearchStudioService(_generator(answer), owner)
    with pytest.raises(ValueError, match="invalid Mind Map or Timeline"):
        await service.generate(
            server_request(output_kind=kind), lambda _: None, asyncio.Event()
        )
    assert not server.writes


@pytest.mark.parametrize("kind", ["mind_map", "timeline"])
def test_diagram_does_not_construct_a_local_owner_request(kind):
    with pytest.raises(ValueError, match="available Studio output owner"):
        ResearchStudioRequest(
            "op", LOCAL, kind, (SOURCE,), ResearchGenerationConfig(), ROUTE, 1
        )


@pytest.mark.asyncio
async def test_selected_source_change_while_generating_prevents_diagram_publication():
    server = _ArtifactServer()
    owner = ServerStudioAdapter(lambda ref: server)
    service = ResearchStudioService(
        _generator("- [2021] - Opened [S1]", after_completion=()), owner
    )
    with pytest.raises(ValueError, match="Selected sources changed"):
        await service.generate(
            server_request(output_kind="timeline"), lambda _: None, asyncio.Event()
        )
    assert not server.writes
