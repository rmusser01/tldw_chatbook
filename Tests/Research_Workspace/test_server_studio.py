from dataclasses import replace

import pytest

from Tests.Research_Workspace.test_studio_models import ROUTE, SERVER, SOURCE
from Tests.Research_Workspace.test_studio_generation import request
from tldw_chatbook.Research_Workspace.server_studio import ServerStudioAdapter
from tldw_chatbook.Research_Workspace.studio_generation import (
    StudioCitation,
    StudioGeneratedOutput,
)

pytestmark = pytest.mark.asyncio


def server_request(kind="report", **updates):
    return replace(
        request(kind),
        workspace=SERVER,
        source_snapshot=(replace(SOURCE, ref=SERVER),),
        processing_route=replace(
            ROUTE,
            data_source="server",
            processor="server-chat",
            endpoint_class="server",
            remote=True,
        ),
        **updates,
    )


class _ArtifactServer:
    def __init__(self):
        self.saved = {}
        self.writes = []
        self.drop_provenance = False
        self.drop_version = False
        self.after_write = lambda: None

    async def list_workspace_artifacts(self, workspace_id):
        assert workspace_id == SERVER.workspace_id
        return list(self.saved.values())

    async def create_workspace_artifact(self, workspace_id, payload):
        assert workspace_id == SERVER.workspace_id
        self.writes.append(("create", payload.id))
        row = {**payload.model_dump(), "workspace_id": workspace_id, "version": 1}
        if self.drop_provenance:
            row.pop("producer_metadata")
        if self.drop_version:
            row.pop("version")
        self.saved[payload.id] = row
        self.after_write()
        return row

    async def update_workspace_artifact(self, workspace_id, artifact_id, payload):
        assert workspace_id == SERVER.workspace_id
        row = self.saved[artifact_id]
        assert row["version"] == payload.version
        self.writes.append(("update", artifact_id, payload.version))
        row.update(payload.model_dump(exclude_unset=True), version=payload.version + 1)
        return row


async def test_server_studio_uses_canonical_artifact_versions_and_reloads_provenance():
    server = _ArtifactServer()

    def client_for(ref):
        assert ref == SERVER
        return server

    owner = ServerStudioAdapter(client_for)
    turn = server_request()
    body = StudioGeneratedOutput(
        "complete",
        text="A saved server report [S1].",
        citations=(StudioCitation("S1", turn.source_snapshot[0]),),
    )
    first = await owner.save(turn, body)
    assert first.status == "complete" and first.owner_version == 1
    assert await ServerStudioAdapter(client_for).save(turn, body) == first
    assert len(server.writes) == 1
    metadata = server.saved[first.owner_id]["producer_metadata"]["research_generation"]
    assert metadata["operation_id"] == turn.operation_id
    assert metadata["source_snapshot"][0]["catalog_item_id"] == SOURCE.catalog_item_id
    assert body.text not in repr(metadata)
    assert server.saved[first.owner_id]["producer_metadata"]["research_citations"] == [
        {
            "marker": "S1",
            "source_id": SOURCE.source_id,
            "catalog_item_id": SOURCE.catalog_item_id,
        }
    ]
    assert (await owner.list_outputs(SERVER)).items == (first,)
    assert (await owner.resolve_output(first))["content"] == body.text

    second = await owner.save(
        server_request(operation_id="replace-1"), body, replace_output=first
    )
    assert second.owner_id == first.owner_id and second.owner_version == 2
    with pytest.raises(ValueError, match="changed"):
        await owner.save(
            server_request(operation_id="replace-2"), body, replace_output=first
        )
    assert server.writes == [("create", first.owner_id), ("update", first.owner_id, 1)]


async def test_server_output_requires_a_confirmed_owner_receipt_and_scoped_authority():
    server = _ArtifactServer()
    owner = ServerStudioAdapter(lambda _ref: server)
    server.drop_provenance = True
    with pytest.raises(ValueError, match="confirm.*provenance"):
        await owner.save(
            server_request(),
            StudioGeneratedOutput("complete", text="Unconfirmed content"),
        )
    with pytest.raises(ValueError, match="Server workspace"):
        await owner.save(
            request(), StudioGeneratedOutput("complete", text="Local content")
        )
    assert (await owner.list_outputs(SERVER)).items == ()


async def test_server_output_requires_a_proven_version_and_current_profile_after_write():
    server = _ArtifactServer()
    current = server
    owner = ServerStudioAdapter(lambda _ref: current)
    generated = StudioGeneratedOutput("complete", text="Unconfirmed")
    server.drop_version = True
    with pytest.raises(ValueError, match="revision"):
        await owner.save(server_request(), generated)
    server.drop_version = False

    def switch_client():
        nonlocal current
        current = _ArtifactServer()

    server.after_write = switch_client
    server.saved.clear()
    with pytest.raises(ValueError, match="selected server changed"):
        await owner.save(server_request(), generated)
    assert len(server.saved) == 1 and not current.saved


async def test_server_output_rejects_unselected_citations_before_writing():
    server = _ArtifactServer()
    owner = ServerStudioAdapter(lambda _ref: server)
    with pytest.raises(ValueError, match="citation"):
        await owner.save(
            server_request(),
            StudioGeneratedOutput(
                "complete",
                text="Foreign [S1]",
                citations=(StudioCitation("S1", SOURCE),),
            ),
        )
    assert not server.writes
