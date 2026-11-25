from dataclasses import replace

import pytest

from Tests.Research_Workspace.test_studio_models import ROUTE, SERVER, SOURCE
from Tests.Research_Workspace.test_studio_generation import request
from tldw_chatbook.Research_Workspace.server_studio import ServerStudioAdapter
from tldw_chatbook.tldw_api.exceptions import APIResponseError
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
        self.exports = []

    async def list_workspace_artifacts(self, workspace_id):
        assert workspace_id == SERVER.workspace_id
        return list(self.saved.values())

    async def create_workspace_artifact(self, workspace_id, payload):
        assert workspace_id == SERVER.workspace_id
        self.writes.append(("create", payload.id))
        row = {
            **payload.model_dump(),
            "workspace_id": workspace_id,
            "version": 1,
            "review_state": "draft",
            "artifact_version_id": f"{payload.id}:v1",
        }
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
        row["artifact_version_id"] = f"{artifact_id}:v{row['version']}"
        return row

    async def _request(self, method, path, *, json_data=None, params=None, headers=None):
        if method == "GET" and path.endswith("/study-outputs"):
            raise APIResponseError(404, "Not Found")
        assert method == "POST"
        artifact_id = path.split("/")[-2]
        assert (
            path
            == f"/api/v1/workspaces/{SERVER.workspace_id}/artifacts/{artifact_id}/exports"
        )
        assert json_data == {
            "format": "md",
            "artifact_version_id": self.saved[artifact_id]["artifact_version_id"],
        }
        self.exports.append(json_data)
        return {
            "workspace_id": SERVER.workspace_id,
            "artifact_id": artifact_id,
            "artifact_version_id": json_data["artifact_version_id"],
            "review_state": "accepted",
            "format": "md",
            "content_type": "text/markdown",
            "content": "Reviewed report",
            "bytes": 15,
        }


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


async def test_server_export_requires_an_accepted_exact_version_and_edit_reopens_review():
    server = _ArtifactServer()
    owner = ServerStudioAdapter(lambda _ref: server)
    artifact = await owner.save(
        server_request(), StudioGeneratedOutput("complete", text="One report")
    )
    with pytest.raises(ValueError, match="accept"):
        await owner.export_output(artifact, "markdown")
    assert server.exports == []
    await owner.accept_output(artifact)
    (accepted,) = (await owner.list_outputs(SERVER)).items
    assert accepted.owner_version == artifact.owner_version + 1
    assert await owner.export_output(accepted, "markdown") == b"Reviewed report"
    with pytest.raises(ValueError, match="current version"):
        await owner.export_output(artifact, "markdown")
    await owner.edit_output(accepted, "New text")
    (edited,) = (await owner.list_outputs(SERVER)).items
    with pytest.raises(ValueError, match="accept"):
        await owner.export_output(edited, "markdown")
    assert len(server.exports) == 1


async def test_native_workspace_data_tables_confirm_and_recover_the_canonical_owner():
    import asyncio
    from uuid import UUID

    from tldw_chatbook.Research_Workspace.studio_service import ResearchStudioService

    native_id = str(UUID(int=44))
    source = replace(SOURCE, ref=SERVER, catalog_item_id="17", catalog_item_version=3, association_version=2)
    route = replace(ROUTE, data_source="server", processor="server-chat", endpoint_class="server")

    class TablesServer(_ArtifactServer):
        def __init__(self):
            super().__init__()
            self.launches = []
            self.detail = {"uuid": native_id, "name": "Native table", "prompt": "Extract", "version": 4,
                           "status": "ready", "row_count": 1, "workspace_tag": f"workspace:{SERVER.workspace_id}"}

        async def _request(self, method, path, *, json_data=None, params=None, headers=None):
            if path.endswith("/research-data-tables"):
                if method == "GET":
                    assert params == {"provider": route.provider, "model": route.model}
                    return {**params, "contract": "research-workspace-data-tables.v1",
                            "available": True, "max_sources": 20}
                self.launches.append(json_data)
                assert json_data["sources"][0]["catalog_item_version"] == 3
                assert json_data["sources"][0]["catalog_item_id"] == 17
                return {"job_id": 54, "job_uuid": str(UUID(int=54)), "status": "completed",
                        "table": self.detail, "publication_status": "ready", "published_version": 4}
            if path.endswith(f"/data-tables/{native_id}"):
                return {"table": self.detail, "columns": [], "rows": [], "sources": [],
                        "rows_limit": 12, "rows_offset": 0}
            return await super()._request(method, path, json_data=json_data, params=params, headers=headers)

    server = TablesServer()
    adapter = ServerStudioAdapter(lambda ref: server)
    caps = await adapter.native_capabilities(SERVER, route, (source,), "r1")
    assert caps["generate_data_table"].available
    req = replace(
        server_request(), output_kind="data_table", source_snapshot=(source,), processing_route=route,
    )
    studio = ResearchStudioService(None, adapter)

    def lost_after_write():
        raise RuntimeError("lost link response")

    server.after_write = lost_after_write
    with pytest.raises(RuntimeError, match="lost link"):
        await studio.generate(req, lambda _: None, asyncio.Event())
    saved = await studio.retry(req.workspace, req.operation_id)
    assert len(server.launches) == len(server.writes) == 1
    assert saved.owner_id == native_id and saved.owner_version == 4
    [link] = server.saved.values()
    assert not link["content"]
    assert link["producer_metadata"]["research_native_owner"]["owner_id"] == native_id
    assert (await adapter.resolve_output(saved))["table"]["uuid"] == native_id
    server.detail["version"] += 1
    assert await adapter.resolve_output(saved) is None
