"""Verify the primary Studio request and receipt shapes over the HTTP client."""

from dataclasses import replace
import json

import httpx
import pytest

from Tests.Research_Workspace.test_server_studio import server_request
from Tests.Research_Workspace.test_studio_models import SERVER
from tldw_chatbook.Research_Workspace.contracts import ResearchCapability
from tldw_chatbook.Research_Workspace.server_studio import ServerStudioAdapter
from tldw_chatbook.Research_Workspace.studio_generation import StudioGeneratedOutput
from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_server_studio_writes_exact_workspace_and_uses_only_owner_receipts():
    records = {}
    writes = []
    path = f"/api/v1/workspaces/{SERVER.workspace_id}/artifacts"

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/study-outputs"):
            return httpx.Response(404, json={"detail": "Not Found"})
        if request.method == "GET":
            assert request.url.path == path
            return httpx.Response(200, json=list(records.values()))
        assert request.headers["X-API-KEY"] == "studio-contract-fixture"
        fields = json.loads(request.content)
        writes.append((request.method, request.url.path, fields))
        if request.method == "POST":
            assert request.url.path == path
            artifact_id = fields["id"]
            assert artifact_id not in records
            records[artifact_id] = {
                **fields,
                "workspace_id": SERVER.workspace_id,
                "artifact_version_id": f"{artifact_id}:v1",
                "version": 1,
                "review_state": "draft",
            }
            return httpx.Response(201, json=records[artifact_id])
        artifact_id = request.url.path.rsplit("/", 1)[-1]
        assert request.method == "PUT" and request.url.path == f"{path}/{artifact_id}"
        row = records[artifact_id]
        assert row["version"] == fields["version"]
        row.update(
            fields,
            version=fields["version"] + 1,
            artifact_version_id=f"{artifact_id}:v2",
        )
        return httpx.Response(200, json=row)

    transport = httpx.MockTransport(respond)
    address = "https://studio-contract.invalid"
    client = TLDWAPIClient(address, token="studio-contract-fixture")

    def qualified_connection(ref):
        if ref != SERVER:
            raise ValueError("The captured Server identity changed")
        return client

    async with httpx.AsyncClient(
        base_url=address,
        transport=transport,
        headers={"X-API-KEY": "studio-contract-fixture"},
    ) as http:
        client._client = http
        studio = ServerStudioAdapter(qualified_connection)
        generated = StudioGeneratedOutput("complete", text="An owner-backed result")
        for kind in ("summary", "report", "compare_sources"):
            sources = server_request().source_snapshot
            if kind == "compare_sources":
                sources = (
                    *sources,
                    replace(
                        sources[0], source_id="second-source", catalog_item_id="37"
                    ),
                )
            request = replace(
                server_request(),
                output_kind=kind,
                operation_id=f"server-{kind}",
                source_snapshot=sources,
            )
            output = await studio.save(request, generated)
            assert (
                output.owner_version == 1
                and (await studio.resolve_output(output))["content"] == generated.text
            )
            assert (
                await ServerStudioAdapter(qualified_connection).save(request, generated)
                == output
            )
            assert writes[-1][2]["artifact_type"] == kind
            assert (
                writes[-1][2]["producer_metadata"]["research_generation"][
                    "operation_id"
                ]
                == f"server-{kind}"
            )

        assert len(writes) == 3
        report = next(
            output
            for output in (await studio.list_outputs(SERVER)).items
            if output.output_kind == "report"
        )
        await studio.edit_output(report, "Reviewed content")
        assert writes[-1] == (
            "PUT",
            f"{path}/{report.owner_id}",
            {"version": 1, "content": "Reviewed content", "review_state": "draft"},
        )
        with pytest.raises(ValueError, match="identity changed"):
            await studio.list_outputs(replace(SERVER, principal_id="another-account"))
        capability = ResearchCapability(
            True, "available", "Server Chat is ready.", "server"
        )
        available = studio.capabilities(capability, writable=True)
        assert available["save_workspace_artifact"].available
        assert not await studio.study_available(SERVER)
        for key in ("create_deck", "create_quiz"):
            assert not available[key].available
        assert len(writes) == 4
