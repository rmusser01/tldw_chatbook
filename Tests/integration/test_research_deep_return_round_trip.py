"""The bound Server launch and bundle contract survives a client restart."""

import json

import httpx
import pytest

from tldw_chatbook.Research_Interop.research_scope_service import ResearchScopeService
from tldw_chatbook.Research_Interop.server_research_service import ServerResearchService
from tldw_chatbook.Research_Workspace.chat_models import ResearchSourceVersionRef
from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef
from tldw_chatbook.Research_Workspace.deep_research_bridge import (
    DeepResearchBridge,
    ResearchReturnMismatch,
)
from tldw_chatbook.Research_Workspace.deep_research_context import ResearchLaunchContext
from tldw_chatbook.Research_Workspace.overlay_store import (
    ResearchPresentationOverlayStore,
)
from tldw_chatbook.tldw_api import TLDWAPIClient
from tldw_chatbook.tldw_api.exceptions import APIConnectionError
from tldw_chatbook.tldw_api.workspace_research_runs import WorkspaceResearchRunsClient


@pytest.mark.asyncio
async def test_server_launch_replays_and_validates_native_bundle_origin(tmp_path):
    ref = QualifiedWorkspaceRef("server", "study", "private-profile", "jwt-sub:4")
    source = ResearchSourceVersionRef(ref, "selected", "43", "document", 2, 5, ("fts",))
    context = ResearchLaunchContext(
        ref, "exact-launch", (source,), "Where is the evidence?", "2026-11-24T20:00:00Z"
    )
    expected_origin = {
        "workspace_id": "study",
        "owner_user_id": "4",
        "launch": context.server_payload(),
    }
    run = {
        "id": "rs_exact",
        "phase": "packaging",
        "status": "completed",
        "workspace_launch": expected_origin,
    }
    writes = []
    returned_id = "rs_exact"
    launch_path = "/api/v1/workspaces/study/research-runs"
    run_path = "/api/v1/research/runs/rs_exact"

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-TLDW-Expected-User-ID"] == "4"
        assert request.headers["X-API-KEY"] == "fixture-account"
        path = request.url.path
        if path == launch_path:
            if request.method == "GET":
                return httpx.Response(
                    200, json={"contract": "research-workspace-runs.v1"}
                )
            assert request.method == "POST"
            assert b"private-profile" not in request.content
            writes.append(json.loads(request.content))
            if len(writes) == 1:
                raise httpx.ReadTimeout(
                    "Fixture lost the first launch acknowledgement", request=request
                )
            return httpx.Response(200, json=run)
        if path == run_path:
            assert request.method == "GET"
            return httpx.Response(200, json=run)
        assert path == f"{run_path}/bundle"
        return httpx.Response(
            200,
            json={
                "run_id": returned_id,
                "workspace_launch": expected_origin,
                "question": context.query,
                "report_markdown": "# Persisted report",
                "source_inventory": [{"id": "source-1"}],
            },
        )

    host = "https://research-fixture.invalid"
    async with httpx.AsyncClient(
        base_url=host,
        transport=httpx.MockTransport(respond),
        headers={"X-API-KEY": "fixture-account"},
    ) as transport:

        def connect():
            root = TLDWAPIClient(host, token="fixture-account")
            root._client = transport
            workspace = WorkspaceResearchRunsClient(
                root, ref.workspace_id, ref.principal_id
            )
            scope = ResearchScopeService(
                local_service=None,
                server_service=ServerResearchService(root, workspace_runs=workspace),
            )
            return workspace, DeepResearchBridge(
                scope,
                ResearchPresentationOverlayStore(tmp_path / "private" / "overlay.json"),
            )

        workspace, original = connect()
        assert await workspace.supported()
        with pytest.raises(APIConnectionError):
            await original.launch_server(context)
        _, resumed = connect()
        target = await resumed.launch_server(context)
        assert target == context.target("rs_exact")
        assert writes == [context.server_payload(), context.server_payload()]
        _, reopened = connect()
        preview = await reopened.preview_return(target)
        assert preview.report == "# Persisted report" and preview.source_count == 1
        returned_id = "rs_other"
        with pytest.raises(ResearchReturnMismatch, match="different run"):
            await reopened.preview_return(target)
