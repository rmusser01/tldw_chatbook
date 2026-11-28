"""Use the authenticated Server transport for a qualified Workspace run."""

from __future__ import annotations

from urllib.parse import quote

from .exceptions import APIResponseError
from .research_runs_schemas import ResearchRunResponse


class WorkspaceResearchRunsClient:
    """Read the launch capability and preserve the captured account on requests."""

    def __init__(self, client, workspace_id: str, principal_id: str) -> None:
        self.client = client
        self.workspace_id = workspace_id
        self.principal_id = principal_id
        self.path = f"/api/v1/workspaces/{quote(workspace_id, safe='')}/research-runs"
        subject = principal_id.removeprefix("jwt-sub:")
        self.headers = (
            {"X-TLDW-Expected-User-ID": subject} if subject != principal_id else None
        )

    async def supported(self) -> bool:
        try:
            result = await self.client._request("GET", self.path, headers=self.headers)
        except APIResponseError as exc:
            if exc.status_code in {404, 405}:
                return False
            raise
        return (
            isinstance(result, dict)
            and result.get("contract") == "research-workspace-runs.v1"
        )

    async def launch(self, payload: dict) -> ResearchRunResponse:
        try:
            raw = await self.client._request(
                "POST", self.path, json_data=payload, headers=self.headers
            )
        except APIResponseError as exc:
            if exc.status_code in {404, 405}:
                raise ValueError(
                    "This Server lacks Workspace Research Runs, or the workspace is unavailable"
                ) from None
            raise
        return ResearchRunResponse.model_validate(raw)

    async def detail(self, run_id: str) -> ResearchRunResponse:
        raw = await self.client._request(
            "GET",
            f"/api/v1/research/runs/{quote(run_id, safe='')}",
            headers=self.headers,
        )
        return ResearchRunResponse.model_validate(raw)

    async def bundle(self, run_id: str) -> dict:
        return await self.client._request(
            "GET",
            f"/api/v1/research/runs/{quote(run_id, safe='')}/bundle",
            headers=self.headers,
        )
