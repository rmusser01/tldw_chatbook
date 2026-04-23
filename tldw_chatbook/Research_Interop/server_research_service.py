"""Server-backed Research Sessions service."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from tldw_chatbook.tldw_api.research_runs_schemas import (
    ResearchCheckpointPatchApproveRequest,
    ResearchRunCreateRequest,
)

from .research_models import ResearchArtifact, ResearchRun
from .research_normalizers import normalize_research_artifact, normalize_research_run

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import TLDWAPIClient


class ServerResearchService:
    """Adapter for the current tldw_server deep research run contract."""

    def __init__(self, client: TLDWAPIClient | None):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any] | None) -> "ServerResearchService":
        from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client_from_config

        return cls(client=build_runtime_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server research operations.")
        return self.client

    async def create_run(
        self,
        *,
        query: str,
        source_policy: str = "balanced",
        autonomy_mode: str = "checkpointed",
        limits_json: dict[str, Any] | None = None,
        provider_overrides: dict[str, Any] | None = None,
        chat_handoff: dict[str, Any] | None = None,
        follow_up: dict[str, Any] | None = None,
        **_: Any,
    ) -> ResearchRun:
        response = await self._require_client().create_research_run(
            ResearchRunCreateRequest(
                query=query,
                source_policy=source_policy,
                autonomy_mode=autonomy_mode,
                limits_json=limits_json,
                provider_overrides=provider_overrides,
                chat_handoff=chat_handoff,
                follow_up=follow_up,
            )
        )
        return normalize_research_run("server", response)

    async def list_runs(self, *, limit: int = 25, **_: Any) -> list[ResearchRun]:
        return [
            normalize_research_run("server", record)
            for record in await self._require_client().list_research_runs(limit=limit)
        ]

    async def get_run(self, run_id: str) -> ResearchRun:
        return normalize_research_run(
            "server",
            await self._require_client().get_research_run(run_id),
        )

    async def pause_run(self, run_id: str) -> ResearchRun:
        return normalize_research_run(
            "server",
            await self._require_client().pause_research_run(run_id),
        )

    async def resume_run(self, run_id: str) -> ResearchRun:
        return normalize_research_run(
            "server",
            await self._require_client().resume_research_run(run_id),
        )

    async def cancel_run(self, run_id: str) -> ResearchRun:
        return normalize_research_run(
            "server",
            await self._require_client().cancel_research_run(run_id),
        )

    async def get_artifact(self, run_id: str, artifact_name: str) -> ResearchArtifact:
        response = await self._require_client().get_research_artifact(run_id, artifact_name)
        data = response.model_dump(mode="json")
        data["run_id"] = run_id
        return normalize_research_artifact("server", data)

    async def get_bundle(self, run_id: str) -> dict[str, Any]:
        return await self._require_client().get_research_bundle(run_id)

    async def patch_and_approve_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        patch_payload: dict[str, Any] | None = None,
    ) -> ResearchRun:
        return normalize_research_run(
            "server",
            await self._require_client().patch_and_approve_research_checkpoint(
                run_id,
                checkpoint_id,
                ResearchCheckpointPatchApproveRequest(patch_payload=patch_payload),
            ),
        )
