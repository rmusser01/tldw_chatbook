"""Local standalone Research Sessions service."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.DB.Research_DB import ResearchDatabase

from .research_models import ResearchArtifact, ResearchRun
from .research_normalizers import normalize_research_artifact, normalize_research_run


class LocalResearchService:
    """Local-first research session lifecycle without server execution."""

    def __init__(self, db: ResearchDatabase):
        self.db = db

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
        return normalize_research_run(
            "local",
            self.db.create_run(
                query=query,
                source_policy=source_policy,
                autonomy_mode=autonomy_mode,
                limits_json=limits_json,
                provider_overrides=provider_overrides,
                chat_handoff=chat_handoff,
                follow_up=follow_up,
            ),
        )

    async def list_runs(self, *, limit: int = 25, **_: Any) -> list[ResearchRun]:
        return [
            normalize_research_run("local", record)
            for record in self.db.list_runs(limit=limit)
        ]

    async def get_run(self, run_id: str) -> ResearchRun:
        return normalize_research_run("local", self.db.get_run(run_id))

    async def resume_run(self, run_id: str) -> ResearchRun:
        return normalize_research_run(
            "local",
            self.db.update_run_state(
                run_id,
                status="running",
                control_state="running",
                progress_message="Local research session resumed.",
            ),
        )

    async def pause_run(self, run_id: str) -> ResearchRun:
        return normalize_research_run(
            "local",
            self.db.update_run_state(
                run_id,
                control_state="paused",
                progress_message="Local research session paused.",
            ),
        )

    async def cancel_run(self, run_id: str) -> ResearchRun:
        return normalize_research_run(
            "local",
            self.db.update_run_state(
                run_id,
                status="cancelled",
                control_state="cancelled",
                progress_message="Local research session cancelled.",
            ),
        )

    async def save_artifact(
        self,
        run_id: str,
        *,
        artifact_name: str,
        content_type: str,
        content: Any,
        phase: str | None = None,
        job_id: str | None = None,
    ) -> ResearchArtifact:
        return normalize_research_artifact(
            "local",
            self.db.save_artifact(
                run_id,
                artifact_name=artifact_name,
                content_type=content_type,
                content=content,
                phase=phase,
                job_id=job_id,
            ),
        )

    async def get_artifact(self, run_id: str, artifact_name: str) -> ResearchArtifact:
        return normalize_research_artifact(
            "local",
            self.db.get_artifact(run_id, artifact_name),
        )

    async def get_bundle(self, run_id: str) -> dict[str, Any]:
        return self.db.get_bundle(run_id)
