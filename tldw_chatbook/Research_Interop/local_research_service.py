"""Local standalone Research Sessions service."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from tldw_chatbook.DB.Research_DB import ResearchDatabase

from .research_models import ResearchArtifact, ResearchRun
from .research_normalizers import normalize_research_artifact, normalize_research_run


class LocalResearchService:
    """Local-first research session lifecycle without server execution."""

    def __init__(
        self,
        db: ResearchDatabase,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ):
        self.db = db
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def configure_notification_dispatch(
        self,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ) -> None:
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def _dispatch_local_notification(
        self,
        *,
        title: str,
        message: str,
        severity: str = "info",
        run: ResearchRun,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        dispatcher = getattr(self, "notification_dispatch_service", None)
        dispatch = getattr(dispatcher, "dispatch", None)
        if not callable(dispatch):
            return None
        try:
            return dispatch(
                app=getattr(self, "notification_app", None),
                category="research",
                title=title,
                message=message,
                severity=severity,
                source_backend="local",
                source_entity_id=run.id,
                source_entity_kind="research_run",
                payload={
                    "run_id": run.id,
                    "query": run.query,
                    "status": run.status,
                    "phase": run.phase,
                    **(payload or {}),
                },
            )
        except Exception:
            return None

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
        run = normalize_research_run(
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
        self._dispatch_local_notification(
            title="Local research session created",
            message=f"Local research session created: {run.query}",
            run=run,
            payload={"action": "created"},
        )
        return run

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
        run = normalize_research_run(
            "local",
            self.db.update_run_state(
                run_id,
                status="cancelled",
                control_state="cancelled",
                progress_message="Local research session cancelled.",
            ),
        )
        self._dispatch_local_notification(
            title="Local research session cancelled",
            message=f"Local research session cancelled: {run.query}",
            severity="warning",
            run=run,
            payload={"action": "cancelled"},
        )
        return run

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

    async def stream_run_events(self, run_id: str, *, after_id: int = 0):
        try:
            cursor = int(after_id)
        except (TypeError, ValueError):
            cursor = 0

        run = await self.get_run(run_id)
        if cursor < 1:
            yield {
                "event": "snapshot",
                "id": "1",
                "data": {"run": asdict(run)},
            }

        bundle = await self.get_bundle(run_id)
        if cursor < 2 and bundle:
            yield {
                "event": "bundle",
                "id": "2",
                "data": {
                    "artifact_names": sorted(bundle),
                    "bundle": bundle,
                },
            }
