"""Controller for Research Sessions UI source routing."""

from __future__ import annotations

import inspect
from typing import Any, Mapping


class ResearchController:
    """Thin async controller over the source-aware research scope service."""

    def __init__(self, scope_service: Any):
        self.scope_service = scope_service
        self.current_runs: list[Any] = []
        self.selected_run: Any | None = None

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _require_scope_service(self) -> Any:
        if self.scope_service is None:
            raise ValueError("Research scope service is unavailable.")
        return self.scope_service

    async def load_runs(self, source: str, *, limit: int = 25) -> list[Any]:
        service = self._require_scope_service()
        runs = await self._maybe_await(service.list_runs(mode=source, limit=limit))
        self.current_runs = list(runs or [])
        return self.current_runs

    async def create_run(self, source: str, payload: Mapping[str, Any]) -> Any:
        service = self._require_scope_service()
        created = await self._maybe_await(service.create_run(mode=source, **dict(payload)))
        return created

    async def get_run(self, source: str, run_id: str) -> Any:
        service = self._require_scope_service()
        run = await self._maybe_await(service.get_run(run_id, mode=source))
        self.selected_run = run
        return run

    async def pause_run(self, source: str, run_id: str) -> Any:
        service = self._require_scope_service()
        return await self._maybe_await(service.pause_run(run_id, mode=source))

    async def resume_run(self, source: str, run_id: str) -> Any:
        service = self._require_scope_service()
        return await self._maybe_await(service.resume_run(run_id, mode=source))

    async def cancel_run(self, source: str, run_id: str) -> Any:
        service = self._require_scope_service()
        return await self._maybe_await(service.cancel_run(run_id, mode=source))

    async def get_bundle(self, source: str, run_id: str) -> dict[str, Any]:
        service = self._require_scope_service()
        return await self._maybe_await(service.get_bundle(run_id, mode=source))

    async def get_artifact(self, source: str, run_id: str, artifact_name: str) -> Any:
        service = self._require_scope_service()
        return await self._maybe_await(service.get_artifact(run_id, artifact_name, mode=source))

    async def patch_and_approve_checkpoint(
        self,
        source: str,
        run_id: str,
        checkpoint_id: str,
        patch_payload: Mapping[str, Any] | None = None,
    ) -> Any:
        service = self._require_scope_service()
        updated = await self._maybe_await(
            service.patch_and_approve_checkpoint(
                run_id,
                checkpoint_id,
                mode=source,
                patch_payload=dict(patch_payload or {}) or None,
            )
        )
        self.selected_run = updated
        return updated

    async def stream_run_events(self, source: str, run_id: str, *, after_id: int = 0):
        service = self._require_scope_service()
        async for event in service.stream_run_events(run_id, mode=source, after_id=after_id):
            yield event
