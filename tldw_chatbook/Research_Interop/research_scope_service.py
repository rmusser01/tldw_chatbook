"""Source-aware router for Research Sessions operations."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class ResearchBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class ResearchScopeService:
    """Route research actions to the selected backend without source fallback."""

    _ACTION_IDS: dict[tuple[str, str], str] = {
        ("runs", "list"): "research.runs.list",
        ("runs", "detail"): "research.runs.detail",
        ("runs", "create"): "research.runs.create",
        ("runs", "update"): "research.runs.update",
        ("runs", "launch"): "research.runs.launch",
        ("runs", "observe"): "research.runs.observe",
        ("sessions", "create"): "research.sessions.create",
        ("sessions", "list"): "research.sessions.list",
        ("sessions", "detail"): "research.sessions.detail",
    }

    def __init__(
        self,
        *,
        local_service: Any,
        server_service: Any,
        policy_enforcer: Any | None = None,
    ):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: ResearchBackend | str | None) -> ResearchBackend:
        if mode is None:
            return ResearchBackend.LOCAL
        if isinstance(mode, ResearchBackend):
            return mode
        try:
            return ResearchBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid research backend: {mode}") from exc

    def _service_for_mode(self, mode: ResearchBackend) -> Any:
        if mode == ResearchBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local research backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server research backend is unavailable.")
        return self.server_service

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, mode: ResearchBackend, *, resource: str, action: str) -> None:
        if self.policy_enforcer is None:
            return
        action_prefix = self._ACTION_IDS.get((resource, action))
        if action_prefix is None:
            return
        self.policy_enforcer.require_allowed(action_id=f"{action_prefix}.{mode.value}")

    async def _call(
        self,
        *,
        mode: ResearchBackend | str | None,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        method = getattr(self._service_for_mode(normalized_mode), method_name)
        return await self._maybe_await(method(*args, **(kwargs or {})))

    async def create_run(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="runs", action="create")
        return await self._call(mode=normalized_mode, method_name="create_run", kwargs=kwargs)

    async def list_runs(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        limit: int = 25,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="runs", action="list")
        return await self._call(
            mode=normalized_mode,
            method_name="list_runs",
            kwargs={"limit": limit},
        )

    async def get_run(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="runs", action="detail")
        return await self._call(mode=normalized_mode, method_name="get_run", args=(run_id,))

    async def pause_run(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="runs", action="update")
        return await self._call(mode=normalized_mode, method_name="pause_run", args=(run_id,))

    async def resume_run(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="runs", action="launch")
        return await self._call(mode=normalized_mode, method_name="resume_run", args=(run_id,))

    async def cancel_run(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="runs", action="update")
        return await self._call(mode=normalized_mode, method_name="cancel_run", args=(run_id,))

    async def get_artifact(
        self,
        run_id: str,
        artifact_name: str,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="runs", action="observe")
        return await self._call(
            mode=normalized_mode,
            method_name="get_artifact",
            args=(run_id, artifact_name),
        )

    async def get_bundle(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="runs", action="observe")
        return await self._call(mode=normalized_mode, method_name="get_bundle", args=(run_id,))

    async def patch_and_approve_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        mode: ResearchBackend | str | None = None,
        patch_payload: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ResearchBackend.LOCAL:
            raise ValueError("Local research checkpoints are not available in this slice.")
        self._enforce_policy(normalized_mode, resource="runs", action="update")
        return await self._call(
            mode=normalized_mode,
            method_name="patch_and_approve_checkpoint",
            args=(run_id, checkpoint_id, patch_payload),
        )

    async def stream_run_events(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
        after_id: int = 0,
    ):
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="runs", action="observe")
        method = getattr(self._service_for_mode(normalized_mode), "stream_run_events")
        async for event in method(run_id, after_id=after_id):
            yield event
