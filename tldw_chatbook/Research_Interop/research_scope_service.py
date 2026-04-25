"""Mode-aware routing for the research session/run parity seam."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from .research_normalizers import normalize_research_record


class ResearchBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class ResearchScopeService:
    """Route research operations to local or server backends with policy enforcement."""

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
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

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _action_id(resource: str, action: str, mode: ResearchBackend) -> str:
        return f"research.{resource}.{action}.{mode.value}"

    async def _call_service(self, service: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(service, method_name)
        signature = inspect.signature(method)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return await self._maybe_await(method(*args, **kwargs))

    @staticmethod
    def _normalize_result(mode: ResearchBackend, kind: str, value: Any) -> Any:
        if isinstance(value, list):
            return [normalize_research_record(mode.value, kind, item) for item in value]
        if isinstance(value, dict):
            return normalize_research_record(mode.value, kind, value)
        return value

    async def list_sessions(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("sessions", "list", normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode),
            "list_sessions",
            limit=limit,
            offset=offset,
            status=status,
        )
        return self._normalize_result(normalized_mode, "session", result)

    async def create_session(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        title: str,
        query: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("sessions", "create", normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode),
            "create_session",
            title=title,
            query=query,
            **kwargs,
        )
        return self._normalize_result(normalized_mode, "session", result)

    async def get_session(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        session_id: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("sessions", "detail", normalized_mode))
        result = await self._call_service(self._service_for_mode(normalized_mode), "get_session", session_id)
        return self._normalize_result(normalized_mode, "session", result)

    async def update_session(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        session_id: str,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("sessions", "update", normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode),
            "update_session",
            session_id,
            expected_version=expected_version,
            **kwargs,
        )
        return self._normalize_result(normalized_mode, "session", result)

    async def delete_session(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        session_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("sessions", "delete", normalized_mode))
        return bool(
            await self._call_service(
                self._service_for_mode(normalized_mode),
                "delete_session",
                session_id,
                expected_version=expected_version,
            )
        )

    async def launch_run(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        query: str | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ResearchBackend.SERVER and query is None:
            raise ValueError("query is required for server research runs")
        self._enforce_policy(self._action_id("runs", "launch", normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode),
            "launch_run",
            session_id=session_id,
            query=query,
            **kwargs,
        )
        return self._normalize_result(normalized_mode, "run", result)

    async def list_runs(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
        session_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "list", normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode),
            "list_runs",
            limit=limit,
            offset=offset,
            session_id=session_id,
            status=status,
        )
        return self._normalize_result(normalized_mode, "run", result)

    async def get_run(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "detail", normalized_mode))
        result = await self._call_service(self._service_for_mode(normalized_mode), "get_run", run_id)
        return self._normalize_result(normalized_mode, "run", result)

    async def pause_run(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "update", normalized_mode))
        result = await self._call_service(self._service_for_mode(normalized_mode), "pause_run", run_id)
        return self._normalize_result(normalized_mode, "run", result)

    async def resume_run(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "update", normalized_mode))
        result = await self._call_service(self._service_for_mode(normalized_mode), "resume_run", run_id)
        return self._normalize_result(normalized_mode, "run", result)

    async def cancel_run(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "update", normalized_mode))
        result = await self._call_service(self._service_for_mode(normalized_mode), "cancel_run", run_id)
        return self._normalize_result(normalized_mode, "run", result)

    async def delete_run(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "delete", normalized_mode))
        return bool(
            await self._call_service(
                self._service_for_mode(normalized_mode),
                "delete_run",
                run_id,
                expected_version=expected_version,
            )
        )

    async def observe_run_events(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
        after_id: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "observe", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        method_name = "observe_run_events" if hasattr(service, "observe_run_events") else "list_run_events"
        result = getattr(service, method_name)(run_id, after_id=after_id)
        if inspect.isasyncgen(result):
            return [item async for item in result]
        return list(await self._maybe_await(result))

    async def get_bundle(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "detail", normalized_mode))
        return await self._call_service(self._service_for_mode(normalized_mode), "get_bundle", run_id)

    async def get_artifact(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
        artifact_name: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "detail", normalized_mode))
        return await self._call_service(
            self._service_for_mode(normalized_mode),
            "get_artifact",
            run_id,
            artifact_name,
        )
