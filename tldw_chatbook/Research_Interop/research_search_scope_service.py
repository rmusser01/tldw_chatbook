"""Mode-aware research search routing."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class ResearchSearchBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class ResearchSearchScopeService:
    """Route research search actions while keeping server-provider calls explicit."""

    def __init__(self, *, server_service: Any, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: ResearchSearchBackend | str | None) -> ResearchSearchBackend:
        if mode is None:
            return ResearchSearchBackend.LOCAL
        if isinstance(mode, ResearchSearchBackend):
            return mode
        try:
            return ResearchSearchBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid research search backend: {mode}") from exc

    def _require_server_service(self, mode: ResearchSearchBackend | str | None, feature_name: str) -> Any:
        if self._normalize_mode(mode) != ResearchSearchBackend.SERVER:
            raise ValueError(f"{feature_name} is server-only in this Chatbook parity slice.")
        if self.server_service is None:
            raise ValueError("Server research search service is unavailable.")
        return self.server_service

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is not None:
            self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _provider_action_id(action: str, mode: ResearchSearchBackend) -> str:
        return f"research.search.providers.{action}.{mode.value}"

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def websearch(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode, "Research websearch")
        self._enforce_policy(self._provider_action_id("launch", normalized_mode))
        result = dict(await self._maybe_await(service.websearch(**payload)) or {})
        result.setdefault("backend", "server")
        result.setdefault("entity_kind", "research_websearch")
        return result

    async def paper_search(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        endpoint: str,
        **params: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode, "Paper search")
        self._enforce_policy(self._provider_action_id("launch", normalized_mode))
        result = dict(await self._maybe_await(service.paper_search(endpoint=endpoint, **params)) or {})
        result.setdefault("backend", "server")
        result.setdefault("entity_kind", "paper_search")
        return result

    async def paper_detail(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        endpoint: str,
        **params: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode, "Paper search detail")
        self._enforce_policy(self._provider_action_id("observe", normalized_mode))
        result = dict(await self._maybe_await(service.paper_detail(endpoint=endpoint, **params)) or {})
        result.setdefault("backend", "server")
        result.setdefault("entity_kind", "paper_search_detail")
        return result

    async def paper_ingest(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode, "Paper search ingest")
        self._enforce_policy(self._provider_action_id("launch", normalized_mode))
        result = dict(
            await self._maybe_await(service.paper_ingest(endpoint=endpoint, payload=payload, **params)) or {}
        )
        result.setdefault("backend", "server")
        result.setdefault("entity_kind", "paper_search_ingest")
        return result
