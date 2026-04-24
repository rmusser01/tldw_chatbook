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

    def __init__(self, *, server_service: Any):
        self.server_service = server_service

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
        service = self._require_server_service(mode, "Research websearch")
        result = dict(await self._maybe_await(service.websearch(**payload)) or {})
        result.setdefault("backend", "server")
        result.setdefault("entity_kind", "research_websearch")
        return result
