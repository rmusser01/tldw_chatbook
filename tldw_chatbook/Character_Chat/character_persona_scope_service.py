"""
Mode-aware routing for local and server-backed character/persona catalog access.
"""

from __future__ import annotations

import inspect
from typing import Any


class CharacterPersonaScopeService:
    """Route character and persona catalog reads to the selected backend."""

    def __init__(self, *, local_service: Any, server_service: Any):
        self.local_service = local_service
        self.server_service = server_service

    def _backend(self, mode: str):
        if mode == "server":
            return self.server_service
        return self.local_service

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def list_characters(self, mode: str = "local", limit: int = 100, offset: int = 0) -> Any:
        backend = self._backend(mode)
        if backend is self.local_service and not hasattr(backend, "list_characters"):
            return await self._maybe_await(backend.list_character_cards(limit=limit, offset=offset))
        return await self._maybe_await(backend.list_characters(limit=limit, offset=offset))

    async def list_persona_profiles(
        self,
        mode: str = "local",
        active_only: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        backend = self._backend(mode)
        if backend is self.local_service and not hasattr(backend, "list_persona_profiles"):
            raise AttributeError("Local backend does not implement list_persona_profiles().")
        return await self._maybe_await(
            backend.list_persona_profiles(
                active_only=active_only,
                include_deleted=include_deleted,
                limit=limit,
                offset=offset,
            )
        )
