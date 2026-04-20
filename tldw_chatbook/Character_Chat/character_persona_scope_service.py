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

    def _backend(self, mode: str | None):
        normalized_mode = "local" if mode is None else mode
        if normalized_mode not in {"local", "server"}:
            raise ValueError(f"Invalid character/persona mode: {mode!r}. Expected 'local' or 'server'.")

        if normalized_mode == "server":
            if self.server_service is None:
                raise ValueError("Server character/persona backend is unavailable.")
            return self.server_service

        if self.local_service is None:
            raise ValueError("Local character/persona backend is unavailable.")
        return self.local_service

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def list_characters(self, mode: str = "local", limit: int = 100, offset: int = 0) -> Any:
        backend = self._backend(mode)
        if mode in {None, "local"} and not hasattr(backend, "list_characters"):
            if not hasattr(backend, "list_character_cards"):
                raise ValueError("Local character backend does not provide list_characters() or list_character_cards().")
            return await self._maybe_await(backend.list_character_cards(limit=limit, offset=offset))
        if not hasattr(backend, "list_characters"):
            raise ValueError("Character backend does not provide list_characters().")
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
        if mode in {None, "local"} and not hasattr(backend, "list_persona_profiles"):
            raise ValueError("Local persona profiles are not available yet.")
        if not hasattr(backend, "list_persona_profiles"):
            raise ValueError("Character/persona backend does not provide list_persona_profiles().")
        return await self._maybe_await(
            backend.list_persona_profiles(
                active_only=active_only,
                include_deleted=include_deleted,
                limit=limit,
                offset=offset,
            )
        )
