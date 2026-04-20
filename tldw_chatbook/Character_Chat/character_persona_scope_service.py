"""
Mode-aware routing for local and server-backed character/persona catalog access.
"""

from __future__ import annotations

import asyncio
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

    def _call_backend(self, backend: Any, method_name: str, **kwargs: Any) -> Any:
        method = getattr(backend, method_name)
        result = method(**kwargs)
        if inspect.isawaitable(result):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(result)
            raise RuntimeError(
                "Cannot execute server character/persona operations synchronously inside an active event loop."
            )
        return result

    def list_characters(self, mode: str = "local", limit: int = 100, offset: int = 0) -> Any:
        backend = self._backend(mode)
        if backend is self.local_service and not hasattr(backend, "list_characters"):
            return self._call_backend(backend, "list_character_cards")
        return self._call_backend(backend, "list_characters")

    def list_persona_profiles(
        self,
        mode: str = "local",
        active_only: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        backend = self._backend(mode)
        if backend is self.local_service and not hasattr(backend, "list_persona_profiles"):
            return self._call_backend(backend, "list_character_cards")
        return self._call_backend(
            backend,
            "list_persona_profiles",
        )
