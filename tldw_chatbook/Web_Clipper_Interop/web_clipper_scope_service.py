"""Source-aware routing for remote-owned web-clipper capabilities."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class WebClipperBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class WebClipperScopeService:
    """Route web-clipper actions through a source-aware seam.

    Browser clip capture is server-owned for this parity row. Local mode is an
    explicit unavailable state, not a silent fallback to local note/media writes.
    """

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: WebClipperBackend | str | None) -> WebClipperBackend:
        if mode is None:
            return WebClipperBackend.SERVER
        if isinstance(mode, WebClipperBackend):
            return mode
        try:
            return WebClipperBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid web-clipper backend: {mode}") from exc

    def _require_server_service(self, mode: WebClipperBackend) -> Any:
        if mode == WebClipperBackend.LOCAL:
            raise ValueError("Web clipper is a server-only capability; switch to server mode to capture clips.")
        if self.server_service is None:
            raise ValueError("Server web-clipper backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _action_id(action: str) -> str:
        return f"web_clipper.{action}.server"

    @staticmethod
    def _normalize_response(mode: WebClipperBackend, result: Any) -> Any:
        if not isinstance(result, dict):
            return result
        payload = dict(result)
        payload.setdefault("backend", mode.value)
        clip_id = payload.get("clip_id")
        enrichment_type = payload.get("enrichment_type")
        if clip_id is not None and enrichment_type is not None:
            payload.setdefault("record_id", f"{mode.value}:web_clip_enrichment:{clip_id}:{enrichment_type}")
        elif clip_id is not None:
            payload.setdefault("record_id", f"{mode.value}:web_clip:{clip_id}")
        return payload

    async def _call(
        self,
        *,
        mode: WebClipperBackend | str | None,
        action: str,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id(action))
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(normalized_mode, result)

    async def save_clip(self, *, mode: WebClipperBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action="capture",
            method_name="save_clip",
            kwargs=kwargs,
        )

    async def get_status(self, clip_id: str, *, mode: WebClipperBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action="status",
            method_name="get_status",
            args=(clip_id,),
        )

    async def persist_enrichment(
        self,
        *,
        mode: WebClipperBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action="capture",
            method_name="persist_enrichment",
            kwargs=kwargs,
        )
