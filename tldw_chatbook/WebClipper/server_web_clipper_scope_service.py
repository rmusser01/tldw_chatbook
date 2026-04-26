"""Runtime-policy-aware server Web Clipper scope seam."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Mapping


class WebClipperBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class ServerWebClipperScopeService:
    """Expose server Web Clipper operations while making local unavailability explicit."""

    def __init__(self, server_service: Any, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: WebClipperBackend | str | None) -> WebClipperBackend:
        if mode is None:
            return WebClipperBackend.LOCAL
        if isinstance(mode, WebClipperBackend):
            return mode
        try:
            return WebClipperBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid Web Clipper backend: {mode}") from exc

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _require_server_mode(self, mode: WebClipperBackend | str | None) -> None:
        if self._normalize_mode(mode) != WebClipperBackend.SERVER:
            raise ValueError("Server web clipper requires server mode.")

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is not None:
            self.policy_enforcer.require_allowed(action_id=action_id)

    def _require_service(self) -> Any:
        if self.server_service is None:
            raise ValueError("Server Web Clipper service is unavailable.")
        return self.server_service

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True, mode="json")
        return dict(value)

    @staticmethod
    def _server_clip_id(clip_id: Any) -> str:
        raw_clip_id = str(clip_id or "").strip()
        if raw_clip_id.startswith("server:web_clip:"):
            return raw_clip_id
        return f"server:web_clip:{raw_clip_id}"

    @staticmethod
    def _server_note_id(note_id: Any) -> str:
        raw_note_id = str(note_id or "").strip()
        if not raw_note_id or raw_note_id.startswith("server:note:"):
            return raw_note_id
        return f"server:note:{raw_note_id}"

    def _normalize_clip_payload(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        clip_id = normalized.get("clip_id")
        normalized["id"] = self._server_clip_id(clip_id)
        normalized["backend"] = "server"
        normalized["entity_kind"] = "web_clip"
        note = normalized.get("note")
        if isinstance(note, Mapping):
            note_payload = dict(note)
            note_payload["id"] = self._server_note_id(note_payload.get("id"))
            normalized["note"] = note_payload
        return normalized

    def _normalize_enrichment_payload(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        clip_id = normalized.get("clip_id")
        enrichment_type = normalized.get("enrichment_type") or "unknown"
        normalized["id"] = f"{self._server_clip_id(clip_id)}:enrichment:{enrichment_type}"
        normalized["backend"] = "server"
        normalized["entity_kind"] = "web_clip_enrichment"
        return normalized

    async def save_clip(self, *, mode: WebClipperBackend | str | None = None, **payload: Any) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("web_clipper.capture.server")
        service = self._require_service()
        result = await self._maybe_await(service.save_clip(**payload))
        return self._normalize_clip_payload(result)

    async def get_clip_status(
        self,
        *,
        mode: WebClipperBackend | str | None = None,
        clip_id: str,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("web_clipper.status.server")
        service = self._require_service()
        result = await self._maybe_await(service.get_clip_status(clip_id))
        return self._normalize_clip_payload(result)

    async def persist_enrichment(
        self,
        *,
        mode: WebClipperBackend | str | None = None,
        clip_id: str,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("web_clipper.capture.server")
        service = self._require_service()
        request_payload = {"clip_id": clip_id, **payload}
        result = await self._maybe_await(service.persist_enrichment(clip_id, **request_payload))
        return self._normalize_enrichment_payload(result)
