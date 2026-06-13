"""Runtime-policy-aware server outputs scope seam."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Mapping


class OutputBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class ServerOutputsScopeService:
    """Expose server outputs operations while keeping local unavailability explicit."""

    def __init__(self, server_service: Any, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: OutputBackend | str | None) -> OutputBackend:
        if mode is None:
            return OutputBackend.LOCAL
        if isinstance(mode, OutputBackend):
            return mode
        try:
            return OutputBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid outputs backend: {mode}") from exc

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _require_server_mode(self, mode: OutputBackend | str | None) -> None:
        if self._normalize_mode(mode) != OutputBackend.SERVER:
            raise ValueError("Server outputs require server mode.")

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is not None:
            self.policy_enforcer.require_allowed(action_id=action_id)

    def _require_service(self) -> Any:
        if self.server_service is None:
            raise ValueError("Server outputs service is unavailable.")
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
    def _server_id(prefix: str, value: Any) -> str:
        raw_value = str(value or "").strip()
        if raw_value.startswith(f"server:{prefix}:"):
            return raw_value
        return f"server:{prefix}:{raw_value}"

    def _normalize_template(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        source_id = normalized.get("source_id", normalized.get("id"))
        normalized["source_id"] = source_id
        normalized["id"] = self._server_id("output_template", source_id)
        normalized["backend"] = "server"
        normalized["entity_kind"] = "output_template"
        return normalized

    def _normalize_output(self, payload: Any, *, entity_kind: str = "output_artifact") -> dict[str, Any]:
        normalized = self._as_dict(payload)
        source_id = normalized.get("source_id", normalized.get("id"))
        normalized["source_id"] = source_id
        normalized["id"] = self._server_id("output", source_id)
        normalized["backend"] = "server"
        normalized["entity_kind"] = entity_kind
        return normalized

    def _normalize_template_list(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        normalized["items"] = [self._normalize_template(item) for item in normalized.get("items", [])]
        normalized["backend"] = "server"
        normalized["entity_kind"] = "output_template_list"
        return normalized

    def _normalize_output_list(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        normalized["items"] = [self._normalize_output(item) for item in normalized.get("items", [])]
        normalized["backend"] = "server"
        normalized["entity_kind"] = "output_artifact_list"
        return normalized

    def _normalize_simple(self, payload: Any, *, entity_kind: str) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        normalized["backend"] = "server"
        normalized["entity_kind"] = entity_kind
        return normalized

    async def list_output_templates(
        self,
        *,
        mode: OutputBackend | str | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.templates.list.server")
        result = await self._maybe_await(self._require_service().list_output_templates(q=q, limit=limit, offset=offset))
        return self._normalize_template_list(result)

    async def create_output_template(
        self,
        *,
        mode: OutputBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.templates.create.server")
        result = await self._maybe_await(self._require_service().create_output_template(**payload))
        return self._normalize_template(result)

    async def get_output_template(
        self,
        *,
        mode: OutputBackend | str | None = None,
        template_id: int,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.templates.detail.server")
        result = await self._maybe_await(self._require_service().get_output_template(template_id))
        return self._normalize_template(result)

    async def update_output_template(
        self,
        *,
        mode: OutputBackend | str | None = None,
        template_id: int,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.templates.update.server")
        result = await self._maybe_await(self._require_service().update_output_template(template_id, **payload))
        return self._normalize_template(result)

    async def delete_output_template(
        self,
        *,
        mode: OutputBackend | str | None = None,
        template_id: int,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.templates.delete.server")
        result = await self._maybe_await(self._require_service().delete_output_template(template_id))
        normalized = self._normalize_simple(result, entity_kind="output_template_delete")
        normalized["template_id"] = template_id
        return normalized

    async def preview_output_template(
        self,
        *,
        mode: OutputBackend | str | None = None,
        template_id: int,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.templates.detail.server")
        result = await self._maybe_await(self._require_service().preview_output_template(template_id, **payload))
        normalized = self._normalize_simple(result, entity_kind="output_template_preview")
        normalized["template_id"] = template_id
        return normalized

    async def list_outputs(
        self,
        *,
        mode: OutputBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.artifacts.list.server")
        result = await self._maybe_await(self._require_service().list_outputs(**payload))
        return self._normalize_output_list(result)

    async def list_deleted_outputs(
        self,
        *,
        mode: OutputBackend | str | None = None,
        page: int = 1,
        size: int = 50,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.artifacts.list.server")
        result = await self._maybe_await(self._require_service().list_deleted_outputs(page=page, size=size))
        return self._normalize_output_list(result)

    async def create_output(
        self,
        *,
        mode: OutputBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.render_jobs.launch.server")
        result = await self._maybe_await(self._require_service().create_output(**payload))
        return self._normalize_output(result, entity_kind="output_render_result")

    async def get_output(
        self,
        *,
        mode: OutputBackend | str | None = None,
        output_id: int,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.artifacts.detail.server")
        result = await self._maybe_await(self._require_service().get_output(output_id))
        return self._normalize_output(result)

    async def update_output(
        self,
        *,
        mode: OutputBackend | str | None = None,
        output_id: int,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.artifacts.update.server")
        result = await self._maybe_await(self._require_service().update_output(output_id, **payload))
        return self._normalize_output(result)

    async def delete_output(
        self,
        *,
        mode: OutputBackend | str | None = None,
        output_id: int,
        hard: bool = False,
        delete_file: bool = False,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.artifacts.delete.server")
        result = await self._maybe_await(
            self._require_service().delete_output(output_id, hard=hard, delete_file=delete_file)
        )
        normalized = self._normalize_simple(result, entity_kind="output_delete")
        normalized["output_id"] = output_id
        return normalized

    async def download_output(
        self,
        *,
        mode: OutputBackend | str | None = None,
        output_id: int,
    ) -> bytes:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.artifacts.detail.server")
        return await self._maybe_await(self._require_service().download_output(output_id))

    async def download_output_by_name(
        self,
        *,
        mode: OutputBackend | str | None = None,
        title: str,
        format: str | None = None,
    ) -> bytes:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.artifacts.detail.server")
        return await self._maybe_await(self._require_service().download_output_by_name(title, format=format))

    async def purge_outputs(
        self,
        *,
        mode: OutputBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("outputs.artifacts.delete.server")
        result = await self._maybe_await(self._require_service().purge_outputs(**payload))
        return self._normalize_simple(result, entity_kind="output_purge")
