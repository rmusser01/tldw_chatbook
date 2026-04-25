"""Mode-aware routing for output templates and artifacts."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class OutputsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_BACKEND_UNAVAILABLE_CAPABILITY = {
    "operation_id": "outputs.local_backend.local",
    "source": "local",
    "supported": False,
    "reason_code": "local_backend_unavailable",
    "user_message": "Local managed output templates, artifacts, and render jobs are not wired in the current Chatbook client.",
    "affected_action_ids": [
        "outputs.templates.list.local",
        "outputs.templates.detail.local",
        "outputs.templates.create.local",
        "outputs.templates.update.local",
        "outputs.templates.delete.local",
        "outputs.artifacts.list.local",
        "outputs.artifacts.detail.local",
        "outputs.artifacts.create.local",
        "outputs.artifacts.update.local",
        "outputs.artifacts.delete.local",
        "outputs.render_jobs.launch.local",
        "outputs.render_jobs.list.local",
        "outputs.render_jobs.detail.local",
        "outputs.render_jobs.observe.local",
    ],
}

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "outputs.render_jobs.observe.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server output API supports synchronous template preview and artifact creation, but not first-class render-job listing, detail, or observation.",
        "affected_action_ids": [
            "outputs.render_jobs.list.server",
            "outputs.render_jobs.detail.server",
            "outputs.render_jobs.observe.server",
        ],
    },
]


class OutputsScopeService:
    """Route output/template actions to the selected backend and normalize records."""

    def __init__(self, *, local_service: Any = None, server_service: Any = None, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: OutputsBackend | str | None) -> OutputsBackend:
        if mode is None:
            return OutputsBackend.SERVER
        if isinstance(mode, OutputsBackend):
            return mode
        try:
            return OutputsBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid outputs backend: {mode}") from exc

    def _service_for_mode(self, mode: OutputsBackend) -> Any:
        if mode == OutputsBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local outputs backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server outputs backend is unavailable.")
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
    def _action_id(resource: str, action: str, mode: OutputsBackend) -> str:
        return f"outputs.{resource}.{action}.{mode.value}"

    @staticmethod
    def _normalize_record(mode: OutputsBackend, kind: str, value: dict[str, Any]) -> dict[str, Any]:
        payload = dict(value or {})
        payload.setdefault("backend", mode.value)
        source_id = payload.get("id") or payload.get(f"{kind}_id")
        if source_id is not None:
            payload.setdefault("record_id", f"{mode.value}:output_{kind}:{source_id}")
        return payload

    def _normalize_response(self, mode: OutputsBackend, kind: str | None, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault("backend", mode.value)
        if kind and "items" in payload and isinstance(payload["items"], list):
            payload["items"] = [
                self._normalize_record(mode, kind, item) if isinstance(item, dict) else item
                for item in payload["items"]
            ]
            return payload
        if kind:
            return self._normalize_record(mode, kind, payload)
        return payload

    def list_unsupported_capabilities(
        self,
        *,
        mode: OutputsBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == OutputsBackend.LOCAL:
            if self.local_service is None:
                return [dict(_LOCAL_BACKEND_UNAVAILABLE_CAPABILITY)]
            return []
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def _call(
        self,
        *,
        mode: OutputsBackend | str | None,
        resource: str,
        action: str,
        method_name: str,
        normalize_kind: str | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(resource, action, normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(normalized_mode, normalize_kind, result)

    async def list_templates(self, *, mode: OutputsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="templates",
            action="list",
            method_name="list_templates",
            normalize_kind="template",
            kwargs=kwargs,
        )

    async def get_template(self, *, mode: OutputsBackend | str | None = None, template_id: int) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="templates",
            action="detail",
            method_name="get_template",
            normalize_kind="template",
            args=(template_id,),
        )

    async def create_template(self, *, mode: OutputsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="templates",
            action="create",
            method_name="create_template",
            normalize_kind="template",
            kwargs=kwargs,
        )

    async def update_template(
        self,
        template_id: int,
        *,
        mode: OutputsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="templates",
            action="update",
            method_name="update_template",
            normalize_kind="template",
            args=(template_id,),
            kwargs=kwargs,
        )

    async def delete_template(self, template_id: int, *, mode: OutputsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="templates",
            action="delete",
            method_name="delete_template",
            args=(template_id,),
        )

    async def preview_template(
        self,
        *,
        mode: OutputsBackend | str | None = None,
        template_id: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="render_jobs",
            action="launch",
            method_name="preview_template",
            args=(template_id,),
            kwargs=kwargs,
        )

    async def list_artifacts(self, *, mode: OutputsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="artifacts",
            action="list",
            method_name="list_artifacts",
            normalize_kind="artifact",
            kwargs=kwargs,
        )

    async def list_deleted_artifacts(
        self,
        *,
        mode: OutputsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="artifacts",
            action="list",
            method_name="list_deleted_artifacts",
            normalize_kind="artifact",
            kwargs=kwargs,
        )

    async def get_artifact(self, *, mode: OutputsBackend | str | None = None, output_id: int) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="artifacts",
            action="detail",
            method_name="get_artifact",
            normalize_kind="artifact",
            args=(output_id,),
        )

    async def create_artifact(self, *, mode: OutputsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="artifacts",
            action="create",
            method_name="create_artifact",
            normalize_kind="artifact",
            kwargs=kwargs,
        )

    async def update_artifact(
        self,
        output_id: int,
        *,
        mode: OutputsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="artifacts",
            action="update",
            method_name="update_artifact",
            normalize_kind="artifact",
            args=(output_id,),
            kwargs=kwargs,
        )

    async def delete_artifact(
        self,
        output_id: int,
        *,
        mode: OutputsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="artifacts",
            action="delete",
            method_name="delete_artifact",
            args=(output_id,),
            kwargs=kwargs,
        )

    async def purge_artifacts(self, *, mode: OutputsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="artifacts",
            action="delete",
            method_name="purge_artifacts",
            kwargs=kwargs,
        )
