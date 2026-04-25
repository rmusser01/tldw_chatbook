"""Mode-aware routing for chunking-template and collection admin surfaces."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from .rag_admin_normalizers import normalize_collection_record, normalize_template_record


class RAGAdminBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class RAGAdminScopeService:
    """Route retrieval-admin actions to local or server backends and normalize outputs."""

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: RAGAdminBackend | str | None) -> RAGAdminBackend:
        if mode is None:
            return RAGAdminBackend.LOCAL
        if isinstance(mode, RAGAdminBackend):
            return mode
        try:
            return RAGAdminBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid RAG admin backend: {mode}") from exc

    def _service_for_mode(self, mode: RAGAdminBackend) -> Any:
        if mode == RAGAdminBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local retrieval-admin backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server retrieval-admin backend is unavailable.")
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
    def _template_action_id(mode: RAGAdminBackend, action: str) -> str:
        return f"rag.template.{action}.{mode.value}"

    @staticmethod
    def _admin_action_id(mode: RAGAdminBackend, action: str) -> str:
        return f"rag.admin.{action}.{mode.value}"

    async def list_templates(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        include_builtin: bool = True,
        include_custom: bool = True,
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._template_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        records = await self._maybe_await(
            service.list_templates(
                include_builtin=include_builtin,
                include_custom=include_custom,
                tags=tags,
                user_id=user_id,
            )
        )
        return [
            normalize_template_record(normalized_mode.value, record)
            for record in list(records or [])
        ]

    async def get_template_detail(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        template_name: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._template_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        record = await self._maybe_await(service.get_template(template_name))
        return normalize_template_record(normalized_mode.value, record)

    async def create_template(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        name: str,
        description: str,
        template: dict[str, Any],
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._template_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        record = await self._maybe_await(
            service.create_template(
                name=name,
                description=description,
                template=template,
                tags=tags,
                user_id=user_id,
            )
        )
        return normalize_template_record(normalized_mode.value, record)

    async def update_template(
        self,
        template_name: str,
        *,
        mode: RAGAdminBackend | str | None = None,
        description: str | None = None,
        template: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._template_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        record = await self._maybe_await(
            service.update_template(
                template_name,
                description=description,
                template=template,
                tags=tags,
            )
        )
        return normalize_template_record(normalized_mode.value, record)

    async def delete_template(
        self,
        template_name: str,
        *,
        mode: RAGAdminBackend | str | None = None,
        hard_delete: bool = False,
    ) -> None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._template_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        await self._maybe_await(service.delete_template(template_name, hard_delete=hard_delete))

    async def get_template_diagnostics(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        diagnostics = await self._maybe_await(service.get_template_diagnostics())
        payload = dict(diagnostics or {})
        payload.setdefault("backend", normalized_mode.value)
        return payload

    async def apply_template(
        self,
        template_name: str,
        *,
        mode: RAGAdminBackend | str | None = None,
        text: str,
        override_options: dict[str, Any] | None = None,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, "apply_template", None)
        if not callable(method):
            raise ValueError(f"{normalized_mode.value.title()} template apply is not available yet.")
        return await self._maybe_await(
            method(
                template_name,
                text=text,
                override_options=override_options,
                include_metadata=include_metadata,
            )
        )

    async def list_collections(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        records = await self._maybe_await(service.list_collections())
        return [
            normalize_collection_record(normalized_mode.value, record)
            for record in list(records or [])
        ]

    async def get_collection_detail(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        collection_name: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        detail = await self._maybe_await(service.get_collection_detail(collection_name))
        return normalize_collection_record(normalized_mode.value, detail)

    async def delete_collection(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        collection_name: str,
    ) -> None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "configure"))
        service = self._service_for_mode(normalized_mode)
        await self._maybe_await(service.delete_collection(collection_name))

    async def reprocess_media(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_id: Any,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, "reprocess_media", None)
        if not callable(method):
            raise ValueError(f"{normalized_mode.value.title()} media reprocess is not available yet.")
        result = await self._maybe_await(method(media_id, **options))
        return dict(result or {})
