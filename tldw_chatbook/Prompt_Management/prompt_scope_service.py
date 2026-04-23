"""Scope-aware routing for local and server-backed prompt operations."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Optional

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    PromptCollectionCreateRequest,
    PromptCollectionUpdateRequest,
    PromptCreateRequest,
    TLDWAPIClient,
)
from .prompt_normalizers import (
    normalize_prompt_collection_list,
    normalize_prompt_collection_record,
    normalize_prompt_list,
    normalize_prompt_record,
    normalize_prompt_version_list,
)


class PromptBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


def _payload_from_fields(
    *,
    name: Optional[str] = None,
    author: Optional[str] = None,
    details: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    prompt_format: Optional[str] = None,
    prompt_schema_version: Optional[int] = None,
    prompt_definition: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "author": author,
        "details": details,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "keywords": keywords,
        "prompt_format": prompt_format,
        "prompt_schema_version": prompt_schema_version,
        "prompt_definition": prompt_definition,
    }
    return {key: value for key, value in payload.items() if value is not None}


def _prompt_create_request_from_payload(payload: dict[str, Any]) -> PromptCreateRequest:
    if not payload.get("name"):
        raise ValueError("Prompt name is required for server prompt saves.")
    return PromptCreateRequest(**payload)


class ServerPromptService:
    """Thin prompt service around the shared server API client."""

    def __init__(self, client: TLDWAPIClient | None):
        self.client = client

    @classmethod
    def from_config(cls, app_config: dict[str, Any]) -> "ServerPromptService":
        return cls(client=build_tldw_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server prompt operations.")
        return self.client

    async def list_prompts(
        self,
        *,
        page: int = 1,
        per_page: int = 10,
        include_deleted: bool = False,
        sort_by: str = "last_modified",
        sort_order: str = "desc",
    ) -> Any:
        return await self._require_client().list_prompts(
            page=page,
            per_page=per_page,
            include_deleted=include_deleted,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    async def get_prompt(self, prompt_identifier: str | int, *, include_deleted: bool = False) -> Any:
        return await self._require_client().get_prompt(prompt_identifier, include_deleted=include_deleted)

    async def create_prompt(self, payload: dict[str, Any]) -> Any:
        return await self._require_client().create_prompt(_prompt_create_request_from_payload(payload))

    async def update_prompt(self, prompt_identifier: str | int, payload: dict[str, Any]) -> Any:
        return await self._require_client().update_prompt(
            prompt_identifier,
            _prompt_create_request_from_payload(payload),
        )

    async def delete_prompt(self, prompt_identifier: str | int) -> Any:
        return await self._require_client().delete_prompt(prompt_identifier)

    async def record_prompt_usage(self, prompt_identifier: str | int) -> Any:
        return await self._require_client().record_prompt_usage(prompt_identifier)

    async def list_prompt_versions(self, prompt_identifier: str | int) -> Any:
        return await self._require_client().list_prompt_versions(prompt_identifier)

    async def restore_prompt_version(self, prompt_identifier: str | int, version: int) -> Any:
        return await self._require_client().restore_prompt_version(prompt_identifier, version)

    async def create_prompt_collection(self, payload: dict[str, Any]) -> Any:
        return await self._require_client().create_prompt_collection(PromptCollectionCreateRequest(**payload))

    async def list_prompt_collections(self, *, limit: int = 200, offset: int = 0) -> Any:
        return await self._require_client().list_prompt_collections(limit=limit, offset=offset)

    async def get_prompt_collection(self, collection_id: int) -> Any:
        return await self._require_client().get_prompt_collection(collection_id)

    async def update_prompt_collection(self, collection_id: int, payload: dict[str, Any]) -> Any:
        return await self._require_client().update_prompt_collection(
            collection_id,
            PromptCollectionUpdateRequest(**payload),
        )


class LocalPromptService:
    """Adapter over the local prompts DB/interop API."""

    def __init__(self, prompt_db: Any):
        self.prompt_db = prompt_db

    def list_prompts(
        self,
        *,
        page: int = 1,
        per_page: int = 10,
        include_deleted: bool = False,
        **_kwargs: Any,
    ) -> Any:
        return self.prompt_db.list_prompts(
            page=page,
            per_page=per_page,
            include_deleted=include_deleted,
        )

    def get_prompt(self, prompt_identifier: str | int, *, include_deleted: bool = False) -> Any:
        if hasattr(self.prompt_db, "fetch_prompt_details"):
            return self.prompt_db.fetch_prompt_details(prompt_identifier, include_deleted=include_deleted)
        return self.prompt_db.get_prompt(prompt_identifier, include_deleted=include_deleted)

    def create_prompt(self, payload: dict[str, Any]) -> Any:
        prompt_id, prompt_uuid, _message = self.prompt_db.add_prompt(
            name=payload.get("name"),
            author=payload.get("author"),
            details=payload.get("details"),
            system_prompt=payload.get("system_prompt"),
            user_prompt=payload.get("user_prompt"),
            keywords=payload.get("keywords"),
            overwrite=False,
            prompt_format=payload.get("prompt_format"),
            prompt_schema_version=payload.get("prompt_schema_version"),
            prompt_definition=payload.get("prompt_definition"),
        )
        identifier = prompt_uuid or prompt_id
        return self.get_prompt(identifier, include_deleted=True)

    def update_prompt(self, prompt_identifier: str | int, payload: dict[str, Any]) -> Any:
        existing = self.get_prompt(prompt_identifier, include_deleted=True)
        if not existing:
            raise ValueError(f"Prompt '{prompt_identifier}' not found.")

        if hasattr(self.prompt_db, "update_prompt_by_id"):
            prompt_uuid, _message = self.prompt_db.update_prompt_by_id(existing["id"], payload)
            return self.get_prompt(prompt_uuid or existing["id"], include_deleted=True)

        prompt_id, prompt_uuid, _message = self.prompt_db.add_prompt(
            name=payload.get("name", existing.get("name")),
            author=payload.get("author", existing.get("author")),
            details=payload.get("details", existing.get("details")),
            system_prompt=payload.get("system_prompt", existing.get("system_prompt")),
            user_prompt=payload.get("user_prompt", existing.get("user_prompt")),
            keywords=payload.get("keywords", existing.get("keywords")),
            overwrite=True,
            prompt_format=payload.get("prompt_format", existing.get("prompt_format")),
            prompt_schema_version=payload.get("prompt_schema_version", existing.get("prompt_schema_version")),
            prompt_definition=payload.get("prompt_definition", existing.get("prompt_definition")),
        )
        return self.get_prompt(prompt_uuid or prompt_id, include_deleted=True)

    def delete_prompt(self, prompt_identifier: str | int) -> Any:
        return self.prompt_db.soft_delete_prompt(prompt_identifier)

    def record_prompt_usage(self, prompt_identifier: str | int) -> Any:
        if hasattr(self.prompt_db, "record_prompt_usage"):
            return self.prompt_db.record_prompt_usage(prompt_identifier)
        return self.get_prompt(prompt_identifier, include_deleted=True)


class PromptScopeService:
    """Route prompt actions to the active local/server backend and normalize outputs."""

    def __init__(self, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: PromptBackend | str | None) -> PromptBackend:
        if mode is None:
            return PromptBackend.LOCAL
        if isinstance(mode, PromptBackend):
            return mode
        try:
            return PromptBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid prompt backend: {mode}") from exc

    def _service_for_mode(self, mode: PromptBackend) -> Any:
        if mode == PromptBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local prompt backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server prompt backend is unavailable.")
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
    def _action_id(mode: PromptBackend, action: str) -> str:
        return f"prompts.{action}.{mode.value}"

    @staticmethod
    def _collection_action_id(action: str) -> str:
        return f"prompts.collections.{action}.server"

    def _require_server_mode_for_collections(self, mode: PromptBackend | str | None) -> PromptBackend:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != PromptBackend.SERVER:
            raise ValueError("Prompt collections require server mode.")
        return normalized_mode

    async def list_prompts(
        self,
        *,
        mode: PromptBackend | str | None = None,
        page: int = 1,
        per_page: int = 10,
        include_deleted: bool = False,
        sort_by: str = "last_modified",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(
            service.list_prompts(
                page=page,
                per_page=per_page,
                include_deleted=include_deleted,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        )
        return normalize_prompt_list(response, backend=normalized_mode.value, page=page, per_page=per_page)

    async def get_prompt(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(
            service.get_prompt(prompt_identifier, include_deleted=include_deleted)
        )
        return normalize_prompt_record(response, backend=normalized_mode.value)

    async def save_prompt(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int | None = None,
        name: Optional[str] = None,
        author: Optional[str] = None,
        details: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        prompt_format: Optional[str] = None,
        prompt_schema_version: Optional[int] = None,
        prompt_definition: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        action = "update" if prompt_identifier not in (None, "") else "create"
        self._enforce_policy(self._action_id(normalized_mode, action))
        service = self._service_for_mode(normalized_mode)
        payload = _payload_from_fields(
            name=name,
            author=author,
            details=details,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            keywords=keywords,
            prompt_format=prompt_format,
            prompt_schema_version=prompt_schema_version,
            prompt_definition=prompt_definition,
        )
        if action == "create":
            response = await self._maybe_await(service.create_prompt(payload))
        else:
            response = await self._maybe_await(service.update_prompt(prompt_identifier, payload))
        return normalize_prompt_record(response, backend=normalized_mode.value)

    async def delete_prompt(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.delete_prompt(prompt_identifier))
        if response == {}:
            return True
        return bool(response)

    async def record_prompt_usage(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "use"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.record_prompt_usage(prompt_identifier))
        return normalize_prompt_record(response, backend=normalized_mode.value)

    async def list_prompt_versions(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "versions"))
        if normalized_mode == PromptBackend.LOCAL:
            raise ValueError("Local prompt version history is unavailable.")
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.list_prompt_versions(prompt_identifier))
        return normalize_prompt_version_list(response, backend=normalized_mode.value)

    async def restore_prompt_version(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
        version: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "restore_version"))
        if normalized_mode == PromptBackend.LOCAL:
            raise ValueError("Local prompt version restore is unavailable.")
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.restore_prompt_version(prompt_identifier, version))
        return normalize_prompt_record(response, backend=normalized_mode.value)

    async def create_prompt_collection(
        self,
        *,
        mode: PromptBackend | str | None = None,
        name: str,
        description: Optional[str] = None,
        prompt_ids: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_mode_for_collections(mode)
        self._enforce_policy(self._collection_action_id("create"))
        service = self._service_for_mode(normalized_mode)
        payload = {
            "name": name,
            "description": description,
            "prompt_ids": list(prompt_ids or []),
        }
        response = await self._maybe_await(service.create_prompt_collection(payload))
        data = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
        collection_id = int(data["collection_id"])
        return {
            "id": f"{normalized_mode.value}:prompt_collection:{collection_id}",
            "backend": normalized_mode.value,
            "collection_id": collection_id,
        }

    async def list_prompt_collections(
        self,
        *,
        mode: PromptBackend | str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_mode_for_collections(mode)
        self._enforce_policy(self._collection_action_id("list"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.list_prompt_collections(limit=limit, offset=offset))
        return normalize_prompt_collection_list(
            response,
            backend=normalized_mode.value,
            limit=limit,
            offset=offset,
        )

    async def get_prompt_collection(
        self,
        *,
        mode: PromptBackend | str | None = None,
        collection_id: int,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_mode_for_collections(mode)
        self._enforce_policy(self._collection_action_id("detail"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.get_prompt_collection(collection_id))
        return normalize_prompt_collection_record(response, backend=normalized_mode.value)

    async def update_prompt_collection(
        self,
        *,
        mode: PromptBackend | str | None = None,
        collection_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        prompt_ids: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_mode_for_collections(mode)
        self._enforce_policy(self._collection_action_id("update"))
        service = self._service_for_mode(normalized_mode)
        payload = {
            key: value
            for key, value in {
                "name": name,
                "description": description,
                "prompt_ids": prompt_ids,
            }.items()
            if value is not None
        }
        response = await self._maybe_await(service.update_prompt_collection(collection_id, payload))
        return normalize_prompt_collection_record(response, backend=normalized_mode.value)


def build_prompt_scope_service(
    *,
    prompt_db: Any,
    app_config: dict[str, Any] | None = None,
    policy_enforcer: Any = None,
) -> PromptScopeService:
    """Build the source-aware prompt service from app startup dependencies."""
    local_service = LocalPromptService(prompt_db) if prompt_db is not None else None
    try:
        server_service = ServerPromptService.from_config(app_config or {})
    except ValueError:
        server_service = ServerPromptService(client=None)

    return PromptScopeService(
        local_service=local_service,
        server_service=server_service,
        policy_enforcer=policy_enforcer,
    )
