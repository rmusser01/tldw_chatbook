"""
Service helpers for server-backed chatbook import/export flows.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.bootstrap import build_server_chatbook_service as build_runtime_server_chatbook_service
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    ChatbookContinueExportRequest,
    ChatbookExportRequest,
    ChatbookImportRequest,
    ReadingExportResponse,
    TLDWAPIClient,
)
from .chatbook_models import ChatbookManifest, ContentType


SelectionKey = Union[str, ContentType]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def build_tldw_api_client_from_config(config: Mapping[str, Any]) -> TLDWAPIClient:
    """Backwards-compatible proxy to the authoritative runtime-policy client factory."""
    return build_runtime_api_client_from_config(config)


def _normalize_selection_key(key: SelectionKey) -> str:
    if isinstance(key, ContentType):
        return key.value
    return str(key).strip().lower()


def normalize_server_content_selections(
    selections: Optional[Mapping[SelectionKey, Iterable[Any]]],
) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    if not selections:
        return normalized

    for key, raw_ids in selections.items():
        normalized_key = _normalize_selection_key(key)
        if not normalized_key or raw_ids is None:
            continue

        normalized_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
        if normalized_ids:
            normalized[normalized_key] = normalized_ids

    return normalized


def find_unsupported_server_import_types(
    selections: Optional[Mapping[SelectionKey, Iterable[Any]]],
) -> List[str]:
    normalized = normalize_server_content_selections(selections)
    return sorted(
        key
        for key, ids in normalized.items()
        if ids and key not in ServerChatbookService.SUPPORTED_SERVER_IMPORT_TYPES
    )


def build_server_chatbook_service_from_config(
    config: Mapping[str, Any],
    *,
    policy_enforcer: Any = None,
) -> tuple["ServerChatbookService", TLDWAPIClient]:
    """Build a server chatbook service plus its owned API client from application config."""
    service = build_runtime_server_chatbook_service(
        app_config=config,
        policy_enforcer=policy_enforcer,
    )
    return service, service.client


def build_server_import_selections_from_manifest(
    manifest: Optional[ChatbookManifest],
    import_media: bool = False,
    import_embeddings: bool = False,
) -> Dict[str, List[str]]:
    selections: Dict[str, List[str]] = {}
    if manifest is None:
        return selections

    for item in manifest.content_items:
        content_type = item.type.value if isinstance(item.type, ContentType) else str(item.type)
        if content_type == ContentType.MEDIA.value and not import_media:
            continue
        if content_type == ContentType.EMBEDDING.value and not import_embeddings:
            continue
        selections.setdefault(content_type, []).append(str(item.id))

    return selections


def get_server_import_blockers_from_manifest(
    manifest: Optional[ChatbookManifest],
    import_media: bool = False,
    import_embeddings: bool = False,
) -> List[str]:
    selections = build_server_import_selections_from_manifest(
        manifest,
        import_media=import_media,
        import_embeddings=import_embeddings,
    )
    return find_unsupported_server_import_types(selections)


def build_server_job_record(
    job_type: str,
    job_result: Mapping[str, Any],
    recorded_at: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "job_type": job_type,
        "job_id": job_result.get("job_id"),
        "status": job_result.get("status", "unknown"),
        "progress_percentage": int(job_result.get("progress_percentage", 0) or 0),
        "chatbook_name": job_result.get("chatbook_name"),
        "download_url": job_result.get("download_url"),
        "successful_items": int(job_result.get("successful_items", 0) or 0),
        "failed_items": int(job_result.get("failed_items", 0) or 0),
        "recorded_at": recorded_at or _utc_now_iso(),
    }


def get_server_job_records(app_instance: Any) -> List[Dict[str, Any]]:
    return list(getattr(app_instance, "_chatbook_server_jobs", []))


def record_server_job(app_instance: Any, job_record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records = get_server_job_records(app_instance)
    records.insert(0, dict(job_record))
    setattr(app_instance, "_chatbook_server_jobs", records)
    return records


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping-like chatbook response, got {type(value).__name__}")


class ServerChatbookService:
    """Thin service around the chatbook endpoints in the shared TLDW API client."""

    SUPPORTED_SERVER_IMPORT_TYPES = {
        ContentType.CONVERSATION.value,
        ContentType.NOTE.value,
        ContentType.CHARACTER.value,
    }

    def __init__(self, client: Optional[TLDWAPIClient], *, policy_enforcer: Any | None = None):
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any] | None,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerChatbookService":
        return cls(
            build_runtime_api_client_from_config(app_config or {}),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server chatbook operations.")
        return self.client

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server chatbook action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _action_id(action: str) -> str:
        return f"chatbooks.{action}.server"

    @staticmethod
    def _job_action_id(job_kind: str, action: str) -> str:
        return f"chatbooks.{job_kind}_jobs.{action}.server"

    def _normalize_selection_key(self, key: SelectionKey) -> str:
        return _normalize_selection_key(key)

    def normalize_content_selections(
        self,
        selections: Optional[Mapping[SelectionKey, Iterable[Any]]],
    ) -> Dict[str, List[str]]:
        return normalize_server_content_selections(selections)

    def validate_server_import_selection(
        self,
        selections: Optional[Mapping[SelectionKey, Iterable[Any]]],
    ) -> List[str]:
        return find_unsupported_server_import_types(selections)

    def build_export_request_payload(
        self,
        name: str,
        description: str,
        selections: Mapping[SelectionKey, Iterable[Any]],
        author: Optional[str] = None,
        include_media: bool = False,
        media_quality: str = "compressed",
        include_embeddings: bool = False,
        include_generated_content: bool = True,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        async_mode: bool = False,
    ) -> ChatbookExportRequest:
        return ChatbookExportRequest(
            name=name,
            description=description,
            content_selections=self.normalize_content_selections(selections),
            author=author,
            include_media=include_media,
            media_quality=media_quality,
            include_embeddings=include_embeddings,
            include_generated_content=include_generated_content,
            tags=tags or [],
            categories=categories or [],
            async_mode=async_mode,
        )

    def build_import_request_payload(
        self,
        selections: Optional[Mapping[SelectionKey, Iterable[Any]]] = None,
        conflict_resolution: Union[str, Any] = "skip",
        prefix_imported: bool = False,
        import_media: bool = False,
        import_embeddings: bool = False,
        async_mode: bool = False,
    ) -> ChatbookImportRequest:
        normalized_selections = self.normalize_content_selections(selections)
        unsupported = self.validate_server_import_selection(normalized_selections)
        if unsupported:
            unsupported_text = ", ".join(unsupported)
            raise ValueError(f"Unsupported server import content types: {unsupported_text}")

        if hasattr(conflict_resolution, "value"):
            conflict_resolution = conflict_resolution.value

        return ChatbookImportRequest(
            content_selections=normalized_selections or None,
            conflict_resolution=str(conflict_resolution),
            prefix_imported=prefix_imported,
            import_media=import_media,
            import_embeddings=import_embeddings,
            async_mode=async_mode,
        )

    def _coerce_export_request(self, request_data: ChatbookExportRequest | Mapping[str, Any]) -> ChatbookExportRequest:
        if isinstance(request_data, ChatbookExportRequest):
            return request_data
        payload = dict(request_data)
        if "content_selections" in payload:
            payload["content_selections"] = self.normalize_content_selections(payload.get("content_selections"))
        return ChatbookExportRequest(**payload)

    def _coerce_continue_export_request(
        self,
        request_data: ChatbookContinueExportRequest | Mapping[str, Any],
    ) -> ChatbookContinueExportRequest:
        if isinstance(request_data, ChatbookContinueExportRequest):
            return request_data
        return ChatbookContinueExportRequest(**dict(request_data))

    def _coerce_import_request(self, request_data: ChatbookImportRequest | Mapping[str, Any]) -> ChatbookImportRequest:
        if isinstance(request_data, ChatbookImportRequest):
            return request_data
        payload = dict(request_data)
        if "content_selections" in payload and payload.get("content_selections") is not None:
            payload["content_selections"] = self.normalize_content_selections(payload.get("content_selections"))
        return ChatbookImportRequest(**payload)

    async def preview_chatbook(self, chatbook_file_path: Union[str, Path]) -> Dict[str, Any]:
        self._enforce(self._action_id("detail"))
        client = self._require_client()
        return await client.preview_chatbook(str(chatbook_file_path))

    async def export_chatbook(self, request_data: ChatbookExportRequest) -> Dict[str, Any]:
        self._enforce(self._action_id("export"))
        client = self._require_client()
        return await client.export_chatbook(self._coerce_export_request(request_data))

    async def continue_chatbook_export(
        self,
        request_data: ChatbookContinueExportRequest | Mapping[str, Any],
    ) -> Dict[str, Any]:
        self._enforce(self._action_id("export"))
        client = self._require_client()
        return _to_plain_dict(await client.continue_chatbook_export(self._coerce_continue_export_request(request_data)))

    async def export_chatbook_from_selection(
        self,
        name: str,
        description: str,
        selections: Mapping[SelectionKey, Iterable[Any]],
        author: Optional[str] = None,
        include_media: bool = False,
        media_quality: str = "compressed",
        include_embeddings: bool = False,
        include_generated_content: bool = True,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        async_mode: bool = False,
    ) -> Dict[str, Any]:
        request = self.build_export_request_payload(
            name=name,
            description=description,
            selections=selections,
            author=author,
            include_media=include_media,
            media_quality=media_quality,
            include_embeddings=include_embeddings,
            include_generated_content=include_generated_content,
            tags=tags,
            categories=categories,
            async_mode=async_mode,
        )
        return await self.export_chatbook(request)

    async def import_chatbook(
        self,
        chatbook_file_path: Union[str, Path],
        request_data: ChatbookImportRequest,
    ) -> Dict[str, Any]:
        self._enforce(self._action_id("import"))
        client = self._require_client()
        return await client.import_chatbook(str(chatbook_file_path), self._coerce_import_request(request_data))

    async def import_chatbook_from_selection(
        self,
        chatbook_file_path: Union[str, Path],
        selections: Optional[Mapping[SelectionKey, Iterable[Any]]] = None,
        conflict_resolution: Union[str, Any] = "skip",
        prefix_imported: bool = False,
        import_media: bool = False,
        import_embeddings: bool = False,
        async_mode: bool = False,
    ) -> Dict[str, Any]:
        request = self.build_import_request_payload(
            selections=selections,
            conflict_resolution=conflict_resolution,
            prefix_imported=prefix_imported,
            import_media=import_media,
            import_embeddings=import_embeddings,
            async_mode=async_mode,
        )
        return await self.import_chatbook(chatbook_file_path, request)

    async def get_export_job(self, job_id: str) -> Dict[str, Any]:
        self._enforce(self._job_action_id("export", "detail"))
        client = self._require_client()
        return _to_plain_dict(await client.get_chatbook_export_job(job_id))

    @staticmethod
    def _download_response_to_record(job_id: str, response: ReadingExportResponse | Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(response, Mapping):
            payload = dict(response)
        else:
            payload = {
                "content": response.content,
                "content_type": response.content_type,
                "content_disposition": response.content_disposition,
                "filename": response.filename,
            }
        payload.setdefault("job_id", job_id)
        return payload

    async def download_export(
        self,
        job_id: str,
        *,
        token: str | None = None,
        exp: int | str | None = None,
    ) -> Dict[str, Any]:
        self._enforce(self._job_action_id("export", "export"))
        client = self._require_client()
        response = await client.download_chatbook_export(job_id, token=token, exp=exp)
        return self._download_response_to_record(job_id, response)

    async def continue_export(self, job_id: str) -> Dict[str, Any]:
        return await self.get_export_job(job_id)

    async def get_import_job(self, job_id: str) -> Dict[str, Any]:
        self._enforce(self._job_action_id("import", "detail"))
        client = self._require_client()
        return _to_plain_dict(await client.get_chatbook_import_job(job_id))

    async def continue_import(self, job_id: str) -> Dict[str, Any]:
        return await self.get_import_job(job_id)

    async def list_export_jobs(self, *, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        self._enforce(self._job_action_id("export", "list"))
        client = self._require_client()
        return _to_plain_dict(await client.list_chatbook_export_jobs(limit=limit, offset=offset))

    async def list_import_jobs(self, *, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        self._enforce(self._job_action_id("import", "list"))
        client = self._require_client()
        return _to_plain_dict(await client.list_chatbook_import_jobs(limit=limit, offset=offset))

    async def cancel_export_job(self, job_id: str) -> Dict[str, Any]:
        self._enforce(self._job_action_id("export", "update"))
        client = self._require_client()
        return _to_plain_dict(await client.cancel_chatbook_export_job(job_id))

    async def cancel_import_job(self, job_id: str) -> Dict[str, Any]:
        self._enforce(self._job_action_id("import", "update"))
        client = self._require_client()
        return _to_plain_dict(await client.cancel_chatbook_import_job(job_id))

    async def remove_export_job(self, job_id: str) -> Dict[str, Any]:
        self._enforce(self._job_action_id("export", "delete"))
        client = self._require_client()
        return _to_plain_dict(await client.remove_chatbook_export_job(job_id))

    async def remove_import_job(self, job_id: str) -> Dict[str, Any]:
        self._enforce(self._job_action_id("import", "delete"))
        client = self._require_client()
        return _to_plain_dict(await client.remove_chatbook_import_job(job_id))

    async def download_export_job(self, job_id: str, destination: Union[str, Path]) -> Path:
        self._enforce(self._job_action_id("export", "export"))
        client = self._require_client()
        response = await client.download_chatbook_export(job_id)
        if isinstance(response, bytes):
            content = response
        elif isinstance(response, Mapping):
            content = response.get("content", b"")
        else:
            content = response.content
        if isinstance(content, str):
            content = content.encode("utf-8")
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(content)
        return destination_path

    async def cleanup_expired_exports(self) -> Dict[str, Any]:
        self._enforce(self._job_action_id("export", "delete"))
        client = self._require_client()
        return _to_plain_dict(await client.cleanup_chatbook_exports())
