"""
Service helpers for server-backed chatbook import/export flows.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from ..tldw_api import ChatbookContinueExportRequest, ChatbookExportRequest, ChatbookImportRequest, TLDWAPIClient
from .chatbook_models import ChatbookManifest, ContentType


SelectionKey = Union[str, ContentType]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def build_tldw_api_client_from_config(config: Mapping[str, Any]) -> TLDWAPIClient:
    """Build a TLDW API client from application config data."""
    api_config = dict(config.get("tldw_api", {}))
    base_url = api_config.get("base_url") or api_config.get("api_url")
    if not base_url:
        raise ValueError("TLDW API base URL is not configured.")

    auth_token = api_config.get("auth_token") or api_config.get("api_key")
    auth_mode = str(api_config.get("auth_mode", "api_key")).lower()

    if auth_mode in {"bearer", "custom_token"}:
        client = TLDWAPIClient(base_url=base_url)
        client.bearer_token = auth_token
        return client

    return TLDWAPIClient(base_url=base_url, token=auth_token)


def build_server_chatbook_service_from_config(config: Mapping[str, Any]) -> tuple["ServerChatbookService", TLDWAPIClient]:
    """Build a server chatbook service plus its owned API client from application config."""
    client = build_tldw_api_client_from_config(config)
    return ServerChatbookService(client=client), client


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
    service = ServerChatbookService(client=None)
    selections = build_server_import_selections_from_manifest(
        manifest,
        import_media=import_media,
        import_embeddings=import_embeddings,
    )
    return service.validate_server_import_selection(selections)


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

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server chatbook operations.")
        return self.client

    def _normalize_selection_key(self, key: SelectionKey) -> str:
        if isinstance(key, ContentType):
            return key.value
        return str(key).strip().lower()

    def normalize_content_selections(
        self,
        selections: Optional[Mapping[SelectionKey, Iterable[Any]]],
    ) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}
        if not selections:
            return normalized

        for key, raw_ids in selections.items():
            normalized_key = self._normalize_selection_key(key)
            if not normalized_key or raw_ids is None:
                continue

            normalized_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
            if normalized_ids:
                normalized[normalized_key] = normalized_ids

        return normalized

    def validate_server_import_selection(
        self,
        selections: Optional[Mapping[SelectionKey, Iterable[Any]]],
    ) -> List[str]:
        normalized = self.normalize_content_selections(selections)
        return sorted(
            key for key, ids in normalized.items()
            if ids and key not in self.SUPPORTED_SERVER_IMPORT_TYPES
        )

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

    async def preview_chatbook(self, chatbook_file_path: Union[str, Path]) -> Dict[str, Any]:
        client = self._require_client()
        return await client.preview_chatbook(str(chatbook_file_path))

    async def export_chatbook(self, request_data: ChatbookExportRequest) -> Dict[str, Any]:
        client = self._require_client()
        return await client.export_chatbook(request_data)

    async def continue_chatbook_export(self, request_data: ChatbookContinueExportRequest) -> Dict[str, Any]:
        client = self._require_client()
        return _to_plain_dict(await client.continue_chatbook_export(request_data))

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
        client = self._require_client()
        return await client.import_chatbook(str(chatbook_file_path), request_data)

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
        client = self._require_client()
        return _to_plain_dict(await client.get_chatbook_export_job(job_id))

    async def continue_export(self, job_id: str) -> Dict[str, Any]:
        return await self.get_export_job(job_id)

    async def get_import_job(self, job_id: str) -> Dict[str, Any]:
        client = self._require_client()
        return _to_plain_dict(await client.get_chatbook_import_job(job_id))

    async def continue_import(self, job_id: str) -> Dict[str, Any]:
        return await self.get_import_job(job_id)

    async def list_export_jobs(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        client = self._require_client()
        return _to_plain_dict(await client.list_chatbook_export_jobs(limit=limit, offset=offset))

    async def list_import_jobs(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        client = self._require_client()
        return _to_plain_dict(await client.list_chatbook_import_jobs(limit=limit, offset=offset))

    async def cancel_export_job(self, job_id: str) -> Dict[str, Any]:
        client = self._require_client()
        return _to_plain_dict(await client.cancel_chatbook_export_job(job_id))

    async def cancel_import_job(self, job_id: str) -> Dict[str, Any]:
        client = self._require_client()
        return _to_plain_dict(await client.cancel_chatbook_import_job(job_id))

    async def remove_export_job(self, job_id: str) -> Dict[str, Any]:
        client = self._require_client()
        return _to_plain_dict(await client.remove_chatbook_export_job(job_id))

    async def cleanup_expired_exports(self) -> Dict[str, Any]:
        client = self._require_client()
        return _to_plain_dict(await client.cleanup_chatbook_exports())

    async def download_export_job(self, job_id: str, destination_path: Union[str, Path]) -> Path:
        client = self._require_client()
        payload = await client.download_chatbook_export(job_id)
        path = Path(destination_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    async def remove_import_job(self, job_id: str) -> Dict[str, Any]:
        client = self._require_client()
        return _to_plain_dict(await client.remove_chatbook_import_job(job_id))
