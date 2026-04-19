"""
Service helpers for server-backed chatbook import/export flows.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from ..tldw_api import ChatbookExportRequest, ChatbookImportRequest, TLDWAPIClient
from .chatbook_models import ContentType


SelectionKey = Union[str, ContentType]


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
        return await client.get_chatbook_export_job(job_id)

    async def continue_export(self, job_id: str) -> Dict[str, Any]:
        return await self.get_export_job(job_id)

    async def get_import_job(self, job_id: str) -> Dict[str, Any]:
        client = self._require_client()
        return await client.get_chatbook_import_job(job_id)

    async def continue_import(self, job_id: str) -> Dict[str, Any]:
        return await self.get_import_job(job_id)
