"""Local chatbook adapter for source-aware prompt/chatbook parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tldw_chatbook.tldw_api.prompt_chatbook_schemas import ChatbookImportRequest

from .chatbook_creator import ChatbookCreator
from .chatbook_importer import ChatbookImporter
from .chatbook_models import ContentType
from .conflict_resolver import ConflictResolution


class LocalChatbookService:
    """Expose local chatbook import/export operations through a service contract."""

    def __init__(self, db_paths: dict[str, str] | None = None):
        self.db_paths = db_paths or {}

    @staticmethod
    def _as_dict(payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return dict(payload)
        model_dump = getattr(payload, "model_dump", None)
        if callable(model_dump):
            return dict(model_dump(mode="json", exclude_none=True))
        return dict(payload)

    @staticmethod
    def _normalize_content_selections(selections: dict[Any, list[Any]] | None) -> dict[ContentType, list[str]]:
        normalized: dict[ContentType, list[str]] = {}
        for content_type, ids in (selections or {}).items():
            key = content_type if isinstance(content_type, ContentType) else ContentType(str(content_type))
            normalized[key] = [str(item_id) for item_id in ids]
        return normalized

    async def preview_chatbook(self, chatbook_file_path: str | Path) -> dict[str, Any]:
        manifest, error = ChatbookImporter(self.db_paths).preview_chatbook(Path(chatbook_file_path))
        return {
            "success": error is None,
            "message": error,
            "manifest": manifest.to_dict() if manifest is not None else None,
        }

    async def export_chatbook(self, request_data: Any) -> dict[str, Any]:
        payload = self._as_dict(request_data)
        output_path = payload.pop("output_path", None)
        if output_path is None:
            raise ValueError("output_path is required for local chatbook export.")

        creator = ChatbookCreator(self.db_paths)
        success, message, dependency_info = creator.create_chatbook(
            name=payload.get("name") or "Chatbook",
            description=payload.get("description") or "",
            content_selections=self._normalize_content_selections(payload.get("content_selections")),
            output_path=Path(output_path),
            author=payload.get("author"),
            include_media=bool(payload.get("include_media", False)),
            media_quality=payload.get("media_quality") or "thumbnail",
            include_embeddings=bool(payload.get("include_embeddings", False)),
            tags=payload.get("tags") or [],
            categories=payload.get("categories") or [],
        )
        return {
            "success": success,
            "message": message,
            "path": str(output_path),
            "dependency_info": dependency_info,
            "name": payload.get("name") or Path(output_path).stem,
        }

    async def import_chatbook(self, chatbook_file_path: str | Path, request_data: Any) -> dict[str, Any]:
        payload = self._as_dict(request_data)
        conflict_value = payload.get("conflict_resolution", ChatbookImportRequest().conflict_resolution)
        conflict_resolution = ConflictResolution(str(conflict_value))
        importer = ChatbookImporter(self.db_paths)
        success, message = importer.import_chatbook(
            Path(chatbook_file_path),
            content_selections=self._normalize_content_selections(payload.get("content_selections")),
            conflict_resolution=conflict_resolution,
            prefix_imported=bool(payload.get("prefix_imported", False)),
            import_media=bool(payload.get("import_media", True)),
            import_embeddings=bool(payload.get("import_embeddings", False)),
        )
        return {
            "success": success,
            "message": message,
            "path": str(chatbook_file_path),
            "name": Path(chatbook_file_path).stem,
        }
