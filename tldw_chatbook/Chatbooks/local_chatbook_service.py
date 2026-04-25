"""Local chatbook adapter for source-aware prompt/chatbook parity."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_chatbook.tldw_api.prompt_chatbook_schemas import ChatbookImportRequest
from tldw_chatbook.Utils.atomic_file_ops import atomic_write_json

from .chatbook_creator import ChatbookCreator
from .chatbook_importer import ChatbookImporter
from .chatbook_models import ContentType
from .conflict_resolver import ConflictResolution


class LocalChatbookService:
    """Expose local chatbook import/export operations through a service contract."""

    def __init__(self, db_paths: dict[str, str] | None = None, *, registry_path: str | Path | None = None):
        self.db_paths = db_paths or {}
        self.registry_path = Path(registry_path).expanduser() if registry_path is not None else self._default_registry_path()

    def _default_registry_path(self) -> Path:
        for key in ("Prompts", "ChaChaNotes", "Media"):
            db_path = self.db_paths.get(key)
            if db_path:
                return Path(db_path).expanduser().with_name("tldw_chatbook_chatbooks.json")
        return Path.home() / ".local" / "share" / "tldw_cli" / "tldw_chatbook_chatbooks.json"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _coerce_string_list(values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values if str(value).strip()]

    @staticmethod
    def _coerce_metadata(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        return dict(value)

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"next_id": 1, "records": []}
        with self.registry_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid local chatbook registry: {self.registry_path}")
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"Invalid local chatbook registry records: {self.registry_path}")
        next_id = int(payload.get("next_id") or 1)
        return {"next_id": next_id, "records": [dict(record) for record in records]}

    def _save_registry(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.registry_path, payload)

    def _find_record(self, registry: dict[str, Any], chatbook_id: int | str) -> dict[str, Any]:
        wanted = str(chatbook_id)
        for record in registry["records"]:
            if str(record.get("chatbook_id")) == wanted or str(record.get("id")) == wanted:
                return record
        raise KeyError(f"Local chatbook not found: {chatbook_id}")

    @staticmethod
    def _record_copy(record: dict[str, Any]) -> dict[str, Any]:
        copied = dict(record)
        copied["tags"] = list(copied.get("tags") or [])
        copied["categories"] = list(copied.get("categories") or [])
        copied["metadata"] = dict(copied.get("metadata") or {})
        return copied

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

    async def list_chatbooks(
        self,
        *,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
        **_: Any,
    ) -> list[dict[str, Any]]:
        registry = self._load_registry()
        records = [self._record_copy(record) for record in registry["records"]]
        query = str(q or "").strip().lower()
        if query:
            records = [
                record
                for record in records
                if query in str(record.get("name") or "").lower()
                or query in str(record.get("description") or "").lower()
                or any(query in tag.lower() for tag in record.get("tags") or [])
                or any(query in category.lower() for category in record.get("categories") or [])
            ]
        return records[int(offset) : int(offset) + int(limit)]

    async def get_chatbook(self, chatbook_id: int | str) -> dict[str, Any]:
        registry = self._load_registry()
        return self._record_copy(self._find_record(registry, chatbook_id))

    async def create_chatbook(
        self,
        *,
        name: str,
        description: str = "",
        file_path: str | Path | None = None,
        tags: list[Any] | None = None,
        categories: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        registry = self._load_registry()
        chatbook_id = int(registry["next_id"])
        now = self._utc_now()
        record = {
            "id": str(chatbook_id),
            "chatbook_id": chatbook_id,
            "name": str(name),
            "description": str(description or ""),
            "file_path": str(file_path) if file_path is not None else None,
            "tags": self._coerce_string_list(tags),
            "categories": self._coerce_string_list(categories),
            "metadata": self._coerce_metadata(metadata),
            "created_at": now,
            "updated_at": now,
        }
        if extra:
            record["metadata"].update({key: value for key, value in extra.items() if value is not None})
        registry["records"].append(record)
        registry["next_id"] = chatbook_id + 1
        self._save_registry(registry)
        return self._record_copy(record)

    async def update_chatbook(self, chatbook_id: int | str, **fields: Any) -> dict[str, Any]:
        registry = self._load_registry()
        record = self._find_record(registry, chatbook_id)
        if "name" in fields:
            record["name"] = str(fields["name"])
        if "description" in fields:
            record["description"] = str(fields["description"] or "")
        if "file_path" in fields:
            file_path = fields["file_path"]
            record["file_path"] = str(file_path) if file_path is not None else None
        if "tags" in fields:
            record["tags"] = self._coerce_string_list(fields["tags"])
        if "categories" in fields:
            record["categories"] = self._coerce_string_list(fields["categories"])
        if "metadata" in fields:
            record["metadata"] = self._coerce_metadata(fields["metadata"])
        record["updated_at"] = self._utc_now()
        self._save_registry(registry)
        return self._record_copy(record)

    async def delete_chatbook(self, chatbook_id: int | str) -> bool:
        registry = self._load_registry()
        wanted = str(chatbook_id)
        remaining = [
            record
            for record in registry["records"]
            if str(record.get("chatbook_id")) != wanted and str(record.get("id")) != wanted
        ]
        if len(remaining) == len(registry["records"]):
            raise KeyError(f"Local chatbook not found: {chatbook_id}")
        registry["records"] = remaining
        self._save_registry(registry)
        return True

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
