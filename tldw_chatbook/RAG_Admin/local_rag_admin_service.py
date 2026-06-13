"""Local retrieval-admin adapter for chunking templates and embedding collections."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from ..Chunking.chunking_interop_library import get_chunking_service
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE

try:
    from ..Embeddings.Chroma_Lib import ChromaDBManager
except Exception:  # pragma: no cover - optional dependency fallback
    ChromaDBManager = None  # type: ignore[assignment]


class LocalRAGAdminService:
    """Wrap existing local chunking-template and Chroma collection operations."""

    def __init__(
        self,
        media_db: Any,
        *,
        app_config: Optional[Mapping[str, Any]] = None,
        user_id: Optional[str] = None,
        chunking_service: Any = None,
        chroma_manager: Any = None,
        media_service: Any = None,
    ):
        self.media_db = media_db
        self.app_config = dict(app_config or {})
        self.user_id = str(user_id or self.app_config.get("USERS_NAME") or "default_user")
        self.chunking_service = chunking_service or (get_chunking_service(media_db) if media_db is not None else None)
        self._chroma_manager = chroma_manager
        self.media_service = media_service

    def _require_chunking_service(self) -> Any:
        if self.chunking_service is None:
            raise ValueError("Local chunking template backend is unavailable.")
        return self.chunking_service

    def _build_chroma_manager(self) -> Any:
        if self._chroma_manager is not None:
            return self._chroma_manager
        if not DEPENDENCIES_AVAILABLE.get("embeddings_rag", False) or ChromaDBManager is None:
            raise ValueError("Local embeddings backend is unavailable.")
        self._chroma_manager = ChromaDBManager(self.user_id, dict(self.app_config))
        return self._chroma_manager

    def _coerce_collection(self, collection: Any) -> dict[str, Any]:
        return {
            "name": getattr(collection, "name", ""),
            "metadata": dict(getattr(collection, "metadata", {}) or {}),
        }

    @staticmethod
    def _parse_template_config(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return {}
            if isinstance(parsed, Mapping):
                return dict(parsed)
        return {}

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if tags is None:
            return []
        candidates = [tags] if isinstance(tags, str) else list(tags)
        return [str(tag) for tag in candidates if str(tag).strip()]

    @classmethod
    def _extract_template_tags(cls, record: Mapping[str, Any], template_config: Mapping[str, Any]) -> list[str]:
        raw_tags = record.get("tags")
        if raw_tags is None:
            raw_tags = template_config.get("tags")
        if raw_tags is None:
            metadata = template_config.get("metadata")
            if isinstance(metadata, Mapping):
                raw_tags = metadata.get("tags")
        return cls._normalize_tags(raw_tags)

    def _decorate_template_record(self, record: Mapping[str, Any] | None) -> dict[str, Any]:
        if not record:
            return {}
        decorated = dict(record)
        template_config = self._parse_template_config(
            decorated.get("template") or decorated.get("template_json")
        )
        decorated["tags"] = self._extract_template_tags(decorated, template_config)
        return decorated

    @staticmethod
    def _with_template_tags(template: Mapping[str, Any], tags: Sequence[str] | None) -> dict[str, Any]:
        payload = dict(template)
        if tags is None:
            return payload
        normalized_tags = LocalRAGAdminService._normalize_tags(tags)
        metadata = dict(payload.get("metadata") or {})
        metadata["tags"] = normalized_tags
        payload["metadata"] = metadata
        payload["tags"] = normalized_tags
        return payload

    def _get_collection(self, collection_name: str) -> Any:
        manager = self._build_chroma_manager()
        return manager.client.get_collection(name=collection_name)

    def _default_collection_name(self, collection_name: Optional[str] = None) -> str:
        if collection_name:
            return str(collection_name)
        return str(self._build_chroma_manager().get_user_default_collection_name())

    def _get_media_for_embedding(self, media_id: int) -> dict[str, Any]:
        if self.media_db is None:
            raise ValueError("Local media DB is required for local embedding generation.")

        getter = getattr(self.media_db, "get_media_by_ids_for_embedding", None)
        if callable(getter):
            rows = list(getter([media_id]) or [])
            if rows:
                return dict(rows[0])

        detail_getter = getattr(self.media_db, "get_media_by_id", None)
        if callable(detail_getter):
            row = detail_getter(media_id)
            if row:
                return dict(row)

        raise ValueError(f"Local media item {media_id} was not found or has no embeddable content.")

    @staticmethod
    def _word_chunks_for_reprocess(content: str, *, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
        words = [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", content)]
        if not words:
            return []
        size = max(1, int(chunk_size))
        overlap = min(max(0, int(chunk_overlap)), size - 1)
        step = max(1, size - overlap)
        chunks: list[dict[str, Any]] = []
        for start in range(0, len(words), step):
            bucket = words[start:start + size]
            if not bucket:
                continue
            chunks.append(
                {
                    "text": " ".join(word for word, _start, _end in bucket),
                    "start_index": bucket[0][1],
                    "end_index": bucket[-1][2],
                }
            )
            if start + size >= len(words):
                break
        return chunks

    def _embedding_ids_for_media(self, media_id: int, *, collection_name: Optional[str] = None) -> list[str]:
        collection_name = self._default_collection_name(collection_name)
        try:
            collection = self._build_chroma_manager().client.get_collection(name=collection_name)
            payload = collection.get(where={"media_id": str(media_id)}, include=["metadatas"])
        except Exception:
            return []
        return [str(item_id) for item_id in payload.get("ids", []) if item_id]

    def _infer_collection_dimension(self, collection: Any, metadata: Mapping[str, Any]) -> int | None:
        dimension = metadata.get("embedding_dimension")
        try:
            return int(dimension) if dimension is not None else None
        except (TypeError, ValueError):
            pass

        try:
            sample = collection.get(limit=1, include=["embeddings"])
        except Exception:
            return None

        embeddings = sample.get("embeddings") or []
        if not embeddings:
            return None
        first_bucket = embeddings[0]
        candidate = first_bucket[0] if isinstance(first_bucket, list) and first_bucket else first_bucket
        if candidate is None or not hasattr(candidate, "__len__"):
            return None
        try:
            return len(candidate)
        except TypeError:
            return None

    def _record_local_media_job(
        self,
        *,
        operation: str,
        media_id: int,
        result: Mapping[str, Any],
        request: Mapping[str, Any] | None = None,
        status: str = "completed",
    ) -> dict[str, Any]:
        prefix = "local-embedding" if operation == "media_embeddings" else "local-reprocess"
        job_id = f"{prefix}-{media_id}-{uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": job_id,
            "job_id": job_id,
            "uuid": job_id,
            "operation": operation,
            "media_id": media_id,
            "status": status,
            "backend": "local",
            "created_at": now,
            "updated_at": now,
            "request": dict(request or {}),
            "result": dict(result),
        }
        self._local_media_jobs[job_id] = record
        return record

    def list_templates(
        self,
        *,
        include_builtin: bool = True,
        include_custom: bool = True,
        tags: Optional[Sequence[str]] = None,
        user_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        templates = [
            self._decorate_template_record(template)
            for template in list(self._require_chunking_service().get_all_templates(include_system=True) or [])
        ]
        if not include_builtin:
            templates = [template for template in templates if not bool(template.get("is_system", False))]
        if not include_custom:
            templates = [template for template in templates if bool(template.get("is_system", False))]
        if tags:
            requested_tags = {str(tag) for tag in tags if str(tag).strip()}
            templates = [
                template
                for template in templates
                if requested_tags.issubset(set(template.get("tags") or []))
            ]
        return templates

    def get_template(self, template_name: str) -> dict[str, Any]:
        template = self._require_chunking_service().get_template_by_name(template_name)
        if not template:
            raise ValueError(f"Chunking template '{template_name}' was not found.")
        return self._decorate_template_record(template)

    def create_template(
        self,
        *,
        name: str,
        description: str,
        template: Mapping[str, Any],
        tags: Optional[Sequence[str]] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        service = self._require_chunking_service()
        template_id = service.create_template(
            name=name,
            description=description,
            template_json=self._with_template_tags(template, tags),
        )
        return self._decorate_template_record(service.get_template_by_id(int(template_id)))

    def update_template(
        self,
        template_name: str,
        *,
        description: Optional[str] = None,
        template: Optional[Mapping[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        service = self._require_chunking_service()
        existing = self.get_template(template_name)
        template_payload: dict[str, Any] | None = None
        if template is not None:
            template_payload = self._with_template_tags(template, tags if tags is not None else existing.get("tags"))
        elif tags is not None:
            template_payload = self._with_template_tags(
                self._parse_template_config(existing.get("template_json")),
                tags,
            )
        service.update_template(
            int(existing["id"]),
            description=description,
            template_json=template_payload,
        )
        return self._decorate_template_record(service.get_template_by_id(int(existing["id"])))

    def delete_template(self, template_name: str, *, hard_delete: bool = False) -> None:
        existing = self.get_template(template_name)
        self._require_chunking_service().delete_template(int(existing["id"]))

    def get_template_diagnostics(self) -> dict[str, Any]:
        service = self._require_chunking_service()
        return {
            "db_class": f"{service.__class__.__module__}.{service.__class__.__name__}",
            "capability": "native",
            "missing_methods": [],
            "fallback_enabled": False,
            "hint": "Local chunking templates use the bundled chunking interop service.",
        }

    def apply_template(
        self,
        template_name: str,
        *,
        text: str,
        override_options: Optional[Mapping[str, Any]] = None,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        record = self.get_template(template_name)
        template_config = self._parse_template_config(record.get("template_json"))
        method, options = self._chunking_options_from_template(template_config)
        options.update(dict(override_options or {}))

        from ..Chunking.Chunk_Lib import Chunker

        chunker = Chunker(options=options, template_manager=object())
        raw_chunks = chunker.chunk_text(text, method=method, use_template=False)
        chunks = [
            chunk.get("text") if isinstance(chunk, Mapping) and "text" in chunk else chunk
            for chunk in list(raw_chunks or [])
        ]
        result: dict[str, Any] = {
            "template_name": template_name,
            "chunks": chunks,
        }
        if include_metadata:
            result["metadata"] = {
                "method": method,
                "options": options,
                "chunk_count": len(chunks),
                "tags": list(record.get("tags") or []),
            }
        return result

    @staticmethod
    def _chunking_options_from_template(template_config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        chunking = template_config.get("chunking")
        if isinstance(chunking, Mapping):
            method = str(chunking.get("method") or "words")
            config = dict(chunking.get("config") or {})
            return method, config

        pipeline = template_config.get("pipeline")
        if isinstance(pipeline, Sequence):
            for stage in pipeline:
                if not isinstance(stage, Mapping) or stage.get("stage") != "chunk":
                    continue
                method = str(stage.get("method") or template_config.get("base_method") or "words")
                return method, dict(stage.get("options") or {})

        method = str(template_config.get("base_method") or "words")
        metadata = template_config.get("metadata")
        options = dict(metadata.get("default_options") or {}) if isinstance(metadata, Mapping) else {}
        return method, options

    def list_collections(self) -> list[dict[str, Any]]:
        manager = self._build_chroma_manager()
        return [self._coerce_collection(collection) for collection in list(manager.list_collections() or [])]

    def get_collection_detail(self, collection_name: str) -> dict[str, Any]:
        collection = self._get_collection(collection_name)
        metadata = dict(getattr(collection, "metadata", {}) or {})

        try:
            count = int(collection.count())
        except Exception:
            count = 0

        return {
            "name": getattr(collection, "name", collection_name),
            "count": count,
            "embedding_dimension": self._infer_collection_dimension(collection, metadata),
            "metadata": metadata,
        }

    def export_collection(
        self,
        collection_name: str,
        *,
        include_embeddings: bool = True,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        collection = self._get_collection(collection_name)
        metadata = dict(getattr(collection, "metadata", {}) or {})
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")
        kwargs: dict[str, Any] = {"include": include}
        if limit is not None:
            kwargs["limit"] = int(limit)
        if offset is not None:
            kwargs["offset"] = int(offset)
        payload = dict(collection.get(**kwargs) or {})
        ids = list(payload.get("ids") or [])
        documents = list(payload.get("documents") or [])
        metadatas = list(payload.get("metadatas") or [])
        embeddings = list(payload.get("embeddings") or [])

        items = []
        for index, item_id in enumerate(ids):
            item: dict[str, Any] = {
                "id": item_id,
                "document": documents[index] if index < len(documents) else None,
                "metadata": metadatas[index] if index < len(metadatas) else {},
            }
            if include_embeddings:
                item["embedding"] = embeddings[index] if index < len(embeddings) else None
            items.append(item)

        try:
            count = int(collection.count())
        except Exception:
            count = len(items)

        return {
            "name": getattr(collection, "name", collection_name),
            "metadata": metadata,
            "count": count,
            "items": items,
            "include_embeddings": include_embeddings,
        }

    def delete_collection(self, collection_name: str) -> None:
        self._build_chroma_manager().delete_collection(collection_name)

    def reprocess_media(self, media_id: Any, **options: Any) -> Any:
        if self.media_service is None:
            raise ValueError("Local media reprocess backend is unavailable.")
        method = getattr(self.media_service, "reprocess_media", None)
        if not callable(method):
            raise ValueError("Local media reprocess backend is unavailable.")
        return method(media_id, **options)
