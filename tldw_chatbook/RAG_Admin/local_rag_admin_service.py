"""Local retrieval-admin adapter for chunking templates and embedding collections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

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
    ):
        self.media_db = media_db
        self.app_config = dict(app_config or {})
        self.user_id = str(user_id or self.app_config.get("USERS_NAME") or "default_user")
        self.chunking_service = chunking_service or (get_chunking_service(media_db) if media_db is not None else None)
        self._chroma_manager = chroma_manager

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

    def list_templates(
        self,
        *,
        include_builtin: bool = True,
        include_custom: bool = True,
        tags: Optional[Sequence[str]] = None,
        user_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        templates = list(self._require_chunking_service().get_all_templates(include_system=True) or [])
        if not include_builtin:
            templates = [template for template in templates if not bool(template.get("is_system", False))]
        if not include_custom:
            templates = [template for template in templates if bool(template.get("is_system", False))]
        if tags:
            return templates
        return templates

    def get_template(self, template_name: str) -> dict[str, Any]:
        template = self._require_chunking_service().get_template_by_name(template_name)
        if not template:
            raise ValueError(f"Chunking template '{template_name}' was not found.")
        return dict(template)

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
        template_id = service.create_template(name=name, description=description, template_json=dict(template))
        return service.get_template_by_id(int(template_id))

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
        service.update_template(
            int(existing["id"]),
            description=description,
            template_json=dict(template) if template is not None else None,
        )
        return service.get_template_by_id(int(existing["id"]))

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

    def delete_collection(self, collection_name: str) -> None:
        self._build_chroma_manager().delete_collection(collection_name)

    def get_media_embeddings_status(
        self,
        media_id: int,
        *,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        normalized_media_id = int(media_id)
        ids = self._embedding_ids_for_media(normalized_media_id, collection_name=collection_name)
        return {
            "media_id": normalized_media_id,
            "has_embeddings": bool(ids),
            "embedding_count": len(ids),
            "collection": self._default_collection_name(collection_name),
            "backend": "local",
        }

    def generate_media_embeddings(
        self,
        media_id: int,
        *,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force_regenerate: bool = False,
        priority: int = 50,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        normalized_media_id = int(media_id)
        row = self._get_media_for_embedding(normalized_media_id)
        content = str(row.get("content") or "").strip()
        if not content:
            raise ValueError(f"Local media item {normalized_media_id} has no embeddable content.")

        collection_name = self._default_collection_name(collection_name)
        if force_regenerate:
            self.delete_media_embeddings(normalized_media_id, collection_name=collection_name)

        title = str(row.get("title") or row.get("url") or f"media-{normalized_media_id}")
        chunk_options = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        self._build_chroma_manager().process_and_store_content(
            content=content,
            media_id=normalized_media_id,
            file_name=title,
            collection_name=collection_name,
            embedding_model_id_override=embedding_model,
            create_embeddings=True,
            chunk_options=chunk_options,
        )
        status = self.get_media_embeddings_status(normalized_media_id, collection_name=collection_name)
        return {
            "media_id": normalized_media_id,
            "status": "completed",
            "embeddings_started": True,
            "embedding_count": status["embedding_count"],
            "collection": collection_name,
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
            "priority": priority,
            "backend": "local",
        }

    def delete_media_embeddings(
        self,
        media_id: int,
        *,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        normalized_media_id = int(media_id)
        collection_name = self._default_collection_name(collection_name)
        ids = self._embedding_ids_for_media(normalized_media_id, collection_name=collection_name)
        self._build_chroma_manager().delete_from_collection(ids, collection_name=collection_name)
        return {
            "media_id": normalized_media_id,
            "status": "success",
            "deleted_count": len(ids),
            "collection": collection_name,
            "backend": "local",
        }
