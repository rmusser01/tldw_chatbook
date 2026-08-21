"""Local retrieval-admin adapter for chunking templates and embedding collections.

The embedding-collection surface operates on the *shared* RAG vector store
(``RAG_Search/ingestion_indexing.get_shared_rag_service().vector_store``) —
the same persistent ChromaDB store that ingestion-time indexing writes and
RAG semantic search reads (task-248). The legacy per-user
``Embeddings/Chroma_Lib.ChromaDBManager`` stack this service previously
wrapped has been removed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ..Chunking.chunking_interop_library import get_chunking_service


class LocalRAGAdminService:
    """Wrap local chunking-template and shared-vector-store collection operations."""

    def __init__(
        self,
        media_db: Any,
        *,
        chunking_service: Any = None,
        vector_store: Any = None,
        media_service: Any = None,
    ):
        self.media_db = media_db
        self.chunking_service = chunking_service or (
            get_chunking_service(media_db) if media_db is not None else None
        )
        self._vector_store = vector_store
        self.media_service = media_service

    def _require_chunking_service(self) -> Any:
        if self.chunking_service is None:
            raise ValueError("Local chunking template backend is unavailable.")
        return self.chunking_service

    def _resolve_vector_store(self) -> Any:
        """Return the vector store RAG search reads (injected or shared service's)."""
        if self._vector_store is not None:
            return self._vector_store
        from ..RAG_Search.ingestion_indexing import get_shared_rag_service

        service = get_shared_rag_service()
        store = getattr(service, "vector_store", None) if service is not None else None
        if store is None:
            raise ValueError("Local embeddings backend is unavailable.")
        return store

    def _require_chroma_client(self) -> Any:
        """Return the raw ChromaDB client behind the shared store.

        Collection detail/export need raw collection access, which only the
        persistent ChromaDB store exposes (the in-memory fallback store has
        no ``client``).
        """
        client = getattr(self._resolve_vector_store(), "client", None)
        if client is None:
            raise ValueError(
                "Local embedding collections require the persistent ChromaDB vector store."
            )
        return client

    def _coerce_collection(self, collection: Any) -> dict[str, Any]:
        if isinstance(collection, str):
            # Some chromadb versions list collection names rather than objects.
            return {"name": collection, "metadata": {}}
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
    def _extract_template_tags(
        cls, record: Mapping[str, Any], template_config: Mapping[str, Any]
    ) -> list[str]:
        raw_tags = record.get("tags")
        if raw_tags is None:
            raw_tags = template_config.get("tags")
        if raw_tags is None:
            metadata = template_config.get("metadata")
            if isinstance(metadata, Mapping):
                raw_tags = metadata.get("tags")
        return cls._normalize_tags(raw_tags)

    def _decorate_template_record(
        self, record: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if not record:
            return {}
        decorated = dict(record)
        template_config = self._parse_template_config(
            decorated.get("template") or decorated.get("template_json")
        )
        decorated["tags"] = self._extract_template_tags(decorated, template_config)
        # (task 10, AC-24a) The listing surface carries validity DATA: a
        # stored-invalid template is listed WITH a flag rather than hidden
        # or silently applied. The flag is computed here (the data half);
        # where it renders is the UI task's. The validator never raises
        # (Task 6 contract: ``{valid, errors, warnings}``), so decoration
        # cannot break a listing.
        validated = {
            key: value
            for key, value in template_config.items()
            if key not in ("name", "description")
        }
        result = self._validate_template_body(validated)
        decorated["template_valid"] = bool(result["valid"])
        issues = [
            f"{issue['field']}: {issue['message']}"
            for issue in result["errors"]
        ]
        if issues:
            decorated["template_validation_errors"] = issues
        return decorated

    @staticmethod
    def _validate_template_body(body: Mapping[str, Any]) -> dict[str, Any]:
        """Run the server-parity validator on a template body.

        Never raises (the validator's own contract, Task 6): returns
        ``{"valid": bool, "errors": [...], "warnings": [...]}`` so callers
        can flag rather than crash. An un-runnable body (not a mapping)
        reports invalid instead of blowing up the listing/apply surface.
        """
        # Lazy: module scope would be circular (RAG_Admin imports this
        # module's package through the scope service).
        from .template_validation import validate_template

        if not isinstance(body, Mapping):
            return {
                "valid": False,
                "errors": [
                    {
                        "field": "template",
                        "message": "template body is not an object",
                    }
                ],
                "warnings": [],
            }
        return validate_template(dict(body))

    def _get_collection(self, collection_name: str) -> Any:
        return self._require_chroma_client().get_collection(name=collection_name)

    def _infer_collection_dimension(
        self, collection: Any, metadata: Mapping[str, Any]
    ) -> int | None:
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
        candidate = (
            first_bucket[0]
            if isinstance(first_bucket, list) and first_bucket
            else first_bucket
        )
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
        templates = [
            self._decorate_template_record(template)
            for template in list(
                self._require_chunking_service().get_all_templates(include_builtin=True)
                or []
            )
        ]
        if not include_builtin:
            templates = [
                template
                for template in templates
                if not bool(template.get("is_builtin", False))
            ]
        if not include_custom:
            templates = [
                template
                for template in templates
                if bool(template.get("is_builtin", False))
            ]
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
        # task-8: tags persist in the v7 ``tags`` column (the interop also
        # moves any body tags there), not embedded in the JSON body.
        template_id = service.create_template(
            name=name,
            description=description,
            template_json=dict(template),
            tags=self._normalize_tags(tags) if tags is not None else None,
        )
        return self._decorate_template_record(
            service.get_template_by_id(int(template_id))
        )

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
            tags=self._normalize_tags(tags) if tags is not None else None,
        )
        return self._decorate_template_record(
            service.get_template_by_id(int(existing["id"]))
        )

    def delete_template(self, template_name: str, *, hard_delete: bool = False) -> None:
        existing = self.get_template(template_name)
        self._require_chunking_service().delete_template(int(existing["id"]))

    def get_template_diagnostics(self) -> dict[str, Any]:
        service = self._require_chunking_service()
        diagnostics = {
            "db_class": f"{service.__class__.__module__}.{service.__class__.__name__}",
            "capability": "native",
            "missing_methods": [],
            "fallback_enabled": False,
            "hint": "Local chunking templates use the bundled chunking interop service.",
        }
        # task-12 (spec §8): the read-only legacy-chunk report rides this
        # diagnostics payload -- the surviving RAG Admin stats surface (the
        # legacy UI was deleted with PR #669; the scope service routes this
        # as the local admin "observe" action). Rendered only when legacy
        # chunks exist (N > 0): a fully stamped library shows nothing
        # (report-only-when-actionable choice; no pre-existing omit-empty
        # convention on this dict). Guarded so a media DB the
        # report can't query never breaks template diagnostics.
        try:
            report = self.get_legacy_chunk_report_line()
        except Exception:  # pragma: no cover - unmigrated/read-only media DB
            report = ""
        if report:
            diagnostics["legacy_chunk_report"] = report
        return diagnostics

    def get_legacy_chunk_report_line(self) -> str:
        """One read-only report line for the RAG Admin stats surface.

        Returns ``"Chunked by an older engine: N items"`` where N is the
        count of media items with live chunk rows persisted before the
        engine-version stamp (NULL ``chunk_engine_version``); an empty
        string when there are none
        (nothing to report -- the library is fully stamped). Read-only by
        construction (``count_chunks_by_engine_version`` is a plain SELECT;
        spec §8: stamp + report only, no re-chunk action).

        Raises:
            Whatever ``db.get_connection().execute`` raises (e.g. a media DB
            too old to have the column) -- callers composing this into a
            larger surface should guard, as ``get_template_diagnostics`` does.
        """
        if self.media_db is None:
            return ""
        legacy = self.count_chunks_by_engine_version(self.media_db).get(
            "legacy", 0
        )
        if legacy <= 0:
            return ""
        return f"Chunked by an older engine: {legacy} items"

    def apply_template(
        self,
        template_name: str,
        *,
        text: str,
        override_options: Optional[Mapping[str, Any]] = None,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        """Apply a stored template to text.

        (task 10, AC-24b) A stored-invalid template body is REFUSED here
        with the named :class:`InvalidTemplateError` -- never an unnamed
        engine error surfacing mid-chunk. The apply path cannot rely on
        validate-on-write alone: stored-invalid rows exist (v6→v7
        conversion can mint them; rows written before the gate existed),
        and they remain deliberately editable (update validates the NEW
        body only) -- so apply is the last line of defense.
        """
        record = self.get_template(template_name)
        template_config = self._parse_template_config(record.get("template_json"))
        validation = self._validate_template_body(
            {
                key: value
                for key, value in template_config.items()
                if key not in ("name", "description")
            }
        )
        if not validation["valid"]:
            from ..Chunking.chunking_interop_library import InvalidTemplateError

            summary = "; ".join(
                f"{issue['field']}: {issue['message']}"
                for issue in validation["errors"][:3]
            )
            raise InvalidTemplateError(
                f"Template '{template_name}' failed validation and was "
                f"refused: {summary}"
            )
        method, options = self._chunking_options_from_template(template_config)
        options.update(dict(override_options or {}))

        from ..Chunking.Chunk_Lib import Chunker

        chunker = Chunker(options=options, template_manager=object())
        raw_chunks = chunker.chunk_text(text, method=method, use_template=False)
        chunks = [
            chunk.get("text")
            if isinstance(chunk, Mapping) and "text" in chunk
            else chunk
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
    def _chunking_options_from_template(
        template_config: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
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
                method = str(
                    stage.get("method") or template_config.get("base_method") or "words"
                )
                return method, dict(stage.get("options") or {})

        method = str(template_config.get("base_method") or "words")
        metadata = template_config.get("metadata")
        options = (
            dict(metadata.get("default_options") or {})
            if isinstance(metadata, Mapping)
            else {}
        )
        return method, options

    def list_collections(self) -> list[dict[str, Any]]:
        client = self._require_chroma_client()
        return [
            self._coerce_collection(collection)
            for collection in list(client.list_collections() or [])
        ]

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
            "embedding_dimension": self._infer_collection_dimension(
                collection, metadata
            ),
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
                item["embedding"] = (
                    embeddings[index] if index < len(embeddings) else None
                )
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
        store = self._resolve_vector_store()
        deleter = getattr(store, "delete_collection", None)
        if not callable(deleter):
            raise ValueError("Local embeddings backend is unavailable.")
        # ChromaVectorStore.delete_collection also resets its cached handle
        # when the active collection is deleted, so delegate to the store.
        # Both bundled stores return a success bool (and log rather than
        # raise); surface an explicit False as a hard failure so the admin
        # seam never reports success for a collection that still exists.
        if deleter(collection_name) is False:
            raise ValueError(
                f"Failed to delete local embedding collection '{collection_name}'."
            )

    def reprocess_media(self, media_id: Any, **options: Any) -> Any:
        if self.media_service is None:
            raise ValueError("Local media reprocess backend is unavailable.")
        method = getattr(self.media_service, "reprocess_media", None)
        if not callable(method):
            raise ValueError("Local media reprocess backend is unavailable.")
        return method(media_id, **options)

    def count_chunks_by_engine_version(self, db: Any) -> dict[str, int]:
        """Count media items per chunking-engine version (read-only).

        task-12 (spec §8): the RAG Admin report surface. The spec's AC counts
        media *items*, not chunk rows — an item with many chunks counts once
        per version it appears under. Rows persisted before the
        engine-version stamp (schema v6 / task-11) carry NULL in
        ``UnvectorizedMediaChunks.chunk_engine_version`` and are reported
        under the ``"legacy"`` key; stamped rows are keyed by their version
        string verbatim. A partially stamped item counts under each version
        it has live rows for.

        Deliberately dependency-light: takes the media DB explicitly (no
        ``__init__`` state -- callable on a bare ``__new__`` instance) so the
        report never needs the chunking service or vector store backends,
        and never mutates anything (a plain SELECT; stamp + report only,
        there is no re-chunk action).

        Args:
            db: The ``MediaDatabase`` (or compatible) holding
                ``UnvectorizedMediaChunks``.

        Returns:
            dict mapping engine version (``"legacy"`` for NULL) to the count
            of media items with non-deleted chunk rows under that version.
            Empty dict when the table has no live rows.
        """
        cursor = db.get_connection().execute(
            "SELECT chunk_engine_version, COUNT(DISTINCT media_id) AS n "
            "FROM UnvectorizedMediaChunks WHERE deleted = 0 "
            "GROUP BY chunk_engine_version"
        )
        counts: dict[str, int] = {}
        for row in cursor.fetchall():
            version = row["chunk_engine_version"]
            counts[version if version is not None else "legacy"] = int(row["n"])
        return counts
