"""Local chatbook adapter for source-aware prompt/chatbook parity."""

from __future__ import annotations

from contextlib import AbstractContextManager
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from tldw_chatbook.Chat.citation_artifact_ownership import (
    ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES,
    ArtifactBackendMode,
    ArtifactOwnerBinding,
    ArtifactOwnerOperation,
    ArtifactOwnerOutboxState,
    CitationArtifactOwnershipCoordinator,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationArtifactOwnerRequest,
    CitationPersistenceUnavailable,
)
from tldw_chatbook.Utils.atomic_file_ops import atomic_write_json

from .chatbook_creator import ChatbookCreator
from .chatbook_importer import ChatbookImporter
from .chatbook_models import ContentType
from .conflict_resolver import ConflictResolution


_REGISTRY_LOCKS_GUARD = threading.Lock()
_REGISTRY_LOCKS: dict[Path, threading.RLock] = {}
_SAFE_PROVENANCE_ERROR = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")
_STALE_OWNER_REQUEST_REASONS = frozenset(
    {"artifact_owner_request_invalid", "fingerprint_key_unavailable"}
)


def _registry_lock(path: Path) -> threading.RLock:
    resolved = path.resolve()
    with _REGISTRY_LOCKS_GUARD:
        return _REGISTRY_LOCKS.setdefault(resolved, threading.RLock())


class LocalChatbookService:
    """Expose local chatbook import/export operations through a service contract."""

    artifact_backend_mode = ArtifactBackendMode.CROSS_STORE
    artifact_store_id = "local-chatbook-json-v1"

    def __init__(
        self,
        db_paths: dict[str, str] | None = None,
        *,
        registry_path: str | Path | None = None,
    ):
        self.db_paths = db_paths or {}
        self.registry_path = (
            Path(registry_path).expanduser()
            if registry_path is not None
            else self._default_registry_path()
        )
        # Guards every load -> mutate -> save span against overlapping registry
        # read-modify-writes from concurrent OS threads (e.g. two overlapping
        # `asyncio.run(...)` exports on separate `@work(thread=True)` workers).
        self._registry_lock = _registry_lock(self.registry_path)
        self._citation_ownership_coordinator: (
            CitationArtifactOwnershipCoordinator | None
        ) = None

    def set_citation_ownership_coordinator(
        self,
        coordinator: CitationArtifactOwnershipCoordinator | None,
    ) -> None:
        """Install the cross-store coordinator after both stores are available."""

        if coordinator is not None and coordinator.artifact_store is not self:
            raise ValueError("citation ownership coordinator store mismatch")
        self._citation_ownership_coordinator = coordinator

    def provenance_collection_guard(self) -> AbstractContextManager[Any]:
        """Hold registry mutation stable through one trace collection."""

        return self._registry_lock

    def _default_registry_path(self) -> Path:
        for key in ("Prompts", "ChaChaNotes", "Media"):
            db_path = self.db_paths.get(key)
            if db_path:
                return (
                    Path(db_path).expanduser().with_name("tldw_chatbook_chatbooks.json")
                )
        return (
            Path.home()
            / ".local"
            / "share"
            / "tldw_cli"
            / "tldw_chatbook_chatbooks.json"
        )

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
            return {"next_id": 1, "records": [], "provenance_outbox": []}
        with self.registry_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid local chatbook registry: {self.registry_path}")
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(
                f"Invalid local chatbook registry records: {self.registry_path}"
            )
        raw_outbox = payload.get("provenance_outbox", [])
        if (
            not isinstance(raw_outbox, list)
            or len(raw_outbox) > ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES
        ):
            raise ValueError(
                f"Invalid local chatbook provenance_outbox: {self.registry_path}"
            )
        try:
            outbox = [
                ArtifactOwnerOperation.model_validate_json(
                    json.dumps(item),
                    strict=True,
                ).model_dump(mode="json")
                for item in raw_outbox
            ]
            normalized_records = []
            for raw_record in records:
                record = dict(raw_record)
                if "provenance_owner" in record:
                    binding = ArtifactOwnerBinding.model_validate_json(
                        json.dumps(record["provenance_owner"]),
                        strict=True,
                    )
                    record_id = str(record.get("chatbook_id") or record.get("id") or "")
                    record_revision = record.get("artifact_revision")
                    if (
                        binding.artifact_store_id != self.artifact_store_id
                        or binding.artifact_id != record_id
                        or isinstance(record_revision, bool)
                        or not isinstance(record_revision, int)
                        or record_revision != binding.artifact_revision
                    ):
                        raise ValueError("artifact owner binding mismatch")
                    record["provenance_owner"] = binding.model_dump(mode="json")
                normalized_records.append(record)
            next_id = int(payload.get("next_id") or 1)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid local chatbook provenance registry: {self.registry_path}"
            ) from None
        return {
            "next_id": next_id,
            "records": normalized_records,
            "provenance_outbox": outbox,
        }

    def _save_registry(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.registry_path, payload)

    def _find_record(
        self, registry: dict[str, Any], chatbook_id: int | str
    ) -> dict[str, Any]:
        wanted = str(chatbook_id)
        for record in registry["records"]:
            if (
                str(record.get("chatbook_id")) == wanted
                or str(record.get("id")) == wanted
            ):
                return record
        raise KeyError(f"Local chatbook not found: {chatbook_id}")

    @staticmethod
    def _record_copy(record: dict[str, Any]) -> dict[str, Any]:
        copied = dict(record)
        copied.pop("provenance_owner", None)
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
    def _normalize_content_selections(
        selections: dict[Any, list[Any]] | None,
    ) -> dict[ContentType, list[str]]:
        normalized: dict[ContentType, list[str]] = {}
        for content_type, ids in (selections or {}).items():
            key = (
                content_type
                if isinstance(content_type, ContentType)
                else ContentType(str(content_type))
            )
            normalized[key] = [str(item_id) for item_id in ids]
        return normalized

    async def preview_chatbook(self, chatbook_file_path: str | Path) -> dict[str, Any]:
        manifest, error = ChatbookImporter(self.db_paths).preview_chatbook(
            Path(chatbook_file_path)
        )
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
                or any(
                    query in category.lower()
                    for category in record.get("categories") or []
                )
            ]
        return records[int(offset) : int(offset) + int(limit)]

    @staticmethod
    def _is_console_saved_artifact(record: dict[str, Any]) -> bool:
        metadata = (
            record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        )
        return (
            str(metadata.get("artifact_source") or "").strip().lower() == "console"
            and str(metadata.get("artifact_kind") or "").strip().lower()
            == "assistant-response"
        )

    @classmethod
    def _home_artifact_sort_key(cls, record: dict[str, Any]) -> tuple[float, int]:
        timestamp = str(
            record.get("updated_at") or record.get("created_at") or ""
        ).strip()
        try:
            normalized = (
                timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
            )
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            timestamp_value = parsed.timestamp()
        except (TypeError, ValueError):
            timestamp_value = 0.0
        try:
            chatbook_id = int(record.get("chatbook_id") or record.get("id") or 0)
        except (TypeError, ValueError):
            chatbook_id = 0
        return timestamp_value, chatbook_id

    def list_home_artifact_snapshot(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return latest Console-saved Chatbook artifacts for synchronous Home rendering."""
        registry = self._load_registry()
        records = [
            self._record_copy(record)
            for record in registry["records"]
            if self._is_console_saved_artifact(record)
        ]
        records.sort(key=self._home_artifact_sort_key, reverse=True)
        return records[: int(limit)]

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
        provenance_owner_request: CitationArtifactOwnerRequest | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        with self._registry_lock:
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
                "artifact_revision": 1,
            }
            if extra:
                record["metadata"].update(
                    {key: value for key, value in extra.items() if value is not None}
                )
            coordinator = self._citation_ownership_coordinator
            if (
                provenance_owner_request is not None
                and coordinator is not None
                and coordinator.writes_enabled
            ):
                operation = self._prepare_optional_link(
                    coordinator,
                    provenance_owner_request,
                    artifact_id=str(chatbook_id),
                    artifact_revision=1,
                )
                if operation is not None:
                    self._append_provenance_operation(registry, operation)
                    record["provenance_owner"] = operation.binding.model_dump(
                        mode="json"
                    )
            registry["records"].append(record)
            registry["next_id"] = chatbook_id + 1
            self._save_registry(registry)
        return self._record_copy(record)

    async def update_chatbook(
        self,
        chatbook_id: int | str,
        *,
        provenance_owner_request: CitationArtifactOwnerRequest | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        with self._registry_lock:
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
            coordinator = self._citation_ownership_coordinator
            previous_binding = self._record_owner_binding(record)
            if previous_binding is not None:
                if coordinator is None or not coordinator.writes_enabled:
                    raise CitationPersistenceUnavailable(
                        "artifact_owner_reconciliation_unavailable"
                    )
                self._append_provenance_operation(
                    registry,
                    coordinator.prepare_unlink_operation(previous_binding),
                )
            can_link = (
                provenance_owner_request is not None
                and coordinator is not None
                and coordinator.writes_enabled
            )
            if previous_binding is not None or can_link:
                next_revision = int(record.get("artifact_revision") or 0) + 1
                link = (
                    self._prepare_optional_link(
                        coordinator,
                        provenance_owner_request,
                        artifact_id=str(record.get("chatbook_id") or record.get("id")),
                        artifact_revision=next_revision,
                    )
                    if can_link
                    else None
                )
                if link is None:
                    record.pop("provenance_owner", None)
                else:
                    self._append_provenance_operation(registry, link)
                    record["provenance_owner"] = link.binding.model_dump(mode="json")
                record["artifact_revision"] = next_revision
            record["updated_at"] = self._utc_now()
            self._save_registry(registry)
        return self._record_copy(record)

    async def delete_chatbook(self, chatbook_id: int | str) -> bool:
        with self._registry_lock:
            registry = self._load_registry()
            wanted = str(chatbook_id)
            record = self._find_record(registry, chatbook_id)
            binding = self._record_owner_binding(record)
            coordinator = self._citation_ownership_coordinator
            if (
                binding is not None
                and coordinator is not None
                and coordinator.writes_enabled
            ):
                try:
                    self._append_provenance_operation(
                        registry,
                        coordinator.prepare_unlink_operation(binding),
                    )
                except CitationPersistenceUnavailable:
                    # A tampered/imported binding is inert. Deleting the ordinary
                    # artifact remains available and cannot mutate trace state.
                    pass
            remaining = [
                record
                for record in registry["records"]
                if str(record.get("chatbook_id")) != wanted
                and str(record.get("id")) != wanted
            ]
            if len(remaining) == len(registry["records"]):
                raise KeyError(f"Local chatbook not found: {chatbook_id}")
            registry["records"] = remaining
            self._save_registry(registry)
        return True

    @staticmethod
    def _prepare_optional_link(
        coordinator: CitationArtifactOwnershipCoordinator,
        request: CitationArtifactOwnerRequest,
        *,
        artifact_id: str,
        artifact_revision: int,
    ) -> ArtifactOwnerOperation | None:
        try:
            return coordinator.prepare_link_operation(
                request,
                artifact_id=artifact_id,
                artifact_revision=artifact_revision,
            )
        except CitationPersistenceUnavailable as exc:
            if exc.reason_code in _STALE_OWNER_REQUEST_REASONS:
                return None
            raise

    @staticmethod
    def _record_owner_binding(
        record: dict[str, Any],
    ) -> ArtifactOwnerBinding | None:
        raw = record.get("provenance_owner")
        if raw is None:
            return None
        return ArtifactOwnerBinding.model_validate_json(
            json.dumps(raw),
            strict=True,
        )

    @staticmethod
    def _append_provenance_operation(
        registry: dict[str, Any],
        operation: ArtifactOwnerOperation,
    ) -> None:
        outbox = registry["provenance_outbox"]
        for existing in outbox:
            if existing["operation_id"] == operation.operation_id:
                current = ArtifactOwnerOperation.model_validate_json(
                    json.dumps(existing),
                    strict=True,
                )
                if (
                    current.operation_kind is not operation.operation_kind
                    or current.binding != operation.binding
                ):
                    raise ValueError("provenance operation identity conflict")
                return
        if len(outbox) >= ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES:
            raise ValueError("local chatbook provenance_outbox is full")
        outbox.append(operation.model_dump(mode="json"))

    def list_provenance_outbox(
        self,
        *,
        limit: int,
    ) -> list[ArtifactOwnerOperation]:
        """Read a bounded copy of the durable cross-store outbox."""

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES
        ):
            raise ValueError("provenance outbox limit is invalid")
        with self._registry_lock:
            registry = self._load_registry()
            return [
                ArtifactOwnerOperation.model_validate_json(
                    json.dumps(item),
                    strict=True,
                )
                for item in registry["provenance_outbox"][:limit]
            ]

    def mark_provenance_operation_acknowledged(self, operation_id: str) -> None:
        """Durably record trace-side application before release."""

        with self._registry_lock:
            registry = self._load_registry()
            for index, item in enumerate(registry["provenance_outbox"]):
                operation = ArtifactOwnerOperation.model_validate_json(
                    json.dumps(item),
                    strict=True,
                )
                if operation.operation_id != operation_id:
                    continue
                if operation.state is ArtifactOwnerOutboxState.ACKNOWLEDGED:
                    return
                registry["provenance_outbox"][index] = operation.model_copy(
                    update={
                        "state": ArtifactOwnerOutboxState.ACKNOWLEDGED,
                        "acknowledged_at": datetime.now(timezone.utc),
                        "error_code": None,
                    }
                ).model_dump(mode="json")
                self._save_registry(registry)
                return
            raise KeyError(f"Unknown provenance operation: {operation_id}")

    def prune_provenance_operation(self, operation_id: str) -> None:
        """Remove only an artifact-acknowledged, trace-finalized entry."""

        with self._registry_lock:
            registry = self._load_registry()
            kept = []
            found = False
            for item in registry["provenance_outbox"]:
                operation = ArtifactOwnerOperation.model_validate_json(
                    json.dumps(item),
                    strict=True,
                )
                if operation.operation_id != operation_id:
                    kept.append(item)
                    continue
                found = True
                if operation.state is not ArtifactOwnerOutboxState.ACKNOWLEDGED:
                    raise ValueError("pending provenance operation cannot be pruned")
            if not found:
                return
            registry["provenance_outbox"] = kept
            self._save_registry(registry)

    def record_provenance_operation_failure(
        self,
        operation_id: str,
        reason_code: str,
    ) -> None:
        """Persist only one bounded sanitized reason code."""

        bounded_reason = str(reason_code)
        if _SAFE_PROVENANCE_ERROR.fullmatch(bounded_reason) is None:
            bounded_reason = "artifact_reconciliation_failed"
        with self._registry_lock:
            registry = self._load_registry()
            for index, item in enumerate(registry["provenance_outbox"]):
                operation = ArtifactOwnerOperation.model_validate_json(
                    json.dumps(item),
                    strict=True,
                )
                if operation.operation_id != operation_id:
                    continue
                registry["provenance_outbox"][index] = operation.model_copy(
                    update={"error_code": bounded_reason}
                ).model_dump(mode="json")
                self._save_registry(registry)
                return

    async def export_chatbook(
        self,
        request_data: Any,
        *,
        progress_callback: Optional[Callable[[Any], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        payload = self._as_dict(request_data)
        output_path = payload.pop("output_path", None)
        if output_path is None:
            raise ValueError("output_path is required for local chatbook export.")

        creator = ChatbookCreator(self.db_paths)
        success, message, dependency_info = creator.create_chatbook(
            name=payload.get("name") or "Chatbook",
            description=payload.get("description") or "",
            content_selections=self._normalize_content_selections(
                payload.get("content_selections")
            ),
            output_path=Path(output_path),
            author=payload.get("author"),
            include_media=bool(payload.get("include_media", False)),
            media_quality=payload.get("media_quality") or "thumbnail",
            include_embeddings=bool(payload.get("include_embeddings", False)),
            tags=payload.get("tags") or [],
            categories=payload.get("categories") or [],
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        cancelled = (
            bool(dependency_info.get("cancelled", False))
            if isinstance(dependency_info, dict)
            else False
        )
        return {
            "success": success,
            "message": message,
            "path": str(output_path),
            "dependency_info": dependency_info,
            "name": payload.get("name") or Path(output_path).stem,
            "cancelled": cancelled,
        }

    async def import_chatbook(
        self, chatbook_file_path: str | Path, request_data: Any
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from tldw_chatbook.tldw_api.prompt_chatbook_schemas import ChatbookImportRequest

        payload = self._as_dict(request_data)
        conflict_value = payload.get(
            "conflict_resolution", ChatbookImportRequest().conflict_resolution
        )
        conflict_resolution = ConflictResolution(str(conflict_value))
        importer = ChatbookImporter(self.db_paths)
        success, message = importer.import_chatbook(
            Path(chatbook_file_path),
            content_selections=self._normalize_content_selections(
                payload.get("content_selections")
            ),
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
