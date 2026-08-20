"""Scope-aware routing for local and server-backed prompt operations."""

from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from ..DB.Prompts_DB import ConflictError, PromptsDatabase
from ..Library.library_content_evidence import LibraryContentEvidence
from ..runtime_policy.bootstrap import (
    build_runtime_api_client_provider_from_config,
    derive_configured_server_binding,
)
from ..runtime_policy.types import PolicyDeniedError

if TYPE_CHECKING:
    from ..tldw_api import PromptCreateRequest, TLDWAPIClient
from .prompt_artifact_codec import deserialize_definition
from .prompt_batch_models import (
    PromptBatchDeleteResult,
    PromptBatchRestoreResult,
    PromptBatchTarget,
    validate_prompt_batch_targets,
)
from .prompt_normalizers import (
    normalize_prompt_collection_list,
    normalize_prompt_collection_record,
    normalize_prompt_history_page,
    normalize_prompt_list,
    normalize_prompt_record,
    normalize_prompt_search,
    normalize_prompt_version_list,
    prepare_retained_snapshot_for_restore,
)
from .prompt_restore_errors import prompt_restore_error_from_conflict
from .prompt_source_capabilities import (
    PromptCapabilityError,
    PromptSourceCapabilities,
    local_prompt_capabilities,
    normalize_server_prompt_capabilities,
    validate_console_artifact_payload,
    validate_prompt_request_size,
)
from .server_prompt_adapter import normalize_artifact_type


_SQLITE_SIGNED_INTEGER_MAX = PromptsDatabase._SQLITE_SIGNED_INTEGER_MAX


def _positive_signed_id(value: Any, *, field_name: str) -> int:
    """Return one strict positive SQLite identifier."""
    if type(value) is not int or value < 1 or value > _SQLITE_SIGNED_INTEGER_MAX:
        raise ValueError(f"{field_name} must be a positive signed 64-bit integer.")
    return value


def _unique_positive_signed_ids(values: Any, *, field_name: str) -> tuple[int, ...]:
    """Return unique, strict positive SQLite identifiers in caller order."""
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must be a sequence of unique identifiers.")
    resolved = tuple(
        _positive_signed_id(value, field_name=field_name) for value in values
    )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field_name} must not contain duplicate identifiers.")
    return resolved


class PromptBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


def _normalize_collection_catalog_args(
    *, query: str, limit: int, offset: int
) -> tuple[str, int, int]:
    """Validate collection catalog inputs without consulting policy or storage."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer.")
    if type(offset) is not int or offset < 0 or offset > _SQLITE_SIGNED_INTEGER_MAX:
        raise ValueError("offset must be a non-negative signed 64-bit integer.")
    return query.strip(), limit, offset


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
    artifact_type: Optional[str] = None,
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
        "artifact_type": (
            normalize_artifact_type(artifact_type)
            if artifact_type is not None
            else None
        ),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _prompt_create_request_from_payload(payload: dict[str, Any]) -> PromptCreateRequest:
    # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
    from ..tldw_api import PromptCreateRequest

    if not payload.get("name"):
        raise ValueError("Prompt name is required for server prompt saves.")
    return PromptCreateRequest(**payload)


def _serialize_server_prompt_request(
    payload: dict[str, Any], *, for_update: bool
) -> dict[str, Any]:
    # Deferred import: keep the runtime API schema boundary out of module import.
    from ..tldw_api.prompt_chatbook_schemas import serialize_prompt_request

    return serialize_prompt_request(
        _prompt_create_request_from_payload(payload), for_update=for_update
    )


class ServerPromptService:
    """Thin prompt service around the shared server API client."""

    def __init__(
        self,
        client: TLDWAPIClient | None = None,
        *,
        client_provider: Any | None = None,
    ):
        self.client = client
        self.client_provider = client_provider

    @classmethod
    def from_config(
        cls,
        app_config: dict[str, Any],
        *,
        client_provider: Any | None = None,
    ) -> "ServerPromptService":
        if client_provider is not None:
            return cls(client=None, client_provider=client_provider)
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
        )

    @classmethod
    def from_server_context_provider(cls, provider: Any) -> "ServerPromptService":
        return cls(client_provider=provider)

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server prompt operations.")

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

    async def get_prompt(
        self, prompt_identifier: str | int, *, include_deleted: bool = False
    ) -> Any:
        return await self._require_client().get_prompt(
            prompt_identifier, include_deleted=include_deleted
        )

    async def create_prompt(self, payload: dict[str, Any]) -> Any:
        return await self._require_client().create_prompt(
            _prompt_create_request_from_payload(payload)
        )

    async def update_prompt(
        self, prompt_identifier: str | int, payload: dict[str, Any]
    ) -> Any:
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

    async def restore_prompt_version(
        self, prompt_identifier: str | int, version: int
    ) -> Any:
        return await self._require_client().restore_prompt_version(
            prompt_identifier, version
        )

    async def get_prompts_health(self) -> dict[str, Any]:
        return await self._require_client().get_prompts_health()

    async def search_prompts(self, **kwargs: Any) -> dict[str, Any]:
        return await self._require_client().search_prompts(**kwargs)

    async def create_prompt_collection(self, payload: dict[str, Any]) -> Any:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import PromptCollectionCreateRequest

        return await self._require_client().create_prompt_collection(
            PromptCollectionCreateRequest(**payload)
        )

    async def list_prompt_collections(
        self, *, limit: int = 200, offset: int = 0
    ) -> Any:
        return await self._require_client().list_prompt_collections(
            limit=limit, offset=offset
        )

    async def get_prompt_collection(self, collection_id: int) -> Any:
        return await self._require_client().get_prompt_collection(collection_id)

    async def update_prompt_collection(
        self, collection_id: int, payload: dict[str, Any]
    ) -> Any:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import PromptCollectionUpdateRequest

        return await self._require_client().update_prompt_collection(
            collection_id,
            PromptCollectionUpdateRequest(**payload),
        )


class LocalPromptService:
    """Adapter over the local prompts DB/interop API."""

    def __init__(self, prompt_db: Any):
        self.prompt_db = prompt_db

    def _require_collection_db(self) -> Any:
        if self.prompt_db is None or not hasattr(self.prompt_db, "get_connection"):
            raise ValueError("Local prompt collection backend is unavailable.")
        self._ensure_collection_schema()
        return self.prompt_db

    def _ensure_collection_schema(self) -> None:
        conn = self.prompt_db.get_connection()
        conn.create_function(
            "PY_CASEFOLD",
            1,
            lambda value: str(value).casefold(),
            deterministic=True,
        )
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS LocalPromptCollections (
                collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                version INTEGER NOT NULL DEFAULT 1,
                deleted INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS LocalPromptCollectionItems (
                collection_id INTEGER NOT NULL,
                prompt_id INTEGER NOT NULL,
                position INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (collection_id, prompt_id),
                FOREIGN KEY (collection_id) REFERENCES LocalPromptCollections(collection_id) ON DELETE CASCADE,
                FOREIGN KEY (prompt_id) REFERENCES Prompts(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_local_prompt_collections_deleted_name
            ON LocalPromptCollections(deleted, name COLLATE NOCASE);
            """
        )
        conn.commit()

    @staticmethod
    def _collection_display_name(
        *, name: str, collection_id: int, collision_count: int
    ) -> str:
        if collision_count > 1:
            return f"{name} · #{collection_id}"
        return name

    @classmethod
    def _collection_mapping(cls, row: Any, *, prompt_ids: list[int]) -> dict[str, Any]:
        collection_id = int(row["collection_id"])
        name = str(row["name"])
        return {
            "collection_id": collection_id,
            "name": name,
            "display_name": cls._collection_display_name(
                name=name,
                collection_id=collection_id,
                collision_count=int(row["collision_count"]),
            ),
            "description": row["description"],
            "prompt_ids": prompt_ids,
        }

    @staticmethod
    def _reject_reserved_name_collision(conn: sqlite3.Connection, *, name: str) -> None:
        collision = conn.execute(
            """
            SELECT 1
            FROM LocalPromptCollections
            WHERE PY_CASEFOLD(name) = ?
            LIMIT 1
            """,
            (name.casefold(),),
        ).fetchone()
        if collision is not None:
            raise ValueError("This prompt collection name is reserved.")

    @staticmethod
    def _collection_id(collection_id: int | str) -> int:
        if type(collection_id) is int:
            resolved = collection_id
        elif isinstance(collection_id, str):
            try:
                resolved = int(collection_id)
            except ValueError as exc:
                raise ValueError("Invalid prompt collection id.") from exc
        else:
            raise ValueError("Invalid prompt collection id.")
        if resolved < 1 or resolved > _SQLITE_SIGNED_INTEGER_MAX:
            raise ValueError("Invalid prompt collection id.")
        return resolved

    @staticmethod
    def _prompt_ids(prompt_ids: Optional[Sequence[int]]) -> list[int]:
        if prompt_ids is None:
            return []
        return list(_unique_positive_signed_ids(prompt_ids, field_name="prompt_ids"))

    @staticmethod
    def _require_active_prompt_ids(
        conn: sqlite3.Connection, prompt_ids: Sequence[int]
    ) -> None:
        for prompt_id in prompt_ids:
            active = conn.execute(
                "SELECT 1 FROM Prompts WHERE id = ? AND deleted = 0",
                (prompt_id,),
            ).fetchone()
            if active is None:
                raise ValueError(
                    "Each prompt reference must identify an active Prompt."
                )

    @staticmethod
    def _require_active_collection_ids(
        conn: sqlite3.Connection, collection_ids: Sequence[int]
    ) -> None:
        for collection_id in collection_ids:
            active = conn.execute(
                """
                SELECT 1
                FROM LocalPromptCollections
                WHERE collection_id = ? AND deleted = 0
                """,
                (collection_id,),
            ).fetchone()
            if active is None:
                raise ValueError(
                    "Prompt memberships must reference active collections."
                )

    def _set_collection_prompt_ids(
        self, conn: sqlite3.Connection, collection_id: int, prompt_ids: list[int]
    ) -> None:
        conn.execute(
            "DELETE FROM LocalPromptCollectionItems WHERE collection_id = ?",
            (collection_id,),
        )
        conn.executemany(
            """
            INSERT INTO LocalPromptCollectionItems (collection_id, prompt_id, position)
            VALUES (?, ?, ?)
            """,
            [
                (collection_id, prompt_id, index)
                for index, prompt_id in enumerate(prompt_ids)
            ],
        )

    def _collection_record(self, collection_id: int) -> dict[str, Any]:
        db = self._require_collection_db()
        conn = db.get_connection()
        row = conn.execute(
            """
            SELECT collection.collection_id,
                   collection.name,
                   collection.description,
                   (
                       SELECT COUNT(*)
                       FROM LocalPromptCollections AS active
                       WHERE active.deleted = 0
                         AND PY_CASEFOLD(active.name) = PY_CASEFOLD(collection.name)
                   ) AS collision_count
            FROM LocalPromptCollections AS collection
            WHERE collection.collection_id = ? AND collection.deleted = 0
            """,
            (collection_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Prompt collection '{collection_id}' not found.")
        prompt_rows = conn.execute(
            """
            SELECT prompt_id
            FROM LocalPromptCollectionItems
            WHERE collection_id = ?
            ORDER BY position ASC, prompt_id ASC
            """,
            (collection_id,),
        ).fetchall()
        return self._collection_mapping(
            row,
            prompt_ids=[int(prompt_row["prompt_id"]) for prompt_row in prompt_rows],
        )

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

    def browse_prompts(
        self,
        *,
        query: str = "",
        collection_id: int | None = None,
        sort_by: str = "last_modified",
        sort_order: str = "desc",
        page: int = 1,
        page_size: int = 50,
    ) -> Any:
        """Delegate one exact browse page to the local database.

        Args:
            query: Prompt search text passed to the database.
            collection_id: Optional local collection identifier.
            sort_by: Database browse sort field.
            sort_order: Database browse sort direction.
            page: Requested one-based page.
            page_size: Requested number of rows per page.

        Returns:
            The raw response from ``prompt_db.browse_prompts``.
        """
        return self.prompt_db.browse_prompts(
            query=query,
            collection_id=collection_id,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
        )

    def count_prompts(self, *, include_deleted: bool = False, **_kwargs: Any) -> int:
        """Count local prompts without fetching a full page.

        Mirrors ``list_prompts`` above: fetches a single row
        (``per_page=1``) purely to read the paginated response's exact
        total.

        Args:
            include_deleted: Whether to include soft-deleted prompts.
            **_kwargs: Accepted and ignored, mirroring ``list_prompts``'s
                permissive signature so callers can forward the same
                kwargs (e.g. ``mode``) uniformly.

        Returns:
            The exact number of matching prompts.
        """
        _prompts, _total_pages, _current_page, total_items = (
            self.prompt_db.list_prompts(
                page=1,
                per_page=1,
                include_deleted=include_deleted,
            )
        )
        return total_items

    def get_prompt(
        self, prompt_identifier: str | int, *, include_deleted: bool = False
    ) -> Any:
        if hasattr(self.prompt_db, "fetch_prompt_details"):
            return self.prompt_db.fetch_prompt_details(
                prompt_identifier, include_deleted=include_deleted
            )
        return self.prompt_db.get_prompt(
            prompt_identifier, include_deleted=include_deleted
        )

    def search_prompts(
        self,
        *,
        query: str,
        limit: int = 10,
        include_deleted: bool = False,
        fts_match_query: Optional[str] = None,
        **_kwargs: Any,
    ) -> Any:
        """Search local prompts via the prompts FTS index.

        Mirrors ``list_prompts``/``count_prompts`` above: delegates straight
        to ``PromptsDatabase.search_prompts``, requesting a single page
        sized to ``limit`` results.

        Args:
            query: Plain user query text, forwarded as ``search_query``
                (used verbatim as the FTS MATCH expression when
                ``fts_match_query`` is not provided).
            limit: Maximum number of prompts to return.
            include_deleted: Whether to include soft-deleted prompts.
            fts_match_query: Optional pre-built FTS5 MATCH string (e.g.
                Library keyword search's plural/singular-widened query)
                overriding the MATCH clause built from ``query``.
            **_kwargs: Accepted and ignored, mirroring ``list_prompts``'s
                permissive signature.

        Returns:
            The list of matching prompt dicts (keywords already attached),
            per ``PromptsDatabase.search_prompts``'s first tuple element.
        """
        fts_kwargs = (
            {"fts_match_query": fts_match_query} if fts_match_query is not None else {}
        )
        results, _total_matches = self.prompt_db.search_prompts(
            search_query=query,
            page=1,
            results_per_page=max(1, int(limit)),
            include_deleted=include_deleted,
            **fts_kwargs,
        )
        return results

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
            artifact_type=payload.get("artifact_type"),
        )
        identifier = prompt_uuid or prompt_id
        return self.get_prompt(identifier, include_deleted=True)

    def update_prompt(
        self, prompt_identifier: str | int, payload: dict[str, Any]
    ) -> Any:
        existing = self.get_prompt(prompt_identifier, include_deleted=True)
        if not existing:
            raise ValueError(f"Prompt '{prompt_identifier}' not found.")

        if hasattr(self.prompt_db, "update_prompt_by_id"):
            expected_version = payload.get("expected_version")
            update_payload = {
                key: value
                for key, value in payload.items()
                if key != "expected_version"
            }
            prompt_uuid, _message = self.prompt_db.update_prompt_by_id(
                existing["id"], update_payload, expected_version=expected_version
            )
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
            prompt_schema_version=payload.get(
                "prompt_schema_version", existing.get("prompt_schema_version")
            ),
            prompt_definition=payload.get(
                "prompt_definition", existing.get("prompt_definition")
            ),
            artifact_type=payload.get("artifact_type", existing.get("artifact_type")),
        )
        return self.get_prompt(prompt_uuid or prompt_id, include_deleted=True)

    def delete_prompt(
        self,
        prompt_identifier: str | int,
        *,
        expected_version: int | None = None,
    ) -> Any:
        """Soft-delete one local Prompt/Recipe conditionally.

        Args:
            prompt_identifier: Numeric id, UUID, or name of the artifact.
            expected_version: Optional active-row version required for deletion.

        Returns:
            The local database delete result.

        Raises:
            InputError: If the identifier or expected version is invalid.
            ConflictError: If no active artifact matches the identifier.
            ExpectedVersionConflictError: If the expected version is stale.
            DatabaseError: If persistence fails.
        """
        return self.prompt_db.soft_delete_prompt(
            prompt_identifier, expected_version=expected_version
        )

    def delete_prompts(
        self, *, targets: tuple[PromptBatchTarget, ...]
    ) -> PromptBatchDeleteResult:
        """Delete one strict local Prompt batch atomically.

        Args:
            targets: Validated Prompt IDs and expected active versions.

        Returns:
            The exact committed database receipt.

        Raises:
            TypeError: If the target container or entries have invalid types.
            ValueError: If targets are empty, duplicated, or invalid.
            ConflictError: If any active Prompt target is missing or stale.
            DatabaseError: If transaction ownership or persistence fails.
        """
        validated_targets = validate_prompt_batch_targets(targets)
        return self.prompt_db.soft_delete_prompts(validated_targets)

    def restore_deleted_prompt(
        self, prompt_identifier: str | int, *, expected_version: int
    ) -> Any:
        """Restore one exact local Prompt/Recipe tombstone.

        Args:
            prompt_identifier: Numeric id, UUID, or name of the tombstone.
            expected_version: Exact deleted-row version required for restore.

        Returns:
            The restored Prompt/Recipe record with canonical keywords.

        Raises:
            InputError: If the identifier or expected version is invalid.
            ExpectedVersionConflictError: If the tombstone version is stale.
            DatabaseError: If recovery metadata is unavailable or persistence fails.
        """
        return self.prompt_db.restore_deleted_prompt(
            prompt_identifier, expected_version=expected_version
        )

    def restore_deleted_prompts(
        self, *, targets: tuple[PromptBatchTarget, ...]
    ) -> PromptBatchRestoreResult:
        """Restore one strict local Prompt batch atomically.

        Args:
            targets: Validated Prompt IDs and exact tombstone versions.

        Returns:
            The exact committed database restore result.

        Raises:
            TypeError: If the target container or entries have invalid types.
            ValueError: If targets are empty, duplicated, or invalid.
            ConflictError: If any Prompt tombstone is missing or stale.
            DatabaseError: If recovery metadata or persistence is unavailable.
        """
        validated_targets = validate_prompt_batch_targets(targets)
        return self.prompt_db.restore_deleted_prompts(validated_targets)

    def record_prompt_usage(self, prompt_identifier: str | int) -> Any:
        if hasattr(self.prompt_db, "record_prompt_usage"):
            return self.prompt_db.record_prompt_usage(prompt_identifier)
        return self.get_prompt(prompt_identifier, include_deleted=True)

    def count_prompt_versions(self, prompt_identifier: str | int) -> int:
        """Return the exact retained create/update count for one local Prompt."""
        prompt = self.get_prompt(prompt_identifier, include_deleted=True)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_identifier}' not found.")
        prompt_uuid = prompt.get("uuid")
        if not prompt_uuid:
            raise ValueError(f"Prompt '{prompt_identifier}' has no UUID.")
        count = self.prompt_db.get_prompt_history_count(prompt_uuid)
        if type(count) is not int or count < 0:
            raise ValueError(
                "Local retained history count must be a non-negative integer."
            )
        return count

    def list_prompt_versions(
        self,
        prompt_identifier: str | int,
        *,
        page_size: int = 25,
        before_change_id: int | None = None,
    ) -> dict[str, Any]:
        prompt = self.get_prompt(prompt_identifier, include_deleted=True)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_identifier}' not found.")
        prompt_uuid = prompt.get("uuid")
        if not prompt_uuid:
            raise ValueError(f"Prompt '{prompt_identifier}' has no UUID.")
        return self.prompt_db.get_prompt_history_entries(
            prompt_uuid,
            page_size=page_size,
            before_change_id=before_change_id,
        )

    def restore_prompt_version(
        self,
        prompt_identifier: str | int,
        *,
        change_id: int,
        version: int,
        expected_version: int,
    ) -> dict[str, Any]:
        prompt = self.get_prompt(prompt_identifier, include_deleted=True)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_identifier}' not found.")
        prompt_uuid = prompt.get("uuid")
        if not prompt_uuid:
            raise ValueError(f"Prompt '{prompt_identifier}' has no UUID.")
        try:
            return self.prompt_db.restore_prompt_history_entry(
                prompt_uuid,
                change_id=change_id,
                version=version,
                expected_version=expected_version,
                snapshot_validator=lambda snapshot: (
                    prepare_retained_snapshot_for_restore(
                        snapshot,
                        capabilities=local_prompt_capabilities(),
                    )
                ),
            )
        except ConflictError as exc:
            classified = prompt_restore_error_from_conflict(exc)
            if classified is not None:
                raise classified from exc
            raise

    def create_prompt_collection(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create one local collection under a serialized case-fold name guard.

        Args:
            payload: Collection name, optional description, and optional Prompt IDs.

        Returns:
            A mapping containing the new positive ``collection_id``.

        Raises:
            ValueError: If the name or Prompt references are invalid, or a stored
                collection reserves the same case-folded name.
        """
        db = self._require_collection_db()
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("Prompt collection name is required.")
        description = payload.get("description")
        prompt_ids = self._prompt_ids(payload.get("prompt_ids"))
        try:
            with db.transaction(immediate=True) as conn:
                self._reject_reserved_name_collision(conn, name=name)
                cursor = conn.execute(
                    """
                    INSERT INTO LocalPromptCollections (name, description)
                    VALUES (?, ?)
                    """,
                    (name, description),
                )
                collection_id = int(cursor.lastrowid)
                self._require_active_prompt_ids(conn, prompt_ids)
                self._set_collection_prompt_ids(conn, collection_id, prompt_ids)
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "Prompt collection creation failed because a name or Prompt reference is invalid."
            ) from exc
        return {"collection_id": collection_id}

    def list_prompt_collections(
        self, *, query: str = "", limit: int = 200, offset: int = 0
    ) -> dict[str, Any]:
        """Return one exact, deterministically ordered local collection page.

        Args:
            query: Literal collection-name substring matched with Python case-fold
                semantics after trimming.
            limit: Positive requested page size, capped at 100.
            offset: Non-negative row offset within the filtered catalog.

        Returns:
            A mapping containing normalized ``limit``/``offset``, exact filtered
            ``total``, and collection records with stable IDs and display names.

        Raises:
            TypeError: If ``query`` is not a string.
            ValueError: If ``limit`` or ``offset`` is outside its accepted bounds,
                or the local collection backend is unavailable.
        """
        query, limit, offset = _normalize_collection_catalog_args(
            query=query, limit=limit, offset=offset
        )
        limit = min(limit, 100)
        db = self._require_collection_db()
        folded_query = query.casefold()
        with db.transaction() as conn:
            total = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM LocalPromptCollections
                    WHERE deleted = 0
                      AND instr(PY_CASEFOLD(name), ?) > 0
                    """,
                    (folded_query,),
                ).fetchone()[0]
            )
            rows = conn.execute(
                """
                WITH active AS (
                    SELECT collection_id,
                           name,
                           description,
                           PY_CASEFOLD(name) AS folded_name
                    FROM LocalPromptCollections
                    WHERE deleted = 0
                ),
                collision_counts AS (
                    SELECT folded_name, COUNT(*) AS collision_count
                    FROM active
                    GROUP BY folded_name
                )
                SELECT active.collection_id,
                       active.name,
                       active.description,
                       collision_counts.collision_count
                FROM active
                JOIN collision_counts USING (folded_name)
                WHERE instr(active.folded_name, ?) > 0
                ORDER BY active.folded_name ASC, active.collection_id ASC
                LIMIT ? OFFSET ?
                """,
                (folded_query, limit, offset),
            ).fetchall()
            collection_ids = [int(row["collection_id"]) for row in rows]
            prompt_ids_by_collection: dict[int, list[int]] = {
                collection_id: [] for collection_id in collection_ids
            }
            if collection_ids:
                placeholders = ", ".join("?" for _ in collection_ids)
                prompt_rows = conn.execute(
                    f"""
                    SELECT collection_id, prompt_id
                    FROM LocalPromptCollectionItems
                    WHERE collection_id IN ({placeholders})
                    ORDER BY collection_id ASC, position ASC, prompt_id ASC
                    """,
                    collection_ids,
                ).fetchall()
                for prompt_row in prompt_rows:
                    prompt_ids_by_collection[int(prompt_row["collection_id"])].append(
                        int(prompt_row["prompt_id"])
                    )
        collections = [
            self._collection_mapping(
                row,
                prompt_ids=prompt_ids_by_collection[int(row["collection_id"])],
            )
            for row in rows
        ]
        return {
            "collections": collections,
            "limit": limit,
            "offset": offset,
            "total": total,
        }

    def get_prompt_collection(self, collection_id: int) -> dict[str, Any]:
        """Return one active local collection selected strictly by ID.

        Args:
            collection_id: Positive local collection identifier.

        Returns:
            The collection record with its Prompt IDs and collision-aware label.

        Raises:
            ValueError: If the identifier is invalid or the collection is inactive.
        """
        return self._collection_record(self._collection_id(collection_id))

    def update_prompt_collection(
        self, collection_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Update one local collection selected strictly by ID.

        Args:
            collection_id: Positive local collection identifier.
            payload: Name, description, and/or replacement Prompt IDs.

        Returns:
            The updated collection record.

        Raises:
            ValueError: If the collection, name, or Prompt references are invalid,
                or a stored collection reserves the same case-folded name.
        """
        resolved_collection_id = self._collection_id(collection_id)
        db = self._require_collection_db()
        updates = {
            key: payload[key] for key in ("name", "description") if key in payload
        }
        if "name" in updates:
            updates["name"] = str(updates["name"] or "").strip()
            if not updates["name"]:
                raise ValueError("Prompt collection name is required.")
        prompt_ids = (
            self._prompt_ids(payload.get("prompt_ids"))
            if "prompt_ids" in payload
            else None
        )
        try:
            with db.transaction(immediate=True) as conn:
                target = conn.execute(
                    """
                    SELECT name
                    FROM LocalPromptCollections
                    WHERE collection_id = ? AND deleted = 0
                    """,
                    (resolved_collection_id,),
                ).fetchone()
                if target is None:
                    raise ValueError("Prompt collection not found.")
                if (
                    "name" in updates
                    and updates["name"].casefold() != str(target["name"]).casefold()
                ):
                    self._reject_reserved_name_collision(conn, name=updates["name"])
                if updates:
                    set_clause = ", ".join(f"{key} = ?" for key in updates)
                    params = list(updates.values()) + [resolved_collection_id]
                    cursor = conn.execute(
                        f"""
                        UPDATE LocalPromptCollections
                        SET {set_clause}, updated_at = CURRENT_TIMESTAMP, version = version + 1
                        WHERE collection_id = ? AND deleted = 0
                        """,
                        params,
                    )
                    if cursor.rowcount == 0:
                        raise ValueError(
                            f"Prompt collection '{collection_id}' not found."
                        )
                if prompt_ids is not None:
                    self._require_active_prompt_ids(conn, prompt_ids)
                    self._set_collection_prompt_ids(
                        conn, resolved_collection_id, prompt_ids
                    )
                    if not updates:
                        cursor = conn.execute(
                            """
                            UPDATE LocalPromptCollections
                            SET updated_at = CURRENT_TIMESTAMP, version = version + 1
                            WHERE collection_id = ? AND deleted = 0
                            """,
                            (resolved_collection_id,),
                        )
                        if cursor.rowcount == 0:
                            raise ValueError(
                                f"Prompt collection '{collection_id}' not found."
                            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "Prompt collection update failed because a name or prompt reference is invalid."
            ) from exc
        return self._collection_record(resolved_collection_id)

    def list_prompt_collection_memberships(
        self, prompt_id: int
    ) -> dict[str, int | tuple[int, ...] | bool]:
        """List active collection memberships for one active local Prompt.

        Args:
            prompt_id: Positive local Prompt identifier.

        Returns:
            A bounded mapping with the Prompt ID, ordered collection IDs, and
            ``changed=False``.

        Raises:
            ValueError: If the Prompt identifier is invalid or inactive.
        """
        resolved_prompt_id = _positive_signed_id(prompt_id, field_name="prompt_id")
        db = self._require_collection_db()
        with db.transaction() as conn:
            self._require_active_prompt_ids(conn, (resolved_prompt_id,))
            rows = conn.execute(
                """
                SELECT item.collection_id
                FROM LocalPromptCollectionItems AS item
                JOIN LocalPromptCollections AS collection
                  ON collection.collection_id = item.collection_id
                WHERE item.prompt_id = ? AND collection.deleted = 0
                ORDER BY item.collection_id ASC
                """,
                (resolved_prompt_id,),
            ).fetchall()
        return {
            "prompt_id": resolved_prompt_id,
            "collection_ids": tuple(int(row["collection_id"]) for row in rows),
            "changed": False,
        }

    def replace_prompt_collection_memberships(
        self, prompt_id: int, collection_ids: Sequence[int]
    ) -> dict[str, int | tuple[int, ...] | bool]:
        """Atomically replace one active Prompt's local collection memberships.

        Args:
            prompt_id: Positive local Prompt identifier.
            collection_ids: Unique positive active collection identifiers.

        Returns:
            A bounded mapping containing the normalized membership set and whether
            it changed.

        Raises:
            ValueError: If an identifier is invalid or inactive, or persistence
                rejects the membership update.
        """
        resolved_prompt_id = _positive_signed_id(prompt_id, field_name="prompt_id")
        resolved_collection_ids = tuple(
            sorted(
                _unique_positive_signed_ids(collection_ids, field_name="collection_ids")
            )
        )
        db = self._require_collection_db()
        try:
            with db.transaction(immediate=True) as conn:
                self._require_active_prompt_ids(conn, (resolved_prompt_id,))
                self._require_active_collection_ids(conn, resolved_collection_ids)
                current_collection_ids = tuple(
                    int(row["collection_id"])
                    for row in conn.execute(
                        """
                        SELECT collection_id
                        FROM LocalPromptCollectionItems
                        WHERE prompt_id = ?
                        ORDER BY collection_id ASC
                        """,
                        (resolved_prompt_id,),
                    ).fetchall()
                )
                if current_collection_ids == resolved_collection_ids:
                    return {
                        "prompt_id": resolved_prompt_id,
                        "collection_ids": resolved_collection_ids,
                        "changed": False,
                    }

                current = set(current_collection_ids)
                requested = set(resolved_collection_ids)
                removed = tuple(sorted(current - requested))
                added = tuple(sorted(requested - current))

                if removed:
                    placeholders = ", ".join("?" for _ in removed)
                    conn.execute(
                        f"""
                        DELETE FROM LocalPromptCollectionItems
                        WHERE prompt_id = ?
                          AND collection_id IN ({placeholders})
                        """,
                        (resolved_prompt_id, *removed),
                    )
                for collection_id in added:
                    position = int(
                        conn.execute(
                            """
                            SELECT COALESCE(MAX(position), -1) + 1
                            FROM LocalPromptCollectionItems
                            WHERE collection_id = ?
                            """,
                            (collection_id,),
                        ).fetchone()[0]
                    )
                    conn.execute(
                        """
                        INSERT INTO LocalPromptCollectionItems
                            (collection_id, prompt_id, position)
                        VALUES (?, ?, ?)
                        """,
                        (collection_id, resolved_prompt_id, position),
                    )

                impacted = tuple(sorted(set(removed) | set(added)))
                if impacted:
                    placeholders = ", ".join("?" for _ in impacted)
                    conn.execute(
                        f"""
                        UPDATE LocalPromptCollections
                        SET updated_at = CURRENT_TIMESTAMP, version = version + 1
                        WHERE deleted = 0
                          AND collection_id IN ({placeholders})
                        """,
                        impacted,
                    )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Prompt collection membership update failed.") from exc
        return {
            "prompt_id": resolved_prompt_id,
            "collection_ids": resolved_collection_ids,
            "changed": True,
        }


class PromptScopeService:
    """Route prompt actions to the active local/server backend and normalize outputs."""

    def __init__(
        self, local_service: Any, server_service: Any, policy_enforcer: Any = None
    ):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer
        self._server_capabilities_cache: PromptSourceCapabilities | None = None

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

    async def get_capabilities(
        self, *, mode: PromptBackend | str
    ) -> PromptSourceCapabilities:
        """Return truthful, immutable capabilities for the selected source."""
        normalized_mode = self._normalize_mode(mode)
        self._service_for_mode(normalized_mode)
        if normalized_mode == PromptBackend.LOCAL:
            return local_prompt_capabilities()
        if self._server_capabilities_cache is not None:
            return self._server_capabilities_cache

        try:
            health = await self._maybe_await(self.server_service.get_prompts_health())
        except Exception:
            # A transient health failure must not invent capabilities or poison retries.
            return normalize_server_prompt_capabilities(None)
        capabilities = normalize_server_prompt_capabilities(health)
        self._server_capabilities_cache = capabilities
        return capabilities

    @staticmethod
    def _source_record(record: Any) -> Mapping[str, Any]:
        if hasattr(record, "model_dump"):
            return record.model_dump(mode="json")
        return record if isinstance(record, Mapping) else {}

    @classmethod
    def _normalize_prompt_record(cls, record: Any, *, backend: str) -> dict[str, Any]:
        """Keep new artifact fields while retaining the established normalizer shape."""
        normalized = normalize_prompt_record(record, backend=backend)
        source = cls._source_record(record)
        system_flag = source.get("has_system_prompt")
        user_flag = source.get("has_user_prompt")
        normalized["has_system_prompt"] = (
            bool(system_flag)
            if isinstance(system_flag, (bool, int)) and system_flag in (0, 1)
            else bool(str(source.get("system_prompt") or "").strip())
        )
        normalized["has_user_prompt"] = (
            bool(user_flag)
            if isinstance(user_flag, (bool, int)) and user_flag in (0, 1)
            else bool(str(source.get("user_prompt") or "").strip())
        )
        return normalized

    @classmethod
    def _normalize_prompt_list(
        cls, response: Any, *, backend: str, page: int, per_page: int
    ) -> dict[str, Any]:
        normalized = normalize_prompt_list(
            response, backend=backend, page=page, per_page=per_page
        )
        if isinstance(response, tuple) and len(response) == 4:
            source_items = response[0]
        else:
            source_items = cls._source_record(response).get("items", [])
        normalized["items"] = [
            cls._normalize_prompt_record(item, backend=backend) for item in source_items
        ]
        return normalized

    @staticmethod
    def _action_id(mode: PromptBackend, action: str) -> str:
        return f"prompts.{action}.{mode.value}"

    @staticmethod
    def _collection_action_id(mode: PromptBackend, action: str) -> str:
        return f"prompts.collections.{action}.{mode.value}"

    @staticmethod
    def _local_membership_mode(mode: PromptBackend | str) -> PromptBackend:
        if mode != PromptBackend.LOCAL and mode != PromptBackend.LOCAL.value:
            raise ValueError("Prompt collection memberships are local-only.")
        return PromptBackend.LOCAL

    @staticmethod
    def _membership_outcome(
        response: Any, *, prompt_id: int, changed: bool | None = None
    ) -> dict[str, int | tuple[int, ...] | bool]:
        if not isinstance(response, Mapping):
            raise ValueError("Local Prompt membership response must be a mapping.")
        response_prompt_id = _positive_signed_id(
            response.get("prompt_id"), field_name="response prompt_id"
        )
        if response_prompt_id != prompt_id:
            raise ValueError(
                "Local Prompt membership response prompt_id does not match the request."
            )
        collection_ids = tuple(
            sorted(
                _unique_positive_signed_ids(
                    response.get("collection_ids"), field_name="collection_ids"
                )
            )
        )
        if changed is None:
            changed = response.get("changed")
            if type(changed) is not bool:
                raise ValueError(
                    "Local Prompt membership response changed flag must be boolean."
                )
        return {
            "prompt_id": prompt_id,
            "collection_ids": collection_ids,
            "changed": changed,
        }

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
        return self._normalize_prompt_list(
            response, backend=normalized_mode.value, page=page, per_page=per_page
        )

    async def browse_prompts(
        self,
        *,
        mode: PromptBackend | str = "local",
        query: str = "",
        collection_id: int | None = None,
        sort_by: str = "last_modified",
        sort_order: str = "desc",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """Browse one normalized page from the local Prompt library.

        Args:
            mode: Backend selection; only ``local`` is accepted.
            query: Prompt search text; surrounding whitespace is removed.
            collection_id: Optional positive local collection identifier.
            sort_by: ``last_modified`` or ``name``; case and surrounding
                whitespace are normalized.
            sort_order: ``asc`` or ``desc``; case and surrounding whitespace
                are normalized.
            page: Requested positive one-based page.
            page_size: Requested positive page size, capped at 100.

        Returns:
            A normalized page mapping containing stable local item IDs, exact
            totals, the adapter-resolved current page, and effective page size.

        Raises:
            TypeError: If a textual scope value has the wrong type.
            ValueError: If the backend or another scope value is invalid, or
                the local backend is unavailable.
            PolicyDeniedError: If runtime policy denies local Prompt listing.
        """
        mode_value = mode.value if isinstance(mode, PromptBackend) else mode
        if not isinstance(mode_value, str) or mode_value.strip().lower() != "local":
            raise ValueError("Prompt browsing is local-only.")
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if collection_id is not None and (
            type(collection_id) is not int or collection_id <= 0
        ):
            raise ValueError("collection_id must be a positive integer or None.")
        if not isinstance(sort_by, str):
            raise TypeError("sort_by must be a string.")
        sort_by = sort_by.strip().lower()
        if sort_by not in {"last_modified", "name"}:
            raise ValueError("sort_by must be 'last_modified' or 'name'.")
        if not isinstance(sort_order, str):
            raise TypeError("sort_order must be a string.")
        sort_order = sort_order.strip().lower()
        if sort_order not in {"asc", "desc"}:
            raise ValueError("sort_order must be 'asc' or 'desc'.")
        if type(page) is not int or page <= 0:
            raise ValueError("page must be a positive integer.")
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("page_size must be a positive integer.")
        query = query.strip()
        page_size = min(page_size, 100)

        self._enforce_policy("prompts.list.local")
        service = self._service_for_mode(PromptBackend.LOCAL)
        response = await self._maybe_await(
            service.browse_prompts(
                query=query,
                collection_id=collection_id,
                sort_by=sort_by,
                sort_order=sort_order,
                page=page,
                page_size=page_size,
            )
        )
        return self._normalize_prompt_list(
            response,
            backend="local",
            page=page,
            per_page=page_size,
        )

    async def count_prompts(self, *, mode: PromptBackend | str = "local") -> int:
        """Count prompts in the given backend without fetching a full page.

        Mirrors ``NotesScopeService.count_notes``: reuses the existing
        ``list`` policy action rather than a dedicated ``count`` action
        (no such capability exists in the runtime policy registry), and
        only the local backend exposes a count-only seam today -- there is
        no server-side count-only endpoint, only a paginated ``list_prompts``
        whose total would require a full fetch to read.

        Args:
            mode: Backend to count in; only the local backend is supported
                today (see Raises). Defaults to ``"local"``.

        Returns:
            The exact number of non-deleted prompts in the local backend.

        Raises:
            ValueError: For the server backend, or when the resolved
                backend is unavailable.
        """
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "list"))
        if normalized_mode != PromptBackend.LOCAL:
            raise ValueError(
                "Server prompt counts are not supported; use list_prompts for a scoped total."
            )
        service = self._service_for_mode(normalized_mode)
        return int(await self._maybe_await(service.count_prompts()))

    async def get_library_user_content_evidence(
        self, *, mode: PromptBackend | str = "local"
    ) -> LibraryContentEvidence:
        """Return tri-state evidence for active user-owned prompts."""
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == PromptBackend.LOCAL:
            total = await self.count_prompts(mode=normalized_mode)
            return (
                LibraryContentEvidence.HAS_USER_CONTENT
                if total > 0
                else LibraryContentEvidence.EMPTY
            )

        self._enforce_policy(self._action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        payload = await self._maybe_await(
            service.list_prompts(
                page=1,
                per_page=1,
                include_deleted=False,
                sort_by="last_modified",
                sort_order="desc",
            )
        )
        payload = self._source_record(payload)
        if not payload:
            return LibraryContentEvidence.UNKNOWN
        items = payload.get("items")
        total = payload.get("total_items")
        if (
            type(total) is not int
            or total < 0
            or not isinstance(items, list)
            or len(items) > 1
        ):
            return LibraryContentEvidence.UNKNOWN
        if total == 0:
            return (
                LibraryContentEvidence.EMPTY
                if not items
                else LibraryContentEvidence.UNKNOWN
            )
        return LibraryContentEvidence.UNKNOWN

    async def search_prompts(
        self,
        *,
        mode: PromptBackend | str = "local",
        query: str,
        limit: int = 10,
        include_deleted: bool = False,
        fts_match_query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search one source, using paginated server listing for an empty query."""
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == PromptBackend.SERVER and not query.strip():
            page = await self.list_prompts(
                mode=normalized_mode,
                page=1,
                per_page=limit,
                include_deleted=include_deleted,
            )
            return page["items"]

        try:
            self._enforce_policy(self._action_id(normalized_mode, "list"))
        except (PermissionError, PolicyDeniedError) as exc:
            raise PromptCapabilityError(normalized_mode.value, "search") from exc
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == PromptBackend.SERVER:
            capabilities = await self.get_capabilities(mode=normalized_mode)
            if not capabilities.search:
                raise PromptCapabilityError(normalized_mode.value, "search")
            try:
                response = await self._maybe_await(
                    service.search_prompts(
                        search_query=query,
                        page=1,
                        results_per_page=limit,
                        include_deleted=include_deleted,
                    )
                )
            except (PermissionError, PolicyDeniedError) as exc:
                raise PromptCapabilityError(normalized_mode.value, "search") from exc
            return normalize_prompt_search(response, backend=normalized_mode.value)

        local_kwargs = (
            {"fts_match_query": fts_match_query} if fts_match_query is not None else {}
        )
        response = await self._maybe_await(
            service.search_prompts(
                query=query,
                limit=limit,
                include_deleted=include_deleted,
                **local_kwargs,
            )
        )
        return [
            self._normalize_prompt_record(item, backend=normalized_mode.value)
            for item in response or ()
        ]

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
        return self._normalize_prompt_record(response, backend=normalized_mode.value)

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
        artifact_type: Optional[str] = None,
        expected_version: Optional[int] = None,
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
            artifact_type=artifact_type,
        )
        raw_definition = deserialize_definition(payload.get("prompt_definition"))
        definition_kind = (
            raw_definition.get("kind") if raw_definition is not None else None
        )
        definition_version = (
            raw_definition.get("schema_version") if raw_definition is not None else None
        )
        is_console_v2_candidate = (
            (
                type(payload.get("prompt_schema_version")) is int
                and payload.get("prompt_schema_version") == 2
            )
            or (type(definition_version) is int and definition_version == 2)
            or definition_kind
            in {
                "block_prompt",
                "block_recipe",
                "single_text_recipe",
            }
        )
        capabilities = None
        if is_console_v2_candidate:
            capabilities = await self.get_capabilities(mode=normalized_mode)
            payload = validate_console_artifact_payload(payload, capabilities)
        if (
            action == "update"
            and normalized_mode == PromptBackend.LOCAL
            and expected_version is not None
        ):
            payload["expected_version"] = expected_version
        if capabilities is not None:
            if normalized_mode == PromptBackend.SERVER:
                payload = _serialize_server_prompt_request(
                    payload, for_update=action == "update"
                )
            validate_prompt_request_size(payload, capabilities)
        if action == "create":
            response = await self._maybe_await(service.create_prompt(payload))
        else:
            response = await self._maybe_await(
                service.update_prompt(prompt_identifier, payload)
            )
        return self._normalize_prompt_record(response, backend=normalized_mode.value)

    async def delete_prompt(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
        expected_version: int | None = None,
    ) -> bool:
        """Delete a Prompt/Recipe through its selected backend.

        Args:
            mode: Backend selection; defaults to the service's configured mode.
            prompt_identifier: Backend-specific identifier for the artifact.
            expected_version: Optional local version required for deletion.

        Returns:
            ``True`` when the backend reports a successful deletion.

        Raises:
            ValueError: If the mode or backend service is unavailable.
            PolicyDeniedError: If policy rejects the delete action.
            ExpectedVersionConflictError: If a local expected version is stale.
        """
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == PromptBackend.LOCAL and expected_version is not None:
            response = await self._maybe_await(
                service.delete_prompt(
                    prompt_identifier,
                    expected_version=expected_version,
                )
            )
        else:
            response = await self._maybe_await(service.delete_prompt(prompt_identifier))
        if response == {}:
            return True
        return bool(response)

    async def delete_prompts(
        self,
        *,
        mode: PromptBackend | str | None = None,
        targets: tuple[PromptBatchTarget, ...],
    ) -> PromptBatchDeleteResult:
        """Delete one strict local Prompt batch in a single transaction.

        Args:
            mode: Backend selection, which must resolve to the local backend.
            targets: Prompt IDs and exact active versions to delete.

        Returns:
            The exact committed local database receipt.

        Raises:
            TypeError: If the target container or entries have invalid types.
            ValueError: If mode, targets, or local capability are invalid.
            PolicyDeniedError: If policy rejects local Prompt deletion.
        """
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != PromptBackend.LOCAL:
            raise ValueError("Prompt batch delete is local-only.")
        validated_targets = validate_prompt_batch_targets(targets)
        self._enforce_policy(self._action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        delete_prompts = getattr(service, "delete_prompts", None)
        if not callable(delete_prompts):
            raise ValueError("Local Prompt batch delete is unavailable.")
        return await self._maybe_await(delete_prompts(targets=validated_targets))

    async def restore_deleted_prompt(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
        expected_version: int,
    ) -> dict[str, Any]:
        """Restore one exact local Prompt/Recipe tombstone.

        Resurrection is local-only until a server capability explicitly owns
        an equivalent contract. It is governed as an ordinary conditional
        update and returned through the same normalized record envelope used
        by the rest of the Library Prompt surface.

        Args:
            mode: Backend selection, which must resolve to the local backend.
            prompt_identifier: Numeric id, UUID, or name of the tombstone.
            expected_version: Exact deleted-row version required for restore.

        Returns:
            The normalized restored Prompt/Recipe record.

        Raises:
            ValueError: If restore is requested outside the local backend or
                the local service does not expose restore capability.
            PolicyDeniedError: If policy rejects the conditional update.
            ExpectedVersionConflictError: If the tombstone version is stale.
        """
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != PromptBackend.LOCAL:
            raise ValueError("Deleted Prompt restore is local-only.")
        self._enforce_policy(self._action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        restore_prompt = getattr(service, "restore_deleted_prompt", None)
        if not callable(restore_prompt):
            raise ValueError("Local Prompt restore is unavailable.")
        response = await self._maybe_await(
            restore_prompt(
                prompt_identifier,
                expected_version=expected_version,
            )
        )
        return self._normalize_prompt_record(response, backend=normalized_mode.value)

    async def restore_deleted_prompts(
        self,
        *,
        mode: PromptBackend | str | None = None,
        targets: tuple[PromptBatchTarget, ...],
    ) -> PromptBatchRestoreResult:
        """Restore one strict local Prompt batch in a single transaction.

        Args:
            mode: Backend selection, which must resolve to the local backend.
            targets: Prompt IDs and exact tombstone versions to restore.

        Returns:
            The exact committed local database restore result.

        Raises:
            TypeError: If the target container or entries have invalid types.
            ValueError: If mode, targets, or local capability are invalid.
            PolicyDeniedError: If policy rejects the local Prompt update.
        """
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != PromptBackend.LOCAL:
            raise ValueError("Prompt batch restore is local-only.")
        validated_targets = validate_prompt_batch_targets(targets)
        self._enforce_policy(self._action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        restore_prompts = getattr(service, "restore_deleted_prompts", None)
        if not callable(restore_prompts):
            raise ValueError("Local Prompt batch restore is unavailable.")
        return await self._maybe_await(restore_prompts(targets=validated_targets))

    async def record_prompt_usage(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "use"))
        service = self._service_for_mode(normalized_mode)
        current = await self._maybe_await(
            service.get_prompt(prompt_identifier, include_deleted=False)
        )
        normalized_current = self._normalize_prompt_record(
            current, backend=normalized_mode.value
        )
        if normalized_current.get("artifact_type") == "recipe":
            raise ValueError(
                "Recipes cannot be used directly. Save a Prompt copy before use."
            )
        if normalized_current.get("artifact_type") != "prompt":
            raise ValueError(
                "Only Prompt artifacts can be used directly. Save a supported "
                "Prompt copy before use."
            )
        response = await self._maybe_await(
            service.record_prompt_usage(prompt_identifier)
        )
        return self._normalize_prompt_record(response, backend=normalized_mode.value)

    async def list_prompt_versions(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
        page_size: int = 25,
        before_change_id: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(f"prompts.versions.list.{normalized_mode.value}")
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == PromptBackend.LOCAL:
            response = await self._maybe_await(
                service.list_prompt_versions(
                    prompt_identifier,
                    page_size=page_size,
                    before_change_id=before_change_id,
                )
            )
            return normalize_prompt_history_page(
                response,
                backend="local",
                capabilities=local_prompt_capabilities(),
            )
        response = await self._maybe_await(
            service.list_prompt_versions(prompt_identifier)
        )
        return normalize_prompt_version_list(response, backend=normalized_mode.value)

    async def count_prompt_versions(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
    ) -> int:
        """Return an exact local retained-history count without loading a page."""
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(f"prompts.versions.list.{normalized_mode.value}")
        if normalized_mode == PromptBackend.SERVER:
            raise PromptCapabilityError("server", "retained history count")
        service = self._service_for_mode(normalized_mode)
        count = await self._maybe_await(
            service.count_prompt_versions(prompt_identifier)
        )
        if type(count) is not int or count < 0:
            raise ValueError("Retained history count must be a non-negative integer.")
        return count

    async def restore_prompt_version(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
        version: int,
        change_id: int | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(f"prompts.versions.restore.{normalized_mode.value}")
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == PromptBackend.LOCAL:
            if change_id is None or expected_version is None:
                raise ValueError(
                    "Local retained restore requires change_id and expected_version."
                )
            return await self._maybe_await(
                service.restore_prompt_version(
                    prompt_identifier,
                    change_id=change_id,
                    version=version,
                    expected_version=expected_version,
                )
            )
        response = await self._maybe_await(
            service.restore_prompt_version(prompt_identifier, version)
        )
        return self._normalize_prompt_record(response, backend=normalized_mode.value)

    async def create_prompt_collection(
        self,
        *,
        mode: PromptBackend | str | None = None,
        name: str,
        description: Optional[str] = None,
        prompt_ids: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._collection_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        payload = {
            "name": name,
            "description": description,
            "prompt_ids": list(prompt_ids or []),
        }
        response = await self._maybe_await(service.create_prompt_collection(payload))
        data = (
            response.model_dump(mode="json")
            if hasattr(response, "model_dump")
            else dict(response)
        )
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
        query: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List one validated collection page from the selected backend.

        Args:
            mode: Local or server Prompt backend.
            query: Literal local collection-name search. Server collection search
                is unsupported; an empty value preserves the existing server API.
            limit: Positive requested page size. Local pages are capped at 100.
            offset: Non-negative row offset.

        Returns:
            A normalized collection page with stable source-qualified IDs.

        Raises:
            TypeError: If ``query`` is not a string.
            ValueError: If a bound is invalid or server search is requested.
            PolicyDeniedError: If runtime policy denies the validated action.
        """
        normalized_mode = self._normalize_mode(mode)
        query, limit, offset = _normalize_collection_catalog_args(
            query=query, limit=limit, offset=offset
        )
        if normalized_mode == PromptBackend.SERVER and query:
            raise ValueError("Server prompt collection search is not supported.")
        if normalized_mode == PromptBackend.LOCAL:
            limit = min(limit, 100)
        self._enforce_policy(self._collection_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        kwargs: dict[str, Any] = {"limit": limit, "offset": offset}
        if normalized_mode == PromptBackend.LOCAL:
            kwargs["query"] = query
        response = await self._maybe_await(service.list_prompt_collections(**kwargs))
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
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._collection_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.get_prompt_collection(collection_id))
        return normalize_prompt_collection_record(
            response, backend=normalized_mode.value
        )

    async def update_prompt_collection(
        self,
        *,
        mode: PromptBackend | str | None = None,
        collection_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        prompt_ids: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._collection_action_id(normalized_mode, "update"))
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
        response = await self._maybe_await(
            service.update_prompt_collection(collection_id, payload)
        )
        return normalize_prompt_collection_record(
            response, backend=normalized_mode.value
        )

    async def list_prompt_collection_memberships(
        self,
        *,
        mode: PromptBackend | str = PromptBackend.LOCAL,
        prompt_id: int,
    ) -> dict[str, int | tuple[int, ...] | bool]:
        """List one active local Prompt's collection memberships.

        Args:
            mode: Backend selection; only ``local`` is accepted.
            prompt_id: Positive local Prompt identifier.

        Returns:
            A bounded mapping with ordered collection IDs and ``changed=False``.

        Raises:
            ValueError: If the mode or Prompt identifier is invalid, the Prompt is
                inactive, or the local response is malformed.
            PolicyDeniedError: If runtime policy denies local collection detail.
        """
        normalized_mode = self._local_membership_mode(mode)
        resolved_prompt_id = _positive_signed_id(prompt_id, field_name="prompt_id")
        self._enforce_policy(self._collection_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(
            service.list_prompt_collection_memberships(resolved_prompt_id)
        )
        return self._membership_outcome(
            response, prompt_id=resolved_prompt_id, changed=False
        )

    async def replace_prompt_collection_memberships(
        self,
        *,
        mode: PromptBackend | str = PromptBackend.LOCAL,
        prompt_id: int,
        collection_ids: Sequence[int],
    ) -> dict[str, int | tuple[int, ...] | bool]:
        """Replace one active local Prompt's collection memberships atomically.

        Args:
            mode: Backend selection; only ``local`` is accepted.
            prompt_id: Positive local Prompt identifier.
            collection_ids: Unique positive active local collection identifiers.

        Returns:
            A bounded mapping containing the normalized membership set and whether
            it changed.

        Raises:
            ValueError: If the mode, Prompt ID, or collection IDs are invalid or
                inactive, or the local response is malformed.
            PolicyDeniedError: If runtime policy denies local collection update.
        """
        normalized_mode = self._local_membership_mode(mode)
        resolved_prompt_id = _positive_signed_id(prompt_id, field_name="prompt_id")
        resolved_collection_ids = tuple(
            sorted(
                _unique_positive_signed_ids(collection_ids, field_name="collection_ids")
            )
        )
        self._enforce_policy(self._collection_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(
            service.replace_prompt_collection_memberships(
                resolved_prompt_id, resolved_collection_ids
            )
        )
        outcome = self._membership_outcome(response, prompt_id=resolved_prompt_id)
        if outcome["collection_ids"] != resolved_collection_ids:
            raise ValueError(
                "Local Prompt membership response collection_ids do not match the request."
            )
        return outcome


def _build_server_prompt_service_from_config(
    app_config: dict[str, Any] | None,
) -> ServerPromptService:
    """Build a lazy server prompt service when app config contains a server binding."""
    if not derive_configured_server_binding(app_config).server_configured:
        return ServerPromptService(client=None)
    return ServerPromptService.from_config(app_config or {})


def build_prompt_scope_service(
    *,
    prompt_db: Any,
    app_config: dict[str, Any] | None = None,
    policy_enforcer: Any = None,
    server_service: Any = None,
    client_provider: Any | None = None,
) -> PromptScopeService:
    """Build the source-aware prompt service from app startup dependencies."""
    local_service = LocalPromptService(prompt_db) if prompt_db is not None else None
    if server_service is None:
        if client_provider is not None:
            server_service = ServerPromptService.from_server_context_provider(
                client_provider
            )
        else:
            server_service = _build_server_prompt_service_from_config(app_config)

    return PromptScopeService(
        local_service=local_service,
        server_service=server_service,
        policy_enforcer=policy_enforcer,
    )
