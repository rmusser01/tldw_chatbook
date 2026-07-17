"""Scope-aware routing for local and server-backed prompt operations."""

from __future__ import annotations

import inspect
import sqlite3
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from ..runtime_policy.bootstrap import (
    build_runtime_api_client_provider_from_config,
    derive_configured_server_binding,
)
if TYPE_CHECKING:
    from ..tldw_api import PromptCreateRequest, TLDWAPIClient
from .prompt_normalizers import (
    normalize_prompt_collection_list,
    normalize_prompt_collection_record,
    normalize_prompt_list,
    normalize_prompt_record,
    normalize_prompt_version_list,
)


class PromptBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


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
    }
    return {key: value for key, value in payload.items() if value is not None}


def _prompt_create_request_from_payload(payload: dict[str, Any]) -> PromptCreateRequest:
    # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
    from ..tldw_api import PromptCreateRequest

    if not payload.get("name"):
        raise ValueError("Prompt name is required for server prompt saves.")
    return PromptCreateRequest(**payload)


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

    async def get_prompt(self, prompt_identifier: str | int, *, include_deleted: bool = False) -> Any:
        return await self._require_client().get_prompt(prompt_identifier, include_deleted=include_deleted)

    async def create_prompt(self, payload: dict[str, Any]) -> Any:
        return await self._require_client().create_prompt(_prompt_create_request_from_payload(payload))

    async def update_prompt(self, prompt_identifier: str | int, payload: dict[str, Any]) -> Any:
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

    async def restore_prompt_version(self, prompt_identifier: str | int, version: int) -> Any:
        return await self._require_client().restore_prompt_version(prompt_identifier, version)

    async def create_prompt_collection(self, payload: dict[str, Any]) -> Any:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import PromptCollectionCreateRequest

        return await self._require_client().create_prompt_collection(PromptCollectionCreateRequest(**payload))

    async def list_prompt_collections(self, *, limit: int = 200, offset: int = 0) -> Any:
        return await self._require_client().list_prompt_collections(limit=limit, offset=offset)

    async def get_prompt_collection(self, collection_id: int) -> Any:
        return await self._require_client().get_prompt_collection(collection_id)

    async def update_prompt_collection(self, collection_id: int, payload: dict[str, Any]) -> Any:
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
    def _collection_id(collection_id: int | str) -> int:
        try:
            resolved = int(collection_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid prompt collection id.") from exc
        if resolved < 1:
            raise ValueError("Invalid prompt collection id.")
        return resolved

    @staticmethod
    def _prompt_ids(prompt_ids: Optional[list[int]]) -> list[int]:
        resolved: list[int] = []
        for prompt_id in prompt_ids or []:
            try:
                value = int(prompt_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("Prompt collection prompt_ids must be integers.") from exc
            if value < 1:
                raise ValueError("Prompt collection prompt_ids must be positive integers.")
            resolved.append(value)
        return resolved

    def _set_collection_prompt_ids(self, conn: sqlite3.Connection, collection_id: int, prompt_ids: list[int]) -> None:
        conn.execute(
            "DELETE FROM LocalPromptCollectionItems WHERE collection_id = ?",
            (collection_id,),
        )
        conn.executemany(
            """
            INSERT INTO LocalPromptCollectionItems (collection_id, prompt_id, position)
            VALUES (?, ?, ?)
            """,
            [(collection_id, prompt_id, index) for index, prompt_id in enumerate(prompt_ids)],
        )

    def _collection_record(self, collection_id: int) -> dict[str, Any]:
        db = self._require_collection_db()
        conn = db.get_connection()
        row = conn.execute(
            """
            SELECT collection_id, name, description
            FROM LocalPromptCollections
            WHERE collection_id = ? AND deleted = 0
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
        return {
            "collection_id": int(row["collection_id"]),
            "name": row["name"],
            "description": row["description"],
            "prompt_ids": [int(prompt_row["prompt_id"]) for prompt_row in prompt_rows],
        }

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
        _prompts, _total_pages, _current_page, total_items = self.prompt_db.list_prompts(
            page=1,
            per_page=1,
            include_deleted=include_deleted,
        )
        return total_items

    def get_prompt(self, prompt_identifier: str | int, *, include_deleted: bool = False) -> Any:
        if hasattr(self.prompt_db, "fetch_prompt_details"):
            return self.prompt_db.fetch_prompt_details(prompt_identifier, include_deleted=include_deleted)
        return self.prompt_db.get_prompt(prompt_identifier, include_deleted=include_deleted)

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
        fts_kwargs = {"fts_match_query": fts_match_query} if fts_match_query is not None else {}
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
        )
        identifier = prompt_uuid or prompt_id
        return self.get_prompt(identifier, include_deleted=True)

    def update_prompt(self, prompt_identifier: str | int, payload: dict[str, Any]) -> Any:
        existing = self.get_prompt(prompt_identifier, include_deleted=True)
        if not existing:
            raise ValueError(f"Prompt '{prompt_identifier}' not found.")

        if hasattr(self.prompt_db, "update_prompt_by_id"):
            prompt_uuid, _message = self.prompt_db.update_prompt_by_id(existing["id"], payload)
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
            prompt_schema_version=payload.get("prompt_schema_version", existing.get("prompt_schema_version")),
            prompt_definition=payload.get("prompt_definition", existing.get("prompt_definition")),
        )
        return self.get_prompt(prompt_uuid or prompt_id, include_deleted=True)

    def delete_prompt(self, prompt_identifier: str | int) -> Any:
        return self.prompt_db.soft_delete_prompt(prompt_identifier)

    def record_prompt_usage(self, prompt_identifier: str | int) -> Any:
        if hasattr(self.prompt_db, "record_prompt_usage"):
            return self.prompt_db.record_prompt_usage(prompt_identifier)
        return self.get_prompt(prompt_identifier, include_deleted=True)

    def create_prompt_collection(self, payload: dict[str, Any]) -> dict[str, Any]:
        db = self._require_collection_db()
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("Prompt collection name is required.")
        description = payload.get("description")
        prompt_ids = self._prompt_ids(payload.get("prompt_ids"))
        conn = db.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO LocalPromptCollections (name, description)
                    VALUES (?, ?)
                    """,
                    (name, description),
                )
                collection_id = int(cursor.lastrowid)
                self._set_collection_prompt_ids(conn, collection_id, prompt_ids)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Prompt collection '{name}' already exists or references missing prompts.") from exc
        return {"collection_id": collection_id}

    def list_prompt_collections(self, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        db = self._require_collection_db()
        conn = db.get_connection()
        total = int(
            conn.execute(
                "SELECT COUNT(*) FROM LocalPromptCollections WHERE deleted = 0"
            ).fetchone()[0]
        )
        rows = conn.execute(
            """
            SELECT collection_id
            FROM LocalPromptCollections
            WHERE deleted = 0
            ORDER BY name COLLATE NOCASE ASC, collection_id ASC
            LIMIT ? OFFSET ?
            """,
            (max(1, int(limit)), max(0, int(offset))),
        ).fetchall()
        return {
            "collections": [self._collection_record(int(row["collection_id"])) for row in rows],
            "limit": int(limit),
            "offset": int(offset),
            "total": total,
        }

    def get_prompt_collection(self, collection_id: int) -> dict[str, Any]:
        return self._collection_record(self._collection_id(collection_id))

    def update_prompt_collection(self, collection_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        db = self._require_collection_db()
        resolved_collection_id = self._collection_id(collection_id)
        updates = {
            key: payload[key]
            for key in ("name", "description")
            if key in payload
        }
        if "name" in updates:
            updates["name"] = str(updates["name"] or "").strip()
            if not updates["name"]:
                raise ValueError("Prompt collection name is required.")
        prompt_ids = self._prompt_ids(payload.get("prompt_ids")) if "prompt_ids" in payload else None
        conn = db.get_connection()
        try:
            with conn:
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
                        raise ValueError(f"Prompt collection '{collection_id}' not found.")
                if prompt_ids is not None:
                    self._set_collection_prompt_ids(conn, resolved_collection_id, prompt_ids)
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
                            raise ValueError(f"Prompt collection '{collection_id}' not found.")
        except sqlite3.IntegrityError as exc:
            raise ValueError("Prompt collection update failed because a name or prompt reference is invalid.") from exc
        return self._collection_record(resolved_collection_id)


class PromptScopeService:
    """Route prompt actions to the active local/server backend and normalize outputs."""

    def __init__(self, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

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

    @staticmethod
    def _action_id(mode: PromptBackend, action: str) -> str:
        return f"prompts.{action}.{mode.value}"

    @staticmethod
    def _collection_action_id(mode: PromptBackend, action: str) -> str:
        return f"prompts.collections.{action}.{mode.value}"

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
        return normalize_prompt_list(response, backend=normalized_mode.value, page=page, per_page=per_page)

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

    async def search_prompts(
        self,
        *,
        mode: PromptBackend | str = "local",
        query: str,
        limit: int = 10,
        include_deleted: bool = False,
        fts_match_query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search prompts in the given backend, normalized like ``list_prompts``.

        Mirrors ``NotesScopeService.search_notes``'s ``fts_match_query``
        pass-through: forwarded to the local backend only when provided, so
        existing local backends/test fakes without the parameter keep
        working unchanged. Like ``count_prompts``, only the local backend is
        supported today -- there is no server-side prompt search endpoint.

        Args:
            mode: Backend to search in; only the local backend is supported
                today (see Raises). Defaults to ``"local"``.
            query: Plain user query text.
            limit: Maximum number of prompts to return.
            include_deleted: Whether to include soft-deleted prompts.
            fts_match_query: Optional pre-built FTS5 MATCH string (e.g.
                Library keyword search's plural/singular-widened query)
                overriding the MATCH clause built from ``query`` in the
                local backend.

        Returns:
            Normalized prompt records (see ``normalize_prompt_record``);
            each carries the local integer id under ``local_id`` (its
            ``id`` is the composite ``"local:prompt:<id>"`` string -- use
            ``local_id`` for any caller that needs the raw prompt id, e.g.
            to open the prompt editor).

        Raises:
            ValueError: For the server backend, or when the resolved
                backend is unavailable.
        """
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "list"))
        if normalized_mode != PromptBackend.LOCAL:
            raise ValueError(
                "Server prompt search is not supported; use list_prompts for a scoped page."
            )
        service = self._service_for_mode(normalized_mode)
        local_kwargs = {"fts_match_query": fts_match_query} if fts_match_query is not None else {}
        response = await self._maybe_await(
            service.search_prompts(
                query=query,
                limit=limit,
                include_deleted=include_deleted,
                **local_kwargs,
            )
        )
        return [normalize_prompt_record(item, backend=normalized_mode.value) for item in response or ()]

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
        return normalize_prompt_record(response, backend=normalized_mode.value)

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
        )
        if action == "create":
            response = await self._maybe_await(service.create_prompt(payload))
        else:
            response = await self._maybe_await(service.update_prompt(prompt_identifier, payload))
        return normalize_prompt_record(response, backend=normalized_mode.value)

    async def delete_prompt(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.delete_prompt(prompt_identifier))
        if response == {}:
            return True
        return bool(response)

    async def record_prompt_usage(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "use"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.record_prompt_usage(prompt_identifier))
        return normalize_prompt_record(response, backend=normalized_mode.value)

    async def list_prompt_versions(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "versions"))
        if normalized_mode == PromptBackend.LOCAL:
            raise ValueError("Local prompt version history is unavailable.")
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.list_prompt_versions(prompt_identifier))
        return normalize_prompt_version_list(response, backend=normalized_mode.value)

    async def restore_prompt_version(
        self,
        *,
        mode: PromptBackend | str | None = None,
        prompt_identifier: str | int,
        version: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id(normalized_mode, "restore_version"))
        if normalized_mode == PromptBackend.LOCAL:
            raise ValueError("Local prompt version restore is unavailable.")
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.restore_prompt_version(prompt_identifier, version))
        return normalize_prompt_record(response, backend=normalized_mode.value)

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
        data = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
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
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._collection_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        response = await self._maybe_await(service.list_prompt_collections(limit=limit, offset=offset))
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
        return normalize_prompt_collection_record(response, backend=normalized_mode.value)

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
        response = await self._maybe_await(service.update_prompt_collection(collection_id, payload))
        return normalize_prompt_collection_record(response, backend=normalized_mode.value)


def _build_server_prompt_service_from_config(app_config: dict[str, Any] | None) -> ServerPromptService:
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
            server_service = ServerPromptService.from_server_context_provider(client_provider)
        else:
            server_service = _build_server_prompt_service_from_config(app_config)

    return PromptScopeService(
        local_service=local_service,
        server_service=server_service,
        policy_enforcer=policy_enforcer,
    )
