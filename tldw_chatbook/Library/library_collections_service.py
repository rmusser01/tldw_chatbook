"""Service seam for Library Collections local management."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from typing import Protocol
from uuid import uuid4

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.library_collections_state import (
    LIBRARY_COLLECTIONS_DESCRIPTION_MAX_LENGTH,
    LIBRARY_COLLECTIONS_NAME_MAX_LENGTH,
    _collection_name_validation,
)
from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input


DEFAULT_LIBRARY_COLLECTIONS_LIST_LIMIT = 200
MAX_LIBRARY_COLLECTIONS_LIST_LIMIT = 500
_STORAGE_FAILURE_MESSAGE = "Library Collections storage failed."


class LibraryCollectionsServiceError(Exception):
    """Base exception for Library Collections service failures."""


class InvalidLibraryCollectionName(LibraryCollectionsServiceError):
    """Raised when a Collection name fails validation."""


class InvalidLibraryCollectionDescription(LibraryCollectionsServiceError):
    """Raised when a Collection description fails validation."""


class DuplicateLibraryCollectionName(LibraryCollectionsServiceError):
    """Raised when a normalized Collection name already exists."""


class DuplicateLibraryCollectionItem(LibraryCollectionsServiceError):
    """Raised when a source item is already in the selected Collection."""


class LibraryCollectionNotFound(LibraryCollectionsServiceError):
    """Raised when a Collection operation targets a missing Collection."""


@dataclass(frozen=True)
class LibraryCollectionRecord:
    """Service record for one local Library Collection."""

    collection_id: str
    name: str
    description: str
    item_count: int
    source_authority: str
    sync_status: str
    created_at: str
    updated_at: str


class LibraryCollectionsService(Protocol):
    """Protocol implemented by Library Collections backends."""

    def list_collections(
        self,
        limit: int = DEFAULT_LIBRARY_COLLECTIONS_LIST_LIMIT,
    ) -> tuple[LibraryCollectionRecord, ...]:
        """List active Library Collections."""

    def get_collection(self, collection_id: str) -> LibraryCollectionRecord | None:
        """Return one active Library Collection if it exists."""

    def create_collection(
        self,
        name: str,
        *,
        description: str = "",
    ) -> LibraryCollectionRecord:
        """Create a local Library Collection."""

    def rename_collection(
        self,
        collection_id: str,
        name: str,
        *,
        description: str | None = None,
    ) -> LibraryCollectionRecord:
        """Rename a local Library Collection."""

    def delete_collection(self, collection_id: str) -> bool:
        """Soft-delete a local Library Collection."""


class LocalLibraryCollectionsService:
    """Local SQLite implementation of Library Collections contracts."""

    def __init__(
        self,
        db: LibraryCollectionsDB,
        *,
        id_factory: Callable[[], str] | None = None,
        now_factory: Callable[[], str] | None = None,
    ) -> None:
        self.db = db
        self._id_factory = id_factory or (lambda: f"collection-{uuid4().hex}")
        self._now_factory = now_factory or _utc_now

    def list_collections(
        self,
        limit: int = DEFAULT_LIBRARY_COLLECTIONS_LIST_LIMIT,
    ) -> tuple[LibraryCollectionRecord, ...]:
        safe_limit = _validate_list_limit(limit)
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        collection.collection_id,
                        collection.name,
                        collection.description,
                        collection.created_at,
                        collection.updated_at,
                        COUNT(item.membership_id) AS item_count
                    FROM library_collections AS collection
                    LEFT JOIN library_collection_items AS item
                        ON item.collection_id = collection.collection_id
                    WHERE collection.deleted_at IS NULL
                    GROUP BY collection.collection_id
                    ORDER BY collection.created_at ASC, collection.name COLLATE NOCASE ASC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_record_from_row(row) for row in rows)

    def get_collection(self, collection_id: str) -> LibraryCollectionRecord | None:
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT
                        collection.collection_id,
                        collection.name,
                        collection.description,
                        collection.created_at,
                        collection.updated_at,
                        COUNT(item.membership_id) AS item_count
                    FROM library_collections AS collection
                    LEFT JOIN library_collection_items AS item
                        ON item.collection_id = collection.collection_id
                    WHERE collection.deleted_at IS NULL
                        AND collection.collection_id = ?
                    GROUP BY collection.collection_id
                    """,
                    (collection_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return _record_from_row(row) if row is not None else None

    def create_collection(
        self,
        name: str,
        *,
        description: str = "",
    ) -> LibraryCollectionRecord:
        safe_name = self._validate_name(name)
        safe_description = self._validate_description(description)
        self._ensure_unique_name(safe_name)
        collection_id = self._id_factory()
        now = self._now_factory()
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO library_collections (
                        collection_id,
                        name,
                        description,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (collection_id, safe_name, safe_description, now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateLibraryCollectionName(
                f"Collection name already exists: {safe_name}"
            ) from exc
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        collection = self.get_collection(collection_id)
        if collection is None:
            raise LibraryCollectionsServiceError("Collection creation failed.")
        return collection

    def rename_collection(
        self,
        collection_id: str,
        name: str,
        *,
        description: str | None = None,
    ) -> LibraryCollectionRecord:
        existing = self.get_collection(collection_id)
        if existing is None:
            raise LibraryCollectionNotFound(collection_id)
        safe_name = self._validate_name(name)
        safe_description = (
            existing.description
            if description is None
            else self._validate_description(description)
        )
        self._ensure_unique_name(safe_name, excluding_collection_id=collection_id)
        now = self._now_factory()
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    UPDATE library_collections
                    SET name = ?,
                        description = ?,
                        updated_at = ?
                    WHERE collection_id = ?
                        AND deleted_at IS NULL
                    """,
                    (safe_name, safe_description, now, collection_id),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateLibraryCollectionName(
                f"Collection name already exists: {safe_name}"
            ) from exc
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        collection = self.get_collection(collection_id)
        if collection is None:
            raise LibraryCollectionNotFound(collection_id)
        return collection

    def delete_collection(self, collection_id: str) -> bool:
        now = self._now_factory()
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE library_collections
                    SET deleted_at = ?,
                        updated_at = ?
                    WHERE collection_id = ?
                        AND deleted_at IS NULL
                    """,
                    (now, now, collection_id),
                )
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return cursor.rowcount > 0

    def add_item_to_collection(
        self,
        collection_id: str,
        *,
        source_type: str,
        source_id: str,
        title: str = "",
    ) -> str:
        if self.get_collection(collection_id) is None:
            raise LibraryCollectionNotFound(collection_id)
        safe_source_type = _validate_required_value(source_type, "source_type")
        safe_source_id = _validate_required_value(source_id, "source_id")
        safe_title = _collapse_text(title)[:500]
        membership_id = self._id_factory()
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO library_collection_items (
                        membership_id,
                        collection_id,
                        source_type,
                        source_id,
                        title,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        membership_id,
                        collection_id,
                        safe_source_type,
                        safe_source_id,
                        safe_title,
                        self._now_factory(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateLibraryCollectionItem(
                "Source item already belongs to this Collection."
            ) from exc
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return membership_id

    def _validate_name(self, value: str) -> str:
        name, reason = _collection_name_validation(value)
        if reason:
            raise InvalidLibraryCollectionName(reason)
        return name

    def _validate_description(self, value: str) -> str:
        description = _collapse_text(value)
        if len(description) > LIBRARY_COLLECTIONS_DESCRIPTION_MAX_LENGTH:
            raise InvalidLibraryCollectionDescription(
                "Collection descriptions must be 500 characters or fewer."
            )
        if not validate_text_input(
            description,
            max_length=LIBRARY_COLLECTIONS_DESCRIPTION_MAX_LENGTH,
            allow_html=False,
        ):
            raise InvalidLibraryCollectionDescription(
                "Enter a safe Collection description."
            )
        description = sanitize_string(
            description,
            max_length=LIBRARY_COLLECTIONS_DESCRIPTION_MAX_LENGTH,
        )
        description = _collapse_text(description)
        if not validate_text_input(
            description,
            max_length=LIBRARY_COLLECTIONS_DESCRIPTION_MAX_LENGTH,
            allow_html=False,
        ):
            raise InvalidLibraryCollectionDescription(
                "Enter a safe Collection description."
            )
        return description

    def _ensure_unique_name(
        self,
        name: str,
        *,
        excluding_collection_id: str | None = None,
    ) -> None:
        try:
            with self.db.connection() as conn:
                query = """
                    SELECT collection_id, deleted_at
                    FROM library_collections
                    WHERE name = ? COLLATE NOCASE
                """
                params: list[str] = [name]
                if excluding_collection_id:
                    query += " AND collection_id != ?"
                    params.append(excluding_collection_id)
                row = conn.execute(query, tuple(params)).fetchone()
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc

        if row is None:
            return
        if row["deleted_at"] is not None:
            raise DuplicateLibraryCollectionName(
                "A deleted Collection already used this name."
            )
        raise DuplicateLibraryCollectionName(
            f"Collection name already exists: {name}"
        )


def _record_from_row(row) -> LibraryCollectionRecord:
    return LibraryCollectionRecord(
        collection_id=str(row["collection_id"]),
        name=str(row["name"]),
        description=str(row["description"] or ""),
        item_count=max(0, int(row["item_count"] or 0)),
        source_authority="local",
        sync_status="local-only",
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _collapse_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _validate_list_limit(limit: int) -> int:
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_LIBRARY_COLLECTIONS_LIST_LIMIT
    return min(max(parsed, 1), MAX_LIBRARY_COLLECTIONS_LIST_LIMIT)


def _validate_required_value(value: str, field_name: str) -> str:
    collapsed = _collapse_text(value)
    if not collapsed:
        raise LibraryCollectionsServiceError(f"{field_name} is required.")
    if len(collapsed) > LIBRARY_COLLECTIONS_NAME_MAX_LENGTH:
        raise LibraryCollectionsServiceError(f"{field_name} is too long.")
    return collapsed
