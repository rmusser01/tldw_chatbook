"""Service seam for Library Collections local management."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import sqlite3
from typing import Protocol
from uuid import uuid4

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.library_content_evidence import LibraryContentEvidence
from tldw_chatbook.Library.library_tool_contract import (
    LIBRARY_ITEM_TYPES,
    make_public_id,
    normalize_display_text,
)
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

    def get_library_user_content_evidence(self) -> LibraryContentEvidence:
        """Return tri-state evidence for active local Collections.

        Returns:
            Evidence that active local Collections exist or are
            authoritatively absent.
        """

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

    def restore_collection(self, collection_id: str) -> LibraryCollectionRecord:
        """Restore one soft-deleted local Library Collection.

        Args:
            collection_id: Stable identifier of the deleted Collection.

        Returns:
            The restored Collection with its retained membership count.

        Raises:
            LibraryCollectionNotFound: If no deleted Collection matches the id.
            LibraryCollectionsServiceError: If local persistence fails.
        """

    def list_library_collections(self, *, limit: int = 20, offset: int = 0) -> dict:
        """Page active Collections with an exact total for Library agent tools.

        Args:
            limit: Maximum number of Collections to return.
            offset: Number of Collections to skip.

        Returns:
            A bounded page containing items, total, offset, and limit.
        """

    def search_library_collections(
        self, *, query: str, limit: int = 20, offset: int = 0
    ) -> dict:
        """Search active Collections by name, description, or member title.

        Args:
            query: Literal case-insensitive search text.
            limit: Maximum number of Collections to return.
            offset: Number of matching Collections to skip.

        Returns:
            A bounded page with exact total and match evidence.
        """

    def get_library_collection(
        self, collection_id: str, *, limit: int = 20, offset: int = 0
    ) -> dict | None:
        """Return one active Collection plus a bounded membership page.

        Args:
            collection_id: Stable Collection identifier.
            limit: Maximum number of members to return.
            offset: Number of members to skip.

        Returns:
            The Collection and bounded member page, or None when absent.
        """


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

    def get_library_user_content_evidence(self) -> LibraryContentEvidence:
        """Return tri-state evidence for active local Collections.

        Returns:
            Evidence that active local Collections exist or are
            authoritatively absent.
        """
        try:
            with self.db.connection() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) AS count FROM library_collections "
                    "WHERE deleted_at IS NULL"
                ).fetchone()["count"]
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return (
            LibraryContentEvidence.HAS_USER_CONTENT
            if total > 0
            else LibraryContentEvidence.EMPTY
        )

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

    def restore_collection(self, collection_id: str) -> LibraryCollectionRecord:
        """Restore one soft-deleted Collection without changing membership.

        Args:
            collection_id: Stable identifier of the deleted Collection.

        Returns:
            The restored Collection with its retained membership count.

        Raises:
            LibraryCollectionNotFound: If no deleted Collection matches the id.
            LibraryCollectionsServiceError: If local persistence fails.
        """
        now = self._now_factory()
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE library_collections
                    SET deleted_at = NULL,
                        updated_at = ?
                    WHERE collection_id = ?
                        AND deleted_at IS NOT NULL
                    """,
                    (now, collection_id),
                )
                if cursor.rowcount < 1:
                    raise LibraryCollectionNotFound(collection_id)
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
        if row is None:
            raise LibraryCollectionNotFound(collection_id)
        return _record_from_row(row)

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

    # --- Library read seams (task-1337, plan Task 4) ---
    #
    # Agent-facing list/search/member-page operations. Search is restricted
    # to Collection name, description, and direct stored member titles --
    # member content is never resolved or inlined. Supported member source
    # identities map through the shared public-ID codec; unsupported or
    # unencodable identities fall back to an opaque reference.

    def list_library_collections(self, *, limit: int = 20, offset: int = 0) -> dict:
        """Page active Collections with an exact total.

        Ordering matches ``list_collections`` (``created_at ASC, name
        COLLATE NOCASE ASC``). Count and page are read in one read-only
        snapshot (``read_transaction``), so this pure read never takes the
        write lock (task-15466).

        Args:
            limit: Maximum number of Collections to return.
            offset: Number of Collections to skip.

        Returns:
            A bounded page containing items, total, offset, and limit.

        Raises:
            LibraryCollectionsServiceError: If the local store cannot be read.
        """
        try:
            with self.db.read_transaction() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) AS count FROM library_collections "
                    "WHERE deleted_at IS NULL"
                ).fetchone()["count"]
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
                    ORDER BY collection.created_at ASC,
                             collection.name COLLATE NOCASE ASC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return {
            "items": [_library_collection_item(row) for row in rows],
            "total": int(total),
            "offset": offset,
            "limit": limit,
        }

    def search_library_collections(
        self, *, query: str, limit: int = 20, offset: int = 0
    ) -> dict:
        """Search active Collections, returning one page plus exact total.

        Match branches (OR, deduplicated by Collection row): case-insensitive
        exact name, name substring, description substring, and direct stored
        member-title substring. LIKE input is escaped so wildcards match
        literally. Exact-name hits rank first, then the list ordering.

        Args:
            query: Literal case-insensitive search text.
            limit: Maximum number of Collections to return.
            offset: Number of matching Collections to skip.

        Returns:
            A bounded page with exact total and match evidence.

        Raises:
            LibraryCollectionsServiceError: If the local store cannot be read.
        """
        like_pattern = f"%{_escape_collection_like(query)}%"
        branches = [
            "LOWER(collection.name) = LOWER(?)",
            "collection.name LIKE ? ESCAPE '\\'",
            "collection.description LIKE ? ESCAPE '\\'",
            "EXISTS (SELECT 1 FROM library_collection_items AS item "
            "WHERE item.collection_id = collection.collection_id "
            "AND item.title LIKE ? ESCAPE '\\')",
        ]
        params: list = [query, like_pattern, like_pattern, like_pattern]
        where_clause = " OR ".join(f"({branch})" for branch in branches)
        hit_selects = ", ".join(
            f"({branch}) AS hit_{index}" for index, branch in enumerate(branches)
        )
        try:
            with self.db.read_transaction() as conn:
                total = conn.execute(
                    f"SELECT COUNT(*) AS count FROM library_collections AS collection "
                    f"WHERE collection.deleted_at IS NULL AND ({where_clause})",
                    tuple(params),
                ).fetchone()["count"]
                rows = conn.execute(
                    f"""
                    SELECT
                        collection.collection_id,
                        collection.name,
                        collection.description,
                        collection.created_at,
                        collection.updated_at,
                        (SELECT COUNT(*) FROM library_collection_items AS item
                            WHERE item.collection_id = collection.collection_id)
                            AS item_count,
                        {hit_selects}
                    FROM library_collections AS collection
                    WHERE collection.deleted_at IS NULL AND ({where_clause})
                    ORDER BY (LOWER(collection.name) = LOWER(?)) DESC,
                             collection.created_at ASC,
                             collection.name COLLATE NOCASE ASC
                    LIMIT ? OFFSET ?
                    """,
                    tuple(params + params + [query, limit, offset]),
                ).fetchall()
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        items = []
        for row in rows:
            item = _library_collection_item(row)
            matched_fields = set()
            if row["hit_0"] or row["hit_1"]:
                matched_fields.add("name")
            if row["hit_2"]:
                matched_fields.add("description")
            if row["hit_3"]:
                matched_fields.add("member_title")
            item["matched_fields"] = sorted(matched_fields)
            items.append(item)
        return {"items": items, "total": int(total), "offset": offset, "limit": limit}

    def get_library_collection(
        self, collection_id: str, *, limit: int = 20, offset: int = 0
    ) -> dict | None:
        """Return one active Collection plus a bounded membership page.

        Members are ordered ``created_at ASC, membership_id ASC`` and
        projected to opaque agent-safe shapes: supported source types
        (media/note/prompt/skill/conversation/collection) receive a
        type-prefixed ``item_id`` from the shared public-ID codec;
        unsupported or unencodable sources receive ``item_id=None`` plus an
        opaque ``source_ref``. Member titles are display-bounded and member
        content is never inlined. Returns None when no active Collection
        matches ``collection_id``.

        Args:
            collection_id: Stable Collection identifier.
            limit: Maximum number of members to return.
            offset: Number of members to skip.

        Returns:
            The Collection and bounded member page, or None when absent.

        Raises:
            LibraryCollectionsServiceError: If the local store cannot be read.
        """
        try:
            with self.db.read_transaction() as conn:
                collection = conn.execute(
                    """
                    SELECT collection_id, name, description, created_at, updated_at
                    FROM library_collections
                    WHERE collection_id = ? AND deleted_at IS NULL
                    """,
                    (collection_id,),
                ).fetchone()
                if collection is None:
                    return None
                member_total = conn.execute(
                    "SELECT COUNT(*) AS count FROM library_collection_items "
                    "WHERE collection_id = ?",
                    (collection_id,),
                ).fetchone()["count"]
                member_rows = conn.execute(
                    """
                    SELECT membership_id, source_type, source_id, title
                    FROM library_collection_items
                    WHERE collection_id = ?
                    ORDER BY created_at ASC, membership_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    (collection_id, limit, offset),
                ).fetchall()
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        members = [_library_collection_member(row) for row in member_rows]
        return {
            "collection_id": str(collection["collection_id"]),
            "name": str(collection["name"]),
            "description": str(collection["description"] or ""),
            "created_at": str(collection["created_at"]),
            "updated_at": str(collection["updated_at"]),
            "member_total": int(member_total),
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(members) < int(member_total),
            "members": members,
        }

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
        raise DuplicateLibraryCollectionName(f"Collection name already exists: {name}")


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
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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


def _library_collection_item(row) -> dict:
    """Project a collections row into the agent-safe Library item shape."""
    return {
        "collection_id": str(row["collection_id"]),
        "name": str(row["name"]),
        "description": str(row["description"] or ""),
        "item_count": max(0, int(row["item_count"] or 0)),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _escape_collection_like(value: str) -> str:
    """Escape LIKE metacharacters so user input matches literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _opaque_source_ref(source_type: str, source_id: str) -> str:
    """Opaque reference for members whose source has no Library item type."""
    raw = f"{source_type}:{source_id}".encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"ref:{encoded}"


def _library_collection_member(row) -> dict:
    """Project a membership row into the agent-safe member shape.

    Supported source types receive a type-prefixed public ``item_id`` that
    round-trips through the shared ID codec. Unsupported types -- or
    supported types whose stored backing identity cannot be encoded (for
    example path-like identities) -- fall back to an opaque ``source_ref``
    instead of leaking the raw identity or failing the whole page. The raw
    ``source_id`` itself is never exposed. Member titles are
    display-normalized and bounded at 160 bytes by the shared contract.
    """
    source_type = str(row["source_type"])
    source_id = str(row["source_id"])
    item_id = None
    source_ref = None
    normalized_type = source_type.lower()
    if normalized_type in LIBRARY_ITEM_TYPES:
        try:
            item_id = make_public_id(normalized_type, source_id)
        except ValueError:
            source_ref = _opaque_source_ref(source_type, source_id)
    else:
        source_ref = _opaque_source_ref(source_type, source_id)
    title, title_truncated = normalize_display_text(row["title"], max_bytes=160)
    return {
        "membership_id": str(row["membership_id"]),
        "source_type": source_type,
        "item_id": item_id,
        "source_ref": source_ref,
        "title": title,
        "title_truncated": title_truncated,
    }
