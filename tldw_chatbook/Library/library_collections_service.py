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


DEFAULT_LIBRARY_COLLECTIONS_LIST_LIMIT = 200
MAX_LIBRARY_COLLECTIONS_LIST_LIMIT = 500
MAX_SQLITE_COLLECTIONS_OFFSET = 2**63 - 1
_STORAGE_FAILURE_MESSAGE = "Library Collections storage failed."


class LibraryCollectionsServiceError(Exception):
    """Base exception for Library Collections service failures."""


class LegacyCollectionsReadOnlyError(LibraryCollectionsServiceError):
    """The superseded generic Collections tables are recovery-only."""

    reason = "legacy_read_only"
    recovery = "Use the legacy Collections inspector or JSON recovery export."

    def __init__(self) -> None:
        super().__init__(self.reason)


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
        """Reject writes to the superseded generic Collections tables."""

    def rename_collection(
        self,
        collection_id: str,
        name: str,
        *,
        description: str | None = None,
    ) -> LibraryCollectionRecord:
        """Reject writes to the superseded generic Collections tables."""

    def delete_collection(self, collection_id: str) -> bool:
        """Reject writes to the superseded generic Collections tables."""

    def restore_collection(self, collection_id: str) -> LibraryCollectionRecord:
        """Reject writes to the superseded generic Collections tables."""

    def add_item_to_collection(
        self,
        collection_id: str,
        *,
        source_type: str,
        source_id: str,
        title: str = "",
    ) -> str:
        """Reject writes to the superseded generic Collections tables."""

    def list_library_collections(self, *, limit: int = 20, offset: int = 0) -> dict:
        """Page active Collections with an exact total for Library agent tools.

        Args:
            limit: Maximum number of Collections to return.
            offset: Number of Collections to skip.

        Returns:
            A bounded page containing items, total, offset, and limit.
        """

    def locate_library_collection_page(
        self, collection_id: str, *, limit: int = 20
    ) -> dict | None:
        """Return the owning top-level page for one active Collection ID."""

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
                    ORDER BY collection.created_at ASC,
                             collection.name COLLATE NOCASE ASC,
                             collection.collection_id ASC
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
        """Reject writes to the superseded generic Collections tables."""
        raise LegacyCollectionsReadOnlyError

    def rename_collection(
        self,
        collection_id: str,
        name: str,
        *,
        description: str | None = None,
    ) -> LibraryCollectionRecord:
        """Reject writes to the superseded generic Collections tables."""
        raise LegacyCollectionsReadOnlyError

    def delete_collection(self, collection_id: str) -> bool:
        """Reject writes to the superseded generic Collections tables."""
        raise LegacyCollectionsReadOnlyError

    def restore_collection(self, collection_id: str) -> LibraryCollectionRecord:
        """Reject writes to the superseded generic Collections tables."""
        raise LegacyCollectionsReadOnlyError

    def add_item_to_collection(
        self,
        collection_id: str,
        *,
        source_type: str,
        source_id: str,
        title: str = "",
    ) -> str:
        """Reject writes to the superseded generic Collections tables."""
        raise LegacyCollectionsReadOnlyError

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
        COLLATE NOCASE ASC, collection_id ASC``). Count and page are read in
        one read-only snapshot (``read_transaction``), so this pure read never
        takes the write lock (task-15466).

        Args:
            limit: Maximum number of Collections to return.
            offset: Number of Collections to skip.

        Returns:
            A bounded page containing items, total, offset, and limit.

        Raises:
            LibraryCollectionsServiceError: If the local store cannot be read.
        """
        safe_limit = _validate_collection_page_limit(limit)
        safe_offset = _validate_collection_page_offset(offset)
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
                             collection.name COLLATE NOCASE ASC,
                             collection.collection_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    (safe_limit, safe_offset),
                ).fetchall()
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return {
            "items": [_library_collection_item(row) for row in rows],
            "total": int(total),
            "offset": safe_offset,
            "limit": safe_limit,
        }

    def locate_library_collection_page(
        self, collection_id: str, *, limit: int = 20
    ) -> dict | None:
        """Return one stable ID's rank-derived owning page.

        Rank metadata, exact total, and page rows are read from one SQLite
        snapshot under the same ordering as :meth:`list_library_collections`.
        The method returns ``None`` when the target is absent or soft-deleted.

        Args:
            collection_id: Stable Collection identifier to locate.
            limit: Exact number of rows in each owning page.

        Returns:
            Owning-page rows and rank metadata, or ``None`` when the target
            is absent or soft-deleted.

        Raises:
            LibraryCollectionsServiceError: If inputs are invalid or the
                local store cannot be read.
        """

        safe_collection_id = _validate_collection_id(collection_id)
        safe_limit = _validate_collection_page_limit(limit)
        try:
            with self.db.read_transaction() as conn:
                location = conn.execute(
                    """
                    WITH ranked AS (
                        SELECT
                            collection_id,
                            ROW_NUMBER() OVER (
                                ORDER BY created_at ASC,
                                         name COLLATE NOCASE ASC,
                                         collection_id ASC
                            ) - 1 AS target_rank,
                            COUNT(*) OVER () AS total
                        FROM library_collections
                        WHERE deleted_at IS NULL
                    )
                    SELECT target_rank, total
                    FROM ranked
                    WHERE collection_id = ?
                    """,
                    (safe_collection_id,),
                ).fetchone()
                if location is None:
                    return None
                target_rank = int(location["target_rank"])
                total = int(location["total"])
                offset = (target_rank // safe_limit) * safe_limit
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
                             collection.name COLLATE NOCASE ASC,
                             collection.collection_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    (safe_limit, offset),
                ).fetchall()
        except sqlite3.Error as exc:
            raise LibraryCollectionsServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return {
            "items": [_library_collection_item(row) for row in rows],
            "total": total,
            "limit": safe_limit,
            "offset": offset,
            "page": offset // safe_limit + 1,
            "target_id": safe_collection_id,
            "target_rank": target_rank,
            "target_index": target_rank - offset,
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


def _validate_list_limit(limit: int) -> int:
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_LIBRARY_COLLECTIONS_LIST_LIMIT
    return min(max(parsed, 1), MAX_LIBRARY_COLLECTIONS_LIST_LIMIT)


def _validate_collection_page_limit(limit: int) -> int:
    if type(limit) is not int or not 1 <= limit <= MAX_LIBRARY_COLLECTIONS_LIST_LIMIT:
        raise LibraryCollectionsServiceError(
            "limit must be an integer between 1 and 500."
        )
    return limit


def _validate_collection_page_offset(offset: int) -> int:
    if type(offset) is not int or not 0 <= offset <= MAX_SQLITE_COLLECTIONS_OFFSET:
        raise LibraryCollectionsServiceError(
            "offset must be a non-negative signed 64-bit integer."
        )
    return offset


def _validate_collection_id(collection_id: str) -> str:
    if (
        type(collection_id) is not str
        or not collection_id
        or collection_id != collection_id.strip()
        or len(collection_id) > 200
    ):
        raise LibraryCollectionsServiceError(
            "collection_id must be stable non-blank text of at most 200 characters."
        )
    return collection_id


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
