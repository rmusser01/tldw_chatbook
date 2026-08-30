"""SQLite repository for local Database Note folder hierarchy operations."""

from __future__ import annotations

import sqlite3
import unicodedata
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import NoReturn

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    FolderCapabilityError,
    FolderCollisionError,
    FolderConflictError,
    FolderMutationResult,
    FolderValidationError,
    NoteFolder,
    NoteFolderChildPage,
    NoteFolderMembership,
    NoteFolderManagedStatus,
    NoteFolderPage,
    NotePlacementPage,
    NotePlacementRecord,
    NoteTreeLocation,
    NoteTreeMutationContext,
    NoteTreePathStep,
    RestoredManagedMembershipReview,
    join_normalized_folder_path,
    normalize_folder_name,
)
from tldw_chatbook.Utils.fts5_match_forms import build_phrase_match_query

_FOLDER_COLUMNS = (
    "id, parent_id, name, normalized_name, path, normalized_path, version, deleted, "
    "modified_at"
)
_NOTE_COLUMNS = (
    "id, title, content, created_at, last_modified, deleted, client_id, version"
)
_COLLISION_PREFLIGHT_CHUNK_SIZE = 400
_MEMBERSHIP_QUERY_CHUNK_SIZE = 400
_MEMBERSHIP_ID_INSERT_ATTEMPTS = 3
_MAX_NOTE_TREE_PAGE_SIZE = 500
_TREE_SEARCH_NOTE_LIMIT = 250
_TREE_SEARCH_FOLDER_LIMIT = 500
_TREE_SEARCH_MEMBERSHIP_LIMIT = 1000
_FOLDER_PATH_SEGMENT_LIMIT = 64
_CALLER_FOLDER_ID_MAX_LENGTH = 256
_ASCII_ALNUM = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)
_CALLER_FOLDER_ID_CHARACTERS = _ASCII_ALNUM | frozenset("_.:-")
_MEMBERSHIP_COLUMNS = (
    "id, folder_id, note_id, ownership, owner_id, owner_active, version"
)
_MANAGED_ANCESTOR_SHADOW_SQL = (
    "m.ownership = 'managed' AND EXISTS ("
    "SELECT 1 FROM note_folder_memberships AS child_m "
    "INDEXED BY idx_note_folder_memberships_active_note "
    "JOIN note_folders AS child_f "
    "ON child_f.id = child_m.folder_id AND child_f.deleted = 0 "
    "WHERE child_m.deleted = 0 AND child_m.owner_active = 1 "
    "AND child_m.ownership = 'managed' "
    "AND child_m.note_id = m.note_id "
    "AND child_m.owner_id = m.owner_id "
    "AND substr(child_f.normalized_path, 1, "
    "length(f.normalized_path) + 1) = f.normalized_path || '/'"
    ")"
)


class LocalNoteFolderRepository:
    """Own local folder SQL while sharing the application's ChaChaNotes handle."""

    def __init__(self, db: CharactersRAGDB) -> None:
        """Create a repository over an already-initialized database handle."""
        if not isinstance(db, CharactersRAGDB):
            raise TypeError("db must be a CharactersRAGDB instance")
        self.db = db

    def create_folder(
        self,
        *,
        name: str,
        parent_id: str | None,
        folder_id: str | None = None,
    ) -> NoteFolder:
        """Create an active folder beneath an active parent.

        Args:
            name: User-visible name for the new folder.
            parent_id: Active parent folder identifier, or None for a root.
            folder_id: Optional caller-owned opaque identifier.

        Returns:
            The newly created folder.

        Raises:
            FolderCollisionError: If the normalized active path already exists.
            FolderValidationError: If the name or parent cannot be used.
            FolderConflictError: If database contention prevents the mutation.
        """
        if folder_id is None:
            selected_folder_id = str(uuid.uuid4())
        else:
            selected_folder_id = validate_deterministic_folder_id(folder_id)
        normalized = normalize_folder_name(name)
        now = _utc_timestamp()
        normalized_path: str | None = None

        try:
            with self.db.transaction() as cursor:
                parent_path = ""
                parent_normalized_path = ""
                if parent_id is not None:
                    parent_row = cursor.execute(
                        f"SELECT {_FOLDER_COLUMNS} FROM note_folders "
                        "WHERE id = ? AND deleted = 0",
                        (parent_id,),
                    ).fetchone()
                    if parent_row is None:
                        raise FolderValidationError(
                            "Parent folder does not exist or is inactive."
                        )
                    _require_manual_folder_subtree(cursor, parent_id)
                    parent_path = str(parent_row["path"])
                    parent_normalized_path = str(parent_row["normalized_path"])

                path = _join_display_folder_path(parent_path, normalized.display)
                normalized_path = join_normalized_folder_path(
                    parent_normalized_path, normalized.key
                )
                cursor.execute(
                    """
                    INSERT INTO note_folders(
                        id, parent_id, name, normalized_name, path,
                        normalized_path, version, deleted, created_at, modified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
                    """,
                    (
                        selected_folder_id,
                        parent_id,
                        normalized.display,
                        normalized.key,
                        path,
                        normalized_path,
                        now,
                        now,
                    ),
                )
                inserted = cursor.execute(
                    f"SELECT {_FOLDER_COLUMNS} FROM note_folders WHERE id = ?",
                    (selected_folder_id,),
                ).fetchone()
                if inserted is None:  # pragma: no cover - SQLite guarantees this
                    raise FolderValidationError("Created folder could not be read.")
                return _folder_from_row(inserted)
        except sqlite3.IntegrityError as exc:
            if (
                getattr(exc, "sqlite_errorcode", None)
                == sqlite3.SQLITE_CONSTRAINT_UNIQUE
                and normalized_path is not None
                and "note_folders.normalized_path" in str(exc)
            ):
                raise FolderCollisionError(
                    "An active folder already uses the normalized path."
                ) from exc
            raise FolderValidationError("Folder could not be created.") from exc
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)

    def get_folder_by_path(self, folder_segments: Iterable[str]) -> NoteFolder | None:
        """Return one active folder by an exact normalized segment path."""
        if isinstance(folder_segments, (str, bytes)):
            raise FolderValidationError(
                "folder_segments must be a collection of path segments."
            )
        try:
            iterator = iter(folder_segments)
        except TypeError as exc:
            raise FolderValidationError(
                "folder_segments must be a collection of path segments."
            ) from exc

        normalized_path = ""
        count = 0
        for count, segment in enumerate(iterator, start=1):
            if count > _FOLDER_PATH_SEGMENT_LIMIT:
                raise FolderValidationError(
                    "folder_segments exceeds the allowed range."
                )
            normalized = normalize_folder_name(segment)
            normalized_path = join_normalized_folder_path(
                normalized_path, normalized.key
            )
        if count == 0:
            raise FolderValidationError("folder_segments must identify a folder.")

        with self.db.transaction() as cursor:
            row = cursor.execute(
                f"SELECT {_FOLDER_COLUMNS} FROM note_folders "
                "WHERE normalized_path = ? AND deleted = 0",
                (normalized_path,),
            ).fetchone()
        return _folder_from_row(row) if row is not None else None

    def get_folder(
        self, folder_id: str, *, include_deleted: bool = False
    ) -> NoteFolder | None:
        """Return one folder by exact identifier.

        Args:
            folder_id: Folder identifier to load.
            include_deleted: Whether a matching soft-deleted row may be returned.

        Returns:
            The matching folder, or None when it is unavailable.
        """
        _validate_folder_id(folder_id, field="folder_id")
        deleted_clause = "" if include_deleted else " AND deleted = 0"
        with self.db.transaction() as cursor:
            row = cursor.execute(
                f"SELECT {_FOLDER_COLUMNS} FROM note_folders "
                f"WHERE id = ?{deleted_clause}",
                (folder_id,),
            ).fetchone()
        return _folder_from_row(row) if row is not None else None

    def list_children(
        self, *, parent_id: str | None, limit: int, offset: int
    ) -> NoteFolderPage:
        """Return a bounded page of active direct children."""
        _validate_int_bound(
            "limit", limit, minimum=1, maximum=_MAX_NOTE_TREE_PAGE_SIZE
        )
        _validate_int_bound("offset", offset, minimum=0)
        if parent_id is not None:
            _validate_folder_id(parent_id, field="parent_id")
        parent_predicate = "parent_id IS NULL" if parent_id is None else "parent_id = ?"
        parent_params: tuple[object, ...] = () if parent_id is None else (parent_id,)
        with self.db.transaction() as cursor:
            total_row = cursor.execute(
                f"SELECT COUNT(*) AS total FROM note_folders "
                f"WHERE deleted = 0 AND {parent_predicate}",
                parent_params,
            ).fetchone()
            rows = cursor.execute(
                f"SELECT {_FOLDER_COLUMNS} FROM note_folders "
                f"WHERE deleted = 0 AND {parent_predicate} "
                "ORDER BY normalized_name, id LIMIT ? OFFSET ?",
                (*parent_params, limit, offset),
            ).fetchall()
        folders = tuple(_folder_from_row(row) for row in rows)
        total = int(total_row["total"] if total_row is not None else 0)
        returned_end = offset + len(folders)
        return NoteFolderPage(
            folders=folders,
            memberships=(),
            notes=(),
            total_folders=total,
            total_notes=0,
            next_offset=returned_end if folders and returned_end < total else None,
            # Preserve ``next_offset`` for callers of the original child-list API
            # while exposing the unambiguous folder cursor used by tree batching.
            next_folder_offset=(
                returned_end if folders and returned_end < total else None
            ),
        )

    def page_child_folders(
        self, *, parent_id: str | None, limit: int, offset: int
    ) -> NoteFolderChildPage:
        """Return an exact page of active direct child folders.

        Args:
            parent_id: Exact parent identifier, or ``None`` for root folders.
            limit: Maximum folders to return.
            offset: Zero-based folder offset.

        Returns:
            Direct children with exact total and bidirectional page cursors.

        Raises:
            FolderValidationError: If an identifier or page bound is invalid.
        """
        _validate_int_bound(
            "limit", limit, minimum=1, maximum=_MAX_NOTE_TREE_PAGE_SIZE
        )
        _validate_int_bound("offset", offset, minimum=0)
        if parent_id is not None:
            _validate_folder_id(parent_id, field="parent_id")
        parent_predicate = "parent_id IS NULL" if parent_id is None else "parent_id = ?"
        parent_params: tuple[object, ...] = () if parent_id is None else (parent_id,)
        with self.db.transaction() as cursor:
            total = int(
                cursor.execute(
                    "SELECT COUNT(*) AS total FROM note_folders "
                    f"WHERE deleted = 0 AND {parent_predicate}",
                    parent_params,
                ).fetchone()["total"]
            )
            rows = cursor.execute(
                f"SELECT {_FOLDER_COLUMNS} FROM note_folders "
                f"WHERE deleted = 0 AND {parent_predicate} "
                "ORDER BY normalized_name, id LIMIT ? OFFSET ?",
                (*parent_params, limit, offset),
            ).fetchall()
            folders = tuple(_folder_from_row(row) for row in rows)
            managed_rows = _load_managed_folder_rows(
                cursor, tuple(folder.folder_id for folder in folders)
            )
        managed_by_id = {
            str(row["folder_id"]): (
                None if row["owner_active"] is None else bool(row["owner_active"])
            )
            for row in managed_rows
        }
        end = offset + len(folders)
        return NoteFolderChildPage(
            folders=folders,
            total_folders=total,
            start_offset=offset,
            previous_offset=_previous_page_offset(offset, limit, total),
            next_offset=end if end < total else None,
            folder_statuses=tuple(
                NoteFolderManagedStatus(
                    folder.folder_id,
                    (
                        "normal"
                        if managed_by_id.get(folder.folder_id) is None
                        else "protected"
                        if managed_by_id[folder.folder_id]
                        else "inactive_managed"
                    ),
                )
                for folder in folders
            ),
        )

    def page_note_placements(
        self, *, parent_id: str | None, limit: int, offset: int
    ) -> NotePlacementPage:
        """Return an exact page of visible note placements beneath one parent.

        Args:
            parent_id: Exact folder identifier, or ``None`` for Unfiled notes.
            limit: Maximum placements to return.
            offset: Zero-based placement offset.

        Returns:
            Visible placement rows with exact total and page cursors.

        Raises:
            FolderValidationError: If an identifier or page bound is invalid.
        """
        _validate_int_bound(
            "limit", limit, minimum=1, maximum=_MAX_NOTE_TREE_PAGE_SIZE
        )
        _validate_int_bound("offset", offset, minimum=0)
        if parent_id is not None:
            _validate_folder_id(parent_id, field="parent_id")

        with self.db.transaction() as cursor:
            if parent_id is None:
                unfiled_from_sql = (
                    "FROM notes AS n WHERE n.deleted = 0 AND NOT EXISTS ("
                    "SELECT 1 FROM note_folder_memberships AS m "
                    "JOIN note_folders AS f "
                    "ON f.id = m.folder_id AND f.deleted = 0 "
                    "WHERE m.note_id = n.id AND m.deleted = 0 "
                    "AND m.owner_active = 1)"
                )
                total = int(
                    cursor.execute(
                        f"SELECT COUNT(*) AS total {unfiled_from_sql}"
                    ).fetchone()["total"]
                )
                rows = cursor.execute(
                    f"SELECT n.{_NOTE_COLUMNS.replace(', ', ', n.')} "
                    f"{unfiled_from_sql} "
                    "ORDER BY n.title COLLATE NOCASE, n.id LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
                placements = tuple(
                    NotePlacementRecord(
                        note=_note_from_row(row), folder_id=None, membership=None
                    )
                    for row in rows
                )
            else:
                effective_memberships_sql = (
                    "WITH effective_memberships AS ("
                    "SELECT n.id, n.title, n.content, n.created_at, "
                    "n.last_modified, n.deleted, n.client_id, n.version, "
                    "m.id AS membership_id, m.folder_id AS membership_folder_id, "
                    "m.note_id AS membership_note_id, "
                    "m.ownership AS membership_ownership, "
                    "m.owner_id AS membership_owner_id, "
                    "m.owner_active AS membership_owner_active, "
                    "m.version AS membership_version "
                    "FROM note_folder_memberships AS m "
                    "JOIN note_folders AS f "
                    "ON f.id = m.folder_id AND f.deleted = 0 "
                    "JOIN notes AS n ON n.id = m.note_id AND n.deleted = 0 "
                    "WHERE m.folder_id = ? AND m.deleted = 0 "
                    f"AND m.owner_active = 1 AND NOT ({_MANAGED_ANCESTOR_SHADOW_SQL})"
                    ") "
                )
                total = int(
                    cursor.execute(
                        f"{effective_memberships_sql}"
                        "SELECT COUNT(*) AS total FROM effective_memberships",
                        (parent_id,),
                    ).fetchone()["total"]
                )
                rows = cursor.execute(
                    f"{effective_memberships_sql}"
                    "SELECT * FROM effective_memberships "
                    "ORDER BY title COLLATE NOCASE, id, membership_id "
                    "LIMIT ? OFFSET ?",
                    (parent_id, limit, offset),
                ).fetchall()
                placements = tuple(
                    NotePlacementRecord(
                        note=_note_from_row(row),
                        folder_id=str(row["membership_folder_id"]),
                        membership=NoteFolderMembership(
                            membership_id=str(row["membership_id"]),
                            folder_id=str(row["membership_folder_id"]),
                            note_id=str(row["membership_note_id"]),
                            ownership=row["membership_ownership"],
                            owner_id=str(row["membership_owner_id"]),
                            owner_active=bool(row["membership_owner_active"]),
                            version=int(row["membership_version"]),
                        ),
                    )
                    for row in rows
                )

            managed_rows = (
                _load_managed_folder_rows(cursor, (parent_id,))
                if parent_id is not None
                else ()
            )

        end = offset + len(placements)
        if parent_id is None:
            folder_statuses: tuple[NoteFolderManagedStatus, ...] = ()
        elif managed_rows and managed_rows[0]["owner_active"] is not None:
            folder_statuses = (
                NoteFolderManagedStatus(
                    parent_id,
                    (
                        "protected"
                        if bool(managed_rows[0]["owner_active"])
                        else "inactive_managed"
                    ),
                ),
            )
        else:
            folder_statuses = (NoteFolderManagedStatus(parent_id, "normal"),)
        return NotePlacementPage(
            placements=placements,
            total_placements=total,
            start_offset=offset,
            previous_offset=_previous_page_offset(offset, limit, total),
            next_offset=end if end < total else None,
            folder_statuses=folder_statuses,
        )

    def locate_note_tree_folder(
        self, *, folder_id: str, page_size: int
    ) -> NoteTreeLocation | None:
        """Locate one active folder in the exact paged tree.

        Args:
            folder_id: Exact active folder identifier.
            page_size: Folder page size used by the tree.

        Returns:
            The root-to-folder location, or None when the folder is inactive.

        Raises:
            FolderValidationError: If an identifier or page bound is invalid.
        """
        _validate_folder_id(folder_id, field="folder_id")
        _validate_int_bound(
            "page_size", page_size, minimum=1, maximum=_MAX_NOTE_TREE_PAGE_SIZE
        )
        with self.db.transaction() as cursor:
            path = _load_note_tree_path(
                cursor, folder_id=folder_id, page_size=page_size
            )
        if not path:
            return None
        return NoteTreeLocation(
            placement_id=FolderPlacementId.folder(folder_id),
            note_id=None,
            membership_id=None,
            path=path,
            placement_offset=None,
        )

    def locate_note_tree_placement(
        self,
        *,
        note_id: str,
        page_size: int,
        preferred_folder_id: str | None = None,
        preferred_membership_id: str | None = None,
    ) -> NoteTreeLocation | None:
        """Locate the preferred surviving placement of one active note.

        Args:
            note_id: Exact active note identifier.
            page_size: Placement page size used by the tree.
            preferred_folder_id: Folder to prefer after exact membership lookup.
            preferred_membership_id: Exact surviving membership to prefer.

        Returns:
            A filed or Unfiled location, or None when the note is inactive.

        Raises:
            FolderValidationError: If an identifier or page bound is invalid.
        """
        _validate_folder_id(note_id, field="note_id")
        _validate_int_bound(
            "page_size", page_size, minimum=1, maximum=_MAX_NOTE_TREE_PAGE_SIZE
        )
        if preferred_folder_id is not None:
            _validate_folder_id(preferred_folder_id, field="preferred_folder_id")
        if preferred_membership_id is not None:
            _validate_folder_id(
                preferred_membership_id, field="preferred_membership_id"
            )

        with self.db.transaction() as cursor:
            selected = cursor.execute(
                f"""
                SELECT n.{_NOTE_COLUMNS.replace(", ", ", n.")},
                       m.id AS membership_id,
                       m.folder_id AS membership_folder_id
                FROM note_folder_memberships AS m
                INDEXED BY idx_note_folder_memberships_active_note
                JOIN note_folders AS f
                  ON f.id = m.folder_id AND f.deleted = 0
                JOIN notes AS n ON n.id = m.note_id AND n.deleted = 0
                WHERE m.note_id = ? AND m.deleted = 0 AND m.owner_active = 1
                  AND NOT ({_MANAGED_ANCESTOR_SHADOW_SQL})
                ORDER BY
                    CASE
                        WHEN ? IS NOT NULL AND m.id = ? THEN 0
                        WHEN ? IS NOT NULL AND m.folder_id = ? THEN 1
                        ELSE 2
                    END,
                    f.normalized_path, f.id, m.id
                LIMIT 1
                """,
                (
                    note_id,
                    preferred_membership_id,
                    preferred_membership_id,
                    preferred_folder_id,
                    preferred_folder_id,
                ),
            ).fetchone()
            if selected is None:
                note = cursor.execute(
                    f"SELECT {_NOTE_COLUMNS} FROM notes WHERE id = ? AND deleted = 0",
                    (note_id,),
                ).fetchone()
                if note is None:
                    return None
                rank = int(
                    cursor.execute(
                        """
                        SELECT COUNT(*) AS rank
                        FROM notes AS candidate
                        WHERE candidate.deleted = 0
                          AND NOT EXISTS (
                              SELECT 1
                              FROM note_folder_memberships AS m
                              JOIN note_folders AS f
                                ON f.id = m.folder_id AND f.deleted = 0
                              WHERE m.note_id = candidate.id
                                AND m.deleted = 0 AND m.owner_active = 1
                          )
                          AND (
                              candidate.title COLLATE NOCASE < ? COLLATE NOCASE
                              OR (
                                  candidate.title = ? COLLATE NOCASE
                                  AND candidate.id < ?
                              )
                          )
                        """,
                        (note["title"], note["title"], note_id),
                    ).fetchone()["rank"]
                )
                return NoteTreeLocation(
                    placement_id=FolderPlacementId.unfiled(note_id),
                    note_id=note_id,
                    membership_id=None,
                    path=(),
                    placement_offset=(rank // page_size) * page_size,
                )

            folder_id = str(selected["membership_folder_id"])
            membership_id = str(selected["membership_id"])
            rank = int(
                cursor.execute(
                    f"""
                    SELECT COUNT(*) AS rank
                    FROM note_folder_memberships AS m
                    JOIN note_folders AS f
                      ON f.id = m.folder_id AND f.deleted = 0
                    JOIN notes AS n ON n.id = m.note_id AND n.deleted = 0
                    WHERE m.folder_id = ? AND m.deleted = 0 AND m.owner_active = 1
                      AND NOT ({_MANAGED_ANCESTOR_SHADOW_SQL})
                      AND (
                          n.title COLLATE NOCASE < ? COLLATE NOCASE
                          OR (
                              n.title = ? COLLATE NOCASE
                              AND (
                                  n.id < ? OR (n.id = ? AND m.id < ?)
                              )
                          )
                      )
                    """,
                    (
                        folder_id,
                        selected["title"],
                        selected["title"],
                        note_id,
                        note_id,
                        membership_id,
                    ),
                ).fetchone()["rank"]
            )
            path = _load_note_tree_path(
                cursor, folder_id=folder_id, page_size=page_size
            )
            if not path:
                return None

        return NoteTreeLocation(
            placement_id=FolderPlacementId.note(folder_id, note_id, membership_id),
            note_id=note_id,
            membership_id=membership_id,
            path=path,
            placement_offset=(rank // page_size) * page_size,
        )

    def load_note_tree_mutation_context(
        self,
        *,
        folder_ids: Iterable[str] = (),
        note_ids: Iterable[str] = (),
        include_folder_subtrees: bool = False,
    ) -> NoteTreeMutationContext:
        """Return exact folder branches affected by note-tree mutations.

        Args:
            folder_ids: Active or recently changed folder identifiers.
            note_ids: Note identifiers whose active placement parents are needed.
            include_folder_subtrees: Whether affected folder subtrees are included.

        Returns:
            Deterministic involved folders, parents, ancestors, and placements.

        Raises:
            FolderValidationError: If an input collection or flag is invalid.
        """
        normalized_folder_ids = _normalize_ids(folder_ids, field="folder_ids")
        normalized_note_ids = _normalize_ids(note_ids, field="note_ids")
        if not isinstance(include_folder_subtrees, bool):
            raise FolderValidationError("include_folder_subtrees must be a boolean.")

        with self.db.transaction() as cursor:
            requested_rows: list[sqlite3.Row] = []
            for chunk in _chunks(normalized_folder_ids, _MEMBERSHIP_QUERY_CHUNK_SIZE):
                requested_rows.extend(
                    cursor.execute(
                        f"SELECT {_FOLDER_COLUMNS} FROM note_folders "
                        f"WHERE id IN ({_placeholders(len(chunk))}) ORDER BY id",
                        chunk,
                    ).fetchall()
                )

            involved_by_id = {str(row["id"]): row for row in requested_rows}
            if include_folder_subtrees:
                for row in requested_rows:
                    for subtree_row in _load_subtree(
                        cursor, row, deleted=bool(row["deleted"])
                    ):
                        involved_by_id[str(subtree_row["id"])] = subtree_row
            involved_rows = tuple(
                involved_by_id[folder_id] for folder_id in sorted(involved_by_id)
            )
            involved_ids = tuple(str(row["id"]) for row in involved_rows)
            direct_parent_ids = {
                str(row["parent_id"]) if row["parent_id"] is not None else None
                for row in involved_rows
            }

            ancestor_ids: set[str] = set()
            for chunk in _chunks(involved_ids, _MEMBERSHIP_QUERY_CHUNK_SIZE):
                rows = cursor.execute(
                    f"""
                    WITH RECURSIVE ancestors(id, parent_id, depth) AS (
                        SELECT id, parent_id, 0
                        FROM note_folders
                        WHERE id IN ({_placeholders(len(chunk))})
                        UNION ALL
                        SELECT parent.id, parent.parent_id, child.depth + 1
                        FROM note_folders AS parent
                        JOIN ancestors AS child ON parent.id = child.parent_id
                    )
                    SELECT DISTINCT id FROM ancestors WHERE depth > 0 ORDER BY id
                    """,
                    chunk,
                ).fetchall()
                ancestor_ids.update(str(row["id"]) for row in rows)

            placement_parent_ids: set[str] = set()
            for chunk in _chunks(normalized_note_ids, _MEMBERSHIP_QUERY_CHUNK_SIZE):
                rows = cursor.execute(
                    f"""
                    SELECT DISTINCT m.folder_id
                    FROM note_folder_memberships AS m
                    JOIN note_folders AS f
                      ON f.id = m.folder_id AND f.deleted = 0
                    WHERE m.note_id IN ({_placeholders(len(chunk))})
                      AND m.deleted = 0 AND m.owner_active = 1
                    ORDER BY m.folder_id
                    """,
                    chunk,
                ).fetchall()
                placement_parent_ids.update(str(row["folder_id"]) for row in rows)

        return NoteTreeMutationContext(
            folder_ids=tuple(sorted(involved_ids)),
            parent_ids=tuple(
                sorted(
                    direct_parent_ids,
                    key=lambda value: (value is not None, value or ""),
                )
            ),
            ancestor_ids=tuple(sorted(ancestor_ids)),
            placement_parent_ids=tuple(sorted(placement_parent_ids)),
        )

    def search_note_tree_placements(
        self, *, query: str, limit: int, offset: int
    ) -> NotePlacementPage:
        """Return one coherent exact page of content/path-matched placements.

        Args:
            query: Plain text matched against note FTS and normalized folder paths.
            limit: Maximum placements to return.
            offset: Zero-based placement offset.

        Returns:
            Exact visible placements plus page-local folder ancestors.

        Raises:
            FolderValidationError: If the query or page bounds are invalid.
        """
        normalized_query = _normalize_folder_search_query(query)
        _validate_int_bound(
            "limit", limit, minimum=1, maximum=_MAX_NOTE_TREE_PAGE_SIZE
        )
        _validate_int_bound("offset", offset, minimum=0)
        fts_query = build_phrase_match_query(query)

        with self.db.transaction() as cursor:
            if not normalized_query:
                return NotePlacementPage(
                    placements=(),
                    total_placements=0,
                    start_offset=offset,
                    previous_offset=_previous_page_offset(offset, limit, 0),
                    next_offset=None,
                )

            if fts_query:
                matching_notes_sql = (
                    "SELECT n.id FROM notes_fts "
                    "JOIN notes AS n ON n.rowid = notes_fts.rowid "
                    "WHERE notes_fts MATCH ? AND n.deleted = 0"
                )
                query_params: tuple[object, ...] = (fts_query, normalized_query)
            else:
                matching_notes_sql = "SELECT id FROM notes WHERE 0"
                query_params = (normalized_query,)

            placement_cte = f"""
                WITH matching_notes AS ({matching_notes_sql}),
                effective_memberships AS (
                    SELECT n.{_NOTE_COLUMNS.replace(", ", ", n.")},
                           m.id AS membership_id,
                           m.folder_id AS membership_folder_id,
                           m.note_id AS membership_note_id,
                           m.ownership AS membership_ownership,
                           m.owner_id AS membership_owner_id,
                           m.owner_active AS membership_owner_active,
                           m.version AS membership_version,
                           f.normalized_path AS folder_normalized_path
                    FROM note_folder_memberships AS m
                    JOIN note_folders AS f
                      ON f.id = m.folder_id AND f.deleted = 0
                    JOIN notes AS n ON n.id = m.note_id AND n.deleted = 0
                    WHERE m.deleted = 0 AND m.owner_active = 1
                      AND NOT ({_MANAGED_ANCESTOR_SHADOW_SQL})
                ),
                visible_placements AS (
                    SELECT 0 AS placement_kind, effective_memberships.*
                    FROM effective_memberships
                    WHERE id IN (SELECT id FROM matching_notes)
                       OR instr(folder_normalized_path, ?) > 0
                    UNION ALL
                    SELECT 1 AS placement_kind,
                           n.{_NOTE_COLUMNS.replace(", ", ", n.")},
                           NULL AS membership_id,
                           NULL AS membership_folder_id,
                           NULL AS membership_note_id,
                           NULL AS membership_ownership,
                           NULL AS membership_owner_id,
                           NULL AS membership_owner_active,
                           NULL AS membership_version,
                           '' AS folder_normalized_path
                    FROM notes AS n
                    WHERE n.deleted = 0 AND n.id IN (SELECT id FROM matching_notes)
                      AND NOT EXISTS (
                          SELECT 1
                          FROM note_folder_memberships AS m
                          JOIN note_folders AS f
                            ON f.id = m.folder_id AND f.deleted = 0
                          WHERE m.note_id = n.id
                            AND m.deleted = 0 AND m.owner_active = 1
                      )
                )
            """
            total = int(
                cursor.execute(
                    f"{placement_cte}SELECT COUNT(*) AS total FROM visible_placements",
                    query_params,
                ).fetchone()["total"]
            )
            rows = cursor.execute(
                f"{placement_cte}"
                "SELECT * FROM visible_placements "
                "ORDER BY placement_kind, folder_normalized_path, "
                "title COLLATE NOCASE, id, membership_id "
                "LIMIT ? OFFSET ?",
                (*query_params, limit, offset),
            ).fetchall()

            folder_ids = tuple(
                sorted(
                    {
                        str(row["membership_folder_id"])
                        for row in rows
                        if row["membership_folder_id"] is not None
                    }
                )
            )
            ancestor_rows: Sequence[sqlite3.Row] = ()
            if folder_ids:
                ancestor_rows = cursor.execute(
                    f"""
                    WITH RECURSIVE ancestors(id) AS (
                        SELECT id FROM note_folders
                        WHERE deleted = 0
                          AND id IN ({_placeholders(len(folder_ids))})
                        UNION
                        SELECT folder.parent_id
                        FROM note_folders AS folder
                        JOIN ancestors ON ancestors.id = folder.id
                        WHERE folder.deleted = 0 AND folder.parent_id IS NOT NULL
                    )
                    SELECT note_folders.{_FOLDER_COLUMNS.replace(", ", ", note_folders.")}
                    FROM note_folders
                    JOIN ancestors ON ancestors.id = note_folders.id
                    WHERE note_folders.deleted = 0
                    ORDER BY note_folders.normalized_path, note_folders.id
                    """,
                    folder_ids,
                ).fetchall()

        placements = tuple(
            NotePlacementRecord(
                note=_note_from_row(row),
                folder_id=(
                    str(row["membership_folder_id"])
                    if row["membership_folder_id"] is not None
                    else None
                ),
                membership=(
                    NoteFolderMembership(
                        membership_id=str(row["membership_id"]),
                        folder_id=str(row["membership_folder_id"]),
                        note_id=str(row["membership_note_id"]),
                        ownership=row["membership_ownership"],
                        owner_id=str(row["membership_owner_id"]),
                        owner_active=bool(row["membership_owner_active"]),
                        version=int(row["membership_version"]),
                    )
                    if row["membership_id"] is not None
                    else None
                ),
            )
            for row in rows
        )
        end = offset + len(placements)
        return NotePlacementPage(
            placements=placements,
            total_placements=total,
            start_offset=offset,
            previous_offset=_previous_page_offset(offset, limit, total),
            next_offset=end if end < total else None,
            ancestor_folders=tuple(_folder_from_row(row) for row in ancestor_rows),
        )

    def load_tree_batch(
        self,
        *,
        expanded_folder_ids: Iterable[str],
        note_limit: int,
        note_offset: int = 0,
        folder_limit: int = 500,
        folder_offset: int = 0,
        membership_limit: int = 1000,
        membership_offset: int = 0,
        load_notes: bool = True,
    ) -> NoteFolderPage:
        """Bulk-load roots or the immediate contents of expanded folders.

        Args:
            expanded_folder_ids: Folders whose immediate contents should load.
            note_limit: Maximum notes to return when note loading is enabled.
            note_offset: Offset into the bounded note page.
            folder_limit: Maximum child folders to return.
            folder_offset: Offset into the bounded folder page.
            membership_limit: Maximum memberships for the current note page.
            membership_offset: Offset into the bounded membership page.
            load_notes: Whether to query notes and their memberships. Disable this
                when only an independent folder cursor remains.

        Returns:
            One bounded folder-tree page with independent continuation cursors.

        Raises:
            FolderValidationError: If a bound or flag is invalid.
        """
        _validate_int_bound("note_limit", note_limit, minimum=1, maximum=1000)
        _validate_int_bound("note_offset", note_offset, minimum=0)
        _validate_int_bound(
            "folder_limit",
            folder_limit,
            minimum=1,
            maximum=_MAX_NOTE_TREE_PAGE_SIZE,
        )
        _validate_int_bound("folder_offset", folder_offset, minimum=0)
        _validate_int_bound(
            "membership_limit", membership_limit, minimum=1, maximum=1000
        )
        _validate_int_bound("membership_offset", membership_offset, minimum=0)
        if not isinstance(load_notes, bool):
            raise FolderValidationError("load_notes must be a boolean.")
        expanded_ids = _normalize_folder_ids(expanded_folder_ids)
        with self.db.transaction() as cursor:
            if expanded_ids:
                placeholders = _placeholders(len(expanded_ids))
                note_candidates_sql = (
                    f"SELECT DISTINCT n.{_NOTE_COLUMNS.replace(', ', ', n.')} "
                    "FROM notes AS n "
                    "JOIN note_folder_memberships AS m "
                    "ON m.note_id = n.id AND m.deleted = 0 "
                    "JOIN note_folders AS f "
                    "ON f.id = m.folder_id AND f.deleted = 0 "
                    f"WHERE n.deleted = 0 AND m.folder_id IN ({placeholders})"
                )
                folder_rows = cursor.execute(
                    f"SELECT {_FOLDER_COLUMNS}, COUNT(*) OVER() AS _total_folders "
                    "FROM note_folders WHERE deleted = 0 "
                    f"AND parent_id IN ({placeholders}) "
                    "ORDER BY normalized_path, id LIMIT ? OFFSET ?",
                    (*expanded_ids, folder_limit, folder_offset),
                ).fetchall()
                note_rows = (
                    cursor.execute(
                        f"SELECT candidate.*, COUNT(*) OVER() AS _total_notes FROM ("
                        f"{note_candidates_sql}"
                        ") AS candidate ORDER BY title COLLATE NOCASE, id "
                        "LIMIT ? OFFSET ?",
                        (*expanded_ids, note_limit, note_offset),
                    ).fetchall()
                    if load_notes
                    else ()
                )
            else:
                folder_rows = cursor.execute(
                    f"SELECT {_FOLDER_COLUMNS}, COUNT(*) OVER() AS _total_folders "
                    "FROM note_folders WHERE deleted = 0 AND parent_id IS NULL "
                    "ORDER BY normalized_path, id LIMIT ? OFFSET ?",
                    (folder_limit, folder_offset),
                ).fetchall()
                note_rows = (
                    cursor.execute(
                        f"SELECT candidate.*, COUNT(*) OVER() AS _total_notes FROM ("
                        f"SELECT n.{_NOTE_COLUMNS.replace(', ', ', n.')} "
                        "FROM notes AS n WHERE n.deleted = 0 AND NOT EXISTS ("
                        "SELECT 1 FROM note_folder_memberships AS m "
                        "JOIN note_folders AS f "
                        "ON f.id = m.folder_id AND f.deleted = 0 "
                        "WHERE m.note_id = n.id AND m.deleted = 0 "
                        "AND m.owner_active = 1"
                        ")"
                        ") AS candidate ORDER BY title COLLATE NOCASE, id "
                        "LIMIT ? OFFSET ?",
                        (note_limit, note_offset),
                    ).fetchall()
                    if load_notes
                    else ()
                )

            page_note_ids = tuple(str(row["id"]) for row in note_rows)
            if expanded_ids and page_note_ids:
                page_notes_cte = (
                    "WITH page_notes AS (SELECT candidate.id FROM ("
                    f"{note_candidates_sql}"
                    ") AS candidate ORDER BY title COLLATE NOCASE, id "
                    "LIMIT ? OFFSET ?) "
                )
                membership_from_sql = (
                    "FROM note_folder_memberships AS m "
                    "JOIN note_folders AS f "
                    "ON f.id = m.folder_id AND f.deleted = 0 "
                    "JOIN page_notes AS page ON page.id = m.note_id "
                    "WHERE m.deleted = 0 "
                    f"AND m.folder_id IN ({placeholders}) "
                )
                membership_params = (
                    *expanded_ids,
                    note_limit,
                    note_offset,
                    *expanded_ids,
                )
                membership_rows = cursor.execute(
                    f"{page_notes_cte}"
                    "SELECT m.id, m.folder_id, m.note_id, m.ownership, m.owner_id, "
                    "m.owner_active, m.version, "
                    "COUNT(*) OVER() AS _total_memberships "
                    f"{membership_from_sql}"
                    "ORDER BY f.normalized_path, m.note_id, m.ownership, "
                    "m.owner_id, m.id LIMIT ? OFFSET ?",
                    (*membership_params, membership_limit, membership_offset),
                ).fetchall()
            else:
                membership_rows = ()

            managed_folder_rows = _load_managed_folder_rows(
                cursor, (str(row["id"]) for row in folder_rows)
            )

            total_memberships = (
                int(membership_rows[0]["_total_memberships"])
                if membership_rows
                else 0
            )
            if (
                expanded_ids
                and page_note_ids
                and not membership_rows
                and membership_offset
            ):
                total_memberships = int(
                    cursor.execute(
                        f"{page_notes_cte}SELECT COUNT(*) AS total "
                        f"{membership_from_sql}",
                        membership_params,
                    ).fetchone()["total"]
                )

            total_folders = (
                int(folder_rows[0]["_total_folders"]) if folder_rows else 0
            )
            if not folder_rows and folder_offset:
                if expanded_ids:
                    total_folders = int(
                        cursor.execute(
                            "SELECT COUNT(*) AS total FROM note_folders "
                            "WHERE deleted = 0 "
                            f"AND parent_id IN ({_placeholders(len(expanded_ids))})",
                            expanded_ids,
                        ).fetchone()["total"]
                    )
                else:
                    total_folders = int(
                        cursor.execute(
                            "SELECT COUNT(*) AS total FROM note_folders "
                            "WHERE deleted = 0 AND parent_id IS NULL"
                        ).fetchone()["total"]
                    )
            total_notes = int(note_rows[0]["_total_notes"]) if note_rows else 0
            if not note_rows and note_offset:
                if expanded_ids:
                    total_notes = int(
                        cursor.execute(
                            "SELECT COUNT(DISTINCT n.id) AS total FROM notes AS n "
                            "JOIN note_folder_memberships AS m ON m.note_id = n.id "
                            "AND m.deleted = 0 "
                            "JOIN note_folders AS f ON f.id = m.folder_id "
                            "AND f.deleted = 0 "
                            "WHERE n.deleted = 0 "
                            f"AND m.folder_id IN ({_placeholders(len(expanded_ids))})",
                            expanded_ids,
                        ).fetchone()["total"]
                    )
                else:
                    total_notes = int(
                        cursor.execute(
                            "SELECT COUNT(*) AS total FROM notes AS n "
                            "WHERE n.deleted = 0 AND NOT EXISTS ("
                            "SELECT 1 FROM note_folder_memberships AS m "
                            "JOIN note_folders AS f ON f.id = m.folder_id "
                            "AND f.deleted = 0 WHERE m.note_id = n.id "
                            "AND m.deleted = 0 AND m.owner_active = 1)"
                        ).fetchone()["total"]
                    )

        folders = tuple(_folder_from_row(row) for row in folder_rows)
        memberships = tuple(_membership_from_row(row) for row in membership_rows)
        notes = tuple(_note_from_row(row) for row in note_rows)
        note_end = note_offset + len(notes)
        folder_end = folder_offset + len(folders)
        membership_end = membership_offset + len(memberships)
        managed_folder_ids = tuple(
            str(row["folder_id"])
            for row in managed_folder_rows
            if row["owner_active"] is not None
        )
        inactive_managed_folder_ids = tuple(
            str(row["folder_id"])
            for row in managed_folder_rows
            if row["owner_active"] is not None and not bool(row["owner_active"])
        )
        return NoteFolderPage(
            folders=folders,
            memberships=memberships,
            notes=notes,
            total_folders=total_folders,
            total_notes=total_notes,
            next_offset=note_end if notes and note_end < total_notes else None,
            next_folder_offset=(
                folder_end if folders and folder_end < total_folders else None
            ),
            total_memberships=total_memberships,
            next_membership_offset=(
                membership_end
                if memberships and membership_end < total_memberships
                else None
            ),
            managed_folder_ids=managed_folder_ids,
            inactive_managed_folder_ids=inactive_managed_folder_ids,
            unfiled_note_ids=(
                tuple(str(row["id"]) for row in note_rows)
                if not expanded_ids
                else ()
            ),
        )

    def attach_manual(
        self,
        *,
        folder_id: str,
        note_id: str,
        expected_note_version: int | None = None,
    ) -> NoteFolderMembership:
        """Attach one user-owned placement, reviving its latest history."""
        _validate_folder_id(folder_id, field="folder_id")
        _validate_folder_id(note_id, field="note_id")
        if expected_note_version is not None:
            _validate_expected_version(expected_note_version)
        try:
            with self.db.transaction() as cursor, _mutation_savepoint(cursor):
                _require_active_membership_targets(
                    cursor, folder_ids=(folder_id,), note_ids=(note_id,)
                )
                row = _ensure_manual_membership(
                    cursor,
                    folder_id=folder_id,
                    note_id=note_id,
                    now=_utc_timestamp(),
                    expected_note_version=expected_note_version,
                )
                return _membership_from_row(row)
        except sqlite3.IntegrityError as exc:
            _raise_membership_integrity_error(exc)
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)

    def load_tree_search(
        self, *, note_ids: Iterable[str], folder_query: str = ""
    ) -> NoteFolderPage:
        """Load bounded content/path matches plus their complete breadcrumbs."""
        normalized_note_ids = _normalize_ids(note_ids, field="note_ids")
        if len(normalized_note_ids) > _TREE_SEARCH_NOTE_LIMIT:
            raise FolderValidationError("note_ids exceeds the allowed range.")
        normalized_folder_query = _normalize_folder_search_query(folder_query)
        if not normalized_note_ids and not normalized_folder_query:
            return NoteFolderPage(
                folders=(),
                memberships=(),
                notes=(),
                total_folders=0,
                total_notes=0,
                next_offset=None,
            )
        with self.db.transaction() as cursor:
            path_note_ids: tuple[str, ...] = ()
            if normalized_folder_query:
                path_note_rows = cursor.execute(
                    """
                    SELECT DISTINCT m.note_id
                    FROM note_folder_memberships AS m
                    JOIN note_folders AS f ON f.id = m.folder_id
                    JOIN notes AS n ON n.id = m.note_id
                    WHERE m.deleted = 0 AND f.deleted = 0 AND n.deleted = 0
                      AND instr(f.normalized_path, ?) > 0
                    ORDER BY m.note_id
                    LIMIT ?
                    """,
                    (normalized_folder_query, _TREE_SEARCH_NOTE_LIMIT + 1),
                ).fetchall()
                path_note_ids = tuple(str(row["note_id"]) for row in path_note_rows)
                if len(path_note_ids) > _TREE_SEARCH_NOTE_LIMIT:
                    raise FolderValidationError(
                        "Folder search has too many notes; narrow the search."
                    )
            selected_note_ids = tuple(
                sorted({*normalized_note_ids, *path_note_ids})
            )
            if len(selected_note_ids) > _TREE_SEARCH_NOTE_LIMIT:
                raise FolderValidationError(
                    "Folder search has too many notes; narrow the search."
                )
            if not selected_note_ids:
                return NoteFolderPage(
                    folders=(),
                    memberships=(),
                    notes=(),
                    total_folders=0,
                    total_notes=0,
                    next_offset=None,
                )

            membership_predicates: list[str] = []
            membership_parameters: list[str] = []
            if normalized_note_ids:
                content_placeholders = _placeholders(len(normalized_note_ids))
                membership_predicates.append(
                    f"m.note_id IN ({content_placeholders})"
                )
                membership_parameters.extend(normalized_note_ids)
            if normalized_folder_query:
                membership_predicates.append("instr(f.normalized_path, ?) > 0")
                membership_parameters.append(normalized_folder_query)
            membership_match = " OR ".join(membership_predicates)

            folder_rows = cursor.execute(
                f"""
                WITH RECURSIVE matched_folders(folder_id) AS (
                    SELECT DISTINCT m.folder_id
                    FROM note_folder_memberships AS m
                    JOIN note_folders AS f ON f.id = m.folder_id
                    WHERE m.deleted = 0 AND f.deleted = 0
                      AND ({membership_match})
                ),
                ancestors(folder_id) AS (
                    SELECT folder_id FROM matched_folders
                    UNION
                    SELECT f.parent_id
                    FROM note_folders AS f
                    JOIN ancestors ON ancestors.folder_id = f.id
                    WHERE f.parent_id IS NOT NULL AND f.deleted = 0
                )
                SELECT {_FOLDER_COLUMNS}
                FROM note_folders
                JOIN ancestors ON ancestors.folder_id = note_folders.id
                WHERE note_folders.deleted = 0
                ORDER BY note_folders.normalized_path, note_folders.id
                LIMIT ?
                """,
                (*membership_parameters, _TREE_SEARCH_FOLDER_LIMIT + 1),
            ).fetchall()
            membership_rows = cursor.execute(
                f"""
                SELECT m.{_MEMBERSHIP_COLUMNS.replace(', ', ', m.')}
                FROM note_folder_memberships AS m
                JOIN note_folders AS f ON f.id = m.folder_id
                WHERE m.deleted = 0 AND f.deleted = 0
                  AND ({membership_match})
                ORDER BY f.normalized_path, m.note_id, m.ownership, m.owner_id, m.id
                LIMIT ?
                """,
                (*membership_parameters, _TREE_SEARCH_MEMBERSHIP_LIMIT + 1),
            ).fetchall()
            selected_placeholders = _placeholders(len(selected_note_ids))
            note_rows = cursor.execute(
                f"""
                SELECT n.{_NOTE_COLUMNS.replace(', ', ', n.')},
                       NOT EXISTS (
                           SELECT 1
                           FROM note_folder_memberships AS m
                           JOIN note_folders AS f ON f.id = m.folder_id
                           WHERE m.note_id = n.id AND m.deleted = 0
                             AND f.deleted = 0 AND m.owner_active = 1
                       ) AS _unfiled
                FROM notes AS n
                WHERE n.deleted = 0 AND n.id IN ({selected_placeholders})
                ORDER BY n.title COLLATE NOCASE, n.id
                """,
                selected_note_ids,
            ).fetchall()
            if (
                len(folder_rows) > _TREE_SEARCH_FOLDER_LIMIT
                or len(membership_rows) > _TREE_SEARCH_MEMBERSHIP_LIMIT
            ):
                raise FolderValidationError(
                    "Folder search has too many placements; narrow the search."
                )
            managed_folder_rows = _load_managed_folder_rows(
                cursor, (str(row["id"]) for row in folder_rows)
            )

        managed_folder_ids = tuple(
            str(row["folder_id"])
            for row in managed_folder_rows
            if row["owner_active"] is not None
        )
        return NoteFolderPage(
            folders=tuple(_folder_from_row(row) for row in folder_rows),
            memberships=tuple(_membership_from_row(row) for row in membership_rows),
            notes=tuple(_note_from_row(row) for row in note_rows),
            total_folders=len(folder_rows),
            total_notes=len(note_rows),
            next_offset=None,
            total_memberships=len(membership_rows),
            managed_folder_ids=managed_folder_ids,
            inactive_managed_folder_ids=tuple(
                str(row["folder_id"])
                for row in managed_folder_rows
                if row["owner_active"] is not None and not bool(row["owner_active"])
            ),
            unfiled_note_ids=tuple(
                str(row["id"]) for row in note_rows if bool(row["_unfiled"])
            ),
        )

    def detach_manual(
        self, *, folder_id: str, note_id: str, expected_version: int
    ) -> bool:
        """Soft-delete one exact active manual placement optimistically."""
        _validate_folder_id(folder_id, field="folder_id")
        _validate_folder_id(note_id, field="note_id")
        _validate_expected_version(expected_version)
        try:
            with self.db.transaction() as cursor, _mutation_savepoint(cursor):
                row = cursor.execute(
                    "SELECT id, version FROM note_folder_memberships "
                    "WHERE folder_id = ? AND note_id = ? AND ownership = 'manual' "
                    "AND owner_id = '' AND deleted = 0",
                    (folder_id, note_id),
                ).fetchone()
                if row is None:
                    return False
                if int(row["version"]) != expected_version:
                    raise FolderConflictError(
                        "Membership version does not match expected_version."
                    )
                cursor.execute(
                    "UPDATE note_folder_memberships SET deleted = 1, "
                    "version = version + 1, modified_at = ? "
                    "WHERE id = ? AND version = ? AND deleted = 0 "
                    "AND ownership = 'manual' AND owner_id = ''",
                    (_utc_timestamp(), row["id"], expected_version),
                )
                _require_one_membership_update(cursor)
                return True
        except sqlite3.IntegrityError as exc:
            _raise_membership_integrity_error(exc)
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)

    def list_memberships(
        self, *, note_ids: Iterable[str], include_inactive: bool = False
    ) -> tuple[NoteFolderMembership, ...]:
        """Return active placements for a bounded, normalized note-ID set."""
        normalized_note_ids = _normalize_ids(note_ids, field="note_ids")
        if not isinstance(include_inactive, bool):
            raise FolderValidationError("include_inactive must be a boolean.")
        if not normalized_note_ids:
            return ()
        rows: list[sqlite3.Row] = []
        with self.db.transaction() as cursor:
            for chunk in _chunks(normalized_note_ids, _MEMBERSHIP_QUERY_CHUNK_SIZE):
                placeholders = _placeholders(len(chunk))
                inactive_clause = (
                    ""
                    if include_inactive
                    else " AND (ownership = 'manual' OR owner_active = 1)"
                )
                rows.extend(
                    cursor.execute(
                        f"SELECT {_MEMBERSHIP_COLUMNS} FROM note_folder_memberships "
                        f"WHERE deleted = 0 AND note_id IN ({placeholders})"
                        f"{inactive_clause}",
                        chunk,
                    ).fetchall()
                )
        rows.sort(
            key=lambda row: (
                str(row["note_id"]),
                str(row["folder_id"]),
                str(row["ownership"]),
                str(row["owner_id"]),
                str(row["id"]),
            )
        )
        return tuple(_membership_from_row(row) for row in rows)

    def get_exact_manual_membership(
        self,
        *,
        folder_id: str,
        note_id: str,
        include_deleted: bool = False,
    ) -> tuple[NoteFolderMembership, bool] | None:
        """Read one exact manual placement, optionally including its tombstone.

        Args:
            folder_id: Opaque identifier of the containing folder.
            note_id: Opaque identifier of the placed note.
            include_deleted: Whether a deleted placement may be returned.

        Returns:
            The matching placement and its deletion flag, or ``None`` when no
            eligible placement exists.

        Raises:
            FolderValidationError: If an identifier or ``include_deleted`` is
                invalid.
        """

        _validate_folder_id(folder_id, field="folder_id")
        _validate_folder_id(note_id, field="note_id")
        if not isinstance(include_deleted, bool):
            raise FolderValidationError("include_deleted must be a boolean.")
        deleted_clause = "" if include_deleted else " AND deleted = 0"
        with self.db.transaction() as cursor:
            row = cursor.execute(
                f"SELECT {_MEMBERSHIP_COLUMNS}, deleted "
                "FROM note_folder_memberships "
                "WHERE folder_id = ? AND note_id = ? "
                "AND ownership = 'manual' AND owner_id = ''"
                f"{deleted_clause} ORDER BY deleted, modified_at DESC, id DESC LIMIT 1",
                (folder_id, note_id),
            ).fetchone()
        if row is None:
            return None
        return _membership_from_row(row), bool(row["deleted"])

    def has_managed_folder_ownership(self, folder_id: str) -> bool:
        """Return whether an active managed placement owns this folder subtree.

        Args:
            folder_id: Opaque identifier of the folder to inspect.

        Returns:
            ``True`` when active managed ownership exists; otherwise ``False``.

        Raises:
            FolderValidationError: If ``folder_id`` is invalid.
        """

        _validate_folder_id(folder_id, field="folder_id")
        with self.db.transaction() as cursor:
            rows = _load_managed_folder_rows(cursor, (folder_id,))
            return bool(rows and rows[0]["owner_active"] is not None)

    def reconcile_managed(
        self, *, owner_id: str, desired: Iterable[tuple[str, str]]
    ) -> tuple[NoteFolderMembership, ...]:
        """Converge only one sync owner's managed placements."""
        _validate_owner_id(owner_id)
        desired_pairs = _normalize_desired_memberships(desired)
        desired_set = set(desired_pairs)
        try:
            with self.db.transaction() as cursor, _mutation_savepoint(cursor):
                _require_active_membership_targets(
                    cursor,
                    folder_ids=tuple(pair[0] for pair in desired_pairs),
                    note_ids=tuple(pair[1] for pair in desired_pairs),
                )
                rows = cursor.execute(
                    "SELECT id, folder_id, note_id, ownership, owner_id, "
                    "owner_active, version, deleted, modified_at "
                    "FROM note_folder_memberships WHERE ownership = 'managed' "
                    "AND owner_id = ? "
                    "ORDER BY folder_id, note_id, modified_at DESC, id DESC",
                    (owner_id,),
                ).fetchall()
                active_by_pair: dict[tuple[str, str], sqlite3.Row] = {}
                deleted_by_pair: dict[tuple[str, str], sqlite3.Row] = {}
                for row in rows:
                    pair = (str(row["folder_id"]), str(row["note_id"]))
                    if bool(row["deleted"]):
                        deleted_by_pair.setdefault(pair, row)
                    else:
                        active_by_pair[pair] = row

                now = _utc_timestamp()
                for pair in sorted(set(active_by_pair) - desired_set):
                    row = active_by_pair[pair]
                    cursor.execute(
                        "UPDATE note_folder_memberships SET deleted = 1, "
                        "version = version + 1, modified_at = ? "
                        "WHERE id = ? AND version = ? AND deleted = 0 "
                        "AND ownership = 'managed' AND owner_id = ?",
                        (now, row["id"], row["version"], owner_id),
                    )
                    _require_one_membership_update(cursor)

                for folder_id, note_id in desired_pairs:
                    pair = (folder_id, note_id)
                    active = active_by_pair.get(pair)
                    if active is not None:
                        if not bool(active["owner_active"]):
                            cursor.execute(
                                "UPDATE note_folder_memberships SET owner_active = 1, "
                                "version = version + 1, modified_at = ? "
                                "WHERE id = ? AND version = ? AND deleted = 0 "
                                "AND ownership = 'managed' AND owner_id = ? "
                                "AND owner_active = 0",
                                (now, active["id"], active["version"], owner_id),
                            )
                            _require_one_membership_update(cursor)
                        continue
                    deleted = deleted_by_pair.get(pair)
                    if deleted is not None:
                        cursor.execute(
                            "UPDATE note_folder_memberships SET deleted = 0, "
                            "owner_active = 1, version = version + 1, modified_at = ? "
                            "WHERE id = ? AND version = ? AND deleted = 1 "
                            "AND ownership = 'managed' AND owner_id = ?",
                            (now, deleted["id"], deleted["version"], owner_id),
                        )
                        _require_one_membership_update(cursor)
                    else:
                        _insert_membership(
                            cursor,
                            folder_id=folder_id,
                            note_id=note_id,
                            ownership="managed",
                            owner_id=owner_id,
                            now=now,
                        )

                result_rows = cursor.execute(
                    f"SELECT {_MEMBERSHIP_COLUMNS} FROM note_folder_memberships "
                    "WHERE ownership = 'managed' AND owner_id = ? "
                    "AND deleted = 0 AND owner_active = 1 "
                    "ORDER BY note_id, folder_id, ownership, owner_id, id",
                    (owner_id,),
                ).fetchall()
                return tuple(
                    _membership_from_row(row)
                    for row in result_rows
                    if (str(row["folder_id"]), str(row["note_id"])) in desired_set
                )
        except sqlite3.IntegrityError as exc:
            _raise_membership_integrity_error(exc)
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)

    def convert_owner_to_manual(self, *, owner_id: str) -> int:
        """Convert one owner's active managed placements to manual placements."""
        _validate_owner_id(owner_id)
        try:
            with self.db.transaction() as cursor, _mutation_savepoint(cursor):
                managed_rows = cursor.execute(
                    "SELECT id, folder_id, note_id, version "
                    "FROM note_folder_memberships WHERE ownership = 'managed' "
                    "AND owner_id = ? AND deleted = 0 "
                    "ORDER BY folder_id, note_id, id",
                    (owner_id,),
                ).fetchall()
                if not managed_rows:
                    return 0
                now = _utc_timestamp()
                for row in managed_rows:
                    _ensure_manual_membership(
                        cursor,
                        folder_id=str(row["folder_id"]),
                        note_id=str(row["note_id"]),
                        now=now,
                    )
                for row in managed_rows:
                    cursor.execute(
                        "UPDATE note_folder_memberships SET deleted = 1, "
                        "version = version + 1, modified_at = ? "
                        "WHERE id = ? AND version = ? AND deleted = 0 "
                        "AND ownership = 'managed' AND owner_id = ?",
                        (now, row["id"], row["version"], owner_id),
                    )
                    _require_one_membership_update(cursor)
                return len(managed_rows)
        except sqlite3.IntegrityError as exc:
            _raise_membership_integrity_error(exc)
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)

    def remove_owner_memberships(self, *, owner_id: str) -> int:
        """Soft-delete only one owner's active managed placements."""
        _validate_owner_id(owner_id)
        try:
            with self.db.transaction() as cursor, _mutation_savepoint(cursor):
                rows = cursor.execute(
                    "SELECT id, version FROM note_folder_memberships "
                    "WHERE ownership = 'managed' AND owner_id = ? AND deleted = 0 "
                    "ORDER BY id",
                    (owner_id,),
                ).fetchall()
                if not rows:
                    return 0
                now = _utc_timestamp()
                for row in rows:
                    cursor.execute(
                        "UPDATE note_folder_memberships SET deleted = 1, "
                        "version = version + 1, modified_at = ? "
                        "WHERE id = ? AND version = ? AND deleted = 0 "
                        "AND ownership = 'managed' AND owner_id = ?",
                        (now, row["id"], row["version"], owner_id),
                    )
                    _require_one_membership_update(cursor)
                return len(rows)
        except sqlite3.IntegrityError as exc:
            _raise_membership_integrity_error(exc)
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)

    def mark_unknown_owners_inactive(
        self, *, active_owner_ids: Iterable[str]
    ) -> int:
        """Converge restored managed-owner flags against the known owner set."""
        known_owners = set(_normalize_owner_ids(active_owner_ids))
        try:
            with self.db.transaction() as cursor, _mutation_savepoint(cursor):
                rows = cursor.execute(
                    "SELECT id, owner_id, owner_active, version "
                    "FROM note_folder_memberships WHERE ownership = 'managed' "
                    "AND deleted = 0 ORDER BY owner_id, id"
                ).fetchall()
                changes: list[tuple[sqlite3.Row, bool]] = []
                for row in rows:
                    owner_active = str(row["owner_id"]) in known_owners
                    if bool(row["owner_active"]) != owner_active:
                        changes.append((row, owner_active))
                if not changes:
                    return 0
                now = _utc_timestamp()
                for row, owner_active in changes:
                    cursor.execute(
                        "UPDATE note_folder_memberships SET owner_active = ?, "
                        "version = version + 1, modified_at = ? "
                        "WHERE id = ? AND version = ? AND deleted = 0 "
                        "AND ownership = 'managed' AND owner_id = ? "
                        "AND owner_active = ?",
                        (
                            int(owner_active),
                            now,
                            row["id"],
                            row["version"],
                            row["owner_id"],
                            row["owner_active"],
                        ),
                    )
                    _require_one_membership_update(cursor)
                return len(changes)
        except sqlite3.IntegrityError as exc:
            _raise_membership_integrity_error(exc)
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)

    def list_restore_reviews(
        self,
    ) -> tuple[RestoredManagedMembershipReview, ...]:
        """Group inactive active managed placements by restored owner."""
        rows = self.db.get_connection().execute(
            "SELECT id, owner_id, note_id, folder_id "
            "FROM note_folder_memberships WHERE ownership = 'managed' "
            "AND deleted = 0 AND owner_active = 0 "
            "ORDER BY owner_id, id"
        ).fetchall()
        grouped: dict[str, dict[str, set[str]]] = {}
        for row in rows:
            owner = str(row["owner_id"])
            group = grouped.setdefault(
                owner, {"memberships": set(), "notes": set(), "folders": set()}
            )
            group["memberships"].add(str(row["id"]))
            group["notes"].add(str(row["note_id"]))
            group["folders"].add(str(row["folder_id"]))
        return tuple(
            RestoredManagedMembershipReview(
                owner_id=owner,
                membership_ids=tuple(sorted(group["memberships"])),
                note_count=len(group["notes"]),
                folder_count=len(group["folders"]),
            )
            for owner, group in sorted(grouped.items())
        )

    def rename_folder(
        self, folder_id: str, *, name: str, expected_version: int
    ) -> FolderMutationResult:
        """Rename an active folder and rewrite every active descendant path."""
        _validate_folder_id(folder_id, field="folder_id")
        _validate_expected_version(expected_version)
        normalized = normalize_folder_name(name)
        try:
            with self.db.transaction() as cursor, _mutation_savepoint(cursor):
                target = _require_target(
                    cursor,
                    folder_id=folder_id,
                    expected_version=expected_version,
                    deleted=False,
                )
                _require_manual_folder_subtree(cursor, folder_id)
                subtree = _load_subtree(cursor, target, deleted=False)
                parent = _load_destination_parent(
                    cursor, parent_id=target["parent_id"]
                )
                parent_path = str(parent["path"]) if parent is not None else ""
                parent_normalized_path = (
                    str(parent["normalized_path"]) if parent is not None else ""
                )
                target_path = _join_display_folder_path(
                    parent_path, normalized.display
                )
                target_normalized_path = join_normalized_folder_path(
                    parent_normalized_path, normalized.key
                )
                rewritten = _rewrite_subtree_paths(
                    subtree,
                    target_path=target_path,
                    target_normalized_path=target_normalized_path,
                )
                _preflight_active_paths(cursor, rewritten)
                now = _utc_timestamp()
                for row, path, normalized_path in rewritten:
                    if row["id"] == folder_id:
                        cursor.execute(
                            "UPDATE note_folders SET name = ?, normalized_name = ?, "
                            "path = ?, normalized_path = ?, version = version + 1, "
                            "modified_at = ? WHERE id = ? AND version = ? "
                            "AND deleted = 0",
                            (
                                normalized.display,
                                normalized.key,
                                path,
                                normalized_path,
                                now,
                                row["id"],
                                row["version"],
                            ),
                        )
                    else:
                        cursor.execute(
                            "UPDATE note_folders SET path = ?, normalized_path = ?, "
                            "version = version + 1, modified_at = ? "
                            "WHERE id = ? AND version = ? AND deleted = 0",
                            (path, normalized_path, now, row["id"], row["version"]),
                        )
                    _require_one_folder_update(cursor)
                return _mutation_result(cursor, folder_id, subtree)
        except sqlite3.IntegrityError as exc:
            _raise_mutation_integrity_error(exc)
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)

    def move_folder(
        self,
        folder_id: str,
        *,
        parent_id: str | None,
        expected_version: int,
    ) -> FolderMutationResult:
        """Move an active folder and its active subtree beneath a new parent."""
        _validate_folder_id(folder_id, field="folder_id")
        if parent_id is not None:
            _validate_folder_id(parent_id, field="parent_id")
        _validate_expected_version(expected_version)
        try:
            with self.db.transaction() as cursor, _mutation_savepoint(cursor):
                target = _require_target(
                    cursor,
                    folder_id=folder_id,
                    expected_version=expected_version,
                    deleted=False,
                )
                _require_manual_folder_subtree(cursor, folder_id)
                subtree = _load_subtree(cursor, target, deleted=False)
                parent = _load_destination_parent(cursor, parent_id=parent_id)
                if parent_id is not None:
                    _require_manual_folder_subtree(cursor, parent_id)
                if parent_id is not None and _has_ancestor(
                    cursor, folder_id=parent_id, ancestor_id=folder_id
                ):
                    raise FolderValidationError(
                        "A folder cannot be moved beneath itself or its descendant."
                    )
                parent_path = str(parent["path"]) if parent is not None else ""
                parent_normalized_path = (
                    str(parent["normalized_path"]) if parent is not None else ""
                )
                target_path = _join_display_folder_path(
                    parent_path, str(target["name"])
                )
                target_normalized_path = join_normalized_folder_path(
                    parent_normalized_path, str(target["normalized_name"])
                )
                rewritten = _rewrite_subtree_paths(
                    subtree,
                    target_path=target_path,
                    target_normalized_path=target_normalized_path,
                )
                _preflight_active_paths(cursor, rewritten)
                now = _utc_timestamp()
                for row, path, normalized_path in rewritten:
                    if row["id"] == folder_id:
                        cursor.execute(
                            "UPDATE note_folders SET parent_id = ?, path = ?, "
                            "normalized_path = ?, version = version + 1, "
                            "modified_at = ? WHERE id = ? AND version = ? "
                            "AND deleted = 0",
                            (
                                parent_id,
                                path,
                                normalized_path,
                                now,
                                row["id"],
                                row["version"],
                            ),
                        )
                    else:
                        cursor.execute(
                            "UPDATE note_folders SET path = ?, normalized_path = ?, "
                            "version = version + 1, modified_at = ? "
                            "WHERE id = ? AND version = ? AND deleted = 0",
                            (path, normalized_path, now, row["id"], row["version"]),
                        )
                    _require_one_folder_update(cursor)
                return _mutation_result(cursor, folder_id, subtree)
        except sqlite3.IntegrityError as exc:
            _raise_mutation_integrity_error(exc)
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)

    def soft_delete_folder(
        self, folder_id: str, *, expected_version: int
    ) -> FolderMutationResult:
        """Soft-delete an active folder and its complete active subtree."""
        _validate_folder_id(folder_id, field="folder_id")
        _validate_expected_version(expected_version)
        try:
            with self.db.transaction() as cursor, _mutation_savepoint(cursor):
                target = _require_target(
                    cursor,
                    folder_id=folder_id,
                    expected_version=expected_version,
                    deleted=False,
                )
                _require_manual_folder_subtree(cursor, folder_id)
                subtree = _load_subtree(cursor, target, deleted=False)
                now = _unique_deleted_folder_timestamp(cursor)
                for row in subtree:
                    cursor.execute(
                        "UPDATE note_folders SET deleted = 1, version = version + 1, "
                        "modified_at = ? WHERE id = ? AND version = ? "
                        "AND deleted = 0",
                        (now, row["id"], row["version"]),
                    )
                    _require_one_folder_update(cursor)
                return _mutation_result(cursor, folder_id, subtree)
        except sqlite3.IntegrityError as exc:
            _raise_mutation_integrity_error(exc)
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)

    def restore_folder(
        self, folder_id: str, *, expected_version: int
    ) -> FolderMutationResult:
        """Restore a deleted stored-path subtree after atomic validation."""
        _validate_folder_id(folder_id, field="folder_id")
        _validate_expected_version(expected_version)
        try:
            with self.db.transaction() as cursor, _mutation_savepoint(cursor):
                target = _require_target(
                    cursor,
                    folder_id=folder_id,
                    expected_version=expected_version,
                    deleted=True,
                )
                subtree = _load_restore_cohort(cursor, target)
                subtree_ids = {str(row["id"]) for row in subtree}
                target_parent_id = (
                    str(target["parent_id"])
                    if target["parent_id"] is not None
                    else None
                )
                external_parent_ids = {
                    str(row["parent_id"])
                    for row in subtree
                    if row["parent_id"] is not None
                    and str(row["parent_id"]) not in subtree_ids
                }
                if external_parent_ids:
                    placeholders = _placeholders(len(external_parent_ids))
                    active_parent_rows = cursor.execute(
                        f"SELECT id FROM note_folders WHERE deleted = 0 "
                        f"AND id IN ({placeholders})",
                        tuple(sorted(external_parent_ids)),
                    ).fetchall()
                    active_parent_ids = {
                        str(row["id"]) for row in active_parent_rows
                    }
                    if active_parent_ids != external_parent_ids:
                        raise FolderValidationError(
                            "A restored folder's external parent is missing "
                            "or inactive."
                        )

                if target_parent_id is None:
                    parent_path = ""
                    parent_normalized_path = ""
                elif target_parent_id in subtree_ids:
                    parent_path = str(target["path"]).rsplit("/", 1)[0]
                    parent_normalized_path = str(target["normalized_path"]).rsplit(
                        "/", 1
                    )[0]
                else:
                    external_parent = _load_destination_parent(
                        cursor, parent_id=target_parent_id
                    )
                    if external_parent is None:  # pragma: no cover - ID is non-null
                        raise FolderValidationError(
                            "A restored folder's external parent is unavailable."
                        )
                    parent_path = str(external_parent["path"])
                    parent_normalized_path = str(
                        external_parent["normalized_path"]
                    )
                target_name = normalize_folder_name(str(target["name"]))
                if (
                    target_name.display != str(target["name"])
                    or target_name.key != str(target["normalized_name"])
                ):
                    raise FolderValidationError(
                        "Restored folder name normalization is inconsistent."
                    )
                target_path = _join_display_folder_path(
                    parent_path, target_name.display
                )
                target_normalized_path = join_normalized_folder_path(
                    parent_normalized_path, target_name.key
                )
                rewritten = _rewrite_subtree_paths(
                    subtree,
                    target_path=target_path,
                    target_normalized_path=target_normalized_path,
                )
                _preflight_active_paths(cursor, rewritten)
                now = _utc_timestamp()
                for row, path, normalized_path in rewritten:
                    cursor.execute(
                        "UPDATE note_folders SET path = ?, normalized_path = ?, "
                        "deleted = 0, version = version + 1, modified_at = ? "
                        "WHERE id = ? AND version = ? AND deleted = 1",
                        (
                            path,
                            normalized_path,
                            now,
                            row["id"],
                            row["version"],
                        ),
                    )
                    _require_one_folder_update(cursor)
                return _mutation_result(cursor, folder_id, subtree)
        except sqlite3.IntegrityError as exc:
            _raise_mutation_integrity_error(exc)
        except sqlite3.OperationalError as exc:
            _raise_mutation_operational_error(exc)
        except CharactersRAGDBError as exc:
            _raise_wrapped_repository_error(exc)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _unique_deleted_folder_timestamp(cursor: sqlite3.Cursor) -> str:
    """Return a millisecond UTC deletion marker unused by folder tombstones."""
    candidate = _utc_timestamp()
    while cursor.execute(
        "SELECT 1 FROM note_folders WHERE deleted = 1 AND modified_at = ? LIMIT 1",
        (candidate,),
    ).fetchone() is not None:
        normalized_candidate = (
            f"{candidate[:-1]}+00:00" if candidate.endswith("Z") else candidate
        )
        parsed = datetime.fromisoformat(normalized_candidate)
        candidate = (parsed + timedelta(milliseconds=1)).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
    return candidate


@contextmanager
def _mutation_savepoint(cursor: sqlite3.Cursor) -> Iterator[None]:
    """Give one multi-row mutation an atomic boundary inside any transaction."""
    name = f"note_folder_mutation_{uuid.uuid4().hex}"
    cursor.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        cursor.execute(f"ROLLBACK TO SAVEPOINT {name}")
        cursor.execute(f"RELEASE SAVEPOINT {name}")
        raise
    else:
        cursor.execute(f"RELEASE SAVEPOINT {name}")


def _require_target(
    cursor: sqlite3.Cursor,
    *,
    folder_id: str,
    expected_version: int,
    deleted: bool,
) -> sqlite3.Row:
    row = cursor.execute(
        f"SELECT {_FOLDER_COLUMNS} FROM note_folders WHERE id = ?", (folder_id,)
    ).fetchone()
    if row is None or bool(row["deleted"]) is not deleted:
        state = "deleted" if deleted else "active"
        raise FolderValidationError(f"Target folder is not {state}.")
    if int(row["version"]) != expected_version:
        raise FolderConflictError("Folder version does not match expected_version.")
    return row


def _load_destination_parent(
    cursor: sqlite3.Cursor, *, parent_id: object
) -> sqlite3.Row | None:
    if parent_id is None:
        return None
    _validate_folder_id(parent_id, field="parent_id")
    row = cursor.execute(
        f"SELECT {_FOLDER_COLUMNS} FROM note_folders "
        "WHERE id = ? AND deleted = 0",
        (parent_id,),
    ).fetchone()
    if row is None:
        raise FolderValidationError(
            "Destination parent does not exist or is inactive."
        )
    _validate_absolute_display_path(str(row["path"]))
    _validate_absolute_normalized_path(str(row["normalized_path"]))
    return row


def _has_ancestor(
    cursor: sqlite3.Cursor, *, folder_id: str, ancestor_id: str
) -> bool:
    """Return whether an active folder's parent-ID chain reaches an ancestor."""
    row = cursor.execute(
        """
        WITH RECURSIVE ancestors(id, parent_id) AS (
            SELECT id, parent_id
              FROM note_folders
             WHERE id = ? AND deleted = 0
            UNION
            SELECT parent.id, parent.parent_id
              FROM note_folders AS parent
              JOIN ancestors AS child ON parent.id = child.parent_id
             WHERE parent.deleted = 0
        )
        SELECT 1 FROM ancestors WHERE id = ? LIMIT 1
        """,
        (folder_id, ancestor_id),
    ).fetchone()
    return row is not None


def _load_note_tree_path(
    cursor: sqlite3.Cursor, *, folder_id: str, page_size: int
) -> tuple[NoteTreePathStep, ...]:
    """Load one active root-to-folder path with parent-relative page offsets."""
    rows = cursor.execute(
        """
        WITH RECURSIVE path(id, parent_id, normalized_name, depth) AS (
            SELECT id, parent_id, normalized_name, 0
            FROM note_folders
            WHERE id = ? AND deleted = 0
            UNION ALL
            SELECT parent.id, parent.parent_id, parent.normalized_name, child.depth + 1
            FROM note_folders AS parent
            JOIN path AS child ON parent.id = child.parent_id
            WHERE parent.deleted = 0
        )
        SELECT path.id, path.parent_id, path.depth,
               (
                   SELECT COUNT(*)
                   FROM note_folders AS sibling
                   WHERE sibling.deleted = 0
                     AND sibling.parent_id IS path.parent_id
                     AND (
                         sibling.normalized_name < path.normalized_name
                         OR (
                             sibling.normalized_name = path.normalized_name
                             AND sibling.id < path.id
                         )
                     )
               ) AS parent_rank
        FROM path
        ORDER BY path.depth DESC
        """,
        (folder_id,),
    ).fetchall()
    if not rows:
        return ()
    if rows[0]["parent_id"] is not None:
        raise FolderValidationError("Folder path does not reach an active root.")
    return tuple(
        NoteTreePathStep(
            folder_id=str(row["id"]),
            parent_id=(str(row["parent_id"]) if row["parent_id"] is not None else None),
            containing_offset=(int(row["parent_rank"]) // page_size) * page_size,
        )
        for row in rows
    )


def _load_subtree(
    cursor: sqlite3.Cursor, target: sqlite3.Row, *, deleted: bool
) -> tuple[sqlite3.Row, ...]:
    root_path = str(target["path"])
    root_normalized_path = str(target["normalized_path"])
    _validate_absolute_display_path(root_path)
    _validate_absolute_normalized_path(root_normalized_path)
    rows = cursor.execute(
        f"SELECT {_FOLDER_COLUMNS} FROM note_folders WHERE deleted = ? AND ("
        "normalized_path = ? OR ("
        "length(normalized_path) > length(?) AND "
        "substr(normalized_path, 1, length(?) + 1) = ? || '/')) "
        "ORDER BY length(normalized_path), normalized_path, id",
        (
            int(deleted),
            root_normalized_path,
            root_normalized_path,
            root_normalized_path,
            root_normalized_path,
        ),
    ).fetchall()
    if not rows or str(rows[0]["id"]) != str(target["id"]):
        raise FolderValidationError("Folder subtree is inconsistent.")
    return tuple(rows)


def _load_restore_cohort(
    cursor: sqlite3.Cursor, target: sqlite3.Row
) -> tuple[sqlite3.Row, ...]:
    """Load only deleted descendants changed by the target's delete operation."""
    marker = str(target["modified_at"])
    root_normalized_path = str(target["normalized_path"])
    _validate_absolute_display_path(str(target["path"]))
    _validate_absolute_normalized_path(root_normalized_path)
    rows = cursor.execute(
        "WITH RECURSIVE cohort AS ("
        "SELECT * FROM note_folders "
        "WHERE id = ? AND deleted = 1 AND modified_at = ? "
        "UNION "
        "SELECT child.* FROM note_folders AS child "
        "JOIN cohort AS parent ON child.parent_id = parent.id "
        "WHERE child.deleted = 1 AND child.modified_at = ? "
        "AND length(child.normalized_path) > length(?) "
        "AND substr(child.normalized_path, 1, length(?) + 1) = ? || '/'"
        ") "
        f"SELECT {_FOLDER_COLUMNS} FROM cohort "
        "ORDER BY length(normalized_path), normalized_path, id",
        (
            target["id"],
            marker,
            marker,
            root_normalized_path,
            root_normalized_path,
            root_normalized_path,
        ),
    ).fetchall()
    if not rows or str(rows[0]["id"]) != str(target["id"]):
        raise FolderValidationError("Folder restore cohort is inconsistent.")
    return tuple(rows)


def _rewrite_subtree_paths(
    subtree: Sequence[sqlite3.Row],
    *,
    target_path: str,
    target_normalized_path: str,
) -> list[tuple[sqlite3.Row, str, str]]:
    old_path = str(subtree[0]["path"])
    old_normalized_path = str(subtree[0]["normalized_path"])
    rewritten = [
        (
            row,
            _replace_subtree_prefix(str(row["path"]), old_path, target_path),
            _replace_subtree_prefix(
                str(row["normalized_path"]),
                old_normalized_path,
                target_normalized_path,
            ),
        )
        for row in subtree
    ]
    _validate_rewritten_paths(rewritten)
    return rewritten


def _replace_subtree_prefix(value: str, old_prefix: str, new_prefix: str) -> str:
    if value == old_prefix:
        return new_prefix
    boundary = f"{old_prefix}/"
    if not value.startswith(boundary):
        raise FolderValidationError(
            "A descendant path does not share the target folder prefix."
        )
    return f"{new_prefix}{value[len(old_prefix):]}"


def _validate_rewritten_paths(
    rewritten: Sequence[tuple[sqlite3.Row, str, str]],
) -> None:
    normalized_paths: set[str] = set()
    for _row, path, normalized_path in rewritten:
        _validate_absolute_display_path(path)
        _validate_absolute_normalized_path(normalized_path)
        if normalized_path in normalized_paths:
            raise FolderCollisionError(
                "The folder subtree contains duplicate normalized paths."
            )
        normalized_paths.add(normalized_path)


def _preflight_active_paths(
    cursor: sqlite3.Cursor,
    rewritten: Sequence[tuple[sqlite3.Row, str, str]],
) -> None:
    _validate_rewritten_paths(rewritten)
    subtree_ids = {str(row["id"]) for row, _path, _normalized in rewritten}
    desired_paths = tuple(
        sorted({normalized for _row, _path, normalized in rewritten})
    )
    for start in range(0, len(desired_paths), _COLLISION_PREFLIGHT_CHUNK_SIZE):
        chunk = desired_paths[start : start + _COLLISION_PREFLIGHT_CHUNK_SIZE]
        placeholders = _placeholders(len(chunk))
        active_rows = cursor.execute(
            "SELECT id, normalized_path FROM note_folders WHERE deleted = 0 "
            f"AND normalized_path IN ({placeholders})",
            chunk,
        ).fetchall()
        if any(str(active["id"]) not in subtree_ids for active in active_rows):
            raise FolderCollisionError(
                "An active folder already uses a resulting normalized path."
            )


def _mutation_result(
    cursor: sqlite3.Cursor, folder_id: str, subtree: Sequence[sqlite3.Row]
) -> FolderMutationResult:
    target = cursor.execute(
        f"SELECT {_FOLDER_COLUMNS} FROM note_folders WHERE id = ?", (folder_id,)
    ).fetchone()
    if target is None:  # pragma: no cover - target remains stable in the transaction
        raise FolderValidationError("Mutated folder could not be read.")
    return FolderMutationResult(
        folder=_folder_from_row(target),
        affected_folder_ids=tuple(str(row["id"]) for row in subtree),
    )


def _require_one_folder_update(cursor: sqlite3.Cursor) -> None:
    """Reject a mutation when its optimistic row snapshot is no longer current."""
    if cursor.rowcount != 1:
        raise FolderConflictError("Folder changed during mutation.")


def _require_one_membership_update(cursor: sqlite3.Cursor) -> None:
    """Reject a membership update whose optimistic snapshot became stale."""
    if cursor.rowcount != 1:
        raise FolderConflictError("Membership changed during mutation.")


def _insert_membership(
    cursor: sqlite3.Cursor,
    *,
    folder_id: str,
    note_id: str,
    ownership: str,
    owner_id: str,
    now: str,
    expected_note_version: int | None = None,
) -> sqlite3.Row:
    membership_id = ""
    for attempt in range(_MEMBERSHIP_ID_INSERT_ATTEMPTS):
        membership_id = str(uuid.uuid4())
        try:
            if expected_note_version is None:
                cursor.execute(
                    "INSERT INTO note_folder_memberships("
                    "id, folder_id, note_id, ownership, owner_id, owner_active, "
                    "version, deleted, created_at, modified_at"
                    ") VALUES (?, ?, ?, ?, ?, 1, 1, 0, ?, ?)",
                    (
                        membership_id,
                        folder_id,
                        note_id,
                        ownership,
                        owner_id,
                        now,
                        now,
                    ),
                )
            else:
                cursor.execute(
                    "INSERT INTO note_folder_memberships("
                    "id, folder_id, note_id, ownership, owner_id, owner_active, "
                    "version, deleted, created_at, modified_at"
                    ") SELECT ?, ?, ?, ?, ?, 1, 1, 0, ?, ? "
                    "WHERE EXISTS (SELECT 1 FROM notes "
                    "WHERE id = ? AND deleted = 0 AND version = ?)",
                    (
                        membership_id,
                        folder_id,
                        note_id,
                        ownership,
                        owner_id,
                        now,
                        now,
                        note_id,
                        expected_note_version,
                    ),
                )
                _require_one_membership_update(cursor)
            break
        except sqlite3.IntegrityError as exc:
            if not _is_membership_id_collision(exc):
                raise
            if attempt == _MEMBERSHIP_ID_INSERT_ATTEMPTS - 1:
                raise FolderValidationError(
                    "Membership ID allocation failed."
                ) from exc
    row = cursor.execute(
        f"SELECT {_MEMBERSHIP_COLUMNS} FROM note_folder_memberships WHERE id = ?",
        (membership_id,),
    ).fetchone()
    if row is None:  # pragma: no cover - SQLite guarantees the inserted row
        raise FolderValidationError("Created membership could not be read.")
    return row


def _ensure_manual_membership(
    cursor: sqlite3.Cursor,
    *,
    folder_id: str,
    note_id: str,
    now: str,
    expected_note_version: int | None = None,
) -> sqlite3.Row:
    active = cursor.execute(
        f"SELECT {_MEMBERSHIP_COLUMNS} FROM note_folder_memberships "
        "WHERE folder_id = ? AND note_id = ? AND ownership = 'manual' "
        "AND owner_id = '' AND deleted = 0",
        (folder_id, note_id),
    ).fetchone()
    if active is not None:
        if expected_note_version is not None:
            cursor.execute(
                "UPDATE note_folder_memberships SET owner_active = owner_active "
                "WHERE id = ? AND deleted = 0 AND ownership = 'manual' "
                "AND owner_id = '' AND EXISTS (SELECT 1 FROM notes "
                "WHERE id = ? AND deleted = 0 AND version = ?)",
                (active["id"], note_id, expected_note_version),
            )
            _require_one_membership_update(cursor)
        return active
    deleted = cursor.execute(
        "SELECT id, version FROM note_folder_memberships "
        "WHERE folder_id = ? AND note_id = ? AND ownership = 'manual' "
        "AND owner_id = '' AND deleted = 1 "
        "ORDER BY modified_at DESC, id DESC LIMIT 1",
        (folder_id, note_id),
    ).fetchone()
    if deleted is None:
        return _insert_membership(
            cursor,
            folder_id=folder_id,
            note_id=note_id,
            ownership="manual",
            owner_id="",
            now=now,
            expected_note_version=expected_note_version,
        )
    if expected_note_version is None:
        cursor.execute(
            "UPDATE note_folder_memberships SET deleted = 0, owner_active = 1, "
            "version = version + 1, modified_at = ? "
            "WHERE id = ? AND version = ? AND deleted = 1 "
            "AND ownership = 'manual' AND owner_id = ''",
            (now, deleted["id"], deleted["version"]),
        )
    else:
        cursor.execute(
            "UPDATE note_folder_memberships SET deleted = 0, owner_active = 1, "
            "version = version + 1, modified_at = ? "
            "WHERE id = ? AND version = ? AND deleted = 1 "
            "AND ownership = 'manual' AND owner_id = '' "
            "AND EXISTS (SELECT 1 FROM notes "
            "WHERE id = ? AND deleted = 0 AND version = ?)",
            (
                now,
                deleted["id"],
                deleted["version"],
                note_id,
                expected_note_version,
            ),
        )
    _require_one_membership_update(cursor)
    row = cursor.execute(
        f"SELECT {_MEMBERSHIP_COLUMNS} FROM note_folder_memberships WHERE id = ?",
        (deleted["id"],),
    ).fetchone()
    if row is None:  # pragma: no cover - the optimistic update preserves the row
        raise FolderValidationError("Revived membership could not be read.")
    return row


def _require_active_membership_targets(
    cursor: sqlite3.Cursor,
    *,
    folder_ids: Sequence[str],
    note_ids: Sequence[str],
) -> None:
    _require_active_ids(
        cursor,
        table="note_folders",
        ids=tuple(sorted(set(folder_ids))),
        field="folder",
    )
    _require_active_ids(
        cursor,
        table="notes",
        ids=tuple(sorted(set(note_ids))),
        field="note",
    )


def _require_active_ids(
    cursor: sqlite3.Cursor,
    *,
    table: str,
    ids: Sequence[str],
    field: str,
) -> None:
    found: set[str] = set()
    for chunk in _chunks(ids, _MEMBERSHIP_QUERY_CHUNK_SIZE):
        placeholders = _placeholders(len(chunk))
        rows = cursor.execute(
            f"SELECT id FROM {table} WHERE deleted = 0 AND id IN ({placeholders})",
            chunk,
        ).fetchall()
        found.update(str(row["id"]) for row in rows)
    if len(found) != len(ids):
        raise FolderValidationError(
            f"Every desired {field} must exist and be active."
        )


def _raise_mutation_integrity_error(exc: sqlite3.IntegrityError) -> NoReturn:
    if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
        raise FolderCollisionError(
            "An active folder already uses a resulting normalized path."
        ) from exc
    raise FolderValidationError("Folder mutation violated stored constraints.") from exc


def _raise_membership_integrity_error(exc: sqlite3.IntegrityError) -> NoReturn:
    if _is_active_membership_owner_collision(exc):
        raise FolderConflictError("Membership changed during mutation.") from exc
    raise FolderValidationError(
        "Membership mutation violated stored constraints."
    ) from exc


def _is_membership_id_collision(exc: sqlite3.IntegrityError) -> bool:
    return (
        getattr(exc, "sqlite_errorcode", None)
        == sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY
        and "note_folder_memberships.id" in str(exc)
    )


def _is_active_membership_owner_collision(exc: sqlite3.IntegrityError) -> bool:
    if getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_CONSTRAINT_UNIQUE:
        return False
    message = str(exc)
    return all(
        column in message
        for column in (
            "note_folder_memberships.folder_id",
            "note_folder_memberships.note_id",
            "note_folder_memberships.ownership",
            "note_folder_memberships.owner_id",
        )
    )


def _raise_mutation_operational_error(exc: sqlite3.OperationalError) -> NoReturn:
    """Translate SQLite writer/snapshot contention into a stable domain conflict."""
    if _is_sqlite_contention(exc):
        raise FolderConflictError("Folder changed during mutation.") from exc
    raise exc


def _is_sqlite_contention(exc: sqlite3.OperationalError) -> bool:
    error_code = getattr(exc, "sqlite_errorcode", None)
    primary_code = error_code & 0xFF if isinstance(error_code, int) else None
    return primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}


def _raise_wrapped_repository_error(exc: CharactersRAGDBError) -> NoReturn:
    """Translate wrapped commit contention while preserving other DB failures."""
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, sqlite3.OperationalError) and _is_sqlite_contention(
            current
        ):
            raise FolderConflictError("Folder changed during mutation.") from current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    raise exc


def _require_manual_folder_subtree(
    cursor: sqlite3.Cursor, folder_id: str
) -> None:
    """Reject direct folder mutations that would alter a sync-owned subtree."""
    managed = cursor.execute(
        """
        WITH RECURSIVE subtree(folder_id) AS (
            SELECT id FROM note_folders WHERE id = ? AND deleted = 0
            UNION ALL
            SELECT child.id
            FROM note_folders AS child
            JOIN subtree AS parent ON child.parent_id = parent.folder_id
            WHERE child.deleted = 0
        )
        SELECT 1
        FROM note_folder_memberships AS membership
        JOIN subtree ON subtree.folder_id = membership.folder_id
        WHERE membership.deleted = 0 AND membership.ownership = 'managed'
        LIMIT 1
        """,
        (folder_id,),
    ).fetchone()
    if managed is not None:
        raise FolderCapabilityError(
            reason_code="sync_managed_folder",
            user_message=(
                "This folder is managed by sync; change its sync root instead."
            ),
        )


def _load_managed_folder_rows(
    cursor: sqlite3.Cursor, folder_ids: Iterable[str]
) -> Sequence[sqlite3.Row]:
    """Return authoritative managed state for each requested active subtree root."""
    normalized_folder_ids = _normalize_ids(folder_ids, field="folder_ids")
    if not normalized_folder_ids:
        return ()
    requested_values = ", ".join(
        f"(?, {ordinal})" for ordinal in range(len(normalized_folder_ids))
    )
    return cursor.execute(
        f"""
        WITH RECURSIVE requested_roots(root_id, ordinal) AS (
            VALUES {requested_values}
        ),
        subtree(root_id, folder_id) AS (
            SELECT requested.root_id, root.id
            FROM requested_roots AS requested
            JOIN note_folders AS root ON root.id = requested.root_id
            WHERE root.deleted = 0
            UNION ALL
            SELECT subtree.root_id, descendant.id
            FROM subtree
            JOIN note_folders AS descendant
                INDEXED BY idx_note_folders_active_parent
                ON descendant.parent_id = subtree.folder_id
            WHERE descendant.deleted = 0
        )
        SELECT requested.root_id AS folder_id,
               MIN(membership.owner_active) AS owner_active
        FROM requested_roots AS requested
        LEFT JOIN subtree ON subtree.root_id = requested.root_id
        LEFT JOIN note_folder_memberships AS membership
            INDEXED BY idx_note_folder_memberships_active_folder
            ON membership.folder_id = subtree.folder_id
           AND membership.deleted = 0
           AND membership.ownership = 'managed'
        GROUP BY requested.ordinal, requested.root_id
        ORDER BY requested.ordinal
        """,
        normalized_folder_ids,
    ).fetchall()


def _normalize_folder_search_query(query: str) -> str:
    """Normalize a bounded user breadcrumb query for stored path matching."""
    if not isinstance(query, str):
        raise FolderValidationError("folder_query must be text.")
    display = query.strip()
    if not display:
        return ""
    if len(display) > 200 or "\x00" in display:
        raise FolderValidationError("folder_query exceeds the allowed range.")
    normalized = unicodedata.normalize("NFKC", display).casefold()
    return "/".join(part.strip() for part in normalized.split("/"))


def _folder_from_row(row: sqlite3.Row) -> NoteFolder:
    return NoteFolder(
        folder_id=str(row["id"]),
        parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
        name=str(row["name"]),
        path=str(row["path"]),
        normalized_path=str(row["normalized_path"]),
        version=int(row["version"]),
        deleted=bool(row["deleted"]),
    )


def _membership_from_row(row: sqlite3.Row) -> NoteFolderMembership:
    return NoteFolderMembership(
        membership_id=str(row["id"]),
        folder_id=str(row["folder_id"]),
        note_id=str(row["note_id"]),
        ownership=row["ownership"],
        owner_id=str(row["owner_id"]),
        owner_active=bool(row["owner_active"]),
        version=int(row["version"]),
    )


def _note_from_row(row: sqlite3.Row) -> dict[str, object]:
    return {column: row[column] for column in _NOTE_COLUMNS.split(", ")}


def _previous_page_offset(offset: int, limit: int, total: int) -> int | None:
    if offset == 0:
        return None
    return min(max(0, offset - limit), max(0, total - limit))


def _join_display_folder_path(parent_path: str, child_name: str) -> str:
    if not isinstance(parent_path, str):
        raise FolderValidationError("Folder display paths must be text.")
    parent = parent_path.rstrip("/")
    if parent:
        if not parent.startswith("/"):
            raise FolderValidationError("Parent display path must be absolute.")
        components = parent[1:].split("/")
        if any(
            not component
            or component in {".", ".."}
            or "\\" in component
            or "\x00" in component
            for component in components
        ):
            raise FolderValidationError("Parent display path is invalid.")
    elif parent_path not in {"", "/"}:
        raise FolderValidationError("Parent display path is invalid.")
    return f"{parent}/{child_name}" if parent else f"/{child_name}"


def _validate_absolute_display_path(path: str) -> None:
    if not isinstance(path, str) or not path.startswith("/") or path == "/":
        raise FolderValidationError("Folder display path must be absolute.")
    components = path[1:].split("/")
    if any(
        not component
        or component in {".", ".."}
        or "\\" in component
        or "\x00" in component
        for component in components
    ):
        raise FolderValidationError("Folder display path is invalid.")


def _validate_absolute_normalized_path(path: str) -> None:
    if not isinstance(path, str) or not path.startswith("/") or path == "/":
        raise FolderValidationError("Normalized folder path must be absolute.")
    rebuilt = ""
    for component in path[1:].split("/"):
        rebuilt = join_normalized_folder_path(rebuilt, component)
    if rebuilt != path:
        raise FolderValidationError("Normalized folder path is not canonical.")


def _validate_folder_id(folder_id: object, *, field: str) -> None:
    if not isinstance(folder_id, str) or not folder_id:
        raise FolderValidationError(f"{field} must be a non-empty string.")


def validate_deterministic_folder_id(folder_id: object) -> str:
    """Validate and return one caller-owned deterministic folder identifier."""
    if (
        type(folder_id) is not str
        or not 1 <= len(folder_id) <= _CALLER_FOLDER_ID_MAX_LENGTH
        or folder_id[0] not in _ASCII_ALNUM
        or any(character not in _CALLER_FOLDER_ID_CHARACTERS for character in folder_id)
    ):
        raise FolderValidationError("folder_id is invalid.")
    return folder_id


def _validate_expected_version(expected_version: object) -> None:
    _validate_int_bound("expected_version", expected_version, minimum=1)


def _validate_int_bound(
    field: str, value: object, *, minimum: int, maximum: int | None = None
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FolderValidationError(f"{field} is outside the allowed range.")
    if maximum is not None and value > maximum:
        raise FolderValidationError(f"{field} is outside the allowed range.")


def _normalize_folder_ids(folder_ids: Iterable[str]) -> tuple[str, ...]:
    if isinstance(folder_ids, (str, bytes)):
        raise FolderValidationError("expanded_folder_ids must be a collection of IDs.")
    unique_values: set[str] = set()
    try:
        iterator = iter(folder_ids)
    except TypeError as exc:
        raise FolderValidationError(
            "expanded_folder_ids must be a collection of IDs."
        ) from exc
    for item_count, folder_id in enumerate(iterator, start=1):
        if item_count > 100:
            raise FolderValidationError(
                "expanded_folder_ids exceeds the allowed range."
            )
        _validate_folder_id(folder_id, field="expanded folder ID")
        unique_values.add(folder_id)
    return tuple(sorted(unique_values))


def _normalize_ids(values: Iterable[str], *, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise FolderValidationError(f"{field} must be a collection of IDs.")
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise FolderValidationError(f"{field} must be a collection of IDs.") from exc
    for value in normalized:
        _validate_folder_id(value, field=f"{field} item")
    return tuple(sorted(set(normalized)))


def _validate_owner_id(owner_id: object) -> None:
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise FolderValidationError("owner_id must be a non-empty string.")


def _normalize_owner_ids(owner_ids: Iterable[str]) -> tuple[str, ...]:
    if isinstance(owner_ids, (str, bytes)):
        raise FolderValidationError("active_owner_ids must be a collection of IDs.")
    try:
        values = tuple(owner_ids)
    except TypeError as exc:
        raise FolderValidationError(
            "active_owner_ids must be a collection of IDs."
        ) from exc
    for owner_id in values:
        _validate_owner_id(owner_id)
    return tuple(sorted(set(values)))


def _normalize_desired_memberships(
    desired: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if isinstance(desired, (str, bytes)):
        raise FolderValidationError("desired must be a collection of ID pairs.")
    try:
        values = tuple(desired)
    except TypeError as exc:
        raise FolderValidationError(
            "desired must be a collection of ID pairs."
        ) from exc
    normalized: set[tuple[str, str]] = set()
    for value in values:
        if isinstance(value, (str, bytes)):
            raise FolderValidationError("Each desired placement must be an ID pair.")
        try:
            pair = tuple(value)
        except TypeError as exc:
            raise FolderValidationError(
                "Each desired placement must be an ID pair."
            ) from exc
        if len(pair) != 2:
            raise FolderValidationError("Each desired placement must be an ID pair.")
        folder_id, note_id = pair
        _validate_folder_id(folder_id, field="desired folder ID")
        _validate_folder_id(note_id, field="desired note ID")
        normalized.add((folder_id, note_id))
    return tuple(sorted(normalized))


def _chunks(values: Sequence[str], size: int) -> Iterator[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def _placeholders(count: int) -> str:
    if count < 1:
        raise FolderValidationError("At least one placeholder is required.")
    return ",".join("?" for _ in range(count))
