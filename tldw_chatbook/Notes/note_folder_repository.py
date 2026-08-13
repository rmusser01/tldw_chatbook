"""SQLite repository for local Database Note folder hierarchy operations."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import sqlite3
from typing import Iterable, Iterator, NoReturn, Sequence
import uuid

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.note_folder_models import (
    FolderCollisionError,
    FolderConflictError,
    FolderMutationResult,
    FolderValidationError,
    NoteFolder,
    NoteFolderMembership,
    NoteFolderPage,
    join_normalized_folder_path,
    normalize_folder_name,
)


_FOLDER_COLUMNS = (
    "id, parent_id, name, normalized_name, path, normalized_path, version, deleted, "
    "modified_at"
)
_NOTE_COLUMNS = (
    "id, title, content, created_at, last_modified, deleted, client_id, version"
)
_COLLISION_PREFLIGHT_CHUNK_SIZE = 400


class LocalNoteFolderRepository:
    """Own local folder SQL while sharing the application's ChaChaNotes handle."""

    def __init__(self, db: CharactersRAGDB) -> None:
        """Create a repository over an already-initialized database handle."""
        if not isinstance(db, CharactersRAGDB):
            raise TypeError("db must be a CharactersRAGDB instance")
        self.db = db

    def create_folder(self, *, name: str, parent_id: str | None) -> NoteFolder:
        """Create an active folder beneath an active parent, if supplied."""
        normalized = normalize_folder_name(name)
        folder_id = str(uuid.uuid4())
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
                        folder_id,
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
                    (folder_id,),
                ).fetchone()
                if inserted is None:  # pragma: no cover - SQLite guarantees this
                    raise FolderValidationError("Created folder could not be read.")
                return _folder_from_row(inserted)
        except sqlite3.IntegrityError as exc:
            if (
                getattr(exc, "sqlite_errorcode", None)
                == sqlite3.SQLITE_CONSTRAINT_UNIQUE
                and normalized_path is not None
                and self.db.get_connection()
                .execute(
                    "SELECT 1 FROM note_folders "
                    "WHERE deleted = 0 AND normalized_path = ?",
                    (normalized_path,),
                )
                .fetchone()
                is not None
            ):
                raise FolderCollisionError(
                    "An active folder already uses the normalized path."
                ) from exc
            raise FolderValidationError("Folder could not be created.") from exc

    def get_folder(
        self, folder_id: str, *, include_deleted: bool = False
    ) -> NoteFolder | None:
        """Return one exact folder ID, excluding deleted rows by default."""
        _validate_folder_id(folder_id, field="folder_id")
        deleted_clause = "" if include_deleted else " AND deleted = 0"
        row = self.db.get_connection().execute(
            f"SELECT {_FOLDER_COLUMNS} FROM note_folders WHERE id = ?{deleted_clause}",
            (folder_id,),
        ).fetchone()
        return _folder_from_row(row) if row is not None else None

    def list_children(
        self, *, parent_id: str | None, limit: int, offset: int
    ) -> NoteFolderPage:
        """Return a bounded page of active direct children."""
        _validate_int_bound("limit", limit, minimum=1, maximum=500)
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
        )

    def load_tree_batch(
        self, *, expanded_folder_ids: Iterable[str], note_limit: int
    ) -> NoteFolderPage:
        """Bulk-load roots or the immediate contents of expanded folders."""
        _validate_int_bound("note_limit", note_limit, minimum=1, maximum=1000)
        expanded_ids = _normalize_folder_ids(expanded_folder_ids)
        with self.db.transaction() as cursor:
            if expanded_ids:
                placeholders = _placeholders(len(expanded_ids))
                folder_rows = cursor.execute(
                    f"SELECT {_FOLDER_COLUMNS}, COUNT(*) OVER() AS _total_folders "
                    "FROM note_folders WHERE deleted = 0 "
                    f"AND parent_id IN ({placeholders}) "
                    "ORDER BY normalized_path, id",
                    expanded_ids,
                ).fetchall()
                membership_rows = cursor.execute(
                    "SELECT m.id, m.folder_id, m.note_id, m.ownership, m.owner_id, "
                    "m.owner_active, m.version "
                    "FROM note_folder_memberships AS m "
                    "JOIN note_folders AS f "
                    "ON f.id = m.folder_id AND f.deleted = 0 "
                    f"WHERE m.deleted = 0 AND m.folder_id IN ({placeholders}) "
                    "ORDER BY f.normalized_path, m.note_id, m.ownership, "
                    "m.owner_id, m.id",
                    expanded_ids,
                ).fetchall()
                note_rows = cursor.execute(
                    f"SELECT candidate.*, COUNT(*) OVER() AS _total_notes FROM ("
                    f"SELECT DISTINCT n.{_NOTE_COLUMNS.replace(', ', ', n.')} "
                    "FROM notes AS n "
                    "JOIN note_folder_memberships AS m "
                    "ON m.note_id = n.id AND m.deleted = 0 "
                    "JOIN note_folders AS f "
                    "ON f.id = m.folder_id AND f.deleted = 0 "
                    f"WHERE n.deleted = 0 AND m.folder_id IN ({placeholders})"
                    ") AS candidate ORDER BY title COLLATE NOCASE, id LIMIT ?",
                    (*expanded_ids, note_limit),
                ).fetchall()
            else:
                folder_rows = cursor.execute(
                    f"SELECT {_FOLDER_COLUMNS}, COUNT(*) OVER() AS _total_folders "
                    "FROM note_folders WHERE deleted = 0 AND parent_id IS NULL "
                    "ORDER BY normalized_path, id",
                    (),
                ).fetchall()
                membership_rows = cursor.execute(
                    "SELECT m.id, m.folder_id, m.note_id, m.ownership, m.owner_id, "
                    "m.owner_active, m.version FROM note_folder_memberships AS m "
                    "WHERE 0",
                    (),
                ).fetchall()
                note_rows = cursor.execute(
                    f"SELECT candidate.*, COUNT(*) OVER() AS _total_notes FROM ("
                    f"SELECT n.{_NOTE_COLUMNS.replace(', ', ', n.')} "
                    "FROM notes AS n WHERE n.deleted = 0 AND NOT EXISTS ("
                    "SELECT 1 FROM note_folder_memberships AS m "
                    "JOIN note_folders AS f "
                    "ON f.id = m.folder_id AND f.deleted = 0 "
                    "WHERE m.note_id = n.id AND m.deleted = 0 "
                    "AND m.owner_active = 1"
                    ")"
                    ") AS candidate ORDER BY title COLLATE NOCASE, id LIMIT ?",
                    (note_limit,),
                ).fetchall()

        folders = tuple(_folder_from_row(row) for row in folder_rows)
        memberships = tuple(_membership_from_row(row) for row in membership_rows)
        notes = tuple(_note_from_row(row) for row in note_rows)
        total_folders = (
            int(folder_rows[0]["_total_folders"]) if folder_rows else 0
        )
        total_notes = int(note_rows[0]["_total_notes"]) if note_rows else 0
        return NoteFolderPage(
            folders=folders,
            memberships=memberships,
            notes=notes,
            total_folders=total_folders,
            total_notes=total_notes,
            next_offset=note_limit if total_notes > len(notes) else None,
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
                subtree = _load_subtree(cursor, target, deleted=False)
                parent = _load_destination_parent(cursor, parent_id=parent_id)
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


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _unique_deleted_folder_timestamp(cursor: sqlite3.Cursor) -> str:
    """Return a millisecond UTC deletion marker unused by folder tombstones."""
    candidate = _utc_timestamp()
    while cursor.execute(
        "SELECT 1 FROM note_folders WHERE deleted = 1 AND modified_at = ? LIMIT 1",
        (candidate,),
    ).fetchone() is not None:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
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


def _raise_mutation_integrity_error(exc: sqlite3.IntegrityError) -> NoReturn:
    if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
        raise FolderCollisionError(
            "An active folder already uses a resulting normalized path."
        ) from exc
    raise FolderValidationError("Folder mutation violated stored constraints.") from exc


def _raise_mutation_operational_error(exc: sqlite3.OperationalError) -> NoReturn:
    """Translate SQLite writer/snapshot contention into a stable domain conflict."""
    error_code = getattr(exc, "sqlite_errorcode", None)
    primary_code = error_code & 0xFF if isinstance(error_code, int) else None
    if primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        raise FolderConflictError("Folder changed during mutation.") from exc
    raise exc


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
    try:
        values = tuple(folder_ids)
    except TypeError as exc:
        raise FolderValidationError(
            "expanded_folder_ids must be a collection of IDs."
        ) from exc
    for folder_id in values:
        _validate_folder_id(folder_id, field="expanded folder ID")
    return tuple(sorted(set(values)))


def _placeholders(count: int) -> str:
    if count < 1:
        raise FolderValidationError("At least one placeholder is required.")
    return ",".join("?" for _ in range(count))
