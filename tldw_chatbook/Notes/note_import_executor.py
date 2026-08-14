"""Synchronous, replay-safe local target operations for Database Notes imports.

This module intentionally contains no plan loop or receipt orchestration.  It is
the narrow target boundary used by that later executor work.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NoReturn

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.Notes.note_folder_models import (
    FolderCapabilityError,
    FolderCollisionError,
    FolderConflictError,
    FolderValidationError,
    NoteFolder,
)
from tldw_chatbook.Notes.note_folder_repository import (
    LocalNoteFolderRepository,
    validate_deterministic_folder_id,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    MAX_IMPORT_DEPTH,
    MAX_IMPORT_KEYWORD_LENGTH,
    MAX_IMPORT_KEYWORDS_PER_NOTE,
    ParsedNotePayload,
)

_OPAQUE_ID_MAX_LENGTH = 256


class ImportTargetError(RuntimeError):
    """Base class for safe local import target failures."""

    message = "The import target operation failed."

    def __init__(self) -> None:
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ImportTargetRetryableError(ImportTargetError):
    """A temporary local target failure that may succeed on retry."""

    message = "The import target is temporarily unavailable."


class ImportTargetConflictError(ImportTargetError):
    """The local target no longer matches the approved operation."""

    message = "The import target conflicts with the approved operation."


class ImportTargetPermanentError(ImportTargetError):
    """A validation or capability failure that retry cannot resolve."""

    message = "The import target cannot apply this operation."


class ImportTargetInternalError(RuntimeError):
    """A privacy-safe fatal target failure caused by an unexpected internal fault."""

    message = "The import target encountered an internal failure."

    def __init__(self) -> None:
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True, slots=True, repr=False)
class LocalTargetFolder:
    """Private-safe immutable folder state used by import execution."""

    folder_id: str
    name: str
    path: str
    normalized_path: str

    def __repr__(self) -> str:
        return "LocalTargetFolder(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class LocalTargetNote:
    """Private-safe immutable note state used only for reconciliation."""

    note_id: str
    title: str
    content: str
    version: int
    keywords: tuple[str, ...]

    def __repr__(self) -> str:
        return "LocalTargetNote(<private>)"


class LocalNoteImportTarget:
    """Apply synchronous deterministic operations to local Database Notes."""

    def __init__(
        self,
        *,
        db: CharactersRAGDB,
        folder_repository: LocalNoteFolderRepository,
    ) -> None:
        try:
            if not isinstance(db, CharactersRAGDB):
                raise TypeError("db must be a CharactersRAGDB instance.")
            if not isinstance(folder_repository, LocalNoteFolderRepository):
                raise TypeError(
                    "folder_repository must be a LocalNoteFolderRepository instance."
                )
            if folder_repository.db is not db:
                raise ValueError("Target components must share one database.")
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            _raise_translated(exc)
        self._folders = folder_repository
        self._user_id = db.client_id
        self._db = db

    def __repr__(self) -> str:
        return "LocalNoteImportTarget(<private>)"

    def ensure_folder(
        self,
        *,
        segments: Iterable[str],
        folder_id: str,
        allow_existing: bool,
    ) -> LocalTargetFolder:
        """Ensure one exact folder path has an approved identity or reuse policy."""
        try:
            validate_deterministic_folder_id(folder_id)
            if not isinstance(allow_existing, bool):
                raise TypeError("allow_existing must be a boolean.")
            copied_segments = _copy_segments(segments)
            existing = self._folders.get_folder_by_path(copied_segments)
            if existing is not None:
                return _project_folder(
                    _reconcile_folder(
                        existing,
                        folder_id=folder_id,
                        allow_existing=allow_existing,
                    )
                )

            identity_owner = self._folders.get_folder(folder_id, include_deleted=True)
            if identity_owner is not None:
                raise ImportTargetConflictError from None

            parent_id: str | None = None
            if len(copied_segments) > 1:
                parent = self._folders.get_folder_by_path(copied_segments[:-1])
                if parent is None:
                    raise ImportTargetPermanentError from None
                parent_id = parent.folder_id

            try:
                return _project_folder(
                    self._folders.create_folder(
                        name=copied_segments[-1],
                        parent_id=parent_id,
                        folder_id=folder_id,
                    )
                )
            except (FolderCollisionError, FolderValidationError):
                winner = self._folders.get_folder_by_path(copied_segments)
                if winner is not None:
                    return _project_folder(
                        _reconcile_folder(
                            winner,
                            folder_id=folder_id,
                            allow_existing=allow_existing,
                        )
                    )
                if (
                    self._folders.get_folder(folder_id, include_deleted=True)
                    is not None
                ):
                    raise ImportTargetConflictError from None
                raise
        except (ImportTargetError, ImportTargetInternalError):
            raise
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            _raise_translated(exc)

    def read_note(self, *, note_id: str) -> LocalTargetNote | None:
        """Read one active note as a frozen private reconciliation projection."""
        try:
            _validate_opaque_id(note_id)
            with self._db.transaction() as cursor:
                return self._read_note(cursor, note_id)
        except (ImportTargetError, ImportTargetInternalError):
            raise
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            _raise_translated(exc)

    def create_note(
        self, *, note_id: str, payload: ParsedNotePayload
    ) -> LocalTargetNote:
        """Create a deterministic note and its exact keywords, or reconcile it."""
        try:
            _validate_opaque_id(note_id)
            _validate_payload(payload)
            with self._db.transaction() as cursor:
                existing = self._read_note(cursor, note_id)
                if existing is not None:
                    if _note_matches(existing, payload):
                        return existing
                    raise ImportTargetConflictError from None

                self._insert_note(cursor, note_id, payload)
                self._sync_keywords(cursor, note_id, payload.keywords)
                created = self._read_note(cursor, note_id)
                if created is None or not _note_matches(created, payload):
                    raise ImportTargetPermanentError from None
                return created
        except (ImportTargetError, ImportTargetInternalError):
            raise
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            _raise_translated(exc)

    def replace_note(
        self,
        *,
        note_id: str,
        expected_version: int,
        payload: ParsedNotePayload,
    ) -> LocalTargetNote:
        """Optimistically replace note text and exact keywords once."""
        try:
            _validate_opaque_id(note_id)
            _validate_expected_version(expected_version)
            _validate_payload(payload)
            with self._db.transaction() as cursor:
                current = self._read_note(cursor, note_id)
                if current is None:
                    raise ImportTargetConflictError from None
                if current.version == expected_version + 1:
                    if _note_matches(current, payload):
                        return current
                    raise ImportTargetConflictError from None
                if current.version != expected_version:
                    raise ImportTargetConflictError from None

                if not self._update_note(
                    cursor,
                    note_id=note_id,
                    expected_version=expected_version,
                    payload=payload,
                ):
                    raise ImportTargetConflictError from None
                self._sync_keywords(cursor, note_id, payload.keywords)
                result = self._read_note(cursor, note_id)
                if (
                    result is None
                    or result.version != expected_version + 1
                    or not _note_matches(result, payload)
                ):
                    raise ImportTargetConflictError from None
                return result
        except (ImportTargetError, ImportTargetInternalError):
            raise
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            _raise_translated(exc)

    def keywords_match(self, *, note_id: str, keywords: Iterable[str]) -> bool:
        """Return whether a note has exactly the desired canonical keyword set."""
        try:
            _validate_opaque_id(note_id)
            desired = _normalize_keywords(keywords)
            with self._db.transaction() as cursor:
                if not self._active_note_exists(cursor, note_id):
                    raise ImportTargetPermanentError from None
                current = self._keyword_rows(cursor, note_id)
                return _keyword_keys_from_rows(current) == set(desired)
        except (ImportTargetError, ImportTargetInternalError):
            raise
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            _raise_translated(exc)

    def sync_keywords(self, *, note_id: str, keywords: Iterable[str]) -> None:
        """Make one active note's canonical keyword links exactly match desired."""
        try:
            _validate_opaque_id(note_id)
            desired = _normalize_keywords(keywords)
            with self._db.transaction() as cursor:
                if not self._active_note_exists(cursor, note_id):
                    raise ImportTargetPermanentError from None
                self._sync_normalized_keywords(cursor, note_id, desired)
        except (ImportTargetError, ImportTargetInternalError):
            raise
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            _raise_translated(exc)

    def attach_membership(self, *, folder_id: str, note_id: str) -> None:
        """Idempotently attach one active note to one active manual folder."""
        try:
            _validate_opaque_id(folder_id)
            _validate_opaque_id(note_id)
            self._folders.attach_manual(folder_id=folder_id, note_id=note_id)
        except (ImportTargetError, ImportTargetInternalError):
            raise
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            _raise_translated(exc)

    def _read_note(
        self, cursor: sqlite3.Cursor, note_id: str
    ) -> LocalTargetNote | None:
        row = cursor.execute(
            "SELECT id, title, content, version "
            "FROM notes WHERE id = ? AND deleted = 0",
            (note_id,),
        ).fetchone()
        if row is None:
            return None
        keywords = self._keyword_rows(cursor, note_id)
        try:
            row_note_id = row["id"]
            title = row["title"]
            content = row["content"]
            version = row["version"]
            keyword_values = tuple(keyword["keyword"] for keyword in keywords)
        except (KeyError, TypeError):
            raise ImportTargetPermanentError from None
        if (
            not isinstance(row_note_id, str)
            or row_note_id != note_id
            or not isinstance(title, str)
            or not isinstance(content, str)
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
            or not all(isinstance(keyword, str) for keyword in keyword_values)
        ):
            raise ImportTargetPermanentError from None
        return LocalTargetNote(
            note_id=row_note_id,
            title=title,
            content=content,
            version=version,
            keywords=keyword_values,
        )

    @staticmethod
    def _active_note_exists(cursor: sqlite3.Cursor, note_id: str) -> bool:
        return (
            cursor.execute(
                "SELECT 1 FROM notes WHERE id = ? AND deleted = 0", (note_id,)
            ).fetchone()
            is not None
        )

    @staticmethod
    def _keyword_rows(cursor: sqlite3.Cursor, note_id: str) -> list[sqlite3.Row]:
        return cursor.execute(
            """
            SELECT k.id, k.keyword, k.version, k.deleted
            FROM keywords AS k
            JOIN note_keywords AS nk ON nk.keyword_id = k.id
            WHERE nk.note_id = ? AND k.deleted = 0
            ORDER BY k.keyword COLLATE NOCASE
            """,
            (note_id,),
        ).fetchall()

    @staticmethod
    def _linked_keyword_rows(cursor: sqlite3.Cursor, note_id: str) -> list[sqlite3.Row]:
        return cursor.execute(
            """
            SELECT k.id, k.keyword, k.version, k.deleted
            FROM keywords AS k
            JOIN note_keywords AS nk ON nk.keyword_id = k.id
            WHERE nk.note_id = ?
            ORDER BY k.keyword COLLATE NOCASE
            """,
            (note_id,),
        ).fetchall()

    def _insert_note(
        self,
        cursor: sqlite3.Cursor,
        note_id: str,
        payload: ParsedNotePayload,
    ) -> None:
        timestamp = _utc_timestamp()
        cursor.execute(
            """
            INSERT INTO notes (
                id, title, content, created_at, last_modified,
                deleted, client_id, version
            )
            VALUES (?, ?, ?, ?, ?, 0, ?, 1)
            """,
            (
                note_id,
                payload.title.strip(),
                payload.content,
                timestamp,
                timestamp,
                self._user_id,
            ),
        )

    def _update_note(
        self,
        cursor: sqlite3.Cursor,
        *,
        note_id: str,
        expected_version: int,
        payload: ParsedNotePayload,
    ) -> bool:
        result = cursor.execute(
            """
            UPDATE notes
            SET title = ?, content = ?, last_modified = ?, version = ?, client_id = ?
            WHERE id = ? AND version = ? AND deleted = 0
            """,
            (
                payload.title.strip(),
                payload.content,
                _utc_timestamp(),
                expected_version + 1,
                self._user_id,
                note_id,
                expected_version,
            ),
        )
        return result.rowcount == 1

    def _sync_keywords(
        self,
        cursor: sqlite3.Cursor,
        note_id: str,
        keywords: Iterable[str],
    ) -> None:
        self._sync_normalized_keywords(cursor, note_id, _normalize_keywords(keywords))

    def _sync_normalized_keywords(
        self,
        cursor: sqlite3.Cursor,
        note_id: str,
        desired: dict[str, str],
    ) -> None:
        current_rows = self._linked_keyword_rows(cursor, note_id)
        current_by_key: dict[str, tuple[int, bool]] = {}
        for row in current_rows:
            try:
                keyword_id = row["id"]
                keyword_text = row["keyword"]
                deleted = row["deleted"]
            except (KeyError, TypeError):
                raise ImportTargetPermanentError from None
            if isinstance(keyword_id, bool) or not isinstance(keyword_id, int):
                raise ImportTargetPermanentError from None
            if not isinstance(keyword_text, str):
                raise ImportTargetPermanentError from None
            if deleted not in (0, 1):
                raise ImportTargetPermanentError from None
            current_by_key[_sqlite_nocase_key(keyword_text)] = (
                keyword_id,
                deleted == 0,
            )

        for key, (keyword_id, _active) in tuple(current_by_key.items()):
            if key not in desired:
                self._unlink_keyword(cursor, note_id, keyword_id)
                current_by_key.pop(key)

        for key, keyword_text in desired.items():
            if key in current_by_key:
                _keyword_id, active = current_by_key[key]
                if not active:
                    self._ensure_keyword(cursor, keyword_text)
                continue
            keyword_id = self._ensure_keyword(cursor, keyword_text)
            self._link_keyword(cursor, note_id, keyword_id)

    def _ensure_keyword(self, cursor: sqlite3.Cursor, keyword_text: str) -> int:
        row = cursor.execute(
            "SELECT id, version, deleted FROM keywords WHERE keyword = ?",
            (keyword_text,),
        ).fetchone()
        if row is None:
            timestamp = _utc_timestamp()
            result = cursor.execute(
                """
                INSERT INTO keywords (
                    keyword, created_at, last_modified, deleted, client_id, version
                )
                VALUES (?, ?, ?, 0, ?, 1)
                """,
                (keyword_text, timestamp, timestamp, self._user_id),
            )
            keyword_id = result.lastrowid
            if isinstance(keyword_id, bool) or not isinstance(keyword_id, int):
                raise ImportTargetPermanentError from None
            return keyword_id

        keyword_id = row["id"]
        version = row["version"]
        deleted = row["deleted"]
        if (
            isinstance(keyword_id, bool)
            or not isinstance(keyword_id, int)
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
            or deleted not in (0, 1)
        ):
            raise ImportTargetPermanentError from None
        if deleted == 0:
            return keyword_id

        result = cursor.execute(
            """
            UPDATE keywords
            SET keyword = ?, last_modified = ?, deleted = 0,
                client_id = ?, version = ?
            WHERE id = ? AND version = ? AND deleted = 1
            """,
            (
                keyword_text,
                _utc_timestamp(),
                self._user_id,
                version + 1,
                keyword_id,
                version,
            ),
        )
        if result.rowcount != 1:
            raise ImportTargetConflictError from None
        return keyword_id

    def _link_keyword(
        self, cursor: sqlite3.Cursor, note_id: str, keyword_id: int
    ) -> None:
        timestamp = _utc_timestamp()
        result = cursor.execute(
            """
            INSERT OR IGNORE INTO note_keywords (note_id, keyword_id, created_at)
            VALUES (?, ?, ?)
            """,
            (note_id, keyword_id, timestamp),
        )
        if result.rowcount > 0:
            self._record_keyword_link_change(
                cursor,
                note_id=note_id,
                keyword_id=keyword_id,
                operation="create",
                timestamp=timestamp,
                payload={
                    "note_id": note_id,
                    "keyword_id": keyword_id,
                    "created_at": timestamp,
                },
            )

    def _unlink_keyword(
        self, cursor: sqlite3.Cursor, note_id: str, keyword_id: int
    ) -> None:
        result = cursor.execute(
            "DELETE FROM note_keywords WHERE note_id = ? AND keyword_id = ?",
            (note_id, keyword_id),
        )
        if result.rowcount > 0:
            timestamp = _utc_timestamp()
            self._record_keyword_link_change(
                cursor,
                note_id=note_id,
                keyword_id=keyword_id,
                operation="delete",
                timestamp=timestamp,
                payload={"note_id": note_id, "keyword_id": keyword_id},
            )

    def _record_keyword_link_change(
        self,
        cursor: sqlite3.Cursor,
        *,
        note_id: str,
        keyword_id: int,
        operation: str,
        timestamp: str,
        payload: dict[str, object],
    ) -> None:
        cursor.execute(
            """
            INSERT INTO sync_log (
                entity, entity_id, operation, timestamp, client_id, version, payload
            )
            VALUES ('note_keywords', ?, ?, ?, ?, 1, ?)
            """,
            (
                f"{note_id}_{keyword_id}",
                operation,
                timestamp,
                self._user_id,
                json.dumps(payload),
            ),
        )


def _project_folder(folder: NoteFolder) -> LocalTargetFolder:
    values = (
        folder.folder_id,
        folder.name,
        folder.path,
        folder.normalized_path,
    )
    if any(type(value) is not str for value in values):
        raise ImportTargetPermanentError from None
    return LocalTargetFolder(
        folder_id=folder.folder_id,
        name=folder.name,
        path=folder.path,
        normalized_path=folder.normalized_path,
    )


def _reconcile_folder(
    folder: NoteFolder, *, folder_id: str, allow_existing: bool
) -> NoteFolder:
    if folder.folder_id == folder_id or allow_existing:
        return folder
    raise ImportTargetConflictError from None


def _copy_segments(segments: Iterable[str]) -> tuple[str, ...]:
    if isinstance(segments, (str, bytes)):
        raise FolderValidationError("Folder path must be a collection of segments.")
    try:
        iterator = iter(segments)
    except TypeError:
        raise FolderValidationError(
            "Folder path must be a collection of segments."
        ) from None
    copied: list[str] = []
    for count, segment in enumerate(iterator, start=1):
        if count > MAX_IMPORT_DEPTH:
            raise FolderValidationError("Folder path exceeds the allowed range.")
        copied.append(segment)
    if not copied:
        raise FolderValidationError("Folder path must identify a folder.")
    return tuple(copied)


def _validate_opaque_id(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _OPAQUE_ID_MAX_LENGTH
        or "\x00" in value
    ):
        raise ValueError("Opaque identifier is invalid.")


def _validate_expected_version(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Expected version is invalid.")


def _validate_payload(payload: object) -> None:
    if not isinstance(payload, ParsedNotePayload):
        raise TypeError("payload must be a ParsedNotePayload instance.")
    _normalize_keywords(payload.keywords)


def _normalize_keywords(keywords: Iterable[str]) -> dict[str, str]:
    if isinstance(keywords, (str, bytes)):
        raise TypeError("Keywords must be a collection of text values.")
    try:
        iterator = iter(keywords)
    except TypeError:
        raise TypeError("Keywords must be a collection of text values.") from None
    normalized: dict[str, str] = {}
    for count, value in enumerate(iterator, start=1):
        if count > MAX_IMPORT_KEYWORDS_PER_NOTE:
            raise ValueError("Keywords exceed the allowed range.")
        if not isinstance(value, str):
            raise TypeError("Keywords must contain text values.")
        display = value.strip()
        if not display or len(display) > MAX_IMPORT_KEYWORD_LENGTH or "\x00" in display:
            raise ValueError("Keyword is invalid.")
        normalized.setdefault(_sqlite_nocase_key(display), display)
    return dict(sorted(normalized.items()))


def _sqlite_nocase_key(value: str) -> str:
    return value.translate(
        str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
    )


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _keyword_keys_from_rows(rows: object) -> set[str]:
    if not isinstance(rows, list):
        raise ImportTargetPermanentError from None
    keys: set[str] = set()
    for row in rows:
        try:
            value = row["keyword"]
        except (KeyError, TypeError):
            raise ImportTargetPermanentError from None
        if not isinstance(value, str):
            raise ImportTargetPermanentError from None
        keys.add(_sqlite_nocase_key(value))
    return keys


def _note_matches(note: LocalTargetNote, payload: ParsedNotePayload) -> bool:
    desired_keywords = set(_normalize_keywords(payload.keywords))
    actual_keywords = {_sqlite_nocase_key(keyword) for keyword in note.keywords}
    return (
        note.title == payload.title.strip()
        and note.content == payload.content
        and actual_keywords == desired_keywords
    )


def _contains_sqlite_contention(exc: Exception) -> bool:
    pending: list[Exception] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, sqlite3.OperationalError):
            error_code = getattr(current, "sqlite_errorcode", None)
            primary_code = error_code & 0xFF if isinstance(error_code, int) else None
            if primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                return True
            lowered = str(current).casefold()
            if "database" in lowered and ("locked" in lowered or "busy" in lowered):
                return True
        if isinstance(current.__cause__, Exception):
            pending.append(current.__cause__)
        if isinstance(current.__context__, Exception):
            pending.append(current.__context__)
    return False


def _raise_translated(exc: Exception) -> NoReturn:
    if not isinstance(
        exc,
        (
            CharactersRAGDBError,
            FolderCapabilityError,
            FolderCollisionError,
            FolderConflictError,
            FolderValidationError,
            sqlite3.Error,
            TypeError,
            ValueError,
        ),
    ):
        raise ImportTargetInternalError from None
    if _contains_sqlite_contention(exc):
        raise ImportTargetRetryableError from None
    if isinstance(
        exc,
        (
            FolderCollisionError,
            FolderConflictError,
            sqlite3.IntegrityError,
        ),
    ):
        raise ImportTargetConflictError from None
    if isinstance(
        exc,
        (
            CharactersRAGDBError,
            FolderCapabilityError,
            FolderValidationError,
            sqlite3.Error,
            TypeError,
            ValueError,
        ),
    ):
        raise ImportTargetPermanentError from None
    raise ImportTargetInternalError from None


__all__ = [
    "ImportTargetConflictError",
    "ImportTargetError",
    "ImportTargetInternalError",
    "ImportTargetPermanentError",
    "ImportTargetRetryableError",
    "LocalNoteImportTarget",
    "LocalTargetFolder",
    "LocalTargetNote",
]
