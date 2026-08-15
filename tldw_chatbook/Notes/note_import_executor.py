"""Synchronous local target operations and durable one-time import execution."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid5

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.Notes.note_folder_models import (
    FolderCapabilityError,
    FolderCollisionError,
    FolderConflictError,
    FolderValidationError,
    NoteFolder,
    join_normalized_folder_path,
    normalize_folder_name,
)
from tldw_chatbook.Notes.note_folder_repository import (
    LocalNoteFolderRepository,
    validate_deterministic_folder_id,
)
from tldw_chatbook.Notes.note_import_execution_models import (
    ApprovedNoteImportPlan,
    ImportEffectState,
    ImportExecutionProgress,
    ImportExecutionReceipt,
    ImportItemOutcome,
    ImportSessionState,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    MAX_IMPORT_DEPTH,
    MAX_IMPORT_KEYWORD_LENGTH,
    MAX_IMPORT_KEYWORDS_PER_NOTE,
    ImportAction,
    ImportPreviewItem,
    ParsedNotePayload,
    ProposedFolderMembership,
    RootCollisionChoice,
)
from tldw_chatbook.Notes.note_import_receipts import (
    EffectTransition,
    ImportEffectCategory,
    ImportEffectRecord,
    ImportReceiptTransitionError,
    ImportSessionSnapshot,
    NoteImportReceiptRepository,
    _folder_path_digest,
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


class _ImportTargetValidationError(Exception):
    """Private marker for explicit target-boundary input validation failures."""

    message = "The import target input is invalid."

    def __init__(self) -> None:
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class _ImportTargetContractError(Exception):
    """Private marker for malformed component or database target state."""

    message = "The import target contract is invalid."

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
        translated_error: ImportTargetError | ImportTargetInternalError | None = None
        try:
            if not isinstance(db, CharactersRAGDB):
                raise _ImportTargetValidationError
            if not isinstance(folder_repository, LocalNoteFolderRepository):
                raise _ImportTargetValidationError
            if folder_repository.db is not db:
                raise _ImportTargetValidationError
            self._folders = folder_repository
            self._user_id = db.client_id
            self._db = db
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            translated_error = _translate_exception(exc)
        if translated_error is not None:
            raise translated_error from None

    def __repr__(self) -> str:
        return "LocalNoteImportTarget(<private>)"

    def ensure_folder(
        self,
        *,
        segments: Iterable[str],
        folder_id: str,
        allow_existing: bool,
    ) -> LocalTargetFolder:
        """Ensure one exact folder path has an approved identity or reuse policy.

        Args:
            segments: Ordered folder names from the selected root to the leaf.
            folder_id: Deterministic approved identity for a newly created path.
            allow_existing: Whether an existing path may be reused safely.

        Returns:
            The reconciled existing folder or newly created folder projection.

        Raises:
            ImportTargetConflictError: If the approved identity or reuse policy
                conflicts with current folder state.
            ImportTargetError: If target validation or persistence fails.
            ImportTargetInternalError: If an unexpected target failure occurs.
        """
        translated_error: ImportTargetError | ImportTargetInternalError | None = None
        try:
            validate_deterministic_folder_id(folder_id)
            if not isinstance(allow_existing, bool):
                raise _ImportTargetValidationError
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
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            translated_error = _translate_exception(exc)
        if translated_error is not None:
            raise translated_error from None
        raise ImportTargetInternalError from None  # pragma: no cover

    def read_note(self, *, note_id: str) -> LocalTargetNote | None:
        """Read one active note as a frozen private reconciliation projection."""
        translated_error: ImportTargetError | ImportTargetInternalError | None = None
        try:
            _validate_opaque_id(note_id)
            with self._db.transaction() as cursor:
                return self._read_note(cursor, note_id)
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            translated_error = _translate_exception(exc)
        if translated_error is not None:
            raise translated_error from None
        raise ImportTargetInternalError from None  # pragma: no cover

    def create_note(
        self, *, note_id: str, payload: ParsedNotePayload
    ) -> LocalTargetNote:
        """Create a deterministic note and its exact keywords, or reconcile it."""
        translated_error: ImportTargetError | ImportTargetInternalError | None = None
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
                    raise _ImportTargetContractError from None
                return created
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            translated_error = _translate_exception(exc)
        if translated_error is not None:
            raise translated_error from None
        raise ImportTargetInternalError from None  # pragma: no cover

    def replace_note(
        self,
        *,
        note_id: str,
        expected_version: int,
        payload: ParsedNotePayload,
    ) -> LocalTargetNote:
        """Optimistically replace note text and exact keywords once."""
        translated_error: ImportTargetError | ImportTargetInternalError | None = None
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
                    raise _ImportTargetContractError from None
                return result
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            translated_error = _translate_exception(exc)
        if translated_error is not None:
            raise translated_error from None
        raise ImportTargetInternalError from None  # pragma: no cover

    def keywords_match(self, *, note_id: str, keywords: Iterable[str]) -> bool:
        """Return whether a note has exactly the desired canonical keyword set."""
        translated_error: ImportTargetError | ImportTargetInternalError | None = None
        try:
            _validate_opaque_id(note_id)
            desired = _normalize_keywords(keywords)
            with self._db.transaction() as cursor:
                if not self._active_note_exists(cursor, note_id):
                    raise ImportTargetPermanentError from None
                current = self._keyword_rows(cursor, note_id)
                return _keyword_keys_from_rows(current) == set(desired)
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            translated_error = _translate_exception(exc)
        if translated_error is not None:
            raise translated_error from None
        raise ImportTargetInternalError from None  # pragma: no cover

    def sync_keywords(self, *, note_id: str, keywords: Iterable[str]) -> None:
        """Make one active note's canonical keyword links exactly match desired."""
        translated_error: ImportTargetError | ImportTargetInternalError | None = None
        try:
            _validate_opaque_id(note_id)
            desired = _normalize_keywords(keywords)
            with self._db.transaction() as cursor:
                if not self._active_note_exists(cursor, note_id):
                    raise ImportTargetPermanentError from None
                self._sync_normalized_keywords(cursor, note_id, desired)
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            translated_error = _translate_exception(exc)
        if translated_error is not None:
            raise translated_error from None

    def attach_membership(
        self,
        *,
        folder_id: str,
        note_id: str,
        expected_note_version: int,
    ) -> None:
        """Idempotently attach one active note to one active manual folder."""
        translated_error: ImportTargetError | ImportTargetInternalError | None = None
        try:
            _validate_opaque_id(folder_id)
            _validate_opaque_id(note_id)
            _validate_expected_version(expected_note_version)
            self._folders.attach_manual(
                folder_id=folder_id,
                note_id=note_id,
                expected_note_version=expected_note_version,
            )
        except Exception as exc:  # noqa: BLE001 - translate target boundary failures
            translated_error = _translate_exception(exc)
        if translated_error is not None:
            raise translated_error from None

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
        except (IndexError, KeyError, TypeError):
            raise _ImportTargetContractError from None
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
            raise _ImportTargetContractError from None
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
            except (IndexError, KeyError, TypeError):
                raise _ImportTargetContractError from None
            if isinstance(keyword_id, bool) or not isinstance(keyword_id, int):
                raise _ImportTargetContractError from None
            if not isinstance(keyword_text, str):
                raise _ImportTargetContractError from None
            if deleted not in (0, 1):
                raise _ImportTargetContractError from None
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
                raise _ImportTargetContractError from None
            return keyword_id

        try:
            keyword_id = row["id"]
            version = row["version"]
            deleted = row["deleted"]
        except (IndexError, KeyError, TypeError):
            raise _ImportTargetContractError from None
        if (
            isinstance(keyword_id, bool)
            or not isinstance(keyword_id, int)
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
            or deleted not in (0, 1)
        ):
            raise _ImportTargetContractError from None
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
        raise _ImportTargetContractError from None
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
        raise _ImportTargetValidationError


def _validate_expected_version(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _ImportTargetValidationError


def _validate_payload(payload: object) -> None:
    if not isinstance(payload, ParsedNotePayload):
        raise _ImportTargetValidationError
    _normalize_keywords(payload.keywords)


def _normalize_keywords(keywords: Iterable[str]) -> dict[str, str]:
    if isinstance(keywords, (str, bytes)):
        raise _ImportTargetValidationError
    try:
        iterator = iter(keywords)
    except TypeError:
        raise _ImportTargetValidationError from None
    normalized: dict[str, str] = {}
    for count, value in enumerate(iterator, start=1):
        if count > MAX_IMPORT_KEYWORDS_PER_NOTE:
            raise _ImportTargetValidationError
        if not isinstance(value, str):
            raise _ImportTargetValidationError
        display = value.strip()
        if not display or len(display) > MAX_IMPORT_KEYWORD_LENGTH or "\x00" in display:
            raise _ImportTargetValidationError
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
        raise _ImportTargetContractError from None
    keys: set[str] = set()
    for row in rows:
        try:
            value = row["keyword"]
        except (IndexError, KeyError, TypeError):
            raise _ImportTargetContractError from None
        if not isinstance(value, str):
            raise _ImportTargetContractError from None
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


def _translate_exception(
    exc: Exception,
) -> ImportTargetError | ImportTargetInternalError:
    if isinstance(exc, ImportTargetRetryableError):
        return ImportTargetRetryableError()
    if isinstance(exc, ImportTargetConflictError):
        return ImportTargetConflictError()
    if isinstance(exc, ImportTargetPermanentError):
        return ImportTargetPermanentError()
    if isinstance(exc, ImportTargetError):
        return ImportTargetError()
    if isinstance(exc, ImportTargetInternalError):
        return ImportTargetInternalError()
    if isinstance(exc, _ImportTargetContractError):
        return ImportTargetInternalError()
    if isinstance(exc, _ImportTargetValidationError):
        return ImportTargetPermanentError()
    expected_family = (
        CharactersRAGDBError,
        FolderCapabilityError,
        FolderCollisionError,
        FolderConflictError,
        FolderValidationError,
        sqlite3.Error,
    )
    if not isinstance(exc, expected_family):
        return ImportTargetInternalError()
    try:
        if _contains_sqlite_contention(exc):
            return ImportTargetRetryableError()
    except Exception:  # noqa: BLE001 - exception inspection must remain privacy-safe
        return ImportTargetInternalError()
    if isinstance(
        exc,
        (
            FolderCollisionError,
            FolderConflictError,
            sqlite3.IntegrityError,
        ),
    ):
        return ImportTargetConflictError()
    if isinstance(
        exc,
        (
            CharactersRAGDBError,
            FolderCapabilityError,
            FolderValidationError,
            sqlite3.Error,
        ),
    ):
        return ImportTargetPermanentError()
    return ImportTargetInternalError()


@dataclass(frozen=True, slots=True)
class _ExecutionFailure:
    """One bounded operational failure safe to persist in the receipt ledger."""

    reason_code: str
    retryable: bool


class NoteImportExecutor:
    """Execute one approved local import plan with per-effect durable receipts."""

    def __init__(
        self,
        *,
        target: LocalNoteImportTarget,
        receipt_repository: NoteImportReceiptRepository,
        batch_size: int = 25,
    ) -> None:
        if type(target) is not LocalNoteImportTarget:
            raise TypeError("target must be a LocalNoteImportTarget.")
        if type(receipt_repository) is not NoteImportReceiptRepository:
            raise TypeError("receipt_repository must be a NoteImportReceiptRepository.")
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer.")
        if not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100.")
        self._target = target
        self._receipts = receipt_repository
        self._batch_size = batch_size

    def __repr__(self) -> str:
        return "NoteImportExecutor(<private>)"

    def execute(
        self,
        approved: ApprovedNoteImportPlan,
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[ImportExecutionProgress], None] | None = None,
        _recovering_interruption: bool | None = None,
    ) -> ImportExecutionReceipt:
        """Execute one freshly approved plan and return its durable receipt."""
        if type(approved) is not ApprovedNoteImportPlan:
            raise TypeError("approved must be an ApprovedNoteImportPlan.")
        if cancel_event is not None and type(cancel_event) is not threading.Event:
            raise TypeError("cancel_event must be a threading.Event when provided.")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable when provided.")
        snapshot = self._receipts.begin(approved, batch_size=self._batch_size)
        recovering_interruption = (
            snapshot.state is ImportSessionState.RUNNING
            if _recovering_interruption is None
            else _recovering_interruption
        )
        if snapshot.state is ImportSessionState.COMPLETED:
            receipt = self._receipts.aggregate_receipt(approved.approval_id)
            _publish_receipt_progress(receipt, progress_callback)
            return receipt
        if snapshot.state is ImportSessionState.PENDING:
            self._receipts.transition_session(
                approved.approval_id,
                ImportSessionState.RUNNING,
            )
            snapshot = self._receipts.load_session_snapshot(approved.approval_id)
        elif snapshot.state is not ImportSessionState.RUNNING:
            raise ImportReceiptTransitionError(
                "Execution requires a pending or interrupted running receipt session."
            )
        self._publish_progress(approved.approval_id, progress_callback)
        if cancel_event is not None and cancel_event.is_set():
            return self._cancel(approved.approval_id, progress_callback)

        folder_bindings, folder_failures, folders_cancelled = self._execute_folders(
            approved,
            snapshot.folder_effects,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        if folders_cancelled:
            return self._cancel(approved.approval_id, progress_callback)
        payload_effects_by_item: dict[str, dict[int | None, ImportEffectRecord]] = {}
        for effect in snapshot.payload_effects:
            if effect.item_id is None:
                raise ImportReceiptTransitionError(
                    "Payload receipt authority does not match the approved plan."
                )
            payload_effects_by_item.setdefault(effect.item_id, {})[
                effect.payload_index
            ] = effect
        membership_effects_by_item: dict[str, list[ImportEffectRecord]] = {}
        for effect in snapshot.membership_effects:
            if effect.item_id is None:
                raise ImportReceiptTransitionError(
                    "Membership receipt authority does not match the approved plan."
                )
            membership_effects_by_item.setdefault(effect.item_id, []).append(effect)
        items_by_id = {item.item_id: item for item in snapshot.items}
        items = approved.plan.items
        for batch_start in range(0, len(items), self._batch_size):
            if cancel_event is not None and cancel_event.is_set():
                return self._cancel(approved.approval_id, progress_callback)
            for item in items[batch_start : batch_start + self._batch_size]:
                durable_item = items_by_id.get(item.item_id)
                if durable_item is None:
                    raise ImportReceiptTransitionError(
                        "Item receipt authority does not match the approved plan."
                    )
                if durable_item.outcome is not ImportItemOutcome.PENDING:
                    continue
                self._execute_item(
                    approved,
                    item=item,
                    payload_effects=payload_effects_by_item.get(item.item_id, {}),
                    membership_effects=tuple(
                        membership_effects_by_item.get(item.item_id, ())
                    ),
                    folder_bindings=folder_bindings,
                    folder_failures=folder_failures,
                    recovering_interruption=recovering_interruption,
                )
            self._publish_progress(approved.approval_id, progress_callback)

        running_receipt = self._receipts.aggregate_receipt(approved.approval_id)
        final_state = (
            ImportSessionState.NEEDS_ATTENTION
            if running_receipt.failed
            else ImportSessionState.COMPLETED
        )
        self._receipts.transition_session(approved.approval_id, final_state)
        receipt = self._receipts.aggregate_receipt(approved.approval_id)
        _publish_receipt_progress(receipt, progress_callback)
        return receipt

    async def execute_async(
        self,
        approved: ApprovedNoteImportPlan,
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[ImportExecutionProgress], None] | None = None,
    ) -> ImportExecutionReceipt:
        """Run the complete synchronous executor away from the caller's event loop."""
        loop = asyncio.get_running_loop()

        def publish(progress: ImportExecutionProgress) -> None:
            if progress_callback is not None:
                loop.call_soon_threadsafe(progress_callback, progress)

        return await asyncio.to_thread(
            self.execute,
            approved,
            cancel_event=cancel_event,
            progress_callback=publish,
        )

    def retry_failed(
        self,
        approved: ApprovedNoteImportPlan,
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[ImportExecutionProgress], None] | None = None,
    ) -> ImportExecutionReceipt:
        """Retry only unfinished or explicitly retryable work in one session."""
        if type(approved) is not ApprovedNoteImportPlan:
            raise TypeError("approved must be an ApprovedNoteImportPlan.")
        snapshot = self._receipts.begin(approved, batch_size=self._batch_size)
        recovering_interruption = snapshot.state is ImportSessionState.RUNNING
        if snapshot.state is ImportSessionState.CANCELLED:
            self._receipts.resume_cancelled(
                approved,
                batch_size=self._batch_size,
            )
        elif snapshot.state is ImportSessionState.NEEDS_ATTENTION:
            self._reset_retryable_rows(approved.approval_id, snapshot)
            self._receipts.transition_session(
                approved.approval_id,
                ImportSessionState.RUNNING,
            )
        elif snapshot.state not in {
            ImportSessionState.PENDING,
            ImportSessionState.RUNNING,
            ImportSessionState.COMPLETED,
        }:
            raise ImportReceiptTransitionError(
                "The receipt session cannot be retried from its current state."
            )
        return self.execute(
            approved,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            _recovering_interruption=recovering_interruption,
        )

    def _reset_retryable_rows(
        self,
        approval_id: str,
        snapshot: ImportSessionSnapshot,
    ) -> None:
        for item in snapshot.items:
            if item.outcome is ImportItemOutcome.FAILED and item.retryable:
                self._receipts.reset_retryable_item(
                    approval_id,
                    item_id=item.item_id,
                )
        refreshed = self._receipts.load_session_snapshot(approval_id)
        pending_item_ids = {
            item.item_id
            for item in refreshed.items
            if item.outcome is ImportItemOutcome.PENDING
        }
        for effect in (*refreshed.payload_effects, *refreshed.membership_effects):
            if (
                effect.item_id in pending_item_ids
                and effect.state is ImportEffectState.FAILED
                and effect.retryable
            ):
                self._receipts.reset_retryable_effect(
                    approval_id,
                    category=effect.category,
                    effect_id=effect.effect_id,
                )
        refreshed = self._receipts.load_session_snapshot(approval_id)
        for effect in reversed(refreshed.folder_effects):
            if effect.state is ImportEffectState.FAILED and effect.retryable:
                self._receipts.reset_retryable_effect(
                    approval_id,
                    category=effect.category,
                    effect_id=effect.effect_id,
                )

    def _cancel(
        self,
        approval_id: str,
        progress_callback: Callable[[ImportExecutionProgress], None] | None,
    ) -> ImportExecutionReceipt:
        self._receipts.transition_session(approval_id, ImportSessionState.CANCELLED)
        receipt = self._receipts.aggregate_receipt(approval_id)
        _publish_receipt_progress(receipt, progress_callback)
        return receipt

    def _publish_progress(
        self,
        approval_id: str,
        progress_callback: Callable[[ImportExecutionProgress], None] | None,
    ) -> None:
        if progress_callback is None:
            return
        _publish_receipt_progress(
            self._receipts.aggregate_receipt(approval_id),
            progress_callback,
        )

    def _execute_folders(
        self,
        approved: ApprovedNoteImportPlan,
        folder_effects: tuple[ImportEffectRecord, ...],
        *,
        cancel_event: threading.Event | None,
        progress_callback: Callable[[ImportExecutionProgress], None] | None,
    ) -> tuple[
        dict[str, str],
        dict[tuple[str, ...], _ExecutionFailure],
        bool,
    ]:
        ordered_paths = _required_folder_paths(approved)
        effects_by_digest = {
            effect.folder_path_digest: effect for effect in folder_effects
        }
        if None in effects_by_digest or len(effects_by_digest) != len(folder_effects):
            raise ImportReceiptTransitionError(
                "Folder receipt authority does not match the approved plan."
            )
        if {_folder_path_digest(path) for path in ordered_paths} != set(
            effects_by_digest
        ):
            raise ImportReceiptTransitionError(
                "Folder receipt authority does not match the approved plan."
            )

        bindings: dict[str, str] = {}
        failures: dict[tuple[str, ...], _ExecutionFailure] = {}
        normalized_owners: dict[str, tuple[str, ...]] = {}
        for batch_start in range(0, len(ordered_paths), self._batch_size):
            if cancel_event is not None and cancel_event.is_set():
                return bindings, failures, True
            batch = ordered_paths[batch_start : batch_start + self._batch_size]
            for path in batch:
                effect = effects_by_digest[_folder_path_digest(path)]
                inherited = _first_folder_failure(path, failures)
                if inherited is not None:
                    self._fail_effect(approved.approval_id, effect, inherited)
                    failures[path] = inherited
                    continue
                normalized_path = _normalized_folder_path(path)
                if normalized_path in normalized_owners:
                    failure = _ExecutionFailure("folder_conflict", False)
                    self._fail_effect(approved.approval_id, effect, failure)
                    failures[path] = failure
                    continue
                if effect.state is ImportEffectState.APPLIED:
                    if effect.target_folder_id is None:
                        raise ImportReceiptTransitionError(
                            "Applied folder receipt is missing its target identity."
                        )
                    bindings[_folder_path_digest(path)] = effect.target_folder_id
                    normalized_owners[normalized_path] = path
                    continue
                if effect.state is ImportEffectState.FAILED:
                    failure = _failure_from_effect(effect)
                    failures[path] = failure
                    continue
                deterministic_id = _deterministic_folder_id(
                    approved.approval_id,
                    normalized_path,
                )
                try:
                    folder = self._target.ensure_folder(
                        segments=path,
                        folder_id=deterministic_id,
                        allow_existing=_allows_existing_root(approved, path),
                    )
                except ImportTargetInternalError:
                    raise
                except ImportTargetError as error:
                    failure = _failure_for_target_error(error, folder=True)
                    self._fail_effect(approved.approval_id, effect, failure)
                    failures[path] = failure
                    continue
                self._receipts.transition_effects(
                    approved.approval_id,
                    (
                        EffectTransition(
                            category=ImportEffectCategory.FOLDER,
                            effect_id=effect.effect_id,
                            state=ImportEffectState.APPLIED,
                            target_folder_id=folder.folder_id,
                        ),
                    ),
                )
                bindings[_folder_path_digest(path)] = folder.folder_id
                normalized_owners[normalized_path] = path
            self._publish_progress(approved.approval_id, progress_callback)
            if cancel_event is not None and cancel_event.is_set():
                return bindings, failures, True
        return bindings, failures, False

    def _execute_item(
        self,
        approved: ApprovedNoteImportPlan,
        *,
        item: ImportPreviewItem,
        payload_effects: dict[int | None, ImportEffectRecord],
        membership_effects: tuple[ImportEffectRecord, ...],
        folder_bindings: dict[str, str],
        folder_failures: dict[tuple[str, ...], _ExecutionFailure],
        recovering_interruption: bool,
    ) -> None:
        if item.selected_action is ImportAction.SKIP:
            self._receipts.transition_item(
                approved.approval_id,
                item.item_id,
                ImportItemOutcome.SKIPPED,
            )
            return

        if len(membership_effects) != len(item.memberships):
            raise ImportReceiptTransitionError(
                "Membership receipt authority does not match the approved plan."
            )
        memberships = tuple(zip(item.memberships, membership_effects, strict=True))
        memberships_by_payload: dict[
            int, list[tuple[ProposedFolderMembership, ImportEffectRecord]]
        ] = {}
        for membership_pair in memberships:
            memberships_by_payload.setdefault(
                membership_pair[0].payload_index,
                [],
            ).append(membership_pair)
        failures: list[_ExecutionFailure] = []
        observed_version: int | None = None

        if item.selected_action is ImportAction.CREATE_NEW:
            for payload_index, payload in enumerate(item.payloads):
                effect = payload_effects.get(payload_index)
                if effect is None:
                    raise ImportReceiptTransitionError(
                        "Payload receipt authority does not match the approved plan."
                    )
                note_id = _deterministic_note_id(
                    approved.approval_id,
                    item.item_id,
                    payload_index,
                )
                unit_memberships = tuple(memberships_by_payload.get(payload_index, ()))
                if effect.state is ImportEffectState.FAILED:
                    failures.append(_failure_from_effect(effect))
                    continue
                blocked = _membership_folder_failure(unit_memberships, folder_failures)
                if blocked is not None:
                    transitions = [
                        _failed_effect_transition(
                            effect,
                            blocked,
                            target_note_id=note_id,
                        )
                    ]
                    for membership, membership_effect in unit_memberships:
                        membership_failure = _membership_folder_failure(
                            ((membership, membership_effect),),
                            folder_failures,
                        )
                        transitions.append(
                            _failed_effect_transition(
                                membership_effect,
                                membership_failure or blocked,
                            )
                        )
                    self._receipts.transition_effects(
                        approved.approval_id,
                        tuple(transitions),
                    )
                    failures.append(blocked)
                    continue
                if effect.state is ImportEffectState.APPLIED:
                    if effect.reason_code == "note_conflict":
                        failure = self._record_applied_payload_conflict(
                            approved,
                            effect=effect,
                            note_id=note_id,
                            memberships=unit_memberships,
                            folder_bindings=folder_bindings,
                        )
                        failures.append(failure)
                        continue
                    if effect.reason_code is not None or effect.retryable:
                        raise ImportReceiptTransitionError(
                            "Applied payload reconciliation metadata is invalid."
                        )
                    note = self._target.read_note(note_id=note_id)
                    if (
                        note is None
                        or effect.target_note_id != note_id
                        or note.version != effect.observed_version
                        or not _note_matches(note, payload)
                    ):
                        failure = self._record_applied_payload_conflict(
                            approved,
                            effect=effect,
                            note_id=note_id,
                            memberships=unit_memberships,
                            folder_bindings=folder_bindings,
                        )
                        failures.append(failure)
                        continue
                else:
                    try:
                        note = self._target.create_note(
                            note_id=note_id, payload=payload
                        )
                    except ImportTargetInternalError:
                        raise
                    except ImportTargetError as error:
                        failure = _failure_for_target_error(
                            error,
                            folder=False,
                            create=True,
                        )
                        self._fail_effect(
                            approved.approval_id,
                            effect,
                            failure,
                            target_note_id=note_id,
                        )
                        failures.append(failure)
                        continue
                    self._apply_payload_effect(
                        approved.approval_id,
                        effect,
                        note_id=note.note_id,
                        observed_version=note.version,
                    )
                membership_failure = self._execute_memberships(
                    approved,
                    note_id=note.note_id,
                    expected_note_version=note.version,
                    memberships=unit_memberships,
                    folder_bindings=folder_bindings,
                )
                if membership_failure is not None:
                    failures.append(membership_failure)
        else:
            if item.match is None or item.match.note_version is None:
                raise ImportReceiptTransitionError(
                    "Update execution requires approved target authority."
                )
            note_id = item.match.note_id
            executable_memberships: list[
                tuple[ProposedFolderMembership, ImportEffectRecord]
            ] = []
            blocked_transitions: list[EffectTransition] = []
            for membership, membership_effect in memberships:
                blocked = _membership_folder_failure(
                    ((membership, membership_effect),),
                    folder_failures,
                )
                if blocked is None:
                    executable_memberships.append((membership, membership_effect))
                    continue
                blocked_transitions.append(
                    _failed_effect_transition(membership_effect, blocked)
                )
                failures.append(blocked)
            if blocked_transitions:
                self._receipts.transition_effects(
                    approved.approval_id,
                    tuple(blocked_transitions),
                )
            note_operation_failed = False
            try:
                if item.replace_content:
                    effect = payload_effects.get(0)
                    if effect is None:
                        raise ImportReceiptTransitionError(
                            "Payload receipt authority does not match the approved plan."
                        )
                    if effect.state is ImportEffectState.FAILED:
                        failures.append(_failure_from_effect(effect))
                        note_operation_failed = True
                        note = None
                    elif effect.state is ImportEffectState.APPLIED:
                        if effect.reason_code == "note_conflict":
                            failure = self._record_applied_payload_conflict(
                                approved,
                                effect=effect,
                                note_id=note_id,
                                memberships=tuple(executable_memberships),
                                folder_bindings=folder_bindings,
                            )
                            failures.append(failure)
                            note_operation_failed = True
                            note = None
                        elif effect.reason_code is not None or effect.retryable:
                            raise ImportReceiptTransitionError(
                                "Applied payload reconciliation metadata is invalid."
                            )
                        else:
                            note = self._target.read_note(note_id=note_id)
                            if (
                                note is None
                                or effect.target_note_id != note_id
                                or note.version != effect.observed_version
                                or not _note_matches(note, item.payloads[0])
                            ):
                                failure = self._record_applied_payload_conflict(
                                    approved,
                                    effect=effect,
                                    note_id=note_id,
                                    memberships=tuple(executable_memberships),
                                    folder_bindings=folder_bindings,
                                )
                                failures.append(failure)
                                note_operation_failed = True
                                note = None
                    else:
                        note = self._target.replace_note(
                            note_id=note_id,
                            expected_version=item.match.note_version,
                            payload=item.payloads[0],
                        )
                        self._apply_payload_effect(
                            approved.approval_id,
                            effect,
                            note_id=note.note_id,
                            observed_version=note.version,
                        )
                else:
                    note = self._target.read_note(note_id=note_id)
                    if note is None or note.version != item.match.note_version:
                        raise ImportTargetConflictError
                if note is not None:
                    observed_version = note.version
            except ImportTargetInternalError:
                raise
            except ImportTargetError as error:
                note_operation_failed = True
                failure = (
                    _ExecutionFailure("note_conflict", False)
                    if (
                        recovering_interruption
                        and item.replace_content
                        and isinstance(error, ImportTargetConflictError)
                    )
                    else _failure_for_target_error(error, folder=False)
                )
                effect = payload_effects.get(0)
                if effect is not None:
                    self._fail_effect(
                        approved.approval_id,
                        effect,
                        failure,
                        target_note_id=note_id,
                    )
                else:
                    for membership, membership_effect in executable_memberships:
                        folder_id = folder_bindings.get(
                            _folder_path_digest(tuple(membership.folder_segments))
                        )
                        if folder_id is None:
                            raise ImportReceiptTransitionError(
                                "Membership folder authority is not durably applied."
                            )
                        self._fail_effect(
                            approved.approval_id,
                            membership_effect,
                            failure,
                            target_note_id=note_id,
                            target_folder_id=folder_id,
                        )
                failures.append(failure)
            if not note_operation_failed:
                membership_failure = self._execute_memberships(
                    approved,
                    note_id=note_id,
                    expected_note_version=note.version,
                    memberships=tuple(executable_memberships),
                    folder_bindings=folder_bindings,
                )
                if membership_failure is not None:
                    failures.append(membership_failure)

        if failures:
            failure = _summarize_failures(failures)
            self._receipts.transition_item(
                approved.approval_id,
                item.item_id,
                ImportItemOutcome.FAILED,
                reason_code=failure.reason_code,
                retryable=failure.retryable,
            )
            return
        outcome = (
            ImportItemOutcome.IMPORTED
            if item.selected_action is ImportAction.CREATE_NEW
            else ImportItemOutcome.UPDATED
        )
        self._receipts.transition_item(
            approved.approval_id,
            item.item_id,
            outcome,
            observed_version=(
                observed_version
                if item.selected_action is ImportAction.UPDATE_EXISTING
                else None
            ),
        )

    def _record_applied_payload_conflict(
        self,
        approved: ApprovedNoteImportPlan,
        *,
        effect: ImportEffectRecord,
        note_id: str,
        memberships: tuple[tuple[ProposedFolderMembership, ImportEffectRecord], ...],
        folder_bindings: dict[str, str],
    ) -> _ExecutionFailure:
        failure = _ExecutionFailure("note_conflict", False)
        if effect.reason_code is None:
            self._receipts.annotate_applied_payload_reconciliation_conflict(
                approved.approval_id,
                effect_id=effect.effect_id,
            )
        elif effect.reason_code != "note_conflict" or effect.retryable:
            raise ImportReceiptTransitionError(
                "Applied payload reconciliation metadata is invalid."
            )
        transitions: list[EffectTransition] = []
        for membership, membership_effect in memberships:
            if membership_effect.state is not ImportEffectState.PENDING:
                continue
            folder_id = folder_bindings.get(
                _folder_path_digest(tuple(membership.folder_segments))
            )
            if folder_id is None:
                raise ImportReceiptTransitionError(
                    "Membership folder authority is not durably applied."
                )
            transitions.append(
                _failed_effect_transition(
                    membership_effect,
                    failure,
                    target_note_id=note_id,
                    target_folder_id=folder_id,
                )
            )
        if transitions:
            self._receipts.transition_effects(
                approved.approval_id,
                tuple(transitions),
            )
        return failure

    def _execute_memberships(
        self,
        approved: ApprovedNoteImportPlan,
        *,
        note_id: str,
        expected_note_version: int,
        memberships: tuple[tuple[ProposedFolderMembership, ImportEffectRecord], ...],
        folder_bindings: dict[str, str],
    ) -> _ExecutionFailure | None:
        failures: list[_ExecutionFailure] = []
        for membership, effect in memberships:
            path_digest = _folder_path_digest(tuple(membership.folder_segments))
            folder_id = folder_bindings.get(path_digest)
            if folder_id is None:
                raise ImportReceiptTransitionError(
                    "Membership folder authority is not durably applied."
                )
            if effect.state is ImportEffectState.APPLIED:
                if (
                    effect.target_note_id != note_id
                    or effect.target_folder_id != folder_id
                ):
                    raise ImportReceiptTransitionError(
                        "Applied membership receipt has conflicting authority."
                    )
                continue
            if effect.state is ImportEffectState.FAILED:
                failures.append(_failure_from_effect(effect))
                continue
            try:
                self._target.attach_membership(
                    folder_id=folder_id,
                    note_id=note_id,
                    expected_note_version=expected_note_version,
                )
            except ImportTargetInternalError:
                raise
            except ImportTargetError as error:
                failure = _failure_for_target_error(error, folder=False)
                self._fail_effect(
                    approved.approval_id,
                    effect,
                    failure,
                    target_note_id=note_id,
                    target_folder_id=folder_id,
                )
                failures.append(failure)
                continue
            self._receipts.transition_effects(
                approved.approval_id,
                (
                    EffectTransition(
                        category=ImportEffectCategory.MEMBERSHIP,
                        effect_id=effect.effect_id,
                        state=ImportEffectState.APPLIED,
                        target_note_id=note_id,
                        target_folder_id=folder_id,
                    ),
                ),
            )
        return _summarize_failures(failures) if failures else None

    def _apply_payload_effect(
        self,
        approval_id: str,
        effect: ImportEffectRecord,
        *,
        note_id: str,
        observed_version: int,
    ) -> None:
        self._receipts.transition_effects(
            approval_id,
            (
                EffectTransition(
                    category=ImportEffectCategory.PAYLOAD,
                    effect_id=effect.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id=note_id,
                    observed_version=observed_version,
                ),
            ),
        )

    def _fail_effect(
        self,
        approval_id: str,
        effect: ImportEffectRecord,
        failure: _ExecutionFailure,
        *,
        target_note_id: str | None = None,
        target_folder_id: str | None = None,
    ) -> None:
        self._receipts.transition_effects(
            approval_id,
            (
                _failed_effect_transition(
                    effect,
                    failure,
                    target_note_id=target_note_id,
                    target_folder_id=target_folder_id,
                ),
            ),
        )


def _required_folder_paths(
    approved: ApprovedNoteImportPlan,
) -> tuple[tuple[str, ...], ...]:
    required: set[tuple[str, ...]] = set()
    for item in approved.plan.items:
        if item.selected_action is ImportAction.SKIP or not item.add_membership:
            continue
        for membership in item.memberships:
            path = tuple(membership.folder_segments)
            required.update(path[:depth] for depth in range(1, len(path) + 1))
    proposed_ordinals = {
        path: ordinal
        for ordinal, path in enumerate(approved.plan.proposed_folder_paths)
    }
    if required.difference(proposed_ordinals):
        raise ImportReceiptTransitionError(
            "The approved plan is missing required folder authority."
        )
    return tuple(
        sorted(required, key=lambda path: (len(path), proposed_ordinals[path]))
    )


def _normalized_folder_path(path: tuple[str, ...]) -> str:
    normalized_path = ""
    for segment in path:
        normalized_path = join_normalized_folder_path(
            normalized_path,
            normalize_folder_name(segment).key,
        )
    return normalized_path


def _deterministic_folder_id(approval_id: str, normalized_path: str) -> str:
    return str(uuid5(UUID(approval_id), f"folder:{normalized_path}"))


def _deterministic_note_id(
    approval_id: str,
    item_id: str,
    payload_index: int,
) -> str:
    return str(uuid5(UUID(approval_id), f"note:{item_id}:{payload_index}"))


def _allows_existing_root(
    approved: ApprovedNoteImportPlan,
    path: tuple[str, ...],
) -> bool:
    collision = approved.plan.root_collision
    return bool(
        len(path) == 1
        and collision is not None
        and collision.collides
        and collision.choice is RootCollisionChoice.USE_EXISTING
        and path[0] == collision.proposed_label
    )


def _first_folder_failure(
    path: tuple[str, ...],
    failures: dict[tuple[str, ...], _ExecutionFailure],
) -> _ExecutionFailure | None:
    for depth in range(1, len(path)):
        failure = failures.get(path[:depth])
        if failure is not None:
            return failure
    return None


def _membership_folder_failure(
    memberships: tuple[tuple[ProposedFolderMembership, ImportEffectRecord], ...],
    folder_failures: dict[tuple[str, ...], _ExecutionFailure],
) -> _ExecutionFailure | None:
    failures: list[_ExecutionFailure] = []
    for membership, _effect in memberships:
        path = tuple(membership.folder_segments)
        for depth in range(1, len(path) + 1):
            failure = folder_failures.get(path[:depth])
            if failure is not None:
                failures.append(failure)
    return _summarize_failures(failures) if failures else None


def _failed_effect_transition(
    effect: ImportEffectRecord,
    failure: _ExecutionFailure,
    *,
    target_note_id: str | None = None,
    target_folder_id: str | None = None,
) -> EffectTransition:
    return EffectTransition(
        category=effect.category,
        effect_id=effect.effect_id,
        state=ImportEffectState.FAILED,
        reason_code=failure.reason_code,
        retryable=failure.retryable,
        target_note_id=target_note_id,
        target_folder_id=target_folder_id,
    )


def _summarize_failures(failures: list[_ExecutionFailure]) -> _ExecutionFailure:
    return next(
        (failure for failure in failures if failure.retryable),
        failures[0],
    )


def _failure_for_target_error(
    error: ImportTargetError,
    *,
    folder: bool,
    create: bool = False,
) -> _ExecutionFailure:
    if isinstance(error, ImportTargetRetryableError):
        return _ExecutionFailure("database_busy", True)
    if isinstance(error, ImportTargetConflictError):
        return _ExecutionFailure(
            (
                "folder_conflict"
                if folder
                else "note_conflict"
                if create
                else "version_conflict"
            ),
            False,
        )
    if isinstance(error, ImportTargetPermanentError):
        return _ExecutionFailure("target_invalid", False)
    return _ExecutionFailure("target_failure", False)


def _failure_from_effect(effect: ImportEffectRecord) -> _ExecutionFailure:
    if effect.state is not ImportEffectState.FAILED or effect.reason_code is None:
        raise ImportReceiptTransitionError(
            "Failed effect receipt is missing its bounded failure metadata."
        )
    return _ExecutionFailure(effect.reason_code, effect.retryable)


def _publish_receipt_progress(
    receipt: ImportExecutionReceipt,
    callback: Callable[[ImportExecutionProgress], None] | None,
) -> None:
    if callback is None:
        return
    callback(
        ImportExecutionProgress(
            state=receipt.state,
            total=receipt.total,
            completed=receipt.completed,
            imported=receipt.imported,
            updated=receipt.updated,
            skipped=receipt.skipped,
            failed=receipt.failed,
            retryable=receipt.retryable,
            reason_code=receipt.reason_code,
        )
    )


__all__ = [
    "ImportTargetConflictError",
    "ImportTargetError",
    "ImportTargetInternalError",
    "ImportTargetPermanentError",
    "ImportTargetRetryableError",
    "LocalNoteImportTarget",
    "LocalTargetFolder",
    "LocalTargetNote",
    "NoteImportExecutor",
]
