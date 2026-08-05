"""Serialized lifecycle owner for the local TTS generation-profile store."""

from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import stat
import threading
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast
from unicodedata import category as _unicode_category
from unicodedata import normalize as _unicode_normalize
from uuid import UUID, uuid4

from tldw_chatbook.DB.private_sqlite import (
    backup_connection_to_private,
    backup_open_connections_to_private,
    connect_private_sqlite,
    copy_private_sqlite,
)
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_schema import (
    ASSIGNED_PROFILE_JOIN_SELECT,
    decode_assigned_snapshot,
    decode_assignment,
    decode_profile,
    decode_utc_datetime,
    encode_assignment,
    encode_profile,
    encode_uuid,
    open_profile_store,
    validate_profile_candidate,
    validate_profile_store_rows,
)
from tldw_chatbook.TTS.profile_store_lock import (
    ProfileStoreLease,
    ProfileStoreLockMode,
)
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    FrozenJsonOptions,
    ProfileBackupReceipt,
    ProfileRepositoryState,
    ProfileRestoreReceipt,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileCollisionSnapshot,
    TTSProfileDraft,
    TTSProfilePage,
)
from tldw_chatbook.Utils.path_validation import validate_path_simple


_T = TypeVar("_T")
_PATH_TYPE = type(Path())
_CHARACTER_REF_TYPE = CharacterRef
_TTS_GENERATION_PROFILE_TYPE = TTSGenerationProfile
_TTS_PROFILE_DRAFT_TYPE = TTSProfileDraft
_MAX_SEARCH_CHARACTERS = 128
_MAX_NORMALIZED_SEARCH_CHARACTERS = 512
_MAX_NORMALIZED_SEARCH_BYTES = 2_048
_UNSAFE_SEARCH_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_unicode_ord = ord
_monotonic = time.monotonic
# SQLite extended result codes are ABI-stable.  Keeping the exact values here
# also supports Python builds that do not expose every named sqlite3 constant.
_SQLITE_CONSTRAINT_FOREIGNKEY = 787
_SQLITE_CONSTRAINT_PRIMARYKEY = 1_555
_SQLITE_CONSTRAINT_TRIGGER = 1_811
_SQLITE_CONSTRAINT_UNIQUE = 2_067
_STORE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_INITIALIZATION_LOCK_TIMEOUT_SECONDS = 0.1
_RESTORE_BACKUP_PAGE_BATCH = 64
_RESTORE_PROGRESS_OPCODE_INTERVAL = 1_000
_RESTORE_REBIND_TIMEOUT_SECONDS = 5.0
_TransactionOperation = Literal[
    "create",
    "read",
    "update",
    "delete",
    "assignment_set",
    "assignment_remove",
]
_PROFILE_SELECT = """
SELECT
    profile_id,
    display_name,
    normalized_name,
    provider_id,
    model_id,
    voice_id,
    response_format,
    speed,
    options_json,
    revision,
    created_at,
    updated_at
FROM tts_generation_profiles
"""
_ASSIGNMENT_SELECT = """
SELECT
    source,
    authority_id,
    character_id,
    profile_id,
    created_at,
    updated_at
FROM character_tts_assignments
"""


@dataclass(frozen=True, slots=True)
class _OperationAdmission(Generic[_T]):
    """One generation-bound worker submission awaiting publication."""

    generation: int
    future: Future[_T]


@dataclass(slots=True)
class _IntegrityEvidence:
    """Exact schema-owned values and statement error for one mutation."""

    profile_id: UUID | None
    normalized_name: str | None = None
    statement_error: sqlite3.IntegrityError | None = None


@dataclass(frozen=True, slots=True)
class _PersistedAssignment:
    """One fully decoded assignment row including persistence timestamps."""

    assignment: CharacterTTSAssignment
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class _DestinationSnapshot:
    """One canonical backup destination admitted before worker submission."""

    path: Path
    parent_identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _CandidateSnapshot:
    """One exact standalone restore candidate and its admission identity."""

    path: Path
    identity: tuple[int, int, int, int, int, int]


def _repository_error(code: str) -> ProfileRepositoryError:
    return ProfileRepositoryError(code)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_exact_profile_id(value: object) -> UUID:
    if type(value) is not UUID:
        raise _repository_error("operation_failed")
    profile_id = cast(UUID, value)
    validation_error: BaseException | None = None
    validated: UUID | None = None
    try:
        validated = UUID(str(profile_id))
        if validated != profile_id:
            raise ValueError
    except BaseException as error:
        validation_error = error
    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("operation_failed")
    assert validated is not None
    return validated


def _validate_optional_profile_id(value: object) -> UUID | None:
    if value is None:
        return None
    return _validate_exact_profile_id(value)


def _validate_optional_profile(
    value: object,
) -> TTSGenerationProfile | None:
    """Return an exact canonical profile snapshot or reject the boundary."""

    if value is None:
        return None
    if type(value) is not _TTS_GENERATION_PROFILE_TYPE:
        raise _repository_error("operation_failed")
    profile = cast(TTSGenerationProfile, value)
    validation_error: BaseException | None = None
    validated: TTSGenerationProfile | None = None
    try:
        validated = TTSGenerationProfile(
            profile_id=profile.profile_id,
            display_name=profile.display_name,
            normalized_name=profile.normalized_name,
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
            revision=profile.revision,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )
        if validated != profile:
            raise ValueError
    except BaseException as error:
        validation_error = error
    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("operation_failed")
    assert validated is not None
    return validated


def _validate_draft(value: object) -> TTSProfileDraft:
    if type(value) is not _TTS_PROFILE_DRAFT_TYPE:
        raise _repository_error("operation_failed")
    draft = cast(TTSProfileDraft, value)
    validation_error: BaseException | None = None
    validated: TTSProfileDraft | None = None
    try:
        validated = TTSProfileDraft(
            display_name=draft.display_name,
            provider_id=draft.provider_id,
            model_id=draft.model_id,
            voice_id=draft.voice_id,
            response_format=draft.response_format,
            speed=draft.speed,
            options=draft.options,
        )
        if validated != draft:
            raise ValueError
    except BaseException as error:
        validation_error = error
    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("operation_failed")
    assert validated is not None
    return validated


def _validate_expected_revision(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise _repository_error("operation_failed")
    return cast(int, value)


def _validate_expected_generation(value: object) -> int:
    if type(value) is not int or value < 0:
        raise _repository_error("operation_failed")
    return cast(int, value)


def _validate_character_ref(value: object) -> CharacterRef:
    if type(value) is not _CHARACTER_REF_TYPE:
        raise _repository_error("operation_failed")
    character_ref = cast(CharacterRef, value)
    validation_error: BaseException | None = None
    validated: CharacterRef | None = None
    try:
        validated = CharacterRef(
            source=character_ref.source,
            authority_id=character_ref.authority_id,
            character_id=character_ref.character_id,
        )
        if validated != character_ref:
            raise ValueError
    except BaseException as error:
        validation_error = error
    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("operation_failed")
    assert validated is not None
    return validated


def _is_unsafe_search_character(character: str) -> bool:
    category = _unicode_category(character)
    code_point = _unicode_ord(character)
    if type(category) is not str or type(code_point) is not int:
        raise ValueError
    return (
        category in _UNSAFE_SEARCH_CATEGORIES
        or 0xFDD0 <= code_point <= 0xFDEF
        or code_point & 0xFFFF in (0xFFFE, 0xFFFF)
    )


def _normalize_search(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or len(value) > _MAX_SEARCH_CHARACTERS:
        raise _repository_error("operation_failed")

    processing_error: BaseException | None = None
    raw_unsafe = False
    normalized_unsafe = False
    trimmed = ""
    normalized: str | None = None
    normalized_byte_count: int | None = None
    try:
        raw_unsafe = any(_is_unsafe_search_character(character) for character in value)
        if not raw_unsafe:
            trimmed = value.strip()
            if trimmed:
                normalized_value = _unicode_normalize("NFKC", trimmed)
                if type(normalized_value) is not str:
                    raise ValueError
                normalized = normalized_value.casefold()
                if type(normalized) is not str:
                    raise ValueError
                normalized_unsafe = any(
                    _is_unsafe_search_character(character) for character in normalized
                )
                if not normalized_unsafe:
                    normalized_byte_count = len(normalized.encode("utf-8"))
    except BaseException as error:
        processing_error = error

    if processing_error is not None:
        if not isinstance(processing_error, Exception):
            raise processing_error
        raise _repository_error("operation_failed")
    if raw_unsafe or normalized_unsafe:
        raise _repository_error("operation_failed")
    if not trimmed:
        return None
    assert normalized is not None
    assert normalized_byte_count is not None
    if (
        len(normalized) > _MAX_NORMALIZED_SEARCH_CHARACTERS
        or normalized_byte_count > _MAX_NORMALIZED_SEARCH_BYTES
    ):
        raise _repository_error("operation_failed")
    return normalized


def _validate_page_limit(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 100:
        raise _repository_error("operation_failed")
    return cast(int, value)


def _validate_page_offset(value: object) -> int:
    if type(value) is not int or value < 0:
        raise _repository_error("operation_failed")
    return cast(int, value)


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _full_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_database_path(database_path: Path, failure_code: str) -> Path:
    """Resolve one configured database path without creating filesystem state."""

    resolution_error: BaseException | None = None
    resolved: Path | None = None
    try:
        resolved = database_path.resolve(strict=False)
    except BaseException as error:
        resolution_error = error
    if resolution_error is not None:
        if not isinstance(resolution_error, Exception):
            raise resolution_error
        raise _repository_error(failure_code)
    if type(resolved) is not _PATH_TYPE or not resolved.is_absolute():
        raise _repository_error(failure_code)
    return resolved


def _reserved_store_paths(database_path: Path) -> tuple[Path, ...]:
    return (
        database_path,
        database_path.with_name(f"{database_path.name}.lock"),
        *(
            database_path.with_name(f"{database_path.name}{suffix}")
            for suffix in _STORE_SIDECAR_SUFFIXES
        ),
    )


def _validate_backup_destination(
    destination: object,
    database_path: Path,
) -> _DestinationSnapshot:
    """Validate and canonicalize one safe publication target."""

    if type(destination) is not _PATH_TYPE:
        raise _repository_error("backup_failed")

    validation_error: BaseException | None = None
    snapshot: _DestinationSnapshot | None = None
    try:
        exact_destination = cast(Path, destination)
        validate_path_simple(exact_destination, require_exists=False)
        if os.path.lexists(exact_destination) and exact_destination.is_symlink():
            raise ValueError
        resolved_destination = exact_destination.resolve(strict=False)
        parent = resolved_destination.parent.resolve(strict=True)
        parent_state = parent.stat()
        if not stat.S_ISDIR(parent_state.st_mode):
            raise ValueError

        reserved = _reserved_store_paths(database_path)
        if resolved_destination in reserved:
            raise ValueError
        if os.path.lexists(resolved_destination):
            destination_state = resolved_destination.stat()
            if not stat.S_ISREG(destination_state.st_mode):
                raise ValueError
            destination_identity = _stat_identity(destination_state)
            for reserved_path in reserved:
                if not os.path.lexists(reserved_path):
                    continue
                if _stat_identity(reserved_path.stat()) == destination_identity:
                    raise ValueError
        snapshot = _DestinationSnapshot(
            path=resolved_destination,
            parent_identity=_stat_identity(parent_state),
        )
    except BaseException as error:
        validation_error = error

    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("backup_failed")
    assert snapshot is not None
    return snapshot


def _validate_restore_timeout(value: object) -> float:
    if type(value) not in (int, float):
        raise _repository_error("restore_failed")
    try:
        normalized = float(cast(int | float, value))
    except Exception:
        raise _repository_error("restore_failed") from None
    if not math.isfinite(normalized) or normalized <= 0:
        raise _repository_error("restore_failed")
    return normalized


def _validate_restore_candidate_path(
    candidate: object,
    database_path: Path,
) -> _CandidateSnapshot:
    """Validate one exact non-store regular-file identity without mutation."""

    if type(candidate) is not _PATH_TYPE:
        raise _repository_error("restore_failed")

    validation_error: BaseException | None = None
    snapshot: _CandidateSnapshot | None = None
    try:
        exact_candidate = cast(Path, candidate)
        validate_path_simple(exact_candidate, require_exists=True)
        if exact_candidate.is_symlink():
            raise ValueError
        resolved_candidate = exact_candidate.resolve(strict=True)
        candidate_state = resolved_candidate.stat()
        if not stat.S_ISREG(candidate_state.st_mode):
            raise ValueError
        reserved = _reserved_store_paths(database_path)
        if resolved_candidate in reserved:
            raise ValueError
        candidate_identity = _stat_identity(candidate_state)
        for reserved_path in reserved:
            if not os.path.lexists(reserved_path):
                continue
            if _stat_identity(reserved_path.stat()) == candidate_identity:
                raise ValueError
        snapshot = _CandidateSnapshot(
            path=resolved_candidate,
            identity=_full_stat_identity(candidate_state),
        )
    except BaseException as error:
        validation_error = error

    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("restore_failed")
    assert snapshot is not None
    return snapshot


def _candidate_is_unchanged(candidate: _CandidateSnapshot) -> bool:
    try:
        return (
            not candidate.path.is_symlink()
            and _full_stat_identity(candidate.path.stat()) == candidate.identity
        )
    except Exception:
        return False


def _read_monotonic() -> float:
    timing_error: BaseException | None = None
    value: object = None
    try:
        value = _monotonic()
    except BaseException as error:
        timing_error = error
    if timing_error is not None:
        if not isinstance(timing_error, Exception):
            raise timing_error
        raise _repository_error("restore_failed")
    if type(value) not in (int, float):
        raise _repository_error("restore_failed")
    normalized = float(cast(int | float, value))
    if not math.isfinite(normalized):
        raise _repository_error("restore_failed")
    return normalized


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - _read_monotonic()
    if not math.isfinite(remaining):
        raise _repository_error("restore_failed")
    return remaining


def _require_restore_time(deadline: float) -> None:
    """Require positive remaining restore time at one worker boundary."""

    if _remaining_seconds(deadline) <= 0:
        raise _repository_error("restore_failed")


def _run_with_restore_progress(
    connection: sqlite3.Connection,
    deadline: float,
    operation: Callable[[], _T],
) -> _T:
    """Run SQLite work with one deadline-aware VM progress handler."""

    callback_error: BaseException | None = None
    body_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    progress_installed = False
    result: _T | None = None

    def interrupt_after_deadline() -> int:
        nonlocal callback_error
        try:
            _require_restore_time(deadline)
        except BaseException as error:
            callback_error = error
            return 1
        return 0

    try:
        _require_restore_time(deadline)
        connection.set_progress_handler(
            interrupt_after_deadline,
            _RESTORE_PROGRESS_OPCODE_INTERVAL,
        )
        progress_installed = True
        result = operation()
        _require_restore_time(deadline)
    except BaseException as error:
        body_error = error

    if progress_installed:
        try:
            connection.set_progress_handler(None, 0)
        except BaseException as error:
            cleanup_error = error

    if callback_error is not None:
        body_error = callback_error
    _raise_with_cleanup_precedence(body_error, cleanup_error)
    return cast(_T, result)


def _unlink_path_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _escape_like_literal(value: str) -> str:
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _read_sqlite_errorcode(error: sqlite3.IntegrityError) -> int | None:
    """Read one exact extended result code without inspecting error text."""

    code: object = None
    code_error: BaseException | None = None
    try:
        code = error.sqlite_errorcode
    except BaseException as caught:
        code_error = caught

    if code_error is not None:
        if not isinstance(code_error, Exception):
            raise code_error
        return None
    if type(code) is not int:
        return None
    return cast(int, code)


def _profile_conflict_evidence(
    connection: sqlite3.Connection,
    operation_kind: _TransactionOperation,
    profile_id: UUID | None,
    normalized_name: str | None,
) -> bool | None:
    """Check exact create/update conflict rows inside the failed transaction."""

    inspection_error: BaseException | None = None
    conflict_exists: bool | None = None
    try:
        if type(profile_id) is not UUID or type(normalized_name) is not str:
            raise ValueError
        encoded_profile_id = encode_uuid(profile_id)
        if operation_kind == "create":
            row = connection.execute(
                """
                SELECT profile_id, normalized_name
                FROM tts_generation_profiles
                WHERE profile_id = ? OR normalized_name = ?
                LIMIT 1
                """,
                (encoded_profile_id, normalized_name),
            ).fetchone()
        elif operation_kind == "update":
            row = connection.execute(
                """
                SELECT profile_id, normalized_name
                FROM tts_generation_profiles
                WHERE profile_id != ? AND normalized_name = ?
                LIMIT 1
                """,
                (encoded_profile_id, normalized_name),
            ).fetchone()
        else:
            raise ValueError

        if row is None:
            conflict_exists = False
        elif len(row) == 2 and type(row[0]) is str and type(row[1]) is str:
            stored_profile_id = cast(str, row[0])
            stored_normalized_name = cast(str, row[1])
            if operation_kind == "create" and (
                stored_profile_id == encoded_profile_id
                or stored_normalized_name == normalized_name
            ):
                conflict_exists = True
            elif (
                operation_kind == "update"
                and stored_profile_id != encoded_profile_id
                and stored_normalized_name == normalized_name
            ):
                conflict_exists = True
    except BaseException as error:
        inspection_error = error

    if inspection_error is not None:
        if not isinstance(inspection_error, Exception):
            raise inspection_error
        return None
    return conflict_exists


def _profile_has_assignment(
    connection: sqlite3.Connection,
    profile_id: UUID | None,
) -> bool | None:
    """Check one exact schema-owned delete restriction, failing closed."""

    inspection_error: BaseException | None = None
    assignment_exists: bool | None = None
    try:
        if type(profile_id) is not UUID:
            raise ValueError
        encoded_profile_id = encode_uuid(profile_id)
        row = connection.execute(
            """
            SELECT profile_id
            FROM character_tts_assignments
            WHERE profile_id = ?
            LIMIT 1
            """,
            (encoded_profile_id,),
        ).fetchone()
        if row is None:
            assignment_exists = False
        elif len(row) == 1 and type(row[0]) is str and row[0] == encoded_profile_id:
            assignment_exists = True
    except BaseException as error:
        inspection_error = error

    if inspection_error is not None:
        if not isinstance(inspection_error, Exception):
            raise inspection_error
        return None
    return assignment_exists


def _has_integrity_conflict_evidence(
    connection: sqlite3.Connection,
    error: sqlite3.IntegrityError,
    operation_kind: _TransactionOperation,
    evidence: _IntegrityEvidence | None,
) -> bool:
    """Require an exact extended code and matching row under the held lock."""

    sqlite_errorcode = _read_sqlite_errorcode(error)
    if not connection.in_transaction or evidence is None:
        return False
    if operation_kind in ("create", "update") and sqlite_errorcode in (
        _SQLITE_CONSTRAINT_PRIMARYKEY,
        _SQLITE_CONSTRAINT_UNIQUE,
    ):
        return (
            _profile_conflict_evidence(
                connection,
                operation_kind,
                evidence.profile_id,
                evidence.normalized_name,
            )
            is True
        )
    if operation_kind == "delete" and sqlite_errorcode in (
        _SQLITE_CONSTRAINT_FOREIGNKEY,
        _SQLITE_CONSTRAINT_TRIGGER,
    ):
        return _profile_has_assignment(connection, evidence.profile_id) is True
    return False


def _fresh_repository_error(
    error: ProfileRepositoryError,
) -> ProfileRepositoryError:
    """Recreate one structured error without its traceback, chain, or notes."""

    code: object = "operation_failed"
    code_error: BaseException | None = None
    try:
        code = error.code
    except BaseException as caught:
        code_error = caught
    if code_error is not None and not isinstance(code_error, Exception):
        raise code_error
    if code_error is not None or type(code) is not str:
        code = "operation_failed"
    return ProfileRepositoryError(cast(str, code))


def _raise_operation_error(error: BaseException) -> None:
    """Preserve safe repository/control-flow errors and bound every other error."""

    if not isinstance(error, Exception):
        raise error
    if isinstance(error, ProfileRepositoryError):
        raise _fresh_repository_error(error)
    raise _repository_error("operation_failed")


def _raise_with_cleanup_precedence(
    primary_error: BaseException | None,
    *cleanup_errors: BaseException | None,
) -> None:
    """Apply the hardened cleanup precedence used by adjacent profile modules."""

    if primary_error is not None and not isinstance(primary_error, Exception):
        raise primary_error
    for cleanup_error in cleanup_errors:
        if cleanup_error is not None and not isinstance(cleanup_error, Exception):
            raise cleanup_error
    if any(cleanup_error is not None for cleanup_error in cleanup_errors):
        raise _repository_error("operation_failed")
    if isinstance(primary_error, ProfileRepositoryError):
        raise _fresh_repository_error(primary_error)
    if primary_error is not None:
        raise _repository_error("operation_failed")


def _raise_cleanup_errors(*errors: BaseException | None) -> None:
    """Preserve the first control-flow cleanup signal or report a safe failure."""

    for error in errors:
        if error is not None and not isinstance(error, Exception):
            raise error
    if any(error is not None for error in errors):
        raise _repository_error("operation_failed")


def _retrieve_future_exception(future: asyncio.Future[_T]) -> None:
    """Mark one wrapper exception retrieved without changing await behavior."""

    try:
        future.exception()
    except BaseException:
        pass


class TTSProfileRepository:
    """Own one serialized profile-store connection and its lifecycle generation.

    Construction is deliberately pure. The executor, worker thread, shared
    lease, filesystem, and SQLite connection are first touched by :meth:`open`.
    """

    def __init__(
        self,
        database_path: Path,
        *,
        _clock: Callable[[], datetime] | None = None,
        _uuid_factory: Callable[[], UUID] | None = None,
    ) -> None:
        """Create an initially closed, reopenable repository.

        Args:
            database_path: Exact local profile-store path.
            _clock: Private deterministic UTC-clock seam.
            _uuid_factory: Private deterministic UUID4 seam.

        Raises:
            ProfileRepositoryError: If a constructor input is invalid.
        """

        if (
            type(database_path) is not _PATH_TYPE
            or (_clock is not None and not callable(_clock))
            or (_uuid_factory is not None and not callable(_uuid_factory))
        ):
            raise _repository_error("operation_failed")

        self._database_path = database_path
        self._clock = _utc_now if _clock is None else _clock
        self._uuid_factory = uuid4 if _uuid_factory is None else _uuid_factory
        self._state = ProfileRepositoryState.CLOSED
        self._generation = 0
        self._terminal = False
        self._state_lock = threading.Lock()
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._lifecycle_lock: asyncio.Lock | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._executor_shutdown = False
        self._connection: sqlite3.Connection | None = None
        self._lease: ProfileStoreLease | None = None
        self._active_database_path: Path | None = None
        self._store_established = False
        self._pending_futures: set[Future[object]] = set()
        self._open_completion: asyncio.Task[ProfileStoreResult[None]] | None = None

    @property
    def state(self) -> ProfileRepositoryState:
        """Return the current public lifecycle state."""

        with self._state_lock:
            return self._state

    @property
    def generation(self) -> int:
        """Return the current monotonic lifecycle generation."""

        with self._state_lock:
            return self._generation

    @property
    def terminal(self) -> bool:
        """Return whether definitive close has made ``closed`` terminal."""

        with self._state_lock:
            return self._terminal

    def _active_path_for_operation(self, failure_code: str) -> Path:
        """Snapshot the open lifecycle's canonical path without filesystem I/O."""

        with self._state_lock:
            state_error = self._normal_state_error_locked()
            active_path = self._active_database_path
        if state_error is not None:
            raise _repository_error(state_error)
        if active_path is None:
            raise _repository_error(failure_code)
        return active_path

    def _require_configured_path_matches(
        self,
        active_path: Path,
        failure_code: str,
    ) -> None:
        """Fail closed when the configured path no longer resolves as opened."""

        current_path = _canonical_database_path(self._database_path, failure_code)
        if current_path != active_path:
            raise _repository_error(failure_code)

    def _worker_active_path(self) -> Path:
        """Return the worker-owned canonical path without re-resolving config."""

        active_path = self._active_database_path
        if active_path is None:
            raise _repository_error("invalid_state")
        return active_path

    async def open(self) -> ProfileStoreResult[None]:
        """Open the profile store or retry one unavailable open attempt.

        Returns:
            The active lifecycle generation with a ``None`` value.

        Raises:
            ProfileRepositoryError: If the state is invalid, the store cannot
                be opened safely, or the repository was definitively closed.
            BaseException: A worker control-flow signal, after partial
                ownership has been cleaned.
        """

        lifecycle_lock = self._bind_or_check_loop()
        with self._state_lock:
            shared_completion = self._open_completion
        if shared_completion is not None:
            return await self._await_open_completion(shared_completion)

        async with lifecycle_lock:
            with self._state_lock:
                if self._terminal:
                    raise _repository_error("terminal")
                if self._state is ProfileRepositoryState.OPEN:
                    return ProfileStoreResult(
                        generation=self._generation,
                        value=None,
                    )
                state_error = self._open_state_error_locked()
                if state_error is not None:
                    raise _repository_error(state_error)
                self._generation += 1
                generation = self._generation
                executor = self._executor

            if executor is None:
                executor_error: BaseException | None = None
                created_executor: ThreadPoolExecutor | None = None
                try:
                    created_executor = ThreadPoolExecutor(max_workers=1)
                except BaseException as error:
                    executor_error = error

                if executor_error is not None:
                    with self._state_lock:
                        self._state = ProfileRepositoryState.UNAVAILABLE
                    _raise_operation_error(executor_error)
                assert created_executor is not None
                with self._state_lock:
                    self._executor = created_executor
                    self._executor_shutdown = False
                executor = created_executor

            submission_error: BaseException | None = None
            open_future: Future[None] | None = None
            try:
                open_future = executor.submit(self._worker_open)
            except BaseException as error:
                submission_error = error

            if submission_error is not None:
                with self._state_lock:
                    self._state = ProfileRepositoryState.UNAVAILABLE
                _raise_operation_error(submission_error)
            assert open_future is not None

            completion = asyncio.create_task(self._finish_open(generation, open_future))
            with self._state_lock:
                self._open_completion = completion
            return await self._await_open_completion(completion)

    async def _await_open_completion(
        self,
        completion: asyncio.Task[ProfileStoreResult[None]],
    ) -> ProfileStoreResult[None]:
        """Join one open attempt and clear its marker only after settlement."""

        self._bind_or_check_loop()
        try:
            return await self._await_lifecycle_completion(completion)
        finally:
            with self._state_lock:
                if completion.done() and self._open_completion is completion:
                    self._open_completion = None

    def _open_state_error_locked(self) -> str | None:
        if self._state is ProfileRepositoryState.RESTORING:
            return "restoring"
        if self._state not in (
            ProfileRepositoryState.CLOSED,
            ProfileRepositoryState.UNAVAILABLE,
        ):
            return "invalid_state"
        if self._executor_shutdown:
            return "terminal"
        return None

    async def _finish_open(
        self,
        generation: int,
        open_future: Future[None],
    ) -> ProfileStoreResult[None]:
        self._bind_or_check_loop()
        open_error: BaseException | None = None
        try:
            await asyncio.wrap_future(open_future)
        except BaseException as error:
            open_error = error

        with self._state_lock:
            generation_changed = self._generation != generation or self._terminal
            if open_error is None and not generation_changed:
                self._state = ProfileRepositoryState.OPEN
            else:
                self._state = ProfileRepositoryState.UNAVAILABLE

        if open_error is not None:
            _raise_operation_error(open_error)
        if generation_changed:
            raise _repository_error("stale")
        return ProfileStoreResult(generation=generation, value=None)

    def _worker_open(self) -> None:
        """Acquire shared ownership and open the long-lived connection."""

        if self._connection is not None or self._lease is not None:
            self._worker_cleanup()

        lease: ProfileStoreLease | None = None
        connection: sqlite3.Connection | None = None
        active_path: Path | None = None
        body_error: BaseException | None = None
        try:
            active_path = _canonical_database_path(
                self._database_path,
                "operation_failed",
            )
            try:
                lease, connection = self._worker_open_existing(active_path)
            except ProfileRepositoryError as error:
                if self._store_established or error.code not in {
                    "missing",
                    "schema_partial",
                }:
                    raise
                try:
                    self._worker_initialize_store(active_path)
                except ProfileRepositoryError as initialization_error:
                    if initialization_error.code != "lock_timeout":
                        raise
                    # Another fresh process may have won initialization and
                    # downgraded to its long-lived shared lease before this
                    # contender acquired exclusive ownership. Re-read under
                    # shared ownership instead of failing a healthy first open.
                    lease, connection = self._worker_open_existing(active_path)
                else:
                    lease, connection = self._worker_open_existing(active_path)
            if connection is None:
                raise _repository_error("operation_failed")
            validate_profile_store_rows(connection)
            self._require_configured_path_matches(
                active_path,
                "operation_failed",
            )
        except BaseException as error:
            body_error = error

        if body_error is None:
            assert lease is not None
            assert active_path is not None
            self._lease = lease
            self._connection = connection
            self._active_database_path = active_path
            self._store_established = True
            return

        connection_error: BaseException | None = None
        lease_error: BaseException | None = None
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                connection_error = error
                self._connection = connection
        if lease is not None and connection_error is not None:
            self._lease = lease
        elif lease is not None:
            try:
                lease.release()
            except BaseException as error:
                lease_error = error
                self._lease = lease
        if self._connection is not None or self._lease is not None:
            self._active_database_path = active_path
        else:
            self._active_database_path = None
        _raise_with_cleanup_precedence(
            body_error,
            connection_error,
            lease_error,
        )

    def _worker_open_existing(
        self,
        active_path: Path,
    ) -> tuple[ProfileStoreLease, sqlite3.Connection]:
        """Open an established store while holding cooperative shared ownership."""

        lease = ProfileStoreLease(
            active_path,
            ProfileStoreLockMode.SHARED,
        )
        connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        release_error: BaseException | None = None
        try:
            lease.acquire()
            connection = open_profile_store(active_path, must_exist=True)
            if connection is None:
                raise _repository_error("operation_failed")
        except BaseException as error:
            body_error = error

        if body_error is None:
            assert connection is not None
            return lease, connection

        try:
            lease.release()
        except BaseException as error:
            release_error = error
            self._lease = lease
            self._active_database_path = active_path
        _raise_with_cleanup_precedence(body_error, release_error)
        raise AssertionError("unreachable")

    def _worker_initialize_store(self, active_path: Path) -> None:
        """Create or migrate one store only while holding exclusive ownership."""

        lease = ProfileStoreLease(
            active_path,
            ProfileStoreLockMode.EXCLUSIVE,
            timeout_seconds=_INITIALIZATION_LOCK_TIMEOUT_SECONDS,
        )
        connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        connection_error: BaseException | None = None
        lease_error: BaseException | None = None
        try:
            lease.acquire()
            connection = open_profile_store(active_path)
            validate_profile_store_rows(connection)
            self._require_configured_path_matches(
                active_path,
                "operation_failed",
            )
        except BaseException as error:
            body_error = error

        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                connection_error = error
                self._connection = connection
                self._lease = lease
                self._active_database_path = active_path
        if connection_error is None:
            try:
                lease.release()
            except BaseException as error:
                lease_error = error
                self._lease = lease
                self._active_database_path = active_path
        _raise_with_cleanup_precedence(
            body_error,
            connection_error,
            lease_error,
        )

    async def create_profile(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID | None = None,
        *,
        expected_generation: int | None = None,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Create one immutable profile at revision 1.

        Args:
            draft: Exact validated profile draft.
            profile_id: Optional exact caller-selected UUID. When omitted, the
                repository generates a UUID4 on its serialized worker.
            expected_generation: Optional exact lifecycle generation when the
                create is derived from caller-held repository state.

        Returns:
            The active generation and exact persisted profile.

        Raises:
            ProfileRepositoryError: If inputs, state, persistence, or
                uniqueness checks fail safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_draft = _validate_draft(draft)
        validated_profile_id = _validate_optional_profile_id(profile_id)
        validated_generation = (
            None
            if expected_generation is None
            else _validate_expected_generation(expected_generation)
        )
        return await self._submit_operation(
            lambda connection: self._worker_create_profile(
                connection,
                validated_draft,
                validated_profile_id,
            ),
            expected_generation=validated_generation,
        )

    async def create_profile_with_assignment(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID,
        character_ref: CharacterRef,
        *,
        expected_generation: int,
        expected_current_profile_id: UUID | None,
    ) -> ProfileStoreResult[AssignedTTSProfileSnapshot]:
        """Atomically create one profile and set one exact assignment."""

        validated_draft = _validate_draft(draft)
        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_character_ref = _validate_character_ref(character_ref)
        validated_generation = _validate_expected_generation(expected_generation)
        validated_current_profile_id = _validate_optional_profile_id(
            expected_current_profile_id
        )
        return await self._submit_operation(
            lambda connection: self._worker_create_profile_with_assignment(
                connection,
                validated_draft,
                validated_profile_id,
                validated_character_ref,
                validated_generation,
                validated_current_profile_id,
            ),
            expected_generation=validated_generation,
        )

    async def get_profile(
        self,
        profile_id: UUID,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Load and fully decode one profile by exact UUID.

        Args:
            profile_id: Exact profile UUID.

        Returns:
            The active generation and immutable decoded profile.

        Raises:
            ProfileRepositoryError: If the input, state, row, or SQLite access
                fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_profile_id = _validate_exact_profile_id(profile_id)
        return await self._submit_operation(
            lambda connection: self._worker_get_profile(
                connection,
                validated_profile_id,
            )
        )

    async def get_profile_collisions(
        self,
        profile_id: UUID,
        draft: TTSProfileDraft,
    ) -> ProfileStoreResult[TTSProfileCollisionSnapshot]:
        """Read exact rows matching a portable UUID hint or normalized name."""

        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_draft = _validate_draft(draft)
        return await self._submit_operation(
            lambda connection: self._worker_get_profile_collisions(
                connection,
                validated_profile_id,
                validated_draft.normalized_name,
            )
        )

    async def list_profiles(
        self,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ProfileStoreResult[TTSProfilePage]:
        """List one stable bounded page and the full filtered result count.

        Search is trimmed, normalized with Unicode NFKC, and case-folded like
        persisted profile names. Empty or whitespace-only search lists all
        profiles. SQL LIKE metacharacters and the explicit escape character
        are always treated literally.

        Args:
            search: Optional exact string of at most 128 characters whose
                normalized form remains within the repository's bounded
                search policy.
            limit: Exact integer page size from 1 through 100.
            offset: Exact nonnegative integer result offset.

        Returns:
            The active generation and an immutable profile page.

        Raises:
            ProfileRepositoryError: If inputs, state, decoding, or SQLite
                access fail safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        normalized_search = _normalize_search(search)
        validated_limit = _validate_page_limit(limit)
        validated_offset = _validate_page_offset(offset)
        return await self._submit_operation(
            lambda connection: self._worker_list_profiles(
                connection,
                normalized_search,
                validated_limit,
                validated_offset,
            )
        )

    async def update_profile(
        self,
        profile_id: UUID,
        expected_revision: int,
        draft: TTSProfileDraft,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Replace one profile only at the exact editor revision.

        Args:
            profile_id: Exact profile UUID.
            expected_revision: Exact positive revision loaded by the editor.
            draft: Exact replacement profile draft.
            expected_generation: Exact nonnegative lifecycle generation loaded
                by the editor.

        Returns:
            The active generation and immutable updated profile.

        Raises:
            ProfileRepositoryError: If inputs, state, optimistic revision,
                uniqueness, row decoding, or SQLite access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_revision = _validate_expected_revision(expected_revision)
        validated_draft = _validate_draft(draft)
        validated_generation = _validate_expected_generation(expected_generation)
        return await self._submit_operation(
            lambda connection: self._worker_update_profile(
                connection,
                validated_profile_id,
                validated_revision,
                validated_draft,
            ),
            expected_generation=validated_generation,
        )

    async def delete_profile(
        self,
        profile_id: UUID,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[None]:
        """Delete exactly one unreferenced profile by UUID.

        Args:
            profile_id: Exact profile UUID.
            expected_generation: Exact nonnegative lifecycle generation loaded
                with the profile.

        Returns:
            The active generation paired with ``None``.

        Raises:
            ProfileRepositoryError: If the input or state is invalid, the row
                is missing or referenced, or SQLite access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_generation = _validate_expected_generation(expected_generation)
        return await self._submit_operation(
            lambda connection: self._worker_delete_profile(
                connection,
                validated_profile_id,
            ),
            expected_generation=validated_generation,
        )

    async def assignment_count(
        self,
        profile_id: UUID,
    ) -> ProfileStoreResult[int]:
        """Count assignments to one existing profile across all authorities.

        Args:
            profile_id: Exact profile UUID.

        Returns:
            The active generation and nonnegative assignment count.

        Raises:
            ProfileRepositoryError: If the input, profile, count row, state, or
                SQLite access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_profile_id = _validate_exact_profile_id(profile_id)
        return await self._submit_operation(
            lambda connection: self._worker_assignment_count(
                connection,
                validated_profile_id,
            )
        )

    async def set_assignment(
        self,
        character_ref: CharacterRef,
        profile_id: UUID,
        *,
        expected_generation: int,
        expected_profile_revision: int,
        expected_current_profile_id: UUID | None,
        expected_profile: TTSGenerationProfile | None = None,
    ) -> ProfileStoreResult[CharacterTTSAssignment]:
        """Create or replace one exact authority-scoped assignment.

        Args:
            character_ref: Exact validated source, authority, and character.
            profile_id: Exact existing profile UUID.
            expected_generation: Exact nonnegative lifecycle generation loaded
                with the selected profile and current assignment.
            expected_profile_revision: Exact positive revision of the selected
                profile.
            expected_current_profile_id: Exact currently assigned profile UUID,
                or ``None`` when the character was observed as unassigned.
            expected_profile: Optional exact immutable selected-profile snapshot.
                When supplied, a delete/recreate with the same UUID and revision
                is rejected as a conflict.

        Returns:
            The active generation and persisted assignment.

        Raises:
            ProfileRepositoryError: If inputs, state, optimistic expectations,
                persistence, foreign-key checks, row decoding, or SQLite
                access fail safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_character_ref = _validate_character_ref(character_ref)
        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_generation = _validate_expected_generation(expected_generation)
        validated_profile_revision = _validate_expected_revision(
            expected_profile_revision
        )
        validated_current_profile_id = _validate_optional_profile_id(
            expected_current_profile_id
        )
        validated_expected_profile = _validate_optional_profile(expected_profile)
        return await self._submit_operation(
            lambda connection: self._worker_set_assignment(
                connection,
                validated_character_ref,
                validated_profile_id,
                validated_generation,
                validated_profile_revision,
                validated_current_profile_id,
                validated_expected_profile,
            ),
            expected_generation=validated_generation,
        )

    async def remove_assignment(
        self,
        character_ref: CharacterRef,
        *,
        expected_generation: int,
        expected_profile_id: UUID,
    ) -> ProfileStoreResult[None]:
        """Remove one exact authority-scoped assignment idempotently.

        Args:
            character_ref: Exact validated source, authority, and character.
            expected_generation: Exact nonnegative lifecycle generation loaded
                with the assignment.
            expected_profile_id: Exact profile UUID observed on the assignment.

        Returns:
            The active generation paired with ``None``.

        Raises:
            ProfileRepositoryError: If an input, state, optimistic expectation,
                persistence, or SQLite access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_character_ref = _validate_character_ref(character_ref)
        validated_generation = _validate_expected_generation(expected_generation)
        validated_profile_id = _validate_exact_profile_id(expected_profile_id)
        return await self._submit_operation(
            lambda connection: self._worker_remove_assignment(
                connection,
                validated_character_ref,
                validated_generation,
                validated_profile_id,
            ),
            expected_generation=validated_generation,
        )

    async def get_assigned_profile(
        self,
        character_ref: CharacterRef,
    ) -> ProfileStoreResult[AssignedTTSProfileSnapshot | None]:
        """Read one exact assignment and immutable profile revision by JOIN.

        Args:
            character_ref: Exact validated source, authority, and character.

        Returns:
            The active generation and joined snapshot, or ``None`` when the
            exact character is unassigned.

        Raises:
            ProfileRepositoryError: If the input, state, joined row, or SQLite
                access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_character_ref = _validate_character_ref(character_ref)
        return await self._submit_operation(
            lambda connection: self._worker_get_assigned_profile(
                connection,
                validated_character_ref,
            )
        )

    async def backup_to(
        self,
        destination: Path,
    ) -> ProfileStoreResult[ProfileBackupReceipt]:
        """Publish one validated SQLite online-backup snapshot atomically.

        Args:
            destination: Exact non-store path for the completed snapshot.

        Returns:
            The active generation and safe backup metadata.

        Raises:
            ProfileRepositoryError: If path admission, state, backup,
                validation, or atomic publication fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        if type(destination) is not _PATH_TYPE:
            raise _repository_error("backup_failed")
        exact_destination = cast(Path, destination)
        active_path = self._active_path_for_operation("backup_failed")
        return await self._submit_operation(
            lambda connection: self._worker_backup_to(
                connection,
                exact_destination,
                active_path,
            )
        )

    def _worker_backup_to(
        self,
        connection: sqlite3.Connection,
        destination_path: Path,
        active_path: Path,
    ) -> ProfileBackupReceipt:
        """Create and atomically publish one worker-owned online backup."""

        destination = _validate_backup_destination(destination_path, active_path)
        temporary_path: Path | None = None
        destination_connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        published = False
        receipt: ProfileBackupReceipt | None = None
        try:
            if self._worker_active_path() != active_path:
                raise _repository_error("backup_failed")
            self._require_configured_path_matches(active_path, "backup_failed")
            # Validate the clock before any destination publication.
            created_at = self._clock()
            ProfileBackupReceipt(created_at=created_at, byte_count=0)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.path.name}.",
                suffix=".backup",
                dir=destination.path.parent,
            )
            temporary_path = Path(temporary_name)
            os.close(descriptor)
            destination_connection = connect_private_sqlite(
                "tts.profile_backup",
                temporary_path,
                must_exist=True,
                isolation_level=None,
            )
            self._worker_online_backup(connection, destination_connection)
            destination_connection.close()
            destination_connection = None
            self._worker_validate_standalone_snapshot(temporary_path)
            temporary_state = temporary_path.stat()
            if not stat.S_ISREG(temporary_state.st_mode):
                raise _repository_error("backup_failed")
            receipt = ProfileBackupReceipt(
                created_at=created_at,
                byte_count=temporary_state.st_size,
            )

            self._require_configured_path_matches(active_path, "backup_failed")
            current_destination = _validate_backup_destination(
                destination.path,
                active_path,
            )
            if current_destination.parent_identity != destination.parent_identity:
                raise _repository_error("backup_failed")
            _fsync_file(temporary_path)
            os.replace(temporary_path, destination.path)
            published = True
            _fsync_directory(destination.path.parent)
        except BaseException as error:
            body_error = error

        if destination_connection is not None:
            try:
                destination_connection.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if temporary_path is not None:
            if not published:
                try:
                    _unlink_path_if_present(temporary_path)
                except BaseException as error:
                    cleanup_errors.append(error)
            for suffix in _STORE_SIDECAR_SUFFIXES:
                try:
                    _unlink_path_if_present(
                        temporary_path.with_name(f"{temporary_path.name}{suffix}")
                    )
                except BaseException as error:
                    cleanup_errors.append(error)

        if body_error is not None or cleanup_errors:
            for candidate_error in (body_error, *cleanup_errors):
                if candidate_error is not None and not isinstance(
                    candidate_error,
                    Exception,
                ):
                    raise candidate_error
            raise _repository_error("backup_failed")
        assert receipt is not None
        return receipt

    def _worker_online_backup(
        self,
        source: sqlite3.Connection,
        destination: sqlite3.Connection,
        *,
        deadline: float | None = None,
    ) -> None:
        """Copy one complete SQLite snapshot through the online-backup API."""

        progress_guard = (
            None if deadline is None else lambda: _require_restore_time(deadline)
        )
        if progress_guard is not None:
            progress_guard()
        backup_open_connections_to_private(
            "tts.profile_backup",
            source,
            destination,
            progress_guard=progress_guard,
        )
        if progress_guard is not None:
            progress_guard()

    def _worker_require_full_integrity(
        self,
        connection: sqlite3.Connection,
        *,
        deadline: float | None = None,
    ) -> None:
        """Require SQLite's full integrity check on one worker-owned handle."""

        callback_error: BaseException | None = None
        body_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        results: list[object] | None = None
        progress_installed = False

        def interrupt_after_deadline() -> int:
            nonlocal callback_error
            try:
                assert deadline is not None
                _require_restore_time(deadline)
            except BaseException as error:
                callback_error = error
                return 1
            return 0

        try:
            if deadline is not None:
                _require_restore_time(deadline)
                connection.set_progress_handler(
                    interrupt_after_deadline,
                    _RESTORE_PROGRESS_OPCODE_INTERVAL,
                )
                progress_installed = True
            results = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        except BaseException as error:
            body_error = error
        if progress_installed:
            try:
                connection.set_progress_handler(None, 0)
            except BaseException as error:
                cleanup_error = error

        if callback_error is not None:
            body_error = callback_error
        elif body_error is None and deadline is not None:
            try:
                _require_restore_time(deadline)
            except BaseException as error:
                body_error = error

        mapped_errors: list[BaseException | None] = []
        for candidate_error in (body_error, cleanup_error):
            if candidate_error is None or not isinstance(candidate_error, Exception):
                mapped_errors.append(candidate_error)
            elif isinstance(candidate_error, ProfileRepositoryError):
                mapped_errors.append(candidate_error)
            else:
                mapped_errors.append(_repository_error("schema_corrupt"))
        _raise_with_cleanup_precedence(*mapped_errors)
        if results != ["ok"]:
            raise _repository_error("schema_corrupt")

    def _worker_validate_standalone_snapshot(
        self,
        path: Path,
        *,
        deadline: float | None = None,
    ) -> None:
        """Run shared schema/domain checks plus full integrity on one snapshot."""

        check_deadline = (
            None if deadline is None else lambda: _require_restore_time(deadline)
        )
        validate_profile_candidate(
            path,
            check_deadline=check_deadline,
        )
        connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        close_error: BaseException | None = None
        try:
            if check_deadline is not None:
                check_deadline()
            connection = connect_private_sqlite(
                "tts.profile_snapshot",
                path,
                read_only=True,
                immutable=True,
                isolation_level=None,
            )
            if check_deadline is not None:
                check_deadline()
            self._worker_require_full_integrity(
                connection,
                deadline=deadline,
            )
        except BaseException as error:
            body_error = error
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                close_error = error
        _raise_with_cleanup_precedence(body_error, close_error)

    def _worker_restore(
        self,
        candidate_path: Path,
        deadline: float,
        generation: int,
        active_path: Path,
    ) -> ProfileRestoreReceipt:
        """Run one exclusive staged restore and race-free shared rebind."""

        stage_path: Path | None = None
        recovery_path: Path | None = None
        exclusive_lease: ProfileStoreLease | None = None
        rebound_lease: ProfileStoreLease | None = None
        rebound_connection: sqlite3.Connection | None = None
        primary_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        stage_cleanup_errors: list[BaseException] = []
        replaced = False
        receipt: ProfileRestoreReceipt | None = None
        try:
            if self._worker_active_path() != active_path:
                raise _repository_error("restore_failed")
            _require_restore_time(deadline)
            self._require_configured_path_matches(active_path, "restore_failed")
            candidate = _validate_restore_candidate_path(
                candidate_path,
                active_path,
            )
            _require_restore_time(deadline)
            restored_at = self._clock()
            ProfileRestoreReceipt(
                restored_at=restored_at,
                profile_count=0,
                assignment_count=0,
            )
            self._worker_close_for_restore(deadline)

            remaining = _remaining_seconds(deadline)
            if remaining <= 0:
                raise _repository_error("restore_failed")
            exclusive_lease = ProfileStoreLease(
                active_path,
                ProfileStoreLockMode.EXCLUSIVE,
                timeout_seconds=remaining,
            )
            exclusive_lease.acquire()
            _require_restore_time(deadline)
            self._require_configured_path_matches(active_path, "restore_failed")

            if not _candidate_is_unchanged(candidate):
                raise _repository_error("restore_failed")
            _require_restore_time(deadline)
            stage_path = self._worker_stage_candidate(candidate, deadline)
            _require_restore_time(deadline)
            self._require_configured_path_matches(active_path, "restore_failed")
            recovery_path = self._worker_create_recovery_backup(
                restored_at,
                deadline,
            )
            _require_restore_time(deadline)
            self._require_configured_path_matches(active_path, "restore_failed")
            self._worker_remove_live_sidecars(deadline=deadline)
            _require_restore_time(deadline)
            self._require_configured_path_matches(active_path, "restore_failed")
            _fsync_file(stage_path)
            _require_restore_time(deadline)
            _fsync_directory(stage_path.parent)
            _require_restore_time(deadline)
            self._require_configured_path_matches(active_path, "restore_failed")
            _require_restore_time(deadline)
            os.replace(stage_path, active_path)
            replaced = True
            stage_path = None
            _fsync_directory(active_path.parent)
            _require_restore_time(deadline)
            self._require_configured_path_matches(active_path, "restore_failed")

            _require_restore_time(deadline)

            def deadline_check() -> None:
                _require_restore_time(deadline)

            scoped = open_profile_store(
                active_path,
                must_exist=True,
                check_deadline=deadline_check,
            )
            scoped_error: BaseException | None = None
            try:
                _require_restore_time(deadline)
                self._worker_require_full_integrity(
                    scoped,
                    deadline=deadline,
                )
                validate_profile_store_rows(
                    scoped,
                    check_deadline=deadline_check,
                )
                self._worker_store_counts(scoped, deadline=deadline)
            except BaseException as error:
                scoped_error = error
            close_error: BaseException | None = None
            try:
                scoped.close()
            except BaseException as error:
                close_error = error
                self._connection = scoped
                self._lease = exclusive_lease
                exclusive_lease = None
            _raise_with_cleanup_precedence(scoped_error, close_error)

            assert exclusive_lease is not None
            _require_restore_time(deadline)
            exclusive_lease.release()
            exclusive_lease = None
            remaining = _remaining_seconds(deadline)
            if remaining <= 0:
                raise _repository_error("restore_failed")
            rebound_lease = ProfileStoreLease(
                active_path,
                ProfileStoreLockMode.SHARED,
                timeout_seconds=min(_RESTORE_REBIND_TIMEOUT_SECONDS, remaining),
            )
            rebound_lease.acquire()
            _require_restore_time(deadline)
            rebound_connection = open_profile_store(
                active_path,
                must_exist=True,
                check_deadline=deadline_check,
            )
            # Validate the authoritative long-lived handle, not only the
            # scoped pre-handoff handle.
            self._worker_require_full_integrity(
                rebound_connection,
                deadline=deadline,
            )
            validate_profile_store_rows(
                rebound_connection,
                check_deadline=deadline_check,
            )
            profile_count, assignment_count = self._worker_store_counts(
                rebound_connection,
                deadline=deadline,
            )
            receipt = ProfileRestoreReceipt(
                restored_at=restored_at,
                profile_count=profile_count,
                assignment_count=assignment_count,
            )
            self._require_configured_path_matches(active_path, "restore_failed")
            _require_restore_time(deadline)
            with self._state_lock:
                _require_restore_time(deadline)
                if (
                    self._generation != generation
                    or self._terminal
                    or self._state is not ProfileRepositoryState.RESTORING
                ):
                    raise _repository_error("stale")
                self._lease = rebound_lease
                self._connection = rebound_connection
                rebound_lease = None
                rebound_connection = None
                self._state = ProfileRepositoryState.OPEN
        except BaseException as error:
            primary_error = error

        if stage_path is not None:
            stage_cleanup_errors.extend(self._worker_remove_temporary_store(stage_path))
        if rebound_connection is not None:
            try:
                rebound_connection.close()
            except BaseException as error:
                cleanup_errors.append(error)
                self._connection = rebound_connection
                self._lease = rebound_lease
                rebound_lease = None
            rebound_connection = None
        if rebound_lease is not None:
            try:
                rebound_lease.release()
            except BaseException as error:
                cleanup_errors.append(error)
                self._lease = rebound_lease
                rebound_lease = None
        if exclusive_lease is not None:
            if self._connection is not None:
                self._lease = exclusive_lease
                exclusive_lease = None
            else:
                try:
                    exclusive_lease.release()
                except BaseException as error:
                    cleanup_errors.append(error)
                    self._lease = exclusive_lease
                    exclusive_lease = None

        if primary_error is None and not cleanup_errors and not stage_cleanup_errors:
            assert receipt is not None
            return receipt

        rebound_ok = False
        if not replaced and not cleanup_errors:
            try:
                if self._connection is not None and self._lease is not None:
                    if (
                        self._lease.mode is not ProfileStoreLockMode.SHARED
                        or not self._lease.acquired
                    ):
                        raise _repository_error("restore_failed")
                    validate_profile_store_rows(self._connection)
                    self._worker_store_counts(self._connection)
                    rebound_ok = True
                elif self._connection is None and self._lease is None:
                    self._worker_rebind_current_store()
                    rebound_ok = True
            except BaseException as error:
                cleanup_errors.append(error)
        with self._state_lock:
            if (
                self._generation == generation
                and not self._terminal
                and self._state is ProfileRepositoryState.RESTORING
            ):
                self._state = (
                    ProfileRepositoryState.OPEN
                    if rebound_ok
                    else ProfileRepositoryState.UNAVAILABLE
                )
        if not rebound_ok and self._connection is None and self._lease is None:
            self._active_database_path = None

        for candidate_error in (
            primary_error,
            *stage_cleanup_errors,
            *cleanup_errors,
        ):
            if candidate_error is not None and not isinstance(
                candidate_error,
                Exception,
            ):
                raise candidate_error
        if rebound_ok and isinstance(primary_error, ProfileRepositoryError):
            code = primary_error.code
            if code in {
                "corrupt_data",
                "lock_timeout",
                "schema_corrupt",
                "schema_partial",
                "schema_unsupported",
            }:
                raise _repository_error(code)
        # Once created, recovery evidence is deliberately retained on failure.
        _ = recovery_path
        raise _repository_error("restore_failed")

    def _worker_close_for_restore(self, deadline: float) -> None:
        connection = self._connection
        lease = self._lease
        if connection is None or lease is None:
            raise _repository_error("invalid_state")
        _require_restore_time(deadline)
        timeout_row = connection.execute("PRAGMA busy_timeout").fetchone()
        if (
            timeout_row is None
            or len(timeout_row) != 1
            or type(timeout_row[0]) is not int
            or timeout_row[0] < 0
        ):
            raise _repository_error("restore_failed")
        original_timeout_ms = cast(int, timeout_row[0])
        remaining = _remaining_seconds(deadline)
        if remaining <= 0:
            raise _repository_error("restore_failed")
        bounded_timeout_ms = min(original_timeout_ms, int(remaining * 1_000))
        checkpoint: object = None
        body_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            connection.execute(f"PRAGMA busy_timeout = {bounded_timeout_ms}")
            checkpoint = _run_with_restore_progress(
                connection,
                deadline,
                lambda: connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone(),
            )
        except BaseException as error:
            body_error = error
        try:
            connection.execute(f"PRAGMA busy_timeout = {original_timeout_ms}")
        except BaseException as error:
            cleanup_error = error
        _raise_with_cleanup_precedence(body_error, cleanup_error)
        if (
            checkpoint is None
            or not isinstance(checkpoint, (tuple, sqlite3.Row))
            or len(checkpoint) != 3
            or any(type(value) is not int for value in checkpoint)
            or tuple(checkpoint) != (0, 0, 0)
        ):
            raise _repository_error("restore_failed")
        connection.close()
        self._connection = None
        lease.release()
        self._lease = None
        _require_restore_time(deadline)

    def _worker_stage_candidate(
        self,
        candidate: _CandidateSnapshot,
        deadline: float,
    ) -> Path:
        active_path = self._worker_active_path()
        stage_path: Path | None = None
        body_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        try:
            _require_restore_time(deadline)
            descriptor, stage_name = tempfile.mkstemp(
                prefix=f".{active_path.name}.",
                suffix=".restore-stage.sqlite3",
                dir=active_path.parent,
            )
            stage_path = Path(stage_name)
            os.close(descriptor)
            copy_private_sqlite(
                "tts.profile_restore_stage",
                candidate.path,
                stage_path,
                progress_guard=lambda: _require_restore_time(deadline),
            )
            if not _candidate_is_unchanged(candidate):
                raise _repository_error("restore_failed")
            _require_restore_time(deadline)
            self._worker_validate_standalone_snapshot(
                stage_path,
                deadline=deadline,
            )
        except BaseException as error:
            body_error = error

        if isinstance(body_error, sqlite3.DatabaseError):
            body_error = _repository_error("schema_corrupt")
        if body_error is not None or cleanup_errors:
            if stage_path is not None:
                cleanup_errors.extend(self._worker_remove_temporary_store(stage_path))
            _raise_with_cleanup_precedence(body_error, *cleanup_errors)
        assert stage_path is not None
        return stage_path

    def _worker_create_recovery_backup(
        self,
        restored_at: datetime,
        deadline: float,
    ) -> Path:
        active_path = self._worker_active_path()
        source: sqlite3.Connection | None = None
        recovery_path: Path | None = None
        body_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        try:
            _require_restore_time(deadline)
            timestamp = restored_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            descriptor, recovery_name = tempfile.mkstemp(
                prefix=f"{active_path.name}.pre-restore-{timestamp}-",
                suffix=".recovery.sqlite3",
                dir=active_path.parent,
            )
            recovery_path = Path(recovery_name)
            os.close(descriptor)
            source = open_profile_store(
                active_path,
                must_exist=True,
                check_deadline=lambda: _require_restore_time(deadline),
            )
            backup_connection_to_private(
                "tts.profile_recovery",
                source,
                active_path,
                recovery_path,
                progress_guard=lambda: _require_restore_time(deadline),
            )
            try:
                source.close()
            except BaseException:
                self._connection = source
                source = None
                raise
            else:
                source = None
            self._worker_validate_standalone_snapshot(
                recovery_path,
                deadline=deadline,
            )
            _require_restore_time(deadline)
            _fsync_file(recovery_path)
            _require_restore_time(deadline)
            _fsync_directory(recovery_path.parent)
            _require_restore_time(deadline)
        except BaseException as error:
            body_error = error

        for connection in (source,):
            if connection is None:
                continue
            try:
                connection.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if body_error is not None or cleanup_errors:
            if recovery_path is not None:
                cleanup_errors.extend(
                    self._worker_remove_temporary_store(recovery_path)
                )
            _raise_with_cleanup_precedence(body_error, *cleanup_errors)
        assert recovery_path is not None
        return recovery_path

    def _worker_remove_live_sidecars(self, *, deadline: float) -> None:
        database_path = self._worker_active_path()
        rollback_journal = database_path.with_name(f"{database_path.name}-journal")
        _require_restore_time(deadline)
        try:
            rollback_journal.lstat()
        except FileNotFoundError:
            pass
        else:
            raise _repository_error("restore_failed")
        _require_restore_time(deadline)
        for suffix in ("-wal", "-shm"):
            sidecar = database_path.with_name(f"{database_path.name}{suffix}")
            _require_restore_time(deadline)
            try:
                state = sidecar.lstat()
            except FileNotFoundError:
                _require_restore_time(deadline)
                continue
            _require_restore_time(deadline)
            if not stat.S_ISREG(state.st_mode):
                raise _repository_error("restore_failed")
            _require_restore_time(deadline)
            sidecar.unlink()
            _require_restore_time(deadline)

    def _worker_remove_temporary_store(self, path: Path) -> list[BaseException]:
        errors: list[BaseException] = []
        for target in (
            path,
            *(
                path.with_name(f"{path.name}{suffix}")
                for suffix in _STORE_SIDECAR_SUFFIXES
            ),
        ):
            try:
                _unlink_path_if_present(target)
            except BaseException as error:
                errors.append(error)
        return errors

    def _worker_store_counts(
        self,
        connection: sqlite3.Connection,
        *,
        deadline: float | None = None,
    ) -> tuple[int, int]:
        def read_counts() -> tuple[int, int]:
            counts: list[int] = []
            for statement in (
                "SELECT COUNT(*) FROM tts_generation_profiles",
                "SELECT COUNT(*) FROM character_tts_assignments",
            ):
                if deadline is not None:
                    _require_restore_time(deadline)
                row = connection.execute(statement).fetchone()
                if (
                    row is None
                    or len(row) != 1
                    or type(row[0]) is not int
                    or row[0] < 0
                ):
                    raise _repository_error("corrupt_data")
                counts.append(cast(int, row[0]))
            if deadline is not None:
                _require_restore_time(deadline)
            return (counts[0], counts[1])

        if deadline is None:
            return read_counts()
        return _run_with_restore_progress(
            connection,
            deadline,
            read_counts,
        )

    def _worker_rebind_current_store(self) -> None:
        active_path = self._worker_active_path()
        if self._connection is not None and self._lease is not None:
            validate_profile_store_rows(self._connection)
            self._worker_store_counts(self._connection)
            return

        cleanup_error: BaseException | None = None
        try:
            self._worker_cleanup()
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error

        lease = ProfileStoreLease(
            active_path,
            ProfileStoreLockMode.SHARED,
            timeout_seconds=_RESTORE_REBIND_TIMEOUT_SECONDS,
        )
        connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        try:
            lease.acquire()
            connection = open_profile_store(
                active_path,
                must_exist=True,
            )
            validate_profile_store_rows(connection)
            self._worker_store_counts(connection)
        except BaseException as error:
            body_error = error
        if body_error is None:
            assert connection is not None
            self._lease = lease
            self._connection = connection
            self._active_database_path = active_path
            return

        connection_error: BaseException | None = None
        lease_error: BaseException | None = None
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                connection_error = error
                self._connection = connection
                self._lease = lease
                connection = None
        if connection_error is None:
            try:
                lease.release()
            except BaseException as error:
                lease_error = error
                self._lease = lease
        if self._connection is not None or self._lease is not None:
            self._active_database_path = active_path
        else:
            self._active_database_path = None
        _raise_with_cleanup_precedence(body_error, connection_error, lease_error)

    async def restore_from(
        self,
        candidate: Path,
        timeout_seconds: int | float = 5.0,
    ) -> ProfileStoreResult[ProfileRestoreReceipt]:
        """Atomically restore one validated standalone profile-store snapshot.

        The timeout is enforced cooperatively between bounded SQLite backup
        page batches, SQLite VM progress callbacks, row decodes, and filesystem
        boundaries. One in-flight kernel filesystem call cannot be interrupted.

        Args:
            candidate: Exact standalone candidate path.
            timeout_seconds: Positive finite quiescence/exclusive-lock budget.

        Returns:
            The admitted lifecycle generation and safe restore metadata.

        Raises:
            ProfileRepositoryError: If admission, quiescence, validation,
                locking, replacement, or lifecycle rebind fails safely.
            BaseException: A caller control-flow signal after lifecycle
                settlement and cleanup.
        """

        timeout = _validate_restore_timeout(timeout_seconds)
        if type(candidate) is not _PATH_TYPE:
            raise _repository_error("restore_failed")
        exact_candidate = cast(Path, candidate)
        active_path = self._active_path_for_operation("restore_failed")
        deadline = _read_monotonic() + timeout
        if not math.isfinite(deadline):
            raise _repository_error("restore_failed")

        lifecycle_lock = self._bind_or_check_loop()
        remaining = _remaining_seconds(deadline)
        if remaining <= 0:
            raise _repository_error("restore_failed")
        try:
            await asyncio.wait_for(lifecycle_lock.acquire(), timeout=remaining)
        except TimeoutError:
            raise _repository_error("restore_failed") from None

        try:
            with self._state_lock:
                state_error = self._normal_state_error_locked()
                if state_error is not None:
                    raise _repository_error(state_error)
                self._state = ProfileRepositoryState.RESTORING
                self._generation += 1
                generation = self._generation
                pending = tuple(self._pending_futures)
                executor = self._executor

            setup_error: BaseException | None = None
            completion: (
                asyncio.Task[ProfileStoreResult[ProfileRestoreReceipt]] | None
            ) = None
            try:
                for future in pending:
                    future.cancel()

                completion = asyncio.create_task(
                    self._finish_restore(
                        exact_candidate,
                        deadline,
                        generation,
                        pending,
                        executor,
                        active_path,
                    )
                )
            except BaseException as error:
                setup_error = error
            if setup_error is not None:
                with self._state_lock:
                    if (
                        self._generation == generation
                        and not self._terminal
                        and self._state is ProfileRepositoryState.RESTORING
                    ):
                        self._state = ProfileRepositoryState.OPEN
                if not isinstance(setup_error, Exception):
                    raise setup_error
                raise _repository_error("restore_failed")
            assert completion is not None
            return await self._await_lifecycle_completion(completion)
        finally:
            lifecycle_lock.release()

    async def _finish_restore(
        self,
        candidate: Path,
        deadline: float,
        generation: int,
        pending: tuple[Future[object], ...],
        executor: ThreadPoolExecutor | None,
        active_path: Path,
    ) -> ProfileStoreResult[ProfileRestoreReceipt]:
        """Quiesce old work, run restore on the worker, and publish safely."""

        self._bind_or_check_loop()
        pre_worker_error: BaseException | None = None
        restore_future: Future[ProfileRestoreReceipt] | None = None
        try:
            running = tuple(future for future in pending if not future.done())
            if running:
                wrappers = tuple(asyncio.wrap_future(future) for future in running)
                drain = asyncio.gather(*wrappers, return_exceptions=True)
                remaining = _remaining_seconds(deadline)
                if remaining <= 0:
                    raise _repository_error("restore_failed")
                try:
                    await asyncio.wait_for(asyncio.shield(drain), timeout=remaining)
                except TimeoutError:
                    raise _repository_error("restore_failed") from None

            remaining = _remaining_seconds(deadline)
            if remaining <= 0 or executor is None:
                raise _repository_error("restore_failed")
            restore_future = executor.submit(
                self._worker_restore,
                candidate,
                deadline,
                generation,
                active_path,
            )
        except BaseException as error:
            pre_worker_error = error

        if pre_worker_error is not None:
            with self._state_lock:
                if (
                    self._generation == generation
                    and not self._terminal
                    and self._state is ProfileRepositoryState.RESTORING
                ):
                    self._state = ProfileRepositoryState.OPEN
            if not isinstance(pre_worker_error, Exception):
                raise pre_worker_error
            raise _repository_error("restore_failed")
        assert restore_future is not None

        worker_error: BaseException | None = None
        receipt: ProfileRestoreReceipt | None = None
        try:
            receipt = await asyncio.wrap_future(restore_future)
        except BaseException as error:
            worker_error = error
        if worker_error is not None:
            _raise_operation_error(worker_error)

        with self._state_lock:
            if (
                self._generation != generation
                or self._terminal
                or self._state is not ProfileRepositoryState.OPEN
            ):
                raise _repository_error("stale")
        assert receipt is not None
        return ProfileStoreResult(generation=generation, value=receipt)

    def _worker_create_profile(
        self,
        connection: sqlite3.Connection,
        draft: TTSProfileDraft,
        profile_id: UUID | None,
    ) -> TTSGenerationProfile:
        evidence = _IntegrityEvidence(
            profile_id=profile_id,
            normalized_name=draft.normalized_name,
        )
        return self._worker_transaction(
            connection,
            lambda: self._worker_insert_profile(
                connection,
                draft,
                profile_id,
                evidence,
            ),
            operation_kind="create",
            immediate=True,
            integrity_evidence=evidence,
        )

    def _worker_insert_profile(
        self,
        connection: sqlite3.Connection,
        draft: TTSProfileDraft,
        profile_id: UUID | None,
        evidence: _IntegrityEvidence,
    ) -> TTSGenerationProfile:
        persisted_id = profile_id if profile_id is not None else self._worker_new_uuid()
        evidence.profile_id = persisted_id
        timestamp = self._clock()
        profile = TTSGenerationProfile(
            profile_id=persisted_id,
            display_name=draft.display_name,
            normalized_name=draft.normalized_name,
            provider_id=draft.provider_id,
            model_id=draft.model_id,
            voice_id=draft.voice_id,
            response_format=draft.response_format,
            speed=draft.speed,
            options=cast(FrozenJsonOptions, draft.options),
            revision=1,
            created_at=timestamp,
            updated_at=timestamp,
        )
        parameters = encode_profile(profile)
        try:
            connection.execute(
                """
                INSERT INTO tts_generation_profiles (
                    profile_id,
                    display_name,
                    normalized_name,
                    provider_id,
                    model_id,
                    voice_id,
                    response_format,
                    speed,
                    options_json,
                    revision,
                    created_at,
                    updated_at
                ) VALUES (
                    :profile_id,
                    :display_name,
                    :normalized_name,
                    :provider_id,
                    :model_id,
                    :voice_id,
                    :response_format,
                    :speed,
                    :options_json,
                    :revision,
                    :created_at,
                    :updated_at
                )
                """,
                parameters,
            )
        except sqlite3.IntegrityError as error:
            evidence.statement_error = error
            raise
        return self._worker_require_round_trip(connection, persisted_id, profile)

    def _worker_create_profile_with_assignment(
        self,
        connection: sqlite3.Connection,
        draft: TTSProfileDraft,
        profile_id: UUID,
        character_ref: CharacterRef,
        expected_generation: int,
        expected_current_profile_id: UUID | None,
    ) -> AssignedTTSProfileSnapshot:
        evidence = _IntegrityEvidence(
            profile_id=profile_id,
            normalized_name=draft.normalized_name,
        )

        def create_and_assign() -> AssignedTTSProfileSnapshot:
            self._worker_require_generation(expected_generation)
            profile = self._worker_insert_profile(
                connection,
                draft,
                profile_id,
                evidence,
            )
            assignment = self._worker_set_assignment_exact(
                connection,
                character_ref,
                profile.profile_id,
                expected_generation,
                profile.revision,
                expected_current_profile_id,
                profile,
            )
            return AssignedTTSProfileSnapshot(
                assignment=assignment,
                profile=profile,
            )

        return self._worker_transaction(
            connection,
            create_and_assign,
            operation_kind="create",
            immediate=True,
            integrity_evidence=evidence,
        )

    def _worker_get_profile(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
    ) -> TTSGenerationProfile:
        row = connection.execute(
            f"{_PROFILE_SELECT} WHERE profile_id = ?",
            (encode_uuid(profile_id),),
        ).fetchone()
        if row is None:
            raise _repository_error("missing")
        return decode_profile(row)

    def _worker_get_profile_collisions(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        normalized_name: str,
    ) -> TTSProfileCollisionSnapshot:
        def read_collisions() -> TTSProfileCollisionSnapshot:
            profile_id_row = connection.execute(
                f"{_PROFILE_SELECT} WHERE profile_id = ?",
                (encode_uuid(profile_id),),
            ).fetchone()
            normalized_name_row = connection.execute(
                f"{_PROFILE_SELECT} WHERE normalized_name = ?",
                (normalized_name,),
            ).fetchone()
            return TTSProfileCollisionSnapshot(
                profile_id_match=(
                    None if profile_id_row is None else decode_profile(profile_id_row)
                ),
                normalized_name_match=(
                    None
                    if normalized_name_row is None
                    else decode_profile(normalized_name_row)
                ),
            )

        return self._worker_transaction(
            connection,
            read_collisions,
            operation_kind="read",
            immediate=False,
        )

    def _worker_list_profiles(
        self,
        connection: sqlite3.Connection,
        normalized_search: str | None,
        limit: int,
        offset: int,
    ) -> TTSProfilePage:
        def read_page() -> TTSProfilePage:
            if normalized_search is None:
                where_clause = ""
                filter_parameters: tuple[object, ...] = ()
            else:
                where_clause = " WHERE normalized_name LIKE ? ESCAPE '!'"
                filter_parameters = (f"%{_escape_like_literal(normalized_search)}%",)

            count_row = connection.execute(
                f"SELECT COUNT(*) FROM tts_generation_profiles{where_clause}",
                filter_parameters,
            ).fetchone()
            if (
                count_row is None
                or len(count_row) != 1
                or type(count_row[0]) is not int
                or count_row[0] < 0
            ):
                raise _repository_error("corrupt_data")
            total = cast(int, count_row[0])
            rows = connection.execute(
                (
                    f"{_PROFILE_SELECT}{where_clause} "
                    "ORDER BY normalized_name ASC, profile_id ASC "
                    "LIMIT ? OFFSET ?"
                ),
                (*filter_parameters, limit, offset),
            ).fetchall()
            profiles = tuple(decode_profile(row) for row in rows)
            return TTSProfilePage(profiles=profiles, total=total)

        return self._worker_transaction(
            connection,
            read_page,
            operation_kind="read",
            immediate=False,
        )

    def _worker_update_profile(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        expected_revision: int,
        draft: TTSProfileDraft,
    ) -> TTSGenerationProfile:
        evidence = _IntegrityEvidence(
            profile_id=profile_id,
            normalized_name=draft.normalized_name,
        )

        def update() -> TTSGenerationProfile:
            stored = self._worker_get_profile(connection, profile_id)
            if stored.revision != expected_revision:
                raise _repository_error("conflict")
            updated = TTSGenerationProfile(
                profile_id=profile_id,
                display_name=draft.display_name,
                normalized_name=draft.normalized_name,
                provider_id=draft.provider_id,
                model_id=draft.model_id,
                voice_id=draft.voice_id,
                response_format=draft.response_format,
                speed=draft.speed,
                options=cast(FrozenJsonOptions, draft.options),
                revision=stored.revision + 1,
                created_at=stored.created_at,
                updated_at=self._clock(),
            )
            parameters = encode_profile(updated)
            parameters["expected_revision"] = expected_revision
            try:
                cursor = connection.execute(
                    """
                    UPDATE tts_generation_profiles
                    SET
                        display_name = :display_name,
                        normalized_name = :normalized_name,
                        provider_id = :provider_id,
                        model_id = :model_id,
                        voice_id = :voice_id,
                        response_format = :response_format,
                        speed = :speed,
                        options_json = :options_json,
                        revision = :revision,
                        updated_at = :updated_at
                    WHERE profile_id = :profile_id
                        AND revision = :expected_revision
                    """,
                    parameters,
                )
            except sqlite3.IntegrityError as error:
                evidence.statement_error = error
                raise
            if cursor.rowcount != 1:
                raise _repository_error("conflict")
            return self._worker_require_round_trip(
                connection,
                profile_id,
                updated,
            )

        return self._worker_transaction(
            connection,
            update,
            operation_kind="update",
            immediate=True,
            integrity_evidence=evidence,
        )

    def _worker_delete_profile(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
    ) -> None:
        evidence = _IntegrityEvidence(profile_id=profile_id)

        def delete() -> None:
            self._worker_get_profile(connection, profile_id)
            encoded_profile_id = encode_uuid(profile_id)
            try:
                cursor = connection.execute(
                    "DELETE FROM tts_generation_profiles WHERE profile_id = ?",
                    (encoded_profile_id,),
                )
            except sqlite3.IntegrityError as error:
                evidence.statement_error = error
                raise
            if cursor.rowcount == 0:
                raise _repository_error("missing")
            if cursor.rowcount != 1:
                raise _repository_error("corrupt_data")

        self._worker_transaction(
            connection,
            delete,
            operation_kind="delete",
            immediate=True,
            integrity_evidence=evidence,
        )

    def _worker_assignment_count(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
    ) -> int:
        def count() -> int:
            self._worker_get_profile(connection, profile_id)
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM character_tts_assignments
                WHERE profile_id = ?
                """,
                (encode_uuid(profile_id),),
            ).fetchone()
            if row is None or len(row) != 1 or type(row[0]) is not int or row[0] < 0:
                raise _repository_error("corrupt_data")
            return cast(int, row[0])

        return self._worker_transaction(
            connection,
            count,
            operation_kind="read",
            immediate=False,
        )

    def _worker_require_generation(self, expected_generation: int) -> None:
        with self._state_lock:
            state_error = self._worker_state_error_locked(expected_generation)
        if state_error is not None:
            raise _repository_error(state_error)

    def _worker_set_assignment(
        self,
        connection: sqlite3.Connection,
        character_ref: CharacterRef,
        profile_id: UUID,
        expected_generation: int,
        expected_profile_revision: int,
        expected_current_profile_id: UUID | None,
        expected_profile: TTSGenerationProfile | None,
    ) -> CharacterTTSAssignment:
        return self._worker_transaction(
            connection,
            lambda: self._worker_set_assignment_exact(
                connection,
                character_ref,
                profile_id,
                expected_generation,
                expected_profile_revision,
                expected_current_profile_id,
                expected_profile,
            ),
            operation_kind="assignment_set",
            immediate=True,
        )

    def _worker_set_assignment_exact(
        self,
        connection: sqlite3.Connection,
        character_ref: CharacterRef,
        profile_id: UUID,
        expected_generation: int,
        expected_profile_revision: int,
        expected_current_profile_id: UUID | None,
        expected_profile: TTSGenerationProfile | None,
    ) -> CharacterTTSAssignment:
        self._worker_require_generation(expected_generation)
        selected_profile = self._worker_get_profile(connection, profile_id)
        if selected_profile.revision != expected_profile_revision:
            raise _repository_error("conflict")
        if expected_profile is not None and selected_profile != expected_profile:
            raise _repository_error("conflict")
        existing = self._worker_get_persisted_assignment(connection, character_ref)
        current_profile_id = None if existing is None else existing.assignment.profile_id
        if current_profile_id != expected_current_profile_id:
            raise _repository_error("conflict")
        assignment = CharacterTTSAssignment(
            character_ref=character_ref,
            profile_id=profile_id,
        )
        timestamp = self._clock()
        created_at = timestamp if existing is None else existing.created_at
        updated_at = timestamp if existing is None else max(existing.updated_at, timestamp)
        expected = _PersistedAssignment(
            assignment=assignment,
            created_at=created_at,
            updated_at=updated_at,
        )
        parameters = encode_assignment(
            assignment,
            created_at=created_at,
            updated_at=updated_at,
        )
        cursor = connection.execute(
            """
            INSERT INTO character_tts_assignments (
                source,
                authority_id,
                character_id,
                profile_id,
                created_at,
                updated_at
            ) VALUES (
                :source,
                :authority_id,
                :character_id,
                :profile_id,
                :created_at,
                :updated_at
            )
            ON CONFLICT(source, authority_id, character_id)
            DO UPDATE SET
                profile_id = excluded.profile_id,
                updated_at = excluded.updated_at
            """,
            parameters,
        )
        if cursor.rowcount != 1:
            raise _repository_error("corrupt_data")
        persisted = self._worker_get_persisted_assignment(connection, character_ref)
        if persisted != expected:
            raise _repository_error("corrupt_data")
        return persisted.assignment

    def _worker_remove_assignment(
        self,
        connection: sqlite3.Connection,
        character_ref: CharacterRef,
        expected_generation: int,
        expected_profile_id: UUID,
    ) -> None:
        def remove_exact() -> None:
            self._worker_require_generation(expected_generation)
            existing = self._worker_get_persisted_assignment(
                connection,
                character_ref,
            )
            if existing is None:
                return
            if existing.assignment.profile_id != expected_profile_id:
                raise _repository_error("conflict")
            cursor = connection.execute(
                """
                DELETE FROM character_tts_assignments
                WHERE source = ?
                    AND authority_id = ?
                    AND character_id = ?
                    AND profile_id = ?
                """,
                (
                    character_ref.source,
                    character_ref.authority_id,
                    character_ref.character_id,
                    encode_uuid(expected_profile_id),
                ),
            )
            if cursor.rowcount != 1:
                raise _repository_error("corrupt_data")
            if (
                self._worker_get_persisted_assignment(connection, character_ref)
                is not None
            ):
                raise _repository_error("corrupt_data")

        self._worker_transaction(
            connection,
            remove_exact,
            operation_kind="assignment_remove",
            immediate=True,
        )

    def _worker_get_persisted_assignment(
        self,
        connection: sqlite3.Connection,
        character_ref: CharacterRef,
    ) -> _PersistedAssignment | None:
        row = connection.execute(
            (
                f"{_ASSIGNMENT_SELECT} "
                "WHERE source = ? AND authority_id = ? AND character_id = ?"
            ),
            (
                character_ref.source,
                character_ref.authority_id,
                character_ref.character_id,
            ),
        ).fetchone()
        if row is None:
            return None
        assignment = decode_assignment(row)
        if assignment.character_ref != character_ref:
            raise _repository_error("corrupt_data")
        created_at = decode_utc_datetime(row["created_at"])
        updated_at = decode_utc_datetime(row["updated_at"])
        if created_at > updated_at:
            raise _repository_error("corrupt_data")
        return _PersistedAssignment(
            assignment=assignment,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _worker_get_assigned_profile(
        self,
        connection: sqlite3.Connection,
        character_ref: CharacterRef,
    ) -> AssignedTTSProfileSnapshot | None:
        row = connection.execute(
            (
                f"{ASSIGNED_PROFILE_JOIN_SELECT} "
                "WHERE a.source = ? "
                "AND a.authority_id = ? "
                "AND a.character_id = ?"
            ),
            (
                character_ref.source,
                character_ref.authority_id,
                character_ref.character_id,
            ),
        ).fetchone()
        if row is None:
            return None
        snapshot = decode_assigned_snapshot(row)
        if snapshot.assignment.character_ref != character_ref:
            raise _repository_error("corrupt_data")
        return snapshot

    def _worker_require_round_trip(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        expected: TTSGenerationProfile,
    ) -> TTSGenerationProfile:
        decoded = self._worker_get_profile(connection, profile_id)
        if decoded != expected:
            raise _repository_error("corrupt_data")
        return decoded

    def _worker_new_uuid(self) -> UUID:
        generated = self._uuid_factory()
        if type(generated) is not UUID or generated.version != 4:
            raise _repository_error("operation_failed")
        return generated

    def _worker_transaction(
        self,
        connection: sqlite3.Connection,
        operation: Callable[[], _T],
        *,
        operation_kind: _TransactionOperation,
        immediate: bool,
        integrity_evidence: _IntegrityEvidence | None = None,
    ) -> _T:
        body_error: BaseException | None = None
        value: _T | None = None
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            value = operation()
            self._commit_transaction(connection)
        except BaseException as error:
            body_error = error

        if body_error is None:
            return cast(_T, value)

        integrity_conflict = False
        classification_error: BaseException | None = None
        if (
            isinstance(body_error, sqlite3.IntegrityError)
            and integrity_evidence is not None
            and integrity_evidence.statement_error is body_error
        ):
            try:
                integrity_conflict = _has_integrity_conflict_evidence(
                    connection,
                    body_error,
                    operation_kind,
                    integrity_evidence,
                )
            except BaseException as error:
                classification_error = error

        rollback_error: BaseException | None = None
        try:
            self._rollback_transaction(connection)
            if connection.in_transaction:
                raise _repository_error("operation_failed")
        except BaseException as error:
            rollback_error = error

        cleanup_error: BaseException | None = None
        if rollback_error is not None:
            with self._state_lock:
                if not self._terminal and self._state is ProfileRepositoryState.OPEN:
                    self._state = ProfileRepositoryState.UNAVAILABLE
            try:
                self._worker_cleanup()
            except BaseException as error:
                cleanup_error = error

        if isinstance(body_error, sqlite3.IntegrityError):
            if classification_error is not None:
                body_error = classification_error
            else:
                body_error = _repository_error(
                    "conflict" if integrity_conflict else "operation_failed"
                )
        _raise_with_cleanup_precedence(
            body_error,
            rollback_error,
            cleanup_error,
        )
        raise AssertionError("unreachable")

    def _commit_transaction(self, connection: sqlite3.Connection) -> None:
        """Commit one worker-owned transaction.

        This small boundary also permits deterministic fault injection without
        exposing a public repository test hook.
        """

        connection.commit()

    def _rollback_transaction(self, connection: sqlite3.Connection) -> None:
        """Roll back one worker-owned transaction."""

        connection.rollback()

    async def _submit_operation(
        self,
        operation: Callable[[sqlite3.Connection], _T],
        *,
        expected_generation: int | None = None,
    ) -> ProfileStoreResult[_T]:
        """Submit and publish one normal generation-bound operation."""

        self._bind_or_check_loop()
        admission = self._admit_operation(
            operation,
            expected_generation=expected_generation,
        )
        return await self._publish_operation(admission)

    def _admit_operation(
        self,
        operation: Callable[[sqlite3.Connection], _T],
        *,
        expected_generation: int | None = None,
    ) -> _OperationAdmission[_T]:
        """Synchronously capture state/generation and register a worker future."""

        self._bind_or_check_loop()
        if not callable(operation):
            raise _repository_error("operation_failed")

        submission_error: BaseException | None = None
        future: Future[_T] | None = None
        with self._state_lock:
            state_error = self._normal_state_error_locked()
            if state_error is not None:
                raise _repository_error(state_error)
            generation = self._generation
            if expected_generation is not None and expected_generation != generation:
                raise _repository_error("stale")
            executor = self._executor
            if executor is None or self._executor_shutdown:
                raise _repository_error("invalid_state")
            try:
                future = executor.submit(
                    self._worker_operation,
                    generation,
                    operation,
                )
            except BaseException as error:
                submission_error = error
            if future is not None:
                self._pending_futures.add(cast(Future[object], future))

        if submission_error is not None:
            _raise_operation_error(submission_error)
        assert future is not None
        future.add_done_callback(self._discard_pending_future)
        return _OperationAdmission(generation=generation, future=future)

    def _normal_state_error_locked(self) -> str | None:
        if self._terminal:
            return "terminal"
        if self._state is ProfileRepositoryState.CLOSED:
            return "closed"
        if self._state is ProfileRepositoryState.RESTORING:
            return "restoring"
        if self._state is ProfileRepositoryState.UNAVAILABLE:
            return "unavailable"
        if self._state is not ProfileRepositoryState.OPEN:
            return "invalid_state"
        return None

    def _discard_pending_future(self, future: Future[_T]) -> None:
        with self._state_lock:
            self._pending_futures.discard(cast(Future[object], future))

    def _worker_operation(
        self,
        generation: int,
        operation: Callable[[sqlite3.Connection], _T],
    ) -> _T:
        """Check freshness immediately before invoking one SQLite operation."""

        with self._state_lock:
            state_error = self._worker_state_error_locked(generation)
            connection = self._connection
        if state_error is not None:
            raise _repository_error(state_error)
        if connection is None:
            raise _repository_error("invalid_state")

        operation_error: BaseException | None = None
        value: _T | None = None
        try:
            value = operation(connection)
        except BaseException as error:
            operation_error = error
        if operation_error is not None:
            _raise_operation_error(operation_error)
        return cast(_T, value)

    def _worker_state_error_locked(self, generation: int) -> str | None:
        if generation != self._generation:
            return "stale"
        return self._normal_state_error_locked()

    async def _publish_operation(
        self,
        admission: _OperationAdmission[_T],
    ) -> ProfileStoreResult[_T]:
        """Await a shielded worker future and publish only if it remains current."""

        self._bind_or_check_loop()
        wrapped_future = asyncio.wrap_future(admission.future)
        wrapped_future.add_done_callback(_retrieve_future_exception)
        worker_cancelled = False
        worker_error: BaseException | None = None
        try:
            value = await asyncio.shield(wrapped_future)
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling() > 0:
                raise
            worker_cancelled = wrapped_future.cancelled()
            if not worker_cancelled:
                raise
            value = cast(_T, None)
        except BaseException as error:
            worker_error = error
            value = cast(_T, None)

        if worker_cancelled:
            raise _repository_error("stale")
        if worker_error is not None:
            _raise_operation_error(worker_error)

        with self._state_lock:
            state_error = self._worker_state_error_locked(admission.generation)
        if state_error is not None:
            raise _repository_error(state_error)
        return ProfileStoreResult(
            generation=admission.generation,
            value=value,
        )

    async def close(self) -> ProfileStoreResult[None]:
        """Definitively close the repository and shut down its worker once."""

        lifecycle_lock = self._bind_or_check_loop()
        async with lifecycle_lock:
            with self._state_lock:
                if self._terminal:
                    return ProfileStoreResult(
                        generation=self._generation,
                        value=None,
                    )
                self._generation += 1
                generation = self._generation
                self._terminal = True
                self._state = ProfileRepositoryState.CLOSED
                executor = self._executor
                pending = tuple(self._pending_futures)

            for future in pending:
                future.cancel()

            if executor is None:
                return ProfileStoreResult(generation=generation, value=None)

            completion = asyncio.create_task(self._finish_close(executor, pending))
            await self._await_lifecycle_completion(completion)
            return ProfileStoreResult(generation=generation, value=None)

    async def _finish_close(
        self,
        executor: ThreadPoolExecutor,
        pending: tuple[Future[object], ...],
    ) -> None:
        """Drain admitted work, clean worker ownership, and shut down off-loop."""

        self._bind_or_check_loop()
        if pending:
            await asyncio.gather(
                *(asyncio.shield(asyncio.wrap_future(future)) for future in pending),
                return_exceptions=True,
            )

        cleanup_error: BaseException | None = None
        cleanup_future: Future[None] | None = None
        try:
            cleanup_future = executor.submit(self._worker_cleanup)
        except BaseException as error:
            cleanup_error = error

        if cleanup_future is not None:
            try:
                await asyncio.wrap_future(cleanup_future)
            except BaseException as error:
                cleanup_error = error

        shutdown_error: BaseException | None = None
        with self._state_lock:
            self._executor_shutdown = True
        try:
            await asyncio.to_thread(
                executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
        except BaseException as error:
            shutdown_error = error
        finally:
            with self._state_lock:
                if self._executor is executor:
                    self._executor = None

        _raise_cleanup_errors(cleanup_error, shutdown_error)

    def _worker_cleanup(self) -> None:
        """Close SQLite before releasing the shared lease on the worker."""

        connection = self._connection
        lease = self._lease
        connection_error: BaseException | None = None
        lease_error: BaseException | None = None

        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                connection_error = error
            if connection_error is None:
                self._connection = None

        if lease is not None and connection_error is None:
            try:
                lease.release()
            except BaseException as error:
                lease_error = error
            if lease_error is None:
                self._lease = None

        if self._connection is None and self._lease is None:
            self._active_database_path = None

        _raise_cleanup_errors(connection_error, lease_error)

    async def _await_lifecycle_completion(
        self,
        completion: asyncio.Task[_T],
    ) -> _T:
        """Delay caller cancellation until a lifecycle transition settles."""

        self._bind_or_check_loop()
        cancellation: asyncio.CancelledError | None = None
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except BaseException:
                break

        completion_error: BaseException | None = None
        result: _T | None = None
        try:
            result = completion.result()
        except BaseException as error:
            completion_error = error

        if cancellation is not None:
            raise cancellation
        if completion_error is not None:
            _raise_operation_error(completion_error)
        return cast(_T, result)

    def _bind_or_check_loop(self) -> asyncio.Lock:
        """Bind first async use and reject every later foreign-loop caller."""

        running_loop: asyncio.AbstractEventLoop | None = None
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if running_loop is None:
            raise _repository_error("invalid_state")

        wrong_loop = False
        lifecycle_lock: asyncio.Lock | None = None
        with self._state_lock:
            if self._owner_loop is None:
                lifecycle_lock = asyncio.Lock()
                self._owner_loop = running_loop
                self._lifecycle_lock = lifecycle_lock
            elif self._owner_loop is not running_loop:
                wrong_loop = True
            else:
                lifecycle_lock = self._lifecycle_lock

        if wrong_loop or lifecycle_lock is None:
            raise _repository_error("invalid_state")
        return lifecycle_lock
