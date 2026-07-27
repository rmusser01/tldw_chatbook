"""Serialized lifecycle owner for the local TTS generation-profile store."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast
from unicodedata import category as _unicode_category
from unicodedata import normalize as _unicode_normalize
from uuid import UUID, uuid4

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_schema import (
    decode_profile,
    encode_profile,
    encode_uuid,
    open_profile_store,
)
from tldw_chatbook.TTS.profile_store_lock import (
    ProfileStoreLease,
    ProfileStoreLockMode,
)
from tldw_chatbook.TTS.profile_types import (
    ProfileRepositoryState,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileDraft,
    TTSProfilePage,
)


_T = TypeVar("_T")
_PATH_TYPE = type(Path())
_MAX_SEARCH_CHARACTERS = 128
_MAX_NORMALIZED_SEARCH_CHARACTERS = 512
_MAX_NORMALIZED_SEARCH_BYTES = 2_048
_UNSAFE_SEARCH_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_unicode_ord = ord
# SQLite extended result codes are ABI-stable.  Keeping the exact values here
# also supports Python builds that do not expose every named sqlite3 constant.
_SQLITE_CONSTRAINT_FOREIGNKEY = 787
_SQLITE_CONSTRAINT_PRIMARYKEY = 1_555
_SQLITE_CONSTRAINT_TRIGGER = 1_811
_SQLITE_CONSTRAINT_UNIQUE = 2_067
_TransactionOperation = Literal["create", "read", "update", "delete"]
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


@dataclass(frozen=True, slots=True)
class _OperationAdmission(Generic[_T]):
    """One generation-bound worker submission awaiting publication."""

    generation: int
    future: Future[_T]


@dataclass(slots=True)
class _IntegrityEvidence:
    """Exact schema-owned values permitted to support one conflict."""

    profile_id: UUID | None
    normalized_name: str | None = None


def _repository_error(code: str) -> ProfileRepositoryError:
    return ProfileRepositoryError(code)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_exact_profile_id(value: object) -> UUID:
    if type(value) is not UUID:
        raise _repository_error("operation_failed")
    return cast(UUID, value)


def _validate_optional_profile_id(value: object) -> UUID | None:
    if value is None:
        return None
    return _validate_exact_profile_id(value)


def _validate_draft(value: object) -> TTSProfileDraft:
    if type(value) is not TTSProfileDraft:
        raise _repository_error("operation_failed")
    return cast(TTSProfileDraft, value)


def _validate_expected_revision(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise _repository_error("operation_failed")
    return cast(int, value)


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
        body_error: BaseException | None = None
        try:
            lease = ProfileStoreLease(
                self._database_path,
                ProfileStoreLockMode.SHARED,
            )
            lease.acquire()
            connection = open_profile_store(self._database_path)
            if connection is None:
                raise _repository_error("operation_failed")
        except BaseException as error:
            body_error = error

        if body_error is None:
            assert lease is not None
            self._lease = lease
            self._connection = connection
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
        _raise_with_cleanup_precedence(
            body_error,
            connection_error,
            lease_error,
        )

    async def create_profile(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID | None = None,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Create one immutable profile at revision 1.

        Args:
            draft: Exact validated profile draft.
            profile_id: Optional exact caller-selected UUID. When omitted, the
                repository generates a UUID4 on its serialized worker.

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
        return await self._submit_operation(
            lambda connection: self._worker_create_profile(
                connection,
                validated_draft,
                validated_profile_id,
            )
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
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Replace one profile only at the exact editor revision.

        Args:
            profile_id: Exact profile UUID.
            expected_revision: Exact positive revision loaded by the editor.
            draft: Exact replacement profile draft.

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
        return await self._submit_operation(
            lambda connection: self._worker_update_profile(
                connection,
                validated_profile_id,
                validated_revision,
                validated_draft,
            )
        )

    async def delete_profile(
        self,
        profile_id: UUID,
    ) -> ProfileStoreResult[None]:
        """Delete exactly one unreferenced profile by UUID.

        Args:
            profile_id: Exact profile UUID.

        Returns:
            The active generation paired with ``None``.

        Raises:
            ProfileRepositoryError: If the input or state is invalid, the row
                is missing or referenced, or SQLite access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_profile_id = _validate_exact_profile_id(profile_id)
        return await self._submit_operation(
            lambda connection: self._worker_delete_profile(
                connection,
                validated_profile_id,
            )
        )

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

        def create() -> TTSGenerationProfile:
            persisted_id = (
                profile_id if profile_id is not None else self._worker_new_uuid()
            )
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
                options=draft.options,
                revision=1,
                created_at=timestamp,
                updated_at=timestamp,
            )
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
                encode_profile(profile),
            )
            return self._worker_require_round_trip(
                connection,
                persisted_id,
                profile,
            )

        return self._worker_transaction(
            connection,
            create,
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
                options=draft.options,
                revision=stored.revision + 1,
                created_at=stored.created_at,
                updated_at=self._clock(),
            )
            parameters = encode_profile(updated)
            parameters["expected_revision"] = expected_revision
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
            integrity_evidence=_IntegrityEvidence(
                profile_id=profile_id,
                normalized_name=draft.normalized_name,
            ),
        )

    def _worker_delete_profile(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
    ) -> None:
        def delete() -> None:
            self._worker_get_profile(connection, profile_id)
            cursor = connection.execute(
                "DELETE FROM tts_generation_profiles WHERE profile_id = ?",
                (encode_uuid(profile_id),),
            )
            if cursor.rowcount == 0:
                raise _repository_error("missing")
            if cursor.rowcount != 1:
                raise _repository_error("corrupt_data")

        self._worker_transaction(
            connection,
            delete,
            operation_kind="delete",
            immediate=True,
            integrity_evidence=_IntegrityEvidence(profile_id=profile_id),
        )

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
        if isinstance(body_error, sqlite3.IntegrityError):
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
    ) -> ProfileStoreResult[_T]:
        """Submit and publish one normal generation-bound operation."""

        self._bind_or_check_loop()
        admission = self._admit_operation(operation)
        return await self._publish_operation(admission)

    def _admit_operation(
        self,
        operation: Callable[[sqlite3.Connection], _T],
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
