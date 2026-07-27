"""Transactional CRUD tests for the serialized TTS profile repository."""

from __future__ import annotations

import asyncio
import multiprocessing
import sqlite3
import threading
import traceback
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

import tldw_chatbook.TTS.profile_repository as profile_repository
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_schema import encode_assignment
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    ProfileRepositoryState,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileDraft,
    TTSProfilePage,
    canonical_json_options,
)


CREATED_AT = datetime(2026, 7, 26, 12, 0, 0, 123456, tzinfo=UTC)
UPDATED_AT = datetime(2026, 7, 26, 13, 30, 0, 654321, tzinfo=UTC)
LATER_AT = datetime(2026, 7, 26, 14, 45, 0, 111111, tzinfo=UTC)
GENERATED_ID = UUID("10000000-0000-4000-8000-000000000001")
CALLER_ID = UUID("20000000-0000-0000-8000-000000000002")


class _ControlFlow(BaseException):
    """Test-only caller control-flow signal."""


class _HostileSQLiteErrorCode(sqlite3.IntegrityError):
    """Integrity error whose extended-code lookup raises a supplied signal."""

    def __init__(self, signal: BaseException) -> None:
        super().__init__("UNIQUE constraint failed: secret-hostile-code-message")
        self._signal = signal

    def __getattribute__(self, name: str) -> Any:
        if name == "sqlite_errorcode":
            raise object.__getattribute__(self, "_signal")
        return super().__getattribute__(name)


class _SequenceCallable:
    """Return a deterministic sequence and fail if code consumes too much."""

    def __init__(self, values: Iterator[Any]) -> None:
        self._values = values

    def __call__(self) -> Any:
        return next(self._values)


class _TextSubclass(str):
    """An inexact public-boundary string."""


class _CharacterRefSubclass(CharacterRef):
    """An inexact public-boundary character reference."""


class _UUIDSubclass(UUID):
    """An inexact public-boundary UUID."""


class _FalseyCallable:
    """A private seam whose truth value must not be inspected."""

    def __init__(self, value: object) -> None:
        self._value = value

    def __call__(self) -> object:
        return self._value

    def __bool__(self) -> bool:
        raise AssertionError("constructor inspected callable truthiness")


class _BeginIntegrityConnection:
    """Minimal transaction connection that fails before its body can run."""

    def __init__(self, error: sqlite3.IntegrityError) -> None:
        self.error = error
        self.in_transaction = False
        self.rollback_attempted = False

    def execute(self, statement: str) -> None:
        assert statement in {"BEGIN", "BEGIN IMMEDIATE"}
        raise self.error

    def rollback(self) -> None:
        self.rollback_attempted = True


class _StaticCursor:
    """Minimal cursor returning one controlled row."""

    def __init__(self, row: object) -> None:
        self._row = row

    def fetchone(self) -> object:
        return self._row


class _CountRowConnection:
    """Delegate SQLite except for the assignment-count result row."""

    def __init__(self, connection: sqlite3.Connection, row: object) -> None:
        self._connection = connection
        self._row = row

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    def execute(
        self,
        statement: str,
        parameters: tuple[object, ...] = (),
    ) -> object:
        if "SELECT COUNT(*) FROM character_tts_assignments" in " ".join(
            statement.split()
        ):
            return _StaticCursor(self._row)
        return self._connection.execute(statement, parameters)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


def _draft(
    display_name: str = "Narrator",
    *,
    provider_id: str = "openai",
    model_id: str = "tts-1-hd/音声",
    voice_id: str | None = "alloy/声",
    response_format: str = "mp3",
    speed: float = 1.25,
    options: object | None = None,
) -> TTSProfileDraft:
    return TTSProfileDraft(
        display_name=display_name,
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        response_format=response_format,
        speed=speed,
        options=cast(
            Any,
            {
                "nested": {"items": [True, 2, 3.5, None]},
                "locale": "日本語",
            }
            if options is None
            else options,
        ),
    )


@asynccontextmanager
async def _opened_repository(
    database_path: Path,
    *,
    clock: Callable[[], datetime] | None = None,
    uuid_factory: Callable[[], UUID] | None = None,
) -> AsyncIterator[profile_repository.TTSProfileRepository]:
    repository = profile_repository.TTSProfileRepository(
        database_path,
        _clock=clock or (lambda: CREATED_AT),
        _uuid_factory=uuid_factory or (lambda: GENERATED_ID),
    )
    await repository.open()
    try:
        yield repository
    finally:
        await repository.close()


def _assert_safe_error(
    error: ProfileRepositoryError,
    code: str,
    *secrets: str,
) -> None:
    assert type(error) is ProfileRepositoryError
    assert error.code == code
    assert str(error) == f"TTS profile repository failed: {code}"
    assert error.__cause__ is None
    assert error.__context__ is None
    visible = " ".join(
        (
            str(error),
            repr(error),
            "".join(traceback.format_exception(error)),
            *(str(note) for note in getattr(error, "__notes__", ())),
        )
    )
    for secret in secrets:
        if secret:
            assert secret not in visible


def _external_execute(
    database_path: Path,
    statement: str,
    parameters: tuple[object, ...] = (),
) -> None:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(statement, parameters)
    finally:
        connection.close()


def _external_insert_assignment(
    database_path: Path,
    profile_id: UUID,
    character_id: str,
) -> None:
    assignment = CharacterTTSAssignment(
        character_ref=CharacterRef(
            source="local",
            authority_id="barrier-authority",
            character_id=character_id,
        ),
        profile_id=profile_id,
    )
    encoded = encode_assignment(
        assignment,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )
    _external_execute(
        database_path,
        """
        INSERT INTO character_tts_assignments (
            source, authority_id, character_id, profile_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            encoded["source"],
            encoded["authority_id"],
            encoded["character_id"],
            encoded["profile_id"],
            encoded["created_at"],
            encoded["updated_at"],
        ),
    )


def _install_insert_constraint_trigger(
    database_path: Path,
    constraint_kind: str,
) -> None:
    scripts = {
        "notnull": """
            CREATE TABLE constraint_probe(value TEXT NOT NULL);
            CREATE TRIGGER force_profile_insert_constraint
            BEFORE INSERT ON tts_generation_profiles
            BEGIN
                INSERT INTO constraint_probe(value) VALUES (NULL);
            END;
        """,
        "check": """
            CREATE TABLE constraint_probe(value INTEGER CHECK(value > 0));
            CREATE TRIGGER force_profile_insert_constraint
            BEFORE INSERT ON tts_generation_profiles
            BEGIN
                INSERT INTO constraint_probe(value) VALUES (-1);
            END;
        """,
        "trigger": """
            CREATE TRIGGER force_profile_insert_constraint
            BEFORE INSERT ON tts_generation_profiles
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'UNIQUE constraint failed: secret-real-trigger-message'
                );
            END;
        """,
        "datatype": """
            CREATE TABLE constraint_probe(value INTEGER) STRICT;
            CREATE TRIGGER force_profile_insert_constraint
            BEFORE INSERT ON tts_generation_profiles
            BEGIN
                INSERT INTO constraint_probe(value) VALUES ('not-an-integer');
            END;
        """,
        "rowid": """
            CREATE TABLE constraint_probe(value TEXT);
            INSERT INTO constraint_probe(rowid, value) VALUES (1, 'first');
            CREATE TRIGGER force_profile_insert_constraint
            BEFORE INSERT ON tts_generation_profiles
            BEGIN
                INSERT INTO constraint_probe(rowid, value) VALUES (1, 'duplicate');
            END;
        """,
        "foreignkey": """
            CREATE TABLE constraint_parent(id INTEGER PRIMARY KEY);
            CREATE TABLE constraint_probe(
                parent_id INTEGER REFERENCES constraint_parent(id)
            );
            CREATE TRIGGER force_profile_insert_constraint
            BEFORE INSERT ON tts_generation_profiles
            BEGIN
                INSERT INTO constraint_probe(parent_id) VALUES (1);
            END;
        """,
    }
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(scripts[constraint_kind])
    finally:
        connection.close()


def _install_unrelated_uniqueness_trigger(
    database_path: Path,
    operation: str,
    constraint_kind: str,
) -> None:
    trigger_operation = {"create": "INSERT", "update": "UPDATE"}[operation]
    if constraint_kind == "unique":
        probe_schema = "value TEXT UNIQUE"
        seed = "INSERT INTO unrelated_conflict_probe(value) VALUES ('occupied');"
        violation = "INSERT INTO unrelated_conflict_probe(value) VALUES ('occupied');"
    else:
        probe_schema = "id INTEGER PRIMARY KEY"
        seed = "INSERT INTO unrelated_conflict_probe(id) VALUES (1);"
        violation = "INSERT INTO unrelated_conflict_probe(id) VALUES (1);"

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.executescript(
            f"""
            CREATE TABLE unrelated_conflict_probe({probe_schema});
            {seed}
            CREATE TRIGGER force_unrelated_profile_constraint
            BEFORE {trigger_operation} ON tts_generation_profiles
            BEGIN
                {violation}
            END;
            """
        )
    finally:
        connection.close()


def _assert_real_insert_constraint_code(
    database_path: Path,
    expected_code: int,
) -> None:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError) as caught:
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "f0000000-0000-4000-8000-00000000000f",
                    "Constraint Probe",
                    "constraint probe",
                    "openai",
                    "tts-1",
                    "alloy",
                    "mp3",
                    1.0,
                    "{}",
                    1,
                    CREATED_AT.isoformat(),
                    CREATED_AT.isoformat(),
                ),
            )
        assert caught.value.sqlite_errorcode == expected_code
    finally:
        connection.close()


def _assert_real_update_constraint_code(
    database_path: Path,
    profile_id: UUID,
    expected_code: int,
) -> None:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        with pytest.raises(sqlite3.IntegrityError) as caught:
            connection.execute(
                """
                UPDATE tts_generation_profiles
                SET display_name = display_name
                WHERE profile_id = ?
                """,
                (str(profile_id),),
            )
        assert caught.value.sqlite_errorcode == expected_code
    finally:
        connection.close()


def _install_no_action_delete_probe(
    database_path: Path,
    profile_id: UUID,
) -> None:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE delete_constraint_probe(
                profile_id TEXT NOT NULL
                    REFERENCES tts_generation_profiles(profile_id)
            )
            """
        )
        connection.execute(
            "INSERT INTO delete_constraint_probe(profile_id) VALUES (?)",
            (str(profile_id),),
        )
    finally:
        connection.close()


def _assert_real_delete_constraint_code(
    database_path: Path,
    profile_id: UUID,
    expected_code: int,
) -> None:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError) as caught:
            connection.execute(
                "DELETE FROM tts_generation_profiles WHERE profile_id = ?",
                (str(profile_id),),
            )
        assert caught.value.sqlite_errorcode == expected_code
    finally:
        connection.close()


def _spawn_profile_collision(
    database_path: str,
    connection: Connection,
    release: Any,
    display_name: str,
) -> None:
    async def create_after_release() -> tuple[object, ...]:
        repository = profile_repository.TTSProfileRepository(Path(database_path))
        opened = False
        try:
            await repository.open()
            opened = True
            connection.send(("ready", None))
            if not release.wait(10.0):
                raise TimeoutError("parent did not release profile collision")
            try:
                created = await repository.create_profile(
                    _draft(display_name),
                    profile_id=GENERATED_ID,
                )
            except ProfileRepositoryError as error:
                return (
                    "failure",
                    error.code,
                    str(error),
                    error.__cause__ is None,
                    error.__context__ is None,
                    tuple(str(note) for note in getattr(error, "__notes__", ())),
                )
            return (
                "success",
                str(created.value.profile_id),
                created.value.normalized_name,
            )
        finally:
            if opened:
                await repository.close()

    try:
        outcome = asyncio.run(create_after_release())
        connection.send(("outcome", outcome))
    except BaseException as error:
        try:
            connection.send(("child_error", type(error).__name__))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


def _spawn_assignment_delete_race(
    database_path: str,
    connection: Connection,
    release: Any,
    operation: str,
    profile_id: str,
) -> None:
    async def race_after_release() -> tuple[object, ...]:
        repository = profile_repository.TTSProfileRepository(Path(database_path))
        opened = False
        try:
            await repository.open()
            opened = True
            connection.send(("ready", operation))
            if not release.wait(10.0):
                raise TimeoutError("parent did not release assignment race")
            try:
                if operation == "set":
                    result = await repository.set_assignment(
                        CharacterRef(
                            source="server",
                            authority_id="server-profile:principal",
                            character_id="shared-character",
                        ),
                        UUID(profile_id),
                    )
                else:
                    result = await repository.delete_profile(UUID(profile_id))
            except ProfileRepositoryError as error:
                return (
                    operation,
                    "failure",
                    error.code,
                    str(error),
                    error.__cause__ is None,
                    error.__context__ is None,
                    tuple(str(note) for note in getattr(error, "__notes__", ())),
                )
            return (operation, "success", result.generation)
        finally:
            if opened:
                await repository.close()

    try:
        outcome = asyncio.run(race_after_release())
        connection.send(("outcome", outcome))
    except BaseException as error:
        try:
            connection.send(("child_error", type(error).__name__))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


async def _initialize_empty_repository(database_path: Path) -> None:
    async with _opened_repository(database_path) as repository:
        assert (await repository.list_profiles()).value.total == 0


async def _assert_spawned_race_store_is_usable(database_path: Path) -> None:
    async with _opened_repository(database_path) as repository:
        page = (await repository.list_profiles()).value
        assert page.total == 1
        assert len(page.profiles) == 1
        winner = page.profiles[0]
        assert winner.profile_id == GENERATED_ID
        assert winner.normalized_name == "race name"
        assert winner.revision == 1

        recovered = await repository.create_profile(
            _draft("After Race"),
            profile_id=CALLER_ID,
        )
        assert recovered.value.profile_id == CALLER_ID
        assert (await repository.get_profile(CALLER_ID)).value == recovered.value


def _fail_next_commit(
    monkeypatch: pytest.MonkeyPatch,
    repository: profile_repository.TTSProfileRepository,
    error: BaseException,
) -> threading.Event:
    original_commit = repository._commit_transaction
    failed = False
    attempted = threading.Event()

    def fail_once(connection: sqlite3.Connection) -> None:
        nonlocal failed
        if not failed:
            failed = True
            attempted.set()
            raise error
        original_commit(connection)

    monkeypatch.setattr(repository, "_commit_transaction", fail_once)
    return attempted


def _fail_next_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    repository: profile_repository.TTSProfileRepository,
    error: BaseException,
) -> threading.Event:
    original_round_trip = repository._worker_require_round_trip
    failed = False
    attempted = threading.Event()

    def fail_once(
        connection: sqlite3.Connection,
        profile_id: UUID,
        expected: TTSGenerationProfile,
    ) -> TTSGenerationProfile:
        nonlocal failed
        if not failed:
            failed = True
            attempted.set()
            raise error
        return original_round_trip(connection, profile_id, expected)

    monkeypatch.setattr(repository, "_worker_require_round_trip", fail_once)
    return attempted


def _expected_integrity_error(code: int, secret: str) -> sqlite3.IntegrityError:
    error = sqlite3.IntegrityError(secret)
    setattr(error, "sqlite_errorcode", code)
    return error


def _record_integrity_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    original_classification = profile_repository._has_integrity_conflict_evidence
    operations: list[str] = []

    def record(
        connection: sqlite3.Connection,
        error: sqlite3.IntegrityError,
        operation_kind: Any,
        evidence: Any,
    ) -> bool:
        operations.append(cast(str, operation_kind))
        return original_classification(
            connection,
            error,
            operation_kind,
            evidence,
        )

    monkeypatch.setattr(
        profile_repository,
        "_has_integrity_conflict_evidence",
        record,
    )
    return operations


def _pause_next_rollback(
    monkeypatch: pytest.MonkeyPatch,
    repository: profile_repository.TTSProfileRepository,
) -> tuple[threading.Event, threading.Event]:
    original_rollback = repository._rollback_transaction
    rolled_back = threading.Event()
    resume = threading.Event()
    paused = False

    def pause_after_rollback(connection: sqlite3.Connection) -> None:
        nonlocal paused
        original_rollback(connection)
        if paused:
            return
        paused = True
        rolled_back.set()
        if not resume.wait(10.0):
            raise RuntimeError("test did not resume rollback")

    monkeypatch.setattr(
        repository,
        "_rollback_transaction",
        pause_after_rollback,
    )
    return rolled_back, resume


def test_private_clock_and_uuid_seams_remain_constructor_pure(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "profiles.sqlite3"

    repository = profile_repository.TTSProfileRepository(
        database_path,
        _clock=cast(Callable[[], datetime], _FalseyCallable(CREATED_AT)),
        _uuid_factory=cast(Callable[[], UUID], _FalseyCallable(GENERATED_ID)),
    )

    assert repository.state.value == "closed"
    assert repository.generation == 0
    assert not database_path.parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "args", "kwargs", "secret"),
    [
        ("create_profile", (object(),), {}, ""),
        (
            "create_profile",
            (_draft(),),
            {"profile_id": "secret-profile-id"},
            "secret-profile-id",
        ),
        ("get_profile", ("secret-get-id",), {}, "secret-get-id"),
        ("get_profile", (GENERATED_ID.bytes,), {}, ""),
        ("list_profiles", (), {"search": object()}, ""),
        (
            "list_profiles",
            (),
            {"search": _TextSubclass("secret-subclass")},
            "secret-subclass",
        ),
        (
            "list_profiles",
            (),
            {"search": "\x00secret-null-search"},
            "secret-null-search",
        ),
        ("list_profiles", (), {"search": "s" * 129}, "s" * 129),
        ("list_profiles", (), {"limit": True}, ""),
        ("list_profiles", (), {"limit": 0}, ""),
        ("list_profiles", (), {"limit": 101}, ""),
        ("list_profiles", (), {"offset": True}, ""),
        ("list_profiles", (), {"offset": -1}, ""),
        ("update_profile", ("secret-update-id", 1, _draft()), {}, "secret-update-id"),
        ("update_profile", (GENERATED_ID, True, _draft()), {}, ""),
        ("update_profile", (GENERATED_ID, 0, _draft()), {}, ""),
        ("update_profile", (GENERATED_ID, 1.0, _draft()), {}, ""),
        ("update_profile", (GENERATED_ID, 1, object()), {}, ""),
        ("delete_profile", ("secret-delete-id",), {}, "secret-delete-id"),
        ("assignment_count", ("secret-count-id",), {}, "secret-count-id"),
        (
            "assignment_count",
            (_UUIDSubclass(str(GENERATED_ID)),),
            {},
            "",
        ),
        (
            "set_assignment",
            (object(), GENERATED_ID),
            {},
            "",
        ),
        (
            "set_assignment",
            (
                _CharacterRefSubclass(
                    source="local",
                    authority_id="secret-authority-subclass",
                    character_id="secret-character-subclass",
                ),
                GENERATED_ID,
            ),
            {},
            "secret-authority-subclass",
        ),
        (
            "set_assignment",
            (
                CharacterRef(
                    source="local",
                    authority_id="secret-authority",
                    character_id="secret-character",
                ),
                "secret-assignment-profile",
            ),
            {},
            "secret-assignment-profile",
        ),
        ("remove_assignment", (object(),), {}, ""),
        (
            "get_assigned_profile",
            (
                _CharacterRefSubclass(
                    source="server",
                    authority_id="secret-server-subclass",
                    character_id="secret-character-subclass",
                ),
            ),
            {},
            "secret-server-subclass",
        ),
    ],
)
async def test_invalid_public_inputs_fail_before_worker_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    secret: str,
) -> None:
    repository = profile_repository.TTSProfileRepository(
        tmp_path / "must-not-exist" / "profiles.sqlite3"
    )
    submitted = False

    async def forbidden_submission(
        _operation: Callable[[sqlite3.Connection], object],
    ) -> ProfileStoreResult[object]:
        nonlocal submitted
        submitted = True
        raise AssertionError("invalid input reached worker submission")

    monkeypatch.setattr(repository, "_submit_operation", forbidden_submission)

    with pytest.raises(ProfileRepositoryError) as caught:
        await getattr(repository, method_name)(*args, **kwargs)

    _assert_safe_error(caught.value, "operation_failed", secret)
    assert submitted is False
    assert not tmp_path.joinpath("must-not-exist").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "include_profile_id"),
    [
        ("set_assignment", True),
        ("remove_assignment", False),
        ("get_assigned_profile", False),
    ],
)
async def test_mutated_exact_character_ref_fails_before_worker_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    include_profile_id: bool,
) -> None:
    secret = " secret-mutated-exact-authority "
    character_ref = CharacterRef(
        source="local",
        authority_id="valid-authority",
        character_id="valid-character",
    )
    object.__setattr__(character_ref, "authority_id", secret)
    repository = profile_repository.TTSProfileRepository(
        tmp_path / "must-not-exist" / "profiles.sqlite3"
    )
    submitted = False

    async def forbidden_submission(
        _operation: Callable[[sqlite3.Connection], object],
    ) -> ProfileStoreResult[object]:
        nonlocal submitted
        submitted = True
        raise AssertionError("malformed exact CharacterRef reached worker submission")

    monkeypatch.setattr(repository, "_submit_operation", forbidden_submission)
    arguments: tuple[object, ...] = (
        (character_ref, GENERATED_ID) if include_profile_id else (character_ref,)
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await getattr(repository, method_name)(*arguments)

    _assert_safe_error(caught.value, "operation_failed", secret)
    assert not submitted
    assert not tmp_path.joinpath("must-not-exist").exists()


@pytest.mark.asyncio
async def test_assignment_snapshots_character_ref_before_queued_worker_runs(
    tmp_path: Path,
) -> None:
    persisted_profile_id = UUID("fd000000-0000-4000-8000-00000000000d")
    caller_profile_id = UUID(str(persisted_profile_id))
    admitted_ref = CharacterRef(
        source="local",
        authority_id="admitted-authority",
        character_id="admitted-character",
    )
    worker_entered = threading.Event()
    worker_resume = threading.Event()

    async with _opened_repository(tmp_path / "queued-snapshot.sqlite3") as repository:
        await repository.create_profile(
            _draft("Queued Snapshot"),
            profile_id=persisted_profile_id,
        )

        def block_worker(_connection: sqlite3.Connection) -> None:
            worker_entered.set()
            if not worker_resume.wait(10.0):
                raise RuntimeError("test did not resume queued worker")

        blocker = repository._admit_operation(block_worker)
        blocker_publication = asyncio.create_task(
            repository._publish_operation(blocker)
        )
        assert await asyncio.to_thread(worker_entered.wait, 10.0)
        queued = asyncio.create_task(
            repository.set_assignment(admitted_ref, caller_profile_id)
        )
        await asyncio.sleep(0)
        object.__setattr__(admitted_ref, "authority_id", "mutated-authority")
        object.__setattr__(admitted_ref, "character_id", "mutated-character")
        object.__setattr__(
            caller_profile_id,
            "int",
            UUID("fd100000-0000-4000-8000-00000000000d").int,
        )
        worker_resume.set()

        await blocker_publication
        assigned = await queued

        expected_ref = CharacterRef(
            source="local",
            authority_id="admitted-authority",
            character_id="admitted-character",
        )
        mutated_ref = CharacterRef(
            source="local",
            authority_id="mutated-authority",
            character_id="mutated-character",
        )
        assert assigned.value == CharacterTTSAssignment(
            expected_ref,
            persisted_profile_id,
        )
        admitted = (await repository.get_assigned_profile(expected_ref)).value
        assert admitted is not None
        assert admitted.assignment.character_ref == expected_ref
        assert (await repository.get_assigned_profile(mutated_ref)).value is None


@pytest.mark.asyncio
async def test_valid_crud_uses_lifecycle_state_lane_before_open(tmp_path: Path) -> None:
    repository = profile_repository.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.get_profile(GENERATED_ID)

    _assert_safe_error(caught.value, "closed")
    assert not (tmp_path / "profiles.sqlite3").exists()


@pytest.mark.asyncio
async def test_hostile_search_normalizer_fails_safely_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-normalizer-store.sqlite3"
    secret = "secret-hostile-unicode-normalizer"
    repository = profile_repository.TTSProfileRepository(database_path)
    submitted = False

    async def forbidden_submission(
        _operation: Callable[[sqlite3.Connection], object],
    ) -> ProfileStoreResult[object]:
        nonlocal submitted
        submitted = True
        raise AssertionError("failed normalization reached worker submission")

    def hostile_normalizer(_form: str, _value: str) -> str:
        try:
            raise RuntimeError(secret)
        except RuntimeError:
            raise ValueError(secret)

    monkeypatch.setattr(repository, "_submit_operation", forbidden_submission)
    monkeypatch.setattr(
        profile_repository,
        "_unicode_normalize",
        hostile_normalizer,
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.list_profiles(search="ordinary")

    _assert_safe_error(
        caught.value,
        "operation_failed",
        secret,
        str(database_path),
    )
    assert submitted is False
    assert not database_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_search", "visible_secret"),
    [
        pytest.param(
            "\tsecret-leading-tab",
            "secret-leading-tab",
            id="leading-tab",
        ),
        pytest.param(
            "\nsecret-leading-newline",
            "secret-leading-newline",
            id="leading-newline",
        ),
        pytest.param(
            "\x01secret-leading-control",
            "secret-leading-control",
            id="leading-control",
        ),
        pytest.param(
            "secret-trailing-tab\t",
            "secret-trailing-tab",
            id="trailing-tab",
        ),
        pytest.param(
            "secret-trailing-newline\n",
            "secret-trailing-newline",
            id="trailing-newline",
        ),
        pytest.param(
            "secret-trailing-control\x1f",
            "secret-trailing-control",
            id="trailing-control",
        ),
        pytest.param("\t \n", "", id="control-only-mixed-space"),
        pytest.param(
            "\u200bsecret-format",
            "secret-format",
            id="format-control",
        ),
        pytest.param(
            "\ud800secret-surrogate",
            "secret-surrogate",
            id="surrogate",
        ),
        pytest.param(
            "\ufdd0secret-noncharacter-start",
            "secret-noncharacter-start",
            id="noncharacter-fdd0",
        ),
        pytest.param(
            "\ufdefsecret-noncharacter-end",
            "secret-noncharacter-end",
            id="noncharacter-fdef",
        ),
        pytest.param(
            "\ufffesecret-plane0-fffe",
            "secret-plane0-fffe",
            id="plane0-fffe",
        ),
        pytest.param(
            "\uffffsecret-plane0-ffff",
            "secret-plane0-ffff",
            id="plane0-ffff",
        ),
        pytest.param(
            "\U0001fffesecret-plane1-fffe",
            "secret-plane1-fffe",
            id="plane1-fffe",
        ),
        pytest.param(
            "\U0010ffffsecret-plane16-ffff",
            "secret-plane16-ffff",
            id="plane16-ffff",
        ),
    ],
)
async def test_raw_unsafe_search_fails_safely_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_search: str,
    visible_secret: str,
) -> None:
    database_path = tmp_path / "secret-unsafe-search-store.sqlite3"
    repository = profile_repository.TTSProfileRepository(database_path)
    submitted = False

    async def forbidden_submission(
        _operation: Callable[[sqlite3.Connection], object],
    ) -> ProfileStoreResult[object]:
        nonlocal submitted
        submitted = True
        raise AssertionError("unsafe raw search reached worker submission")

    monkeypatch.setattr(repository, "_submit_operation", forbidden_submission)

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.list_profiles(search=raw_search)

    _assert_safe_error(
        caught.value,
        "operation_failed",
        visible_secret,
        str(database_path),
    )
    assert repr(raw_search) not in "".join(traceback.format_exception(caught.value))
    assert submitted is False
    assert not database_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("helper_name", "secret"),
    [
        ("_unicode_category", "secret-hostile-unicode-category"),
        ("_unicode_ord", "secret-hostile-unicode-ord"),
    ],
)
async def test_hostile_search_character_inspection_fails_safely_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    secret: str,
) -> None:
    database_path = tmp_path / "secret-character-inspector-store.sqlite3"
    repository = profile_repository.TTSProfileRepository(database_path)
    submitted = False

    async def forbidden_submission(
        _operation: Callable[[sqlite3.Connection], object],
    ) -> ProfileStoreResult[object]:
        nonlocal submitted
        submitted = True
        raise AssertionError("failed character inspection reached submission")

    def hostile_inspector(_character: str) -> object:
        try:
            raise RuntimeError(secret)
        except RuntimeError:
            raise ValueError(secret)

    monkeypatch.setattr(repository, "_submit_operation", forbidden_submission)
    monkeypatch.setattr(
        profile_repository,
        helper_name,
        hostile_inspector,
        raising=False,
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.list_profiles(search="ordinary")

    _assert_safe_error(
        caught.value,
        "operation_failed",
        secret,
        str(database_path),
    )
    assert submitted is False
    assert not database_path.exists()


@pytest.mark.asyncio
async def test_create_generates_uuid4_or_retains_exact_caller_uuid_and_round_trips(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    generated_draft = _draft("  Ｎａｒｒａｔｏｒ 音声  ")
    caller_draft = _draft(
        "Caller Profile",
        model_id="model/exact/2",
        voice_id=None,
        response_format="ogg",
        speed=0.75,
        options={"seed": 7, "labels": ["声", False]},
    )
    generated_factory = _SequenceCallable(iter((GENERATED_ID,)))

    async with _opened_repository(
        database_path,
        clock=lambda: CREATED_AT,
        uuid_factory=cast(Callable[[], UUID], generated_factory),
    ) as repository:
        generated_result = await repository.create_profile(generated_draft)
        caller_result = await repository.create_profile(
            caller_draft,
            profile_id=CALLER_ID,
        )

        assert generated_result.generation == 1
        assert caller_result.generation == 1
        generated = generated_result.value
        caller = caller_result.value
        assert type(generated) is TTSGenerationProfile
        assert generated.profile_id == GENERATED_ID
        assert generated.profile_id.version == 4
        assert generated.display_name == "Ｎａｒｒａｔｏｒ 音声"
        assert generated.normalized_name == "narrator 音声"
        assert generated.provider_id == generated_draft.provider_id
        assert generated.model_id == generated_draft.model_id
        assert generated.voice_id == generated_draft.voice_id
        assert generated.response_format == generated_draft.response_format
        assert generated.speed == generated_draft.speed
        assert canonical_json_options(generated.options) == canonical_json_options(
            generated_draft.options
        )
        assert generated.revision == 1
        assert generated.created_at == CREATED_AT
        assert generated.updated_at == CREATED_AT
        assert generated.created_at.tzinfo is UTC

        assert caller.profile_id == CALLER_ID
        assert caller.display_name == caller_draft.display_name
        assert caller.provider_id == caller_draft.provider_id
        assert caller.model_id == caller_draft.model_id
        assert caller.voice_id is None
        assert caller.response_format == caller_draft.response_format
        assert caller.speed == caller_draft.speed
        assert canonical_json_options(caller.options) == canonical_json_options(
            caller_draft.options
        )
        assert caller.revision == 1
        assert caller.created_at == caller.updated_at == CREATED_AT

        with pytest.raises((AttributeError, TypeError)):
            setattr(generated, "display_name", "mutated")
        with pytest.raises(TypeError):
            cast(Any, generated.options)["new"] = "mutated"


@pytest.mark.asyncio
async def test_create_conflicts_do_not_overwrite_uuid_or_normalized_name(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    first_id = UUID("30000000-0000-4000-8000-000000000003")
    other_id = UUID("40000000-0000-4000-8000-000000000004")
    third_id = UUID("50000000-0000-4000-8000-000000000005")

    async with _opened_repository(database_path) as repository:
        first = (
            await repository.create_profile(
                _draft("  Ｓｔｒａｓｓｅ ｶﾀｶﾅ  "),
                profile_id=first_id,
            )
        ).value

        with pytest.raises(ProfileRepositoryError) as duplicate_id:
            await repository.create_profile(
                _draft("A different display name"),
                profile_id=first_id,
            )
        _assert_safe_error(duplicate_id.value, "conflict")

        with pytest.raises(ProfileRepositoryError) as duplicate_name:
            await repository.create_profile(
                _draft("STRASSE カタカナ"),
                profile_id=other_id,
            )
        _assert_safe_error(duplicate_name.value, "conflict")

        with pytest.raises(ProfileRepositoryError) as casefold_name:
            await repository.create_profile(
                _draft("strasse カタカナ"),
                profile_id=third_id,
            )
        _assert_safe_error(casefold_name.value, "conflict")

        assert (await repository.get_profile(first_id)).value == first
        page = (await repository.list_profiles()).value
        assert page.profiles == (first,)
        assert page.total == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("constraint_kind", "expected_sqlite_code"),
    [
        ("notnull", 1299),
        ("check", 275),
        ("trigger", 1811),
        ("datatype", 3091),
        ("rowid", 2579),
        ("foreignkey", 787),
    ],
)
async def test_create_maps_real_unexpected_extended_constraints_to_operation_failed(
    tmp_path: Path,
    constraint_kind: str,
    expected_sqlite_code: int,
) -> None:
    database_path = tmp_path / f"secret-{constraint_kind}-constraint.sqlite3"
    trigger_secret = "secret-real-trigger-message"

    async with _opened_repository(database_path) as repository:
        await asyncio.to_thread(
            _install_insert_constraint_trigger,
            database_path,
            constraint_kind,
        )
        await asyncio.to_thread(
            _assert_real_insert_constraint_code,
            database_path,
            expected_sqlite_code,
        )

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.create_profile(
                _draft("Unexpected Constraint"),
                profile_id=GENERATED_ID,
            )

        _assert_safe_error(
            caught.value,
            "operation_failed",
            trigger_secret,
            str(database_path),
        )
        assert (await repository.list_profiles()).value.total == 0

        await asyncio.to_thread(
            _external_execute,
            database_path,
            "DROP TRIGGER force_profile_insert_constraint",
        )
        recovered = await repository.create_profile(
            _draft("Recovered"),
            profile_id=GENERATED_ID,
        )
        assert recovered.value.profile_id == GENERATED_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize(
    ("constraint_kind", "expected_sqlite_code"),
    [("unique", 2067), ("primarykey", 1555)],
)
async def test_unrelated_expected_extended_code_without_owned_row_evidence_fails(
    tmp_path: Path,
    operation: str,
    constraint_kind: str,
    expected_sqlite_code: int,
) -> None:
    database_path = tmp_path / (
        f"secret-{operation}-{constraint_kind}-false-conflict.sqlite3"
    )
    profile_id = UUID("f0500000-0000-4000-8000-00000000000f")
    raw_name = "Evidence ' OR 1=1 -- secret-evidence-name"
    draft = _draft(raw_name)

    async with _opened_repository(database_path) as repository:
        original: TTSGenerationProfile | None = None
        if operation == "update":
            original = (
                await repository.create_profile(
                    _draft("Original Evidence Target"),
                    profile_id=profile_id,
                )
            ).value

        await asyncio.to_thread(
            _install_unrelated_uniqueness_trigger,
            database_path,
            operation,
            constraint_kind,
        )
        if operation == "create":
            await asyncio.to_thread(
                _assert_real_insert_constraint_code,
                database_path,
                expected_sqlite_code,
            )
        else:
            await asyncio.to_thread(
                _assert_real_update_constraint_code,
                database_path,
                profile_id,
                expected_sqlite_code,
            )

        with pytest.raises(ProfileRepositoryError) as failed:
            if operation == "create":
                await repository.create_profile(draft, profile_id=profile_id)
            else:
                await repository.update_profile(profile_id, 1, draft)

        _assert_safe_error(
            failed.value,
            "operation_failed",
            raw_name,
            str(database_path),
        )
        page = (await repository.list_profiles()).value
        if operation == "create":
            assert page.total == 0
        else:
            assert original is not None
            assert page.profiles == (original,)

        await asyncio.to_thread(
            _external_execute,
            database_path,
            "DROP TRIGGER force_unrelated_profile_constraint",
        )
        if operation == "create":
            recovered = await repository.create_profile(
                draft,
                profile_id=profile_id,
            )
        else:
            recovered = await repository.update_profile(
                profile_id,
                1,
                draft,
            )
        assert recovered.value.normalized_name == draft.normalized_name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "sqlite_errorcode"),
    [
        ("create", 2067),
        ("create", 1555),
        ("create", 787),
        ("update", 2067),
        ("update", 1555),
        ("update", 787),
        ("delete", 787),
        ("delete", 2067),
        ("delete", 1555),
        ("create", 19),
        ("create", 25363),
        ("create", None),
    ],
)
async def test_integrity_extended_code_without_owned_evidence_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    sqlite_errorcode: int | None,
) -> None:
    database_path = tmp_path / "secret-constructed-constraint.sqlite3"
    profile_id = UUID("f1000000-0000-4000-8000-00000000000f")
    secret = "secret-constructed-constraint-message"
    error = sqlite3.IntegrityError(f"UNIQUE constraint failed: {secret}")
    if sqlite_errorcode is not None:
        setattr(error, "sqlite_errorcode", sqlite_errorcode)

    async with _opened_repository(database_path) as repository:
        if operation != "create":
            await repository.create_profile(
                _draft("Existing"),
                profile_id=profile_id,
            )

        def fail_encode(_profile: TTSGenerationProfile) -> dict[str, object]:
            raise error

        def fail_get(
            _connection: sqlite3.Connection,
            _profile_id: UUID,
        ) -> TTSGenerationProfile:
            raise error

        with monkeypatch.context() as patch:
            if operation in {"create", "update"}:
                patch.setattr(profile_repository, "encode_profile", fail_encode)
            else:
                patch.setattr(repository, "_worker_get_profile", fail_get)

            with pytest.raises(ProfileRepositoryError) as caught:
                if operation == "create":
                    await repository.create_profile(
                        _draft("Injected Create"),
                        profile_id=profile_id,
                    )
                elif operation == "update":
                    await repository.update_profile(
                        profile_id,
                        1,
                        _draft("Injected Update"),
                    )
                else:
                    await repository.delete_profile(profile_id)

        _assert_safe_error(
            caught.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        page = (await repository.list_profiles()).value
        assert page.total == (0 if operation == "create" else 1)


@pytest.mark.asyncio
async def test_hostile_integrity_error_code_maps_safely_without_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-hostile-error-code.sqlite3"
    secret = "secret-hostile-error-code-context"
    error = _HostileSQLiteErrorCode(ValueError(secret))

    async with _opened_repository(database_path) as repository:

        def fail_encode(_profile: TTSGenerationProfile) -> dict[str, object]:
            raise error

        with monkeypatch.context() as patch:
            patch.setattr(profile_repository, "encode_profile", fail_encode)
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.create_profile(
                    _draft("Hostile Code"),
                    profile_id=GENERATED_ID,
                )

        _assert_safe_error(
            caught.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert (await repository.list_profiles()).value.total == 0


@pytest.mark.asyncio
async def test_integrity_error_code_control_flow_is_preserved_after_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    signal = _ControlFlow()

    async with _opened_repository(database_path) as repository:
        original = (
            await repository.create_profile(
                _draft("Control Flow Code"),
                profile_id=GENERATED_ID,
            )
        ).value

        def raise_signal(_error: sqlite3.IntegrityError) -> int | None:
            raise signal

        with monkeypatch.context() as patch:
            patch.setattr(
                profile_repository,
                "_read_sqlite_errorcode",
                raise_signal,
            )
            with pytest.raises(_ControlFlow) as caught:
                await repository.create_profile(
                    _draft("Control Flow Code"),
                    profile_id=CALLER_ID,
                )

        assert caught.value is signal
        assert (await repository.list_profiles()).value.profiles == (original,)
        recovered = await repository.create_profile(
            _draft("Control Flow Recovered"),
            profile_id=CALLER_ID,
        )
        assert recovered.value.profile_id == CALLER_ID


@pytest.mark.asyncio
async def test_get_missing_and_corrupt_rows_fail_with_safe_specific_codes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "secret-profile-path.sqlite3"
    profile_id = UUID("60000000-0000-4000-8000-000000000006")
    corrupt_secret = "secret-corrupt-options"

    async with _opened_repository(database_path) as repository:
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing", str(database_path))

        await repository.create_profile(_draft(), profile_id=profile_id)
        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            UPDATE tts_generation_profiles
            SET options_json = ?
            WHERE profile_id = ?
            """,
            (f'{{"{corrupt_secret}":NaN}}', str(profile_id)),
        )

        with pytest.raises(ProfileRepositoryError) as corrupt:
            await repository.get_profile(profile_id)
        _assert_safe_error(
            corrupt.value,
            "corrupt_data",
            corrupt_secret,
            str(database_path),
        )


@pytest.mark.asyncio
async def test_list_is_stable_paginated_immutable_and_reports_filtered_total(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    entries = (
        (UUID("00000000-0000-4000-8000-000000000004"), "Zulu"),
        (UUID("00000000-0000-4000-8000-000000000002"), "beta"),
        (UUID("00000000-0000-4000-8000-000000000003"), "Äther"),
        (UUID("00000000-0000-4000-8000-000000000001"), "Alpha"),
    )

    async with _opened_repository(database_path) as repository:
        for profile_id, display_name in entries:
            await repository.create_profile(
                _draft(display_name),
                profile_id=profile_id,
            )

        first_page = (await repository.list_profiles(limit=2, offset=0)).value
        second_page = (await repository.list_profiles(limit=2, offset=2)).value
        beyond = (await repository.list_profiles(limit=100, offset=20)).value
        single = (await repository.list_profiles(limit=1, offset=1)).value

        assert type(first_page) is TTSProfilePage
        assert type(first_page.profiles) is tuple
        expected_names = tuple(
            sorted(
                (display_name for _, display_name in entries),
                key=lambda name: (
                    _draft(name).normalized_name,
                    str(
                        next(profile_id for profile_id, item in entries if item == name)
                    ),
                ),
            )
        )
        assert (
            tuple(
                profile.display_name
                for profile in first_page.profiles + second_page.profiles
            )
            == expected_names
        )
        assert (
            first_page.total == second_page.total == beyond.total == single.total == 4
        )
        assert len(first_page.profiles) == len(second_page.profiles) == 2
        assert beyond.profiles == ()
        assert single.profiles == (first_page.profiles[1],)


@pytest.mark.asyncio
async def test_list_search_normalizes_case_and_treats_like_metacharacters_literal(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    names = (
        "100% Real",
        "under_score",
        "Bang! Voice",
        "Ｃａｆé 音声",
        "Straße",
        "Ordinary",
    )

    async with _opened_repository(database_path) as repository:
        for index, name in enumerate(names, start=1):
            await repository.create_profile(
                _draft(name),
                profile_id=UUID(f"70000000-0000-4000-8000-{index:012d}"),
            )

        percent = (await repository.list_profiles(search="%")).value
        underscore = (await repository.list_profiles(search="_")).value
        escape = (await repository.list_profiles(search="!")).value
        unicode_case = (await repository.list_profiles(search="  CAFÉ 音声  ")).value
        casefold = (await repository.list_profiles(search="STRASSE")).value
        empty = (await repository.list_profiles(search="")).value
        spaces_only = (await repository.list_profiles(search="   ")).value

        assert tuple(profile.display_name for profile in percent.profiles) == (
            "100% Real",
        )
        assert percent.total == 1
        assert tuple(profile.display_name for profile in underscore.profiles) == (
            "under_score",
        )
        assert underscore.total == 1
        assert tuple(profile.display_name for profile in escape.profiles) == (
            "Bang! Voice",
        )
        assert escape.total == 1
        assert tuple(profile.display_name for profile in unicode_case.profiles) == (
            "Ｃａｆé 音声",
        )
        assert unicode_case.total == 1
        assert tuple(profile.display_name for profile in casefold.profiles) == (
            "Straße",
        )
        assert casefold.total == 1
        assert empty == spaces_only
        assert empty.total == len(names)


@pytest.mark.asyncio
async def test_update_uses_optimistic_revision_and_preserves_winner_exactly(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    profile_id = UUID("80000000-0000-4000-8000-000000000008")
    clock = _SequenceCallable(iter((CREATED_AT, UPDATED_AT)))

    async with _opened_repository(
        database_path,
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        created = (
            await repository.create_profile(_draft("Shared"), profile_id=profile_id)
        ).value
        editor_a = (await repository.get_profile(profile_id)).value
        editor_b = (await repository.get_profile(profile_id)).value
        winner_draft = _draft(
            "Shared Updated",
            model_id="winner-model",
            voice_id=None,
            response_format="flac",
            speed=2.0,
            options={"winner": {"exact": [1, "声"]}},
        )
        loser_draft = _draft(
            "Loser Value",
            model_id="loser-model",
            options={"must": "not persist"},
        )

        winner = (
            await repository.update_profile(
                profile_id,
                editor_a.revision,
                winner_draft,
            )
        ).value
        with pytest.raises(ProfileRepositoryError) as stale:
            await repository.update_profile(
                profile_id,
                editor_b.revision,
                loser_draft,
            )
        _assert_safe_error(stale.value, "conflict")

        stored = (await repository.get_profile(profile_id)).value
        assert winner.revision == created.revision + 1 == 2
        assert winner.created_at == created.created_at == CREATED_AT
        assert winner.updated_at == UPDATED_AT
        assert winner.display_name == winner_draft.display_name
        assert winner.model_id == winner_draft.model_id
        assert winner.voice_id == winner_draft.voice_id
        assert winner.response_format == winner_draft.response_format
        assert winner.speed == winner_draft.speed
        assert canonical_json_options(winner.options) == canonical_json_options(
            winner_draft.options
        )
        assert stored == winner
        assert stored != editor_b


@pytest.mark.asyncio
async def test_update_allows_display_spelling_change_with_same_normalized_key(
    tmp_path: Path,
) -> None:
    profile_id = UUID("81000000-0000-4000-8000-000000000008")
    clock = _SequenceCallable(iter((CREATED_AT, UPDATED_AT)))

    async with _opened_repository(
        tmp_path / "profiles.sqlite3",
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        original = (
            await repository.create_profile(
                _draft("Straße"),
                profile_id=profile_id,
            )
        ).value
        updated = (
            await repository.update_profile(
                profile_id,
                original.revision,
                _draft("STRASSE"),
            )
        ).value

        assert original.normalized_name == updated.normalized_name == "strasse"
        assert updated.display_name == "STRASSE"
        assert updated.revision == 2


@pytest.mark.asyncio
async def test_update_missing_or_name_collision_rolls_back_without_partial_changes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    first_id = UUID("82000000-0000-4000-8000-000000000008")
    second_id = UUID("83000000-0000-4000-8000-000000000008")
    missing_id = UUID("84000000-0000-4000-8000-000000000008")
    clock = _SequenceCallable(iter((CREATED_AT, CREATED_AT, UPDATED_AT, LATER_AT)))

    async with _opened_repository(
        database_path,
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        first = (
            await repository.create_profile(_draft("First"), profile_id=first_id)
        ).value
        second = (
            await repository.create_profile(_draft("Second"), profile_id=second_id)
        ).value

        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.update_profile(
                missing_id,
                1,
                _draft("Missing"),
            )
        _assert_safe_error(missing.value, "missing")

        with pytest.raises(ProfileRepositoryError) as collision:
            await repository.update_profile(
                second_id,
                second.revision,
                _draft("Ｆｉｒｓｔ"),
            )
        _assert_safe_error(collision.value, "conflict")

        assert (await repository.get_profile(first_id)).value == first
        assert (await repository.get_profile(second_id)).value == second
        recovered = (
            await repository.update_profile(
                second_id,
                second.revision,
                _draft("Third"),
            )
        ).value
        assert recovered.revision == 2
        assert recovered.display_name == "Third"


@pytest.mark.asyncio
async def test_delete_removes_exactly_one_profile_and_missing_is_safe(
    tmp_path: Path,
) -> None:
    profile_id = UUID("90000000-0000-4000-8000-000000000009")
    other_id = UUID("91000000-0000-4000-8000-000000000009")

    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        await repository.create_profile(_draft("Delete"), profile_id=profile_id)
        other = (
            await repository.create_profile(_draft("Keep"), profile_id=other_id)
        ).value

        deleted = await repository.delete_profile(profile_id)

        assert deleted == ProfileStoreResult(generation=1, value=None)
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing")
        assert (await repository.get_profile(other_id)).value == other

        with pytest.raises(ProfileRepositoryError) as missing_delete:
            await repository.delete_profile(profile_id)
        _assert_safe_error(missing_delete.value, "missing")


@pytest.mark.asyncio
async def test_delete_rejects_corrupt_target_without_discarding_it(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "secret-corrupt-delete.sqlite3"
    profile_id = UUID("91500000-0000-4000-8000-000000000009")
    corrupt_secret = "secret-delete-corrupt-options"

    async with _opened_repository(database_path) as repository:
        original = (
            await repository.create_profile(
                _draft("Corrupt Delete"),
                profile_id=profile_id,
            )
        ).value
        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            UPDATE tts_generation_profiles
            SET options_json = ?
            WHERE profile_id = ?
            """,
            (f'{{"{corrupt_secret}":NaN}}', str(profile_id)),
        )

        with pytest.raises(ProfileRepositoryError) as corrupt:
            await repository.delete_profile(profile_id)
        _assert_safe_error(
            corrupt.value,
            "corrupt_data",
            corrupt_secret,
            str(database_path),
        )

        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            UPDATE tts_generation_profiles
            SET options_json = ?
            WHERE profile_id = ?
            """,
            (canonical_json_options(original.options), str(profile_id)),
        )
        assert (await repository.get_profile(profile_id)).value == original


@pytest.mark.asyncio
async def test_foreign_key_restricted_delete_trigger_maps_to_conflict_for_assignment(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    profile_id = UUID("92000000-0000-4000-8000-000000000009")
    assignment = CharacterTTSAssignment(
        character_ref=CharacterRef(
            source="local",
            authority_id="authority-1",
            character_id="character-1",
        ),
        profile_id=profile_id,
    )

    async with _opened_repository(database_path) as repository:
        profile = (
            await repository.create_profile(
                _draft("Assigned"),
                profile_id=profile_id,
            )
        ).value
        encoded = encode_assignment(
            assignment,
            created_at=CREATED_AT,
            updated_at=CREATED_AT,
        )
        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            INSERT INTO character_tts_assignments (
                source, authority_id, character_id, profile_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                encoded["source"],
                encoded["authority_id"],
                encoded["character_id"],
                encoded["profile_id"],
                encoded["created_at"],
                encoded["updated_at"],
            ),
        )
        await asyncio.to_thread(
            _assert_real_delete_constraint_code,
            database_path,
            profile_id,
            1811,
        )

        with pytest.raises(ProfileRepositoryError) as conflict:
            await repository.delete_profile(profile_id)

        _assert_safe_error(conflict.value, "conflict")
        assert (await repository.get_profile(profile_id)).value == profile


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt", range(5))
async def test_delete_captures_assignment_conflict_before_post_rollback_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt: int,
) -> None:
    database_path = tmp_path / f"remove-assignment-barrier-{attempt}.sqlite3"
    profile_id = UUID("92100000-0000-4000-8000-000000000009")

    async with _opened_repository(database_path) as repository:
        profile = (
            await repository.create_profile(
                _draft("Barrier Assigned"),
                profile_id=profile_id,
            )
        ).value
        await asyncio.to_thread(
            _external_insert_assignment,
            database_path,
            profile_id,
            f"remove-{attempt}",
        )
        rolled_back, resume = _pause_next_rollback(monkeypatch, repository)
        deletion = asyncio.create_task(repository.delete_profile(profile_id))
        try:
            assert await asyncio.to_thread(rolled_back.wait, 10.0)
            await asyncio.to_thread(
                _external_execute,
                database_path,
                "DELETE FROM character_tts_assignments WHERE profile_id = ?",
                (str(profile_id),),
            )
        finally:
            resume.set()

        with pytest.raises(ProfileRepositoryError) as conflict:
            await deletion

        _assert_safe_error(conflict.value, "conflict")
        assert (await repository.get_profile(profile_id)).value == profile


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt", range(5))
async def test_delete_captures_missing_assignment_before_post_rollback_addition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt: int,
) -> None:
    database_path = tmp_path / f"add-assignment-barrier-{attempt}.sqlite3"
    profile_id = UUID("92200000-0000-4000-8000-000000000009")
    secret = f"secret-addition-trigger-{attempt}"

    async with _opened_repository(database_path) as repository:
        profile = (
            await repository.create_profile(
                _draft("Barrier Unassigned"),
                profile_id=profile_id,
            )
        ).value
        await asyncio.to_thread(
            _external_execute,
            database_path,
            f"""
            CREATE TRIGGER force_profile_delete_constraint
            BEFORE DELETE ON tts_generation_profiles
            BEGIN
                SELECT RAISE(ABORT, '{secret}');
            END
            """,
        )
        rolled_back, resume = _pause_next_rollback(monkeypatch, repository)
        deletion = asyncio.create_task(repository.delete_profile(profile_id))
        try:
            assert await asyncio.to_thread(rolled_back.wait, 10.0)
            await asyncio.to_thread(
                _external_insert_assignment,
                database_path,
                profile_id,
                f"add-{attempt}",
            )
        finally:
            resume.set()

        with pytest.raises(ProfileRepositoryError) as failed:
            await deletion

        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert (await repository.get_profile(profile_id)).value == profile


@pytest.mark.asyncio
async def test_unrelated_foreign_key_no_action_delete_maps_to_operation_failed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    profile_id = UUID("92500000-0000-4000-8000-000000000009")

    async with _opened_repository(database_path) as repository:
        profile = (
            await repository.create_profile(
                _draft("No Action Assigned"),
                profile_id=profile_id,
            )
        ).value
        await asyncio.to_thread(
            _install_no_action_delete_probe,
            database_path,
            profile_id,
        )
        await asyncio.to_thread(
            _assert_real_delete_constraint_code,
            database_path,
            profile_id,
            787,
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.delete_profile(profile_id)

        _assert_safe_error(failed.value, "operation_failed")
        assert (await repository.get_profile(profile_id)).value == profile

        await asyncio.to_thread(
            _external_execute,
            database_path,
            "DROP TABLE delete_constraint_probe",
        )
        assert (await repository.delete_profile(profile_id)).value is None


@pytest.mark.asyncio
async def test_unrelated_delete_trigger_maps_to_operation_failed_without_assignment(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "secret-unrelated-delete-trigger.sqlite3"
    profile_id = UUID("92600000-0000-4000-8000-000000000009")
    secret = "secret-unrelated-delete-trigger-message"

    async with _opened_repository(database_path) as repository:
        profile = (
            await repository.create_profile(
                _draft("Unrelated Trigger"),
                profile_id=profile_id,
            )
        ).value
        await asyncio.to_thread(
            _external_execute,
            database_path,
            f"""
            CREATE TRIGGER force_profile_delete_constraint
            BEFORE DELETE ON tts_generation_profiles
            BEGIN
                SELECT RAISE(ABORT, 'FOREIGN KEY constraint failed: {secret}');
            END
            """,
        )
        await asyncio.to_thread(
            _assert_real_delete_constraint_code,
            database_path,
            profile_id,
            1811,
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.delete_profile(profile_id)

        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert (await repository.get_profile(profile_id)).value == profile

        await asyncio.to_thread(
            _external_execute,
            database_path,
            "DROP TRIGGER force_profile_delete_constraint",
        )
        assert (await repository.delete_profile(profile_id)).value is None


@pytest.mark.asyncio
async def test_delete_trigger_post_check_failure_maps_safely_to_operation_failed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "secret-delete-post-check.sqlite3"
    profile_id = UUID("92700000-0000-4000-8000-000000000009")
    secret = "secret-delete-post-check-trigger"

    async with _opened_repository(database_path) as repository:
        profile = (
            await repository.create_profile(
                _draft("Post Check Failure"),
                profile_id=profile_id,
            )
        ).value
        await asyncio.to_thread(
            _external_execute,
            database_path,
            "DROP TABLE character_tts_assignments",
        )
        await asyncio.to_thread(
            _external_execute,
            database_path,
            f"""
            CREATE TRIGGER force_profile_delete_constraint
            BEFORE DELETE ON tts_generation_profiles
            BEGIN
                SELECT RAISE(ABORT, '{secret}');
            END
            """,
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.delete_profile(profile_id)

        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert (await repository.get_profile(profile_id)).value == profile


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update", "delete"])
@pytest.mark.parametrize("sqlite_errorcode", [2067, 1555])
async def test_expected_integrity_from_commit_never_classifies_as_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    sqlite_errorcode: int,
) -> None:
    database_path = tmp_path / (
        f"commit-integrity-{operation}-{sqlite_errorcode}.sqlite3"
    )
    profile_id = UUID("a0000000-0000-4000-8000-00000000000a")
    secret = f"secret-{operation}-commit-integrity-{sqlite_errorcode}"
    original: TTSGenerationProfile | None = None

    async with _opened_repository(database_path) as repository:
        if operation != "create":
            original = (
                await repository.create_profile(
                    _draft("Commit Original"),
                    profile_id=profile_id,
                )
            ).value

        classifications = _record_integrity_classification(monkeypatch)
        attempted = _fail_next_commit(
            monkeypatch,
            repository,
            _expected_integrity_error(sqlite_errorcode, secret),
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            if operation == "create":
                await repository.create_profile(
                    _draft("Commit Create"),
                    profile_id=profile_id,
                )
            elif operation == "update":
                await repository.update_profile(
                    profile_id,
                    1,
                    _draft("Commit Updated"),
                )
            else:
                await repository.delete_profile(profile_id)

        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert attempted.is_set()
        assert classifications == []

        if operation == "create":
            assert (await repository.list_profiles()).value.total == 0
            recovered = await repository.create_profile(
                _draft("Commit Create"),
                profile_id=profile_id,
            )
            assert recovered.value.profile_id == profile_id
        elif operation == "update":
            assert original is not None
            assert (await repository.get_profile(profile_id)).value == original
            recovered = await repository.update_profile(
                profile_id,
                original.revision,
                _draft("Commit Recovered"),
            )
            assert recovered.value.revision == original.revision + 1
        else:
            assert original is not None
            assert (await repository.get_profile(profile_id)).value == original
            assert (await repository.delete_profile(profile_id)).value is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize("sqlite_errorcode", [2067, 1555])
async def test_expected_integrity_from_round_trip_never_classifies_as_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    sqlite_errorcode: int,
) -> None:
    database_path = tmp_path / (
        f"round-trip-integrity-{operation}-{sqlite_errorcode}.sqlite3"
    )
    profile_id = UUID("a1000000-0000-4000-8000-00000000000a")
    secret = f"secret-{operation}-round-trip-integrity-{sqlite_errorcode}"
    original: TTSGenerationProfile | None = None

    async with _opened_repository(database_path) as repository:
        if operation == "update":
            original = (
                await repository.create_profile(
                    _draft("Round Trip Original"),
                    profile_id=profile_id,
                )
            ).value

        classifications = _record_integrity_classification(monkeypatch)
        attempted = _fail_next_round_trip(
            monkeypatch,
            repository,
            _expected_integrity_error(sqlite_errorcode, secret),
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            if operation == "create":
                await repository.create_profile(
                    _draft("Round Trip Create"),
                    profile_id=profile_id,
                )
            else:
                await repository.update_profile(
                    profile_id,
                    1,
                    _draft("Round Trip Updated"),
                )

        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert attempted.is_set()
        assert classifications == []

        if operation == "create":
            assert (await repository.list_profiles()).value.total == 0
            recovered = await repository.create_profile(
                _draft("Round Trip Create"),
                profile_id=profile_id,
            )
            assert recovered.value.profile_id == profile_id
        else:
            assert original is not None
            assert (await repository.get_profile(profile_id)).value == original
            recovered = await repository.update_profile(
                profile_id,
                original.revision,
                _draft("Round Trip Recovered"),
            )
            assert recovered.value.revision == original.revision + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update", "delete"])
@pytest.mark.parametrize("sqlite_errorcode", [2067, 1555])
async def test_expected_integrity_from_begin_never_classifies_as_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    sqlite_errorcode: int,
) -> None:
    database_path = tmp_path / (
        f"begin-integrity-{operation}-{sqlite_errorcode}.sqlite3"
    )
    secret = f"secret-{operation}-begin-integrity-{sqlite_errorcode}"
    failed_connection = _BeginIntegrityConnection(
        _expected_integrity_error(sqlite_errorcode, secret)
    )
    mutation_attempted = False

    def mutation() -> None:
        nonlocal mutation_attempted
        mutation_attempted = True

    async with _opened_repository(database_path) as repository:
        classifications = _record_integrity_classification(monkeypatch)

        with pytest.raises(ProfileRepositoryError) as failed:
            repository._worker_transaction(
                cast(sqlite3.Connection, cast(Any, failed_connection)),
                mutation,
                operation_kind=cast(Any, operation),
                immediate=True,
            )

        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert not mutation_attempted
        assert failed_connection.rollback_attempted
        assert classifications == []

        recovered = await repository.create_profile(
            _draft(f"After {operation} Begin"),
            profile_id=GENERATED_ID,
        )
        assert recovered.value.profile_id == GENERATED_ID


@pytest.mark.asyncio
async def test_create_failure_before_commit_rolls_back_and_repository_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-create-store.sqlite3"
    profile_id = UUID("a0000000-0000-4000-8000-00000000000a")
    secret = "secret-create-commit-failure"

    async with _opened_repository(database_path) as repository:
        attempted = _fail_next_commit(
            monkeypatch,
            repository,
            sqlite3.OperationalError(secret),
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.create_profile(
                _draft("Rollback Create"),
                profile_id=profile_id,
            )
        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert attempted.is_set()

        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing")
        recovered = (
            await repository.create_profile(
                _draft("Rollback Create"),
                profile_id=profile_id,
            )
        ).value
        assert recovered.revision == 1


@pytest.mark.asyncio
async def test_rollback_failure_quarantines_connection_and_retry_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-rollback-store.sqlite3"
    profile_id = UUID("a5000000-0000-4000-8000-00000000000a")
    commit_secret = "secret-commit-before-rollback-failure"
    rollback_secret = "secret-rollback-failure"
    repository = profile_repository.TTSProfileRepository(
        database_path,
        _clock=lambda: CREATED_AT,
        _uuid_factory=lambda: GENERATED_ID,
    )
    await repository.open()
    try:
        _fail_next_commit(
            monkeypatch,
            repository,
            sqlite3.OperationalError(commit_secret),
        )

        def failed_rollback(_connection: sqlite3.Connection) -> None:
            raise sqlite3.OperationalError(rollback_secret)

        monkeypatch.setattr(
            repository,
            "_rollback_transaction",
            failed_rollback,
            raising=False,
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.create_profile(
                _draft("Rollback Quarantine"),
                profile_id=profile_id,
            )
        _assert_safe_error(
            failed.value,
            "operation_failed",
            commit_secret,
            rollback_secret,
            str(database_path),
        )
        assert repository.state is ProfileRepositoryState.UNAVAILABLE

        with pytest.raises(ProfileRepositoryError) as unavailable:
            await repository.get_profile(profile_id)
        _assert_safe_error(unavailable.value, "unavailable")

        reopened = await repository.open()
        assert reopened.generation == 2
        assert repository.state is ProfileRepositoryState.OPEN
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing")
        recovered = (
            await repository.create_profile(
                _draft("Rollback Quarantine"),
                profile_id=profile_id,
            )
        ).value
        assert recovered.revision == 1
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_update_failure_before_commit_rolls_back_and_repository_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-update-store.sqlite3"
    profile_id = UUID("b0000000-0000-4000-8000-00000000000b")
    secret = "secret-update-commit-failure"
    clock = _SequenceCallable(iter((CREATED_AT, UPDATED_AT, LATER_AT)))

    async with _opened_repository(
        database_path,
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        original = (
            await repository.create_profile(
                _draft("Original"),
                profile_id=profile_id,
            )
        ).value
        attempted = _fail_next_commit(
            monkeypatch,
            repository,
            sqlite3.OperationalError(secret),
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.update_profile(
                profile_id,
                original.revision,
                _draft("Must Roll Back", model_id="rolled-back-model"),
            )
        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert attempted.is_set()
        assert (await repository.get_profile(profile_id)).value == original

        recovered = (
            await repository.update_profile(
                profile_id,
                original.revision,
                _draft("Recovered", model_id="recovered-model"),
            )
        ).value
        assert recovered.revision == 2
        assert recovered.display_name == "Recovered"
        assert recovered.updated_at == LATER_AT


@pytest.mark.asyncio
async def test_delete_failure_before_commit_rolls_back_and_repository_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-delete-store.sqlite3"
    profile_id = UUID("c0000000-0000-4000-8000-00000000000c")
    secret = "secret-delete-commit-failure"

    async with _opened_repository(database_path) as repository:
        original = (
            await repository.create_profile(
                _draft("Survivor"),
                profile_id=profile_id,
            )
        ).value
        attempted = _fail_next_commit(
            monkeypatch,
            repository,
            sqlite3.OperationalError(secret),
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.delete_profile(profile_id)
        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert attempted.is_set()
        assert (await repository.get_profile(profile_id)).value == original

        assert (await repository.delete_profile(profile_id)).value is None
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing")


@pytest.mark.asyncio
async def test_mutation_control_flow_is_preserved_after_transaction_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_id = UUID("d0000000-0000-4000-8000-00000000000d")
    signal = _ControlFlow()

    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        _fail_next_commit(monkeypatch, repository, signal)

        with pytest.raises(_ControlFlow) as caught:
            await repository.create_profile(
                _draft("Control Flow"),
                profile_id=profile_id,
            )

        assert caught.value is signal
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing")
        assert (
            await repository.create_profile(
                _draft("Control Flow"),
                profile_id=profile_id,
            )
        ).value.profile_id == profile_id


@pytest.mark.asyncio
async def test_hostile_codec_failure_is_recreated_without_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-codec-store.sqlite3"
    profile_id = UUID("e0000000-0000-4000-8000-00000000000e")
    secret = "secret-hostile-codec-context"

    async with _opened_repository(database_path) as repository:
        await repository.create_profile(_draft(), profile_id=profile_id)

        def hostile_decode(_row: object) -> TTSGenerationProfile:
            try:
                raise RuntimeError(secret)
            except RuntimeError:
                raise ValueError(secret)

        monkeypatch.setattr(profile_repository, "decode_profile", hostile_decode)

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.get_profile(profile_id)

        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )


@pytest.mark.parametrize("attempt", range(5))
def test_spawned_repositories_resolve_sqlite_constraint_race_safely(
    tmp_path: Path,
    attempt: int,
) -> None:
    database_path = tmp_path / f"profiles-{attempt}.sqlite3"
    asyncio.run(_initialize_empty_repository(database_path))
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    display_names = ("Ｒａｃｅ Ｎａｍｅ", "race name")
    receivers: list[Connection] = []
    child_connections: list[Connection] = []
    started_processes: list[Any] = []
    exitcodes: list[int | None] = []
    outcomes: list[tuple[object, ...]] = []

    try:
        for display_name in display_names:
            receiver, child_connection = context.Pipe(duplex=False)
            process = context.Process(
                target=_spawn_profile_collision,
                args=(
                    str(database_path),
                    child_connection,
                    release,
                    display_name,
                ),
            )
            receivers.append(receiver)
            child_connections.append(child_connection)
            process.start()
            started_processes.append(process)
            child_connection.close()

        for receiver in receivers:
            assert receiver.poll(15.0), "spawned repository did not report ready"
            assert receiver.recv() == ("ready", None)

        release.set()
        for receiver in receivers:
            assert receiver.poll(15.0), "spawned repository did not report outcome"
            message = receiver.recv()
            assert message[0] == "outcome", message
            outcomes.append(cast(tuple[object, ...], message[1]))
    finally:
        release.set()
        for child_connection in child_connections:
            child_connection.close()
        for process in started_processes:
            process.join(10.0)
            if process.is_alive():
                process.terminate()
                process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join(5.0)
            exitcodes.append(process.exitcode)
        for receiver in receivers:
            receiver.close()
        for process in started_processes:
            process.close()

    assert exitcodes == [0, 0]
    successes = [outcome for outcome in outcomes if outcome[0] == "success"]
    failures = [outcome for outcome in outcomes if outcome[0] == "failure"]
    assert successes == [("success", str(GENERATED_ID), "race name")]
    assert failures == [
        (
            "failure",
            "conflict",
            "TTS profile repository failed: conflict",
            True,
            True,
            (),
        )
    ]
    asyncio.run(_assert_spawned_race_store_is_usable(database_path))


@pytest.mark.asyncio
async def test_assignments_are_exactly_authority_scoped_and_replace_one_target(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "authority-scoped.sqlite3"
    first_profile_id = UUID("f1000000-0000-4000-8000-000000000001")
    second_profile_id = UUID("f2000000-0000-4000-8000-000000000002")
    timestamps = tuple(CREATED_AT + timedelta(seconds=index) for index in range(8))
    clock = _SequenceCallable(iter(timestamps))
    refs = (
        CharacterRef("local", "local-database-one", "shared-character"),
        CharacterRef("local", "local-database-two", "shared-character"),
        CharacterRef("server", "server-profile:principal-one", "shared-character"),
        CharacterRef("server", "server-profile:principal-two", "shared-character"),
    )

    async with _opened_repository(
        database_path,
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        await repository.create_profile(
            _draft("First Assignment Profile"),
            profile_id=first_profile_id,
        )
        await repository.create_profile(
            _draft("Second Assignment Profile"),
            profile_id=second_profile_id,
        )

        for character_ref in refs:
            assigned = await repository.set_assignment(
                character_ref,
                first_profile_id,
            )
            assert assigned == ProfileStoreResult(
                generation=1,
                value=CharacterTTSAssignment(character_ref, first_profile_id),
            )

        assert (await repository.assignment_count(first_profile_id)).value == 4
        connection = sqlite3.connect(database_path)
        try:
            before = connection.execute(
                """
                SELECT created_at, updated_at
                FROM character_tts_assignments
                WHERE source = ? AND authority_id = ? AND character_id = ?
                """,
                (refs[2].source, refs[2].authority_id, refs[2].character_id),
            ).fetchone()
        finally:
            connection.close()
        assert before is not None

        replaced = await repository.set_assignment(refs[2], second_profile_id)

        assert replaced.value == CharacterTTSAssignment(refs[2], second_profile_id)
        assert replaced.generation == 1
        assert (await repository.assignment_count(first_profile_id)).value == 3
        assert (await repository.assignment_count(second_profile_id)).value == 1
        for character_ref in (refs[0], refs[1], refs[3]):
            snapshot = (await repository.get_assigned_profile(character_ref)).value
            assert snapshot is not None
            assert snapshot.assignment == CharacterTTSAssignment(
                character_ref,
                first_profile_id,
            )
        replaced_snapshot = (await repository.get_assigned_profile(refs[2])).value
        assert replaced_snapshot is not None
        assert replaced_snapshot.assignment.profile_id == second_profile_id

        connection = sqlite3.connect(database_path)
        try:
            after = connection.execute(
                """
                SELECT created_at, updated_at
                FROM character_tts_assignments
                WHERE source = ? AND authority_id = ? AND character_id = ?
                """,
                (refs[2].source, refs[2].authority_id, refs[2].character_id),
            ).fetchone()
        finally:
            connection.close()
        assert after is not None
        assert after[0] == before[0]
        assert after[1] > before[1]


@pytest.mark.asyncio
async def test_assignment_missing_profile_is_missing_not_zero_or_partial(
    tmp_path: Path,
) -> None:
    missing_profile_id = UUID("f3000000-0000-4000-8000-000000000003")
    character_ref = CharacterRef(
        "server",
        "server-profile:' OR 1=1 --",
        "character:' OR 1=1 --",
    )

    async with _opened_repository(
        tmp_path / "missing-assignment.sqlite3"
    ) as repository:
        with pytest.raises(ProfileRepositoryError) as count_missing:
            await repository.assignment_count(missing_profile_id)
        _assert_safe_error(count_missing.value, "missing", str(missing_profile_id))

        with pytest.raises(ProfileRepositoryError) as set_missing:
            await repository.set_assignment(character_ref, missing_profile_id)
        _assert_safe_error(
            set_missing.value,
            "missing",
            str(missing_profile_id),
            character_ref.authority_id,
            character_ref.character_id,
        )

        assert (await repository.get_assigned_profile(character_ref)).value is None
        assert (await repository.remove_assignment(character_ref)).value is None


@pytest.mark.asyncio
async def test_orphan_assignment_is_corrupt_while_genuine_unassigned_is_none(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "secret-orphan-assignment.sqlite3"
    profile_id = UUID("f3100000-0000-4000-8000-000000000003")
    assigned_ref = CharacterRef(
        "server",
        "secret-orphan-authority",
        "secret-orphan-character",
    )
    unassigned_ref = CharacterRef(
        "server",
        "secret-unassigned-authority",
        "secret-unassigned-character",
    )

    async with _opened_repository(database_path) as repository:
        await repository.create_profile(_draft("Orphaned"), profile_id=profile_id)
        await repository.set_assignment(assigned_ref, profile_id)
        assert (await repository.get_assigned_profile(unassigned_ref)).value is None

        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            assert connection.execute("PRAGMA foreign_keys").fetchone() == (0,)
            connection.execute(
                "DELETE FROM tts_generation_profiles WHERE profile_id = ?",
                (str(profile_id),),
            )
        finally:
            connection.close()

        with pytest.raises(ProfileRepositoryError) as corrupt:
            await repository.get_assigned_profile(assigned_ref)
        _assert_safe_error(
            corrupt.value,
            "corrupt_data",
            str(profile_id),
            assigned_ref.authority_id,
            assigned_ref.character_id,
            str(database_path),
        )
        assert (await repository.get_assigned_profile(unassigned_ref)).value is None


@pytest.mark.asyncio
async def test_remove_assignment_is_exact_idempotent_and_does_not_probe_target(
    tmp_path: Path,
) -> None:
    profile_id = UUID("f4000000-0000-4000-8000-000000000004")
    exact_ref = CharacterRef("local", "detached-database", "same-character")
    other_ref = CharacterRef("local", "other-database", "same-character")

    async with _opened_repository(tmp_path / "detach.sqlite3") as repository:
        await repository.create_profile(_draft("Detach"), profile_id=profile_id)
        await repository.set_assignment(exact_ref, profile_id)
        await repository.set_assignment(other_ref, profile_id)

        first = await repository.remove_assignment(exact_ref)
        second = await repository.remove_assignment(exact_ref)

        assert first == ProfileStoreResult(generation=1, value=None)
        assert second == ProfileStoreResult(generation=1, value=None)
        assert (await repository.get_assigned_profile(exact_ref)).value is None
        assert (await repository.get_assigned_profile(other_ref)).value is not None
        assert (await repository.assignment_count(profile_id)).value == 1


@pytest.mark.asyncio
async def test_joined_read_returns_immutable_exact_revision_snapshot(
    tmp_path: Path,
) -> None:
    profile_id = UUID("f5000000-0000-4000-8000-000000000005")
    character_ref = CharacterRef("server", "stable-server-authority", "character-5")
    clock = _SequenceCallable(iter((CREATED_AT, UPDATED_AT, LATER_AT)))

    async with _opened_repository(
        tmp_path / "joined.sqlite3",
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        created = (
            await repository.create_profile(
                _draft("Joined Original"),
                profile_id=profile_id,
            )
        ).value
        await repository.set_assignment(character_ref, profile_id)

        joined = await repository.get_assigned_profile(character_ref)
        assert joined == ProfileStoreResult(
            generation=1,
            value=AssignedTTSProfileSnapshot(
                assignment=CharacterTTSAssignment(character_ref, profile_id),
                profile=created,
            ),
        )

        updated = (
            await repository.update_profile(
                profile_id,
                created.revision,
                _draft("Joined Updated", model_id="new-model"),
            )
        ).value
        current = (await repository.get_assigned_profile(character_ref)).value

        assert joined.value is not None
        assert joined.value.profile == created
        assert joined.value.profile is not created
        assert joined.value.profile.revision == 1
        assert joined.value.profile.display_name == "Joined Original"
        assert current is not None
        assert current.profile == updated
        assert current.profile.revision == 2


@pytest.mark.asyncio
async def test_delete_profile_is_blocked_by_repository_assignment(
    tmp_path: Path,
) -> None:
    profile_id = UUID("f6000000-0000-4000-8000-000000000006")
    character_ref = CharacterRef("local", "delete-protection", "character-6")

    async with _opened_repository(tmp_path / "protected-delete.sqlite3") as repository:
        profile = (
            await repository.create_profile(
                _draft("Protected"),
                profile_id=profile_id,
            )
        ).value
        await repository.set_assignment(character_ref, profile_id)

        with pytest.raises(ProfileRepositoryError) as conflict:
            await repository.delete_profile(profile_id)

        _assert_safe_error(conflict.value, "conflict")
        assert (await repository.get_profile(profile_id)).value == profile
        assert (await repository.get_assigned_profile(character_ref)).value is not None


@pytest.mark.asyncio
async def test_assignment_mutation_commit_failures_roll_back_and_recover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-assignment-commit.sqlite3"
    profile_id = UUID("f7000000-0000-4000-8000-000000000007")
    character_ref = CharacterRef(
        "local",
        "secret-assignment-authority",
        "secret-assignment-character",
    )

    async with _opened_repository(database_path) as repository:
        await repository.create_profile(
            _draft("Commit Assignment"), profile_id=profile_id
        )
        set_secret = "secret-set-assignment-commit"
        _fail_next_commit(
            monkeypatch,
            repository,
            sqlite3.OperationalError(set_secret),
        )

        with pytest.raises(ProfileRepositoryError) as set_failed:
            await repository.set_assignment(character_ref, profile_id)
        _assert_safe_error(
            set_failed.value,
            "operation_failed",
            set_secret,
            character_ref.authority_id,
            character_ref.character_id,
            str(database_path),
        )
        assert (await repository.get_assigned_profile(character_ref)).value is None

        await repository.set_assignment(character_ref, profile_id)
        remove_secret = "secret-remove-assignment-commit"
        _fail_next_commit(
            monkeypatch,
            repository,
            sqlite3.OperationalError(remove_secret),
        )
        with pytest.raises(ProfileRepositoryError) as remove_failed:
            await repository.remove_assignment(character_ref)
        _assert_safe_error(
            remove_failed.value,
            "operation_failed",
            remove_secret,
            character_ref.authority_id,
            character_ref.character_id,
            str(database_path),
        )
        assert (await repository.get_assigned_profile(character_ref)).value is not None
        assert (await repository.remove_assignment(character_ref)).value is None


@pytest.mark.asyncio
async def test_unrelated_assignment_foreign_key_trigger_is_not_missing(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "secret-unrelated-assignment-fk.sqlite3"
    profile_id = UUID("f8000000-0000-4000-8000-000000000008")
    character_ref = CharacterRef("server", "trigger-authority", "trigger-character")
    secret = "secret-unrelated-assignment-fk"

    async with _opened_repository(database_path) as repository:
        await repository.create_profile(
            _draft("Trigger Assignment"), profile_id=profile_id
        )
        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            CREATE TABLE assignment_constraint_parent(id INTEGER PRIMARY KEY);
            """,
        )
        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
                CREATE TABLE assignment_constraint_probe(
                    parent_id INTEGER REFERENCES assignment_constraint_parent(id)
                );
                CREATE TRIGGER force_unrelated_assignment_fk
                BEFORE INSERT ON character_tts_assignments
                BEGIN
                    INSERT INTO assignment_constraint_probe(parent_id)
                    VALUES (1);
                END;
                """
            )
        finally:
            connection.close()

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.set_assignment(character_ref, profile_id)
        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            character_ref.authority_id,
            character_ref.character_id,
            str(database_path),
        )
        assert (await repository.get_assigned_profile(character_ref)).value is None

        await asyncio.to_thread(
            _external_execute,
            database_path,
            "DROP TRIGGER force_unrelated_assignment_fk",
        )
        assert (
            await repository.set_assignment(character_ref, profile_id)
        ).value.profile_id == profile_id


@pytest.mark.asyncio
async def test_missing_assignment_target_preflight_returns_missing_before_unrelated_fk_trigger(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "secret-missing-target-unrelated-fk.sqlite3"
    missing_profile_id = UUID("f8100000-0000-4000-8000-000000000008")
    character_ref = CharacterRef(
        "server",
        "secret-missing-trigger-authority",
        "secret-missing-trigger-character",
    )
    secret = "secret-missing-target-unrelated-fk"

    async with _opened_repository(database_path) as repository:
        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
                CREATE TABLE missing_assignment_constraint_parent(
                    id INTEGER PRIMARY KEY
                );
                CREATE TABLE missing_assignment_constraint_probe(
                    parent_id INTEGER
                        REFERENCES missing_assignment_constraint_parent(id)
                );
                CREATE TRIGGER force_missing_target_unrelated_assignment_fk
                BEFORE INSERT ON character_tts_assignments
                BEGIN
                    INSERT INTO missing_assignment_constraint_probe(parent_id)
                    VALUES (1);
                END;
                """
            )
        finally:
            connection.close()

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.set_assignment(character_ref, missing_profile_id)
        _assert_safe_error(
            failed.value,
            "missing",
            secret,
            str(missing_profile_id),
            character_ref.authority_id,
            character_ref.character_id,
            str(database_path),
        )
        assert (await repository.get_assigned_profile(character_ref)).value is None


@pytest.mark.asyncio
async def test_corrupt_assignment_and_joined_profile_rows_fail_closed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "secret-corrupt-joined.sqlite3"
    profile_id = UUID("f9000000-0000-4000-8000-000000000009")
    character_ref = CharacterRef(
        "server",
        "secret-corrupt-authority",
        "secret-corrupt-character",
    )
    assignment_secret = "secret-corrupt-assignment-timestamp"
    profile_secret = "secret-corrupt-joined-profile"

    async with _opened_repository(database_path) as repository:
        profile = (
            await repository.create_profile(
                _draft("Corrupt Joined"),
                profile_id=profile_id,
            )
        ).value
        await repository.set_assignment(character_ref, profile_id)
        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            UPDATE character_tts_assignments
            SET created_at = ?
            WHERE source = ? AND authority_id = ? AND character_id = ?
            """,
            (
                assignment_secret,
                character_ref.source,
                character_ref.authority_id,
                character_ref.character_id,
            ),
        )
        with pytest.raises(ProfileRepositoryError) as corrupt_assignment:
            await repository.get_assigned_profile(character_ref)
        _assert_safe_error(
            corrupt_assignment.value,
            "corrupt_data",
            assignment_secret,
            character_ref.authority_id,
            character_ref.character_id,
            str(database_path),
        )

        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            UPDATE character_tts_assignments
            SET created_at = ?, updated_at = ?
            WHERE source = ? AND authority_id = ? AND character_id = ?
            """,
            (
                CREATED_AT.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                CREATED_AT.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                character_ref.source,
                character_ref.authority_id,
                character_ref.character_id,
            ),
        )
        await asyncio.to_thread(
            _external_execute,
            database_path,
            "UPDATE tts_generation_profiles SET options_json = ? WHERE profile_id = ?",
            (f'{{"{profile_secret}":NaN}}', str(profile_id)),
        )
        with pytest.raises(ProfileRepositoryError) as corrupt_profile:
            await repository.get_assigned_profile(character_ref)
        _assert_safe_error(
            corrupt_profile.value,
            "corrupt_data",
            profile_secret,
            character_ref.authority_id,
            character_ref.character_id,
            str(database_path),
        )

        await asyncio.to_thread(
            _external_execute,
            database_path,
            "UPDATE tts_generation_profiles SET options_json = ? WHERE profile_id = ?",
            (canonical_json_options(profile.options), str(profile_id)),
        )
        assert (await repository.get_assigned_profile(character_ref)).value is not None


@pytest.mark.asyncio
async def test_hostile_joined_codec_failure_is_bounded_without_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-hostile-joined-codec.sqlite3"
    profile_id = UUID("fa000000-0000-4000-8000-00000000000a")
    character_ref = CharacterRef("local", "codec-authority", "codec-character")
    secret = "secret-hostile-joined-codec"

    async with _opened_repository(database_path) as repository:
        await repository.create_profile(_draft("Codec Joined"), profile_id=profile_id)
        await repository.set_assignment(character_ref, profile_id)

        def hostile_decode(_row: object) -> AssignedTTSProfileSnapshot:
            try:
                raise RuntimeError(secret)
            except RuntimeError:
                raise ValueError(secret)

        monkeypatch.setattr(
            profile_repository,
            "decode_assigned_snapshot",
            hostile_decode,
        )
        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.get_assigned_profile(character_ref)
        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            character_ref.authority_id,
            character_ref.character_id,
            str(database_path),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt_row", [None, (), ("1",), (-1,), (True,)])
async def test_assignment_count_corrupt_rows_fail_closed(
    tmp_path: Path,
    corrupt_row: object,
) -> None:
    database_path = tmp_path / "secret-corrupt-count.sqlite3"
    profile_id = UUID("fb000000-0000-4000-8000-00000000000b")
    repository = profile_repository.TTSProfileRepository(database_path)
    await repository.open()
    try:
        await repository.create_profile(_draft("Corrupt Count"), profile_id=profile_id)
        connection = repository._connection
        assert connection is not None
        repository._connection = cast(
            sqlite3.Connection,
            cast(Any, _CountRowConnection(connection, corrupt_row)),
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.assignment_count(profile_id)
        _assert_safe_error(
            failed.value,
            "corrupt_data",
            str(profile_id),
            str(database_path),
        )
    finally:
        await repository.close()


@pytest.mark.parametrize("attempt", range(3))
def test_spawned_set_delete_race_is_serialized_without_partial_mutation(
    tmp_path: Path,
    attempt: int,
) -> None:
    database_path = tmp_path / f"assignment-delete-race-{attempt}.sqlite3"
    profile_id = UUID("fc000000-0000-4000-8000-00000000000c")

    async def initialize() -> None:
        async with _opened_repository(database_path) as repository:
            await repository.create_profile(
                _draft("Race Assignment"),
                profile_id=profile_id,
            )

    asyncio.run(initialize())
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    receivers: list[Connection] = []
    child_connections: list[Connection] = []
    processes: list[Any] = []
    outcomes: list[tuple[object, ...]] = []
    exitcodes: list[int | None] = []

    try:
        for operation in ("set", "delete"):
            receiver, child_connection = context.Pipe(duplex=False)
            process = context.Process(
                target=_spawn_assignment_delete_race,
                args=(
                    str(database_path),
                    child_connection,
                    release,
                    operation,
                    str(profile_id),
                ),
            )
            receivers.append(receiver)
            child_connections.append(child_connection)
            process.start()
            processes.append(process)
            child_connection.close()

        for receiver in receivers:
            assert receiver.poll(15.0), "spawned assignment race did not become ready"
            assert receiver.recv()[0] == "ready"
        release.set()
        for receiver in receivers:
            assert receiver.poll(15.0), "spawned assignment race returned no outcome"
            message = receiver.recv()
            assert message[0] == "outcome", message
            outcomes.append(cast(tuple[object, ...], message[1]))
    finally:
        release.set()
        for child_connection in child_connections:
            child_connection.close()
        for process in processes:
            process.join(10.0)
            if process.is_alive():
                process.terminate()
                process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join(5.0)
            exitcodes.append(process.exitcode)
        for receiver in receivers:
            receiver.close()
        for process in processes:
            process.close()

    assert exitcodes == [0, 0]
    normalized = {(outcome[0], outcome[1], outcome[2]) for outcome in outcomes}
    assert normalized in (
        {("set", "success", 1), ("delete", "failure", "conflict")},
        {("set", "failure", "missing"), ("delete", "success", 1)},
    )
    for outcome in outcomes:
        if outcome[1] == "failure":
            assert outcome[3] == (f"TTS profile repository failed: {outcome[2]}")
            assert outcome[4:] == (True, True, ())

    async def verify_serialized_outcome() -> None:
        character_ref = CharacterRef(
            "server",
            "server-profile:principal",
            "shared-character",
        )
        async with _opened_repository(database_path) as repository:
            page = (await repository.list_profiles()).value
            snapshot = (await repository.get_assigned_profile(character_ref)).value
            if ("set", "success", 1) in normalized:
                assert page.total == 1
                assert snapshot is not None
                assert snapshot.assignment.profile_id == profile_id
                assert (await repository.assignment_count(profile_id)).value == 1
            else:
                assert page.total == 0
                assert snapshot is None

    asyncio.run(verify_serialized_outcome())


@pytest.mark.asyncio
async def test_crud_sql_runs_only_on_repository_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    event_loop_thread = threading.get_ident()
    traced_threads: list[int] = []
    real_open = profile_repository.open_profile_store

    def traced_open(path: Path) -> sqlite3.Connection:
        connection = real_open(path)
        connection.set_trace_callback(
            lambda _statement: traced_threads.append(threading.get_ident())
        )
        return connection

    monkeypatch.setattr(profile_repository, "open_profile_store", traced_open)
    clock = _SequenceCallable(iter((CREATED_AT, UPDATED_AT)))

    async with _opened_repository(
        database_path,
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        created = (
            await repository.create_profile(
                _draft("Worker"),
                profile_id=GENERATED_ID,
            )
        ).value
        await repository.get_profile(created.profile_id)
        await repository.list_profiles(search="work", limit=1, offset=0)
        await repository.update_profile(
            created.profile_id,
            created.revision,
            _draft("Worker Updated"),
        )
        await repository.delete_profile(created.profile_id)

    assert traced_threads
    assert len(set(traced_threads)) == 1
    assert traced_threads[0] != event_loop_thread
