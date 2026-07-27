"""Lifecycle tests for the serialized TTS profile repository."""

from __future__ import annotations

import asyncio
import gc
import importlib
import os
import sqlite3
import sys
import threading
import traceback
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from uuid import UUID

import pytest

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_schema import (
    open_profile_store,
    validate_profile_candidate,
)
from tldw_chatbook.TTS.profile_store_lock import (
    ProfileStoreLease,
    ProfileStoreLockMode,
)
from tldw_chatbook.TTS.profile_types import (
    ProfileBackupReceipt,
    ProfileRepositoryState,
    ProfileRestoreReceipt,
    ProfileStoreResult,
    TTSProfileDraft,
)


class _ControlFlow(BaseException):
    """A test-only control-flow signal."""


class _RecordingConnection:
    def __init__(
        self,
        events: list[tuple[str, int]],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.close_error = close_error
        self.closed = False

    def close(self) -> None:
        self.events.append(("connection.close", threading.get_ident()))
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class _CloseFailingSQLiteProxy:
    def __init__(self, connection: sqlite3.Connection, secret: str) -> None:
        self.connection = connection
        self.secret = secret
        self.fail_close = True
        self.close_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connection, name)

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError(self.secret)
        self.connection.close()


class _RecordingLease:
    def __init__(
        self,
        events: list[tuple[str, int]],
        *,
        acquire_error: BaseException | None = None,
        release_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.acquire_error = acquire_error
        self.release_error = release_error
        self.acquired = False

    def acquire(self) -> _RecordingLease:
        self.events.append(("lease.acquire", threading.get_ident()))
        if self.acquire_error is not None:
            raise self.acquire_error
        self.acquired = True
        return self

    def release(self) -> None:
        self.events.append(("lease.release", threading.get_ident()))
        self.acquired = False
        if self.release_error is not None:
            raise self.release_error


class _SequencedCloseConnection:
    def __init__(
        self,
        events: list[tuple[str, int]],
        label: str,
        close_errors: list[BaseException | None],
        *,
        before_close: Callable[[], None] | None = None,
    ) -> None:
        self.events = events
        self.label = label
        self.close_errors = close_errors
        self.before_close = before_close
        self.close_calls = 0
        self.closed = False

    def close(self) -> None:
        if self.before_close is not None:
            self.before_close()
        self.events.append((f"{self.label}.close", threading.get_ident()))
        self.close_calls += 1
        error = self.close_errors.pop(0) if self.close_errors else None
        if error is not None:
            raise error
        self.closed = True


class _SequencedReleaseLease:
    def __init__(
        self,
        events: list[tuple[str, int]],
        label: str,
        release_errors: list[BaseException | None],
    ) -> None:
        self.events = events
        self.label = label
        self.release_errors = release_errors
        self.acquire_calls = 0
        self.release_calls = 0
        self.acquired = False

    def acquire(self) -> _SequencedReleaseLease:
        self.events.append((f"{self.label}.acquire", threading.get_ident()))
        self.acquire_calls += 1
        self.acquired = True
        return self

    def release(self) -> None:
        self.events.append((f"{self.label}.release", threading.get_ident()))
        self.release_calls += 1
        error = self.release_errors.pop(0) if self.release_errors else None
        if error is not None:
            raise error
        self.acquired = False


class _ObservedProfileStoreLease(ProfileStoreLease):
    def __init__(
        self,
        database_path: Path,
        mode: ProfileStoreLockMode,
        events: list[tuple[str, int]],
        label: str,
    ) -> None:
        super().__init__(database_path, mode)
        self.events = events
        self.label = label

    def acquire(self) -> ProfileStoreLease:
        self.events.append((f"{self.label}.acquire", threading.get_ident()))
        return super().acquire()

    def release(self) -> None:
        self.events.append((f"{self.label}.release", threading.get_ident()))
        super().release()


class _RecordingExecutor:
    def __init__(
        self,
        max_workers: int,
        events: list[tuple[str, int]],
    ) -> None:
        self.events = events
        self.max_workers = max_workers
        self.shutdown_calls = 0
        self._delegate = RealThreadPoolExecutor(max_workers=max_workers)
        self.events.append(("executor.construct", threading.get_ident()))

    def submit(
        self,
        function: Any,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        return self._delegate.submit(function, *args, **kwargs)

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        self.shutdown_calls += 1
        self.events.append(("executor.shutdown", threading.get_ident()))
        self._delegate.shutdown(wait=wait, cancel_futures=cancel_futures)


def _repository_module() -> ModuleType:
    try:
        return importlib.import_module("tldw_chatbook.TTS.profile_repository")
    except ModuleNotFoundError:
        pytest.fail("TTS profile repository lifecycle is not implemented")
    raise AssertionError("unreachable")


def _repository(database_path: Path) -> Any:
    return _repository_module().TTSProfileRepository(database_path)


def _draft(name: str) -> TTSProfileDraft:
    return TTSProfileDraft(
        display_name=name,
        provider_id="audio_cpp",
        model_id="supertonic",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
    )


async def _create_profile_store(path: Path, *names: str) -> None:
    repository = _repository(path)
    await repository.open()
    try:
        for index, name in enumerate(names, start=1):
            await repository.create_profile(
                _draft(name),
                UUID(f"00000000-0000-4000-8000-{index:012d}"),
            )
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
        assert secret not in visible


def _tainted_repository_error(
    code: str,
    secret: str,
) -> ProfileRepositoryError:
    try:
        raise RuntimeError(secret)
    except RuntimeError:
        try:
            raise ProfileRepositoryError(code)
        except ProfileRepositoryError as error:
            return error


def _try_exclusive_lease(database_path: Path) -> ProfileRepositoryError | None:
    contender = ProfileStoreLease(
        database_path,
        ProfileStoreLockMode.EXCLUSIVE,
        timeout_seconds=0.05,
        check_interval_seconds=0.005,
    )
    try:
        contender.acquire()
    except ProfileRepositoryError as error:
        return error
    try:
        return None
    finally:
        contender.release()


async def _assert_exclusive_lease_blocked(database_path: Path) -> None:
    error = await asyncio.to_thread(_try_exclusive_lease, database_path)
    assert error is not None
    _assert_safe_error(error, "lock_timeout", str(database_path))


async def _wait_thread_event(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 5.0)


async def _run_in_new_loop_thread(
    awaitable_factory: Callable[[], Awaitable[Any]],
) -> tuple[Any | None, BaseException | None]:
    outcomes: list[tuple[Any | None, BaseException | None]] = []

    async def invoke() -> Any:
        return await awaitable_factory()

    def runner() -> None:
        try:
            outcomes.append((asyncio.run(invoke()), None))
        except BaseException as error:
            outcomes.append((None, error))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    await asyncio.to_thread(thread.join, 5.0)
    assert not thread.is_alive()
    assert len(outcomes) == 1
    return outcomes[0]


def _install_fake_store(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    events: list[tuple[str, int]],
    connection: _RecordingConnection,
    *,
    leases: list[_RecordingLease] | None = None,
) -> list[_RecordingLease]:
    recorded_leases = leases if leases is not None else []

    def lease_factory(
        _database_path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _RecordingLease:
        assert mode is ProfileStoreLockMode.SHARED
        lease = _RecordingLease(events)
        recorded_leases.append(lease)
        return lease

    def opener(_database_path: Path) -> Any:
        events.append(("store.open", threading.get_ident()))
        return connection

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(module, "open_profile_store", opener)
    return recorded_leases


def _phase_threads(
    events: list[tuple[str, int]],
    phase: str,
) -> list[int]:
    return [thread_id for name, thread_id in events if name == phase]


def test_constructor_is_pure_and_starts_initial_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "nested" / "profiles.sqlite3"
    entries_before = tuple(tmp_path.iterdir())

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("constructor performed lazy lifecycle work")

    monkeypatch.setattr(module, "ThreadPoolExecutor", forbidden)
    monkeypatch.setattr(module, "ProfileStoreLease", forbidden)
    monkeypatch.setattr(module, "open_profile_store", forbidden)

    repository = module.TTSProfileRepository(database_path)

    assert tuple(tmp_path.iterdir()) == entries_before
    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.generation == 0
    assert repository.terminal is False
    assert repository._executor is None
    assert repository._connection is None
    assert repository._lease is None
    assert repository._owner_loop is None
    assert repository._lifecycle_lock is None


def test_constructor_rejects_non_path_without_exposing_value(tmp_path: Path) -> None:
    secret = str(tmp_path / "secret-profile-store.sqlite3")

    with pytest.raises(ProfileRepositoryError) as caught:
        _repository_module().TTSProfileRepository(secret)

    _assert_safe_error(caught.value, "operation_failed", secret)


@pytest.mark.asyncio
async def test_open_uses_one_worker_for_lease_connection_sql_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    loop_thread = threading.get_ident()
    events: list[tuple[str, int]] = []
    sql_traces: list[tuple[int, str]] = []

    class RecordingLease(ProfileStoreLease):
        def acquire(self) -> ProfileStoreLease:
            events.append(("lease.acquire", threading.get_ident()))
            return super().acquire()

        def release(self) -> None:
            events.append(("lease.release", threading.get_ident()))
            super().release()

    def traced_open(path: Path) -> sqlite3.Connection:
        events.append(("store.open", threading.get_ident()))
        connection = open_profile_store(path)
        connection.set_trace_callback(
            lambda statement: sql_traces.append((threading.get_ident(), statement))
        )
        return connection

    monkeypatch.setattr(module, "ProfileStoreLease", RecordingLease)
    monkeypatch.setattr(module, "open_profile_store", traced_open)
    repository = module.TTSProfileRepository(database_path)

    opened = await repository.open()

    def query(connection: sqlite3.Connection) -> int:
        events.append(("operation", threading.get_ident()))
        return cast(
            int,
            connection.execute(
                "SELECT count(*) FROM tts_generation_profiles"
            ).fetchone()[0],
        )

    result = await repository._submit_operation(query)
    closed = await repository.close()

    worker_threads = {
        thread_id
        for phase, thread_id in events
        if phase in {"lease.acquire", "store.open", "operation", "lease.release"}
    }
    assert opened == ProfileStoreResult(generation=1, value=None)
    assert result == ProfileStoreResult(generation=1, value=0)
    assert closed == ProfileStoreResult(generation=2, value=None)
    assert len(worker_threads) == 1
    assert loop_thread not in worker_threads
    assert sql_traces
    assert {thread_id for thread_id, _statement in sql_traces} == worker_threads
    assert [phase for phase, _thread_id in events].index("lease.acquire") < [
        phase for phase, _thread_id in events
    ].index("store.open")


@pytest.mark.asyncio
async def test_open_and_close_are_idempotent_and_shutdown_once_off_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    executors: list[_RecordingExecutor] = []

    def executor_factory(max_workers: int) -> _RecordingExecutor:
        executor = _RecordingExecutor(max_workers, events)
        executors.append(executor)
        return executor

    monkeypatch.setattr(module, "ThreadPoolExecutor", executor_factory)
    leases = _install_fake_store(
        monkeypatch,
        module,
        events,
        connection,
    )
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    loop_thread = threading.get_ident()

    first_open = await repository.open()
    second_open = await repository.open()
    first_close = await repository.close()
    second_close = await repository.close()

    assert first_open == second_open == ProfileStoreResult(generation=1, value=None)
    assert (
        first_close
        == second_close
        == ProfileStoreResult(
            generation=2,
            value=None,
        )
    )
    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.generation == 2
    assert repository.terminal is True
    assert len(executors) == 1
    assert executors[0].max_workers == 1
    assert executors[0].shutdown_calls == 1
    assert len(leases) == 1
    assert _phase_threads(events, "lease.acquire") == _phase_threads(
        events, "store.open"
    )
    assert _phase_threads(events, "connection.close") == _phase_threads(
        events, "lease.release"
    )
    assert _phase_threads(events, "lease.acquire") == _phase_threads(
        events, "connection.close"
    )
    assert _phase_threads(events, "executor.shutdown")[0] != loop_thread

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()
    _assert_safe_error(caught.value, "terminal", str(tmp_path))
    assert len(executors) == 1


@pytest.mark.asyncio
async def test_close_before_open_is_terminal_without_creating_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    executor_calls = 0

    def forbidden_executor(_max_workers: int) -> Any:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError

    monkeypatch.setattr(module, "ThreadPoolExecutor", forbidden_executor)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    first = await repository.close()
    second = await repository.close()

    assert first == second == ProfileStoreResult(generation=1, value=None)
    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.generation == 1
    assert repository.terminal is True
    assert executor_calls == 0
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()
    _assert_safe_error(caught.value, "terminal")
    assert executor_calls == 0


@pytest.mark.asyncio
async def test_concurrent_open_calls_share_one_attempt_and_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    open_started = threading.Event()
    allow_open = threading.Event()
    leases = _install_fake_store(
        monkeypatch,
        module,
        events,
        connection,
    )

    def blocked_open(_database_path: Path) -> Any:
        events.append(("store.open", threading.get_ident()))
        open_started.set()
        assert allow_open.wait(5.0)
        return connection

    monkeypatch.setattr(module, "open_profile_store", blocked_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    second_started = asyncio.Event()

    first_task = asyncio.create_task(repository.open())

    async def second_open() -> ProfileStoreResult[None]:
        second_started.set()
        return cast(ProfileStoreResult[None], await repository.open())

    second_task = asyncio.create_task(second_open())
    await _wait_thread_event(open_started)
    await second_started.wait()
    allow_open.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first == second == ProfileStoreResult(generation=1, value=None)
    assert len(leases) == 1
    assert len(_phase_threads(events, "store.open")) == 1
    assert repository.generation == 1
    await repository.close()


@pytest.mark.asyncio
async def test_active_open_rejects_foreign_loop_before_joining_shared_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    open_started = threading.Event()
    allow_open = threading.Event()
    open_calls = 0
    _install_fake_store(monkeypatch, module, events, connection)

    def blocked_open(_database_path: Path) -> Any:
        nonlocal open_calls
        open_calls += 1
        open_started.set()
        assert allow_open.wait(5.0)
        return connection

    monkeypatch.setattr(module, "open_profile_store", blocked_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    main_open = asyncio.create_task(repository.open())
    await _wait_thread_event(open_started)

    try:
        result, error = await _run_in_new_loop_thread(repository.open)

        assert result is None
        assert isinstance(error, ProfileRepositoryError)
        _assert_safe_error(error, "invalid_state", "event loop", repr(main_open))
        assert main_open.done() is False
        assert repository.generation == 1
        assert open_calls == 1
    finally:
        allow_open.set()
        await main_open
        await repository.close()


@pytest.mark.asyncio
async def test_operation_caller_paths_reject_foreign_loop_without_worker_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    foreign_work_ran = threading.Event()
    admission = repository._admit_operation(lambda _connection: "main-result")

    async def admit_from_foreign_loop() -> None:
        repository._admit_operation(lambda _connection: foreign_work_ran.set())

    caller_paths: tuple[Callable[[], Awaitable[Any]], ...] = (
        admit_from_foreign_loop,
        lambda: repository._submit_operation(
            lambda _connection: foreign_work_ran.set()
        ),
        lambda: repository._publish_operation(admission),
    )

    try:
        for caller_path in caller_paths:
            result, error = await _run_in_new_loop_thread(caller_path)
            assert result is None
            assert isinstance(error, ProfileRepositoryError)
            _assert_safe_error(error, "invalid_state", "event loop")
        assert foreign_work_ran.is_set() is False
        assert await repository._publish_operation(admission) == ProfileStoreResult(
            generation=1,
            value="main-result",
        )
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_terminal_close_rejects_foreign_loop_but_stays_idempotent_on_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    closed = await repository.close()

    result, error = await _run_in_new_loop_thread(repository.close)

    assert result is None
    assert isinstance(error, ProfileRepositoryError)
    _assert_safe_error(error, "invalid_state", "event loop")
    assert await repository.close() == closed
    assert repository.generation == 2


@pytest.mark.asyncio
async def test_overlapping_failed_open_calls_share_attempt_before_later_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    first_attempt_started = threading.Event()
    release_first_failure = threading.Event()
    attempts = 0
    failures_enabled = True
    leases = _install_fake_store(
        monkeypatch,
        module,
        events,
        connection,
    )

    def controlled_open(_database_path: Path) -> Any:
        nonlocal attempts
        attempts += 1
        events.append(("store.open", threading.get_ident()))
        if failures_enabled:
            if attempts == 1:
                first_attempt_started.set()
                assert release_first_failure.wait(5.0)
            raise ProfileRepositoryError("schema_corrupt")
        return connection

    monkeypatch.setattr(module, "open_profile_store", controlled_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    async def capture_open_error() -> ProfileRepositoryError:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.open()
        return caught.value

    first_task = asyncio.create_task(capture_open_error())
    await _wait_thread_event(first_attempt_started)
    second_invoked = asyncio.Event()

    async def overlapping_open() -> ProfileRepositoryError:
        second_invoked.set()
        return await capture_open_error()

    second_task = asyncio.create_task(overlapping_open())
    await second_invoked.wait()
    release_first_failure.set()
    first_error, second_error = await asyncio.gather(first_task, second_task)

    assert first_error is not second_error
    _assert_safe_error(first_error, "schema_corrupt")
    _assert_safe_error(second_error, "schema_corrupt")
    assert attempts == 1
    assert len(leases) == 1
    assert len(_phase_threads(events, "lease.acquire")) == 1
    assert len(_phase_threads(events, "store.open")) == 1
    assert len(set(_phase_threads(events, "store.open"))) == 1
    assert leases[0].acquired is False
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1

    failures_enabled = False
    retried = await repository.open()

    assert retried == ProfileStoreResult(generation=2, value=None)
    assert attempts == 2
    assert len(leases) == 2
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    await repository.close()


@pytest.mark.asyncio
async def test_cancelled_overlapping_open_waits_for_shared_attempt_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    attempt_started = threading.Event()
    release_failure = threading.Event()
    attempts = 0
    _install_fake_store(monkeypatch, module, events, connection)

    def controlled_open(_database_path: Path) -> Any:
        nonlocal attempts
        attempts += 1
        attempt_started.set()
        assert release_failure.wait(5.0)
        raise ProfileRepositoryError("schema_corrupt")

    monkeypatch.setattr(module, "open_profile_store", controlled_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    first_task = asyncio.create_task(repository.open())
    await _wait_thread_event(attempt_started)
    second_invoked = asyncio.Event()

    async def overlapping_open() -> ProfileStoreResult[None]:
        second_invoked.set()
        return cast(ProfileStoreResult[None], await repository.open())

    second_task = asyncio.create_task(overlapping_open())
    await second_invoked.wait()
    second_task.cancel()
    loop = asyncio.get_running_loop()
    cancellation_checkpoint: asyncio.Future[bool] = loop.create_future()

    def observe_after_cancellation_delivery() -> None:
        loop.call_soon(cancellation_checkpoint.set_result, second_task.done())

    loop.call_soon(observe_after_cancellation_delivery)

    cancelled_before_settlement = await cancellation_checkpoint
    release_failure.set()
    assert cancelled_before_settlement is False
    with pytest.raises(ProfileRepositoryError) as first_error:
        await first_task
    with pytest.raises(asyncio.CancelledError):
        await second_task

    _assert_safe_error(first_error.value, "schema_corrupt")
    assert attempts == 1
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    await repository.close()


@pytest.mark.asyncio
async def test_invalid_existing_store_fails_closed_and_retry_can_recover(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE unrelated(value TEXT)")
    connection.commit()
    connection.close()
    invalid_bytes = database_path.read_bytes()
    repository = _repository(database_path)

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    _assert_safe_error(caught.value, "schema_partial", str(database_path))
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    assert repository.terminal is False
    assert database_path.read_bytes() == invalid_bytes
    executor = repository._executor

    database_path.unlink()
    retried = await repository.open()

    assert retried == ProfileStoreResult(generation=2, value=None)
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    assert repository._executor is executor
    await repository.close()


@pytest.mark.asyncio
async def test_failed_open_cleans_partial_lease_and_maps_hostile_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    secret = str(tmp_path / "secret-store.sqlite3")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    leases = _install_fake_store(
        monkeypatch,
        module,
        events,
        connection,
    )

    def hostile_open(_database_path: Path) -> Any:
        events.append(("store.open", threading.get_ident()))
        raise RuntimeError(secret)

    monkeypatch.setattr(module, "open_profile_store", hostile_open)
    repository = module.TTSProfileRepository(Path(secret))

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    _assert_safe_error(caught.value, "operation_failed", secret)
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    assert repository._executor is not None
    assert len(leases) == 1
    assert leases[0].acquired is False
    assert _phase_threads(events, "lease.acquire") == _phase_threads(
        events, "lease.release"
    )

    def healthy_open(_database_path: Path) -> Any:
        events.append(("store.open", threading.get_ident()))
        return connection

    monkeypatch.setattr(module, "open_profile_store", healthy_open)
    retried = await repository.open()

    assert retried == ProfileStoreResult(generation=2, value=None)
    assert len(leases) == 2
    await repository.close()


@pytest.mark.asyncio
async def test_failed_open_retains_real_shared_lease_until_connection_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    secret = str(tmp_path / "partial-connection-close-failure")
    events: list[tuple[str, int]] = []
    previous_worker_trace: Any = None

    def restore_worker_trace() -> None:
        sys.settrace(previous_worker_trace)

    first_connection = _SequencedCloseConnection(
        events,
        "connection-1",
        [RuntimeError(secret), None],
        before_close=restore_worker_trace,
    )
    second_connection = _RecordingConnection(events)
    leases: list[_ObservedProfileStoreLease] = []
    open_calls = 0
    injected_failure = threading.Event()
    worker_trace_checked = threading.Event()
    opening_error = RuntimeError("injected-open-transition-failure")

    def passthrough_worker_trace(
        _frame: Any,
        _event: str,
        _argument: Any,
    ) -> Any:
        return passthrough_worker_trace

    def lease_factory(
        path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _ObservedProfileStoreLease:
        lease = _ObservedProfileStoreLease(
            path,
            mode,
            events,
            f"lease-{len(leases) + 1}",
        )
        leases.append(lease)
        return lease

    def fail_after_connection_assignment(
        frame: Any,
        event: str,
        _argument: Any,
    ) -> Any:
        if (
            frame.f_code is module.TTSProfileRepository._worker_open.__code__
            and event == "line"
            and frame.f_locals.get("connection") is first_connection
        ):
            injected_failure.set()
            frame.f_trace = None
            raise opening_error
        return fail_after_connection_assignment

    def controlled_open(_database_path: Path) -> Any:
        nonlocal open_calls, previous_worker_trace
        open_calls += 1
        if open_calls == 1:
            if sys.gettrace() is None:
                sys.settrace(passthrough_worker_trace)
            previous_worker_trace = sys.gettrace()
            worker_frame = sys._getframe(1)
            worker_frame.f_trace = fail_after_connection_assignment
            sys.settrace(fail_after_connection_assignment)
            return first_connection
        assert sys.gettrace() is previous_worker_trace
        worker_trace_checked.set()
        return second_connection

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(module, "open_profile_store", controlled_open)
    repository = module.TTSProfileRepository(database_path)

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.open()

        _assert_safe_error(caught.value, "operation_failed", secret)
        assert injected_failure.is_set()
        assert repository._connection is first_connection
        assert repository._lease is leases[0]
        assert first_connection.closed is False
        assert first_connection.close_calls == 1
        assert leases[0].acquired is True
        assert not _phase_threads(events, "lease-1.release")
        await _assert_exclusive_lease_blocked(database_path)

        retried = await repository.open()

        assert retried == ProfileStoreResult(generation=2, value=None)
        assert first_connection.closed is True
        assert first_connection.close_calls == 2
        assert leases[0].acquired is False
        assert repository._connection is second_connection
        assert repository._lease is leases[1]
        assert leases[1].acquired is True
        assert worker_trace_checked.is_set()
        phases = [phase for phase, _thread_id in events]
        assert max(
            index for index, phase in enumerate(phases) if phase == "connection-1.close"
        ) < phases.index("lease-1.release")
        assert phases.index("lease-1.release") < phases.index("lease-2.acquire")
        await _assert_exclusive_lease_blocked(database_path)
    finally:
        if not repository.terminal:
            await repository.close()

    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_is_control", [True, False])
async def test_failed_open_connection_cleanup_control_preserves_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_is_control: bool,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    primary_signal = _ControlFlow()
    cleanup_signal = _ControlFlow()
    primary_error: BaseException
    expected_signal: _ControlFlow
    if primary_is_control:
        primary_error = primary_signal
        expected_signal = primary_signal
    else:
        primary_error = RuntimeError("ordinary-open-transition-failure")
        expected_signal = cleanup_signal
    previous_worker_trace: Any = None

    def restore_worker_trace() -> None:
        sys.settrace(previous_worker_trace)

    connection = _SequencedCloseConnection(
        events,
        "connection",
        [cleanup_signal, None],
        before_close=restore_worker_trace,
    )
    lease = _SequencedReleaseLease(events, "lease", [])

    def lease_factory(
        _database_path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _SequencedReleaseLease:
        assert mode is ProfileStoreLockMode.SHARED
        return lease

    def fail_after_connection_assignment(
        frame: Any,
        event: str,
        _argument: Any,
    ) -> Any:
        if (
            frame.f_code is module.TTSProfileRepository._worker_open.__code__
            and event == "line"
            and frame.f_locals.get("connection") is connection
        ):
            frame.f_trace = None
            raise primary_error
        return fail_after_connection_assignment

    def partial_open(_database_path: Path) -> Any:
        nonlocal previous_worker_trace
        previous_worker_trace = sys.gettrace()
        worker_frame = sys._getframe(1)
        worker_frame.f_trace = fail_after_connection_assignment
        sys.settrace(fail_after_connection_assignment)
        return connection

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(module, "open_profile_store", partial_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    try:
        with pytest.raises(_ControlFlow) as caught:
            await repository.open()

        assert caught.value is expected_signal
        assert repository._connection is connection
        assert repository._lease is lease
        assert connection.closed is False
        assert lease.acquired is True
        assert lease.release_calls == 0

        assert await repository.close() == ProfileStoreResult(
            generation=2,
            value=None,
        )
        assert connection.closed is True
        assert lease.acquired is False
        assert lease.release_calls == 1
    finally:
        if not repository.terminal:
            await repository.close()


@pytest.mark.asyncio
async def test_failed_open_retains_lease_until_retry_cleanup_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    secret = str(tmp_path / "residual-lease-failure")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    leases: list[_SequencedReleaseLease] = []
    open_calls = 0

    def lease_factory(
        _database_path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _SequencedReleaseLease:
        assert mode is ProfileStoreLockMode.SHARED
        release_errors: list[BaseException | None]
        if not leases:
            release_errors = [RuntimeError(secret), RuntimeError(secret), None]
        else:
            release_errors = []
        lease = _SequencedReleaseLease(
            events,
            f"lease-{len(leases) + 1}",
            release_errors,
        )
        leases.append(lease)
        return lease

    def controlled_open(_database_path: Path) -> Any:
        nonlocal open_calls
        open_calls += 1
        events.append((f"store.open-{open_calls}", threading.get_ident()))
        if open_calls == 1:
            raise RuntimeError(secret)
        return connection

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(module, "open_profile_store", controlled_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    try:
        with pytest.raises(ProfileRepositoryError) as first_error:
            await repository.open()
        _assert_safe_error(first_error.value, "operation_failed", secret)
        assert repository._lease is leases[0]
        assert leases[0].acquired is True
        assert leases[0].release_calls == 1

        with pytest.raises(ProfileRepositoryError) as cleanup_error:
            await repository.open()
        _assert_safe_error(cleanup_error.value, "operation_failed", secret)
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert repository._lease is leases[0]
        assert leases[0].acquired is True
        assert leases[0].release_calls == 2
        assert len(leases) == 1
        assert open_calls == 1

        retried = await repository.open()

        assert retried == ProfileStoreResult(generation=3, value=None)
        assert leases[0].release_calls == 3
        assert leases[0].acquired is False
        assert repository._lease is leases[1]
        assert leases[1].acquired is True
        assert open_calls == 2
        phases = [phase for phase, _thread_id in events]
        assert max(
            index for index, phase in enumerate(phases) if phase == "lease-1.release"
        ) < phases.index("lease-2.acquire")
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_is_control", [True, False])
async def test_failed_open_close_retries_retained_lease_with_error_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_is_control: bool,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    primary_signal = _ControlFlow()
    cleanup_signal = _ControlFlow()
    primary_error: BaseException
    expected_signal: _ControlFlow
    if primary_is_control:
        primary_error = primary_signal
        expected_signal = primary_signal
    else:
        primary_error = RuntimeError("ordinary-open-failure")
        expected_signal = cleanup_signal
    lease = _SequencedReleaseLease(
        events,
        "residual-lease",
        [cleanup_signal, None],
    )

    def lease_factory(
        _database_path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _SequencedReleaseLease:
        assert mode is ProfileStoreLockMode.SHARED
        return lease

    def failing_open(_database_path: Path) -> Any:
        raise primary_error

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(module, "open_profile_store", failing_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    with pytest.raises(_ControlFlow) as caught:
        await repository.open()

    try:
        assert caught.value is expected_signal
        assert repository._lease is lease
        assert lease.acquired is True
        assert lease.release_calls == 1

        closed = await repository.close()

        assert closed == ProfileStoreResult(generation=2, value=None)
        assert lease.release_calls == 2
        assert lease.acquired is False
        assert repository._lease is None
    finally:
        if not repository.terminal:
            await repository.close()


@pytest.mark.asyncio
async def test_retry_cleans_residual_connection_before_acquiring_new_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    secret = str(tmp_path / "residual-connection-failure")
    events: list[tuple[str, int]] = []
    first_connection = _SequencedCloseConnection(
        events,
        "connection-1",
        [RuntimeError(secret), None],
    )
    second_connection = _SequencedCloseConnection(events, "connection-2", [])
    connections = [first_connection, second_connection]
    leases: list[_SequencedReleaseLease] = []
    open_calls = 0

    def lease_factory(
        _database_path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _SequencedReleaseLease:
        assert mode is ProfileStoreLockMode.SHARED
        lease = _SequencedReleaseLease(
            events,
            f"lease-{len(leases) + 1}",
            [],
        )
        leases.append(lease)
        return lease

    def controlled_open(_database_path: Path) -> Any:
        nonlocal open_calls
        connection = connections[open_calls]
        open_calls += 1
        events.append((f"store.open-{open_calls}", threading.get_ident()))
        return connection

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(module, "open_profile_store", controlled_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    with repository._state_lock:
        repository._state = ProfileRepositoryState.UNAVAILABLE

    try:
        with pytest.raises(ProfileRepositoryError) as cleanup_error:
            await repository.open()
        _assert_safe_error(cleanup_error.value, "operation_failed", secret)
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert repository._connection is first_connection
        assert first_connection.closed is False
        assert first_connection.close_calls == 1
        assert repository._lease is leases[0]
        assert leases[0].acquired is True
        assert leases[0].release_calls == 0
        assert len(leases) == 1
        assert open_calls == 1

        retried = await repository.open()

        assert retried == ProfileStoreResult(generation=3, value=None)
        assert first_connection.closed is True
        assert first_connection.close_calls == 2
        assert repository._connection is second_connection
        assert len(leases) == 2
        assert open_calls == 2
        phases = [phase for phase, _thread_id in events]
        assert phases.index("connection-1.close") < phases.index("lease-1.release")
        assert phases.index("lease-1.release") < phases.index("lease-2.acquire")
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_open_recreates_structured_error_without_hostile_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    secret = str(tmp_path / "secret-open-context.sqlite3")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    incoming = _tainted_repository_error("schema_corrupt", secret)
    _install_fake_store(monkeypatch, module, events, connection)

    def hostile_open(_database_path: Path) -> Any:
        raise incoming

    monkeypatch.setattr(module, "open_profile_store", hostile_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    assert caught.value is not incoming
    _assert_safe_error(caught.value, "schema_corrupt", secret)
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    await repository.close()


@pytest.mark.asyncio
async def test_open_rejects_missing_connection_and_releases_partial_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    leases = _install_fake_store(
        monkeypatch,
        module,
        events,
        connection,
    )

    def missing_connection(_database_path: Path) -> Any:
        events.append(("store.open", threading.get_ident()))
        return None

    monkeypatch.setattr(module, "open_profile_store", missing_connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    _assert_safe_error(caught.value, "operation_failed")
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    assert len(leases) == 1
    assert leases[0].acquired is False
    assert _phase_threads(events, "lease.acquire") == _phase_threads(
        events, "lease.release"
    )
    await repository.close()


@pytest.mark.asyncio
async def test_cancelled_open_settles_worker_and_publishes_consistent_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    open_started = threading.Event()
    allow_open = threading.Event()
    _install_fake_store(monkeypatch, module, events, connection)

    def blocked_open(_database_path: Path) -> Any:
        events.append(("store.open", threading.get_ident()))
        open_started.set()
        assert allow_open.wait(5.0)
        return connection

    monkeypatch.setattr(module, "open_profile_store", blocked_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    open_task = asyncio.create_task(repository.open())
    await _wait_thread_event(open_started)

    open_task.cancel()
    allow_open.set()
    with pytest.raises(asyncio.CancelledError):
        await open_task

    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 1
    assert repository._connection is connection
    result = await repository._submit_operation(lambda _connection: "usable")
    assert result == ProfileStoreResult(generation=1, value="usable")
    await repository.close()


@pytest.mark.asyncio
async def test_normal_submission_rejects_every_non_open_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(lambda _connection: "closed")
    _assert_safe_error(caught.value, "closed")

    await repository.open()
    with repository._state_lock:
        repository._state = ProfileRepositoryState.RESTORING
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(lambda _connection: "restoring")
    _assert_safe_error(caught.value, "restoring")

    with repository._state_lock:
        repository._state = ProfileRepositoryState.UNAVAILABLE
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(lambda _connection: "unavailable")
    _assert_safe_error(caught.value, "unavailable")

    with repository._state_lock:
        repository._state = ProfileRepositoryState.OPEN
    await repository.close()
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(lambda _connection: "terminal")
    _assert_safe_error(caught.value, "terminal")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("incoming_code", "expected_code"),
    [
        ("missing", "missing"),
        ("hostile-invalid-code", "operation_failed"),
    ],
)
async def test_operation_publication_recreates_structured_error_without_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    incoming_code: str,
    expected_code: str,
) -> None:
    module = _repository_module()
    secret = str(tmp_path / "secret-operation-context.sqlite3")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    incoming = _tainted_repository_error("missing", secret)
    incoming.code = incoming_code
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()

    def hostile_operation(_connection: Any) -> None:
        raise incoming

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(hostile_operation)

    assert caught.value is not incoming
    _assert_safe_error(caught.value, expected_code, secret)
    await repository.close()


@pytest.mark.asyncio
async def test_worker_preflight_and_publication_both_reject_stale_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    first_started = threading.Event()
    release_first = threading.Event()
    second_ran = False

    def first_operation(_connection: Any) -> str:
        first_started.set()
        assert release_first.wait(5.0)
        return "old-result"

    def second_operation(_connection: Any) -> str:
        nonlocal second_ran
        second_ran = True
        return "must-not-run"

    first = repository._admit_operation(first_operation)
    await _wait_thread_event(first_started)
    second = repository._admit_operation(second_operation)
    with repository._state_lock:
        repository._generation += 1
        repository._state = ProfileRepositoryState.RESTORING
    release_first.set()

    with pytest.raises(ProfileRepositoryError) as first_error:
        await repository._publish_operation(first)
    with pytest.raises(ProfileRepositoryError) as second_error:
        await repository._publish_operation(second)

    _assert_safe_error(first_error.value, "stale")
    _assert_safe_error(second_error.value, "stale")
    assert second_ran is False
    assert repository.generation == 2
    await repository.close()


@pytest.mark.asyncio
async def test_cancelled_operation_remains_registered_until_worker_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    operation_started = threading.Event()
    finish_operation = threading.Event()
    writes: list[str] = []

    def write_like_operation(_connection: Any) -> str:
        operation_started.set()
        assert finish_operation.wait(5.0)
        writes.append("committed")
        return "unpublished"

    operation_task = asyncio.create_task(
        repository._submit_operation(write_like_operation)
    )
    await _wait_thread_event(operation_started)
    operation_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation_task

    with repository._state_lock:
        assert len(repository._pending_futures) == 1

    close_admitted = asyncio.Event()
    original_finish_close = repository._finish_close

    async def observed_finish_close(*args: Any, **kwargs: Any) -> None:
        close_admitted.set()
        await original_finish_close(*args, **kwargs)

    monkeypatch.setattr(repository, "_finish_close", observed_finish_close)
    close_task = asyncio.create_task(repository.close())
    await close_admitted.wait()

    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.terminal is True
    assert close_task.done() is False
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(lambda _connection: "too-late")
    _assert_safe_error(caught.value, "terminal")

    finish_operation.set()
    await close_task

    assert writes == ["committed"]
    with repository._state_lock:
        assert not repository._pending_futures


@pytest.mark.asyncio
async def test_caller_cancellation_wins_when_close_cancels_same_queued_future(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    first_started = threading.Event()
    finish_first = threading.Event()
    second_ran = False

    def first_operation(_connection: Any) -> None:
        first_started.set()
        assert finish_first.wait(5.0)

    def second_operation(_connection: Any) -> None:
        nonlocal second_ran
        second_ran = True

    repository._admit_operation(first_operation)
    await _wait_thread_event(first_started)
    second = repository._admit_operation(second_operation)
    publication_started = asyncio.Event()

    async def publish_second() -> asyncio.CancelledError:
        publication_started.set()
        try:
            await repository._publish_operation(second)
        except asyncio.CancelledError as error:
            return error
        raise AssertionError("caller cancellation was not preserved")

    operation_task = asyncio.create_task(publish_second())
    await publication_started.wait()
    second.future.add_done_callback(
        lambda _future: operation_task.cancel("caller-cancellation")
    )
    close_task = asyncio.create_task(repository.close())

    try:
        cancellation = await operation_task
        assert type(cancellation) is asyncio.CancelledError
        assert cancellation.args == ("caller-cancellation",)
        assert operation_task.cancelling() == 1
        assert second_ran is False
    finally:
        finish_first.set()
        await close_task


@pytest.mark.asyncio
async def test_worker_future_cancellation_without_caller_cancellation_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    first_started = threading.Event()
    finish_first = threading.Event()
    second_ran = False

    def first_operation(_connection: Any) -> None:
        first_started.set()
        assert finish_first.wait(5.0)

    def second_operation(_connection: Any) -> None:
        nonlocal second_ran
        second_ran = True

    repository._admit_operation(first_operation)
    await _wait_thread_event(first_started)
    second = repository._admit_operation(second_operation)
    publication_started = asyncio.Event()

    async def publish_second() -> ProfileStoreResult[None]:
        publication_started.set()
        return cast(
            ProfileStoreResult[None],
            await repository._publish_operation(second),
        )

    operation_task = asyncio.create_task(publish_second())
    await publication_started.wait()
    assert second.future.cancel()

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await operation_task
        _assert_safe_error(caught.value, "stale")
        assert second_ran is False
    finally:
        finish_first.set()
        await repository.close()


@pytest.mark.asyncio
async def test_cancelled_operation_late_failure_is_retrieved_without_loop_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    operation_started = threading.Event()
    finish_operation = threading.Event()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled_contexts: list[dict[str, Any]] = []

    def capture_unhandled(
        _loop: asyncio.AbstractEventLoop,
        context: dict[str, Any],
    ) -> None:
        unhandled_contexts.append(context)

    loop.set_exception_handler(capture_unhandled)
    try:

        def late_failure(_connection: Any) -> None:
            operation_started.set()
            assert finish_operation.wait(5.0)
            raise RuntimeError("late-worker-failure")

        operation_task = asyncio.create_task(repository._submit_operation(late_failure))
        await _wait_thread_event(operation_started)
        operation_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation_task

        close_task = asyncio.create_task(repository.close())
        finish_operation.set()
        await close_task
        del operation_task
        gc.collect()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not [
        context
        for context in unhandled_contexts
        if context.get("message") == "Future exception was never retrieved"
    ]


@pytest.mark.asyncio
async def test_close_cancels_queued_work_before_draining_running_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    first_started = threading.Event()
    finish_first = threading.Event()
    second_ran = False

    def first_operation(_connection: Any) -> None:
        first_started.set()
        assert finish_first.wait(5.0)
        events.append(("first.finished", threading.get_ident()))

    def second_operation(_connection: Any) -> None:
        nonlocal second_ran
        second_ran = True

    first = repository._admit_operation(first_operation)
    await _wait_thread_event(first_started)
    second = repository._admit_operation(second_operation)
    close_admitted = asyncio.Event()
    original_finish_close = repository._finish_close

    async def observed_finish_close(*args: Any, **kwargs: Any) -> None:
        close_admitted.set()
        await original_finish_close(*args, **kwargs)

    monkeypatch.setattr(repository, "_finish_close", observed_finish_close)
    close_task = asyncio.create_task(repository.close())
    await close_admitted.wait()

    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.generation == 2
    assert repository.terminal is True
    assert second.future.cancelled()
    assert close_task.done() is False

    finish_first.set()
    await close_task

    assert first.future.done()
    assert second_ran is False
    phases = [phase for phase, _thread_id in events]
    assert phases.index("first.finished") < phases.index("connection.close")
    assert phases.index("connection.close") < phases.index("lease.release")


@pytest.mark.asyncio
async def test_close_failure_retains_real_shared_lease_and_connection_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    secret = str(tmp_path / "secret-close-error")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events, close_error=RuntimeError(secret))
    executors: list[_RecordingExecutor] = []
    leases: list[_ObservedProfileStoreLease] = []

    def executor_factory(max_workers: int) -> _RecordingExecutor:
        executor = _RecordingExecutor(max_workers, events)
        executors.append(executor)
        return executor

    def lease_factory(
        path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _ObservedProfileStoreLease:
        lease = _ObservedProfileStoreLease(
            path,
            mode,
            events,
            f"lease-{len(leases) + 1}",
        )
        leases.append(lease)
        return lease

    monkeypatch.setattr(module, "ThreadPoolExecutor", executor_factory)
    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(module, "open_profile_store", lambda _path: connection)
    repository = module.TTSProfileRepository(database_path)
    await repository.open()

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.close()

        _assert_safe_error(caught.value, "operation_failed", secret)
        assert repository.state is ProfileRepositoryState.CLOSED
        assert repository.terminal is True
        assert repository._connection is connection
        assert repository._lease is leases[0]
        assert connection.closed is False
        assert leases[0].acquired is True
        assert not _phase_threads(events, "lease-1.release")
        assert executors[0].shutdown_calls == 1
        await _assert_exclusive_lease_blocked(database_path)
        assert await repository.close() == ProfileStoreResult(
            generation=2,
            value=None,
        )
        assert executors[0].shutdown_calls == 1
    finally:
        connection.close_error = None
        if not connection.closed:
            connection.close()
        if leases and leases[0].acquired:
            await asyncio.to_thread(leases[0].release)
        repository._connection = None
        repository._lease = None

    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
async def test_close_preserves_connection_cleanup_control_and_retains_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    signal = _ControlFlow()
    connection = _RecordingConnection(events, close_error=signal)
    executors: list[_RecordingExecutor] = []

    def executor_factory(max_workers: int) -> _RecordingExecutor:
        executor = _RecordingExecutor(max_workers, events)
        executors.append(executor)
        return executor

    monkeypatch.setattr(module, "ThreadPoolExecutor", executor_factory)
    leases = _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()

    try:
        with pytest.raises(_ControlFlow) as caught:
            await repository.close()

        assert caught.value is signal
        assert repository._connection is connection
        assert repository._lease is leases[0]
        assert connection.closed is False
        assert leases[0].acquired is True
        assert not _phase_threads(events, "lease.release")
        assert executors[0].shutdown_calls == 1
        assert repository.state is ProfileRepositoryState.CLOSED
        assert repository.terminal is True
    finally:
        connection.close_error = None
        if not connection.closed:
            connection.close()
        if leases[0].acquired:
            leases[0].release()
        repository._connection = None
        repository._lease = None


@pytest.mark.asyncio
async def test_online_backup_serializes_with_write_and_publishes_valid_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    destination = tmp_path / "backup.sqlite3"
    created_at = datetime(2026, 7, 27, 10, 11, 12, 123456, tzinfo=UTC)
    repository = _repository_module().TTSProfileRepository(
        database_path,
        _clock=lambda: created_at,
    )
    await repository.open()
    await repository.create_profile(
        _draft("Before"),
        UUID("00000000-0000-4000-8000-000000000001"),
    )
    commit_started = threading.Event()
    finish_commit = threading.Event()
    real_commit = repository._commit_transaction

    def controlled_commit(connection: sqlite3.Connection) -> None:
        commit_started.set()
        assert finish_commit.wait(5.0)
        real_commit(connection)

    monkeypatch.setattr(repository, "_commit_transaction", controlled_commit)
    write = asyncio.create_task(
        repository.create_profile(
            _draft("After"),
            UUID("00000000-0000-4000-8000-000000000002"),
        )
    )
    await _wait_thread_event(commit_started)
    backup = asyncio.create_task(repository.backup_to(destination))
    await asyncio.sleep(0)

    try:
        assert backup.done() is False
        finish_commit.set()
        write_result, backup_result = await asyncio.gather(write, backup)

        validate_profile_candidate(destination)
        snapshot = open_profile_store(destination, must_exist=True)
        try:
            profile_count = snapshot.execute(
                "SELECT COUNT(*) FROM tts_generation_profiles"
            ).fetchone()[0]
        finally:
            snapshot.close()
        assert write_result.generation == 1
        assert backup_result == ProfileStoreResult(
            generation=1,
            value=ProfileBackupReceipt(
                created_at=created_at,
                byte_count=destination.stat().st_size,
            ),
        )
        assert profile_count == 2
    finally:
        finish_commit.set()
        await repository.close()


@pytest.mark.asyncio
async def test_online_backup_rejects_live_lock_and_sidecar_targets(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    repository = _repository(database_path)
    await repository.open()
    reserved = (
        database_path,
        database_path.with_name(f"{database_path.name}.lock"),
        *(
            database_path.with_name(f"{database_path.name}{suffix}")
            for suffix in (
                "-wal",
                "-shm",
                "-journal",
            )
        ),
    )

    try:
        before = {path: path.read_bytes() for path in reserved if path.is_file()}
        for destination in reserved:
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.backup_to(destination)
            _assert_safe_error(caught.value, "backup_failed", str(destination))
        assert {path: path.read_bytes() for path in before} == before
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_online_backup_rejects_symlink_and_hardlink_aliases_to_live_store(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    symlink_path = tmp_path / "symlink.sqlite3"
    hardlink_path = tmp_path / "hardlink.sqlite3"
    repository = _repository(database_path)
    await repository.open()
    symlink_path.symlink_to(database_path)
    os.link(database_path, hardlink_path)

    try:
        live_bytes = database_path.read_bytes()
        for destination in (symlink_path, hardlink_path):
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.backup_to(destination)
            _assert_safe_error(
                caught.value,
                "backup_failed",
                str(database_path),
                str(destination),
            )
        assert database_path.read_bytes() == live_bytes
        assert symlink_path.is_symlink()
        assert hardlink_path.stat().st_ino == database_path.stat().st_ino
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_online_backup_replace_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    destination = tmp_path / "existing-backup.sqlite3"
    destination.write_bytes(b"trusted-existing-destination")
    secret = str(tmp_path / "secret-backup-replace")
    repository = _repository(database_path)
    await repository.open()
    before = destination.read_bytes()

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError(secret)

    monkeypatch.setattr(module.os, "replace", fail_replace)
    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.backup_to(destination)

        _assert_safe_error(
            caught.value,
            "backup_failed",
            secret,
            str(database_path),
            str(destination),
        )
        assert destination.read_bytes() == before
        assert not tuple(tmp_path.glob(f".{destination.name}.*.backup"))
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_replaces_store_with_validated_candidate_and_safe_receipt(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    restored_at = datetime(2026, 7, 27, 12, 13, 14, 654321, tzinfo=UTC)
    await _create_profile_store(candidate, "Candidate one", "Candidate two")
    repository = _repository_module().TTSProfileRepository(
        database_path,
        _clock=lambda: restored_at,
    )
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )

    try:
        result = await repository.restore_from(candidate)

        assert result == ProfileStoreResult(
            generation=2,
            value=ProfileRestoreReceipt(
                restored_at=restored_at,
                profile_count=2,
                assignment_count=0,
            ),
        )
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == [
            "Candidate one",
            "Candidate two",
        ]
        recoveries = tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        assert len(recoveries) == 1
        validate_profile_candidate(recoveries[0])
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_online_backup_includes_committed_candidate_wal_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    candidate_repository = _repository(candidate)
    await candidate_repository.open()
    await candidate_repository.create_profile(
        _draft("Committed in WAL"),
        UUID("00000000-0000-4000-8000-000000000077"),
    )
    assert candidate.with_name(f"{candidate.name}-wal").is_file()
    repository = _repository(database_path)
    await repository.open()

    try:
        result = await repository.restore_from(candidate)

        assert result.value.profile_count == 1
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == [
            "Committed in WAL"
        ]
    finally:
        await repository.close()
        await candidate_repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timeout",
    [None, True, 0, -1, float("inf"), float("-inf"), float("nan"), "5"],
)
async def test_restore_rejects_invalid_timeout_before_lifecycle_or_file_mutation(
    tmp_path: Path,
    timeout: object,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(
                candidate,
                timeout_seconds=timeout,  # type: ignore[arg-type]
            )

        _assert_safe_error(caught.value, "restore_failed", str(candidate))
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 1
        assert {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        } == before
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_rejects_live_reserved_and_alias_candidates_before_admission(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    repository = _repository(database_path)
    await repository.open()
    hardlink = tmp_path / "hardlink-candidate.sqlite3"
    symlink = tmp_path / "symlink-candidate.sqlite3"
    os.link(database_path, hardlink)
    symlink.symlink_to(database_path)
    candidates = (
        database_path,
        database_path.with_name(f"{database_path.name}.lock"),
        *(
            database_path.with_name(f"{database_path.name}{suffix}")
            for suffix in (
                "-wal",
                "-shm",
                "-journal",
            )
        ),
        hardlink,
        symlink,
    )

    try:
        for candidate in candidates:
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.restore_from(candidate)
            _assert_safe_error(
                caught.value,
                "restore_failed",
                str(candidate),
                str(database_path),
            )
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 1
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_corrupt_restore_candidate_preserves_live_store_and_reopens_advanced(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate-secret.sqlite3"
    candidate.write_bytes(b"not a sqlite database")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    await repository._submit_operation(
        lambda connection: connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchall()
    )
    before_bytes = database_path.read_bytes()

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(
            caught.value,
            "schema_corrupt",
            str(candidate),
            str(database_path),
        )
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert database_path.read_bytes() == before_bytes
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
        restored = await repository.list_profiles()
        assert [profile.display_name for profile in restored.value.profiles] == [
            "Original"
        ]
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_quiescence_timeout_advances_without_file_or_lease_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    operation_started = threading.Event()
    release_operation = threading.Event()

    def blocked_operation(_connection: sqlite3.Connection) -> str:
        operation_started.set()
        assert release_operation.wait(5.0)
        return "old-result"

    old = repository._admit_operation(blocked_operation)
    await _wait_thread_event(operation_started)
    connection = repository._connection
    lease = repository._lease
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate, timeout_seconds=0.01)

        _assert_safe_error(caught.value, "restore_failed")
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert repository._connection is connection
        assert repository._lease is lease
        assert {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        } == before
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
    finally:
        release_operation.set()

    with pytest.raises(ProfileRepositoryError) as stale:
        await repository._publish_operation(old)
    _assert_safe_error(stale.value, "stale")
    assert (await repository.list_profiles()).generation == 2
    await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("variant", "expected_code"),
    [
        ("partial", "schema_corrupt"),
        ("unsupported", "schema_unsupported"),
        ("domain", "corrupt_data"),
        ("foreign_key", "schema_corrupt"),
    ],
)
async def test_invalid_restore_candidates_preserve_original_and_rebind_open(
    tmp_path: Path,
    variant: str,
    expected_code: str,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / f"{variant}-candidate.sqlite3"
    if variant == "partial":
        partial = sqlite3.connect(candidate)
        partial.execute("CREATE TABLE unexpected(value TEXT)")
        partial.execute("PRAGMA user_version = 1")
        partial.close()
    else:
        await _create_profile_store(candidate, "Candidate")
        hostile = sqlite3.connect(candidate)
        if variant == "unsupported":
            hostile.execute("PRAGMA user_version = 2")
        elif variant == "domain":
            hostile.execute("UPDATE tts_generation_profiles SET revision = 0")
        else:
            timestamp = "2026-07-27T12:00:00.000000Z"
            hostile.execute(
                """
                INSERT INTO character_tts_assignments (
                    source, authority_id, character_id, profile_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "local",
                    "authority",
                    "character",
                    "00000000-0000-4000-8000-999999999999",
                    timestamp,
                    timestamp,
                ),
            )
        hostile.commit()
        hostile.close()

    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    await repository._submit_operation(
        lambda connection: connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchall()
    )
    before_bytes = database_path.read_bytes()

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(
            caught.value,
            expected_code,
            str(candidate),
            str(database_path),
        )
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert database_path.read_bytes() == before_bytes
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_cancels_queued_old_write_and_suppresses_running_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    commit_started = threading.Event()
    release_commit = threading.Event()
    second_body_ran = False
    real_commit = repository._commit_transaction
    real_create = repository._worker_create_profile
    commit_calls = 0

    def controlled_commit(connection: sqlite3.Connection) -> None:
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 1:
            commit_started.set()
            assert release_commit.wait(5.0)
        real_commit(connection)

    def observed_create(
        connection: sqlite3.Connection,
        draft: TTSProfileDraft,
        profile_id: UUID | None,
    ) -> Any:
        nonlocal second_body_ran
        if draft.display_name == "Queued old":
            second_body_ran = True
        return real_create(connection, draft, profile_id)

    monkeypatch.setattr(repository, "_commit_transaction", controlled_commit)
    monkeypatch.setattr(repository, "_worker_create_profile", observed_create)
    running = asyncio.create_task(
        repository.create_profile(
            _draft("Running old"),
            UUID("00000000-0000-4000-8000-000000000010"),
        )
    )
    await _wait_thread_event(commit_started)
    queued = asyncio.create_task(
        repository.create_profile(
            _draft("Queued old"),
            UUID("00000000-0000-4000-8000-000000000011"),
        )
    )
    await asyncio.sleep(0)
    restore = asyncio.create_task(repository.restore_from(candidate))
    for _ in range(100):
        if repository.state is ProfileRepositoryState.RESTORING:
            break
        await asyncio.sleep(0)

    assert repository.state is ProfileRepositoryState.RESTORING
    assert repository.generation == 2
    assert restore.done() is False
    release_commit.set()
    running_outcome, queued_outcome, restore_outcome = await asyncio.gather(
        running,
        queued,
        restore,
        return_exceptions=True,
    )

    assert isinstance(running_outcome, ProfileRepositoryError)
    _assert_safe_error(running_outcome, "stale")
    assert isinstance(queued_outcome, ProfileRepositoryError)
    _assert_safe_error(queued_outcome, "stale")
    assert second_body_ran is False
    assert restore_outcome.generation == 2
    assert repository.state is ProfileRepositoryState.OPEN
    page = await repository.list_profiles()
    assert [profile.display_name for profile in page.value.profiles] == ["Candidate"]
    await repository.close()


@pytest.mark.asyncio
async def test_recovery_backup_failure_cleans_stage_and_rebinds_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / "secret-recovery-failure")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )

    def fail_recovery(_restored_at: datetime) -> Path:
        raise RuntimeError(secret)

    monkeypatch.setattr(repository, "_worker_create_recovery_backup", fail_recovery)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
    assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
    page = await repository.list_profiles()
    assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    await repository.close()


@pytest.mark.asyncio
async def test_replace_failure_retains_recovery_cleans_stage_and_rebinds_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / "secret-replace-failure")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    real_replace = os.replace

    def fail_live_replace(source: object, destination: object) -> None:
        if Path(destination) == database_path.resolve():
            raise OSError(secret)
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_live_replace)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
    assert repository.state is ProfileRepositoryState.OPEN
    assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
    recoveries = tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
    assert len(recoveries) == 1
    validate_profile_candidate(recoveries[0])
    page = await repository.list_profiles()
    assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    await repository.close()


@pytest.mark.asyncio
async def test_post_replace_shared_reacquire_failure_is_unavailable_with_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    real_lease_type = module.ProfileStoreLease
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )

    def lease_factory(
        path: Path,
        mode: ProfileStoreLockMode,
        **kwargs: object,
    ) -> Any:
        if mode is ProfileStoreLockMode.SHARED:
            return _RecordingLease(
                [],
                acquire_error=ProfileRepositoryError("lock_timeout"),
            )
        return real_lease_type(path, mode, **kwargs)

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", str(database_path))
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is None
    assert repository._lease is None
    assert len(tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))) == 1
    authoritative = open_profile_store(database_path, must_exist=True)
    try:
        names = [
            row[0]
            for row in authoritative.execute(
                "SELECT display_name FROM tts_generation_profiles"
            )
        ]
    finally:
        authoritative.close()
    assert names == ["Candidate"]
    await repository.close()


@pytest.mark.asyncio
async def test_post_replace_long_lived_reopen_failure_is_unavailable_without_blank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / "secret-reopen-failure")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    real_open = module.open_profile_store
    strict_calls = 0

    def injected_open(path: Path, *, must_exist: bool = False) -> sqlite3.Connection:
        nonlocal strict_calls
        if must_exist:
            strict_calls += 1
            if strict_calls == 3:
                raise RuntimeError(secret)
        return real_open(path, must_exist=must_exist)

    monkeypatch.setattr(module, "open_profile_store", injected_open)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
    assert strict_calls == 3
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is None
    assert repository._lease is None
    assert database_path.stat().st_size > 0
    assert len(tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))) == 1
    authoritative = real_open(database_path, must_exist=True)
    try:
        assert [
            row[0]
            for row in authoritative.execute(
                "SELECT display_name FROM tts_generation_profiles"
            )
        ] == ["Candidate"]
    finally:
        authoritative.close()
    await repository.close()


@pytest.mark.asyncio
async def test_restore_handoff_opens_newer_valid_store_that_wins_before_shared_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    real_lease_type = module.ProfileStoreLease
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    newer = tmp_path / "newer.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    await _create_profile_store(newer, "Newer one", "Newer two")
    repository = _repository(database_path)
    await repository.open()
    replaced_during_handoff = False

    class HandoffLease(real_lease_type):
        def release(self) -> None:
            nonlocal replaced_during_handoff
            super().release()
            if (
                self.mode is ProfileStoreLockMode.EXCLUSIVE
                and not replaced_during_handoff
            ):
                os.replace(newer, database_path)
                replaced_during_handoff = True

    monkeypatch.setattr(module, "ProfileStoreLease", HandoffLease)

    try:
        result = await repository.restore_from(candidate)

        assert replaced_during_handoff is True
        assert result.value.profile_count == 2
        assert result.value.assignment_count == 0
        assert repository.state is ProfileRepositoryState.OPEN
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == [
            "Newer one",
            "Newer two",
        ]
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "expected_mode"),
    [
        ("recovery_source", ProfileStoreLockMode.EXCLUSIVE),
        ("scoped_replacement", ProfileStoreLockMode.EXCLUSIVE),
        ("pre_rebind", ProfileStoreLockMode.SHARED),
        ("post_rebind", ProfileStoreLockMode.SHARED),
    ],
)
async def test_restore_live_connection_close_failure_retains_protecting_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    expected_mode: ProfileStoreLockMode,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / f"secret-{boundary}-close")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    real_open = module.open_profile_store
    real_counts = repository._worker_store_counts
    strict_calls = 0
    proxies: list[_CloseFailingSQLiteProxy] = []

    def injected_open(
        path: Path,
        *,
        must_exist: bool = False,
    ) -> sqlite3.Connection:
        nonlocal strict_calls
        connection = real_open(path, must_exist=must_exist)
        if must_exist:
            strict_calls += 1
            target_call = {
                "recovery_source": 1,
                "scoped_replacement": 2,
                "pre_rebind": 1,
                "post_rebind": 3,
            }[boundary]
            if strict_calls == target_call:
                proxy = _CloseFailingSQLiteProxy(connection, secret)
                proxies.append(proxy)
                return cast(sqlite3.Connection, proxy)
        return connection

    def injected_counts(connection: sqlite3.Connection) -> tuple[int, int]:
        if boundary in {"pre_rebind", "post_rebind"} and any(
            connection is proxy for proxy in proxies
        ):
            raise RuntimeError(secret)
        return real_counts(connection)

    if boundary == "pre_rebind":
        monkeypatch.setattr(
            repository,
            "_worker_create_recovery_backup",
            lambda _restored_at: (_ for _ in ()).throw(RuntimeError(secret)),
        )
    monkeypatch.setattr(module, "open_profile_store", injected_open)
    monkeypatch.setattr(repository, "_worker_store_counts", injected_counts)

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
    assert len(proxies) == 1
    proxy = proxies[0]
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is proxy
    assert repository._lease is not None
    assert repository._lease.mode is expected_mode
    assert repository._lease.acquired is True
    await _assert_exclusive_lease_blocked(database_path)

    proxy.fail_close = False
    await repository.close()
    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "expected_mode"),
    [
        ("exclusive_release", ProfileStoreLockMode.EXCLUSIVE),
        ("rebound_release", ProfileStoreLockMode.SHARED),
    ],
)
async def test_restore_release_failure_retains_residual_lease_for_later_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    expected_mode: ProfileStoreLockMode,
) -> None:
    module = _repository_module()
    real_lease_type = module.ProfileStoreLease
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / f"secret-{boundary}")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    real_open = module.open_profile_store
    strict_calls = 0
    retained: list[Any] = []

    class ControlledReleaseLease(real_lease_type):
        fail_release = True

        def release(self) -> None:
            should_fail = (
                boundary == "exclusive_release"
                and self.mode is ProfileStoreLockMode.EXCLUSIVE
            ) or (
                boundary == "rebound_release"
                and self.mode is ProfileStoreLockMode.SHARED
            )
            if should_fail and self.fail_release:
                retained.append(self)
                raise RuntimeError(secret)
            super().release()

    def injected_open(
        path: Path,
        *,
        must_exist: bool = False,
    ) -> sqlite3.Connection:
        nonlocal strict_calls
        if must_exist:
            strict_calls += 1
            if boundary == "rebound_release" and strict_calls == 3:
                raise RuntimeError(secret)
        return real_open(path, must_exist=must_exist)

    monkeypatch.setattr(module, "ProfileStoreLease", ControlledReleaseLease)
    monkeypatch.setattr(module, "open_profile_store", injected_open)

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
    assert retained
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is None
    assert repository._lease is retained[-1]
    assert repository._lease.mode is expected_mode
    assert repository._lease.acquired is True
    await _assert_exclusive_lease_blocked(database_path)

    repository._lease.fail_release = False
    await repository.close()
    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["clock", "checkpoint", "connection_close"])
async def test_restore_preclose_failure_keeps_usable_original_shared_pair_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / f"secret-{boundary}")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    original_connection = repository._connection
    original_lease = repository._lease
    proxy: _CloseFailingSQLiteProxy | None = None

    if boundary == "clock":
        monkeypatch.setattr(
            repository,
            "_clock",
            lambda: (_ for _ in ()).throw(RuntimeError(secret)),
        )
    elif boundary == "connection_close":
        assert original_connection is not None
        proxy = _CloseFailingSQLiteProxy(original_connection, secret)
        repository._connection = cast(sqlite3.Connection, proxy)
        original_connection = repository._connection
    else:
        assert original_connection is not None
        delegate = original_connection

        class CheckpointFailingProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(delegate, name)

            def execute(
                self,
                sql: str,
                parameters: object = (),
            ) -> Any:
                if sql == "PRAGMA wal_checkpoint(TRUNCATE)":
                    raise RuntimeError(secret)
                return delegate.execute(sql, parameters)

        repository._connection = cast(sqlite3.Connection, CheckpointFailingProxy())
        original_connection = repository._connection

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert repository._connection is original_connection
        assert repository._lease is original_lease
        assert repository._lease is not None
        assert repository._lease.mode is ProfileStoreLockMode.SHARED
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    finally:
        if proxy is not None:
            proxy.fail_close = False
        await repository.close()


@pytest.mark.asyncio
async def test_restore_fsyncs_recovery_directory_entry_before_live_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    events: list[tuple[str, Path]] = []
    real_fsync_file = module._fsync_file
    real_fsync_directory = module._fsync_directory
    real_replace = module.os.replace

    def observed_fsync_file(path: Path) -> None:
        events.append(("file", path))
        real_fsync_file(path)

    def observed_fsync_directory(path: Path) -> None:
        events.append(("directory", path))
        real_fsync_directory(path)

    def observed_replace(source: object, destination: object) -> None:
        events.append(("replace", Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(module, "_fsync_file", observed_fsync_file)
    monkeypatch.setattr(module, "_fsync_directory", observed_fsync_directory)
    monkeypatch.setattr(module.os, "replace", observed_replace)

    try:
        await repository.restore_from(candidate)

        recovery_file_index = next(
            index
            for index, (kind, path) in enumerate(events)
            if kind == "file" and path.name.endswith(".recovery.sqlite3")
        )
        live_replace_index = next(
            index
            for index, (kind, path) in enumerate(events)
            if kind == "replace" and path == database_path
        )
        assert events[recovery_file_index + 1] == ("directory", tmp_path)
        assert recovery_file_index < live_replace_index
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "clock",
    [
        lambda: float("inf"),
        lambda: (_ for _ in ()).throw(RuntimeError("hostile-monotonic")),
    ],
)
async def test_restore_hostile_monotonic_fails_before_lifecycle_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: Callable[[], float],
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    monkeypatch.setattr(module, "_monotonic", clock)

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(caught.value, "restore_failed", str(candidate))
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 1
        assert {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        } == before
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_cancelled_restore_settles_transition_before_propagating_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    worker_started = threading.Event()
    release_worker = threading.Event()
    real_restore = repository._worker_restore

    def blocked_restore(*args: object, **kwargs: object) -> ProfileRestoreReceipt:
        worker_started.set()
        assert release_worker.wait(5.0)
        return real_restore(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(repository, "_worker_restore", blocked_restore)
    restore_task = asyncio.create_task(repository.restore_from(candidate))
    await _wait_thread_event(worker_started)
    restore_task.cancel("caller-cancelled")
    await asyncio.sleep(0)
    assert restore_task.done() is False

    release_worker.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await restore_task

    assert caught.value.args == ("caller-cancelled",)
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    page = await repository.list_profiles()
    assert [profile.display_name for profile in page.value.profiles] == ["Candidate"]
    await repository.close()


@pytest.mark.asyncio
async def test_backup_and_restore_reject_foreign_loop_without_lifecycle_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    destination = tmp_path / "backup.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()

    try:
        for operation in (
            lambda: repository.backup_to(destination),
            lambda: repository.restore_from(candidate),
        ):
            result, error = await _run_in_new_loop_thread(operation)
            assert result is None
            assert isinstance(error, ProfileRepositoryError)
            _assert_safe_error(error, "invalid_state")
            assert repository.state is ProfileRepositoryState.OPEN
            assert repository.generation == 1
        assert destination.exists() is False
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_runs_full_integrity_check_before_publishing_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    statements: list[str] = []
    real_connect = module.sqlite3.connect

    def traced_connect(
        database: object,
        *args: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        connection = real_connect(database, *args, **kwargs)  # type: ignore[arg-type]
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(module.sqlite3, "connect", traced_connect)
    try:
        await repository.restore_from(candidate)

        assert "PRAGMA integrity_check" in statements
        assert repository.state is ProfileRepositoryState.OPEN
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_full_integrity_failure_preserves_original_before_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )

    def fail_integrity(_connection: sqlite3.Connection) -> None:
        raise ProfileRepositoryError("schema_corrupt")

    monkeypatch.setattr(repository, "_worker_require_full_integrity", fail_integrity)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(
        caught.value, "schema_corrupt", str(candidate), str(database_path)
    )
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
    page = await repository.list_profiles()
    assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "checkpoint_row",
    [
        (0, True, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0),
        ("0", 0, 0),
    ],
)
async def test_restore_requires_exact_completed_truncate_checkpoint_evidence(
    tmp_path: Path,
    checkpoint_row: tuple[object, ...],
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    original_connection = repository._connection
    original_lease = repository._lease
    assert original_connection is not None

    class CheckpointCursor:
        def fetchone(self) -> tuple[object, ...]:
            return checkpoint_row

    class CheckpointProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_connection, name)

        def execute(self, sql: str, parameters: object = ()) -> Any:
            if sql == "PRAGMA wal_checkpoint(TRUNCATE)":
                return CheckpointCursor()
            return original_connection.execute(sql, parameters)

    proxy = cast(sqlite3.Connection, CheckpointProxy())
    repository._connection = proxy
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", str(database_path))
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    assert repository._connection is proxy
    assert repository._lease is original_lease
    assert (await repository.list_profiles()).value.total == 0
    await repository.close()


@pytest.mark.asyncio
async def test_restore_refuses_live_rollback_journal_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    journal = database_path.with_name(f"{database_path.name}-journal")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    real_remove_sidecars = repository._worker_remove_live_sidecars
    real_rebind = repository._worker_rebind_current_store
    journal_seen_before_rebind = False

    def inject_journal() -> None:
        journal.touch()
        real_remove_sidecars()

    def observed_rebind() -> None:
        nonlocal journal_seen_before_rebind
        journal_seen_before_rebind = journal.exists()
        real_rebind()

    monkeypatch.setattr(repository, "_worker_remove_live_sidecars", inject_journal)
    monkeypatch.setattr(repository, "_worker_rebind_current_store", observed_rebind)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", str(database_path))
    assert journal_seen_before_rebind is True
    assert repository.state is ProfileRepositoryState.OPEN
    page = await repository.list_profiles()
    assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    assert len(tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))) == 1
    await repository.close()


@pytest.mark.asyncio
async def test_stage_cleanup_failure_does_not_block_original_shared_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / "secret-stage-cleanup")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    real_replace = module.os.replace
    real_remove_temporary = repository._worker_remove_temporary_store
    retained_stage: Path | None = None

    def fail_replace(source: object, destination: object) -> None:
        if Path(destination) == database_path.resolve():
            raise OSError(secret)
        real_replace(source, destination)

    def fail_stage_cleanup(path: Path) -> list[BaseException]:
        nonlocal retained_stage
        if path.name.endswith(".restore-stage.sqlite3"):
            retained_stage = path
            return [RuntimeError(secret)]
        return real_remove_temporary(path)

    monkeypatch.setattr(module.os, "replace", fail_replace)
    monkeypatch.setattr(
        repository,
        "_worker_remove_temporary_store",
        fail_stage_cleanup,
    )
    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository._connection is not None
        assert repository._lease is not None
        assert repository._lease.mode is ProfileStoreLockMode.SHARED
        assert retained_stage is not None and retained_stage.exists()
        assert len(tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))) == 1
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    finally:
        if retained_stage is not None:
            for error in real_remove_temporary(retained_stage):
                if isinstance(error, Exception):
                    raise error
        await repository.close()
