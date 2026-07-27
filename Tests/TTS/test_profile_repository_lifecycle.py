"""Lifecycle tests for the serialized TTS profile repository."""

from __future__ import annotations

import asyncio
import importlib
import sqlite3
import threading
import traceback
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_schema import open_profile_store
from tldw_chatbook.TTS.profile_store_lock import (
    ProfileStoreLease,
    ProfileStoreLockMode,
)
from tldw_chatbook.TTS.profile_types import (
    ProfileRepositoryState,
    ProfileStoreResult,
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


async def _wait_thread_event(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 5.0)


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
async def test_close_maps_cleanup_error_and_still_releases_and_shuts_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    secret = str(tmp_path / "secret-close-error")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events, close_error=RuntimeError(secret))
    executors: list[_RecordingExecutor] = []

    def executor_factory(max_workers: int) -> _RecordingExecutor:
        executor = _RecordingExecutor(max_workers, events)
        executors.append(executor)
        return executor

    monkeypatch.setattr(module, "ThreadPoolExecutor", executor_factory)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.close()

    _assert_safe_error(caught.value, "operation_failed", secret)
    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.terminal is True
    assert len(_phase_threads(events, "lease.release")) == 1
    assert executors[0].shutdown_calls == 1
    assert await repository.close() == ProfileStoreResult(generation=2, value=None)
    assert executors[0].shutdown_calls == 1


@pytest.mark.asyncio
async def test_close_preserves_control_flow_after_remaining_cleanup(
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
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()

    with pytest.raises(_ControlFlow) as caught:
        await repository.close()

    assert caught.value is signal
    assert len(_phase_threads(events, "lease.release")) == 1
    assert executors[0].shutdown_calls == 1
    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.terminal is True
