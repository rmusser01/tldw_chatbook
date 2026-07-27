"""Serialized lifecycle owner for the local TTS generation-profile store."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, cast

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


_T = TypeVar("_T")
_PATH_TYPE = type(Path())


@dataclass(frozen=True, slots=True)
class _OperationAdmission(Generic[_T]):
    """One generation-bound worker submission awaiting publication."""

    generation: int
    future: Future[_T]


def _repository_error(code: str) -> ProfileRepositoryError:
    return ProfileRepositoryError(code)


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

    def __init__(self, database_path: Path) -> None:
        """Create an initially closed, reopenable repository.

        Args:
            database_path: Exact local profile-store path.

        Raises:
            ProfileRepositoryError: If ``database_path`` is not an exact
                platform ``Path`` value.
        """

        if type(database_path) is not _PATH_TYPE:
            raise _repository_error("operation_failed")

        self._database_path = database_path
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
