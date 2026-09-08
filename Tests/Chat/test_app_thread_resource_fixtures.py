"""Real thread/SQLite controls for the opt-in app resource fixtures."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from Tests import app_thread_resource_fixtures as threads
from Tests import console_resource_fixtures as resources


class ThreadDatabase:
    def __init__(self, *, close_error=None):
        self._thread_local = threading.local()
        self.close_error = close_error
        self.closed_on = []

    def connection(self):
        if getattr(self._thread_local, "conn", None) is None:
            self._thread_local.conn = sqlite3.connect(":memory:")
        return self._thread_local.conn

    def close(self):
        self._thread_local.conn.close()
        self._thread_local.conn = None
        self.closed_on.append(threading.get_ident())
        if self.close_error is not None:
            raise self.close_error


def owned_executor(databases):
    assert hasattr(threads, "OwnedAppExecutor"), "missing exact app executor owner"
    return threads.OwnedAppExecutor(databases, max_workers=1)


@pytest.mark.parametrize("operation_fails", [False, True])
async def test_executor_closes_new_owned_handles_on_creator_thread(operation_fails):
    owned, foreign = ThreadDatabase(), ThreadDatabase()
    failure = ValueError("operation failure")
    with owned_executor([owned]) as executor:

        def operation():
            owned.connection()
            foreign.connection()
            if operation_fails:
                raise failure
            return threading.get_ident()

        future = executor.submit(operation)
        if operation_fails:
            with pytest.raises(ValueError) as caught:
                future.result(timeout=2)
            assert caught.value is failure
        else:
            assert future.result(timeout=2) == owned.closed_on[0]
        assert await executor.drain() == []
        assert len(owned.closed_on) == 1
        assert foreign.closed_on == []
        assert executor.submit(
            lambda: foreign.connection().execute("SELECT 1").fetchone()
        ).result(timeout=2) == (1,)
        executor.submit(foreign.close).result(timeout=2)


async def test_executor_preserves_a_borrowed_connection_and_transaction():
    database = ThreadDatabase()
    with owned_executor([database]) as executor:
        # Establish a caller-owned transaction outside the adapter's operation.
        def begin():
            connection = database.connection()
            connection.execute("CREATE TABLE item (value)")
            connection.execute("INSERT INTO item VALUES (1)")
            return connection

        borrowed = ThreadPoolExecutor.submit(executor, begin).result(timeout=2)
        assert executor.submit(lambda: database.connection() is borrowed).result(
            timeout=2
        )
        assert await executor.drain() == []
        assert database.closed_on == []
        assert ThreadPoolExecutor.submit(
            executor, lambda: borrowed.in_transaction
        ).result(timeout=2)
        ThreadPoolExecutor.submit(executor, database.close).result(timeout=2)


async def test_executor_closes_a_new_handle_replacing_an_invalid_borrowed_one():
    database = ThreadDatabase()
    with owned_executor([database]) as executor:

        def invalidate():
            database.connection().close()

        ThreadPoolExecutor.submit(executor, invalidate).result(timeout=2)

        def revive():
            # The held-connection stores transparently replace an invalid handle.
            database._thread_local.conn = sqlite3.connect(":memory:")

        try:
            executor.submit(revive).result(timeout=2)
            assert await executor.drain() == []
            assert len(database.closed_on) == 1
        finally:

            def cleanup_if_needed():
                if getattr(database._thread_local, "conn", None) is not None:
                    database.close()

            ThreadPoolExecutor.submit(executor, cleanup_if_needed).result(timeout=2)


async def test_connection_cleanup_failure_does_not_replace_operation_error():
    cleanup_error, operation_error = (
        RuntimeError("close failed"),
        ValueError("work failed"),
    )
    first = ThreadDatabase(close_error=cleanup_error)
    second = ThreadDatabase()
    with owned_executor([first, second]) as executor:

        def operation():
            first.connection()
            second.connection()
            raise operation_error

        with pytest.raises(ValueError) as caught:
            executor.submit(operation).result(timeout=2)
        assert caught.value is operation_error
        assert await executor.drain() == [cleanup_error]
        assert len(first.closed_on) == len(second.closed_on) == 1


async def test_cancelled_asyncio_wrapper_does_not_complete_resource_drain():
    database = ThreadDatabase()
    started, release = threading.Event(), threading.Event()
    with owned_executor([database]) as executor:

        def operation():
            database.connection()
            started.set()
            assert release.wait(2)

        wrapper = asyncio.get_running_loop().run_in_executor(executor, operation)
        try:
            while not started.is_set():
                await asyncio.sleep(0)
            wrapper.cancel()
            with pytest.raises(asyncio.CancelledError):
                await wrapper
            drain = asyncio.create_task(executor.drain())
            await asyncio.sleep(0)
            assert not drain.done()
            assert not database.closed_on
            release.set()
            assert await drain == []
            assert len(database.closed_on) == 1
        finally:
            release.set()


async def test_drain_timeout_preserves_running_future_for_later_cleanup():
    database = ThreadDatabase()
    started, release = threading.Event(), threading.Event()
    with owned_executor([database]) as executor:

        def operation():
            database.connection()
            started.set()
            assert release.wait(2)

        future = executor.submit(operation)
        try:
            while not started.is_set():
                await asyncio.sleep(0)
            with pytest.raises(TimeoutError):
                await executor.drain(timeout_seconds=0.01)
            assert not future.cancelled()
            assert not database.closed_on
        finally:
            release.set()
            await executor.drain()
        assert len(database.closed_on) == 1


@pytest.mark.parametrize("drain_fails", [False, True])
async def test_resource_stack_requires_worker_completion_before_database_cleanup(
    monkeypatch, tmp_path, drain_fails
):
    lifecycle = resources.close_owned_console_resources.__wrapped__(
        monkeypatch, tmp_path, None
    )
    stack = await anext(lifecycle)
    assert hasattr(stack, "before_close"), "missing worker drain prerequisite"
    events = []
    drain_error = TimeoutError("still running")

    async def drain():
        events.append("drain")
        if drain_fails:
            raise drain_error
        return [ValueError("worker close failed")]

    stack.before_close.append(drain)
    stack.callback(events.append, "database close")
    if drain_fails:
        with pytest.raises(TimeoutError) as caught:
            await anext(lifecycle)
        assert caught.value is drain_error
        assert events == ["drain"]
        stack.close()
    else:
        with pytest.raises(ExceptionGroup) as caught:
            await anext(lifecycle)
        assert [type(error) for error in caught.value.exceptions] == [ValueError]
        assert events == ["drain", "database close"]
    await lifecycle.aclose()


def test_initializer_capture_closes_only_returned_apps_exact_handles(
    monkeypatch, tmp_path
):
    assert hasattr(threads, "capture_app_initialization"), (
        "missing exact initializer owner"
    )
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

    foreign = PromptsDatabase(tmp_path / "foreign.db", client_id="test")
    foreign_connection = foreign.get_connection()
    captured = []

    def build():
        # An unrelated pre-existing instance is touched during construction.
        foreign.get_connection()

        def initialize():
            prompts = PromptsDatabase(tmp_path / "prompts.db", client_id="test")
            media = MediaDatabase(tmp_path / "media.db", client_id="test")
            captured.extend((prompts.get_connection(), media.get_connection()))
            return SimpleNamespace(prompts_db=prompts, media_db=media)

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(initialize).result(timeout=5)

    original = PromptsDatabase._get_thread_connection
    try:
        with ExitStack() as cleanup:
            app = threads.capture_app_initialization(build, monkeypatch, cleanup)
            assert app.prompts_db is not foreign
            assert all(
                connection.execute("SELECT 1").fetchone()[0] == 1
                for connection in captured
            )
        for connection in captured:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                connection.execute("SELECT 1")
        assert foreign_connection.execute("SELECT 1").fetchone()[0] == 1
        assert PromptsDatabase._get_thread_connection is original
    finally:
        foreign.close()
