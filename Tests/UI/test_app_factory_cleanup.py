"""Real factory-owned handle retirement without releasing app references."""

import asyncio
import sqlite3

import pytest

pytestmark = pytest.mark.bootstrap_profile


def test_factory_cleanup_closes_constructor_database_handles():
    from Tests.UI import app_factory

    apps = [app_factory._build_test_app() for _ in range(3)]
    locks = [app._instance_lock_status.handle for app in apps]
    assert all(lock is not None and not lock.closed for lock in locks)
    owners = [
        owner
        for app in apps
        for owner in (
            app.local_library_collections_db,
            app.local_workspace_db,
            app.subscriptions_db,
            app.evaluation_orchestrator.db,
        )
    ]
    connections = [
        connection
        for owner in owners
        for connection in owner._maintenance_participant.connections
    ]
    assert len(connections) == 12
    for connection in connections:
        assert connection.execute("SELECT 1").fetchone()[0] == 1

    app_factory.drain_active_service_patches()
    assert app_factory.drain_created_dirs() == 3

    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
    assert all(not owner._maintenance_participant.connections for owner in owners)
    assert len(apps) == 3
    assert all(lock.closed for lock in locks)


def test_factory_cleanup_keeps_replacement_database_caller_owned():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from Tests.UI import app_factory

    app = app_factory._build_test_app()
    original = app.local_workspace_db
    connection = next(iter(original._maintenance_participant.connections))
    replacement_close = Mock()
    app.local_workspace_db = SimpleNamespace(close=replacement_close)

    assert app_factory.drain_created_dirs() == 1

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    replacement_close.assert_not_called()
    assert app_factory.drain_created_dirs() == 0


def test_factory_cleanup_propagates_database_close_failure():
    from unittest.mock import patch

    from Tests.UI import app_factory

    app = app_factory._build_test_app()
    database = app.evaluation_orchestrator.db
    directory = app_factory._created_dirs[-1]
    error = RuntimeError("injected constructor-owned database close failure")
    with (
        patch.object(database, "close", side_effect=error),
        pytest.raises(RuntimeError) as raised,
    ):
        app_factory.drain_created_dirs()
    assert raised.value is error
    assert directory.exists()

    assert app_factory.drain_created_dirs() == 1
    assert not directory.exists()


def test_unrelated_sync_cleanup_never_requests_an_async_loop(request):
    assert "retire_test_app_owners" not in request.fixturenames
    assert not any(
        name.endswith("_scoped_runner") or name == "event_loop"
        for name in request.fixturenames
    )
    with pytest.raises(RuntimeError, match="no running event loop"):
        asyncio.get_running_loop()


def test_factory_cleanup_keeps_replacement_instance_lock_caller_owned():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from Tests.UI import app_factory

    app = app_factory._build_test_app()
    original = app._instance_lock_status.handle
    borrowed_close = Mock()
    app._instance_lock_status = SimpleNamespace(
        handle=SimpleNamespace(close=borrowed_close)
    )

    assert app_factory.drain_created_dirs() == 1
    assert original.closed
    borrowed_close.assert_not_called()


async def test_app_owner_finalizer_joins_cancelled_native_worker_before_config_reset(
    monkeypatch,
):
    import threading

    from textual.worker import Worker

    from Tests import conftest as root_fixtures
    from Tests.UI import app_factory
    from tldw_chatbook import config

    app = app_factory._build_test_app()
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    reset = root_fixtures._reset_config_database_instances

    def native_work():
        database = config.get_chachanotes_db_lazy()
        assert database is not None
        entered.set()
        try:
            assert release.wait(2.0)
            assert database.get_connection().execute("SELECT 1").fetchone()[0] == 1
        finally:
            finished.set()

    def reset_after_join(module):
        assert finished.is_set(), "config reset preceded native worker completion"
        reset(module)

    monkeypatch.setattr(
        root_fixtures, "_reset_config_database_instances", reset_after_join
    )
    worker = Worker(app, native_work, thread=True)
    task = asyncio.create_task(worker.run())
    assert await asyncio.to_thread(entered.wait, 2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not finished.is_set()
    asyncio.get_running_loop().call_later(0.02, release.set)


@pytest.mark.parametrize("failure", [RuntimeWarning, asyncio.CancelledError])
async def test_app_owner_executor_timeout_does_not_retire_database_or_lock(
    monkeypatch, failure
):
    import warnings
    from types import SimpleNamespace

    from Tests import conftest as root_fixtures
    from Tests.UI import app_factory
    from tldw_chatbook.app_lifecycle import WORKER_CANCELLATION_GRACE_SECONDS

    app = app_factory._build_test_app()
    database = app.local_workspace_db
    connection = next(iter(database._maintenance_participant.connections))
    lock = app._instance_lock_status.handle
    directory = app_factory._created_dirs[-1]

    async def timeout(*, timeout):
        assert timeout == WORKER_CANCELLATION_GRACE_SECONDS
        if failure is asyncio.CancelledError:
            raise asyncio.CancelledError("native executor join cancelled")
        warnings.warn("native executor join timed out", RuntimeWarning, stacklevel=2)

    with monkeypatch.context() as scoped:
        scoped.setattr(asyncio.get_running_loop(), "shutdown_default_executor", timeout)
        node = SimpleNamespace()
        cleanup = root_fixtures.retire_test_app_owners.__wrapped__(
            None, SimpleNamespace(node=node)
        )
        await anext(cleanup)
        with pytest.raises(failure, match="native executor join"):
            await anext(cleanup)
        assert node._test_app_native_workers_uncertain is True
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        assert database._maintenance_participant.connections
        assert directory.exists()
        assert not lock.closed


async def test_unrelated_async_cleanup_does_not_shutdown_the_executor(monkeypatch):
    from tldw_chatbook.app_lifecycle import WORKER_CANCELLATION_GRACE_SECONDS

    loop = asyncio.get_running_loop()
    original = loop.shutdown_default_executor

    async def observed_join(timeout=None):
        assert timeout != WORKER_CANCELLATION_GRACE_SECONDS, (
            "unrelated async test attempted app-owner executor shutdown"
        )
        await original(timeout)

    monkeypatch.setattr(loop, "shutdown_default_executor", observed_join)
    assert await asyncio.to_thread(lambda: 42) == 42


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_app_owner_runtime_failure_does_not_retire_database_or_lock(
    monkeypatch, failure
):
    from types import SimpleNamespace

    from Tests import conftest as root_fixtures
    from Tests.UI import app_factory

    app = app_factory._build_test_app()
    database = app.local_workspace_db
    connection = next(iter(database._maintenance_participant.connections))
    lock = app._instance_lock_status.handle
    directory = app_factory._created_dirs[-1]

    async def failed_disposal():
        raise failure("runtime retirement interrupted")

    with monkeypatch.context() as scoped:
        scoped.setattr(app_factory, "drain_created_runtimes", failed_disposal)
        node = SimpleNamespace()
        cleanup = root_fixtures.retire_test_app_owners.__wrapped__(
            None, SimpleNamespace(node=node)
        )
        await anext(cleanup)
        with pytest.raises(failure, match="runtime retirement interrupted"):
            await anext(cleanup)
        assert node._test_app_native_workers_uncertain is True
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        assert database._maintenance_participant.connections
        assert directory.exists()
        assert not lock.closed
