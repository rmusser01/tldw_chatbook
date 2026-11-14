"""Finite Goal work retires its own real SQLite handles, never borrowed ones."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.Chat.test_goal_cli_verification import (
    test_two_increment_real_cli_repairs_file_with_model_visible_evidence as run_cli,
)
from Tests.Chat.test_goal_conversation_provisioning import service
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture
from Tests.UI.test_console_goal_controls import (
    test_mounted_pause_before_capacity_resume_dispatches_first_iteration as run_mounted,
)

stores = _stores_fixture

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


def current_connections(database):
    participant = database._maintenance_participant
    return tuple(
        connection
        for connection, lease in participant.connections.items()
        if lease.resource_thread is threading.current_thread()
    )


@pytest.mark.parametrize("borrowed", [False, True])
async def test_service_operation_retires_only_fresh_worker_handles(stores, borrowed):
    owner = service(stores)
    dbs = (stores[0], stores[1].db, stores[2].db)

    def work():
        before = []
        if borrowed:
            for database in dbs:
                getter = getattr(database, "get_connection", None)
                if getter is not None:
                    getter()
                else:
                    with database.connection():
                        pass
                before.append(current_connections(database))
        try:
            goal = owner.create(stores[3], launch_id="finite")
            assert goal.status == "ready"
            if borrowed:
                assert [current_connections(db) for db in dbs] == before
                for handles in before:
                    for conn in handles:
                        assert conn.execute("SELECT 1").fetchone()[0] == 1
            else:
                assert all(not current_connections(db) for db in dbs)
        finally:
            for db in dbs:
                close = getattr(db, "close_connection", None) or db.close
                close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        await asyncio.get_running_loop().run_in_executor(executor, work)


@pytest.mark.parametrize("surface", ["cli", "mounted"])
@pytest.mark.parametrize("borrowed", [False, True])
async def test_native_goal_workers_leave_no_new_thread_handles(
    stores, monkeypatch, tmp_path, surface, borrowed
):
    dbs = (stores[0], stores[2].db)
    before = {
        id(conn) for db in dbs for conn in db._maintenance_participant.connections
    }
    observed_threads = set()
    if borrowed:
        from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext

        original = AutomaticWorkContext.check

        def borrowed_check(context):
            if (
                context.goal is None
                or threading.current_thread() is threading.main_thread()
            ):
                return original(context)
            acquired = []
            handles = []
            for db in dbs:
                had = current_connections(db)
                with db.connection() as conn:
                    handles.append(conn)
                if not had:
                    acquired.append(db)
            try:
                result = original(context)
                for conn in handles:
                    assert conn.execute("SELECT 1").fetchone()[0] == 1
                observed_threads.add(threading.current_thread().name)
                return result
            finally:
                for db in acquired:
                    db.close()

        monkeypatch.setattr(AutomaticWorkContext, "check", borrowed_check)
    if surface == "cli":
        await run_cli(stores, monkeypatch, tmp_path)
    else:
        await run_mounted(stores, monkeypatch)
    for runtime in stores[0]._goal_test_runtimes:
        await runtime.dispose()
    retained = [
        (str(db.db_path), lease.resource_thread.name, lease.resource_thread.is_alive())
        for db in dbs
        for conn, lease in db._maintenance_participant.connections.items()
        if id(conn) not in before
        and lease.resource_thread is not threading.main_thread()
    ]
    assert retained == []
    if borrowed:
        assert "console-agent-loop" in observed_threads
        assert any(name.startswith("asyncio_") for name in observed_threads)
        if surface == "cli":
            assert "tool-fs_edit" in observed_threads
    # Exact owners retire their main-thread borrowers before maintenance admission.
    for db in dbs:
        db.close()
        participant = db._maintenance_participant
        participant.close_admission()
        try:
            assert participant.drain(time.monotonic() + 1)
        finally:
            participant.resume()


async def test_cancelled_callback_retains_handle_until_physical_worker_finishes(stores):
    owner = service(stores)
    db = stores[0]
    db.close()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        with db.connection() as conn:
            entered.set()
            assert release.wait(5)
            assert conn.execute("SELECT 1").fetchone()[0] == 1
        finished.set()

    task = asyncio.create_task(owner.call(work))
    assert await asyncio.to_thread(entered.wait, 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    participant = db._maintenance_participant
    participant.close_admission()
    try:
        assert not participant.drain(time.monotonic() + 0.02)
        assert participant.connections
        release.set()
        assert await asyncio.to_thread(finished.wait, 3)
        assert await asyncio.to_thread(participant.drain, time.monotonic() + 3)
    finally:
        release.set()
        participant.resume()


async def test_operation_keeps_memory_and_unrelated_borrowers(stores, tmp_path):
    from types import SimpleNamespace

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    memory = AgentRunsDB(":memory:")
    foreign = AgentRunsDB(tmp_path / "foreign.db")
    try:
        with memory.connection() as memory_conn, foreign.transaction() as foreign_conn:
            foreign_conn.execute("CREATE TABLE held(value INTEGER)")
            foreign_conn.execute("INSERT INTO held VALUES (7)")
            with owned_goal_storage(
                SimpleNamespace(db=memory, persistence=SimpleNamespace())
            ):
                assert memory_conn.execute("SELECT 1").fetchone()[0] == 1
            assert memory_conn.execute("SELECT 1").fetchone()[0] == 1
            assert foreign_conn.in_transaction
            assert foreign_conn.execute("SELECT value FROM held").fetchone()[0] == 7
    finally:
        memory.close()
        foreign.close()
