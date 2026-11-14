"""Actual application shutdown must retire its constructor-held SQLite handles."""

import asyncio
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("borrower", ["cold", "sibling"])
@private_profile_test
async def test_real_app_shutdown_retires_constructor_sqlite_handles(
    request: pytest.FixtureRequest,
    borrower: str,
) -> None:
    from tldw_chatbook.config import save_settings_to_cli_config

    assert save_settings_to_cli_config(
        {"splash_screen": {"enabled": False}, "first_run": {"setup_completed": True}}
    )
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    sibling = None
    if borrower == "sibling":
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        sibling = CharactersRAGDB(app.chachanotes_db.db_path, "retirement-borrower")
        previous = sibling.get_connection()
    owners = {
        "Chat": app.chachanotes_db,
        "Evaluations": app.evaluation_orchestrator.db,
        "Collections": app.local_library_collections_db,
        "Subscriptions": app.subscriptions_db,
        "Workspace": app.local_workspace_db,
    }
    connections = {
        "Chat": owners["Chat"].get_connection(),
        "Evaluations": getattr(owners["Evaluations"]._local, "connection", None),
        "Collections": getattr(owners["Collections"]._thread_local, "conn", None),
        "Subscriptions": getattr(owners["Subscriptions"]._local, "conn", None),
        "Workspace": getattr(owners["Workspace"]._thread_local, "conn", None),
    }
    assert all(connection is not None for connection in connections.values())
    try:
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
        retained = []
        for name, connection in connections.items():
            try:
                connection.execute("SELECT 1").fetchone()
            except sqlite3.ProgrammingError:
                pass
            else:
                retained.append(name)
        assert retained == [], f"Constructor SQLite handles remain live: {retained}"
        if sibling is not None:
            assert previous.execute("SELECT 1").fetchone()[0] == 1
    finally:
        if sibling is not None:
            sibling.close()
        for owner in owners.values():
            owner.close()


def _database_descriptors():
    """Native paths, not Darwin's misleading readlink-only zero census."""
    paths = []
    if sys.platform == "darwin":
        import fcntl

        for descriptor in os.listdir("/dev/fd"):
            try:
                raw = fcntl.fcntl(int(descriptor), 50, bytes(1024))  # F_GETPATH
            except OSError:
                continue
            paths.append(os.fsdecode(raw.split(b"\0", 1)[0]))
    elif sys.platform.startswith("linux"):
        for descriptor in os.listdir("/proc/self/fd"):
            try:
                paths.append(os.readlink(f"/proc/self/fd/{descriptor}"))
            except OSError:
                continue
    else:
        pytest.skip("Native descriptor census is implemented for Darwin and Linux")
    return [
        path
        for path in paths
        if any(suffix in path.lower() for suffix in (".sqlite", ".db", "-wal", "-shm"))
    ]


@pytest.mark.asyncio
@private_profile_test
async def test_repeated_real_apps_retire_before_cleanup_after_physical_executor_join(
    request,
):
    from tldw_chatbook.config import save_settings_to_cli_config

    assert save_settings_to_cli_config(
        {"splash_screen": {"enabled": False}, "first_run": {"setup_completed": True}}
    )
    from tldw_chatbook.app import TldwCli

    def sessions():
        counts = []
        for _ in range(3):
            owners, connections = [], []

            async def actual_app(owners=owners, connections=connections):
                app = TldwCli()
                owners.extend(
                    (
                        app.chachanotes_db,
                        app.evaluation_orchestrator.db,
                        app.local_library_collections_db,
                        app.subscriptions_db,
                        app.local_workspace_db,
                    )
                )
                connections.extend(
                    (
                        app.chachanotes_db.get_connection(),
                        app.evaluation_orchestrator.db._local.connection,
                        app.local_library_collections_db._thread_local.conn,
                        app.subscriptions_db._local.conn,
                        app.local_workspace_db._thread_local.conn,
                    )
                )
                # A real default-executor job proves this is a physical join boundary.
                await asyncio.to_thread(lambda: None)
                async with app.run_test(size=(120, 50)) as pilot:
                    await pilot.pause()

            try:
                with asyncio.Runner() as runner:
                    runner.run(actual_app())
                assert not any(
                    thread.name.startswith("asyncio_")
                    for thread in threading.enumerate()
                )
                for connection in connections:
                    with pytest.raises(
                        sqlite3.ProgrammingError, match="closed database"
                    ):
                        connection.execute("SELECT 1")
                paths = _database_descriptors()
                owned_paths = {str(owner.db_path) for owner in owners}
                assert not [
                    path
                    for path in paths
                    if any(
                        path == owned or path.startswith(owned + "-")
                        for owned in owned_paths
                    )
                ]
                counts.append(len(paths))
            finally:
                # No native measurement may be moved below this test cleanup.
                for owner in owners:
                    owner.close()
        print(f"PHYSICAL_JOIN_DB_FD_COUNTS={counts}")
        assert counts[1:] == counts[:-1], (
            "Database descriptors grew across real app sessions"
        )

    # Only test-owned loops/pool are joined; the pytest caller's executor is untouched.
    with ThreadPoolExecutor(max_workers=1) as executor:
        await asyncio.wrap_future(executor.submit(sessions))
