"""Finite Guardian callbacks retire native handles before their worker exits."""
import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC

import pytest

from tldw_chatbook.DB.base_db import operation_owned_connection, run_owned_db_call
from tldw_chatbook.DB.Guardian_DB import GuardianDB


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_guardian_worker_retires_new_native_handle_before_pool_exit(tmp_path, fail):
    db = GuardianDB(tmp_path / "guardian.db", "finite-test")
    main = db._held_connection()
    handles = []
    def operation():
        connection = db._held_connection()
        handles.append(connection)
        if fail:
            connection.execute("SELECT * FROM missing_finite_guardian_table")
        return db.list_rules()
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.set_default_executor(executor)
        try:
            if fail:
                with pytest.raises(sqlite3.OperationalError):
                    await run_owned_db_call(db, operation)
            else:
                assert await run_owned_db_call(db, operation)
            # Inspection happens on the still-alive originating worker, before
            # executor shutdown or natural interpreter cleanup can hide a leak.
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                await asyncio.to_thread(handles[0].execute, "SELECT 1")
            assert main.execute("SELECT 1").fetchone()[0] == 1
            assert await asyncio.to_thread(lambda: True)
        finally:
            await asyncio.to_thread(db.close)
            db.close()


def test_guardian_scope_preserves_a_borrowed_transaction(tmp_path):
    db = GuardianDB(tmp_path / "guardian.db", "finite-test")
    try:
        connection = db._held_connection()
        connection.execute("BEGIN")
        with operation_owned_connection(db):
            assert db.list_rules()
        assert connection.in_transaction
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        connection.rollback()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_daily_native_work_is_finite_and_notification_keeps_main_thread(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    from tldw_chatbook.Guardian import settings, trend_analyzer
    from tldw_chatbook.Scheduling.scheduler.handlers import (
        guardian_trend_handler as handler_module,
    )

    monkeypatch.setattr(settings, "guardian_setting", lambda key, default=None: True if key == "enabled" else default)
    db = GuardianDB(tmp_path / "daily.db", "daily-test")
    notification_connection = sqlite3.connect(":memory:")
    notification_connection.execute("CREATE TABLE notifications (body TEXT)")
    main_thread = threading.get_ident()
    worker_handles = []
    def analyze(database, **kwargs):
        assert threading.get_ident() != main_thread
        worker_handles.append(database._held_connection())
        return [SimpleNamespace(label="local trend", message="local summary")]
    monkeypatch.setattr(trend_analyzer, "analyze", analyze)
    def dispatch(**kwargs):
        assert threading.get_ident() == main_thread
        notification_connection.execute("INSERT INTO notifications VALUES (?)", (kwargs["message"],))
    deps = handler_module.GuardianTrendDeps(db, SimpleNamespace(dispatch=dispatch))
    handler = handler_module.GuardianTrendHandler(deps_getter=lambda: deps)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.set_default_executor(executor)
        try:
            await handler.handle({"id": "guardian:trends"})
            await asyncio.gather(*tuple(handler_module._SPAWNED_TREND_RUNS))
            assert notification_connection.execute("SELECT body FROM notifications").fetchall() == [("local summary",)]
            assert db.get_meta("trends_last_run")
            for connection in worker_handles:
                with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                    await asyncio.to_thread(connection.execute, "SELECT 1")
            assert await asyncio.to_thread(lambda: True)
        finally:
            await asyncio.to_thread(db.close)
            db.close()
            notification_connection.close()


@pytest.mark.asyncio
async def test_daily_shutdown_drains_physical_analysis_and_suppresses_delivery(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    from tldw_chatbook.Guardian import settings, trend_analyzer
    from tldw_chatbook.Scheduling.scheduler.handlers import (
        guardian_trend_handler as handler_module,
    )

    monkeypatch.setattr(settings, "guardian_setting", lambda key, default=None: True if key == "enabled" else default)
    db = GuardianDB(tmp_path / "daily.db", "daily-test")
    main_thread = threading.get_ident()
    started = asyncio.Event()
    release = threading.Event()
    physical_done = threading.Event()
    loop = asyncio.get_running_loop()
    deliveries = []
    def analyze(database, **kwargs):
        assert threading.get_ident() != main_thread
        database.list_rules()
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        physical_done.set()
        return [SimpleNamespace(label="local trend", message="local summary")]
    monkeypatch.setattr(trend_analyzer, "analyze", analyze)
    deps = handler_module.GuardianTrendDeps(db, SimpleNamespace(dispatch=lambda **kwargs: deliveries.append(kwargs)))
    handler = handler_module.GuardianTrendHandler(deps_getter=lambda: deps)
    try:
        await handler.handle({"id": "guardian:trends"})
        await asyncio.wait_for(started.wait(), 3)
        draining = asyncio.create_task(handler_module.shutdown())
        await asyncio.sleep(0)
        assert not draining.done()
        assert not physical_done.is_set()
        release.set()
        await asyncio.wait_for(draining, 3)
        assert physical_done.is_set()
        assert deliveries == []
        assert db.get_meta("trends_last_run") is None
    finally:
        release.set()
        await asyncio.gather(*tuple(handler_module._SPAWNED_TREND_RUNS), return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_projection_retires_only_its_worker_handle(tmp_path, monkeypatch):
    from datetime import datetime

    from tldw_chatbook.Scheduling.services import guardian_projection
    db = GuardianDB(tmp_path / "projection.db")
    main = db._held_connection()
    handles = []
    original = db.get_meta
    def read(key):
        handles.append(db._held_connection())
        return original(key)
    monkeypatch.setattr(db, "get_meta", read)
    monkeypatch.setattr(guardian_projection, "guardian_setting", lambda *args: True)
    projection = guardian_projection.GuardianProjection(lambda: db)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.set_default_executor(executor)
        try:
            assert await asyncio.to_thread(projection.tasks, datetime.now(UTC))
            assert len(handles) == 1
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                await asyncio.to_thread(handles[0].execute, "SELECT 1")
            assert main.execute("SELECT 1").fetchone()[0] == 1
        finally:
            await asyncio.to_thread(db.close)
            db.close()


@pytest.mark.asyncio
async def test_discovery_reader_retires_only_its_new_native_worker_handle(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from tldw_chatbook.Dreams.profile_sources import read_guardian_topics
    from tldw_chatbook.Guardian import settings

    monkeypatch.setattr(settings, "guardian_setting", lambda *args: True)
    db = GuardianDB(tmp_path / "discovery.db")
    main = db._held_connection()
    handles = []
    original = db.connection

    @contextmanager
    def capture_connection():
        with original() as connection:
            handles.append(connection)
            yield connection

    monkeypatch.setattr(db, "connection", capture_connection)
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.get_running_loop().set_default_executor(executor)
        try:
            assert await asyncio.to_thread(read_guardian_topics, db) == []
            assert len(handles) == 1
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                await asyncio.to_thread(handles[0].execute, "SELECT 1")
            assert main.execute("SELECT 1").fetchone()[0] == 1
            assert await asyncio.to_thread(lambda: True)
        finally:
            await asyncio.to_thread(db.close)
            db.close()
