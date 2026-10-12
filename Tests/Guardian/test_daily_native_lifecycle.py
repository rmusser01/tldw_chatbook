"""Unmocked daily analysis, notification affinity, shutdown and recovery drain."""

import asyncio
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from tldw_chatbook.Backup_Recovery.participants import _repository_participant
from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.Guardian import settings
from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    NotificationDispatchService,
)
from tldw_chatbook.Scheduling.scheduler.handlers import (
    guardian_trend_handler as handler_module,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["complete", "shutdown", "recovery"])
async def test_actual_daily_analysis_native_lifetime_and_drain(
    tmp_path, monkeypatch, outcome
):
    now = datetime(2026, 10, 12, 12, tzinfo=UTC)
    monkeypatch.setattr(
        "tldw_chatbook.Utils.timestamps.utc_now_iso", lambda: now.isoformat()
    )
    monkeypatch.setattr(
        settings,
        "guardian_setting",
        lambda key, default=None: (
            True
            if key == "enabled"
            else default
            if default is not None
            else settings.GUARDIAN_DEFAULTS.get(key)
        ),
    )
    db = GuardianDB(tmp_path / "daily.sqlite", "daily-native")
    rule = db.upsert_rule(name="Science", topic="science", pattern="science")
    for index in range(30):
        db.insert_alert(
            rule_id=rule,
            session_id="captured-chat",
            visit_id="visit",
            topic="science",
            message_digest="d" * 64,
            ts=(now - timedelta(days=index % 7)).isoformat(),
        )
    old = db.insert_alert(
        rule_id=rule,
        session_id="old-chat",
        visit_id="old-visit",
        topic="science",
        message_digest="e" * 64,
        ts=(now - timedelta(days=200)).isoformat(),
    )
    db.close()
    notifications = ClientNotificationsDB(":memory:", "daily-affinity")
    dispatch = NotificationDispatchService(notifications)
    handler = handler_module.GuardianTrendHandler(
        deps_getter=lambda: handler_module.GuardianTrendDeps(db, dispatch)
    )
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    captured = []
    creator = db._get_connection
    main_thread = threading.get_ident()
    first = [True]

    def trace(statement):
        if first[0] and statement.startswith(
            "SELECT topic, ts, crisis_provenance FROM guardian_alerts"
        ):
            first[0] = False
            assert threading.get_ident() != main_thread
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)

    def acquire():
        connection = creator()
        captured.append(connection)
        connection.set_trace_callback(trace)
        return connection

    monkeypatch.setattr(db, "_get_connection", acquire)
    pool = ThreadPoolExecutor(max_workers=1)
    previous = loop._default_executor
    loop.set_default_executor(pool)
    participant = _repository_participant(db)
    try:
        await handler.handle({"id": "guardian:trends"})
        await asyncio.wait_for(entered.wait(), 3)
        # The scheduler handle returned and the event loop remains responsive
        # while the real file-backed SELECT is still admitted on its worker.
        responsive = asyncio.Event()
        loop.call_soon(responsive.set)
        await asyncio.wait_for(responsive.wait(), 1)
        if outcome == "shutdown":
            settling = asyncio.create_task(handler_module.shutdown())
            await asyncio.sleep(0)
            assert not settling.done() and not release.is_set()
        elif outcome == "recovery":
            participant.close_admission()
            assert not participant.drain(time.monotonic() + 0.02)
            participant.resume()
        release.set()
        if outcome == "shutdown":
            await asyncio.wait_for(settling, 3)
        else:
            await asyncio.wait_for(
                asyncio.gather(*tuple(handler_module._SPAWNED_TREND_RUNS)), 3
            )

        def physically_closed():
            assert captured
            for connection in captured:
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    connection.execute("SELECT 1")

        await loop.run_in_executor(pool, physically_closed)
        assert await loop.run_in_executor(pool, lambda: "still alive") == "still alive"
        participant.close_admission()
        assert participant.drain(time.monotonic() + 0.1)
        participant.resume()
        # Inspect persisted effects independently, without reopening the runtime
        # handle whose physical closure was just proved.
        connection = sqlite3.connect(db.db_path)
        try:
            rows = connection.execute(
                "SELECT topic FROM guardian_alerts WHERE rule_id IS NULL"
            ).fetchall()
            assert rows == [("fixation:science",)]
            watermark = connection.execute(
                "SELECT value FROM guardian_meta WHERE key='trends_last_run'"
            ).fetchone()
            old_exists = connection.execute(
                "SELECT 1 FROM guardian_alerts WHERE id=?", (old,)
            ).fetchone()
        finally:
            connection.close()
        inbox = notifications.list_notifications(category="guardian")
        if outcome == "shutdown":
            assert watermark is None and old_exists and inbox == []
        else:
            assert watermark == (now.isoformat(),) and old_exists is None
            assert len(inbox) == 1 and "science" in inbox[0]["message"]
    finally:
        release.set()
        participant.resume()
        await handler_module.shutdown()
        await loop.run_in_executor(pool, db.close)
        loop._default_executor = previous
        pool.shutdown(wait=True)
        notifications.close()
        db.close()


def _native_daily_pair(tmp_path, monkeypatch):
    now = datetime(2026, 10, 12, 12, tzinfo=UTC)
    monkeypatch.setattr(
        "tldw_chatbook.Utils.timestamps.utc_now_iso", lambda: now.isoformat()
    )
    enabled = [True]
    monkeypatch.setattr(
        settings,
        "guardian_setting",
        lambda key, default=None: (
            enabled[0]
            if key == "enabled"
            else default
            if default is not None
            else settings.GUARDIAN_DEFAULTS.get(key)
        ),
    )
    db = GuardianDB(tmp_path / "guardian.sqlite", "native-dispatch")
    rule = db.upsert_rule(name="Science", topic="science", pattern="science")
    for index in range(30):
        db.insert_alert(
            rule_id=rule,
            session_id="chat",
            visit_id="visit",
            topic="science",
            message_digest="d" * 64,
            ts=(now - timedelta(days=index % 7)).isoformat(),
        )
    db.close()
    inbox = ClientNotificationsDB(tmp_path / "inbox.sqlite", "native-dispatch")
    inbox.get_settings()  # Ensure real tables before taking a competing writer.
    inbox.close()
    service = NotificationDispatchService(inbox)
    handler = handler_module.GuardianTrendHandler(
        deps_getter=lambda: handler_module.GuardianTrendDeps(db, service)
    )
    return db, inbox, handler, enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
async def test_daily_file_inbox_contention_keeps_ui_live_and_retires_only_new_handles(
    tmp_path, monkeypatch, borrowed
):
    db, inbox, handler, _ = _native_daily_pair(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)
    previous = loop._default_executor
    loop.set_default_executor(pool)
    locked = threading.Event()
    inserting = threading.Event()
    heartbeat = threading.Event()
    observation = []
    errors = []
    captured = []
    original = inbox._get_connection

    def trace(statement):
        if "INSERT INTO client_notifications" in statement:
            inserting.set()
            loop.call_soon_threadsafe(lambda: loop.call_later(0.02, heartbeat.set))

    def acquire():
        connection = original()
        captured.append((threading.get_ident(), connection))
        connection.set_trace_callback(trace)
        return connection

    monkeypatch.setattr(inbox, "_get_connection", acquire)

    def competing_writer():
        connection = sqlite3.connect(inbox.db_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            locked.set()
            if not inserting.wait(3):
                errors.append("actual notification INSERT never entered")
            # Bounded watchdog also releases a genuinely blocked UI in RED.
            observation.append(heartbeat.wait(0.5))
            connection.rollback()
        finally:
            connection.close()

    writer = threading.Thread(target=competing_writer)
    try:
        if borrowed:
            await loop.run_in_executor(pool, inbox._get_connection)
        writer.start()
        assert await loop.run_in_executor(pool, locked.wait, 3)
        await handler.handle({"id": "guardian:trends"})
        await asyncio.wait_for(
            asyncio.gather(*tuple(handler_module._SPAWNED_TREND_RUNS)), 4
        )
        writer.join(timeout=1)
        assert not writer.is_alive() and not errors
        assert observation == [True], (
            "event-loop heartbeat did not run while native INSERT waited for writer"
        )

        def probe():
            assert captured
            for thread_id, connection in captured:
                assert thread_id == threading.get_ident()
                if borrowed:
                    assert connection.execute("SELECT 1").fetchone()[0] == 1
                else:
                    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                        connection.execute("SELECT 1")
            return threading.get_ident()

        worker = await loop.run_in_executor(pool, probe)
        assert worker == await loop.run_in_executor(pool, threading.get_ident)
        connection = sqlite3.connect(inbox.db_path)
        try:
            assert (
                connection.execute(
                    "SELECT count(*) FROM client_notifications WHERE category='guardian'"
                ).fetchone()[0]
                == 1
            )
        finally:
            connection.close()
    finally:
        heartbeat.set()
        if writer.ident is not None:
            writer.join(timeout=4)
        await handler_module.shutdown()
        await loop.run_in_executor(pool, inbox.close)
        await loop.run_in_executor(pool, db.close)
        loop._default_executor = previous
        pool.shutdown(wait=True)
        inbox.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["disable", "shutdown"])
async def test_daily_file_delivery_rechecks_authority_after_executor_queue(
    tmp_path, monkeypatch, boundary
):
    db, inbox, handler, enabled = _native_daily_pair(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    queued = asyncio.Event()
    occupying = asyncio.Event()
    release = threading.Event()
    active = threading.Event()

    class ObservedPool(ThreadPoolExecutor):
        def submit(self, function, /, *args, **kwargs):
            if active.is_set():
                loop.call_soon_threadsafe(queued.set)
            return super().submit(function, *args, **kwargs)

    pool = ObservedPool(max_workers=1)
    previous = loop._default_executor
    loop.set_default_executor(pool)
    original = handler_module._store_work
    first = [True]

    def occupy():
        active.set()
        loop.call_soon_threadsafe(occupying.set)
        assert release.wait(5)

    async def after_real_analysis(*args):
        result = await original(*args)
        if first[0]:
            first[0] = False
            loop.run_in_executor(pool, occupy)
            await asyncio.wait_for(occupying.wait(), 2)
        return result

    monkeypatch.setattr(handler_module, "_store_work", after_real_analysis)
    shutdown = None
    try:
        await handler.handle({"id": "guardian:trends"})
        await asyncio.wait_for(queued.wait(), 3)
        connection = sqlite3.connect(inbox.db_path)
        try:
            assert (
                connection.execute(
                    "SELECT count(*) FROM client_notifications"
                ).fetchone()[0]
                == 0
            )
        finally:
            connection.close()
        if boundary == "disable":
            enabled[0] = False
        else:
            shutdown = asyncio.create_task(handler_module.shutdown())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not shutdown.done(), (
                "shutdown must retain queued physical delivery until settlement"
            )
        active.clear()
        release.set()
        if shutdown is not None:
            await asyncio.wait_for(shutdown, 3)
        else:
            await asyncio.wait_for(
                asyncio.gather(*tuple(handler_module._SPAWNED_TREND_RUNS)), 3
            )
        assert inbox.list_notifications(category="guardian") == []
        assert db.get_meta("trends_last_run") is None
    finally:
        active.clear()
        release.set()
        await handler_module.shutdown()
        if shutdown is not None:
            await shutdown
        await loop.run_in_executor(pool, inbox.close)
        await loop.run_in_executor(pool, db.close)
        loop._default_executor = previous
        pool.shutdown(wait=True)
        inbox.close()
        db.close()
