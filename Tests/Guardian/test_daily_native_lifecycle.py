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
