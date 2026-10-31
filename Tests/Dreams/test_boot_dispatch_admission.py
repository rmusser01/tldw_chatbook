"""Native scheduler occurrences stay held after Dreams startup timeout/error."""
import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Dreams import track_reconcile, track_service
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection
from tldw_chatbook.Subscriptions import LocalWatchlistsService

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "failure", "incomplete"])
async def test_pending_dreams_stays_held_without_stopping_unrelated_dispatch(
    tmp_path, monkeypatch, outcome,
):
    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting",
                        lambda section, key, default=None: False if key == "enabled" else default)
    path = tmp_path / "dreams.db"
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    subs = SubscriptionsDB(tmp_path / "subscriptions.db")
    monkeypatch.setattr(track_reconcile, "get_subscriptions_db_path", lambda: subs.db_path)
    scheduler = ScheduledTasksDB(tmp_path / "scheduler.db")
    dreams = DreamsDB(path)
    owned = subs.add_subscription("dream", "url", "https://example.com/dream", last_checked="2000-01-01T00:00:00+00:00")
    ordinary = subs.add_subscription("ordinary", "url", "https://example.com/ordinary", last_checked="2000-01-01T00:00:00+00:00")
    for source_id in (owned, ordinary):
        subs.update_subscription(source_id, last_checked="2000-01-01T00:00:00+00:00", check_frequency=60)
    # Authoring updates deliberately do not own the monitor timestamp.
    with subs.transaction() as connection:
        connection.execute("UPDATE subscriptions SET last_checked = ?", ("2000-01-01T00:00:00+00:00",))
    item = dreams.create_tracked_item(mechanism="page", intent="topic", cadence_seconds=3600,
                                    subscription_id=owned, created_by_dreams=1)
    dreams.close_all()
    due = "2026-01-01T00:00:00+00:00"
    protected = scheduler.create_reminder_task(owner_id="local", title="dream event", schedule_kind="one_time",
        next_run_at=due, link_type="dream_tracked_item", link_id=str(item))
    unrelated = scheduler.create_reminder_task(owner_id="local", title="ordinary reminder", schedule_kind="one_time", next_run_at=due)
    sent = []
    async def handler(task):
        sent.append(task["id"])
    loop = SchedulerLoop(scheduler, {"reminder": handler, "watchlist_job": handler},
                         watchlist_projection=WatchlistProjection(subs), clock=lambda: datetime(2026, 1, 2, tzinfo=UTC))
    loop._FIRST_DISPATCH_GATE_TIMEOUT_SECONDS = 0.01
    gate = track_reconcile.first_dispatch_gate(loop)
    app = SimpleNamespace(get_dreams_db=lambda: None, scheduler_loop=loop,
        scheduling_service=SimpleNamespace(db=scheduler),
        local_watchlists_service=LocalWatchlistsService(db_factory=lambda: subs))
    reached, release = asyncio.Event(), asyncio.Event()
    original = track_service.reconcile_tracking
    async def pending(*args, **kwargs):
        reached.set()
        await release.wait()
        return await original(*args, **kwargs)
    async def incomplete(*args, **kwargs):
        return 0
    def broken():
        raise RuntimeError("ownership unavailable")
    if outcome == "timeout":
        monkeypatch.setattr(track_service, "reconcile_tracking", pending)
    elif outcome == "incomplete":
        monkeypatch.setattr(track_service, "reconcile_tracking", incomplete)
    else:
        monkeypatch.setattr(track_reconcile, "get_dreams_db_path", broken)
    boot = asyncio.create_task(track_reconcile.reconcile_at_boot(app, gate))
    try:
        if outcome == "timeout":
            await asyncio.wait_for(reached.wait(), 5)
        else:
            await asyncio.wait_for(boot, 5)
        await loop._await_first_dispatch_gates()
        loop.queue.load()
        assert {f"watchlist:{owned}", f"watchlist:{ordinary}"} <= {row["id"] for row in loop.queue._items}
        await loop.tick()
        expected = {unrelated, f"watchlist:{ordinary}"} if outcome != "failure" else {unrelated}
        assert set(sent) == expected
        row = scheduler.get_reminder_task(protected)
        assert row["enabled"] and row["next_run_at"] == due
        assert not row["last_run_at"]
        if outcome == "timeout":
            release.set()
            await asyncio.wait_for(boot, 5)
            assert gate.dispatch_ready
            assert not subs.get_subscription(owned)["is_active"]
            assert not scheduler.get_reminder_task(protected)["enabled"]
    finally:
        release.set()
        await asyncio.wait_for(boot, 5)
        subs.close()
        scheduler.close()


@pytest.mark.asyncio
async def test_failed_ownership_classification_retires_boot_owned_native_handle(tmp_path, monkeypatch):
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "classification.db"
    seeded = DreamsDB(path)
    seeded.create_tracked_item(mechanism="page", intent="topic", cadence_seconds=3600,
                               subscription_id=1, created_by_dreams=1)
    seeded.close_all()
    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting",
                        lambda section, key, default=None: False if key == "enabled" else default)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    stores, handles = [], []
    def fail_read(self):
        stores.append(self)
        handles.append(self._held_connection())
        raise RuntimeError("ownership read unavailable")
    monkeypatch.setattr(DreamsDB, "dream_owned_subscription_ids", fail_read)
    loop = SimpleNamespace(first_dispatch_gates=[])
    gate = track_reconcile.first_dispatch_gate(loop)
    app = SimpleNamespace(get_dreams_db=lambda: None, scheduler_loop=loop,
                          scheduling_service=None, local_watchlists_service=None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.get_running_loop().set_default_executor(executor)
        try:
            await track_reconcile.reconcile_at_boot(app, gate)
            assert len(handles) == 1
            assert gate.is_set() and not gate.dispatch_ready
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                await asyncio.to_thread(handles[0].execute, "SELECT 1")
            assert await asyncio.to_thread(lambda: True)
        finally:
            for store in stores:
                store.close_all()
