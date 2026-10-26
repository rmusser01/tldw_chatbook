"""TASK-36931: real inbox I/O must release the mounted UI event loop."""

import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.UI.Subscription_Modules.notifications_inbox_controller import (
    NotificationsInboxController,
)
from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import NotificationsPane


async def until(predicate):
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.01)


@asynccontextmanager
async def inbox(tmp_path):
    app = _build_test_app()
    app.client_notifications_db.close()
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    app.client_notifications_db = store
    row = store.insert(category="research", title="Ready", message="Actual inbox row")
    host = DestinationHarness(app, "watchlists_collections")
    try:
        async with host.run_test(size=(180, 50)) as pilot:
            screen = host.screen_stack[-1]
            await until(lambda: bool(screen.query("#wl-tab-notifications")))
            screen.query_one("#wl-tab-notifications", Button).press()
            await until(lambda: len(screen._loaded_notifications) == 1)
            await pilot.pause()
            yield app, screen, store, row
    finally:
        store.close()


async def progress_during_hold(entered, release):
    """Release after loop progress; an external watchdog also bounds broken code."""
    ticks = 0
    await until(entered.is_set)
    while not release.is_set():
        ticks += 1
        if ticks == 3:
            release.set()
            break
        await asyncio.sleep(0.01)
    return ticks


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["mark-read", "dismiss"])
@private_profile_test
async def test_inbox_mutations_persist_and_refresh_without_blocking(
    request, tmp_path, monkeypatch, action
):
    async with inbox(tmp_path) as (_, screen, store, row):
        assert not store.is_memory_db
        entered, release, holder_ready = (threading.Event() for _ in range(3))
        holder_errors = []
        observed_threads = []
        original = store._update_flags

        def update(notification_id, **fields):
            # Open the thread's real connection before signaling the competing
            # writer; the actual UPDATE then follows while its lock is held.
            store._held_connection()
            observed_threads.append(threading.get_ident())
            entered.set()
            return original(notification_id, **fields)

        monkeypatch.setattr(store, "_update_flags", update)

        def holder():
            try:
                with store.transaction() as connection:
                    connection.execute(
                        "UPDATE client_notifications SET is_read = is_read WHERE id = ?",
                        (row["id"],),
                    )
                    holder_ready.set()
                    assert entered.wait(15)
                    # Broken inline SQLite cannot prevent this independent release.
                    release.wait(1)
                    release.set()
            except BaseException as error:
                holder_errors.append(error)
            finally:
                store.close()

        thread = threading.Thread(target=holder)
        beat = None
        try:
            if thread:
                thread.start()
                await until(holder_ready.is_set)
                beat = asyncio.create_task(progress_during_hold(entered, release))
            pane = screen.query_one("#watchlists-notifications-pane", NotificationsPane)
            pane.select_notification_by_id(str(row["id"]))
            selector = f"#notifications-{action}-button"
            # Selection currently recomposes the pane (TASK-33283); await its live controls.
            await until(
                lambda: bool(screen.query(selector))
                and not screen.query_one(selector, Button).disabled
            )
            screen.query_one(selector, Button).press()
            if action == "mark-read":
                await until(
                    lambda: bool(screen._loaded_notifications)
                    and screen._loaded_notifications[0]["is_read"]
                )
            else:
                await until(lambda: not screen._loaded_notifications)
            if beat:
                assert (
                    await beat >= 3
                ), "UI loop made no progress during SQLite contention"
            assert observed_threads and all(
                ident != threading.get_ident() for ident in observed_threads
            ), "SQLite mutation executed on the UI thread"
            persisted = store.get_notification(row["id"])
            assert persisted["is_read"] is (action == "mark-read")
            assert persisted["is_dismissed"] is (action == "dismiss")
            assert not holder_errors
        finally:
            release.set()
            entered.set()
            if thread:
                await asyncio.to_thread(thread.join, 5)
                assert not thread.is_alive()
            if beat:
                await beat


@pytest.mark.asyncio
@private_profile_test
async def test_inbox_read_hold_leaves_ui_loop_responsive(
    request, tmp_path, monkeypatch
):
    async with inbox(tmp_path) as (_, screen, store, row):
        entered, release = threading.Event(), threading.Event()
        observed_threads = []
        original = store._normalize_row

        def normalize(raw):
            # Retain the real SELECT and its materialized row; hold only decoding.
            result = original(raw)
            observed_threads.append(threading.get_ident())
            entered.set()
            release.wait(1)
            release.set()
            return result

        monkeypatch.setattr(store, "_normalize_row", normalize)
        beat = asyncio.create_task(progress_during_hold(entered, release))
        try:
            pane = screen.query_one("#watchlists-notifications-pane", NotificationsPane)
            pane.query_one("#notifications-refresh-button", Button).press()
            assert await beat >= 3, "UI loop made no progress during inbox read"
            await until(
                lambda: not any(
                    not worker.is_finished and worker.group == "wc_notifications"
                    for worker in screen.app.workers
                )
            )
            assert observed_threads and all(
                ident != threading.get_ident() for ident in observed_threads
            )
            assert screen._loaded_notifications[0]["id"] == row["id"]
        finally:
            release.set()
            await beat


@pytest.mark.asyncio
@private_profile_test
async def test_inbox_corrupt_store_reports_load_failure(request, tmp_path):
    async with inbox(tmp_path) as (app, screen, _, row):
        path = tmp_path / "corrupt.db"
        path.write_bytes(b"This is not a SQLite database")
        corrupt = ClientNotificationsDB(path)
        screen._notifications_controller.store = corrupt
        feedback = []
        app.notify = lambda message, **kwargs: feedback.append((message, kwargs))
        try:
            pane = screen.query_one("#watchlists-notifications-pane", NotificationsPane)
            pane.query_one("#notifications-refresh-button", Button).press()
            await until(lambda: bool(feedback))
            assert feedback == [
                ("Failed to load local notifications.", {"severity": "error"})
            ]
            assert screen._loaded_notifications[0]["id"] == row["id"]
            assert screen.is_attached
        finally:
            corrupt.close()


class _PolicyApp:
    def __init__(self, *, allowed=True):
        self.allowed = allowed
        self.action_ids = []

    def require_ui_action_allowed(self, *, action_id):
        self.action_ids.append(action_id)
        return SimpleNamespace(allowed=self.allowed)


@pytest.mark.asyncio
async def test_memory_inbox_preserves_rows_and_mutations_on_owning_thread():
    store = ClientNotificationsDB(":memory:")
    try:
        row = store.insert(category="research", title="Ready", message="In memory")
        controller = NotificationsInboxController(
            app_instance=_PolicyApp(), store=store
        )
        assert (await controller.load_rows())[0]["id"] == row["id"]
        assert await controller.mark_read(row["id"], is_read=True)
        assert await controller.dismiss(row["id"], is_dismissed=True)
        assert await controller.load_rows() == []
        persisted = store.get_notification(row["id"])
        assert persisted["is_read"] and persisted["is_dismissed"]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_inbox_awaitable_overrides_keep_callers_loop_and_return_values(tmp_path):
    import asyncio
    import threading

    store = ClientNotificationsDB(tmp_path / "awaitable.db")
    row = {"id": 7, "title": "Awaitable seam"}
    owner_loop = asyncio.get_running_loop()
    owner_thread = threading.get_ident()

    async def list_rows(**kwargs):
        assert asyncio.get_running_loop() is owner_loop
        assert threading.get_ident() == owner_thread
        assert kwargs["limit"] == 100 and not kwargs["include_dismissed"]
        await asyncio.sleep(0)
        return [row]

    async def mutate(notification_id, **fields):
        assert asyncio.get_running_loop() is owner_loop
        assert threading.get_ident() == owner_thread
        assert notification_id == 7
        await asyncio.sleep(0)
        row.update(fields)
        return True

    store.list_notifications = list_rows
    store.mark_read = mutate
    store.dismiss_notification = mutate
    controller = NotificationsInboxController(app_instance=_PolicyApp(), store=store)
    try:
        assert await controller.load_rows() == [row]
        assert await controller.mark_read(7, is_read=True)
        assert await controller.dismiss(7, is_dismissed=True)
        assert row == {
            "id": 7,
            "title": "Awaitable seam",
            "is_read": True,
            "is_dismissed": True,
        }
    finally:
        store.close()


@pytest.mark.asyncio
async def test_inbox_denial_prevents_store_access(tmp_path):
    store = ClientNotificationsDB(tmp_path / "denied.db")
    app = _PolicyApp(allowed=False)
    controller = NotificationsInboxController(app_instance=app, store=store)
    assert await controller.load_rows() == []
    assert not await controller.mark_read(7, is_read=True)
    assert not await controller.dismiss(7, is_dismissed=True)
    assert not store.db_path.exists()
    assert app.action_ids == [
        "notifications.queue.list.local",
        "notifications.queue.update.local",
        "notifications.queue.update.local",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["mark_read", "dismiss"])
async def test_inbox_sqlite_failure_propagates_without_changed_flags(tmp_path, action):
    import sqlite3

    store = ClientNotificationsDB(tmp_path / "failed.db")
    row = store.insert(category="research", title="Ready", message="Unchanged")
    try:
        with store.connection() as connection:
            connection.execute(
                "CREATE TRIGGER reject_update BEFORE UPDATE ON client_notifications "
                "BEGIN SELECT RAISE(ABORT, 'controlled storage failure'); END"
            )
        controller = NotificationsInboxController(
            app_instance=_PolicyApp(), store=store
        )
        kwargs = {"is_read": True} if action == "mark_read" else {"is_dismissed": True}
        with pytest.raises(sqlite3.IntegrityError, match="controlled storage failure"):
            await getattr(controller, action)(row["id"], **kwargs)
        persisted = store.get_notification(row["id"])
        assert not persisted["is_read"] and not persisted["is_dismissed"]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_custom_sync_inbox_stays_on_callers_thread():
    import threading
    from types import SimpleNamespace

    owner = threading.get_ident()
    row = {"id": 7}

    def list_rows(**kwargs):
        assert threading.get_ident() == owner
        return [row]

    def mutate(notification_id, **fields):
        assert threading.get_ident() == owner
        assert notification_id == 7
        row.update(fields)
        return True

    store = SimpleNamespace(
        is_memory_db=False,
        list_notifications=list_rows,
        mark_read=mutate,
        dismiss_notification=mutate,
    )
    controller = NotificationsInboxController(app_instance=_PolicyApp(), store=store)
    assert await controller.load_rows() == [{"id": 7}]
    assert await controller.mark_read(7, is_read=True)
    assert await controller.dismiss(7, is_dismissed=True)
    assert row == {"id": 7, "is_read": True, "is_dismissed": True}
