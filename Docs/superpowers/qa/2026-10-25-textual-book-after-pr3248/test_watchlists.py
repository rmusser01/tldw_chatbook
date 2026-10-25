from pathlib import Path
import asyncio, json
import pytest
from Tests.private_profile import private_profile_test

OUT = Path(__file__).resolve().parent


async def until(predicate, seconds=12):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@private_profile_test
async def test_notifications_keyboard(request, tmp_path):
    from textual.widgets import Button, DataTable
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import DestinationHarness
    from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import NotificationsPane

    app = _build_test_app()
    from tldw_chatbook.Notifications.client_notifications_db import (
        ClientNotificationsDB,
    )

    app.client_notifications_db.close()
    db = ClientNotificationsDB(tmp_path / "audit-notifications.sqlite")
    app.client_notifications_db = db
    assert not db.is_memory_db
    for letter in ["A", "B", "C", "D"]:
        db.insert_notification(
            category="research", title="Audit " + letter, message="Message " + letter
        )
    host = DestinationHarness(app, "watchlists_collections")
    import sys

    record = dict(
        file_backed=not db.is_memory_db,
        db_path=str(db.db_path),
        production_modules={
            kind.__name__: sys.modules[kind.__module__].__file__
            for kind in (type(db), NotificationsPane)
        },
    )
    try:
        async with host.run_test(size=(180, 50)) as pilot:
            screen = host.screen_stack[-1]
            await until(lambda: bool(screen.query("#wl-tab-notifications")))
            screen.query_one("#wl-tab-notifications", Button).press()
            await until(
                lambda: bool(screen.query("#watchlists-notifications-pane"))
                and len(
                    screen.query_one(
                        "#watchlists-notifications-pane", NotificationsPane
                    ).notifications
                )
                == 4
            )
            await pilot.pause()
            pane = screen.query_one("#watchlists-notifications-pane", NotificationsPane)
            table = pane.query_one("#notifications-table", DataTable)
            table.focus()
            await pilot.pause()

            def sample():
                live = pane.query_one("#notifications-table", DataTable)
                return dict(
                    focus=screen.focused.id if screen.focused else None,
                    same_table=live is table,
                    row=live.cursor_row,
                    selected=(pane.selected_notification or {}).get("title"),
                    screen_selected=(screen.selected_notification or {}).get("title"),
                )

            record["initial"] = sample()
            await pilot.press("down")
            await pilot.pause()
            record["after_first_down"] = sample()
            await pilot.press("down")
            await pilot.pause()
            record["after_second_down"] = sample()
            live = pane.query_one("#notifications-table", DataTable)
            live.focus()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            record["after_refocus_down"] = sample()
            assert record["initial"]["focus"] == "notifications-table"
            assert record["after_first_down"]["selected"] == "Audit C"
            assert record["after_second_down"]["selected"] == "Audit C"
            assert record["after_first_down"]["focus"] != "notifications-table"
            assert record["after_refocus_down"]["selected"] == "Audit B"
    finally:
        (OUT / "notifications-keyboard.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("contended", [False, True])
@private_profile_test
async def test_notification_write_contention(request, monkeypatch, contended, tmp_path):
    import threading, time
    from textual.widgets import Button, DataTable
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import DestinationHarness
    from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import NotificationsPane

    app = _build_test_app()
    from tldw_chatbook.Notifications.client_notifications_db import (
        ClientNotificationsDB,
    )

    app.client_notifications_db.close()
    db = ClientNotificationsDB(tmp_path / "audit-notifications.sqlite")
    app.client_notifications_db = db
    assert not db.is_memory_db
    row = db.insert_notification(
        category="research", title="Audit write", message="Message"
    )
    main_thread = threading.get_ident()
    calls = []
    ticks = []
    holder_ready = threading.Event()
    write_started = threading.Event()
    holder_done = threading.Event()
    holder_error = []
    original = db.mark_read

    def observed(notification_id, *, is_read=True):
        calls.append(dict(thread=threading.get_ident(), start=time.monotonic()))
        write_started.set()
        try:
            return original(notification_id, is_read=is_read)
        finally:
            calls[-1]["end"] = time.monotonic()

    monkeypatch.setattr(db, "mark_read", observed)

    def holder():
        try:
            with db.transaction() as cursor:
                cursor.execute(
                    "UPDATE client_notifications SET is_read = is_read WHERE id = ?",
                    (row["id"],),
                )
                holder_ready.set()
                assert write_started.wait(15)
                time.sleep(0.7)
        except BaseException as exc:
            holder_error.append(type(exc).__name__ + ": " + str(exc))
        finally:
            try:
                db.close()
            except BaseException as exc:
                holder_error.append("close: " + str(exc))
            holder_done.set()

    stop = False

    async def heartbeat():
        while not stop:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    host = DestinationHarness(app, "watchlists_collections")
    thread = None
    beat = None
    record = dict(
        contended=contended,
        main_thread=main_thread,
        calls=calls,
        file_backed=not db.is_memory_db,
        db_path=str(db.db_path),
    )
    try:
        async with host.run_test(size=(180, 50)) as pilot:
            screen = host.screen_stack[-1]
            await until(lambda: bool(screen.query("#wl-tab-notifications")))
            screen.query_one("#wl-tab-notifications", Button).press()
            await until(
                lambda: bool(screen.query("#watchlists-notifications-pane"))
                and len(
                    screen.query_one(
                        "#watchlists-notifications-pane", NotificationsPane
                    ).notifications
                )
                == 1
            )
            await pilot.pause()
            pane = screen.query_one("#watchlists-notifications-pane", NotificationsPane)
            pane.query_one("#notifications-table", DataTable).focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            button = pane.query_one("#notifications-mark-read-button", Button)
            assert not button.disabled
            beat = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.05)
            if contended:
                thread = threading.Thread(target=holder)
                thread.start()
                await until(holder_ready.is_set)
            button.press()
            await until(lambda: bool(calls) and "end" in calls[-1])
            await asyncio.sleep(0.08)
            record["max_heartbeat_gap"] = max(b - a for a, b in zip(ticks, ticks[1:]))
            record["write_seconds"] = calls[-1]["end"] - calls[-1]["start"]
            record["durably_read"] = db.get_notification(row["id"])["is_read"]
            record["holder_error"] = holder_error
            assert calls[-1]["thread"] == main_thread and record["durably_read"]
            if contended:
                assert (
                    record["write_seconds"] >= 0.65
                    and record["max_heartbeat_gap"] >= 0.65
                )
            else:
                # Ordinary mutation correctness is not a speed guarantee.
                assert not holder_error and thread is None
    finally:
        write_started.set()
        stop = True
        if beat:
            await beat
        if thread:
            thread.join(3)
        db.close()
        (OUT / ("notification-contention-" + str(contended) + ".json")).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
