from tldw_chatbook.Notifications import ClientNotificationsDB


def test_client_notifications_store_round_trips_read_and_dismiss(tmp_path):
    db = ClientNotificationsDB(tmp_path / "notifications.db")

    row = db.insert_notification(
        category="watchlists",
        title="Deleted source",
        message="Source deleted with restore window.",
        severity="information",
        source_backend="server",
        source_entity_id="17",
        source_entity_kind="watchlist_source",
        payload={"restore_window_seconds": 10},
    )

    assert row["is_read"] is False
    assert row["is_dismissed"] is False
    assert row["payload"] == {"restore_window_seconds": 10}

    assert db.mark_read(row["id"], is_read=True) is True
    assert db.dismiss_notification(row["id"], is_dismissed=True) is True

    stored = db.list_notifications(limit=10, include_dismissed=True)[0]
    assert stored["is_read"] is True
    assert stored["is_dismissed"] is True
    assert stored["read_at"] is not None
    assert stored["dismissed_at"] is not None


def test_client_notifications_store_filters_dismissed_by_default(tmp_path):
    db = ClientNotificationsDB(tmp_path / "notifications.db")
    kept = db.insert_notification(category="watchlists", title="A", message="A")
    dismissed = db.insert_notification(category="watchlists", title="B", message="B")

    db.dismiss_notification(dismissed["id"], is_dismissed=True)

    rows = db.list_notifications(limit=10)
    assert [row["id"] for row in rows] == [kept["id"]]


def test_client_notifications_store_can_restore_dismissed_rows(tmp_path):
    db = ClientNotificationsDB(tmp_path / "notifications.db")
    row = db.insert_notification(category="watchlists", title="A", message="A")

    db.dismiss_notification(row["id"], is_dismissed=True)
    db.dismiss_notification(row["id"], is_dismissed=False)

    stored = db.get_notification(row["id"])
    assert stored["is_dismissed"] is False
    assert stored["dismissed_at"] is None


def test_client_notifications_memory_store_keeps_state_across_operations():
    db = ClientNotificationsDB(":memory:")

    row = db.insert_notification(category="watchlists", title="A", message="A")

    assert db.get_notification(row["id"])["title"] == "A"
    db.close()
