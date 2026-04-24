from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB


def test_client_notifications_store_round_trips_read_and_dismiss(tmp_path):
    db = ClientNotificationsDB(tmp_path / "notifications.db")
    row = db.insert(
        category="watchlists",
        title="Deleted source",
        message="Source deleted with restore window.",
        severity="info",
        source_backend="server",
        source_entity_id="17",
        source_entity_kind="watchlist_source",
        payload={"restore_window_seconds": 10},
    )
    db.mark_read(row["id"], is_read=True)
    db.dismiss(row["id"], is_dismissed=True)
    assert db.list(limit=10) == []

    stored = db.list(include_dismissed=True, limit=10)[0]
    assert stored["is_read"] is True
    assert stored["is_dismissed"] is True


def test_client_notifications_store_filters_queue_rows(tmp_path):
    db = ClientNotificationsDB(tmp_path / "notifications.db")
    first = db.insert(
        category="watchlists",
        title="Watchlist alert",
        message="Remote source changed.",
        severity="warning",
        source_backend="server",
        source_entity_kind="watchlist_source",
    )
    db.insert(
        category="research",
        title="Local run complete",
        message="Research bundle is ready.",
        severity="info",
        source_backend="local",
        source_entity_kind="research_run",
    )
    db.mark_read(first["id"], is_read=True)

    rows = db.list_notifications(
        limit=10,
        category="watchlists",
        source_backend="server",
        severity="warning",
        is_read=True,
    )

    assert [row["title"] for row in rows] == ["Watchlist alert"]
    assert db.list_notifications(limit=10, category="research", is_read=True) == []


def test_client_notifications_store_round_trips_delivery_preferences(tmp_path):
    db = ClientNotificationsDB(tmp_path / "notifications.db")

    assert db.get_preferences() == {
        "delivery_enabled": True,
        "muted_categories": [],
        "muted_severities": [],
    }

    updated = db.update_preferences(
        delivery_enabled=False,
        muted_categories=["watchlists", "research"],
        muted_severities=["warning"],
    )

    assert updated == {
        "delivery_enabled": False,
        "muted_categories": ["research", "watchlists"],
        "muted_severities": ["warning"],
    }
    assert db.get_preferences() == updated
