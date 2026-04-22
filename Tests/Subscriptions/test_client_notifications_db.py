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
