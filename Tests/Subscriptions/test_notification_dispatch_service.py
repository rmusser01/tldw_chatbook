from unittest.mock import MagicMock

from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Notifications.notification_dispatch_service import NotificationDispatchService


def test_notification_dispatch_service_persists_and_notifies_app(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    service = NotificationDispatchService(store=store)
    app = MagicMock()

    row = service.dispatch(
        app=app,
        category="watchlists",
        title="Deleted source",
        message="Source deleted with restore window.",
        severity="info",
        source_backend="server",
        source_entity_id="17",
        source_entity_kind="watchlist_source",
        payload={"restore_window_seconds": 10},
    )

    assert row["category"] == "watchlists"
    assert row["payload"] == {"restore_window_seconds": 10}
    app.notify.assert_called_once_with("Source deleted with restore window.", severity="information")

    stored = store.list_notifications(limit=10)[0]
    assert stored["title"] == "Deleted source"
    assert stored["is_read"] is False
