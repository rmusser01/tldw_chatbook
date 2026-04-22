from unittest.mock import MagicMock

from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Notifications.notification_dispatch_service import NotificationDispatchService


def test_notification_dispatch_service_uses_toast_when_available(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    service = NotificationDispatchService(store=store)
    app = MagicMock()
    app.show_toast = MagicMock()

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
    app.show_toast.assert_called_once_with(
        message="Source deleted with restore window.",
        severity="info",
        timeout=5.0,
        persistent=False,
    )
    app.notify.assert_not_called()

    stored = store.list(limit=10)[0]
    assert stored["title"] == "Deleted source"
    assert stored["is_read"] is False


def test_notification_dispatch_service_falls_back_to_notify_without_toast(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    service = NotificationDispatchService(store=store)
    class AppWithoutToast:
        def __init__(self) -> None:
            self.notify = MagicMock()

    app = AppWithoutToast()

    service.dispatch(
        app=app,
        category="watchlists",
        title="Deleted source",
        message="Source deleted with restore window.",
        severity="warning",
        source_backend="server",
    )

    app.notify.assert_called_once_with("Source deleted with restore window.", severity="warning", timeout=None)
