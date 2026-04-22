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

    stored = store.list(include_dismissed=True, limit=10)[0]
    assert stored["title"] == "Deleted source"
    assert stored["is_read"] is False


def test_notification_dispatch_service_falls_back_to_notify_when_toast_raises(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    service = NotificationDispatchService(store=store)
    class AppWithFailingToast:
        def __init__(self) -> None:
            self.show_toast = MagicMock(side_effect=RuntimeError("toast failed"))
            self.notify = MagicMock()

    app = AppWithFailingToast()

    row = service.dispatch(
        app=app,
        category="watchlists",
        title="Deleted source",
        message="Source deleted with restore window.",
        severity="warning",
        source_backend="server",
    )

    assert row["category"] == "watchlists"
    app.show_toast.assert_called_once()
    app.notify.assert_called_once_with("Source deleted with restore window.", severity="warning", timeout=None)


def test_notification_dispatch_service_survives_notify_failure_and_missing_hooks(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    service = NotificationDispatchService(store=store)

    class AppWithFailingNotify:
        def __init__(self) -> None:
            self.notify = MagicMock(side_effect=RuntimeError("notify failed"))

    class AppWithoutNotificationHooks:
        pass

    failing_notify_app = AppWithFailingNotify()
    row = service.dispatch(
        app=failing_notify_app,
        category="watchlists",
        title="Deleted source",
        message="Source deleted with restore window.",
        severity="error",
        source_backend="server",
    )
    assert row["category"] == "watchlists"
    failing_notify_app.notify.assert_called_once()

    no_hook_app = AppWithoutNotificationHooks()
    row = service.dispatch(
        app=no_hook_app,
        category="watchlists",
        title="Deleted source",
        message="Source deleted with restore window.",
        severity="info",
        source_backend="server",
    )
    assert row["category"] == "watchlists"
