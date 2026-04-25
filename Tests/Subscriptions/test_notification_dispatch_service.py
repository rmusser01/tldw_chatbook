from types import SimpleNamespace
from unittest.mock import Mock

from tldw_chatbook.Notifications import ClientNotificationsDB, NotificationDispatchService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


def test_notification_dispatch_service_persists_then_notifies(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    app = SimpleNamespace(notify=Mock())
    service = NotificationDispatchService(store=store)

    row = service.dispatch(
        app=app,
        category="watchlists",
        title="Watchlist source deleted",
        message="Server source deleted within restore window.",
        severity="warning",
        source_backend="server",
        source_entity_kind="watchlist_source",
        source_entity_id="17",
        payload={"restore_window_seconds": 10},
    )

    stored = store.get_notification(row["id"])
    assert stored["payload"]["restore_window_seconds"] == 10
    assert stored["source_backend"] == "server"
    app.notify.assert_called_once_with(
        "Watchlist source deleted: Server source deleted within restore window.",
        severity="warning",
        timeout=None,
    )


def test_notification_dispatch_service_uses_toast_when_available(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    app = SimpleNamespace(show_toast=Mock(), notify=Mock())
    service = NotificationDispatchService(store=store)

    service.dispatch(
        app=app,
        category="subscriptions",
        title="New items",
        message="Three new items found.",
        severity="information",
        timeout=2.0,
    )

    app.show_toast.assert_called_once_with(
        message="New items: Three new items found.",
        severity="info",
        timeout=2.0,
        persistent=False,
    )
    app.notify.assert_not_called()


def test_notification_dispatch_service_enforces_policy_before_insert(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    policy = Mock()
    service = NotificationDispatchService(store=store, policy_enforcer=policy)

    service.dispatch(
        category="watchlists",
        title="Source saved",
        message="Saved.",
    )

    policy.require_allowed.assert_called_once_with(
        action_id="notifications.dispatch.launch.local"
    )
    assert len(store.list_notifications(limit=10)) == 1


def test_notification_dispatch_service_hard_stops_denied_ui_policy(tmp_path):
    store = ClientNotificationsDB(tmp_path / "notifications.db")
    policy = SimpleNamespace(
        require_ui_action_allowed=Mock(
            return_value=PolicyDecision(
                allowed=False,
                reason_code="server_unreachable",
                user_message="Blocked.",
                effective_source="local",
                authority_owner="local",
            )
        )
    )
    service = NotificationDispatchService(store=store, policy_enforcer=policy)

    try:
        service.dispatch(category="watchlists", title="Blocked", message="Blocked.")
    except PolicyDeniedError as exc:
        assert exc.reason_code == "server_unreachable"
    else:
        raise AssertionError("Expected PolicyDeniedError")

    assert store.list_notifications(limit=10) == []
