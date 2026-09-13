"""Compatibility-preserving lazy public exports; recovery imports no services."""

from importlib import import_module

_EXPORTS = {
    "ClientNotificationsDB": (".client_notifications_db", "ClientNotificationsDB"),
    "ClientNotificationsService": (
        ".client_notifications_service",
        "ClientNotificationsService",
    ),
    "EventReplayWindow": (".event_state_repository", "EventReplayWindow"),
    "EventRetentionPolicy": (".event_state_repository", "EventRetentionPolicy"),
    "EventStateRepository": (".event_state_repository", "EventStateRepository"),
    "NotificationsScopeService": (
        ".notifications_scope_service",
        "NotificationsScopeService",
    ),
    "NotificationDispatchService": (
        ".notification_dispatch_service",
        "NotificationDispatchService",
    ),
    "ServerNotificationEventObserver": (
        ".server_notification_events",
        "ServerNotificationEventObserver",
    ),
    "build_server_notification_feed": (
        ".server_notification_events",
        "build_server_notification_feed",
    ),
    "normalize_server_notification_event": (
        ".server_notification_events",
        "normalize_server_notification_event",
    ),
    "ServerNotificationsService": (
        ".server_notifications_service",
        "ServerNotificationsService",
    ),
}

__all__ = [
    "ClientNotificationsDB",
    "ClientNotificationsService",
    "EventRetentionPolicy",
    "EventReplayWindow",
    "EventStateRepository",
    "NotificationsScopeService",
    "NotificationDispatchService",
    "ServerNotificationEventObserver",
    "ServerNotificationsService",
    "build_server_notification_feed",
    "normalize_server_notification_event",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
