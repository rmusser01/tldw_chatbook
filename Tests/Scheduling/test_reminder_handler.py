import pytest
from unittest.mock import Mock

from tldw_chatbook.Scheduling.scheduler.handlers.reminder_handler import ReminderHandler
from tldw_chatbook.Notifications.notification_dispatch_service import (
    NotificationDispatchService,
)


@pytest.fixture
def handler():
    return ReminderHandler(dispatch_service=Mock())


@pytest.mark.asyncio
async def test_reminder_handler_dispatches_notification(handler):
    await handler.handle({"id": "1", "title": "T", "body": "B", "link_type": None})
    handler.dispatch_service.dispatch.assert_called_once_with(
        app=None,
        category="reminder",
        title="T",
        message="B",
        source_entity_kind="scheduled_task",
        source_entity_id="1",
    )


@pytest.mark.asyncio
async def test_reminder_handler_uses_default_title_when_missing(handler):
    await handler.handle({"id": "2", "body": "B"})
    handler.dispatch_service.dispatch.assert_called_once_with(
        app=None,
        category="reminder",
        title="Reminder",
        message="B",
        source_entity_kind="scheduled_task",
        source_entity_id="2",
    )


@pytest.mark.asyncio
async def test_reminder_handler_uses_empty_message_when_body_missing(handler):
    await handler.handle({"id": "3", "title": "T"})
    handler.dispatch_service.dispatch.assert_called_once_with(
        app=None,
        category="reminder",
        title="T",
        message="",
        source_entity_kind="scheduled_task",
        source_entity_id="3",
    )


@pytest.mark.asyncio
async def test_reminder_handler_uses_empty_message_when_body_is_none(handler):
    await handler.handle({"id": "4", "title": "T", "body": None})
    handler.dispatch_service.dispatch.assert_called_once_with(
        app=None,
        category="reminder",
        title="T",
        message="",
        source_entity_kind="scheduled_task",
        source_entity_id="4",
    )


@pytest.mark.asyncio
async def test_reminder_handler_allows_missing_id(handler):
    await handler.handle({"title": "T", "body": "B"})
    handler.dispatch_service.dispatch.assert_called_once_with(
        app=None,
        category="reminder",
        title="T",
        message="B",
        source_entity_kind="scheduled_task",
        source_entity_id=None,
    )


@pytest.mark.asyncio
async def test_reminder_handler_is_callable(handler):
    await handler({"id": "5", "title": "T", "body": "B"})
    handler.dispatch_service.dispatch.assert_called_once_with(
        app=None,
        category="reminder",
        title="T",
        message="B",
        source_entity_kind="scheduled_task",
        source_entity_id="5",
    )


@pytest.mark.asyncio
async def test_reminder_handler_passes_app_from_getter():
    app = object()
    service = Mock()
    handler = ReminderHandler(dispatch_service=service, app_getter=lambda: app)
    await handler.handle({"id": "5", "title": "T", "body": "B"})
    assert service.dispatch.call_args.kwargs["app"] is app


@pytest.mark.asyncio
async def test_reminder_handler_tolerates_getter_returning_none():
    service = Mock()
    handler = ReminderHandler(dispatch_service=service, app_getter=lambda: None)
    await handler.handle({"id": "6", "title": "T"})
    assert service.dispatch.call_args.kwargs["app"] is None


class _FakeNotificationStore:
    """Minimal store double: records inserts, reports notifications enabled."""

    def __init__(self):
        self.inserted = []

    def insert_notification(self, **kwargs):
        self.inserted.append(kwargs)
        return dict(kwargs)

    def get_settings(self):
        return {"enabled": True, "toast_enabled": True, "persist_enabled": True}


class _FakeApp:
    """App double exposing only ``notify`` (no ``show_toast``), matching
    ``show_notification``'s real fallback contract."""

    def __init__(self):
        self.notify_calls = []

    def notify(self, message, severity="information", timeout=None):
        self.notify_calls.append(
            {"message": message, "severity": severity, "timeout": timeout}
        )


@pytest.mark.asyncio
async def test_reminder_handler_integration_persists_and_toasts_through_real_dispatch_service():
    """End-to-end through the real NotificationDispatchService (Qodo finding):
    the inbox row must be persisted AND the toast path must fire when an app
    is available, while the no-app case still persists without a toast."""
    store = _FakeNotificationStore()
    service = NotificationDispatchService(store=store)
    app = _FakeApp()
    handler = ReminderHandler(dispatch_service=service, app_getter=lambda: app)

    await handler.handle({"id": "int-1", "title": "Pay rent", "body": "Due today"})

    assert len(store.inserted) == 1
    assert store.inserted[0]["title"] == "Pay rent"
    assert store.inserted[0]["message"] == "Due today"

    assert len(app.notify_calls) == 1
    assert "Pay rent" in app.notify_calls[0]["message"]

    # No-app case: still persists, but never attempts toast delivery.
    handler_no_app = ReminderHandler(dispatch_service=service, app_getter=None)
    await handler_no_app.handle({"id": "int-2", "title": "No app", "body": "Body"})

    assert len(store.inserted) == 2
    assert store.inserted[1]["title"] == "No app"
    assert len(app.notify_calls) == 1  # unchanged
