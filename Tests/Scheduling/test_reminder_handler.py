import pytest
from unittest.mock import Mock

from tldw_chatbook.Scheduling.scheduler.handlers.reminder_handler import ReminderHandler


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
