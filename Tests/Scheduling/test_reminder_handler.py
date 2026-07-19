import pytest
from unittest.mock import Mock

from tldw_chatbook.Scheduling.scheduler.handlers.reminder_handler import ReminderHandler


@pytest.mark.asyncio
async def test_reminder_handler_dispatches_notification():
    dispatch = Mock()
    handler = ReminderHandler(dispatch_service=dispatch)
    await handler.handle({"id": "1", "title": "T", "body": "B", "link_type": None})
    dispatch.dispatch.assert_called_once()
