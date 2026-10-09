"""Regression coverage for chat message artifact actions."""

import pytest
from textual.widgets import Button

from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage
from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced


pytestmark = [pytest.mark.asyncio, pytest.mark.ui]


@pytest.mark.parametrize("widget_class", [ChatMessage, ChatMessageEnhanced])
async def test_ai_messages_expose_save_artifact_action(widget_pilot, widget_class):
    async with await widget_pilot(
        widget_class,
        message="Assistant response",
        role="Assistant",
        generation_complete=True,
    ) as pilot:
        widget = pilot.app.test_widget
        await pilot.pause()

        save_button = widget.query_one("#save-artifact", Button)
        assert save_button is not None
        assert "artifact-button" in save_button.classes
        assert save_button.tooltip == "Save response as Chatbook artifact"


@pytest.mark.parametrize("widget_class", [ChatMessage, ChatMessageEnhanced])
async def test_user_messages_do_not_expose_save_artifact_action(
    widget_pilot, widget_class
):
    async with await widget_pilot(
        widget_class,
        message="User prompt",
        role="User",
    ) as pilot:
        widget = pilot.app.test_widget
        await pilot.pause()

        assert not widget.query("#save-artifact")


@pytest.mark.parametrize("widget_class", [ChatMessage, ChatMessageEnhanced])
async def test_system_messages_do_not_expose_save_artifact_action(
    widget_pilot, widget_class
):
    async with await widget_pilot(
        widget_class,
        message="System status",
        role="System",
    ) as pilot:
        widget = pilot.app.test_widget
        await pilot.pause()

        assert widget.has_class("-ai")
        assert not widget.query("#save-artifact")


@pytest.mark.parametrize("widget_class", [ChatMessage, ChatMessageEnhanced])
async def test_legacy_chunk_entry_point_removed(widget_pilot, widget_class):
    """The watcher-less streaming mutation entry is gone on both classes.

    ``update_message_chunk`` appended to ``message_text`` -- a reactive
    with no watcher, so the appended text never reached the composed
    Markdown body. It had zero production callers; the ``ChatMessageEnhanced``
    copy was removed in TASK-34433 and this file's sibling ``ChatMessage``
    copy in TASK-34437 (the classes are checked together so the pattern
    cannot survive on one seam again).
    """
    async with await widget_pilot(
        widget_class,
        message="",
        role="Assistant",
        generation_complete=False,
    ) as pilot:
        widget = pilot.app.test_widget
        await pilot.pause()

        assert not hasattr(widget, "update_message_chunk")
        # The reactive still carries the message composed at mount time.
        assert widget.message_text == ""
