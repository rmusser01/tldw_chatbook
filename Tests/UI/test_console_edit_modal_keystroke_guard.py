"""TASK-360: the edit modal must not absorb keystrokes typed before it opened.

Pressing `e` on a selected message dispatches through a Button.Pressed hop
and an async modal push; a second `e` pressed in that gap used to land as
text in the late-opening TextArea, silently corrupting the draft (review
finding j6-edit-modal-late-open-keystroke-leak). Keys whose event time
predates the modal's mount are swallowed — the user couldn't have meant
them as text in a modal they couldn't see.
"""

import pytest
from textual.app import App
from textual.events import Key
from textual.widgets import TextArea

from tldw_chatbook.Widgets.Console.console_edit_message_modal import (
    ConsoleEditMessageModal,
)


class _ModalHost(App):
    pass


@pytest.mark.asyncio
async def test_edit_modal_swallows_keys_typed_before_it_opened():
    app = _ModalHost()
    async with app.run_test() as pilot:
        modal = ConsoleEditMessageModal(content="original text")
        app.push_screen(modal)
        await pilot.pause()
        area = modal.query_one("#console-edit-message-body", TextArea)

        stale = Key(key="e", character="e")
        stale.time = modal._opened_at - 5.0
        area.post_message(stale)
        await pilot.pause()
        await pilot.pause()

        assert area.text == "original text"


@pytest.mark.asyncio
async def test_edit_modal_accepts_keys_typed_after_it_opened():
    app = _ModalHost()
    async with app.run_test() as pilot:
        modal = ConsoleEditMessageModal(content="")
        app.push_screen(modal)
        await pilot.pause()
        area = modal.query_one("#console-edit-message-body", TextArea)

        await pilot.press("h", "i")

        assert area.text == "hi"
