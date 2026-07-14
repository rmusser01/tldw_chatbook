"""P1e: the dedicated (non-TTS) conversation attach picker returns string ids."""

import pytest
from textual.app import App
from textual.widgets import Input, ListView

from tldw_chatbook.Widgets.Persona_Widgets.dictionary_attach_picker import DictionaryAttachPicker

pytestmark = pytest.mark.asyncio

CONVS = [
    {"conversation_id": "conv-uuid-a", "title": "Noir case"},
    {"conversation_id": "conv-uuid-b", "title": "Lab notes"},
]


class _Host(App):
    def __init__(self):
        super().__init__()
        self.result = "unset"

    def on_mount(self):
        self.run_worker(self._drive)

    async def _drive(self):
        self.result = await self.push_screen_wait(DictionaryAttachPicker(list(CONVS)))


async def test_picker_returns_selected_string_id():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        picker = app.screen
        picker.query_one("#dict-attach-list", ListView).index = 1  # Lab notes
        await pilot.pause()
        await pilot.click("#dict-attach-confirm")
        await pilot.pause()
    assert app.result == "conv-uuid-b"          # string id, not int


async def test_picker_search_filters():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        picker = app.screen
        picker.query_one("#dict-attach-search", Input).value = "noir"
        await pilot.pause()
        rows = picker.query_one("#dict-attach-list", ListView).children
        assert len(rows) == 1
        # select the only match, confirm
        picker.query_one("#dict-attach-list", ListView).index = 0
        await pilot.pause()
        await pilot.click("#dict-attach-confirm")
        await pilot.pause()
    assert app.result == "conv-uuid-a"


async def test_picker_cancel_returns_none():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#dict-attach-cancel")
        await pilot.pause()
    assert app.result is None
