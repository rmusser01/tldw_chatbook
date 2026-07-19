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


async def test_picker_stale_selection_after_filter_is_not_returned():
    """Regression for Roleplay P1e final-review #3.

    Select a row from the full list, then narrow the search to a DIFFERENT
    conversation without re-selecting. Confirm must not silently reuse the
    old highlighted index against the rebuilt (differently-mapped) row set;
    it must require an explicit re-select instead.
    """
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        picker = app.screen
        # Select row 0 of the FULL 2-row list ("Noir case").
        picker.query_one("#dict-attach-list", ListView).index = 0
        await pilot.pause()
        # Narrow the list to the OTHER conversation only ("Lab notes").
        picker.query_one("#dict-attach-search", Input).value = "lab"
        await pilot.pause()
        rows = picker.query_one("#dict-attach-list", ListView).children
        assert len(rows) == 1
        # Confirm WITHOUT re-selecting: no stale index should carry over.
        await pilot.click("#dict-attach-confirm")
        await pilot.pause()
    assert app.result is None


async def test_picker_cancel_returns_none():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#dict-attach-cancel")
        await pilot.pause()
    assert app.result is None
