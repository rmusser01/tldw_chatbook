"""P1f: the dictionary picker returns int dictionary ids."""

import pytest
from textual.app import App
from textual.widgets import Input, ListView

from tldw_chatbook.Widgets.Persona_Widgets.dictionary_picker import DictionaryPicker

pytestmark = pytest.mark.asyncio

DICTS = [
    {"dictionary_id": 3, "name": "Slang"},
    {"dictionary_id": 7, "name": "Period Vocab"},
]


class _Host(App):
    def __init__(self):
        super().__init__()
        self.result = "unset"

    def on_mount(self):
        self.run_worker(self._drive)

    async def _drive(self):
        self.result = await self.push_screen_wait(DictionaryPicker(list(DICTS)))


async def test_picker_returns_selected_int_id():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        picker = app.screen
        picker.query_one("#dict-pick-list", ListView).index = 1  # Period Vocab
        await pilot.pause()
        await pilot.click("#dict-pick-confirm")
        await pilot.pause()
    assert app.result == 7
    assert isinstance(app.result, int)


async def test_picker_search_filters_by_name():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        picker = app.screen
        picker.query_one("#dict-pick-search", Input).value = "slang"
        await pilot.pause()
        assert len(picker.query_one("#dict-pick-list", ListView).children) == 1
        picker.query_one("#dict-pick-list", ListView).index = 0
        await pilot.pause()
        await pilot.click("#dict-pick-confirm")
        await pilot.pause()
    assert app.result == 3


async def test_picker_cancel_returns_none():
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#dict-pick-cancel")
        await pilot.pause()
    assert app.result is None
