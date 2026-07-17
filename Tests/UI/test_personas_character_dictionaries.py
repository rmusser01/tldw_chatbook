"""P1f: the I/O-free character dictionaries panel."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_dictionaries import (
    PersonasCharacterDictionariesWidget,
    CharacterDictionaryAttachRequested,
    CharacterDictionaryDetachRequested,
)

pytestmark = pytest.mark.asyncio


class _Host(App):
    def compose(self) -> ComposeResult:
        yield PersonasCharacterDictionariesWidget()


async def test_empty_state_when_no_dictionaries():
    async with _Host().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        panel.load_character_dictionaries([])
        await pilot.pause()
        empty = pilot.app.query_one("#personas-char-dicts-empty", Static)
        assert empty.display is True
        assert pilot.app.query_one("#personas-char-dicts-table", DataTable).display is False


async def test_load_renders_rows():
    async with _Host().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        panel.load_character_dictionaries([{"name": "Slang", "entry_count": 2, "enabled": True}])
        await pilot.pause()
        table = pilot.app.query_one("#personas-char-dicts-table", DataTable)
        assert table.row_count == 1
        assert "Slang" in str(table.get_cell_at((0, 0)))


async def test_attach_button_posts_intent():
    posted = []

    class _CaptureHost(_Host):
        def on_character_dictionary_attach_requested(self, m: CharacterDictionaryAttachRequested):
            posted.append(m)

    async with _CaptureHost().run_test(size=(120, 40)) as pilot:
        await pilot.click("#personas-char-dicts-add")
        await pilot.pause()
    assert len(posted) == 1


async def test_detach_button_posts_intent_with_name():
    posted = []

    class _CaptureHost(_Host):
        def on_character_dictionary_detach_requested(self, m: CharacterDictionaryDetachRequested):
            posted.append(m.dictionary_name)

    async with _CaptureHost().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        panel.load_character_dictionaries([{"name": "Slang", "entry_count": 1, "enabled": True}])
        await pilot.pause()
        pilot.app.query_one("#personas-char-dicts-table", DataTable).move_cursor(row=0)
        await pilot.click("#personas-char-dicts-detach")
        await pilot.pause()
    assert posted == ["Slang"]


async def test_duplicate_named_rows_do_not_crash_and_dedup_to_one_row():
    """A hostile/crafted import can produce two same-named embedded blocks.

    ``DataTable.add_row(..., key=str(name))`` would raise ``DuplicateKey`` on
    the second row if the panel didn't dedup first — and that exception would
    propagate uncaught through the import worker (default ``exit_on_error``)
    and exit the whole app. The panel must dedup by name (first wins) so this
    can never happen, regardless of what the screen feeds it.
    """
    async with _Host().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        panel.load_character_dictionaries([
            {"name": "Slang", "entry_count": 1, "enabled": True},
            {"name": "Slang", "entry_count": 2, "enabled": True},
        ])
        await pilot.pause()
        table = pilot.app.query_one("#personas-char-dicts-table", DataTable)
        assert table.row_count == 1
