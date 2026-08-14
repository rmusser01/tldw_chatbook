"""P1f: the I/O-free character dictionaries panel."""

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, DataTable, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_dictionaries import (
    PersonasCharacterDictionariesWidget,
    CharacterDictionaryAttachRequested,
    CharacterDictionaryDetachRequested,
)

pytestmark = pytest.mark.asyncio


class _Host(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield PersonasCharacterDictionariesWidget()


async def test_empty_state_when_no_dictionaries():
    async with _Host().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        panel.load_character_dictionaries([])
        await pilot.pause()
        empty = pilot.app.query_one("#personas-char-dicts-empty", Static)
        assert empty.display is True
        assert (
            pilot.app.query_one("#personas-char-dicts-table", DataTable).display
            is False
        )


async def test_load_renders_rows():
    async with _Host().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        panel.load_character_dictionaries(
            [{"name": "Slang", "entry_count": 2, "enabled": True}]
        )
        await pilot.pause()
        table = pilot.app.query_one("#personas-char-dicts-table", DataTable)
        assert table.row_count == 1
        assert "Slang" in str(table.get_cell_at((0, 0)))


async def test_attach_button_posts_intent():
    posted = []

    class _CaptureHost(_Host):
        def on_character_dictionary_attach_requested(
            self, m: CharacterDictionaryAttachRequested
        ):
            posted.append(m)

    async with _CaptureHost().run_test(size=(120, 40)) as pilot:
        # The section starts collapsed (task-2231): expand to reach Attach.
        await pilot.click("#personas-char-dicts-toggle")
        await pilot.pause()
        await pilot.click("#personas-char-dicts-add")
        await pilot.pause()
    assert len(posted) == 1


async def test_detach_button_posts_intent_with_name():
    posted = []

    class _CaptureHost(_Host):
        def on_character_dictionary_detach_requested(
            self, m: CharacterDictionaryDetachRequested
        ):
            posted.append(m.dictionary_name)

    async with _CaptureHost().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        panel.load_character_dictionaries(
            [{"name": "Slang", "entry_count": 1, "enabled": True}]
        )
        await pilot.pause()
        # The section starts collapsed (task-2231): expand to reach Detach.
        await pilot.click("#personas-char-dicts-toggle")
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
        panel.load_character_dictionaries(
            [
                {"name": "Slang", "entry_count": 1, "enabled": True},
                {"name": "Slang", "entry_count": 2, "enabled": True},
            ]
        )
        await pilot.pause()
        table = pilot.app.query_one("#personas-char-dicts-table", DataTable)
        assert table.row_count == 1


# ===================================================================
# task-2231: the panel is a collapsible section - one line when empty,
# count in the header, and a collapse state that data reloads never reset.
# ===================================================================


async def test_collapsed_by_default_renders_one_line():
    """AC#4: an empty section costs exactly one line, not 16."""
    async with _Host().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        await pilot.pause()
        assert panel.query_one("#personas-char-dicts-body").display is False
        assert (
            str(panel.query_one("#personas-char-dicts-toggle", Button).label)
            == "▸ Dictionaries (0)"
        )
        assert panel.size.height == 1


async def test_toggle_expands_and_collapses_in_place():
    async with _Host().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        await pilot.pause()
        toggle = panel.query_one("#personas-char-dicts-toggle", Button)

        await pilot.click("#personas-char-dicts-toggle")
        await pilot.pause()
        assert panel.query_one("#personas-char-dicts-body").display is True
        assert str(toggle.label) == "▾ Dictionaries (0)"

        # Button presses are debounced for the -active animation window
        # (Textual ignores a re-click while it lasts); wait it out like a
        # real user's second click would.
        await pilot.pause(0.4)
        await pilot.click("#personas-char-dicts-toggle")
        await pilot.pause()
        assert panel.query_one("#personas-char-dicts-body").display is False
        assert str(toggle.label) == "▸ Dictionaries (0)"


async def test_header_count_tracks_rows_without_resetting_collapse_state():
    """AC#3: attach/detach refreshes update the count but keep the user's
    expand/collapse choice (session persistence)."""
    async with _Host().run_test(size=(120, 40)) as pilot:
        panel = pilot.app.query_one(PersonasCharacterDictionariesWidget)
        await pilot.pause()
        toggle = panel.query_one("#personas-char-dicts-toggle", Button)

        panel.load_character_dictionaries(
            [
                {"name": "Slang", "entry_count": 2, "enabled": True},
                {"name": "Cant", "entry_count": 5, "enabled": True},
            ]
        )
        await pilot.pause()
        assert str(toggle.label) == "▸ Dictionaries (2)"
        assert panel.query_one("#personas-char-dicts-body").display is False

        await pilot.click("#personas-char-dicts-toggle")
        await pilot.pause()
        panel.load_character_dictionaries(
            [{"name": "Slang", "entry_count": 2, "enabled": True}]
        )
        await pilot.pause()
        assert str(toggle.label) == "▾ Dictionaries (1)"
        assert panel.query_one("#personas-char-dicts-body").display is True
