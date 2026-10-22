"""Roleplay journeys, one test per journey a frame slice touches (spec 5.7.3).

From frame slice B1: Edit -> type -> the header's unsaved chip appears ->
Ctrl+S -> the chip goes (B1 AC#5). Under both styled tiers, driven like a
user: a click on the card's Edit, keystrokes, Ctrl+S. Since frame slice B2a
both tiers run on a real temporary ChaChaNotes, so the journey ends by
reading the save back. Later slices add the spec's J1-J4 journeys here, each
as a passing prefix test plus a strict ``xfail(raises=NotYetDelivered)``
full-journey test (RC-7).
"""

from __future__ import annotations

import pytest
from textual.widgets import TextArea

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.UI.roleplay_frame_harness import (
    open_styled_roleplay,
    settle,
    styled_tiers,
    wait_until,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Workbench.workbench_widgets import FittedText

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@styled_tiers
async def test_editing_shows_the_unsaved_chip_and_saving_clears_it(
    styled_tier, mock_app_instance, tmp_path
):
    """B1 AC#5 on a real database (frame slice B2a carry-over): Edit, type,
    the chip shows, Ctrl+S, the chip goes; after the editor closes and the
    character loads again, the card shows the saved description, read back
    from the database."""
    db = CharactersRAGDB(tmp_path / "journey.db", "roleplay-journey")
    # "Abel ..." sorts before the built-in "Default Assistant", so the
    # first-paint auto-selection (F-031) loads it without a click.
    char_id = db.add_character_card({"name": "Abel Detective", "description": "Noir"})
    try:
        async with open_styled_roleplay(
            styled_tier, mock_app_instance, size=(120, 36), character_db=db
        ) as pilot:
            screen = pilot.app.screen
            chip = screen.query_one("#personas-header-unsaved", FittedText)
            item = screen.query_one("#personas-header-item", FittedText)
            await wait_until(
                pilot,
                lambda: screen.state.selected_entity_id == str(char_id),
                what="the auto-selected character",
            )
            assert not chip.display

            assert await pilot.click("#personas-card-edit-character")
            await wait_until(pilot, lambda: screen._edit_mode == "edit", what="Edit")
            assert item.value == ("Abel Detective", True)
            assert not chip.display  # opening the editor is not an edit

            screen.query_one("#personas-char-editor-description", TextArea).focus()
            await pilot.pause()
            await pilot.press("x")
            await wait_until(pilot, lambda: chip.display, what="the unsaved chip")
            assert chip.value == "Unsaved changes"

            await pilot.press("ctrl+s")
            await settle(pilot)
            await wait_until(pilot, lambda: not chip.display, what="the chip to go")
            assert screen._edit_mode == "edit"  # save-in-place keeps the editor
            # Save -> reload: leave the editor and load the character again
            # through the screen's own select path (the one a row click takes),
            # which re-reads the record from the database. The card shows the
            # edit, and the database holds it.
            await pilot.press("escape")
            await wait_until(
                pilot, lambda: screen._edit_mode == "view", what="the card"
            )
            await screen._select_character(str(char_id), "Abel Detective")
            await settle(pilot)
            card = str(screen.query_one("#personas-character-card-description").content)
            saved = character_handler_module.fetch_character_by_id(char_id)
            assert "x" in card and card == f"Description: {saved['description']}"
            assert "x" in saved["description"] and saved["description"] != "Noir"
            assert db.get_character_card_by_id(char_id)["version"] > 1
    finally:
        db.close_connection()
