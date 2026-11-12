"""Roleplay journeys, one test per journey a frame slice touches (spec 5.7.3).

From frame slice B1: Edit -> type -> the header's unsaved chip appears ->
Ctrl+S -> the chip goes (B1 AC#5). Under both styled tiers, driven like a
user: a click on the card's Edit, keystrokes, Ctrl+S. Since frame slice B2a
both tiers run on a real temporary ChaChaNotes, so the journey ends by
reading the save back.

From frame slice B2a: J1 fast, ``/ isolde ⏎ ⏎ o`` (spec 1.6, 5.7.3), as a
passing prefix test of the steps delivered so far plus a strict
``xfail(raises=NotYetDelivered)`` full-journey test (RC-7). ``/`` is frame
slice B3's key and ``o`` is B5c's; until B3, Ctrl+F is the prefix's search key
(spec 0.4 keeps ``Ctrl+F > name > Enter > Enter``). Later slices add J1-J4.

From frame slice B2b: J1, import a PNG card and chat with it (spec 5.7.3), the
same way: a passing prefix (Import, the one picker, the card selected and
loaded, read back from the database) and a strict full-journey test whose
Chat now step is B5b-2's and whose provider request is B5c's.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static, TextArea

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.UI.roleplay_frame_harness import (
    ISOLDE,
    NotYetDelivered,
    fake_file_picker,
    highlighted_item_id,
    item_at,
    not_yet_delivered,
    open_styled_roleplay,
    open_volume_db,
    roleplay_full_app,
    select_roleplay_item,
    settle,
    styled_tiers,
    volume_db_path,
    wait_until,
)
from Tests.UI.roleplay_import_fixtures import png_card_bytes
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


# ---- J1 fast: find an existing card (frame slice B2a; spec 1.6, 5.7.3) ----


@pytest.fixture
def volume_348(tmp_path_factory, tmp_path):
    db = open_volume_db(tmp_path, volume_db_path(tmp_path_factory, 348))
    yield db
    db.close_connection()


async def _start_on_another_card(pilot, db) -> str:
    """Load ``Character 0100`` first, so the find must move the cursor and the
    selection: Isolde is row 0 of the name-sorted list, which the first-paint
    auto-selection (F-031) may already have loaded."""
    screen = pilot.app.screen
    other = str(db.get_character_card_by_name("Character 0100")["id"])
    await select_roleplay_item(pilot, screen, f"character:{other}")
    await wait_until(
        pilot,
        lambda: screen.state.selected_entity_id == other,
        what="Character 0100 loaded",
    )
    await settle(pilot)
    assert highlighted_item_id(screen) == f"character:{other}"
    return other


async def test_j1_fast_prefix_ctrl_f_isolde_enter_enter_opens_the_card(volume_348):
    """B2a AC#3 on the 348-character volume profile, on the real app: four
    keystrokes with no waits between them open Isolde's card. ``/`` arrives
    with frame slice B3; Ctrl+F is the same search box (spec 0.4)."""
    isolde = str(volume_348.get_character_card_by_name(ISOLDE)["id"])
    async with roleplay_full_app(size=(120, 36), character_db=volume_348) as pilot:
        screen = pilot.app.screen
        await _start_on_another_card(pilot, volume_348)
        await pilot.press("ctrl+f", *"isolde", "enter", "enter")
        await wait_until(
            pilot,
            lambda: screen.state.selected_entity_id == isolde,
            what="Isolde's card",
        )
        await settle(pilot)
        card_name = screen.query_one("#personas-character-card-name", Static)
        assert str(card_name.render()) == f"Name: {ISOLDE}"
        pane = screen.query_one("#personas-library-pane")
        assert str(pane.border_title) == "Characters 1 of 349"
        assert highlighted_item_id(screen) == f"character:{isolde}"


@pytest.mark.xfail(
    strict=True,
    raises=NotYetDelivered,
    reason="J1 fast: / is frame slice B3's key, o is B5c's",
)
async def test_j1_fast_full_journey_slash_isolde_enter_enter_o(volume_348):
    """``/ isolde ⏎ ⏎ o``: the card opens and ``o`` starts its chat (spec 1.6).
    Every undelivered step raises ``NotYetDelivered``; the slice that delivers
    a step replaces its raise with the step."""
    isolde = str(volume_348.get_character_card_by_name(ISOLDE)["id"])
    async with roleplay_full_app(size=(120, 36), character_db=volume_348) as pilot:
        screen = pilot.app.screen
        await _start_on_another_card(pilot, volume_348)
        screen.query_one("#personas-library-rows").focus()
        await settle(pilot)
        not_yet_delivered("/ focuses the items search from the list", "B3")
        await pilot.press("slash", *"isolde", "enter", "enter")
        await wait_until(
            pilot,
            lambda: screen.state.selected_entity_id == isolde,
            what="Isolde's card",
        )
        not_yet_delivered("o on the loaded row starts a Console chat", "B5c")


# ---- J1: import a PNG card, then chat with it (frame slice B2b; spec 5.7.3) ----

#: The PNG fixture's card name (Tests/UI/roleplay_import_fixtures.py).
J1_CARD = "Wren Halloway"


@pytest.fixture
def j1_db(tmp_path):
    """A file-backed ChaChaNotes with one character (the list needs a first row)."""
    db = CharactersRAGDB(tmp_path / "j1.db", "roleplay-journey-j1")
    db.add_character_card({"name": "Abel Host", "description": "seed"})
    yield db
    db.close_connection()


async def _import_the_png_card(pilot, monkeypatch, tmp_path) -> str:
    """Import (the one picker answers with a PNG card), then wait for it to load.

    Returns:
        The imported card's option id.
    """
    card = tmp_path / "wren.png"
    card.write_bytes(png_card_bytes())
    fake_file_picker(monkeypatch, card)
    screen = pilot.app.screen
    assert await pilot.click("#personas-library-import")
    await wait_until(
        pilot,
        lambda: screen.state.selected_entity_name == J1_CARD,
        what="the imported card loaded",
    )
    await settle(pilot)
    return f"character:{screen.state.selected_entity_id}"


async def test_j1_prefix_import_a_png_card_selects_and_loads_it(
    j1_db, monkeypatch, tmp_path
):
    """J1's delivered steps on the real app: Import, the PNG card through the
    one picker, then its row is the loaded one (the gutter's loaded glyph),
    its card shows, and the database holds it."""
    async with roleplay_full_app(size=(120, 36), character_db=j1_db) as pilot:
        screen = pilot.app.screen
        option_id = await _import_the_png_card(pilot, monkeypatch, tmp_path)
        row = item_at(screen, option_id)
        assert row.active and row.name == J1_CARD
        prompt = screen.query_one("#personas-library-rows").get_option(option_id).prompt
        assert "▸" in prompt.plain.split(J1_CARD)[0]
        card_name = screen.query_one("#personas-character-card-name", Static)
        assert str(card_name.render()) == f"Name: {J1_CARD}"
        stored = j1_db.get_character_card_by_name(J1_CARD)
        assert stored is not None
        assert option_id == f"character:{stored['id']}"


@pytest.mark.xfail(
    strict=True,
    raises=NotYetDelivered,
    reason="J1: Chat now from the card is frame slice B5b-2's, the request B5c's",
)
async def test_j1_full_journey_import_png_chat_now_provider_received_one_request(
    j1_db, monkeypatch, tmp_path
):
    """Import a PNG card, Chat now, and the provider receives one request
    (spec 5.7.3). Every undelivered step raises ``NotYetDelivered``; the slice
    that delivers a step replaces its raise with the step."""
    async with roleplay_full_app(size=(120, 36), character_db=j1_db) as pilot:
        screen = pilot.app.screen
        await _import_the_png_card(pilot, monkeypatch, tmp_path)
        not_yet_delivered("Chat now from the loaded card", "B5b-2")
        assert await pilot.click("#personas-start-chat")
        not_yet_delivered("the provider received one request", "B5c")
        assert screen.state.selected_entity_name == J1_CARD
