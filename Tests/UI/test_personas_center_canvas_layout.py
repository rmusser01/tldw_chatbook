# Tests/UI/test_personas_center_canvas_layout.py
"""Rendered-layout contract for the Roleplay center canvas (task-2231, R2).

The center canvas is ONE scrollable column: the character card fills the
viewport first, and the Dictionaries / World Books attachment sections flow
below it in document order as collapsed one-line sections the user expands
in place. This replaced the bottom-dock workaround, under which two empty
panels owned ~50% of the center, displaced the card entirely at 100x30, and
left a ~10-line dead void between the panels at 170x50.
"""

import pytest
from textual.widgets import Button, Static

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.UI.test_personas_dictionaries import patch_character_paging
from Tests.UI.test_personas_workbench import CHARACTERS, StyledPersonasTestApp
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_card_widget import (
    PersonasCharacterCardWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_dictionaries import (
    PersonasCharacterDictionariesWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_world_books import (
    PersonasCharacterWorldBooksWidget,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def stub_characters(monkeypatch):
    """Same character stubs as the workbench suite (kept local for lint)."""
    monkeypatch.setattr(
        character_handler_module,
        "fetch_all_characters",
        lambda: [dict(c) for c in CHARACTERS],
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: next(
            dict(c) for c in CHARACTERS if str(c["id"]) == str(character_id)
        ),
    )
    patch_character_paging(monkeypatch)


async def _mounted_screen(pilot):
    """First paint auto-selects the first character (F-031): card is shown."""
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return pilot.app.screen


def _center_parts(screen):
    stack = screen.query_one("#personas-detail-stack")
    card = screen.query_one(PersonasCharacterCardWidget)
    wrapper = screen.query_one("#personas-character-attachments")
    dicts = screen.query_one(PersonasCharacterDictionariesWidget)
    world_books = screen.query_one(PersonasCharacterWorldBooksWidget)
    return stack, card, wrapper, dicts, world_books


async def _force_empty_panels(pilot, dicts, world_books) -> None:
    """Pin both sections to their empty state regardless of service stubs."""
    dicts.load_character_dictionaries([])
    world_books.load_world_books([])
    await pilot.pause()


@pytest.mark.parametrize("size", [(170, 50), (100, 30)])
async def test_card_fills_viewport_with_attachments_below_in_document_order(
    mock_app_instance, stub_characters, size
):
    """AC#1/#2/#5: the card owns the whole center viewport at both supported
    sizes; the attachment sections start exactly at the card's bottom edge
    (document order, below the fold) instead of docking space away from it."""
    app = StyledPersonasTestApp(mock_app_instance)
    async with app.run_test(size=size) as pilot:
        screen = await _mounted_screen(pilot)
        stack, card, wrapper, dicts, world_books = _center_parts(screen)
        await _force_empty_panels(pilot, dicts, world_books)

        # The card fills the center viewport exactly...
        assert card.region.y == stack.region.y
        assert card.region.height == stack.region.height
        # ...with a real floor under it (AC#5: never squeezed to a sliver).
        assert card.region.height >= 10
        # ...and the attachment sections flow directly beneath it (no dock,
        # no dead void, no overlap of the visible card).
        assert (
            wrapper.virtual_region.y
            == card.virtual_region.y + card.virtual_region.height
        )
        assert wrapper.virtual_region.y - card.virtual_region.y >= (
            stack.region.height
        ), "attachment sections must start at or below the viewport bottom"


async def test_empty_sections_are_single_adjacent_lines_at_170x50(
    mock_app_instance, stub_characters
):
    """AC#4/#6: two empty sections cost exactly one line each, flush against
    each other - the old layout spent up to 16 lines here and still left a
    dead void between the panels."""
    app = StyledPersonasTestApp(mock_app_instance)
    async with app.run_test(size=(170, 50)) as pilot:
        screen = await _mounted_screen(pilot)
        _, _, wrapper, dicts, world_books = _center_parts(screen)
        await _force_empty_panels(pilot, dicts, world_books)

        assert dicts.virtual_region.height == 1
        assert world_books.virtual_region.height == 1
        assert dicts.virtual_region.y + dicts.virtual_region.height == (
            world_books.virtual_region.y
        ), "no dead void between the two sections"
        assert wrapper.virtual_region.height == 2


async def test_scrolling_the_center_reveals_the_attachment_sections(
    mock_app_instance, stub_characters
):
    """AC#1: the center column scrolls down to the sections (they render
    below the fold, not clipped away)."""
    app = StyledPersonasTestApp(mock_app_instance)
    async with app.run_test(size=(170, 50)) as pilot:
        screen = await _mounted_screen(pilot)
        stack, _, wrapper, dicts, world_books = _center_parts(screen)
        await _force_empty_panels(pilot, dicts, world_books)

        assert stack.max_scroll_y > 0, "attachments must overflow the fold"
        # virtual_region is scroll-independent; Widget.region is NOT clipped
        # by the scroll viewport, so fold math uses virtual-vs-viewport.
        viewport_bottom = stack.scroll_offset.y + stack.region.height
        assert wrapper.virtual_region.y >= viewport_bottom, (
            "sanity: below the fold pre-scroll"
        )

        stack.scroll_end(animate=False)
        await pilot.pause()

        viewport_bottom = stack.scroll_offset.y + stack.region.height
        assert wrapper.virtual_region.y < viewport_bottom, (
            "scrolling must reveal the sections"
        )
        toggle = screen.query_one("#personas-char-dicts-toggle", Button)
        assert stack.region.y <= toggle.region.y < (
            stack.region.y + stack.region.height
        ), "the section header must be on screen after scrolling"


async def test_sections_expand_in_place_and_remember_state_across_reloads(
    mock_app_instance, stub_characters
):
    """AC#3: sections carry their count in the header, expand/collapse in
    place, and keep the user's collapse choice when data reloads."""
    app = StyledPersonasTestApp(mock_app_instance)
    async with app.run_test(size=(170, 50)) as pilot:
        screen = await _mounted_screen(pilot)
        stack, _, _, dicts, _ = _center_parts(screen)

        dicts.load_character_dictionaries(
            [
                {"name": "Slang", "entry_count": 2, "enabled": True},
                {"name": "Cant", "entry_count": 5, "enabled": True},
            ]
        )
        await pilot.pause()

        toggle = screen.query_one("#personas-char-dicts-toggle", Button)
        assert str(toggle.label) == "▸ Dictionaries (2)"
        assert screen.query_one("#personas-char-dicts-body").display is False

        stack.scroll_end(animate=False)
        await pilot.pause()
        await pilot.click("#personas-char-dicts-toggle")
        await pilot.pause()

        assert str(toggle.label) == "▾ Dictionaries (2)"
        assert screen.query_one("#personas-char-dicts-body").display is True

        # A data reload (attach/detach refresh) must not reset the section.
        dicts.load_character_dictionaries(
            [{"name": "Slang", "entry_count": 2, "enabled": True}]
        )
        await pilot.pause()
        assert str(toggle.label) == "▾ Dictionaries (1)"
        assert screen.query_one("#personas-char-dicts-body").display is True

        # Button presses are debounced for the -active animation window;
        # wait it out before collapsing again.
        await pilot.pause(0.4)
        await pilot.click("#personas-char-dicts-toggle")
        await pilot.pause()
        assert str(toggle.label) == "▸ Dictionaries (1)"
        assert screen.query_one("#personas-char-dicts-body").display is False


async def test_card_content_rows_render_inside_the_card_at_100x30(
    mock_app_instance, stub_characters
):
    """AC#5: at the smallest supported size the card's own content (name,
    description, ...) is on screen - empty panels cannot displace it."""
    app = StyledPersonasTestApp(mock_app_instance)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await _mounted_screen(pilot)
        _, card, _, dicts, world_books = _center_parts(screen)
        await _force_empty_panels(pilot, dicts, world_books)

        name_row = screen.query_one("#personas-character-card-name", Static)
        assert "Detective Sam" in str(name_row.renderable)
        assert name_row.region.height > 0
        assert card.region.y <= name_row.region.y < (
            card.region.y + card.region.height
        )
