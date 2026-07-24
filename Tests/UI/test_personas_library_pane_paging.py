"""Mounted tests for the Personas library pane sort/tag/page controls (P3a Task 3)."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    LibraryRow,
    PersonasLibraryPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaPageChanged,
    PersonaSortCycleRequested,
    PersonaTagFilterRequested,
)

pytestmark = pytest.mark.asyncio


class _Host(App):
    def __init__(self):
        super().__init__()
        self.events = []

    def compose(self) -> ComposeResult:
        yield PersonasLibraryPane(id="pane")

    def on_persona_sort_cycle_requested(self, message: PersonaSortCycleRequested) -> None:
        self.events.append(("sort", None))

    def on_persona_tag_filter_requested(self, message: PersonaTagFilterRequested) -> None:
        self.events.append(("tag", None))

    def on_persona_page_changed(self, message: PersonaPageChanged) -> None:
        self.events.append(("page", message.delta))


def _rows(n):
    return tuple(
        LibraryRow(item_id=str(i), kind="character", name=f"c{i:03d}") for i in range(n)
    )


async def test_page_bar_shows_when_total_exceeds_page_size():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            _rows(50), total=130, noun="characters", page_offset=0, page_size=50
        )
        await pilot.pause()
        info = app.query_one("#personas-library-page-info", Static)
        assert "1-50 of 130" in str(info.renderable)
        assert app.query_one("#personas-library-prev", Button).disabled is True
        assert app.query_one("#personas-library-next", Button).disabled is False
        assert app.query_one("#personas-library-pagebar").display is True


async def test_page_bar_hidden_when_fits_one_page():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            _rows(5), total=5, noun="characters", page_offset=0, page_size=50
        )
        await pilot.pause()
        assert app.query_one("#personas-library-pagebar").display is False


async def test_next_prev_post_page_changed():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            _rows(50), total=130, noun="characters", page_offset=50, page_size=50
        )
        await pilot.pause()
        await pilot.click("#personas-library-next")
        await pilot.click("#personas-library-prev")
        await pilot.pause()
        assert ("page", 1) in app.events and ("page", -1) in app.events


async def test_sort_and_tag_buttons_post_intents():
    app = _Host()
    async with app.run_test() as pilot:
        app.query_one(PersonasLibraryPane).set_mode("characters")
        await pilot.pause()
        await pilot.click("#personas-library-sort")
        await pilot.click("#personas-library-tag")
        await pilot.pause()
        assert ("sort", None) in app.events and ("tag", None) in app.events


async def test_tag_button_hidden_for_personas_visible_for_characters():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        pane.set_mode("personas")
        assert app.query_one("#personas-library-tag", Button).display is False
        assert app.query_one("#personas-library-sort", Button).display is True
        pane.set_mode("characters")
        assert app.query_one("#personas-library-tag", Button).display is True


async def test_sort_page_hidden_for_lore():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        pane.set_mode("lore")
        assert app.query_one("#personas-library-sort", Button).display is False
        assert app.query_one("#personas-library-tag", Button).display is False
        # dict/lore call update_rows WITHOUT page kwargs -> no page bar, plain count
        await pane.update_rows((), total=0, noun="world books")
        await pilot.pause()
        assert app.query_one("#personas-library-pagebar").display is False


async def test_set_sort_label_and_set_tag_label():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        pane.set_sort_label("Sort: Recent")
        pane.set_tag_label("Tag: villain")
        assert str(app.query_one("#personas-library-sort", Button).label) == "Sort: Recent"
        assert str(app.query_one("#personas-library-tag", Button).label) == "Tag: villain"


async def test_update_rows_without_page_kwargs_keeps_plain_count():
    """Backward compatibility: dict/lore callers omit page_offset/page_size."""
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        await pane.update_rows(
            (LibraryRow(item_id="1", kind="dictionary", name="D1"),),
            total=1,
            noun="dictionaries",
        )
        await pilot.pause()
        count = app.query_one("#personas-library-count", Static)
        # task-445: a total of exactly 1 reads singular ("1 dictionary").
        assert "1 dictionary" in str(count.renderable)
        assert app.query_one("#personas-library-pagebar").display is False
