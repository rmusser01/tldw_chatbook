"""Mounted tests for the Personas library pane."""

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    LibraryRow,
    PersonasLibraryPane,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaActionRequested,
    PersonaEntitySelected,
    PersonaSearchChanged,
)

pytestmark = pytest.mark.asyncio


class LibraryPaneApp(App):
    def compose(self):
        yield PersonasLibraryPane(id="personas-library-pane")


async def test_pane_renders_search_toolbar_and_empty_state():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        assert pilot.app.query_one("#personas-library-search", Input)
        assert pilot.app.query_one("#personas-library-new", Button)
        assert pilot.app.query_one("#personas-library-import", Button)
        pane.update_rows((), total=0, noun="characters")
        await pilot.pause()
        empty = pilot.app.query_one("#personas-library-empty", Static)
        assert "No characters yet" in str(empty.renderable)


async def test_update_rows_renders_rows_and_count():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        rows = (
            LibraryRow(item_id="1", kind="character", name="Detective Sam"),
            LibraryRow(item_id="2", kind="character", name="Tutor", is_unsaved=True),
        )
        pane.update_rows(rows, total=2, noun="characters")
        await pilot.pause()
        buttons = pilot.app.query(".personas-library-row")
        assert len(buttons) == 2
        assert "is-unsaved" in pilot.app.query_one(
            "#personas-library-row-character-2", Button
        ).classes
        count = pilot.app.query_one("#personas-library-count", Static)
        assert "2 characters" in str(count.renderable)


async def test_filtered_count_shows_n_of_m():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        pane.update_rows(
            (LibraryRow(item_id="1", kind="character", name="Detective Sam"),),
            total=12,
            noun="characters",
            filtered=True,
        )
        await pilot.pause()
        count = pilot.app.query_one("#personas-library-count", Static)
        assert "1 of 12 characters" in str(count.renderable)


async def test_row_press_posts_persona_entity_selected():
    received = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_entity_selected(self, message: PersonaEntitySelected) -> None:
            received.append((message.entity_kind, message.entity_id, message.entity_name))

    app = CaptureApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        pane.update_rows(
            (LibraryRow(item_id="7", kind="character", name="Detective Sam"),),
            total=1,
            noun="characters",
        )
        await pilot.pause()
        await pilot.click("#personas-library-row-character-7")
        await pilot.pause()
    assert received == [("character", "7", "Detective Sam")]


async def test_search_input_posts_search_changed_and_new_posts_create_action():
    searches = []
    actions = []

    class CaptureApp(LibraryPaneApp):
        def on_persona_search_changed(self, message: PersonaSearchChanged) -> None:
            searches.append(message.query)

        def on_persona_action_requested(self, message: PersonaActionRequested) -> None:
            actions.append(message.action)

    app = CaptureApp()
    async with app.run_test() as pilot:
        search = pilot.app.query_one("#personas-library-search", Input)
        search.value = "sam"
        await pilot.pause()
        await pilot.click("#personas-library-new")
        await pilot.pause()
        await pilot.click("#personas-library-import")
        await pilot.pause()
    assert searches[-1] == "sam"
    assert actions == ["create", "import"]


async def test_mark_active_row_applies_is_active_to_selected_only():
    app = LibraryPaneApp()
    async with app.run_test() as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        pane.update_rows(
            (
                LibraryRow(item_id="1", kind="character", name="Detective Sam"),
                LibraryRow(item_id="2", kind="character", name="Tutor"),
            ),
            total=2,
            noun="characters",
        )
        await pilot.pause()
        pane.mark_active_row("character", "2")
        assert "is-active" in pilot.app.query_one("#personas-library-row-character-2", Button).classes
        assert "is-active" not in pilot.app.query_one("#personas-library-row-character-1", Button).classes
