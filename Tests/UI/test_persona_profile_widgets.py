"""Mounted tests for persona profile card and editor widgets."""

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_card_widget import (
    PersonaProfileCardWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    EditPersonaRequested,
    PersonaProfileSaveRequested,
)

pytestmark = pytest.mark.asyncio

PROFILE = {
    "id": "p-1",
    "name": "Archivist",
    "description": "Preserve, organize, retrieve",
    "system_prompt": "You are a meticulous archivist.",
}


class WidgetApp(App):
    def compose(self):
        yield PersonaProfileCardWidget()
        yield PersonaProfileEditorWidget()


async def test_card_shows_profile_and_edit_posts_message():
    received = []

    class CaptureApp(WidgetApp):
        def on_edit_persona_requested(self, message: EditPersonaRequested) -> None:
            received.append(message.persona_id)

    app = CaptureApp()
    async with app.run_test() as pilot:
        card = pilot.app.query_one(PersonaProfileCardWidget)
        card.show_persona(PROFILE)
        await pilot.pause()
        assert "Archivist" in str(
            pilot.app.query_one("#personas-card-name", Static).renderable
        )
        assert "meticulous archivist" in str(
            pilot.app.query_one("#personas-card-system-prompt", Static).renderable
        )
        await pilot.click("#personas-card-edit")
        await pilot.pause()
    assert received == ["p-1"]


async def test_card_edit_disabled_without_persona():
    app = WidgetApp()
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#personas-card-edit", Button).disabled is True


async def test_editor_load_collect_roundtrip():
    app = WidgetApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonaProfileEditorWidget)
        editor.load_persona(PROFILE)
        await pilot.pause()
        assert pilot.app.query_one("#personas-editor-name", Input).value == "Archivist"
        data = editor.collect()
        assert data["name"] == "Archivist"
        assert data["system_prompt"] == "You are a meticulous archivist."
        assert data["id"] == "p-1"


async def test_editor_new_persona_clears_previous_state():
    app = WidgetApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonaProfileEditorWidget)
        editor.load_persona(PROFILE)
        await pilot.pause()
        editor.new_persona()
        await pilot.pause()
        assert pilot.app.query_one("#personas-editor-name", Input).value == ""
        assert "id" not in editor.collect()


async def test_editor_save_posts_collected_data():
    received = []

    class CaptureApp(WidgetApp):
        def on_persona_profile_save_requested(self, message: PersonaProfileSaveRequested) -> None:
            received.append(message.data)

    app = CaptureApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonaProfileEditorWidget)
        editor.new_persona()
        pilot.app.query_one("#personas-editor-name", Input).value = "Mentor"
        await pilot.pause()
        await pilot.click("#personas-editor-save")
        await pilot.pause()
    assert received and received[0]["name"] == "Mentor"


async def test_editor_save_with_empty_name_blocks_and_shows_error():
    received = []

    class CaptureApp(WidgetApp):
        def on_persona_profile_save_requested(self, message: PersonaProfileSaveRequested) -> None:
            received.append(message.data)

    app = CaptureApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonaProfileEditorWidget)
        editor.new_persona()
        await pilot.pause()
        await pilot.click("#personas-editor-save")
        await pilot.pause()
        validation = pilot.app.query_one("#personas-editor-validation", Static)
        assert "name: required" in str(validation.renderable)
    assert received == []
