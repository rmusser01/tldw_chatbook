"""Mounted tests for persona profile card and editor widgets."""

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.tldw_api import PersonaProfileCreate
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


class EditorOnlyApp(App):
    """Isolated harness for editor-only tests — keeps card off-screen concerns away."""

    def compose(self):
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
        # Labeled inline rows, matching the character card's vocabulary.
        assert (
            str(pilot.app.query_one("#personas-card-name", Static).renderable)
            == "Name: Archivist"
        )
        assert (
            str(pilot.app.query_one("#personas-card-description", Static).renderable)
            == "Description: Preserve, organize, retrieve"
        )
        assert (
            str(pilot.app.query_one("#personas-card-system-prompt", Static).renderable)
            == "System prompt: You are a meticulous archivist."
        )
        await pilot.click("#personas-card-edit")
        await pilot.pause()
    assert received == ["p-1"]


async def test_card_hides_empty_rows():
    """Empty description/system prompt rows are hidden - no bare labels."""
    app = WidgetApp()
    async with app.run_test() as pilot:
        card = pilot.app.query_one(PersonaProfileCardWidget)
        card.show_persona({"id": "p-1", "name": "Archivist", "description": ""})
        await pilot.pause()
        assert pilot.app.query_one("#personas-card-name", Static).display is True
        assert pilot.app.query_one("#personas-card-description", Static).display is False
        assert (
            pilot.app.query_one("#personas-card-system-prompt", Static).display is False
        )
        # Re-show with values: the rows come back.
        card.show_persona(PROFILE)
        await pilot.pause()
        assert pilot.app.query_one("#personas-card-description", Static).display is True
        assert (
            pilot.app.query_one("#personas-card-system-prompt", Static).display is True
        )


async def test_card_markup_like_content_renders_literally():
    """Profile text with Rich-markup-looking content must not raise at render."""
    app = WidgetApp()
    async with app.run_test() as pilot:
        card = pilot.app.query_one(PersonaProfileCardWidget)
        card.show_persona(
            {"id": "p-1", "name": "[/x]", "description": "[bold]unclosed"}
        )
        await pilot.pause()  # would raise MarkupError at render with markup on
        assert "[/x]" in str(
            pilot.app.query_one("#personas-card-name", Static).renderable
        )


async def test_card_edit_button_lives_in_toolbar_with_shared_classes():
    app = WidgetApp()
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#personas-card-edit", Button)
        assert button.has_class("console-action-secondary")
        toolbar = button.parent
        assert toolbar is not None and toolbar.has_class("ds-toolbar")


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


async def test_editor_roundtrips_version():
    app = WidgetApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonaProfileEditorWidget)
        editor.load_persona({**PROFILE, "version": 3})
        await pilot.pause()
        assert editor.collect()["version"] == 3
        editor.new_persona()
        await pilot.pause()
        assert "version" not in editor.collect()


async def test_persona_profile_create_schema_accepts_description():
    profile = PersonaProfileCreate(name="x", description="d")
    assert profile.description == "d"


async def test_editor_save_posts_collected_data():
    received = []

    class CaptureApp(EditorOnlyApp):
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

    class CaptureApp(EditorOnlyApp):
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


async def test_long_system_prompt_keeps_edit_toolbar_visible():
    """A 60-line system prompt must not push the Edit button off-screen."""
    long_prompt = "\n".join(f"Line {i}: do something useful here." for i in range(60))
    profile_with_long_prompt = {
        "id": "p-long",
        "name": "Verbose Persona",
        "description": "A persona with a very long system prompt.",
        "system_prompt": long_prompt,
    }

    app = WidgetApp()
    async with app.run_test(size=(80, 24)) as pilot:
        card = pilot.app.query_one(PersonaProfileCardWidget)
        card.show_persona(profile_with_long_prompt)
        await pilot.pause()
        # The Edit button must be clickable (visible on screen) even with a long
        # system prompt filling the scrollable body above it.
        await pilot.click("#personas-card-edit")
        await pilot.pause()  # no exception means the button was reachable
