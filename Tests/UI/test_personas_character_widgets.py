"""Mounted tests for the ds-native Personas character card/editor/transcript widgets."""

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Widgets.CCP_Widgets.ccp_character_card_widget import (
    EditCharacterRequested,
)
from tldw_chatbook.Widgets.CCP_Widgets.ccp_character_editor_widget import (
    CharacterEditorCancelled,
    CharacterSaveRequested,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_card_widget import (
    PersonasCharacterCardWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_conversation_transcript_widget import (
    PersonasConversationTranscriptWidget,
)

pytestmark = pytest.mark.asyncio

CHARACTER = {
    "id": 7,
    "version": 2,
    "name": "Detective Sam",
    "description": "Noir detective",
    "personality": "Wry, persistent",
    "scenario": "A rainy night office",
    "first_message": "The name's Sam. Who's asking?",
    "system_prompt": "You are a noir detective.",
    "post_history_instructions": "Stay terse.",
    "creator_notes": "Keep it 1940s.",
    "creator": "rmusser",
    "character_version": "1.2",
    "tags": ["noir", "detective"],
    "alternate_greetings": ["Evening.", "You again?"],
    "image": b"\x89PNG-fake",
}

#: The exact key structure the legacy CCPCharacterEditorWidget.get_character_data
#: produced for a loaded V2 card: the loaded record's keys plus the editor's
#: own field keys (note ``first_mes``, the legacy alias the save path maps).
LEGACY_EDITOR_KEYS = set(CHARACTER) | {
    "name",
    "description",
    "personality",
    "scenario",
    "first_mes",
    "creator_notes",
    "system_prompt",
    "post_history_instructions",
    "creator",
    "character_version",
    "alternate_greetings",
    "tags",
}


class WidgetApp(App):
    def compose(self):
        yield PersonasCharacterCardWidget()
        yield PersonasCharacterEditorWidget()
        yield PersonasConversationTranscriptWidget()


# ===== Card =====


class TestCharacterCard:
    async def test_placeholder_shown_and_edit_disabled_before_load(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            assert pilot.app.query_one("#personas-character-card-empty").display is True
            assert pilot.app.query_one("#personas-character-card-body").display is False
            assert pilot.app.query_one("#personas-card-edit-character", Button).disabled is True

    async def test_load_populates_fields_and_enables_edit(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            card = pilot.app.query_one(PersonasCharacterCardWidget)
            card.load_character(dict(CHARACTER))
            await pilot.pause()
            assert pilot.app.query_one("#personas-character-card-empty").display is False
            assert pilot.app.query_one("#personas-character-card-body").display is True

            def text(selector: str) -> str:
                return str(pilot.app.query_one(selector, Static).renderable)

            assert "Detective Sam" in text("#personas-character-card-name")
            assert "Noir detective" in text("#personas-character-card-description")
            assert "Wry, persistent" in text("#personas-character-card-personality")
            assert "rainy night" in text("#personas-character-card-scenario")
            assert "Who's asking?" in text("#personas-character-card-first-message")
            assert "noir detective" in text("#personas-character-card-system-prompt")
            assert "Stay terse." in text("#personas-character-card-post-history")
            assert "rmusser" in text("#personas-character-card-creator")
            assert "1.2" in text("#personas-character-card-version")
            assert text("#personas-character-card-tags") == "Tags: noir, detective"
            assert "Alternate greetings: 2" in text("#personas-character-card-alt-greetings")
            assert "Evening." in text("#personas-character-card-greeting-preview")
            assert text("#personas-card-avatar-status") == "Avatar: embedded"
            assert pilot.app.query_one("#personas-card-edit-character", Button).disabled is False

    async def test_load_accepts_first_mes_alias_and_no_avatar(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            card = pilot.app.query_one(PersonasCharacterCardWidget)
            data = dict(CHARACTER)
            data.pop("first_message")
            data.pop("image")
            data["first_mes"] = "Aliased greeting."
            card.load_character(data)
            await pilot.pause()
            assert "Aliased greeting." in str(
                pilot.app.query_one("#personas-character-card-first-message", Static).renderable
            )
            assert (
                str(pilot.app.query_one("#personas-card-avatar-status", Static).renderable)
                == "Avatar: none"
            )

    async def test_edit_posts_legacy_message(self):
        received = []

        class CaptureApp(WidgetApp):
            def on_edit_character_requested(self, message: EditCharacterRequested) -> None:
                received.append(message.character_id)

        app = CaptureApp()
        async with app.run_test() as pilot:
            card = pilot.app.query_one(PersonasCharacterCardWidget)
            card.load_character(dict(CHARACTER))
            await pilot.pause()
            pilot.app.query_one("#personas-card-edit-character", Button).press()
            await pilot.pause()
        assert received == ["7"]

    async def test_default_id_matches_handler_query(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            widget = pilot.app.query_one("#ccp-character-card-view")
            assert isinstance(widget, PersonasCharacterCardWidget)
            assert hasattr(widget, "load_character")

    async def test_load_with_markup_like_content_does_not_raise(self):
        """Field values with Rich-markup-looking text must render literally."""
        app = WidgetApp()
        async with app.run_test() as pilot:
            card = pilot.app.query_one(PersonasCharacterCardWidget)
            data = dict(CHARACTER)
            data["name"] = "[/x]"
            data["description"] = "[bold]unclosed"
            data["tags"] = ["[/tag]"]
            data["alternate_greetings"] = ["[/oops] hi"]
            card.load_character(data)
            await pilot.pause()  # would raise MarkupError at render with markup on
            assert "[/x]" in str(
                pilot.app.query_one("#personas-character-card-name", Static).renderable
            )


# ===== Editor =====


class TestCharacterEditor:
    async def test_load_collect_roundtrip_matches_legacy_key_structure(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            editor.load_character(dict(CHARACTER))
            await pilot.pause()
            assert (
                pilot.app.query_one("#personas-char-editor-name", Input).value
                == "Detective Sam"
            )
            data = editor.get_character_data()
            assert set(data) == LEGACY_EDITOR_KEYS
            assert data["id"] == 7
            assert data["version"] == 2
            assert data["name"] == "Detective Sam"
            assert data["first_mes"] == "The name's Sam. Who's asking?"
            assert data["first_message"] == "The name's Sam. Who's asking?"
            assert data["tags"] == ["noir", "detective"]
            assert data["alternate_greetings"] == ["Evening.", "You again?"]
            assert data["character_version"] == "1.2"
            assert data["creator"] == "rmusser"
            assert data["creator_notes"] == "Keep it 1940s."
            assert data["post_history_instructions"] == "Stay terse."

    async def test_multiline_greeting_survives_untouched_roundtrip(self):
        """A multi-paragraph greeting must not be re-split on an untouched save."""
        app = WidgetApp()
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            data = dict(CHARACTER)
            data["alternate_greetings"] = ["para1\n\npara2"]
            editor.load_character(data)
            await pilot.pause()
            collected = editor.get_character_data()
            assert collected["alternate_greetings"] == ["para1\n\npara2"]

    async def test_edited_greetings_are_reparsed_per_line(self):
        """Once the TextArea is edited, greetings re-parse one per line."""
        app = WidgetApp()
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            data = dict(CHARACTER)
            data["alternate_greetings"] = ["para1\n\npara2"]
            editor.load_character(data)
            await pilot.pause()
            pilot.app.query_one(
                "#personas-char-editor-alt-greetings", TextArea
            ).text = "first\n\nsecond"
            collected = editor.get_character_data()
            assert collected["alternate_greetings"] == ["first", "second"]

    async def test_empty_version_defaults_to_1_0(self):
        """Empty/whitespace Version collects as the new-character default."""
        app = WidgetApp()
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            editor.load_character(dict(CHARACTER))
            await pilot.pause()
            pilot.app.query_one("#personas-char-editor-version", Input).value = "   "
            data = editor.get_character_data()
            assert data["character_version"] == "1.0"

    async def test_first_message_edit_updates_both_aliases(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            editor.load_character(dict(CHARACTER))
            await pilot.pause()
            pilot.app.query_one(
                "#personas-char-editor-first-message", TextArea
            ).text = "New opening line."
            data = editor.get_character_data()
            assert data["first_mes"] == "New opening line."
            # The stale loaded value must not survive in the DB-facing alias.
            assert data["first_message"] == "New opening line."

    async def test_new_character_clears_and_defaults_version(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            editor.load_character(dict(CHARACTER))
            await pilot.pause()
            editor.new_character()
            await pilot.pause()
            assert pilot.app.query_one("#personas-char-editor-name", Input).value == ""
            assert (
                pilot.app.query_one("#personas-char-editor-version", Input).value == "1.0"
            )
            data = editor.get_character_data()
            assert "id" not in data
            assert "version" not in data
            assert data["tags"] == []
            assert data["alternate_greetings"] == []

    async def test_save_posts_legacy_message_with_collected_data(self):
        received = []

        class CaptureApp(WidgetApp):
            def on_character_save_requested(self, message: CharacterSaveRequested) -> None:
                received.append(message.character_data)

        app = CaptureApp()
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            editor.new_character()
            pilot.app.query_one("#personas-char-editor-name", Input).value = "New Hero"
            pilot.app.query_one("#personas-char-editor-tags", Input).value = "brave, kind"
            await pilot.pause()
            pilot.app.query_one("#personas-char-editor-save", Button).press()
            await pilot.pause()
        assert received and received[0]["name"] == "New Hero"
        assert received[0]["tags"] == ["brave", "kind"]

    async def test_save_with_empty_name_blocks_and_shows_error(self):
        received = []

        class CaptureApp(WidgetApp):
            def on_character_save_requested(self, message: CharacterSaveRequested) -> None:
                received.append(message.character_data)

        app = CaptureApp()
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            editor.new_character()
            await pilot.pause()
            pilot.app.query_one("#personas-char-editor-save", Button).press()
            await pilot.pause()
            validation = pilot.app.query_one("#personas-char-editor-validation", Static)
            assert "name: required" in str(validation.renderable)
        assert received == []

    async def test_cancel_posts_legacy_message(self):
        received = []

        class CaptureApp(WidgetApp):
            def on_character_editor_cancelled(self, message: CharacterEditorCancelled) -> None:
                received.append(message)

        app = CaptureApp()
        async with app.run_test() as pilot:
            pilot.app.query_one("#personas-char-editor-cancel", Button).press()
            await pilot.pause()
        assert len(received) == 1

    async def test_advanced_toggle_shows_and_hides_section(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            advanced = pilot.app.query_one("#personas-char-editor-advanced")
            toggle = pilot.app.query_one("#personas-char-editor-advanced-toggle", Button)
            assert advanced.display is False
            assert str(toggle.label) == "Advanced ▸"
            toggle.press()
            await pilot.pause()
            assert advanced.display is True
            assert str(toggle.label) == "Advanced ▾"
            toggle.press()
            await pilot.pause()
            assert advanced.display is False
            assert str(toggle.label) == "Advanced ▸"

    async def test_load_character_collapses_advanced_section(self):
        """Loading (or starting) a character resets Advanced to collapsed."""
        app = WidgetApp()
        async with app.run_test() as pilot:
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            advanced = pilot.app.query_one("#personas-char-editor-advanced")
            toggle = pilot.app.query_one("#personas-char-editor-advanced-toggle", Button)
            toggle.press()
            await pilot.pause()
            assert advanced.display is True
            editor.load_character(dict(CHARACTER))
            await pilot.pause()
            assert advanced.display is False
            assert str(toggle.label) == "Advanced ▸"
            toggle.press()
            await pilot.pause()
            editor.new_character()
            await pilot.pause()
            assert advanced.display is False
            assert str(toggle.label) == "Advanced ▸"

    async def test_no_upload_button_and_avatar_status_is_read_only(self):
        """No handler exists for avatar upload, so the editor must not offer it."""
        app = WidgetApp()
        async with app.run_test() as pilot:
            assert not list(pilot.app.query("#personas-char-editor-avatar-upload"))
            editor = pilot.app.query_one(PersonasCharacterEditorWidget)
            editor.load_character(dict(CHARACTER))
            await pilot.pause()
            assert (
                str(
                    pilot.app.query_one(
                        "#personas-char-editor-avatar-status", Static
                    ).renderable
                )
                == "Avatar: embedded"
            )

    async def test_default_id_matches_screen_query(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            widget = pilot.app.query_one("#ccp-character-editor-view")
            assert isinstance(widget, PersonasCharacterEditorWidget)


# ===== Transcript =====


class TestConversationTranscript:
    async def test_load_messages_renders_role_lines(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            view = pilot.app.query_one(PersonasConversationTranscriptWidget)
            await view.load_messages(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            )
            await pilot.pause()
            lines = list(view.query(".personas-transcript-line"))
            assert len(lines) == 2
            assert "user: Hello" in str(lines[0].renderable)
            assert "personas-transcript-line-user" in lines[0].classes
            assert "assistant: Hi there" in str(lines[1].renderable)
            assert "personas-transcript-line-assistant" in lines[1].classes

    async def test_empty_messages_show_placeholder(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            view = pilot.app.query_one(PersonasConversationTranscriptWidget)
            await view.load_messages([])
            await pilot.pause()
            assert "No messages" in str(
                pilot.app.query_one("#personas-transcript-empty", Static).renderable
            )

    async def test_double_load_replaces_without_crash(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            view = pilot.app.query_one(PersonasConversationTranscriptWidget)
            await view.load_messages([{"role": "user", "content": "First"}])
            await view.load_messages(
                [
                    {"role": "user", "content": "Second"},
                    {"role": "assistant", "content": "Reply"},
                ]
            )
            await pilot.pause()
            lines = list(view.query(".personas-transcript-line"))
            assert len(lines) == 2
            assert "Second" in str(lines[0].renderable)
            # The empty-state placeholder never appears alongside lines.
            assert not list(view.query("#personas-transcript-empty"))

    async def test_set_title_updates_header(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            view = pilot.app.query_one(PersonasConversationTranscriptWidget)
            view.set_title("Case File 7")
            await pilot.pause()
            assert "Case File 7" in str(
                pilot.app.query_one("#personas-transcript-title", Static).renderable
            )

    async def test_markup_like_content_renders_without_raising(self):
        """Message content/titles with Rich-markup-looking text render literally."""
        app = WidgetApp()
        async with app.run_test() as pilot:
            view = pilot.app.query_one(PersonasConversationTranscriptWidget)
            view.set_title("[/bad title]")
            await view.load_messages([{"role": "user", "content": "[/bad]"}])
            await pilot.pause()  # would raise MarkupError at render with markup on
            lines = list(view.query(".personas-transcript-line"))
            assert len(lines) == 1
            assert "user: [/bad]" in str(lines[0].renderable)

    async def test_default_id(self):
        app = WidgetApp()
        async with app.run_test() as pilot:
            assert pilot.app.query_one("#personas-conversation-transcript-view")
