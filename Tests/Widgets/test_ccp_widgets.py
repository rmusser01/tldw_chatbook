"""Focused CCP widget tests for persona-first management."""

from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label, ListView, Select, Static

from tldw_chatbook.UI.Screens.ccp_screen import CCPScreenState
from tldw_chatbook.Widgets.CCP_Widgets import (
    CCPSidebarWidget,
    CCPPersonaCardWidget,
    CCPPersonaEditorWidget,
    EditPersonaRequested,
    PersonaLoadRequested,
    PersonaSaveRequested,
    StartPersonaChatRequested,
)


def _list_text(list_view: ListView) -> str:
    if not list_view.children:
        return ""
    list_item = list_view.children[0]
    try:
        return str(list_item.query_one(Static).render())
    except Exception:
        return str(list_item.query_one(Label).render())


def _select_option_text(select: Select) -> str:
    labels = []
    for option in getattr(select, "_options", []):
        if isinstance(option, tuple):
            labels.append(str(option[0]))
        else:
            labels.append(str(getattr(option, "prompt", option)))
    return "\n".join(labels)


@pytest.fixture
def mock_parent_screen():
    """Create a minimal parent screen double."""
    screen = Mock()
    screen.state = CCPScreenState()
    screen.app_instance = Mock()
    return screen


@pytest.fixture
def sample_persona_data():
    return {
        "id": "persona.local.alice",
        "name": "Alice Persona",
        "mode": "session_scoped",
        "system_prompt": "A helpful persona that stays warm and thoughtful.",
        "version": 3,
    }


class TestCCPSidebarWidget:
    """Sidebar persona coverage."""

    @pytest.mark.asyncio
    async def test_persona_controls_are_present(self, mock_parent_screen):
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPSidebarWidget(parent_screen=mock_parent_screen)

        app = TestApp()
        async with app.run_test() as pilot:
            sidebar = pilot.app.query_one(CCPSidebarWidget)

            assert sidebar.query_one("#ccp-persona-select")
            assert sidebar.query_one("#ccp-load-persona-button")
            assert sidebar.query_one("#ccp-refresh-persona-list-button")

    @pytest.mark.asyncio
    async def test_empty_library_states_explain_assets_and_chat_flow(self, mock_parent_screen):
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPSidebarWidget(parent_screen=mock_parent_screen)

        app = TestApp()
        async with app.run_test() as pilot:
            sidebar = pilot.app.query_one(CCPSidebarWidget)

            sidebar.update_conversation_results([])
            sidebar.update_character_list([])
            sidebar.update_persona_list([])
            sidebar.update_dictionary_list([])

            conversation_text = _list_text(sidebar.query_one("#conv-char-search-results-list", ListView))
            prompt_text = _list_text(sidebar.query_one("#ccp-prompts-listview", ListView))
            worldbook_text = _list_text(sidebar.query_one("#ccp-worldbooks-listview", ListView))
            character_text = _select_option_text(sidebar.query_one("#conv-char-character-select", Select))
            persona_text = _select_option_text(sidebar.query_one("#ccp-persona-select", Select))
            dictionary_text = _select_option_text(sidebar.query_one("#ccp-dictionary-select", Select))

            assert "No conversations yet." in conversation_text
            assert "Chat" in conversation_text
            assert "Import Conversation" in conversation_text
            assert "No characters yet." in character_text
            assert "Create Character" in character_text
            assert "character-backed Chat" in character_text
            assert "No personas yet." in persona_text
            assert "Create Persona" in persona_text
            assert "persona-backed Chat" in persona_text
            assert "No prompts yet." in prompt_text
            assert "Create New Prompt" in prompt_text
            assert "Chat instructions" in prompt_text
            assert "No chat dictionaries yet." in dictionary_text
            assert "Create Dictionary" in dictionary_text
            assert "Chat context" in dictionary_text
            assert "No world books yet." in worldbook_text
            assert "Create World Book" in worldbook_text
            assert "Chat context" in worldbook_text

    @pytest.mark.asyncio
    async def test_load_persona_button_posts_message(self, mock_parent_screen):
        sidebar = CCPSidebarWidget(parent_screen=mock_parent_screen)
        sidebar._persona_select = Mock(value="persona.local.alice")
        sidebar.post_message = Mock()
        sidebar._emit_message_to_app = AsyncMock(
            side_effect=lambda message, _handler_name: sidebar.post_message(message)
        )
        event = Mock()

        await sidebar.handle_load_persona(event)

        event.stop.assert_called_once()
        posted = sidebar.post_message.call_args[0][0]
        assert isinstance(posted, PersonaLoadRequested)
        assert posted.persona_id == "persona.local.alice"


class TestCCPPersonaWidgets:
    """Persona widget export and interaction coverage."""

    def test_persona_message_accepts_string_identifier(self):
        msg = PersonaLoadRequested("persona.local.alice")

        assert msg.persona_id == "persona.local.alice"

    @pytest.mark.asyncio
    async def test_persona_card_widget_is_mountable(self, mock_parent_screen, sample_persona_data):
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CCPPersonaCardWidget(parent_screen=mock_parent_screen)

        app = TestApp()
        async with app.run_test() as pilot:
            widget = pilot.app.query_one(CCPPersonaCardWidget)
            widget.load_persona(sample_persona_data)

            assert widget.id == "ccp-persona-card-view"
            assert widget.persona_data["id"] == "persona.local.alice"

    @pytest.mark.asyncio
    async def test_persona_card_edit_posts_message(self, mock_parent_screen, sample_persona_data):
        widget = CCPPersonaCardWidget(parent_screen=mock_parent_screen)
        widget.persona_data = dict(sample_persona_data)
        widget.post_message = Mock()
        widget._emit_message_to_app = AsyncMock(
            side_effect=lambda message, _handler_name: widget.post_message(message)
        )
        event = Mock()

        await widget.handle_edit(event)

        event.stop.assert_called_once()
        posted = widget.post_message.call_args[0][0]
        assert isinstance(posted, EditPersonaRequested)
        assert posted.persona_id == "persona.local.alice"

    @pytest.mark.asyncio
    async def test_persona_card_start_chat_posts_message(self, mock_parent_screen, sample_persona_data):
        widget = CCPPersonaCardWidget(parent_screen=mock_parent_screen)
        widget.persona_data = dict(sample_persona_data)
        widget.post_message = Mock()
        widget._emit_message_to_app = AsyncMock(
            side_effect=lambda message, _handler_name: widget.post_message(message)
        )
        event = Mock()

        await widget.handle_start_chat(event)

        event.stop.assert_called_once()
        posted = widget.post_message.call_args[0][0]
        assert isinstance(posted, StartPersonaChatRequested)
        assert posted.persona_id == "persona.local.alice"

    @pytest.mark.asyncio
    async def test_persona_editor_loads_and_saves(self, mock_parent_screen, sample_persona_data):
        widget = CCPPersonaEditorWidget(parent_screen=mock_parent_screen)
        widget.persona_data = dict(sample_persona_data)
        name_input = Mock(value="Updated Persona")
        mode_select = Mock(value="persistent_scoped")
        system_prompt_area = Mock(text="Stay concise and pragmatic.")
        widget.query_one = Mock(
            side_effect=lambda selector, _type=None: {
                "#ccp-persona-name": name_input,
                "#ccp-persona-mode": mode_select,
                "#ccp-persona-system-prompt": system_prompt_area,
            }[selector]
        )
        widget.post_message = Mock()
        widget._emit_message_to_app = AsyncMock(
            side_effect=lambda message, _handler_name: widget.post_message(message)
        )
        event = Mock()

        await widget.handle_save(event)

        event.stop.assert_called_once()
        posted = widget.post_message.call_args[0][0]
        assert isinstance(posted, PersonaSaveRequested)
        assert posted.persona_data["name"] == "Updated Persona"
        assert posted.persona_data["mode"] == "persistent_scoped"
        assert posted.persona_data["system_prompt"] == "Stay concise and pragmatic."
