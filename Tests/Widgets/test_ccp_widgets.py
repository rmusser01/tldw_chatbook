"""Focused CCP widget tests for persona-first management."""

from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult

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
