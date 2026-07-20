import pytest
from textual.app import App

from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider
from tldw_chatbook.Widgets.Console.console_context_modal import ConsoleContextModal


class ChatScreenHarness(App):
    COMMANDS = App.COMMANDS | {ConsoleCommandProvider}

    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


@pytest.fixture
def app_instance():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "local-model",
            },
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    return app


@pytest.mark.asyncio
async def test_chat_screen_keybinding_opens_context_modal(app_instance):
    app = ChatScreenHarness(app_instance)

    async with app.run_test(size=(120, 40)) as pilot:
        # Ensure an active session and a message exist.
        screen = app.screen
        controller = screen._ensure_console_chat_controller()
        session = controller.store.ensure_session(title="Test")
        controller.store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Hello",
        )
        await pilot.pause()

        await pilot.press("ctrl+shift+p")
        await pilot.pause()

        assert isinstance(app.screen, ConsoleContextModal)


@pytest.mark.asyncio
async def test_chat_screen_command_palette_opens_context_modal(app_instance):
    app = ChatScreenHarness(app_instance)

    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        controller = screen._ensure_console_chat_controller()
        session = controller.store.ensure_session(title="Test")
        controller.store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Hello",
        )
        await pilot.pause()

        # Open command palette and select the context command.
        await pilot.press("ctrl+p")
        await pilot.pause()
        for character in "view context":
            await pilot.press(character)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ConsoleContextModal)
