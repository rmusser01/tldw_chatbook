import pytest

from textual.app import App, ComposeResult

from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved
from tldw_chatbook.UI.Screens.chatbooks_screen import ChatbooksScreen


@pytest.mark.asyncio
async def test_chatbooks_screen_uses_improved_window(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksScreenApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksScreen(self)

    app = ChatbooksScreenApp()
    async with app.run_test() as pilot:
        assert app.screen.query_one(ChatbooksWindowImproved) is not None
        assert app.screen.query_one("#nav-chat") is not None
        assert app.screen.query_one("#nav-chatbooks") is not None


@pytest.mark.asyncio
async def test_improved_window_exposes_server_action_cards(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

    app = ChatbooksWindowApp()
    async with app.run_test() as pilot:
        window = app.query_one(ChatbooksWindowImproved)
        assert window.query_one("#create-server-action") is not None
        assert window.query_one("#import-server-action") is not None


@pytest.mark.asyncio
async def test_server_create_action_uses_server_mode(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

    app = ChatbooksWindowApp()
    async with app.run_test() as pilot:
        window = app.query_one(ChatbooksWindowImproved)
        recorded = {}

        async def fake_action_create_chatbook(execution_mode="local"):
            recorded["mode"] = execution_mode

        window.action_create_chatbook = fake_action_create_chatbook
        await window.action_create_chatbook_server()

        assert recorded["mode"] == "server"


@pytest.mark.asyncio
async def test_server_import_action_uses_server_mode(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)

    class ChatbooksWindowApp(App):
        def compose(self) -> ComposeResult:
            yield ChatbooksWindowImproved(self)

    app = ChatbooksWindowApp()
    async with app.run_test() as pilot:
        window = app.query_one(ChatbooksWindowImproved)
        recorded = {}

        async def fake_action_import_chatbook(execution_mode="local"):
            recorded["mode"] = execution_mode

        window.action_import_chatbook = fake_action_import_chatbook
        await window.action_import_chatbook_server()

        assert recorded["mode"] == "server"
