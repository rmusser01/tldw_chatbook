"""UX audit smoke tests for top-level shell navigation."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved
from tldw_chatbook.UI.Screens.chatbooks_screen import ChatbooksScreen


class ChatbooksShellSmokeApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ChatbooksScreen(self)


@pytest.mark.asyncio
async def test_chatbooks_screen_keeps_shared_navigation_escape(monkeypatch):
    async def no_refresh(self):
        self.chatbooks = []

    monkeypatch.setattr(ChatbooksWindowImproved, "_refresh_chatbooks", no_refresh)
    app = ChatbooksShellSmokeApp()

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)

        assert app.screen.query_one(ChatbooksWindowImproved) is not None
        assert app.screen.query_one("#nav-chat") is not None
        assert app.screen.query_one("#nav-chatbooks") is not None
