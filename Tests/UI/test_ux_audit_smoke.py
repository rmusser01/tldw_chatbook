"""UX audit smoke tests for top-level shell navigation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, TextArea

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import apply_current_handoff_context
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved
from tldw_chatbook.UI.Screens.chatbooks_screen import ChatbooksScreen
from tldw_chatbook.Widgets.Chat_Widgets.chat_handoff_card import ChatHandoffCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer


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


class _HandoffSmokeHost:
    def __init__(self, payload: ChatHandoffPayload) -> None:
        self.pending_chat_handoff = payload
        self.chat_enhanced_mode = False
        self.notify = Mock()
        self.current_chat_conversation_id = None
        self.current_chat_is_ephemeral = True
        self.current_chat_worker = None
        self.current_ai_message_widget = None
        self._current_chat_handoff_payload = None
        self._current_chat_is_streaming = False

    def set_current_chat_is_streaming(self, value: bool) -> None:
        self._current_chat_is_streaming = value

    def get_current_chat_is_streaming(self) -> bool:
        return self._current_chat_is_streaming


class HandoffFirstSendSmokeApp(App[None]):
    def __init__(self, payload: ChatHandoffPayload) -> None:
        super().__init__()
        self.host = _HandoffSmokeHost(payload)
        self.host.call_later = self.call_later
        self.tab_container: ChatTabContainer | None = None

    def compose(self) -> ComposeResult:
        self.tab_container = ChatTabContainer(self.host)
        yield self.tab_container

    def on_mount(self) -> None:
        self.host.query_one = self.query_one
        self.host.query = self.query


@pytest.mark.asyncio
async def test_handoff_smoke_replays_chat_staging_and_first_send(monkeypatch):
    payload = ChatHandoffPayload(
        source="notes",
        item_type="note",
        title="Field Notes",
        body="Observed confusing empty states.",
        suggested_prompt="Summarize the usability issue.",
    )
    app = HandoffFirstSendSmokeApp(payload)
    observed: dict[str, str | None] = {}

    async def fake_send_handler(app, event):
        tab_container = app.query_one(ChatTabContainer)
        session = tab_container.sessions[tab_container.active_session_id]
        payload = session.session_data.handoff_payload

        app.host._current_chat_handoff_payload = payload
        observed["wrapped_prompt"] = apply_current_handoff_context(
            app.host,
            "Summarize the usability issue.",
        )
        observed["active_handoff_title"] = app.host._current_chat_handoff_payload.title

        payload.status = "sent"
        app.host._current_chat_handoff_payload = None

    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed",
        fake_send_handler,
    )

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.05)
        tab_container = app.tab_container
        assert tab_container is not None

        screen = ChatScreen(app.host)
        screen.chat_window = SimpleNamespace(_tab_container=tab_container)
        await screen._consume_pending_chat_handoff()
        await pilot.pause(0.05)

        handoff_tab_id = tab_container.active_session_id
        assert handoff_tab_id is not None
        session = tab_container.sessions[handoff_tab_id]
        assert app.host.pending_chat_handoff is None
        assert session.session_data.conversation_id is None
        assert session.session_data.is_ephemeral is True
        assert session.session_data.handoff_payload.title == "Field Notes"
        assert session.query_one(ChatHandoffCard).payload.status == "staged"
        assert (
            session.query_one(f"#chat-input-{handoff_tab_id}", TextArea).text
            == "Summarize the usability issue."
        )

        session.query_one(f"#send-stop-chat-{handoff_tab_id}", Button).press()
        await pilot.pause(0.05)

        assert observed["active_handoff_title"] == "Field Notes"
        assert "[Staged context]" in observed["wrapped_prompt"]
        assert "Observed confusing empty states." in observed["wrapped_prompt"]
        assert "[User prompt]\nSummarize the usability issue." in observed["wrapped_prompt"]
        assert session.session_data.handoff_payload.status == "sent"
        assert app.host._current_chat_handoff_payload is None
