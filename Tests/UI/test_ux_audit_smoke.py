"""UX audit smoke tests for top-level shell navigation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, TextArea

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import apply_current_handoff_context
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.notes_scope_models import ScopeType, WorkspaceSubview
from tldw_chatbook.UI.Screens.notes_screen import NotesScreen
from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved
from tldw_chatbook.UI.Screens.chatbooks_screen import ChatbooksScreen
from tldw_chatbook.Widgets.Chat_Widgets.chat_handoff_card import ChatHandoffCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel


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

    async def fake_send_handler(chat_app, event):
        tab_container = app.tab_container
        assert tab_container is not None
        session = tab_container.sessions[tab_container.active_session_id]
        payload = session.session_data.handoff_payload

        chat_app._current_chat_handoff_payload = payload
        observed["wrapped_prompt"] = apply_current_handoff_context(
            chat_app,
            "Summarize the usability issue.",
        )
        observed["active_handoff_title"] = chat_app._current_chat_handoff_payload.title

        payload.status = "sent"
        chat_app._current_chat_handoff_payload = None

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


def _empty_notes_service() -> Mock:
    service = Mock()
    service.list_notes.return_value = []
    return service


class InvalidNotesSelectionSmokeApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        app_instance = SimpleNamespace(
            notes_service=_empty_notes_service(),
            notes_user_id="default_user",
            notes_scope_service=None,
            server_notes_workspace_service=None,
            notify=Mock(),
            open_chat_with_handoff=Mock(),
            open_study_screen=Mock(),
            open_notes_workspace=Mock(),
            call_from_thread=Mock(),
            loguru_logger=Mock(),
            current_selected_note_id=None,
            current_selected_note_version=None,
            current_selected_note_title="",
            current_selected_note_content="",
        )
        self.app_instance = app_instance
        self.screen_under_test = NotesScreen(app_instance)

    def on_mount(self) -> None:
        self.push_screen(self.screen_under_test)


@pytest.mark.asyncio
async def test_invalid_notes_and_workspace_handoffs_do_not_stage_chat_in_smoke():
    app = InvalidNotesSelectionSmokeApp()

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        screen = app.screen_under_test

        note_button = screen.query_one("#notes-use-in-chat-button", Button)
        assert note_button.disabled is True
        assert "Select a note" in str(note_button.tooltip)

        note_button.press()
        await pilot.pause(0.05)

        app.app_instance.open_chat_with_handoff.assert_not_called()

        screen._set_state(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.SOURCES,
            selected_workspace_id="workspace-1",
            selected_workspace_source_id=None,
            selected_workspace_artifact_id=None,
        )
        await pilot.pause(0.05)

        workspace_button = screen.query_one("#workspace-use-in-chat-button", Button)
        source_button = screen.query_one("#workspace-source-use-in-chat-button", Button)
        artifact_button = screen.query_one("#workspace-artifact-use-in-chat-button", Button)

        assert workspace_button.disabled is False
        assert source_button.disabled is True
        assert "Select a workspace source" in str(source_button.tooltip)
        assert artifact_button.disabled is True
        assert "Select a workspace artifact" in str(artifact_button.tooltip)

        source_button.press()
        artifact_button.press()
        await pilot.pause(0.05)

        app.app_instance.open_chat_with_handoff.assert_not_called()


class InvalidMediaSelectionSmokeApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.use_in_chat_requests = 0
        self.panel = MediaViewerPanel(SimpleNamespace(notify=Mock()))

    def compose(self) -> ComposeResult:
        yield self.panel

    @on(MediaViewerPanel.UseInChatRequested)
    def handle_media_use_in_chat(self, event: MediaViewerPanel.UseInChatRequested) -> None:
        self.use_in_chat_requests += 1


@pytest.mark.asyncio
async def test_invalid_media_handoff_selection_does_not_post_request_in_smoke():
    app = InvalidMediaSelectionSmokeApp()

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)

        button = app.panel.query_one("#media-use-in-chat-button", Button)
        assert button.disabled is True
        assert "Select a media item before using it in Chat" in str(button.tooltip)

        button.press()
        await pilot.pause(0.05)

        assert app.use_in_chat_requests == 0
        assert app.panel._build_use_in_chat_event() is None
