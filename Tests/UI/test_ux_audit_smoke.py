"""UX audit smoke tests for top-level shell navigation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, TextArea

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import apply_current_handoff_context
from tldw_chatbook.UI.MediaWindow_v2 import MediaWindow
from tldw_chatbook.UI.SearchWindow import SearchWindow
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState
from tldw_chatbook.UI.Screens.notes_scope_models import ScopeType, WorkspaceSubview
from tldw_chatbook.UI.Screens.notes_screen import NotesScreen
from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved
from tldw_chatbook.UI.Screens.chatbooks_screen import ChatbooksScreen
from tldw_chatbook.UI.Views.RAGSearch import search_rag_window
from tldw_chatbook.UI.Views.RAGSearch.search_rag_window import SearchRAGWindow
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


@pytest.mark.asyncio
async def test_valid_notes_and_workspace_handoffs_stage_app_payloads_in_smoke():
    app = InvalidNotesSelectionSmokeApp()

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        screen = app.screen_under_test

        screen._set_state(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=7,
            selected_note_version=2,
            selected_note_title="Draft Note",
            selected_note_content="Saved body",
        )
        screen.query_one("#notes-editor-area", TextArea).text = "Visible draft body"
        await pilot.pause(0.05)

        note_button = screen.query_one("#notes-use-in-chat-button", Button)
        assert note_button.disabled is False

        note_button.press()
        await pilot.pause(0.05)

        note_payload = app.app_instance.open_chat_with_handoff.call_args.args[0]
        assert note_payload.source == "notes"
        assert note_payload.source_id == "7"
        assert note_payload.title == "Draft Note"
        assert note_payload.body == "Visible draft body"

        app.app_instance.open_chat_with_handoff.reset_mock()
        screen._workspace_context_payload = {
            "workspace": {"id": "workspace-1", "name": "Research"},
            "notes": [
                {
                    "id": "note-1",
                    "title": "Workspace Note",
                    "content": "Workspace note body",
                    "version": 5,
                }
            ],
            "sources": [
                {
                    "id": "source-1",
                    "title": "Transcript",
                    "source_type": "video",
                    "url": "https://example.com/transcript",
                }
            ],
            "artifacts": [
                {
                    "id": "artifact-1",
                    "title": "Quiz Outline",
                    "content": "Artifact body",
                    "version": 3,
                }
            ],
        }
        screen._set_state(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.DETAILS,
            selected_workspace_id="workspace-1",
            selected_workspace_note_id="note-1",
            selected_workspace_source_id="source-1",
            selected_workspace_artifact_id="artifact-1",
        )
        await pilot.pause(0.05)

        workspace_button = screen.query_one("#workspace-use-in-chat-button", Button)
        assert workspace_button.disabled is False

        workspace_button.press()
        await pilot.pause(0.05)

        workspace_payload = app.app_instance.open_chat_with_handoff.call_args.args[0]
        assert workspace_payload.source == "workspace"
        assert workspace_payload.item_type == "workspace"
        assert workspace_payload.source_id == "workspace-1"
        assert workspace_payload.title == "Research"

        app.app_instance.open_chat_with_handoff.reset_mock()
        screen._set_state(workspace_subview=WorkspaceSubview.NOTES)
        screen.query_one("#notes-editor-area", TextArea).text = "Visible workspace note body"
        await pilot.pause(0.05)

        workspace_note_button = screen.query_one("#notes-use-in-chat-button", Button)
        assert workspace_note_button.disabled is False

        workspace_note_button.press()
        await pilot.pause(0.05)

        workspace_note_payload = app.app_instance.open_chat_with_handoff.call_args.args[0]
        assert workspace_note_payload.source == "workspace"
        assert workspace_note_payload.item_type == "workspace-note"
        assert workspace_note_payload.source_id == "note-1"
        assert workspace_note_payload.body == "Visible workspace note body"

        app.app_instance.open_chat_with_handoff.reset_mock()
        screen._set_state(workspace_subview=WorkspaceSubview.SOURCES)
        await pilot.pause(0.05)

        source_button = screen.query_one("#workspace-source-use-in-chat-button", Button)
        artifact_button = screen.query_one("#workspace-artifact-use-in-chat-button", Button)
        assert source_button.disabled is False
        assert artifact_button.disabled is False

        source_button.press()
        await pilot.pause(0.05)

        source_payload = app.app_instance.open_chat_with_handoff.call_args.args[0]
        assert source_payload.source == "workspace"
        assert source_payload.item_type == "workspace-source"
        assert source_payload.workspace_id == "workspace-1"
        assert source_payload.source_id == "source-1"
        assert source_payload.title == "Transcript"

        app.app_instance.open_chat_with_handoff.reset_mock()
        artifact_button.press()
        await pilot.pause(0.05)

        artifact_payload = app.app_instance.open_chat_with_handoff.call_args.args[0]
        assert artifact_payload.source == "workspace"
        assert artifact_payload.item_type == "workspace-artifact"
        assert artifact_payload.workspace_id == "workspace-1"
        assert artifact_payload.source_id == "artifact-1"
        assert artifact_payload.body == "Artifact body"


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


class ValidMediaWindowHandoffSmokeApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.app_instance = SimpleNamespace(
            _media_types_for_ui=[],
            media_runtime_state=MediaRuntimeState(runtime_backend="local"),
            media_reading_scope_service=SimpleNamespace(
                search_media=AsyncMock(return_value={"items": [], "total": 0}),
            ),
            notify=Mock(),
            open_chat_with_handoff=Mock(),
            media_db=None,
        )
        self.window = MediaWindow(self.app_instance)

    def compose(self) -> ComposeResult:
        yield self.window


@pytest.mark.asyncio
async def test_valid_media_handoff_replays_from_mounted_window_to_app_seam():
    app = ValidMediaWindowHandoffSmokeApp()

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        app.window.viewer_panel.load_media(
            {
                "id": "media-1",
                "title": "Lecture",
                "content": "Transcript body",
                "media_type": "video",
            }
        )
        await pilot.pause(0.05)

        button = app.window.viewer_panel.query_one("#media-use-in-chat-button", Button)
        assert button.disabled is False

        button.press()
        await pilot.pause(0.05)

        payload = app.app_instance.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "media"
        assert payload.source_id == "media-1"
        assert payload.title == "Lecture"
        assert payload.body == "Transcript body"


class SearchRAGHandoffSmokeApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.app_instance = SimpleNamespace(
            notify=Mock(),
            api_endpoint="test-endpoint",
            get_authoritative_runtime_source=Mock(return_value="server"),
            open_chat_with_handoff=Mock(),
        )
        self.window = SearchRAGWindow(self.app_instance)

    def compose(self) -> ComposeResult:
        yield self.window


@pytest.mark.asyncio
async def test_valid_rag_search_handoff_replays_from_mounted_window_to_app_seam(tmp_path):
    with (
        patch.dict(search_rag_window.DEPENDENCIES_AVAILABLE, {"embeddings_rag": True}, clear=False),
        patch(
            "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_user_data_dir",
            return_value=tmp_path,
        ),
        patch(
            "tldw_chatbook.UI.Views.RAGSearch.saved_searches_panel.get_user_data_dir",
            return_value=tmp_path,
        ),
    ):
        app = SearchRAGHandoffSmokeApp()

        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause(0.1)
            app.window.search_results = [
                {
                    "title": "Retrieved Chunk",
                    "content": "Evidence body",
                    "source": "notes",
                    "score": 0.91,
                    "metadata": {"document_id": "doc-1"},
                }
            ]
            app.window.total_results = 1
            await app.window._display_results()
            await pilot.pause(0.05)

            button = app.window.query_one("#use-in-chat-0", Button)
            button.press()
            await pilot.pause(0.05)

            payload = app.app_instance.open_chat_with_handoff.call_args.args[0]
            assert payload.source == "search-rag"
            assert payload.item_type == "rag-result"
            assert payload.runtime_backend == "server"
            assert payload.title == "Retrieved Chunk"
            assert payload.body == "Evidence body"


class WebSearchResultHandoffSmokeApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.app_instance = SimpleNamespace(
            notify=Mock(),
            api_endpoint="test-endpoint",
            search_active_sub_tab=None,
            get_authoritative_runtime_source=Mock(return_value="local"),
            open_chat_with_handoff=Mock(),
        )
        self.window = SearchWindow(self.app_instance)

    def compose(self) -> ComposeResult:
        yield self.window


@pytest.mark.asyncio
async def test_valid_web_search_handoff_replays_from_mounted_window_to_app_seam(monkeypatch):
    monkeypatch.setattr("tldw_chatbook.UI.SearchWindow.WEB_SEARCH_AVAILABLE", True)
    app = WebSearchResultHandoffSmokeApp()

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        app.window.web_search_results = [
            {
                "title": "Article",
                "content": "Snippet body",
                "source": "web",
                "metadata": {"url": "https://example.com", "displayUrl": "example.com"},
            }
        ]
        await app.window._render_web_search_result_cards()
        await pilot.pause(0.05)

        button = app.window.query_one("#use-in-chat-0", Button)
        button.press()
        await pilot.pause(0.05)

        payload = app.app_instance.open_chat_with_handoff.call_args.args[0]
        assert payload.source == "search-web"
        assert payload.item_type == "web-result"
        assert payload.metadata["url"] == "https://example.com"
        assert payload.body == "Snippet body"
