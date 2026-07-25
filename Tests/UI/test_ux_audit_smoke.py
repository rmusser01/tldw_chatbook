"""UX audit smoke tests for top-level shell navigation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Select

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import (
    handle_chat_send_button_pressed,
)
from tldw_chatbook.runtime_policy.engine import PolicyEngine
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.MediaWindow_v2 import MediaWindow
from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState
from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved
from tldw_chatbook.UI.Screens.chatbooks_screen import ChatbooksScreen
from tldw_chatbook.UI.Views.RAGSearch import search_rag_window
from tldw_chatbook.UI.Views.RAGSearch.search_rag_window import SearchRAGWindow
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
        assert app.screen.query_one("#nav-console") is not None
        assert app.screen.query_one("#nav-artifacts") is not None


@pytest.mark.asyncio
async def test_tabbed_chat_send_reads_session_specific_input_and_log():
    chat_container = SimpleNamespace(
        is_mounted=True,
        mount=AsyncMock(),
        children=[],
        query=Mock(return_value=[]),
    )
    selectors_seen: list[str] = []

    def fake_widget(**attrs):
        return SimpleNamespace(**attrs)

    widgets = {
        "#chat-input-tab-a": fake_widget(text="Hello from a tab", focus=Mock()),
        "#chat-log-tab-a": chat_container,
        "#chat-api-provider": fake_widget(value=Select.BLANK),
        "#chat-api-model": fake_widget(value=Select.BLANK),
        "#chat-system-prompt": fake_widget(text=""),
        "#chat-temperature": fake_widget(value="0.7"),
        "#chat-top-p": fake_widget(value="0.95"),
        "#chat-min-p": fake_widget(value="0.05"),
        "#chat-top-k": fake_widget(value="50"),
        "#chat-llm-max-tokens": fake_widget(value="1024"),
        "#chat-llm-seed": fake_widget(value=""),
        "#chat-llm-stop": fake_widget(value=""),
        "#chat-llm-response-format": fake_widget(value=Select.BLANK),
        "#chat-llm-n": fake_widget(value="1"),
        "#chat-llm-user-identifier": fake_widget(value=""),
        "#chat-llm-logprobs": fake_widget(value=False),
        "#chat-llm-top-logprobs": fake_widget(value="0"),
        "#chat-llm-logit-bias": fake_widget(text="{}"),
        "#chat-llm-presence-penalty": fake_widget(value="0.0"),
        "#chat-llm-frequency-penalty": fake_widget(value="0.0"),
        "#chat-llm-tools": fake_widget(text="[]"),
        "#chat-llm-tool-choice": fake_widget(value="auto"),
        "#chat-llm-fixed-tokens-kobold": fake_widget(value=False),
        "#chat-strip-thinking-tags-checkbox": fake_widget(value=True),
        "#chat-streaming-enabled-checkbox": fake_widget(value=False),
    }

    class FakeScreen:
        def query_one(self, selector, _widget_type=None):
            selectors_seen.append(selector)
            if selector not in widgets:
                raise AssertionError(f"Unexpected selector: {selector}")
            return widgets[selector]

    app = SimpleNamespace(
        screen=FakeScreen(),
        current_chat_worker=None,
        _current_chat_tab_id="tab-a",
        app_config={"api_settings": {}},
        current_chat_active_character_data=None,
        current_chat_conversation_id=None,
        chachanotes_db=None,
        API_IMPORTS_SUCCESSFUL=True,
    )

    await handle_chat_send_button_pressed(
        app, SimpleNamespace(button=SimpleNamespace(id="send-stop-chat-tab-a"))
    )

    assert selectors_seen[:2] == ["#chat-input-tab-a", "#chat-log-tab-a"]
    assert "#chat-input" not in selectors_seen
    assert "#chat-log" not in selectors_seen
    chat_container.mount.assert_awaited_once()


def _assert_single_handoff_payload(open_chat_with_handoff: Mock) -> ChatHandoffPayload:
    open_chat_with_handoff.assert_called_once()
    return open_chat_with_handoff.call_args.args[0]


class InvalidMediaSelectionSmokeApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.use_in_chat_requests = 0
        self.panel = MediaViewerPanel(SimpleNamespace(notify=Mock()))

    def compose(self) -> ComposeResult:
        yield self.panel

    @on(MediaViewerPanel.UseInChatRequested)
    def handle_media_use_in_chat(
        self, event: MediaViewerPanel.UseInChatRequested
    ) -> None:
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

        payload = _assert_single_handoff_payload(
            app.app_instance.open_chat_with_handoff
        )
        assert payload.source == "media"
        assert payload.source_id == "media-1"
        assert payload.title == "Lecture"
        assert payload.body == "Transcript body"


@pytest.mark.asyncio
async def test_contract_blocked_media_handoff_explains_recovery_without_staging():
    app = ValidMediaWindowHandoffSmokeApp()
    app.app_instance.runtime_policy = SimpleNamespace(
        state=RuntimeSourceState(active_source="local")
    )
    app.app_instance.ui_policy_engine = PolicyEngine(CAPABILITY_REGISTRY)

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        app.window.viewer_panel.load_media(
            {
                "id": "media-1",
                "backend": "server",
                "title": "Server Lecture",
                "content": "Transcript body",
                "media_type": "video",
            }
        )
        await pilot.pause(0.05)

        button = app.window.viewer_panel.query_one("#media-use-in-chat-button", Button)
        assert button.disabled is False

        button.press()
        await pilot.pause(0.05)

        app.app_instance.open_chat_with_handoff.assert_not_called()
        message = app.app_instance.notify.call_args.args[0]
        assert "media.items.detail.server requires server mode" in message
        assert "source authority: runtime_policy/server" in message.lower()
        assert "ux interop: active source local" in message.lower()
        assert "switch source to server" in message.lower()


class SearchRAGHandoffSmokeApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.app_instance = SimpleNamespace(
            notify=Mock(),
            api_endpoint="test-endpoint",
            get_authoritative_runtime_source=Mock(return_value="server"),
            open_chat_with_handoff=Mock(),
            open_console_for_live_work=Mock(),
        )
        self.window = SearchRAGWindow(self.app_instance)

    def compose(self) -> ComposeResult:
        yield self.window


@pytest.mark.asyncio
async def test_valid_rag_search_handoff_replays_from_mounted_window_to_app_seam(
    tmp_path,
):
    with (
        patch.dict(
            search_rag_window.DEPENDENCIES_AVAILABLE,
            {"embeddings_rag": True},
            clear=False,
        ),
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

            payload = _assert_single_handoff_payload(
                app.app_instance.open_chat_with_handoff
            )
            assert payload.source == "search-rag"
            assert payload.item_type == "rag-result"
            assert payload.runtime_backend == "server"
            assert payload.title == "Retrieved Chunk"
            assert payload.body == "Evidence body"


@pytest.mark.asyncio
async def test_valid_rag_search_console_launch_replays_from_mounted_window_to_app_seam(
    tmp_path,
):
    with (
        patch.dict(
            search_rag_window.DEPENDENCIES_AVAILABLE,
            {"embeddings_rag": True},
            clear=False,
        ),
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
                    "metadata": {"document_id": "doc-1", "chunk_id": "chunk-7"},
                }
            ]
            app.window.total_results = 1
            await app.window._display_results()
            await pilot.pause(0.05)

            button = app.window.query_one("#use-in-console-0", Button)
            button.press()
            await pilot.pause(0.05)

            app.app_instance.open_chat_with_handoff.assert_not_called()
            app.app_instance.open_console_for_live_work.assert_called_once_with(
                source="RAG",
                title="Retrieved Chunk",
                payload={
                    "target_id": "search-rag:doc-1",
                    "source_id": "doc-1",
                    "content_ref": "search-rag:doc-1",
                    "runtime_backend": "server",
                    "source": "notes",
                    "score": 0.91,
                    "display_summary": "Evidence body",
                    "suggested_prompt": "Use this retrieved result as context and answer or reason from it carefully.",
                },
                status="ready",
                recovery="Use this retrieved RAG result as Console context, or return to Search/RAG to adjust the query.",
                action_label="Ask from RAG result",
            )


@pytest.mark.asyncio
async def test_rag_search_console_launch_escapes_result_markup_before_staging(tmp_path):
    with (
        patch.dict(
            search_rag_window.DEPENDENCIES_AVAILABLE,
            {"embeddings_rag": True},
            clear=False,
        ),
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
                    "title": "[red]Retrieved[/red] <script>\x00",
                    "content": "[bold]Evidence[/bold] <script>alert(1)</script>\x00",
                    "source": "notes",
                    "score": 0.91,
                    "metadata": {"document_id": "doc-[red]-<script>"},
                }
            ]
            app.window.total_results = 1
            await app.window._display_results()
            await pilot.pause(0.05)

            app.window.query_one("#use-in-console-0", Button).press()
            await pilot.pause(0.05)

            app.app_instance.open_console_for_live_work.assert_called_once()
            call_kwargs = app.app_instance.open_console_for_live_work.call_args.kwargs
            assert call_kwargs["title"] == r"\[red]Retrieved\[/red] &lt;script&gt;"
            assert (
                call_kwargs["payload"]["target_id"]
                == r"search-rag:doc-\[red]-&lt;script&gt;"
            )
            assert call_kwargs["payload"]["source"] == "notes"
            assert call_kwargs["payload"]["display_summary"] == (
                r"\[bold]Evidence\[/bold] &lt;script&gt;alert(1)&lt;/script&gt;"
            )


@pytest.mark.asyncio
async def test_contract_blocked_rag_search_handoff_explains_recovery_without_staging(
    tmp_path,
):
    with (
        patch.dict(
            search_rag_window.DEPENDENCIES_AVAILABLE,
            {"embeddings_rag": True},
            clear=False,
        ),
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
        app.app_instance.runtime_policy = SimpleNamespace(
            state=RuntimeSourceState(active_source="local")
        )
        app.app_instance.ui_policy_engine = PolicyEngine(CAPABILITY_REGISTRY)

        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause(0.1)
            app.window.search_results = [
                {
                    "title": "Server Chunk",
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

            app.app_instance.open_chat_with_handoff.assert_not_called()
            message = app.app_instance.notify.call_args.args[0]
            assert "rag.media_embeddings.search.server requires server mode" in message
            assert "source authority: runtime_policy/server" in message.lower()
            assert "ux interop: active source local" in message.lower()
            assert "switch source to server" in message.lower()


@pytest.mark.asyncio
async def test_contract_blocked_rag_search_console_launch_explains_recovery_without_staging(
    tmp_path,
):
    with (
        patch.dict(
            search_rag_window.DEPENDENCIES_AVAILABLE,
            {"embeddings_rag": True},
            clear=False,
        ),
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
        app.app_instance.runtime_policy = SimpleNamespace(
            state=RuntimeSourceState(active_source="local")
        )
        app.app_instance.ui_policy_engine = PolicyEngine(CAPABILITY_REGISTRY)

        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause(0.1)
            app.window.search_results = [
                {
                    "title": "Server Chunk",
                    "content": "Evidence body",
                    "source": "notes",
                    "score": 0.91,
                    "metadata": {"document_id": "doc-1"},
                }
            ]
            app.window.total_results = 1
            await app.window._display_results()
            await pilot.pause(0.05)

            button = app.window.query_one("#use-in-console-0", Button)
            button.press()
            await pilot.pause(0.05)

            app.app_instance.open_console_for_live_work.assert_not_called()
            app.app_instance.open_chat_with_handoff.assert_not_called()
            message = app.app_instance.notify.call_args.args[0]
            assert "rag.media_embeddings.search.server requires server mode" in message
            assert "source authority: runtime_policy/server" in message.lower()
            assert "ux interop: active source local" in message.lower()
            assert "switch source to server" in message.lower()
