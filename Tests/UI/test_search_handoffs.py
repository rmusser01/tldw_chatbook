from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Markdown

from tldw_chatbook.UI import SearchWindow as search_window_module
from tldw_chatbook.UI.Views.RAGSearch import search_rag_window
from tldw_chatbook.UI.SearchWindow import SearchWindow
from tldw_chatbook.UI.Views.RAGSearch.search_rag_window import SearchRAGWindow
from tldw_chatbook.UI.Views.RAGSearch.search_result import SearchResult


def _search_app(runtime_backend: str = "local") -> Mock:
    app = Mock()
    app.notify = Mock()
    app.api_endpoint = "test-endpoint"
    app.search_active_sub_tab = None
    app.get_authoritative_runtime_source = Mock(return_value=runtime_backend)
    app.open_chat_with_handoff = Mock()
    return app


def test_search_result_builds_use_in_chat_event_with_result_data():
    result = {"title": "Doc", "content": "Snippet", "source": "notes", "score": 0.8}
    card = SearchResult(result, 0)

    event = card._build_use_in_chat_event()

    assert event.index == 0
    assert event.result["title"] == "Doc"


def test_search_window_normalizes_rag_result_payload(tmp_path):
    app = _search_app(runtime_backend="server")
    with patch(
        "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_user_data_dir",
        return_value=tmp_path,
    ):
        window = SearchRAGWindow(app_instance=app)
    result = {
        "title": "Chunk",
        "content": "Retrieved text",
        "source": "notes",
        "score": 0.91,
        "metadata": {"document_id": "doc-1"},
    }

    payload = window._build_search_chat_handoff_payload(result)

    assert payload.source == "search-rag"
    assert payload.item_type == "rag-result"
    assert payload.discovery_owner == "rag_search"
    assert payload.runtime_backend == "server"
    assert payload.source_selector_state == "server"
    assert payload.body == "Retrieved text"
    assert payload.metadata["score"] == 0.91


def test_search_window_normalizes_web_result_payload(tmp_path):
    app = _search_app(runtime_backend="local")
    with patch(
        "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_user_data_dir",
        return_value=tmp_path,
    ):
        window = SearchRAGWindow(app_instance=app)
    result = {
        "title": "Article",
        "content": "Snippet",
        "source": "web",
        "metadata": {"url": "https://example.com", "displayUrl": "example.com"},
    }

    payload = window._build_search_chat_handoff_payload(result)

    assert payload.source == "search-web"
    assert payload.item_type == "web-result"
    assert payload.discovery_owner == "web_search"
    assert payload.source_owner == "server"
    assert payload.source_selector_state == "server"
    assert payload.metadata["url"] == "https://example.com"


def test_search_rag_window_use_in_chat_handler_routes_to_app(tmp_path):
    app = _search_app(runtime_backend="local")
    with patch(
        "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_user_data_dir",
        return_value=tmp_path,
    ):
        window = SearchRAGWindow(app_instance=app)
    event = SearchResult.UseInChatRequested(
        0,
        {"title": "Chunk", "content": "Retrieved text", "source": "notes"},
    )

    window.handle_search_result_use_in_chat(event)

    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "search-rag"
    assert payload.body == "Retrieved text"


def test_search_rag_window_use_in_chat_unavailable_explains_recovery(tmp_path):
    app = _search_app(runtime_backend="local")
    app.open_chat_with_handoff = None
    with patch(
        "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_user_data_dir",
        return_value=tmp_path,
    ):
        window = SearchRAGWindow(app_instance=app)
    event = SearchResult.UseInChatRequested(
        0,
        {"title": "Chunk", "content": "Retrieved text", "source": "notes"},
    )

    window.handle_search_result_use_in_chat(event)

    message = app.notify.call_args.args[0]
    assert "Use in Chat is unavailable" in message
    assert "Open Chat" in message
    assert "try again" in message
    assert app.notify.call_args.kwargs["severity"] == "warning"


@pytest.mark.asyncio
async def test_search_rag_window_web_search_runs_bing_call_in_thread(tmp_path):
    app = _search_app(runtime_backend="local")
    with patch(
        "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_user_data_dir",
        return_value=tmp_path,
    ):
        window = SearchRAGWindow(app_instance=app)
    search_bing = Mock(return_value={"raw": "bing"})

    with (
        patch.object(search_rag_window, "WEB_SEARCH_AVAILABLE", True),
        patch.object(search_rag_window, "search_web_bing", search_bing),
        patch.object(
            search_rag_window,
            "parse_bing_results",
            return_value=[
                {
                    "name": "Article",
                    "snippet": "Snippet",
                    "url": "https://example.com",
                    "displayUrl": "example.com",
                }
            ],
        ),
        patch.object(search_rag_window.asyncio, "to_thread", AsyncMock(return_value={"raw": "bing"})) as to_thread,
    ):
        results = await window._perform_web_search("agent handoff")

    to_thread.assert_awaited_once_with(search_bing, "agent handoff")
    assert results[0]["title"] == "Article"
    assert results[0]["metadata"]["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_search_window_renders_dedicated_web_results_as_cards():
    app = _search_app(runtime_backend="local")
    window = SearchWindow(app_instance=app)
    result = {
        "title": "Article",
        "content": "Snippet",
        "source": "web",
        "metadata": {"url": "https://example.com"},
    }
    window.web_search_results = [result]
    result_container = Mock()
    result_container.remove_children = AsyncMock()
    result_container.mount = AsyncMock()
    window.query_one = Mock(return_value=result_container)

    await window._render_web_search_result_cards()

    result_container.remove_children.assert_awaited_once()
    mounted_card = result_container.mount.call_args.args[0]
    assert isinstance(mounted_card, SearchResult)
    assert mounted_card.result["title"] == "Article"


def test_search_window_dedicated_web_result_handoff_routes_to_app():
    app = _search_app(runtime_backend="local")
    window = SearchWindow(app_instance=app)
    event = SearchResult.UseInChatRequested(
        0,
        {
            "title": "Article",
            "content": "Snippet",
            "source": "web",
            "metadata": {"url": "https://example.com"},
        },
    )

    window.handle_search_result_use_in_chat(event)

    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "search-web"
    assert payload.metadata["url"] == "https://example.com"


def test_search_window_dedicated_web_result_unavailable_explains_recovery():
    app = _search_app(runtime_backend="local")
    app.open_chat_with_handoff = None
    window = SearchWindow(app_instance=app)
    event = SearchResult.UseInChatRequested(
        0,
        {
            "title": "Article",
            "content": "Snippet",
            "source": "web",
            "metadata": {"url": "https://example.com"},
        },
    )

    window.handle_search_result_use_in_chat(event)

    message = app.notify.call_args.args[0]
    assert "Use in Chat is unavailable" in message
    assert "Open Chat" in message
    assert "try again" in message
    assert app.notify.call_args.kwargs["severity"] == "warning"


@pytest.mark.asyncio
async def test_search_window_disabled_web_search_nav_explains_dependency_recovery(monkeypatch):
    monkeypatch.setattr(search_window_module, "WEB_SEARCH_AVAILABLE", False)
    app_instance = _search_app(runtime_backend="local")

    class SearchWindowApp(App):
        def compose(self) -> ComposeResult:
            yield SearchWindow(app_instance=app_instance)

    app = SearchWindowApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()

        disabled_nav = app.query_one("#search-nav-web-search-disabled", Button)
        disabled_message = app.query_one("#search-view-web-search Markdown", Markdown)

        assert disabled_nav.disabled is True
        assert "Web Search requires optional dependencies" in str(disabled_nav.tooltip)
        assert 'pip install -e ".[websearch]"' in str(disabled_nav.tooltip)
        assert "Web Search requires optional dependencies" in disabled_message.source
        assert 'pip install -e ".[websearch]"' in disabled_message.source
