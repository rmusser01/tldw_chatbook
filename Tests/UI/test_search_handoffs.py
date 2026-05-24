from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Markdown

from tldw_chatbook.runtime_policy.engine import PolicyEngine
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI import SearchWindow as search_window_module
from tldw_chatbook.UI.Views.RAGSearch import search_rag_window
from tldw_chatbook.UI.SearchWindow import SearchWindow
from tldw_chatbook.UI.Views.RAGSearch.search_rag_window import SearchRAGWindow
from tldw_chatbook.UI.Views.RAGSearch.search_handoff import (
    build_library_rag_evidence_bundle,
    build_library_rag_console_live_work_payload,
)
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


def test_library_rag_console_payload_preserves_evidence_fields():
    result = {
        "result_id": "note-42:chunk-7",
        "title": "Incident Review",
        "snippet": "Expired credential caused the incident.",
        "source_id": "note-42",
        "chunk_id": "chunk-7",
        "score": 0.93,
        "runtime_backend": "local-fts",
        "citations": [{"label": "Incident Review p.2"}],
    }

    payload = build_library_rag_console_live_work_payload(
        result,
        query="Why did the incident happen?",
    )

    assert {
        key: payload[key]
        for key in (
            "target_id",
            "result_id",
            "query",
            "title",
            "source_id",
            "chunk_id",
            "snippet",
            "citations",
            "score",
            "runtime_backend",
            "source_authority",
            "source_selector_state",
        )
    } == {
        "target_id": "local:library-rag:note-42:chunk-7",
        "result_id": "note-42:chunk-7",
        "query": "Why did the incident happen?",
        "title": "Incident Review",
        "source_id": "note-42",
        "chunk_id": "chunk-7",
        "snippet": "Expired credential caused the incident.",
        "citations": ["Incident Review p.2"],
        "score": 0.93,
        "runtime_backend": "local-fts",
        "source_authority": "local",
        "source_selector_state": "local",
    }
    bundle = payload["evidence_bundle"]
    reference = bundle["references"][0]
    assert bundle["query"] == "Why did the incident happen?"
    assert bundle["status"] == "available"
    assert reference["evidence_id"] == "S1"
    assert reference["source_id"] == "note-42"
    assert reference["snippet"] == "Expired credential caused the incident."
    assert reference["authority_label"] == "Source authority: local"
    assert reference["metadata"]["active_context_eligible"] is True
    assert reference["metadata"]["global_browse_visible"] is True


def test_library_rag_evidence_bundle_blocks_cross_workspace_context():
    bundle = build_library_rag_evidence_bundle(
        {
            "result_id": "note-42:chunk-7",
            "title": "Workspace B Note",
            "snippet": "Workspace B evidence remains visible.",
            "source_id": "note-42",
            "chunk_id": "chunk-7",
            "source_type": "note",
            "runtime_backend": "local-fts",
            "workspace_ids": ("workspace-b",),
            "active_workspace_id": "workspace-a",
        },
        query="Can I use this in Workspace A?",
    )

    payload = bundle.to_payload()
    reference = payload["references"][0]
    assert payload["status"] == "blocked"
    assert reference["status"] == "blocked"
    assert reference["workspace_id"] == "workspace-b"
    assert reference["authority_label"] == (
        "Workspace: workspace-b (blocked for active workspace workspace-a)"
    )
    assert reference["metadata"]["global_browse_visible"] is True
    assert reference["metadata"]["active_context_eligible"] is False
    assert reference["metadata"]["eligibility_reason"] == "cross_workspace"
    assert reference["metadata"]["active_workspace_id"] == "workspace-a"


def test_library_rag_evidence_bundle_preserves_provenance_identity():
    bundle = build_library_rag_evidence_bundle(
        {
            "title": "Server Transcript",
            "snippet": "The server transcript contains the source evidence.",
            "provenance": {
                "source_id": "media-9",
                "chunk_id": "chunk-3",
                "source_type": "media",
                "runtime_backend": "server-rag",
            },
        },
        query="What does the transcript say?",
    )

    payload = bundle.to_payload()
    reference = payload["references"][0]
    assert reference["source_id"] == "media-9"
    assert reference["source_type"] == "media"
    assert reference["source_owner"] == "server"
    assert reference["content_ref"] == "server:library-rag:media-9:chunk-3"
    assert reference["metadata"]["chunk_id"] == "chunk-3"
    assert reference["metadata"]["runtime_backend"] == "server-rag"


def test_library_rag_evidence_bundle_marks_empty_results_missing():
    bundle = build_library_rag_evidence_bundle(
        [],
        query="What evidence is available?",
    )

    payload = bundle.to_payload()
    assert payload["status"] == "missing"
    assert payload["references"] == []
    assert payload["metadata"]["eligible_reference_count"] == 0
    assert payload["metadata"]["blocked_reference_count"] == 0


@pytest.mark.parametrize("status", ("stale", "unknown"))
def test_library_rag_evidence_bundle_preserves_non_missing_reference_status(status):
    bundle = build_library_rag_evidence_bundle(
        {
            "title": "Indexed source",
            "snippet": "Existing evidence has a non-ready state.",
            "source_id": f"{status}-source",
            "evidence_status": status,
        },
        query=f"Show {status} evidence",
    )

    payload = bundle.to_payload()
    reference = payload["references"][0]
    assert payload["status"] == status
    assert reference["status"] == status


def test_library_rag_evidence_bundle_drops_out_of_range_scores():
    payload = build_library_rag_console_live_work_payload(
        {
            "title": "Out of range score",
            "snippet": "Score should not prevent staging evidence.",
            "source_id": "note-99",
            "score": 1.5,
        },
        query="Can this evidence stage?",
    )

    reference = payload["evidence_bundle"]["references"][0]
    assert payload["score"] is None
    assert "score" not in reference


def test_library_rag_console_payload_uses_shared_validation_for_unsafe_text():
    result = {
        "result_id": "note-42:chunk-7",
        "title": "<script>alert('bad')</script>",
        "snippet": "javascript:alert(1)",
        "source_id": "note-42<script>",
        "chunk_id": "chunk-7",
        "runtime_backend": "server-rag",
        "citations": [{"label": "onclick=bad"}],
    }

    payload = build_library_rag_console_live_work_payload(
        result,
        query="javascript:alert(1)",
    )

    assert payload["target_id"] == "server:library-rag:note-42:chunk-7"
    assert payload["result_id"] == "note-42:chunk-7"
    assert payload["query"] == ""
    assert payload["title"] == "Untitled source"
    assert payload["source_id"] == ""
    assert payload["chunk_id"] == "chunk-7"
    assert payload["snippet"] == ""
    assert payload["citations"] == []
    assert payload["source_authority"] == "server"
    assert payload["target_id"].startswith(f"{payload['source_authority']}:")


def test_library_rag_console_payload_target_prefix_matches_source_authority():
    local_payload = build_library_rag_console_live_work_payload(
        {
            "result_id": "local-note:chunk-1",
            "title": "Local Note",
            "runtime_backend": "local-fts",
        },
        query="local query",
    )
    server_payload = build_library_rag_console_live_work_payload(
        {
            "result_id": "server-note:chunk-1",
            "title": "Server Note",
            "runtime_backend": "server-rag",
        },
        query="server query",
    )

    assert local_payload["source_authority"] == "local"
    assert local_payload["target_id"].startswith("local:library-rag:")
    assert local_payload["target_id"].startswith(f"{local_payload['source_authority']}:")
    assert server_payload["source_authority"] == "server"
    assert server_payload["target_id"].startswith("server:library-rag:")
    assert server_payload["target_id"].startswith(f"{server_payload['source_authority']}:")


def test_library_rag_console_payload_helper_documents_contract():
    docstring = inspect.getdoc(build_library_rag_console_live_work_payload)

    assert docstring is not None
    assert "Args:" in docstring
    assert "Returns:" in docstring


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


def test_search_rag_window_use_in_chat_policy_block_explains_recovery(tmp_path):
    app = _search_app(runtime_backend="server")
    app.runtime_policy = SimpleNamespace(state=RuntimeSourceState(active_source="local"))
    app.ui_policy_engine = PolicyEngine(CAPABILITY_REGISTRY)
    with patch(
        "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_user_data_dir",
        return_value=tmp_path,
    ):
        window = SearchRAGWindow(app_instance=app)
    event = SearchResult.UseInChatRequested(
        0,
        {
            "title": "Server Chunk",
            "content": "Retrieved text",
            "source": "notes",
            "metadata": {"document_id": "doc-1"},
        },
    )

    window.handle_search_result_use_in_chat(event)

    app.open_chat_with_handoff.assert_not_called()
    message = app.notify.call_args.args[0]
    assert "rag.media_embeddings.search.server requires server mode" in message
    assert "switch source" in message.lower()
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


def test_search_window_dedicated_web_result_policy_block_explains_recovery():
    app = _search_app(runtime_backend="local")
    app.runtime_policy = SimpleNamespace(state=RuntimeSourceState(active_source="local"))
    app.ui_policy_engine = PolicyEngine(CAPABILITY_REGISTRY)
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

    app.open_chat_with_handoff.assert_not_called()
    message = app.notify.call_args.args[0]
    assert "research.search.providers.launch.server requires server mode" in message
    assert "switch source" in message.lower()
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
