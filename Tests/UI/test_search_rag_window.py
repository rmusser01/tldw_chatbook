# test_search_rag_window.py
# Focused regression tests for the current Search/RAG widget contract

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Button, Checkbox, DataTable, Input, ListView, Select, Static

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.DB.search_history_db import SearchHistoryDB
from tldw_chatbook.RAG_Search.ingestion_indexing import (
    ITEM_TYPE_CONVERSATION,
    ITEM_TYPE_MEDIA,
    ITEM_TYPE_NOTE,
)
from tldw_chatbook.UI.SearchRAGWindow import (
    SearchHistoryDropdown,
    SearchRAGWindow,
    SearchResult,
    SavedSearchesPanel,
)
from tldw_chatbook.UI.Views.RAGSearch import search_rag_window as search_rag_window_module


def _text(widget: Static) -> str:
    """Return plain text from a Textual static-style widget."""
    rendered = widget.render()
    return getattr(rendered, "plain", str(rendered))


@pytest.fixture
def mock_app_instance() -> MagicMock:
    """Create the minimal app surface SearchRAGWindow expects."""
    app = MagicMock()
    app.notify = MagicMock()
    app.api_endpoint = "test-endpoint"
    return app


@pytest.fixture
def temp_user_data_dir(tmp_path: Path) -> Path:
    """Use a temp dir for Search/RAG persistence during tests."""
    return tmp_path


@pytest.fixture
def search_rag_test_env(temp_user_data_dir: Path):
    """Patch Search/RAG persistence and optional dependency state for tests."""
    with patch.dict(
        search_rag_window_module.DEPENDENCIES_AVAILABLE,
        {"embeddings_rag": True},
        clear=False,
    ):
        with patch(
            "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_user_data_dir",
            return_value=temp_user_data_dir,
        ):
            with patch(
                "tldw_chatbook.UI.Views.RAGSearch.saved_searches_panel.get_user_data_dir",
                return_value=temp_user_data_dir,
            ):
                yield


@pytest.mark.ui
class TestSearchRAGWindow:
    """Regression coverage for the current Search/RAG screen contract."""

    def test_window_initialization_uses_current_state_contract(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
    ) -> None:
        """The window should expose the current state model, not legacy aliases."""
        window = SearchRAGWindow(mock_app_instance, id="test-search-window")

        assert window.app_instance == mock_app_instance
        assert window.search_results == []
        assert window.current_page == 1
        assert window.results_per_page == 10
        assert window.total_results == 0
        assert window.is_searching is False
        assert window.current_search_mode == "plain"
        assert window.available_collections == []
        assert window.last_search_config is None

    def test_load_available_collections_does_not_touch_textual_widgets(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
    ) -> None:
        """Collection loading should be safe to run off the Textual UI thread."""
        window = SearchRAGWindow(mock_app_instance, id="test-search-window")
        window.query_one = MagicMock(side_effect=AssertionError("UI touched during load"))

        with patch(
            "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_available_profiles",
            return_value=["default", "research"],
        ):
            assert window._load_available_collections() == ["default", "research"]

    def test_handle_search_worker_runs_on_textual_thread(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
    ) -> None:
        """Search orchestration should keep UI mutations on the Textual thread."""
        window = SearchRAGWindow(mock_app_instance, id="test-search-window")
        window.run_worker = MagicMock()

        window.handle_search(MagicMock())

        assert window.run_worker.call_args.kwargs["exclusive"] is True
        assert window.run_worker.call_args.kwargs["thread"] is False

    @pytest.mark.asyncio
    async def test_apply_available_collections_updates_list_and_select(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Applying loaded collections should mutate Textual widgets on the UI thread."""
        with patch(
            "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_available_profiles",
            return_value=[],
        ):
            async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                window = pilot.app.test_widget

                await window._apply_available_collections(["default"])
                await pilot.pause()

                assert window.available_collections == ["default"]
                assert len(window.query_one("#collections-list", ListView).children) == 1
                assert window.query_one("#collection-select", Select) is not None

    @pytest.mark.asyncio
    async def test_compose_creates_current_ui_elements(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """The mounted window should expose the current search-first layout."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            assert window.query_one("#search-query-input", Input)
            assert window.query_one("#search-button", Button)
            assert window.query_one("#search-mode-select", Select)
            assert window.query_one("#advanced-options")
            assert window.query_one("#results-container-enhanced")
            assert window.query_one("#results-list-enhanced")
            assert window.query_one("#search-history-dropdown", SearchHistoryDropdown)
            assert window.query_one("#filter-media", Checkbox)
            assert window.query_one("#filter-conversations", Checkbox)
            assert window.query_one("#filter-notes", Checkbox)
            assert window.query_one("#saved-searches-panel", SavedSearchesPanel)

    @pytest.mark.asyncio
    async def test_initial_search_empty_state_explains_search_modes_collections_and_chat_handoff(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """First-run Search should explain modes, collection scope, and Chat handoff."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            empty_state = window.query_one("#search-empty-state", Static)
            empty_text = _text(empty_state)

            assert "Plain Search" in empty_text
            assert "RAG" in empty_text
            assert "collections" in empty_text
            assert "Use in Chat" in empty_text
            assert "Chat context" in empty_text

    @pytest.mark.asyncio
    async def test_primary_search_action_is_reachable_in_default_layout(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """The primary Search action should be visible without opening advanced controls."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            search_input = window.query_one("#search-query-input", Input)
            search_button = window.query_one("#search-button", Button)

            assert search_input.display is True
            assert search_button.display is True
            assert search_button.disabled is False
            assert search_button.region.y + search_button.region.height <= pilot.app.size.height

            await pilot.click("#search-button")

    @pytest.mark.asyncio
    async def test_missing_embeddings_dependency_disables_primary_search_action_with_recovery(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Missing RAG dependencies should disable Search with actionable recovery copy."""
        with patch.dict(
            search_rag_window_module.DEPENDENCIES_AVAILABLE,
            {"embeddings_rag": False},
            clear=False,
        ):
            async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                window = pilot.app.test_widget

                search_input = window.query_one("#search-query-input", Input)
                search_button = window.query_one("#search-button", Button)

                assert search_input.disabled is True
                assert "Embeddings not available" in search_input.placeholder
                assert search_button.disabled is True
                tooltip = str(search_button.tooltip)
                assert "Search/RAG queries" in tooltip
                assert "Missing optional dependencies: embeddings_rag" in tooltip
                assert 'pip install -e ".[embeddings_rag]"' in tooltip
                assert 'pip install "tldw_chatbook[embeddings_rag]"' in tooltip
                assert "then restart" in tooltip
                assert window.is_searching is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("web_search_available", [False, True])
    async def test_get_search_config_reads_current_controls(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
        web_search_available: bool,
    ) -> None:
        """Search config should mirror the live control values and reactive state."""
        with patch.object(search_rag_window_module, "WEB_SEARCH_AVAILABLE", web_search_available):
            with patch(
                "tldw_chatbook.UI.Views.RAGSearch.search_event_handlers.WEB_SEARCH_AVAILABLE",
                web_search_available,
            ):
                async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                    window = pilot.app.test_widget

                    window.query_one("#search-mode-select", Select).value = "hybrid"
                    window.query_one("#collection-select", Select).value = "all"
                    window.query_one("#top-k-input", Input).value = "7"
                    window.query_one("#temperature-input", Input).value = "0.35"
                    window.query_one("#filter-media", Checkbox).value = True
                    window.query_one("#filter-conversations", Checkbox).value = False
                    window.query_one("#filter-notes", Checkbox).value = True

                    window.enable_parent_docs = True
                    window.parent_retrieval_strategy = "sentence_window"
                    window.parent_retrieval_size = 256

                    expected_config = {
                        "mode": "hybrid",
                        "collection": "all",
                        "top_k": 7,
                        "temperature": 0.35,
                        "enable_parent_docs": True,
                        "parent_strategy": "sentence_window",
                        "parent_size": 256,
                        "filters": {
                            "media": True,
                            "conversations": False,
                            "notes": True,
                        },
                    }
                    if web_search_available:
                        expected_config["include_web_search"] = False

                    assert window._get_search_config() == expected_config

    @pytest.mark.asyncio
    async def test_action_focus_search_focuses_search_input(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """The focus shortcut should move focus to the search box."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            search_input = window.query_one("#search-query-input", Input)

            window.action_focus_search()
            await pilot.pause()

            assert search_input.has_focus

    @pytest.mark.asyncio
    async def test_display_results_paginates_current_results(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Result rendering should paginate using the current 10-item page size."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            window.search_results = [
                {
                    "title": f"Result {index}",
                    "content": "Preview content",
                    "source": "media",
                    "score": 0.9,
                }
                for index in range(12)
            ]
            window.total_results = len(window.search_results)

            await window._display_results()
            await pilot.pause()

            results_list = window.query_one("#results-list-enhanced")
            page_info = window.query_one("#page-info", Static)

            assert len(results_list.children) == 10
            assert "hidden" not in window.query_one("#pagination-enhanced").classes
            assert "Page 1 of 2" in _text(page_info)
            assert "12 results" in _text(page_info)

    @pytest.mark.asyncio
    async def test_display_results_shows_zero_result_recovery_empty_state(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Zero-result searches should leave a recovery path instead of a blank pane."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            window.search_results = []
            window.total_results = 0
            window.last_search_config = {"query": "missing topic", "mode": "hybrid", "collection": "all"}

            await window._display_results()
            await pilot.pause()

            empty_state = window.query_one("#search-empty-state", Static)
            empty_text = _text(empty_state)

            assert "No results found" in empty_text
            assert "Plain Search" in empty_text
            assert "All Collections" in empty_text
            assert "Ingest" in empty_text
            assert "Use in Chat" in empty_text

    @pytest.mark.asyncio
    async def test_record_search_to_history_persists_entry(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Recording a search should persist history instead of silently logging an error."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            window.last_search_time = 0.42

            window._record_search_to_history(
                query="python widgets",
                search_type="plain",
                filters={"media": True, "notes": False},
                results_count=3,
            )
            await pilot.pause()

            history = window.search_history_db.get_search_history(limit=1)

            assert len(history) == 1
            assert history[0]["query"] == "python widgets"
            assert history[0]["search_type"] == "plain"
            assert history[0]["result_count"] == 3
            assert history[0]["search_params"]["filters"] == {
                "media": True,
                "notes": False,
            }

    @pytest.mark.asyncio
    async def test_saved_search_load_applies_selected_config_to_controls(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Loading a saved search should repopulate the search controls for reuse."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            saved_panel = window.query_one("#saved-searches-panel", SavedSearchesPanel)
            saved_panel.save_search(
                "Agent UX Search",
                {
                    "query": "agent UX",
                    "mode": "hybrid",
                    "collection": "all",
                    "top_k": 7,
                    "temperature": 0.35,
                    "filters": {"media": True, "conversations": False, "notes": True},
                },
            )
            await pilot.pause()

            saved_list = saved_panel.query_one("#saved-searches-list", ListView)
            saved_list.index = 0
            saved_list.action_select_cursor()
            await pilot.pause()

            load_button = saved_panel.query_one("#load-saved-search", Button)
            assert load_button.disabled is False
            load_button.press()
            await pilot.pause()

            assert window.query_one("#search-query-input", Input).value == "agent UX"
            assert window.query_one("#search-mode-select", Select).value == "hybrid"
            assert window.query_one("#top-k-input", Input).value == "7"
            assert window.query_one("#temperature-input", Input).value == "0.35"
            assert window.query_one("#filter-media", Checkbox).value is True
            assert window.query_one("#filter-conversations", Checkbox).value is False
            assert window.query_one("#filter-notes", Checkbox).value is True


@pytest.mark.ui
class TestSemanticHonestStates:
    """Task-250: the standalone Search surface reports semantic-leg states."""

    def test_sources_from_config_maps_filter_checkboxes(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
    ) -> None:
        window = SearchRAGWindow(mock_app_instance, id="test-search-window")

        sources = window._sources_from_config(
            {"filters": {"media": True, "conversations": False, "notes": True}}
        )

        assert sources == {"media": True, "conversations": False, "notes": True}
        # Missing filters default to searching everything
        assert window._sources_from_config({}) == {
            "media": True, "conversations": True, "notes": True,
        }

    @pytest.mark.asyncio
    async def test_perform_hybrid_search_uses_pipeline_contract_and_diagnostics(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
    ) -> None:
        """The hybrid seam must call the real pipeline signature and keep WHY."""
        from tldw_chatbook.RAG_Search.semantic_availability import (
            SEMANTIC_DIAGNOSTICS_KEY,
            SEMANTIC_REASON_DEPS_MISSING,
            SEMANTIC_STATUS_UNAVAILABLE,
            SEMANTIC_UNAVAILABLE_MESSAGES,
        )

        window = SearchRAGWindow(mock_app_instance, id="test-search-window")
        captured = {}

        async def fake_hybrid(app, query, sources, top_k=10, diagnostics=None, **kwargs):
            captured["app"] = app
            captured["query"] = query
            captured["sources"] = sources
            captured["top_k"] = top_k
            if diagnostics is not None:
                diagnostics[SEMANTIC_DIAGNOSTICS_KEY] = {
                    "status": SEMANTIC_STATUS_UNAVAILABLE,
                    "reason": SEMANTIC_REASON_DEPS_MISSING,
                    "message": SEMANTIC_UNAVAILABLE_MESSAGES[SEMANTIC_REASON_DEPS_MISSING],
                }
            return ([{"id": "1", "title": "Doc", "content": "x", "source": "media"}], "ctx")

        with patch.object(search_rag_window_module, "perform_hybrid_rag_search", fake_hybrid):
            results = await window._perform_hybrid_search(
                "query",
                {"top_k": 7, "filters": {"media": True, "conversations": False, "notes": False}},
            )

        assert captured["app"] is mock_app_instance
        assert captured["query"] == "query"
        assert captured["sources"] == {"media": True, "conversations": False, "notes": False}
        assert captured["top_k"] == 7
        assert [r["id"] for r in results] == ["1"]
        semantic_state = window.last_search_diagnostics[SEMANTIC_DIAGNOSTICS_KEY]
        assert semantic_state["status"] == SEMANTIC_STATUS_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_perform_contextual_search_unpacks_pipeline_tuple(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
    ) -> None:
        window = SearchRAGWindow(mock_app_instance, id="test-search-window")

        async def fake_full(app, query, sources, top_k=10, diagnostics=None, **kwargs):
            return ([{"id": "v1", "title": "Vec", "content": "y", "source": "media"}], "ctx")

        with patch.object(search_rag_window_module, "perform_full_rag_pipeline", fake_full):
            results = await window._perform_contextual_search("query", {"top_k": 3})

        assert [r["id"] for r in results] == ["v1"]

    def test_semantic_leg_notice_states(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
    ) -> None:
        from tldw_chatbook.RAG_Search.semantic_availability import (
            SEMANTIC_DIAGNOSTICS_KEY,
            SEMANTIC_EMPTY_INDEX_MESSAGE,
            SEMANTIC_REASON_DEPS_MISSING,
            SEMANTIC_STATUS_EMPTY_INDEX,
            SEMANTIC_STATUS_UNAVAILABLE,
            SEMANTIC_UNAVAILABLE_MESSAGES,
        )

        window = SearchRAGWindow(mock_app_instance, id="test-search-window")

        # No diagnostics -> nothing to report
        window.last_search_diagnostics = {}
        assert window._semantic_leg_notice("hybrid") is None

        # Semantic leg ran fine -> nothing to report
        window.last_search_diagnostics = {
            SEMANTIC_DIAGNOSTICS_KEY: {"status": "ok", "result_count": 2}
        }
        assert window._semantic_leg_notice("hybrid") is None

        # Hybrid with unavailable leg -> keyword-only marker + full reason
        window.last_search_diagnostics = {
            SEMANTIC_DIAGNOSTICS_KEY: {
                "status": SEMANTIC_STATUS_UNAVAILABLE,
                "reason": SEMANTIC_REASON_DEPS_MISSING,
                "message": SEMANTIC_UNAVAILABLE_MESSAGES[SEMANTIC_REASON_DEPS_MISSING],
            }
        }
        marker, message = window._semantic_leg_notice("hybrid")
        assert "keyword-only" in marker
        assert "keyword-only (FTS)" in message
        assert "embeddings" in message

        # Contextual with empty index -> distinct index-empty copy
        window.last_search_diagnostics = {
            SEMANTIC_DIAGNOSTICS_KEY: {
                "status": SEMANTIC_STATUS_EMPTY_INDEX,
                "message": SEMANTIC_EMPTY_INDEX_MESSAGE,
            }
        }
        marker, message = window._semantic_leg_notice("contextual")
        assert marker == "semantic index empty"
        assert message == SEMANTIC_EMPTY_INDEX_MESSAGE


@pytest.mark.ui
class TestSearchHistoryDropdown:
    """Search history auto-complete behavior."""

    @pytest.mark.asyncio
    async def test_show_history_filters_recent_queries(
        self,
        temp_user_data_dir: Path,
        widget_pilot,
    ) -> None:
        """The dropdown should filter stored history against the current query text."""
        db = SearchHistoryDB(temp_user_data_dir / "search_history.db")
        db.record_search("python programming", "plain", [], 120)
        db.record_search("java programming", "plain", [], 80)
        db.record_search("python tutorials", "hybrid", [], 95)

        async with await widget_pilot(SearchHistoryDropdown, search_history_db=db) as pilot:
            dropdown = pilot.app.test_widget

            await dropdown.show_history("python")
            await pilot.pause()

            list_view = dropdown.query_one("#search-history-list", ListView)

            assert dropdown.history_items == ["python tutorials", "python programming"]
            assert len(list_view.children) == 2
            assert "hidden" not in dropdown.classes


@pytest.mark.ui
class TestSavedSearchesPanel:
    """Saved search persistence and list refresh behavior."""

    @pytest.mark.asyncio
    async def test_save_search_persists_and_creates_visible_list_item(
        self,
        temp_user_data_dir: Path,
        widget_pilot,
    ) -> None:
        """Saving the first search should replace the empty state with an actual list entry."""
        with patch(
            "tldw_chatbook.UI.Views.RAGSearch.saved_searches_panel.get_user_data_dir",
            return_value=temp_user_data_dir,
        ):
            async with await widget_pilot(SavedSearchesPanel) as pilot:
                panel = pilot.app.test_widget

                config = {
                    "query": "agent UX",
                    "mode": "hybrid",
                    "filters": {"media": True, "conversations": False, "notes": True},
                }

                panel.save_search("Agent UX Search", config)
                await pilot.pause()

                saved_path = temp_user_data_dir / "saved_searches.json"
                list_view = panel.query_one("#saved-searches-list", ListView)

                assert saved_path.exists()
                assert "Agent UX Search" in panel.saved_searches
                assert len(list_view.children) == 1

                persisted = json.loads(saved_path.read_text())
                assert persisted["Agent UX Search"]["config"] == config

    @pytest.mark.asyncio
    async def test_saved_search_actions_explain_selection_requirement_and_delete_selected(
        self,
        temp_user_data_dir: Path,
        widget_pilot,
    ) -> None:
        """Saved-search actions should explain selection requirements and recover after deletion."""
        with patch(
            "tldw_chatbook.UI.Views.RAGSearch.saved_searches_panel.get_user_data_dir",
            return_value=temp_user_data_dir,
        ):
            async with await widget_pilot(SavedSearchesPanel) as pilot:
                panel = pilot.app.test_widget
                load_button = panel.query_one("#load-saved-search", Button)
                delete_button = panel.query_one("#delete-saved-search", Button)

                assert load_button.disabled is True
                assert "Select a saved search before loading it" in str(load_button.tooltip)
                assert delete_button.disabled is True
                assert "Select a saved search before deleting it" in str(delete_button.tooltip)

                panel.save_search("Agent UX Search", {"query": "agent UX", "mode": "plain"})
                await pilot.pause()

                saved_list = panel.query_one("#saved-searches-list", ListView)
                saved_list.index = 0
                saved_list.action_select_cursor()
                await pilot.pause()

                assert panel.selected_search_name == "Agent UX Search"
                assert load_button.disabled is False
                assert "Load the selected saved search" in str(load_button.tooltip)
                assert delete_button.disabled is False
                assert "Delete the selected saved search" in str(delete_button.tooltip)

                delete_button.press()
                await pilot.pause()

                assert "Agent UX Search" not in panel.saved_searches
                assert panel.selected_search_name is None
                assert load_button.disabled is True
                assert "Select a saved search before loading it" in str(load_button.tooltip)


@pytest.mark.ui
class TestSearchResult:
    """Search result card rendering."""

    @pytest.mark.asyncio
    async def test_result_item_displays_title_preview_and_score(self, widget_pilot) -> None:
        """A result card should render current title, preview, and score affordances."""
        result_data = {
            "title": "Test Title",
            "content": "This is the preview content that should be visible in the card.",
            "source": "media",
            "score": 0.95,
            "metadata": {"author": "Ada", "topic": "UX"},
        }

        async with await widget_pilot(SearchResult, result=result_data, index=0) as pilot:
            result_widget = pilot.app.test_widget

            title = result_widget.query_one(".result-title-enhanced", Static)
            preview = result_widget.query_one(".result-preview-enhanced", Static)
            score = result_widget.query_one(".score-text", Static)

            assert "Test Title" in _text(title)
            assert "preview content" in _text(preview).lower()
            assert "95.0%" == _text(score)


def _notify_messages(mock_app: MagicMock) -> list[str]:
    """Return the text of every notification sent to the mock app."""
    return [str(call.args[0]) for call in mock_app.notify.call_args_list if call.args]


async def _wait_for_indexing_cycle(pilot) -> None:
    """Let a Start Indexing press run its worker plus the follow-up stats refresh."""
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    # Completion schedules the stats-refresh worker; let it finish too.
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


def _ok_summary(**overrides) -> dict:
    summary = {"status": "ok", "indexed": 0, "skipped": 0, "failed": 0, "errors": [], "by_type": {}}
    summary.update(overrides)
    return summary


@pytest.mark.ui
class TestIndexingControls:
    """Task-251: the Maintenance-tab indexing controls trigger real indexing."""

    @pytest.mark.asyncio
    async def test_missing_embeddings_dependency_disables_indexing_controls_with_recovery(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Without embeddings deps the indexing controls are disabled with honest recovery copy."""
        with patch.dict(
            search_rag_window_module.DEPENDENCIES_AVAILABLE,
            {"embeddings_rag": False},
            clear=False,
        ):
            async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                window = pilot.app.test_widget

                start_button = window.query_one("#start-indexing", Button)
                source_select = window.query_one("#index-source-select", Select)

                assert start_button.disabled is True
                tooltip = str(start_button.tooltip)
                assert "Missing optional dependencies: embeddings_rag" in tooltip
                assert 'pip install "tldw_chatbook[embeddings_rag]"' in tooltip
                assert source_select.disabled is True
                assert "embeddings_rag" in str(source_select.tooltip)

    @pytest.mark.asyncio
    async def test_start_indexing_runs_backfill_for_selected_source_and_refreshes_stats(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Pressing Start Indexing runs the real backfill path and re-reads real stats."""
        calls: list[dict] = []

        async def fake_backfill(**kwargs):
            calls.append(kwargs)
            return _ok_summary(indexed=3, skipped=1)

        rag_service = MagicMock()
        rag_service.vector_store.get_collection_stats.return_value = {
            "name": "tldw_rag",
            "count": 42,
        }
        mock_app_instance._rag_service = None

        with patch.object(search_rag_window_module, "backfill_semantic_index", fake_backfill):
            with patch.object(
                search_rag_window_module, "peek_shared_rag_service", return_value=None
            ):
                async with await widget_pilot(
                    SearchRAGWindow, app_instance=mock_app_instance
                ) as pilot:
                    window = pilot.app.test_widget
                    await pilot.app.workers.wait_for_complete()
                    await pilot.pause()

                    # Before any runtime exists, stats honestly say so.
                    table = window.query_one("#index-stats-table", DataTable)
                    assert table.row_count == 1
                    assert any(
                        "not initialized" in str(cell) for cell in table.get_row_at(0)
                    )

                    # Once a runtime exists, completion re-reads real stats from it.
                    mock_app_instance._rag_service = rag_service
                    window.query_one("#index-source-select", Select).value = "notes"
                    await pilot.pause()

                    window.query_one("#start-indexing", Button).press()
                    await _wait_for_indexing_cycle(pilot)

                    assert len(calls) == 1
                    assert calls[0]["item_types"] == (ITEM_TYPE_NOTE,)
                    assert calls[0]["chachanotes_db"] is mock_app_instance.chachanotes_db
                    assert calls[0]["media_db"] is None

                    assert window.query_one("#start-indexing", Button).disabled is False
                    messages = _notify_messages(mock_app_instance)
                    assert any(
                        "Indexing complete" in message and "3" in message
                        for message in messages
                    )

                    row = [str(cell) for cell in table.get_row_at(0)]
                    assert "tldw_rag" in row
                    assert "42" in row

    @pytest.mark.asyncio
    async def test_index_source_select_maps_to_backfill_item_types(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Every source option maps onto the real backfill item-type contract."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            select = window.query_one("#index-source-select", Select)

            expectations = {
                "media": (ITEM_TYPE_MEDIA,),
                "conversations": (ITEM_TYPE_CONVERSATION,),
                "notes": (ITEM_TYPE_NOTE,),
                "all": (ITEM_TYPE_MEDIA, ITEM_TYPE_NOTE, ITEM_TYPE_CONVERSATION),
            }
            for value, expected in expectations.items():
                select.value = value
                await pilot.pause()
                assert window._selected_index_item_types() == expected

    @pytest.mark.asyncio
    async def test_start_indexing_is_single_flight(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """A second activation during a run cannot start a second backfill."""
        gate = threading.Event()
        started = threading.Event()
        calls: list[dict] = []

        async def gated_backfill(**kwargs):
            calls.append(kwargs)
            started.set()
            gate.wait(timeout=5)
            return _ok_summary()

        with patch.object(search_rag_window_module, "backfill_semantic_index", gated_backfill):
            async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                window = pilot.app.test_widget

                window.query_one("#start-indexing", Button).press()
                await pilot.pause()
                assert started.wait(timeout=5) is True

                # While the run is in flight the button is disabled...
                start_button = window.query_one("#start-indexing", Button)
                assert start_button.disabled is True
                # ...and even a programmatic re-activation is refused honestly.
                window._start_indexing_run()
                messages = _notify_messages(mock_app_instance)
                assert any("already running" in message for message in messages)

                gate.set()
                await _wait_for_indexing_cycle(pilot)

                assert len(calls) == 1
                assert window.query_one("#start-indexing", Button).disabled is False

    @pytest.mark.asyncio
    async def test_backfill_failures_surface_last_error(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Failed items produce an honest warning carrying the last error."""

        async def failing_backfill(**kwargs):
            return _ok_summary(
                status="partial", indexed=1, failed=2, errors=["media 3: boom"]
            )

        with patch.object(search_rag_window_module, "backfill_semantic_index", failing_backfill):
            async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                window = pilot.app.test_widget

                window.query_one("#start-indexing", Button).press()
                await _wait_for_indexing_cycle(pilot)

                messages = _notify_messages(mock_app_instance)
                assert any("boom" in message for message in messages)
                assert window.query_one("#start-indexing", Button).disabled is False

    @pytest.mark.asyncio
    async def test_backfill_unavailable_status_notifies_recovery(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """An unavailable backfill (deps/kill switch) reports why instead of silence."""

        async def unavailable_backfill(**kwargs):
            return _ok_summary(status="unavailable")

        with patch.object(
            search_rag_window_module, "backfill_semantic_index", unavailable_backfill
        ):
            async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                window = pilot.app.test_widget

                window.query_one("#start-indexing", Button).press()
                await _wait_for_indexing_cycle(pilot)

                messages = _notify_messages(mock_app_instance)
                assert any("did not run" in message for message in messages)
                assert any("embeddings" in message for message in messages)

    @pytest.mark.asyncio
    async def test_backfill_crash_is_caught_and_reported(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """A raising backfill never kills the app; it notifies and re-enables the button."""

        async def crashing_backfill(**kwargs):
            raise RuntimeError("kapow")

        with patch.object(search_rag_window_module, "backfill_semantic_index", crashing_backfill):
            async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                window = pilot.app.test_widget

                window.query_one("#start-indexing", Button).press()
                await _wait_for_indexing_cycle(pilot)

                messages = _notify_messages(mock_app_instance)
                assert any("kapow" in message for message in messages)
                assert window._indexing_in_flight is False
                assert window.query_one("#start-indexing", Button).disabled is False

    @pytest.mark.asyncio
    async def test_start_indexing_without_source_databases_reports_honestly(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """No reachable source database means an honest error, not a fake run."""
        mock_app_instance.media_db = None
        mock_app_instance.chachanotes_db = None

        with patch.object(
            search_rag_window_module, "backfill_semantic_index", MagicMock()
        ) as backfill:
            async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                window = pilot.app.test_widget

                window.query_one("#start-indexing", Button).press()
                await pilot.pause()

                backfill.assert_not_called()
                messages = _notify_messages(mock_app_instance)
                assert any("no source database" in message for message in messages)

    @pytest.mark.asyncio
    async def test_apply_index_stats_renders_honest_states(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """The stats table shows real counts and honest not-ready/error states."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            table = window.query_one("#index-stats-table", DataTable)

            def row_text() -> str:
                return " ".join(str(cell) for cell in table.get_row_at(0))

            window._apply_index_stats(None)
            assert table.row_count == 1
            assert "not initialized" in row_text()

            window._apply_index_stats({"name": "tldw_rag", "error": "stats exploded"})
            assert table.row_count == 1
            assert "stats exploded" in row_text()

            window._apply_index_stats({"name": "tldw_rag", "count": 7})
            assert table.row_count == 1
            assert "tldw_rag" in row_text()
            assert "7" in row_text()

            window._apply_index_stats({"name": "tldw_rag", "count": 0})
            assert "empty" in row_text()

            # Untrustworthy counts are never displayed as real numbers.
            window._apply_index_stats({"name": "tldw_rag", "count": "7"})
            assert "unavailable" in row_text()


@pytest.mark.ui
class TestWiredSearchChromeControls:
    """Task-251: previously dead chrome either acts for real or is gone."""

    @pytest.mark.asyncio
    async def test_dead_placeholder_controls_are_removed(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Placeholder controls with no cheap real backing are no longer rendered."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            assert not window.query("#create-collection")
            assert not window.query("#delete-collection")
            assert not window.query("#refresh-results")
            assert not window.query("#indexing-progress")

    @pytest.mark.asyncio
    async def test_pagination_buttons_page_through_results(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Previous/Next actually navigate result pages."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            window.search_results = [
                {"title": f"Result {index}", "content": "c", "source": "media", "score": 0.5}
                for index in range(12)
            ]
            window.total_results = 12
            await window._display_results()
            await pilot.pause()

            window.query_one("#next-page", Button).press()
            await pilot.pause()

            assert window.current_page == 2
            assert "Page 2 of 2" in _text(window.query_one("#page-info", Static))
            assert len(window.query_one("#results-list-enhanced").children) == 2

            window.query_one("#prev-page", Button).press()
            await pilot.pause()

            assert window.current_page == 1
            assert "Page 1 of 2" in _text(window.query_one("#page-info", Static))

    @pytest.mark.asyncio
    async def test_export_results_button_uses_export_action(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """The results-header Export button triggers the real export action."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            window.search_results = []

            window.query_one("#export-results", Button).press()
            await pilot.pause()

            assert any(
                "No results to export" in message
                for message in _notify_messages(mock_app_instance)
            )

    @pytest.mark.asyncio
    async def test_refresh_history_controls_honor_selected_range(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """History refresh reloads through the DB honoring the selected time range."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            window.search_history_db = MagicMock()
            window.search_history_db.get_search_history.return_value = []

            window.query_one("#history-range-select", Select).value = "7"
            await pilot.pause()
            window.query_one("#refresh-history", Button).press()
            await pilot.pause()

            assert (
                window.search_history_db.get_search_history.call_args.kwargs["days_back"] == 7
            )

            window.query_one("#history-range-select", Select).value = "all"
            await pilot.pause()

            assert (
                window.search_history_db.get_search_history.call_args.kwargs["days_back"] is None
            )

    @pytest.mark.asyncio
    async def test_refresh_collections_button_triggers_real_refresh(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """The Maintenance refresh button reloads collections and index stats."""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            window._refresh_collections_list = MagicMock()
            window._refresh_index_stats = MagicMock()

            window.query_one("#refresh-collections", Button).press()
            await pilot.pause()

            window._refresh_collections_list.assert_called_once()
            window._refresh_index_stats.assert_called_once()
