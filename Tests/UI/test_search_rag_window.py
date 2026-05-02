# test_search_rag_window.py
# Focused regression tests for the current Search/RAG widget contract

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Button, Checkbox, Input, ListView, Select, Static

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.DB.search_history_db import SearchHistoryDB
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

    @pytest.mark.asyncio
    async def test_get_search_config_reads_current_controls(
        self,
        mock_app_instance: MagicMock,
        search_rag_test_env,
        widget_pilot,
    ) -> None:
        """Search config should mirror the live control values and reactive state."""
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

            assert window._get_search_config() == {
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
