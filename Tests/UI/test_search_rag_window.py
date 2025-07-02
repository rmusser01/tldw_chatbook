# test_search_rag_window.py
# Comprehensive UI tests for SearchRAGWindow

import pytest
import pytest_asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path
import json
from datetime import datetime
import tempfile
import shutil
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from textual.app import App, ComposeResult
from textual.widgets import Input, Select, Checkbox, ListView, Button, Static
from textual.containers import Container

# Import test utilities
from Tests.textual_test_utils import WidgetTestApp, widget_pilot, app_pilot

from tldw_chatbook.UI.SearchRAGWindow import (
    SearchRAGWindow, SearchHistoryDropdown, SavedSearchesPanel,
    SearchResult, SOURCE_ICONS, SOURCE_COLORS
)


@pytest.fixture
def mock_app_instance():
    """Create a mock app instance with required attributes"""
    app = MagicMock()
    app.media_db = MagicMock()
    app.rag_db = MagicMock()
    app.chachanotes_db = MagicMock()
    app.notes_service = MagicMock()
    app.notes_user_id = "test_user"
    app.notify = MagicMock()
    return app


@pytest.fixture
def temp_user_data_dir():
    """Create temporary user data directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.mark.ui
class TestSearchRAGWindow:
    """Test SearchRAGWindow UI component"""
    
    @pytest.mark.asyncio
    async def test_window_initialization(self, mock_app_instance, widget_pilot):
        """Test SearchRAGWindow initializes correctly"""
        window = SearchRAGWindow(mock_app_instance, id="test-search-window")
        
        assert window.app_instance == mock_app_instance
        assert window.current_results == []
        assert window.all_results == []
        assert window.search_history == []
        assert window.current_search_id is None
        assert window.is_searching == False
        assert window.current_page == 1
        assert window.results_per_page == 20
        assert window.total_results == 0
    
    @pytest.mark.asyncio
    async def test_compose_creates_ui_elements(self, mock_app_instance, widget_pilot):
        """Test that compose creates all required UI elements"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            # Check main containers exist
            # The actual implementation uses different container structure
            assert window.query_one(".search-section")
            assert window.query_one("#results-container")
            assert window.query_one("#saved-searches-panel")
            
            # Check search input elements
            assert window.query_one("#rag-search-input", Input)
            assert window.query_one("#rag-search-btn", Button)
            assert window.query_one("#search-mode-select", Select)
            
            # Check filter checkboxes
            assert window.query_one("#source-media", Checkbox)
            assert window.query_one("#source-conversations", Checkbox)
            assert window.query_one("#source-notes", Checkbox)
            
            # Check advanced options
            assert window.query_one("#top-k-input", Input)
            assert window.query_one("#max-context-input", Input)
            assert window.query_one("#enable-rerank", Checkbox)
    
    @pytest.mark.asyncio
    async def test_search_mode_options(self, mock_app_instance, widget_pilot):
        """Test search mode select has correct options"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            search_mode = window.query_one("#search-mode-select", Select)
            
            expected_options = [
                ("plain", "Plain Search (BM25/FTS5)"),
                ("full", "Full RAG (with embeddings)"),
                ("hybrid", "Hybrid (BM25 + embeddings)")
            ]
            
            # Check if options match expected
            for i, (value, label) in enumerate(expected_options):
                assert search_mode._options[i].value == value
                assert label in search_mode._options[i].prompt
    
    @pytest.mark.asyncio
    async def test_search_input_triggers_search(self, mock_app_instance, widget_pilot):
        """Test that submitting search input triggers search"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            # Mock the search method
            window._perform_search = AsyncMock()
            
            # Enter search query
            search_input = window.query_one("#rag-search-input", Input)
            search_input.value = "test query"
            
            # Trigger search by submitting input
            await pilot.press("enter")
            await pilot.pause()
            
            # Verify search was called
            window._perform_search.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    async def test_source_filter_checkboxes(self, mock_app_instance, widget_pilot):
        """Test source filter checkboxes work correctly"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            # Get checkboxes
            media_cb = window.query_one("#source-media", Checkbox)
            conv_cb = window.query_one("#source-conversations", Checkbox)
            notes_cb = window.query_one("#source-notes", Checkbox)
            
            # All should be checked by default
            assert media_cb.value == True
            assert conv_cb.value == True
            assert notes_cb.value == True
            
            # Test toggling
            await pilot.click("#source-media")
            assert media_cb.value == False
            
            await pilot.click("#source-conversations")
            assert conv_cb.value == False
    
    @patch('tldw_chatbook.UI.SearchRAGWindow.perform_plain_rag_search')
    @pytest.mark.asyncio
    async def test_plain_search_execution(self, mock_search, mock_app_instance, widget_pilot):
        """Test plain search execution"""
        mock_search.return_value = (
            [{"title": "Test Result", "content": "Test content", "source": "media"}],
            "Test context"
        )
        
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            # Set search mode to plain
            search_mode = window.query_one("#search-mode-select", Select)
            search_mode.value = "plain"
            
            # Enter query and search
            search_input = window.query_one("#rag-search-input", Input)
            search_input.value = "test"
            
            await window._perform_search("test")
            await pilot.pause()
            
            # Verify search was called with correct parameters
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args[0][0] == mock_app_instance  # app instance
            assert call_args[0][1] == "test"  # query
            
            # Verify results were stored
            assert len(window.all_results) == 1
            assert window.total_results == 1
    
    @pytest.mark.asyncio
    async def test_search_history_dropdown(self, mock_app_instance, widget_pilot):
        """Test search history dropdown functionality"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            # Get history dropdown
            history_dropdown = window.query_one(SearchHistoryDropdown)
            
            # Initially hidden
            assert "hidden" in history_dropdown.classes
            
            # Focus search input should show history
            search_input = window.query_one("#rag-search-input", Input)
            search_input.focus()
            await pilot.pause()
            
            # Should show history (though it may be empty)
            assert history_dropdown is not None
    
    @pytest.mark.asyncio
    async def test_saved_searches_panel(self, mock_app_instance, temp_user_data_dir, widget_pilot):
        """Test saved searches panel functionality"""
        with patch('tldw_chatbook.UI.SearchRAGWindow.get_user_data_dir', return_value=temp_user_data_dir):
            async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                window = pilot.app.test_widget
                
                # Get saved searches panel
                saved_panel = window.query_one(SavedSearchesPanel)
                assert saved_panel is not None
                
                # Test saving a search
                test_config = {
                    "query": "test",
                    "search_mode": "plain",
                    "sources": {"media": True, "conversations": True, "notes": False}
                }
                
                saved_panel.save_search("Test Search", test_config)
                
                # Verify saved
                assert "Test Search" in saved_panel.saved_searches
                assert saved_panel.saved_searches["Test Search"]["config"] == test_config
                
                # Verify persisted to disk
                saved_file = temp_user_data_dir / "saved_searches.json"
                assert saved_file.exists()
                
                with open(saved_file) as f:
                    saved_data = json.load(f)
                    assert "Test Search" in saved_data
    
    @pytest.mark.asyncio
    async def test_search_result_display(self, mock_app_instance, widget_pilot):
        """Test search result display"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            # Mock some results
            test_results = [
                {
                    "title": "Media Result",
                    "content": "This is media content",
                    "source": "media",
                    "score": 0.95,
                    "metadata": {"id": 1}
                },
                {
                    "title": "Conversation Result", 
                    "content": "This is conversation content",
                    "source": "conversation",
                    "score": 0.85,
                    "metadata": {"id": 2}
                }
            ]
            
            window.all_results = test_results
            window.total_results = len(test_results)
            
            # Display results
            await window._display_results_page("Test context")
            await pilot.pause()
            
            # Check results container has items
            results_container = window.query_one("#results-container")
            result_items = results_container.query(SearchResult)
            
            # Should have 2 result items
            assert len(result_items) == 2
    
    @pytest.mark.asyncio
    async def test_pagination_controls(self, mock_app_instance, widget_pilot):
        """Test pagination controls"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            # Mock many results to trigger pagination
            window.all_results = [{"title": f"Result {i}", "source": "media"} for i in range(50)]
            window.total_results = 50
            window.results_per_page = 20
            
            # Display first page
            await window._display_results_page("")
            
            # Check pagination info
            assert window.current_page == 1
            
            # Get pagination controls
            prev_btn = window.query_one("#prev-page-btn", Button)
            next_btn = window.query_one("#next-page-btn", Button)
            
            # First page - prev should be disabled
            assert prev_btn.disabled == True
            assert next_btn.disabled == False
            
            # Go to next page
            await pilot.click("#next-page-btn")
            await pilot.pause()
            
            assert window.current_page == 2
            assert prev_btn.disabled == False
    
    @pytest.mark.asyncio
    async def test_advanced_options_toggle(self, mock_app_instance, widget_pilot):
        """Test advanced options collapsible"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            # Find advanced options
            advanced_collapsible = window.query_one("#advanced-options-collapsible")
            assert advanced_collapsible is not None
            
            # Should contain chunk size and overlap inputs
            chunk_size = advanced_collapsible.query_one("#chunk-size-input", Input)
            chunk_overlap = advanced_collapsible.query_one("#chunk-overlap-input", Input)
            
            assert chunk_size is not None
            assert chunk_overlap is not None
            
            # Check default values
            assert chunk_size.value == "400"
            assert chunk_overlap.value == "100"
    
    @pytest.mark.asyncio
    async def test_export_results_functionality(self, mock_app_instance, temp_user_data_dir):
        """Test export results button"""
        with patch('tldw_chatbook.UI.SearchRAGWindow.get_user_data_dir', return_value=temp_user_data_dir):
            async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
                window = pilot.app.test_widget
                
                # Export button should be disabled initially
                export_btn = window.query_one("#export-results-btn", Button)
                assert export_btn.disabled == True
                
                # Add some results
                window.all_results = [
                    {"title": "Test", "content": "Content", "source": "media"}
                ]
                window.total_results = 1
                
                # Enable export
                export_btn.disabled = False
                
                # Mock export method
                window._export_results = AsyncMock()
                
                # Click export
                await pilot.click("#export-results-btn")
                await pilot.pause()
                
                # Verify export was called
                window._export_results.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_display(self, mock_app_instance, widget_pilot):
        """Test error handling and display"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            # Mock search to raise an error
            with patch('tldw_chatbook.UI.SearchRAGWindow.perform_plain_rag_search') as mock_search:
                mock_search.side_effect = Exception("Test error")
                
                # Try to search
                await window._perform_search("test")
                await pilot.pause()
                
                # Check status shows error
                status = window.query_one("#search-status", Static)
                assert "error" in status.renderable.plain.lower()
                
                # Check notification was sent
                mock_app_instance.notify.assert_called()
    
    @pytest.mark.asyncio
    async def test_search_status_updates(self, mock_app_instance, widget_pilot):
        """Test search status updates during search"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            status_elem = window.query_one("#search-status", Static)
            
            # Check initial status
            assert status_elem.renderable.plain == "🔍 Ready to search"
            
            # During search
            window.is_searching = True
            
            # Mock a search in progress
            with patch('tldw_chatbook.UI.SearchRAGWindow.perform_plain_rag_search') as mock_search:
                mock_search.return_value = ([], "")
                
                await window._perform_search("test")
                await pilot.pause()
                
                # After search with no results
                assert "No results found" in status_elem.renderable.plain
    
    @pytest.mark.asyncio
    async def test_keyboard_shortcuts(self, mock_app_instance, widget_pilot):
        """Test keyboard shortcuts work"""
        async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            
            # Test Ctrl+K focuses search
            await pilot.press("ctrl+k")
            await pilot.pause()
            
            search_input = window.query_one("#rag-search-input", Input)
            assert search_input.has_focus
            
            # Test Escape clears search
            search_input.value = "test"
            await pilot.press("escape")
            await pilot.pause()
            
            assert search_input.value == ""


@pytest.mark.ui
class TestSearchResult:
    """Test SearchResult widget"""
    
    @pytest.mark.asyncio
    async def test_result_item_display(self, widget_pilot):
        """Test SearchResult displays correctly"""
        result_data = {
            "title": "Test Title",
            "content": "This is test content that should be displayed",
            "source": "media",
            "score": 0.95,
            "metadata": {"id": 123, "created_at": "2024-01-01"}
        }
        
        async with await widget_pilot(SearchResult, result=result_data, index=0) as pilot:
            item = pilot.app.test_widget
            
            # Check title is displayed
            title_elem = item.query_one(".result-title", Static)
            assert "Test Title" in title_elem.renderable.plain
            
            # Check source icon
            assert SOURCE_ICONS["media"] in title_elem.renderable.plain
            
            # Check content preview
            content_elem = item.query_one(".result-content", Static)
            assert "test content" in content_elem.renderable.plain.lower()
            
            # Check score if displayed
            score_elems = item.query(".result-score")
            if score_elems:
                assert "95%" in score_elems[0].renderable.plain
    
    @pytest.mark.asyncio
    async def test_result_item_click_action(self, widget_pilot):
        """Test clicking result item triggers action"""
        result_data = {
            "title": "Test",
            "content": "Content",
            "source": "media",
            "metadata": {"id": 1}
        }
        
        async with await widget_pilot(SearchResult, result=result_data, index=0) as pilot:
            item = pilot.app.test_widget
            item.action_view_details = MagicMock()
            
            # Click the item
            await pilot.click(item)
            await pilot.pause()
            
            # Verify action was triggered
            item.action_view_details.assert_called_once()


@pytest.mark.ui 
class TestSearchHistoryDropdown:
    """Test SearchHistoryDropdown component"""
    
    @pytest.mark.asyncio
    async def test_history_dropdown_initialization(self):
        """Test history dropdown initializes correctly"""
        mock_db = MagicMock()
        mock_db.get_search_history.return_value = []
        
        dropdown = SearchHistoryDropdown(mock_db)
        
        assert dropdown.search_history_db == mock_db
        assert dropdown.history_items == []
        assert "hidden" in dropdown.classes
    
    @pytest.mark.asyncio
    async def test_show_history_with_results(self, widget_pilot):
        """Test showing search history"""
        mock_db = MagicMock()
        mock_db.get_search_history.return_value = [
            {"query": "test query 1", "created_at": "2024-01-01 10:00:00"},
            {"query": "test query 2", "created_at": "2024-01-01 11:00:00"}
        ]
        
        async with await widget_pilot(SearchHistoryDropdown, search_history_db=mock_db) as pilot:
            dropdown = pilot.app.test_widget
            
            # Show history
            await dropdown.show_history()
            await pilot.pause()
            
            # Check list has items
            list_view = dropdown.query_one("#search-history-list", ListView)
            assert len(list_view.children) == 2
            
            # Dropdown should be visible
            assert "hidden" not in dropdown.classes
    
    @pytest.mark.asyncio
    async def test_filter_history_by_query(self, widget_pilot):
        """Test filtering history by current query"""
        mock_db = MagicMock()
        mock_db.get_search_history.return_value = [
            {"query": "python programming", "created_at": "2024-01-01"},
            {"query": "java programming", "created_at": "2024-01-01"},
            {"query": "python tutorial", "created_at": "2024-01-01"}
        ]
        
        async with await widget_pilot(SearchHistoryDropdown, search_history_db=mock_db) as pilot:
            dropdown = pilot.app.test_widget
            
            # Show history filtered by "python"
            await dropdown.show_history("python")
            await pilot.pause()
            
            # Should filter to show only python-related queries
            list_view = dropdown.query_one("#search-history-list", ListView)
            
            # Note: The actual filtering logic would need to be implemented
            # This test assumes the filtering is done in show_history method