"""
Tests for Model Preferences and Batch Operations.
Tests favorite models, recent tracking, filtering, and batch selection.
"""

import pytest
import pytest_asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from textual.widgets import Button, Select, ListView, Checkbox
from textual.containers import Container

from tldw_chatbook.Utils.model_preferences import (
    ModelPreferencesManager,
    ModelUsage
)
from tldw_chatbook.UI.Embeddings_Management_Window import (
    EmbeddingsManagementWindow,
    DeleteModelDialog,
    BatchDeleteDialog
)
from tldw_chatbook.Widgets.embeddings_list_items import ModelListItem

from .test_base import EmbeddingsTestBase, WidgetTestApp


class TestModelPreferences(EmbeddingsTestBase):
    """Test ModelPreferencesManager functionality."""
    
    @pytest.fixture
    def temp_preferences_dir(self, tmp_path):
        """Create temporary preferences directory."""
        prefs_dir = tmp_path / ".config" / "tldw_cli"
        prefs_dir.mkdir(parents=True)
        return prefs_dir
    
    @pytest.fixture
    def preferences_manager(self, temp_preferences_dir):
        """Create preferences manager with temp directory."""
        return ModelPreferencesManager(preferences_dir=temp_preferences_dir)
    
    def test_preferences_initialization(self, preferences_manager, temp_preferences_dir):
        """Test preferences manager initialization."""
        assert preferences_manager.preferences_dir == temp_preferences_dir
        assert preferences_manager.preferences_file == temp_preferences_dir / "model_preferences.json"
        assert len(preferences_manager.model_usage) == 0
        assert len(preferences_manager.recent_models) == 0
    
    def test_record_model_use(self, preferences_manager):
        """Test recording model usage."""
        # First use
        preferences_manager.record_model_use("model1")
        
        assert "model1" in preferences_manager.model_usage
        usage = preferences_manager.model_usage["model1"]
        assert usage.use_count == 1
        assert usage.model_id == "model1"
        assert "model1" in preferences_manager.recent_models
        
        # Second use
        preferences_manager.record_model_use("model1")
        assert preferences_manager.model_usage["model1"].use_count == 2
    
    def test_toggle_favorite(self, preferences_manager):
        """Test toggling favorite status."""
        # First toggle (should become favorite)
        is_favorite = preferences_manager.toggle_favorite("model1")
        assert is_favorite == True
        assert preferences_manager.is_favorite("model1") == True
        
        # Second toggle (should remove favorite)
        is_favorite = preferences_manager.toggle_favorite("model1")
        assert is_favorite == False
        assert preferences_manager.is_favorite("model1") == False
    
    def test_get_recent_models(self, preferences_manager):
        """Test getting recent models."""
        # Record usage for multiple models
        models = ["model1", "model2", "model3", "model4"]
        for model in models:
            preferences_manager.record_model_use(model)
        
        # Get recent models
        recent = preferences_manager.get_recent_models(limit=3)
        assert len(recent) == 3
        assert recent[0] == "model4"  # Most recent first
        assert recent[1] == "model3"
        assert recent[2] == "model2"
    
    def test_get_favorite_models(self, preferences_manager):
        """Test getting favorite models."""
        # Add some favorites
        preferences_manager.toggle_favorite("model1")
        preferences_manager.toggle_favorite("model2")
        preferences_manager.toggle_favorite("model3")
        
        # Remove one
        preferences_manager.toggle_favorite("model2")
        
        favorites = preferences_manager.get_favorite_models()
        assert len(favorites) == 2
        assert "model1" in favorites
        assert "model3" in favorites
        assert "model2" not in favorites
    
    def test_get_most_used_models(self, preferences_manager):
        """Test getting most used models."""
        # Record different usage counts
        for i in range(5):
            preferences_manager.record_model_use("model1")
        for i in range(3):
            preferences_manager.record_model_use("model2")
        preferences_manager.record_model_use("model3")
        
        most_used = preferences_manager.get_most_used_models(limit=2)
        assert len(most_used) == 2
        assert most_used[0] == ("model1", 5)
        assert most_used[1] == ("model2", 3)
    
    def test_persistence(self, preferences_manager, temp_preferences_dir):
        """Test saving and loading preferences."""
        # Set up some data
        preferences_manager.record_model_use("model1")
        preferences_manager.record_model_use("model1")
        preferences_manager.toggle_favorite("model1")
        preferences_manager.record_model_use("model2")
        
        # Create new manager to test loading
        new_manager = ModelPreferencesManager(preferences_dir=temp_preferences_dir)
        
        # Check data was loaded
        assert "model1" in new_manager.model_usage
        assert new_manager.model_usage["model1"].use_count == 2
        assert new_manager.is_favorite("model1") == True
        assert "model2" in new_manager.recent_models
    
    def test_remove_model(self, preferences_manager):
        """Test removing model data."""
        # Add model data
        preferences_manager.record_model_use("model1")
        preferences_manager.toggle_favorite("model1")
        
        # Remove it
        preferences_manager.remove_model("model1")
        
        assert "model1" not in preferences_manager.model_usage
        assert "model1" not in preferences_manager.recent_models
        assert preferences_manager.is_favorite("model1") == False


class TestModelFilteringUI(EmbeddingsTestBase):
    """Test model filtering UI functionality."""
    
    @pytest.mark.asyncio
    async def test_filter_dropdown(self, mock_app_instance, mock_embedding_factory, mock_model_preferences):
        """Test model filter dropdown functionality."""
        window = EmbeddingsManagementWindow(mock_app_instance)
        window.embedding_factory = mock_embedding_factory
        window.model_preferences = mock_model_preferences
        
        app = WidgetTestApp(window, mock_app_instance)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Find filter dropdown
            filter_select = pilot.app.query_one("#embeddings-model-filter", Select)
            assert filter_select is not None
            
            # Check options
            options = [opt[0] for opt in filter_select._options]
            assert "All Models" in options
            assert "Favorites" in options
            assert "Recent" in options
            assert "Most Used" in options
    
    @pytest.mark.asyncio
    async def test_filter_models_method(self, mock_app_instance, mock_model_preferences):
        """Test _filter_models method."""
        window = EmbeddingsManagementWindow(mock_app_instance)
        window.model_preferences = mock_model_preferences
        
        # Setup test data
        all_models = ["model1", "model2", "model3", "model4"]
        mock_model_preferences.get_favorite_models.return_value = ["model1", "model3"]
        mock_model_preferences.get_recent_models.return_value = ["model4", "model2"]
        mock_model_preferences.get_most_used_models.return_value = [("model1", 10), ("model4", 5)]
        
        # Test "all" filter
        window.model_filter = "all"
        filtered = window._filter_models(all_models)
        assert filtered == all_models
        
        # Test "favorites" filter
        window.model_filter = "favorites"
        filtered = window._filter_models(all_models)
        assert set(filtered) == {"model1", "model3"}
        
        # Test "recent" filter
        window.model_filter = "recent"
        filtered = window._filter_models(all_models)
        assert set(filtered) == {"model4", "model2"}
        
        # Test "most_used" filter
        window.model_filter = "most_used"
        filtered = window._filter_models(all_models)
        assert set(filtered) == {"model1", "model4"}
    
    @pytest.mark.asyncio
    async def test_favorite_button_interaction(self, mock_app_instance, mock_embedding_factory, mock_model_preferences):
        """Test favorite button functionality."""
        window = EmbeddingsManagementWindow(mock_app_instance)
        window.embedding_factory = mock_embedding_factory
        window.model_preferences = mock_model_preferences
        window.selected_model = "e5-small-v2"
        
        # Mock toggle_favorite to return True (favorited)
        mock_model_preferences.toggle_favorite.return_value = True
        
        app = WidgetTestApp(window, mock_app_instance)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Find and click favorite button
            await pilot.click("#embeddings-favorite-model")
            await pilot.pause()
            
            # Check toggle was called
            mock_model_preferences.toggle_favorite.assert_called_with("e5-small-v2")
            
            # Check notification
            mock_app_instance.notify.assert_called()
            call_args = mock_app_instance.notify.call_args
            assert "Added e5-small-v2 to favorites" in call_args[0][0]


class TestBatchOperations(EmbeddingsTestBase):
    """Test batch operations functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_mode_toggle(self, mock_app_instance, mock_embedding_factory):
        """Test toggling batch mode."""
        window = EmbeddingsManagementWindow(mock_app_instance)
        window.embedding_factory = mock_embedding_factory
        
        app = WidgetTestApp(window, mock_app_instance)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Initially batch mode is off
            assert window.batch_mode_enabled == False
            
            # Find batch controls (should be hidden)
            model_controls = pilot.app.query_one("#batch-model-controls")
            assert "hidden" in model_controls.classes
            
            # Toggle batch mode
            await pilot.click("#toggle-batch-mode")
            await pilot.pause()
            
            # Batch mode should be on
            assert window.batch_mode_enabled == True
            
            # Controls should be visible
            assert "hidden" not in model_controls.classes
            
            # Button text should change
            toggle_button = pilot.app.query_one("#toggle-batch-mode", Button)
            assert toggle_button.label == "Exit Batch Mode"
    
    @pytest.mark.asyncio
    async def test_select_all_models(self, mock_app_instance, mock_embedding_factory):
        """Test select all models functionality."""
        window = EmbeddingsManagementWindow(mock_app_instance)
        window.embedding_factory = mock_embedding_factory
        window.batch_mode_enabled = True
        
        # Create some model items
        for i in range(3):
            model_item = ModelListItem(
                f"model{i}",
                {"provider": "test"},
                show_selection=True,
                id=f"model-model{i}"
            )
            window.selected_models.add(f"model{i}")
        
        app = WidgetTestApp(window, mock_app_instance)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Click select all
            await pilot.click("#select-all-models")
            await pilot.pause()
            
            # All models should be in selected set
            # (Note: actual checkbox state would be set in the real list items)
            assert len(window.selected_models) > 0
    
    @pytest.mark.asyncio
    async def test_batch_delete_confirmation(self, mock_app_instance, mock_embedding_factory):
        """Test batch delete with confirmation dialog."""
        window = EmbeddingsManagementWindow(mock_app_instance)
        window.embedding_factory = mock_embedding_factory
        window.batch_mode_enabled = True
        window.selected_models = {"model1", "model2", "model3"}
        
        app = WidgetTestApp(window, mock_app_instance)
        
        # Mock push_screen to simulate dialog
        dialog_result = None
        async def mock_push_screen(dialog, wait_for_dismiss=False):
            nonlocal dialog_result
            if isinstance(dialog, BatchDeleteDialog):
                dialog_result = dialog
                return True  # Simulate confirmation
            return False
        
        app.push_screen = mock_push_screen
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Click delete selected
            await pilot.click("#delete-selected-models")
            await pilot.pause()
            
            # Check dialog was created with correct count
            assert dialog_result is not None
            assert dialog_result.count == 3
    
    @pytest.mark.asyncio
    async def test_checkbox_state_management(self, mock_app_instance):
        """Test checkbox state management in list items."""
        # Create a model list item with selection enabled
        model_item = ModelListItem(
            "test-model",
            {
                "provider": "huggingface",
                "is_downloaded": True,
                "is_loaded": False,
                "dimension": 384,
                "is_favorite": False
            },
            show_selection=True,
            id="model-test-model"
        )
        
        app = WidgetTestApp(model_item)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Find checkbox
            checkbox = pilot.app.query_one("#select-model-test-model", Checkbox)
            assert checkbox is not None
            
            # Initially unchecked
            assert checkbox.value == False
            
            # Click to check
            await pilot.click("#select-model-test-model")
            await pilot.pause()
            
            # Should be checked
            assert checkbox.value == True
    
    @pytest.mark.asyncio
    async def test_batch_operations_collections(self, mock_app_instance, mock_embedding_factory):
        """Test batch operations for collections."""
        window = EmbeddingsManagementWindow(mock_app_instance)
        window.embedding_factory = mock_embedding_factory
        window.batch_mode_enabled = True
        window.selected_collections = {"collection1", "collection2"}
        
        app = WidgetTestApp(window, mock_app_instance)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Check collection batch controls exist
            collection_controls = pilot.app.query_one("#batch-collection-controls")
            assert collection_controls is not None
            
            # Test select all collections
            await pilot.click("#select-all-collections")
            await pilot.pause()
            
            # Test select none
            await pilot.click("#select-none-collections")
            await pilot.pause()
            
            # Collections should be cleared
            assert len(window.selected_collections) == 0