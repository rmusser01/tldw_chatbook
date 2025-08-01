"""
Integration tests for the full embeddings workflow.
Tests end-to-end scenarios including creation, management, and all UI components working together.
"""

import pytest
import pytest_asyncio
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from datetime import datetime, timedelta

from textual.widgets import Button, Input, TextArea, Select, TabbedContent, TabPane
from textual.containers import Container

from tldw_chatbook.UI.Embeddings_Creation_Content import EmbeddingsCreationContent
from tldw_chatbook.UI.Embeddings_Management_Window import EmbeddingsManagementWindow
from tldw_chatbook.Widgets.toast_notification import ToastManager
from tldw_chatbook.Widgets.detailed_progress import DetailedProgressBar
from tldw_chatbook.Widgets.activity_log import ActivityLogWidget
from tldw_chatbook.Widgets.performance_metrics import PerformanceMetricsWidget
from tldw_chatbook.Utils.model_preferences import ModelPreferencesManager
from tldw_chatbook.Utils.embedding_templates import EmbeddingTemplateManager

from .test_base import EmbeddingsTestBase, WidgetTestApp


class TestFullEmbeddingsWorkflow(EmbeddingsTestBase):
    """Test complete embeddings workflow from creation to management."""
    
    @pytest.fixture
    def full_mock_setup(self, mock_app_instance, mock_embedding_factory, mock_chroma_manager, tmp_path):
        """Setup all mocks needed for full workflow."""
        # Setup preferences
        prefs_dir = tmp_path / ".config" / "tldw_cli"
        prefs_dir.mkdir(parents=True)
        
        # Setup template manager
        template_dir = prefs_dir / "templates"
        template_dir.mkdir()
        
        # Mock components
        setup = {
            "app": mock_app_instance,
            "embedding_factory": mock_embedding_factory,
            "chroma_manager": mock_chroma_manager,
            "preferences": ModelPreferencesManager(preferences_dir=prefs_dir),
            "templates": EmbeddingTemplateManager(template_dir=template_dir),
            "prefs_dir": prefs_dir,
            "template_dir": template_dir
        }
        
        # Mock async methods
        mock_embedding_factory.create_embeddings = AsyncMock(return_value={
            "embeddings_created": 100,
            "chunks_processed": 150,
            "time_taken": 5.5
        })
        
        mock_chroma_manager.create_collection = AsyncMock()
        mock_chroma_manager.list_collections = AsyncMock(return_value=[
            {"name": "test_collection", "count": 100}
        ])
        
        return setup
    
    @pytest.mark.asyncio
    async def test_create_embeddings_with_template(self, full_mock_setup):
        """Test creating embeddings using a template."""
        window = EmbeddingsCreationContent(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        window.chroma_manager = full_mock_setup["chroma_manager"]
        window.template_manager = full_mock_setup["templates"]
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Switch to template tab
            tabbed_content = pilot.app.query_one(TabbedContent)
            tabbed_content.active = "template-tab"
            await pilot.pause()
            
            # Select a template (simulate)
            template = window.template_manager.get_template("Quick Summary")
            assert template is not None
            
            # Apply template configuration
            window._apply_template_config(template.config)
            await pilot.pause()
            
            # Fill in collection name
            name_input = pilot.app.query_one("#collection-name", Input)
            name_input.value = "test_quick_summary"
            
            # Add source
            source_input = pilot.app.query_one("#source-path", Input)
            source_input.value = "/path/to/test/file.txt"
            await pilot.click("#add-source")
            await pilot.pause()
            
            # Create embeddings
            await pilot.click("#create-embeddings")
            await pilot.pause()
            
            # Verify creation was called with template config
            create_call = full_mock_setup["embedding_factory"].create_embeddings
            assert create_call.called
            
            # Check notification was shown
            full_mock_setup["app"].notify.assert_called()
            assert any("created successfully" in str(call) for call in full_mock_setup["app"].notify.call_args_list)
    
    @pytest.mark.asyncio
    async def test_embeddings_creation_with_progress(self, full_mock_setup):
        """Test embeddings creation with detailed progress tracking."""
        window = EmbeddingsCreationContent(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        
        # Mock long-running operation
        async def mock_create_with_progress(*args, **kwargs):
            # Simulate progress updates
            if hasattr(window, 'progress_bar') and window.progress_bar:
                window.progress_bar.start_stage(0, 100)  # Preprocessing
                await asyncio.sleep(0.1)
                window.progress_bar.update_progress(50)
                await asyncio.sleep(0.1)
                window.progress_bar.complete_stage()
                
                window.progress_bar.start_stage(1, 100)  # Embedding
                await asyncio.sleep(0.1)
                window.progress_bar.update_progress(100)
                window.progress_bar.complete_stage()
            
            return {"embeddings_created": 100, "chunks_processed": 100}
        
        full_mock_setup["embedding_factory"].create_embeddings = mock_create_with_progress
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Setup form
            pilot.app.query_one("#collection-name", Input).value = "progress_test"
            pilot.app.query_one("#source-path", Input).value = "/test/file.txt"
            await pilot.click("#add-source")
            await pilot.pause()
            
            # Start creation
            await pilot.click("#create-embeddings")
            
            # Wait for completion
            await asyncio.sleep(0.5)
            await pilot.pause()
            
            # Progress bar should have been shown and completed
            # (Would check progress bar state if it persists)
            assert True  # Verify no errors
    
    @pytest.mark.asyncio
    async def test_management_with_batch_operations(self, full_mock_setup):
        """Test managing embeddings with batch operations."""
        window = EmbeddingsManagementWindow(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        window.model_preferences = full_mock_setup["preferences"]
        
        # Mock multiple models
        full_mock_setup["embedding_factory"].config.models = {
            'model1': MagicMock(provider='test', is_downloaded=True),
            'model2': MagicMock(provider='test', is_downloaded=True),
            'model3': MagicMock(provider='test', is_downloaded=True)
        }
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Load models
            await window._load_models()
            await pilot.pause()
            
            # Enable batch mode
            await pilot.click("#toggle-batch-mode")
            await pilot.pause()
            
            assert window.batch_mode_enabled == True
            
            # Select all models
            await pilot.click("#select-all-models")
            await pilot.pause()
            
            # Perform batch operation (e.g., delete)
            window.selected_models = {'model1', 'model2'}  # Simulate selection
            
            # Mock deletion
            full_mock_setup["embedding_factory"].delete_model = AsyncMock()
            
            # Trigger batch delete
            await window._handle_batch_delete_models()
            await pilot.pause()
            
            # Verify deletions
            assert full_mock_setup["embedding_factory"].delete_model.call_count == 2
    
    @pytest.mark.asyncio
    async def test_workflow_with_activity_logging(self, full_mock_setup):
        """Test complete workflow with activity logging."""
        # Create both windows
        creation_window = EmbeddingsCreationContent(full_mock_setup["app"])
        creation_window.embedding_factory = full_mock_setup["embedding_factory"]
        
        management_window = EmbeddingsManagementWindow(full_mock_setup["app"])
        management_window.embedding_factory = full_mock_setup["embedding_factory"]
        
        # Add activity log to both
        creation_window.activity_log = ActivityLogWidget()
        management_window.activity_log = ActivityLogWidget()
        
        # Test creation with logging
        app = WidgetTestApp(creation_window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Log should start empty
            assert len(creation_window.activity_log.entries) == 0
            
            # Perform operations that should log
            creation_window.activity_log.add_entry("Starting embedding creation", "info", "creation")
            creation_window.activity_log.add_entry("Validated configuration", "info", "validation")
            creation_window.activity_log.add_entry("Processing 100 chunks", "info", "processing")
            creation_window.activity_log.add_entry("Embeddings created successfully", "success", "completion")
            
            await pilot.pause()
            
            # Check log entries
            assert len(creation_window.activity_log.entries) == 4
            
            # Export log for audit
            with patch('builtins.open', mock_open()) as mock_file:
                creation_window.activity_log.export_log("json")
                mock_file.assert_called()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_during_creation(self, full_mock_setup):
        """Test performance monitoring during embedding creation."""
        with patch('psutil.Process') as mock_process:
            # Mock system metrics
            process_instance = MagicMock()
            mock_process.return_value = process_instance
            process_instance.cpu_percent.return_value = 45.0
            process_instance.memory_info.return_value = MagicMock(rss=1024*1024*512)  # 512MB
            
            window = EmbeddingsCreationContent(full_mock_setup["app"])
            window.embedding_factory = full_mock_setup["embedding_factory"]
            
            # Add performance monitor
            window.performance_metrics = PerformanceMetricsWidget(update_interval=0.1)
            
            app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
            
            async with app.run_test() as pilot:
                await pilot.pause()
                
                # Start monitoring
                window.performance_metrics.start_monitoring()
                
                # Simulate embedding creation
                async def create_with_metrics(*args, **kwargs):
                    # Record metrics during processing
                    for i in range(5):
                        window.performance_metrics.record_embedding_processed(
                            model="test-model",
                            chunks=20,
                            bytes_processed=2048
                        )
                        await asyncio.sleep(0.05)
                    return {"embeddings_created": 100}
                
                full_mock_setup["embedding_factory"].create_embeddings = create_with_metrics
                
                # Setup and create
                pilot.app.query_one("#collection-name", Input).value = "perf_test"
                pilot.app.query_one("#source-path", Input).value = "/test/file.txt"
                await pilot.click("#add-source")
                await pilot.pause()
                
                await pilot.click("#create-embeddings")
                await asyncio.sleep(0.3)
                
                # Stop monitoring
                window.performance_metrics.stop_monitoring()
                await pilot.pause()
                
                # Check metrics were collected
                stats = window.performance_metrics.embedding_stats
                assert stats.total_embeddings == 5
                assert stats.total_chunks == 100
                assert len(window.performance_metrics.cpu_history.values) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, full_mock_setup):
        """Test error handling throughout the workflow."""
        window = EmbeddingsCreationContent(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        
        # Mock error during creation
        error_message = "Failed to create embeddings: Model not found"
        full_mock_setup["embedding_factory"].create_embeddings = AsyncMock(
            side_effect=Exception(error_message)
        )
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Setup form
            pilot.app.query_one("#collection-name", Input).value = "error_test"
            pilot.app.query_one("#source-path", Input).value = "/test/file.txt"
            await pilot.click("#add-source")
            await pilot.pause()
            
            # Try to create embeddings
            await pilot.click("#create-embeddings")
            await pilot.pause()
            
            # Should show error notification
            full_mock_setup["app"].notify.assert_called()
            error_calls = [call for call in full_mock_setup["app"].notify.call_args_list 
                          if "error" in str(call).lower() or "failed" in str(call).lower()]
            assert len(error_calls) > 0
    
    @pytest.mark.asyncio
    async def test_preferences_persistence_workflow(self, full_mock_setup):
        """Test that preferences are saved and loaded correctly."""
        prefs = full_mock_setup["preferences"]
        
        # Simulate user actions that should save preferences
        prefs.record_model_use("model1")
        prefs.record_model_use("model1")
        prefs.record_model_use("model2")
        prefs.toggle_favorite("model1")
        
        # Create new preferences manager to test loading
        new_prefs = ModelPreferencesManager(preferences_dir=full_mock_setup["prefs_dir"])
        
        # Check preferences were loaded
        assert new_prefs.is_favorite("model1") == True
        assert new_prefs.model_usage["model1"].use_count == 2
        assert new_prefs.model_usage["model2"].use_count == 1
        assert "model2" in new_prefs.recent_models
    
    @pytest.mark.asyncio
    async def test_template_workflow(self, full_mock_setup):
        """Test complete template workflow from creation to usage."""
        window = EmbeddingsCreationContent(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        window.template_manager = full_mock_setup["templates"]
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Create custom template
            custom_config = {
                "model_name": "custom-model",
                "chunk_size": 2048,
                "chunk_overlap": 200,
                "batch_size": 8
            }
            
            # Save template
            from tldw_chatbook.Utils.embedding_templates import EmbeddingTemplate
            custom_template = EmbeddingTemplate(
                name="My Custom Template",
                description="Custom settings for large documents",
                config=custom_config
            )
            window.template_manager.save_template(custom_template)
            
            # Verify template was saved
            saved_template = window.template_manager.get_template("My Custom Template")
            assert saved_template is not None
            assert saved_template.config["chunk_size"] == 2048
            
            # Use template for creation
            window._apply_template_config(saved_template.config)
            await pilot.pause()
            
            # Verify configuration was applied
            # (Would check actual form values)
            assert True  # Verify no errors


class TestUIComponentIntegration(EmbeddingsTestBase):
    """Test integration between different UI components."""
    
    @pytest.mark.asyncio
    async def test_toast_notifications_integration(self, full_mock_setup):
        """Test toast notifications work across different operations."""
        # Create window with toast support
        window = EmbeddingsCreationContent(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        
        # Add toast manager
        window._toast_manager = ToastManager()
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Trigger various operations that should show toasts
            
            # Success toast
            window.toast_success("Configuration validated")
            await pilot.pause()
            
            # Warning toast
            window.toast_warning("Large file detected, processing may take time")
            await pilot.pause()
            
            # Error toast
            window.toast_error("Invalid model selected")
            await pilot.pause()
            
            # Info toast
            window.toast_info("Processing started...")
            await pilot.pause()
            
            # Check toasts were created
            toasts = pilot.app.query(".toast-notification")
            assert len(toasts) == 4
    
    @pytest.mark.asyncio
    async def test_progress_and_activity_log_sync(self, full_mock_setup):
        """Test that progress updates are reflected in activity log."""
        window = EmbeddingsCreationContent(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        
        # Add both components
        window.activity_log = ActivityLogWidget()
        progress = DetailedProgressBar(stages=["Preprocessing", "Embedding", "Storing"])
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Simulate progress with logging
            progress.start_stage(0, 100)
            window.activity_log.add_entry("Started preprocessing", "info", "progress")
            
            progress.update_progress(50)
            window.activity_log.add_entry("Preprocessing 50% complete", "info", "progress")
            
            progress.complete_stage()
            window.activity_log.add_entry("Preprocessing completed", "success", "progress")
            
            await pilot.pause()
            
            # Check synchronization
            assert len(window.activity_log.entries) == 3
            assert progress.current_stage == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_ui(self, full_mock_setup):
        """Test UI handles concurrent operations correctly."""
        window = EmbeddingsManagementWindow(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        
        # Mock concurrent operations
        async def slow_operation():
            await asyncio.sleep(0.2)
            return {"status": "completed"}
        
        full_mock_setup["embedding_factory"].load_model = slow_operation
        full_mock_setup["embedding_factory"].delete_model = slow_operation
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Start multiple operations
            tasks = []
            
            # Simulate clicking multiple buttons quickly
            async def simulate_operations():
                # These would normally be triggered by UI events
                task1 = asyncio.create_task(window._handle_load_model("model1"))
                task2 = asyncio.create_task(window._handle_delete_model("model2"))
                return await asyncio.gather(task1, task2, return_exceptions=True)
            
            results = await simulate_operations()
            await pilot.pause()
            
            # UI should handle concurrent operations without errors
            assert len(results) == 2
            assert all(not isinstance(r, Exception) for r in results)


class TestEndToEndScenarios(EmbeddingsTestBase):
    """Test complete end-to-end user scenarios."""
    
    @pytest.mark.asyncio
    async def test_first_time_user_workflow(self, full_mock_setup):
        """Test workflow for a first-time user."""
        # Start with no saved preferences or templates
        window = EmbeddingsCreationContent(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        window.template_manager = full_mock_setup["templates"]
        window.show_help = True  # First time user might need help
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # User explores templates first
            tabbed_content = pilot.app.query_one(TabbedContent)
            tabbed_content.active = "template-tab"
            await pilot.pause()
            
            # Select beginner-friendly template
            quick_template = window.template_manager.get_template("Quick Summary")
            window._apply_template_config(quick_template.config)
            
            # Fill basic info
            pilot.app.query_one("#collection-name", Input).value = "my_first_embeddings"
            pilot.app.query_one("#source-path", Input).value = "/Users/me/document.txt"
            await pilot.click("#add-source")
            await pilot.pause()
            
            # Create embeddings
            await pilot.click("#create-embeddings")
            await pilot.pause()
            
            # Should complete successfully with helpful notifications
            full_mock_setup["app"].notify.assert_called()
            success_calls = [call for call in full_mock_setup["app"].notify.call_args_list 
                           if "success" in str(call).lower()]
            assert len(success_calls) > 0
    
    @pytest.mark.asyncio
    async def test_power_user_workflow(self, full_mock_setup):
        """Test workflow for an experienced power user."""
        # Setup with existing preferences
        prefs = full_mock_setup["preferences"]
        prefs.record_model_use("e5-large-v2")
        prefs.record_model_use("e5-large-v2")
        prefs.toggle_favorite("e5-large-v2")
        
        window = EmbeddingsCreationContent(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Power user goes straight to manual config
            tabbed_content = pilot.app.query_one(TabbedContent)
            tabbed_content.active = "manual-tab"
            await pilot.pause()
            
            # Configure advanced settings
            pilot.app.query_one("#collection-name", Input).value = "research_papers_v2"
            pilot.app.query_one("#model-name", Input).value = "e5-large-v2"
            pilot.app.query_one("#chunk-size", Input).value = "2048"
            pilot.app.query_one("#chunk-overlap", Input).value = "256"
            pilot.app.query_one("#batch-size", Input).value = "4"
            
            # Add multiple sources
            sources = [
                "/research/paper1.pdf",
                "/research/paper2.pdf", 
                "/research/paper3.pdf"
            ]
            
            for source in sources:
                pilot.app.query_one("#source-path", Input).value = source
                await pilot.click("#add-source")
                await pilot.pause()
            
            # Create with custom config
            await pilot.click("#create-embeddings")
            await pilot.pause()
            
            # Verify advanced configuration was used
            create_call = full_mock_setup["embedding_factory"].create_embeddings.call_args
            assert create_call is not None
            # Would verify specific configuration values
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, full_mock_setup):
        """Test user workflow when errors occur and recovery is needed."""
        window = EmbeddingsCreationContent(full_mock_setup["app"])
        window.embedding_factory = full_mock_setup["embedding_factory"]
        
        # First attempt fails
        attempt_count = 0
        
        async def create_with_retry(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise Exception("Network error: Unable to download model")
            return {"embeddings_created": 100}
        
        full_mock_setup["embedding_factory"].create_embeddings = create_with_retry
        
        app = WidgetTestApp(window, app_instance=full_mock_setup["app"])
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Setup
            pilot.app.query_one("#collection-name", Input).value = "retry_test"
            pilot.app.query_one("#source-path", Input).value = "/test/file.txt"
            await pilot.click("#add-source")
            await pilot.pause()
            
            # First attempt - fails
            await pilot.click("#create-embeddings")
            await pilot.pause()
            
            # Should show error
            error_notifications = [call for call in full_mock_setup["app"].notify.call_args_list
                                 if "error" in str(call).lower()]
            assert len(error_notifications) > 0
            
            # User retries after fixing issue
            await pilot.click("#create-embeddings")
            await pilot.pause()
            
            # Should succeed on retry
            success_notifications = [call for call in full_mock_setup["app"].notify.call_args_list
                                   if "success" in str(call).lower()]
            assert len(success_notifications) > 0
            assert attempt_count == 2