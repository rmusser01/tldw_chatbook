# test_ingestion_regression.py
"""
Regression tests for the media ingestion UI system.

This test suite compares legacy vs redesigned implementations to ensure:
1. Feature parity between legacy and redesigned windows
2. Configuration compatibility
3. Data validation consistency
4. No regressions in existing functionality

These tests help maintain backward compatibility while transitioning to new UI architecture.
"""

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

# Third-party Libraries
from textual.app import App
from textual.widgets import Button, Input, Select, Checkbox, TextArea
from textual.containers import Container, VerticalScroll

# Local Imports
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Media_Ingest.IngestUIFactory import IngestUIFactory
from tldw_chatbook.Widgets.Media_Ingest.Ingest_Local_Video_Window import VideoIngestWindowRedesigned
from tldw_chatbook.Widgets.Media_Ingest.IngestLocalVideoWindowSimplified import IngestLocalVideoWindowSimplified

# Try to import legacy windows (may not exist for all media types)
try:
    from tldw_chatbook.Widgets.Media_Ingest.IngestLocalVideoWindow import IngestLocalVideoWindow
    LEGACY_VIDEO_AVAILABLE = True
except ImportError:
    LEGACY_VIDEO_AVAILABLE = False

try:
    from tldw_chatbook.Widgets.Media_Ingest.Ingest_Local_Audio_Window import IngestLocalAudioWindow
    LEGACY_AUDIO_AVAILABLE = True
except ImportError:
    LEGACY_AUDIO_AVAILABLE = False


class TestLegacyVsRedesignedParity:
    """Test feature parity between legacy and redesigned implementations."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not LEGACY_VIDEO_AVAILABLE, reason="Legacy video window not available")
    async def test_video_window_essential_elements_parity(self):
        """Test that redesigned video window has all essential elements from legacy version."""
        
        # Test redesigned version
        class RedesignedApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        redesigned_app = RedesignedApp()
        async with redesigned_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Collect essential elements from redesigned window
            redesigned_elements = {
                "browse_files_button": len(redesigned_app.query(Button).filter(lambda b: "Browse" in str(b.label))),
                "clear_files_button": len(redesigned_app.query(Button).filter(lambda b: "Clear" in str(b.label))),
                "title_input": len(redesigned_app.query(Input).filter(lambda i: i.id and "title" in i.id.lower())),
                "author_input": len(redesigned_app.query(Input).filter(lambda i: i.id and "author" in i.id.lower())),
                "keywords_input": len(redesigned_app.query(Input).filter(lambda i: i.id and "keyword" in i.id.lower())),
                "process_button": len(redesigned_app.query(Button).filter(lambda b: "Process" in str(b.label) or "Submit" in str(b.label))),
                "extract_audio_checkbox": len(redesigned_app.query(Checkbox).filter(lambda c: "audio" in str(c.label).lower())),
                "transcription_selects": len(redesigned_app.query(Select).filter(lambda s: s.id and "transcription" in s.id.lower())),
            }
        
        # Test legacy version
        class LegacyApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield IngestLocalVideoWindow(self)
        
        legacy_app = LegacyApp()
        async with legacy_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Collect essential elements from legacy window
            legacy_elements = {
                "browse_files_button": len(legacy_app.query(Button).filter(lambda b: "Browse" in str(b.label))),
                "clear_files_button": len(legacy_app.query(Button).filter(lambda b: "Clear" in str(b.label))),
                "title_input": len(legacy_app.query(Input).filter(lambda i: i.id and "title" in i.id.lower())),
                "author_input": len(legacy_app.query(Input).filter(lambda i: i.id and "author" in i.id.lower())),
                "keywords_input": len(legacy_app.query(Input).filter(lambda i: i.id and "keyword" in i.id.lower())),
                "process_button": len(legacy_app.query(Button).filter(lambda b: "Process" in str(b.label) or "Submit" in str(b.label))),
                "extract_audio_checkbox": len(legacy_app.query(Checkbox).filter(lambda c: "audio" in str(c.label).lower())),
                "transcription_selects": len(legacy_app.query(Select).filter(lambda s: s.id and "transcription" in s.id.lower())),
            }
        
        # Compare feature parity
        for feature, redesigned_count in redesigned_elements.items():
            legacy_count = legacy_elements.get(feature, 0)
            
            # Redesigned should have at least as many features as legacy
            assert redesigned_count >= legacy_count, \
                f"Redesigned window missing {feature}: has {redesigned_count}, legacy has {legacy_count}"

    @pytest.mark.asyncio
    async def test_simplified_vs_redesigned_improvements(self):
        """Test that redesigned window fixes known issues in simplified windows."""
        
        # Test simplified window (known to have issues)
        class SimplifiedApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield IngestLocalVideoWindowSimplified(self)
        
        simplified_app = SimplifiedApp()
        async with simplified_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Count issues in simplified window
            simplified_scroll_containers = simplified_app.query(VerticalScroll)
            simplified_inputs = simplified_app.query(Input)
            
            simplified_inputs_with_styling = 0
            for input_widget in simplified_inputs:
                if "form-input" in input_widget.classes or hasattr(input_widget.styles, 'height'):
                    simplified_inputs_with_styling += 1
        
        # Test redesigned window (should fix the issues)
        class RedesignedApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        redesigned_app = RedesignedApp()
        async with redesigned_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Count improvements in redesigned window
            redesigned_scroll_containers = redesigned_app.query(VerticalScroll)
            redesigned_inputs = redesigned_app.query(Input)
            
            redesigned_inputs_with_styling = 0
            for input_widget in redesigned_inputs:
                if "form-input" in input_widget.classes:
                    redesigned_inputs_with_styling += 1
        
        # Redesigned should fix scrolling issues (single scroll container)
        assert len(redesigned_scroll_containers) == 1, \
            f"Redesigned should have 1 scroll container, found {len(redesigned_scroll_containers)}"
        
        # Simplified may have multiple (which is the problem we're fixing)
        if len(simplified_scroll_containers) > 1:
            assert len(redesigned_scroll_containers) < len(simplified_scroll_containers), \
                "Redesigned should fix double-scrolling issue"
        
        # Redesigned should have better input styling
        if len(redesigned_inputs) > 0:
            redesigned_styling_ratio = redesigned_inputs_with_styling / len(redesigned_inputs)
            assert redesigned_styling_ratio > 0.8, \
                f"Redesigned inputs should be properly styled: {redesigned_styling_ratio:.2f} ratio"

    @pytest.mark.asyncio
    async def test_configuration_compatibility(self):
        """Test that both legacy and redesigned windows work with existing configurations."""
        # Test with various configuration scenarios
        test_configs = [
            {"api_settings": {"openai": {"models": ["gpt-4"]}}},
            {"api_settings": {"anthropic": {"models": ["claude-3-sonnet"]}}},
            {"api_settings": {}},  # Empty config
            None,  # No config
        ]
        
        for config in test_configs:
            # Test redesigned window
            class RedesignedTestApp(App):
                def __init__(self):
                    super().__init__()
                    self.app_config = config
                
                def compose(self):
                    yield VideoIngestWindowRedesigned(self)
            
            redesigned_app = RedesignedTestApp()
            async with redesigned_app.run_test() as pilot:
                await pilot.pause(0.5)
                
                # Should load without crashing
                assert redesigned_app.is_running, f"Redesigned window should handle config: {config}"
                
                video_window = redesigned_app.query_one(VideoIngestWindowRedesigned)
                assert video_window is not None, "Redesigned window should be created"
                
                # Essential elements should be present
                status_dashboard = redesigned_app.query_one("#status-dashboard")
                assert status_dashboard is not None, "Status dashboard should exist with any config"

    @pytest.mark.asyncio
    async def test_data_validation_consistency(self):
        """Test that validation rules are consistent between legacy and redesigned implementations."""
        # Create both window types
        class RedesignedApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        redesigned_app = RedesignedApp()
        async with redesigned_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            redesigned_window = redesigned_app.query_one(VideoIngestWindowRedesigned)
            
            # Test validation rules
            test_cases = [
                ("title-input", ""),  # Empty title
                ("title-input", "a"),  # Too short title
                ("title-input", "Valid Title"),  # Valid title
                ("keywords-input", ""),  # Empty keywords
                ("keywords-input", "valid,keywords,here"),  # Valid keywords
            ]
            
            redesigned_validations = {}
            for field_id, test_value in test_cases:
                error = redesigned_window.validate_field(field_id, test_value)
                redesigned_validations[(field_id, test_value)] = error
        
        # Note: If we had legacy validation to compare against, we would test consistency here
        # For now, we document the expected validation behavior
        
        # Title validation expectations
        assert redesigned_validations[("title-input", "")] is None, "Empty title should be valid (optional)"
        assert redesigned_validations[("title-input", "a")] is not None, "Single character title should be invalid"
        assert redesigned_validations[("title-input", "Valid Title")] is None, "Valid title should pass validation"
        
        # Keywords validation expectations
        assert redesigned_validations[("keywords-input", "")] is None, "Empty keywords should be valid (optional)"
        assert redesigned_validations[("keywords-input", "valid,keywords,here")] is None, "Valid keywords should pass"


class TestBackwardCompatibility:
    """Test that changes don't break existing functionality."""

    @pytest.mark.asyncio
    async def test_factory_backward_compatibility(self):
        """Test that factory still creates working UIs for all previously supported media types."""
        app = TldwCli()
        
        # These media types should have been supported before the redesign
        legacy_media_types = ["video", "audio", "document", "pdf", "ebook", "plaintext"]
        
        for media_type in legacy_media_types:
            # Should create some kind of working UI for each type
            ui_widget = IngestUIFactory.create_ui(app, media_type)
            
            from textual.widget import Widget
            assert isinstance(ui_widget, Widget), \
                f"Factory should still create valid Widget for {media_type}"
            
            assert ui_widget is not None, \
                f"Factory should not return None for previously supported {media_type}"

    @pytest.mark.asyncio
    async def test_removed_simplified_ui_handling(self):
        """Test that requests for removed 'simplified' UI style are handled gracefully."""
        app = TldwCli()
        
        # Mock configuration to request simplified UI (which was removed)
        with patch('tldw_chatbook.config.get_ingest_ui_style', return_value="simplified"):
            # Should not crash, should fallback to working UI
            try:
                ui_widget = IngestUIFactory.create_ui(app, "video")
                
                from textual.widget import Widget
                assert isinstance(ui_widget, Widget), "Should fallback to valid UI when simplified requested"
                assert ui_widget is not None, "Should not return None when simplified requested"
                
                # Should get redesigned version as fallback
                assert isinstance(ui_widget, VideoIngestWindowRedesigned), \
                    "Should fallback to redesigned UI when simplified requested"
                
            except Exception as e:
                pytest.fail(f"Factory should handle removed 'simplified' UI gracefully: {str(e)}")

    @pytest.mark.asyncio
    async def test_ui_style_migration_path(self):
        """Test that users can migrate from old to new UI styles smoothly."""
        app = TldwCli()
        
        # Test migration path: simplified → default → redesigned
        migration_styles = ["simplified", "default", "redesigned", "new"]
        
        for style in migration_styles:
            with patch('tldw_chatbook.config.get_ingest_ui_style', return_value=style):
                try:
                    ui_widget = IngestUIFactory.create_ui(app, "video")
                    
                    from textual.widget import Widget
                    assert isinstance(ui_widget, Widget), f"Style '{style}' should create valid widget"
                    
                    # All these styles should now point to the redesigned implementation for video
                    assert isinstance(ui_widget, VideoIngestWindowRedesigned), \
                        f"Style '{style}' should use redesigned implementation for video"
                    
                except Exception as e:
                    pytest.fail(f"Migration style '{style}' should work: {str(e)}")

    @pytest.mark.asyncio
    async def test_existing_user_workflows_still_work(self):
        """Test that common user workflows from the old UI still work in the new UI."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            video_window = app.query_one(VideoIngestWindowRedesigned)
            
            # Workflow 1: Add files and process (basic workflow)
            test_files = [Path("/tmp/test_video.mp4")]
            video_window.add_files(test_files)
            
            # Process button should be enabled
            video_window.update_submit_state()
            process_button = app.query_one("#process-button", Button)
            assert process_button.disabled == False, "Basic file processing workflow should still work"
            
            # Workflow 2: Fill metadata (common user action)
            await pilot.click("#title-input")
            await pilot.press(*"Test Video")
            await pilot.pause(0.1)
            
            title_input = app.query_one("#title-input", Input)
            assert title_input.value == "Test Video", "Metadata input workflow should still work"
            
            # Workflow 3: Configure options (advanced users)
            extract_audio = app.query_one("#extract-audio-only", Checkbox)
            initial_audio_state = extract_audio.value
            
            await pilot.click("#extract-audio-only")
            await pilot.pause(0.1)
            
            assert extract_audio.value != initial_audio_state, "Option configuration workflow should still work"
            
            # Workflow 4: Clear and start over (common user action)
            video_window.clear_files()
            assert len(video_window.form_data.get("files", [])) == 0, "Clear workflow should still work"

    @pytest.mark.asyncio
    async def test_error_messages_consistency(self):
        """Test that error messages are consistent and helpful across implementations."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            video_window = app.query_one(VideoIngestWindowRedesigned)
            
            # Test validation error messages are helpful
            title_error = video_window.validate_field("title-input", "a")
            assert title_error is not None and "characters" in title_error, \
                "Validation errors should be descriptive and helpful"
            
            # Test error state handling
            video_window.processing_status = video_window.processing_status.model_copy(
                update={"state": "error", "error_message": "Test error message"}
            )
            await pilot.pause(0.1)
            
            # Error should be displayed to user
            assert video_window.processing_status.state == "error", "Error state should be tracked"
            assert video_window.processing_status.error_message == "Test error message", \
                "Error messages should be preserved"