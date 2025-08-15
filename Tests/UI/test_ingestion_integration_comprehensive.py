# test_ingestion_integration_comprehensive.py
"""
Comprehensive integration tests for the media ingestion UI system.

This test suite focuses on:
1. Factory pattern integration across all media types
2. Cross-platform compatibility (different terminal sizes)
3. Complete user workflow testing 
4. Regression testing between legacy and redesigned implementations

These tests are designed to catch Textual best practice violations and ensure
the ingestion UI works correctly across different configurations and scenarios.
"""

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
import asyncio
from typing import List, Dict, Any

# Third-party Libraries
from textual.app import App
from textual.widgets import Button, Input, Select, Checkbox, TextArea, RadioSet, RadioButton, Label, Static
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.pilot import Pilot
from textual.css.query import NoMatches

# Local Imports
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Media_Ingest.IngestUIFactory import IngestUIFactory
from tldw_chatbook.Widgets.Media_Ingest.Ingest_Local_Video_Window import VideoIngestWindowRedesigned
from tldw_chatbook.Widgets.Media_Ingest.base_media_ingest_window import BaseMediaIngestWindow


class TestFactoryPatternIntegration:
    """Test the factory pattern creates appropriate UIs for all media types."""

    @pytest.mark.asyncio
    async def test_factory_creates_all_media_types(self):
        """Test that factory can create UI for every supported media type."""
        # Test both with main app context and minimal test app
        app = TldwCli()
        
        # List all media types that should be supported
        media_types = ["video", "audio", "document", "pdf", "ebook", "plaintext", "xml", "mediawiki"]
        
        for media_type in media_types:
            try:
                ui_widget = IngestUIFactory.create_ui(app, media_type)
                
                # Verify it's a valid Textual widget
                from textual.widget import Widget
                assert isinstance(ui_widget, Widget), \
                    f"Factory should return Widget for {media_type}, got {type(ui_widget)}"
                
                # For redesigned media types, verify correct inheritance
                if media_type in ["video", "audio"]:  # Currently redesigned
                    assert isinstance(ui_widget, BaseMediaIngestWindow), \
                        f"Redesigned {media_type} UI should inherit from BaseMediaIngestWindow"
                
            except Exception as e:
                pytest.fail(f"Factory failed to create {media_type} UI: {str(e)}")

    @pytest.mark.asyncio
    async def test_factory_ui_style_selection(self):
        """Test that factory respects UI style configuration."""
        app = TldwCli()
        
        # Test different UI styles
        ui_styles = ["default", "redesigned", "new", "grid", "wizard", "split"]
        
        for ui_style in ui_styles:
            # Mock the configuration to return our test style
            with patch('tldw_chatbook.config.get_ingest_ui_style', return_value=ui_style):
                try:
                    # Test with video (most likely to have redesigned version)
                    video_ui = IngestUIFactory.create_ui(app, "video")
                    assert video_ui is not None
                    
                    # For redesigned styles, should get the redesigned implementation
                    if ui_style in ["redesigned", "new", "default"]:
                        assert isinstance(video_ui, VideoIngestWindowRedesigned), \
                            f"UI style '{ui_style}' should return VideoIngestWindowRedesigned for video"
                    
                except Exception as e:
                    pytest.fail(f"Factory failed with UI style '{ui_style}': {str(e)}")

    @pytest.mark.asyncio
    async def test_factory_graceful_fallback(self):
        """Test that factory falls back gracefully when redesigned UI not available."""
        app = TldwCli()
        
        # Test with media types that likely don't have redesigned implementations yet
        legacy_media_types = ["document", "pdf", "ebook", "plaintext"]
        
        with patch('tldw_chatbook.config.get_ingest_ui_style', return_value="redesigned"):
            for media_type in legacy_media_types:
                try:
                    ui_widget = IngestUIFactory.create_ui(app, media_type)
                    
                    # Should get a valid widget (legacy implementation)
                    from textual.widget import Widget
                    assert isinstance(ui_widget, Widget), \
                        f"Factory should return valid Widget for {media_type} (legacy fallback)"
                    
                    # Should not be None or raise exception
                    assert ui_widget is not None, \
                        f"Factory should not return None for {media_type} fallback"
                    
                except Exception as e:
                    pytest.fail(f"Factory fallback failed for {media_type}: {str(e)}")

    @pytest.mark.asyncio
    async def test_factory_available_styles_consistency(self):
        """Test that get_available_styles returns valid style names."""
        available_styles = IngestUIFactory.get_available_styles()
        
        # Should return a non-empty list
        assert len(available_styles) > 0, "Factory should return available styles"
        
        # Each style should have a description
        for style in available_styles:
            description = IngestUIFactory.get_style_description(style)
            assert description and description != "Unknown UI style", \
                f"Style '{style}' should have a valid description, got: '{description}'"
        
        # Test that each available style actually works
        app = TldwCli()
        for style in available_styles:
            with patch('tldw_chatbook.config.get_ingest_ui_style', return_value=style):
                try:
                    ui_widget = IngestUIFactory.create_ui(app, "video")
                    assert ui_widget is not None, f"Style '{style}' should create valid UI"
                except Exception as e:
                    pytest.fail(f"Available style '{style}' failed to create UI: {str(e)}")

    @pytest.mark.asyncio  
    async def test_factory_error_handling(self):
        """Test factory error handling with invalid configurations."""
        app = TldwCli()
        
        # Test with invalid UI style
        with patch('tldw_chatbook.config.get_ingest_ui_style', return_value="nonexistent_style"):
            try:
                ui_widget = IngestUIFactory.create_ui(app, "video")
                # Should fallback gracefully, not crash
                assert ui_widget is not None, "Factory should handle invalid UI style gracefully"
            except Exception as e:
                pytest.fail(f"Factory should not crash with invalid UI style: {str(e)}")
        
        # Test with invalid media type
        try:
            ui_widget = IngestUIFactory.create_ui(app, "invalid_media_type")
            # Should either return a valid fallback or handle gracefully
            from textual.widget import Widget
            assert isinstance(ui_widget, Widget) or ui_widget is None, \
                "Factory should handle invalid media type gracefully"
        except Exception as e:
            pytest.fail(f"Factory should not crash with invalid media type: {str(e)}")


class TestCrossPlatformCompatibility:
    """Test UI compatibility across different terminal sizes and platforms."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal_size", [
        (80, 24),    # Standard small terminal
        (120, 40),   # Medium terminal  
        (200, 60),   # Large terminal
        (60, 20),    # Very small terminal
        (300, 80),   # Very large terminal
    ])
    async def test_ui_responsive_design(self, terminal_size):
        """Test that UI adapts correctly to different terminal sizes."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        width, height = terminal_size
        app = TestApp()
        
        async with app.run_test(size=(width, height)) as pilot:
            await pilot.pause(0.5)
            
            # UI should load successfully at any size
            assert app.is_running, f"App should load at terminal size {width}x{height}"
            
            video_window = app.query_one(VideoIngestWindowRedesigned)
            assert video_window is not None, f"Video window should exist at size {width}x{height}"
            
            # Essential elements should be present regardless of size
            status_dashboard = app.query_one("#status-dashboard")
            assert status_dashboard is not None, "Status dashboard should exist at any size"
            
            browse_button = app.query_one("#browse-files", Button)
            assert browse_button is not None, "Browse button should exist at any size"
            
            # Form inputs should be present
            title_input = app.query_one("#title-input", Input)
            assert title_input is not None, "Title input should exist at any size"
            
            # For very small terminals, ensure no horizontal overflow
            if width <= 60:
                # Elements should fit within terminal width
                # (This is a basic check - more sophisticated responsive testing could be added)
                pass

    @pytest.mark.asyncio
    async def test_performance_at_different_sizes(self):
        """Test UI performance doesn't degrade significantly at different terminal sizes."""
        import time
        
        test_sizes = [(80, 24), (200, 60), (300, 80)]
        render_times = []
        
        for width, height in test_sizes:
            class TestApp(App):
                def __init__(self):
                    super().__init__()
                    self.app_config = {"api_settings": {}}
                
                def compose(self):
                    yield VideoIngestWindowRedesigned(self)
            
            app = TestApp()
            
            start_time = time.time()
            async with app.run_test(size=(width, height)) as pilot:
                await pilot.pause(0.5)
            render_time = time.time() - start_time
            render_times.append((f"{width}x{height}", render_time))
        
        # All sizes should render reasonably quickly
        for size_desc, render_time in render_times:
            assert render_time < 3.0, f"Rendering at {size_desc} took too long: {render_time:.2f}s"
        
        # Performance shouldn't degrade drastically with size
        if len(render_times) >= 2:
            min_time = min(t[1] for t in render_times)
            max_time = max(t[1] for t in render_times)
            # Max time shouldn't be more than 3x min time
            assert max_time <= min_time * 3, \
                f"Performance varies too much across sizes: {render_times}"

    @pytest.mark.asyncio
    async def test_scrolling_behavior_different_sizes(self):
        """Test that scrolling works correctly at different terminal sizes."""
        test_sizes = [(80, 20), (120, 30), (200, 50)]  # Heights chosen to force scrolling
        
        for width, height in test_sizes:
            class TestApp(App):
                def __init__(self):
                    super().__init__()
                    self.app_config = {"api_settings": {}}
                
                def compose(self):
                    yield VideoIngestWindowRedesigned(self)
            
            app = TestApp()
            
            async with app.run_test(size=(width, height)) as pilot:
                await pilot.pause(0.5)
                
                # Verify only one scroll container
                scroll_containers = app.query(VerticalScroll)
                assert len(scroll_containers) == 1, \
                    f"Should have exactly 1 scroll container at size {width}x{height}, found {len(scroll_containers)}"
                
                # Test scrolling functionality
                main_scroll = scroll_containers.first()
                initial_scroll_y = main_scroll.scroll_y
                
                # Try to scroll down
                await pilot.press("j", "j", "j")  # Scroll down
                await pilot.pause(0.1)
                
                # Should be able to scroll (might not move if content fits, but shouldn't crash)
                assert main_scroll.scroll_y >= initial_scroll_y, "Should handle scroll input gracefully"


class TestUserWorkflowIntegration:
    """Test complete user workflows from start to finish."""

    @pytest.mark.asyncio
    async def test_complete_video_ingestion_workflow(self):
        """Test complete workflow: file selection → metadata entry → validation → submit ready."""
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
            
            # Step 1: Start with simple mode (default)
            assert video_window.simple_mode == True, "Should start in simple mode"
            
            # Step 2: Add files (simulate file selection)
            test_files = [Path("/tmp/test_video.mp4"), Path("/tmp/another.avi")]
            video_window.add_files(test_files)
            
            # Step 3: Fill in metadata fields
            await pilot.click("#title-input")
            await pilot.press(*"Test Video Title")
            await pilot.pause(0.1)
            
            await pilot.click("#author-input")
            await pilot.press(*"Test Author")
            await pilot.pause(0.1)
            
            await pilot.click("#keywords-input")
            await pilot.press(*"test, video, ingestion")
            await pilot.pause(0.1)
            
            # Step 4: Switch to advanced mode
            await pilot.click("#advanced-mode")
            await pilot.pause(0.2)
            
            assert video_window.simple_mode == False, "Should switch to advanced mode"
            
            # Step 5: Configure advanced options
            extract_audio_checkbox = app.query_one("#extract-audio-only", Checkbox)
            if not extract_audio_checkbox.value:
                await pilot.click("#extract-audio-only")
                await pilot.pause(0.1)
            
            # Step 6: Validate form is ready for submission
            video_window.update_submit_state()
            process_button = app.query_one("#process-button", Button)
            assert process_button.disabled == False, "Process button should be enabled with valid form"
            
            # Step 7: Verify form data is properly collected
            form_data = video_window.get_form_data()
            assert "files" in form_data and len(form_data["files"]) == 2, "Files should be in form data"
            assert "title" in form_data, "Title should be in form data"
            assert "author" in form_data, "Author should be in form data"
            assert "keywords" in form_data, "Keywords should be in form data"

    @pytest.mark.asyncio
    async def test_url_ingestion_workflow(self):
        """Test workflow for URL-based ingestion."""
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
            
            # Step 1: Show URL input section
            url_section = app.query_one("#url-input-section")
            assert "hidden" in url_section.classes, "URL section should be hidden initially"
            
            await pilot.click("#add-urls")
            await pilot.pause(0.2)
            
            assert "hidden" not in url_section.classes, "URL section should be visible after clicking Add URLs"
            
            # Step 2: Enter URLs
            urls_textarea = app.query_one("#urls-textarea", TextArea)
            assert urls_textarea is not None
            
            test_urls = [
                "https://youtube.com/watch?v=test123",
                "https://example.com/video.mp4",
                "https://vimeo.com/123456789"
            ]
            urls_text = "\n".join(test_urls)
            
            await pilot.click("#urls-textarea")
            await pilot.press(*urls_text)
            await pilot.pause(0.1)
            
            # Step 3: Process URLs
            await pilot.click("#process-urls")
            await pilot.pause(0.2)
            
            # Step 4: Verify URLs are added to form data
            form_data = video_window.get_form_data()
            assert "urls" in form_data and len(form_data["urls"]) == len(test_urls), \
                "URLs should be processed and added to form data"
            
            # Step 5: Verify submit button is enabled
            video_window.update_submit_state()
            process_button = app.query_one("#process-button", Button)
            assert process_button.disabled == False, "Process button should be enabled with valid URLs"

    @pytest.mark.asyncio
    async def test_form_validation_workflow(self):
        """Test that form validation provides proper user feedback throughout workflow."""
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
            
            # Step 1: Process button should be disabled initially (no files/URLs)
            process_button = app.query_one("#process-button", Button)
            assert process_button.disabled == True, "Process button should start disabled"
            
            # Step 2: Test title validation
            await pilot.click("#title-input")
            await pilot.press("a")  # Too short
            await pilot.pause(0.1)
            
            # Should trigger validation error
            title_input = app.query_one("#title-input", Input)
            error = video_window.validate_field("title-input", "a")
            assert error is not None and "at least 2 characters" in error, \
                "Should show validation error for short title"
            
            # Step 3: Fix validation error
            await pilot.press("ctrl+a")  # Select all
            await pilot.press(*"Valid Title")
            await pilot.pause(0.1)
            
            error = video_window.validate_field("title-input", "Valid Title")
            assert error is None, "Should clear validation error with valid title"
            
            # Step 4: Add files to enable submit
            test_files = [Path("/tmp/test_video.mp4")]
            video_window.add_files(test_files)
            video_window.update_submit_state()
            
            assert process_button.disabled == False, "Process button should be enabled with valid form"
            
            # Step 5: Test clearing files disables submit
            video_window.clear_files()
            video_window.update_submit_state()
            
            assert process_button.disabled == True, "Process button should be disabled after clearing files"

    @pytest.mark.asyncio
    async def test_mode_switching_preserves_data(self):
        """Test that switching between simple/advanced mode preserves user data."""
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
            
            # Step 1: Fill out form in simple mode
            await pilot.click("#title-input")
            await pilot.press(*"Test Title")
            await pilot.pause(0.1)
            
            await pilot.click("#keywords-input")
            await pilot.press(*"test, keywords")
            await pilot.pause(0.1)
            
            # Step 2: Switch to advanced mode
            await pilot.click("#advanced-mode")
            await pilot.pause(0.2)
            
            # Step 3: Verify data is preserved
            title_input = app.query_one("#title-input", Input)
            keywords_input = app.query_one("#keywords-input", Input)
            
            assert title_input.value == "Test Title", "Title should be preserved when switching modes"
            assert keywords_input.value == "test, keywords", "Keywords should be preserved when switching modes"
            
            # Step 4: Fill advanced options
            enable_analysis = app.query_one("#enable-analysis", Checkbox)
            if not enable_analysis.value:
                await pilot.click("#enable-analysis")
                await pilot.pause(0.1)
            
            # Step 5: Switch back to simple mode
            await pilot.click("#simple-mode")
            await pilot.pause(0.2)
            
            # Step 6: Verify all data is still preserved
            assert title_input.value == "Test Title", "Title should be preserved when switching back to simple"
            assert keywords_input.value == "test, keywords", "Keywords should be preserved when switching back"
            
            # Advanced settings should be remembered even if hidden
            form_data = video_window.get_form_data()
            assert form_data.get("enable_analysis", False) == True, "Advanced settings should be preserved"

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test that users can recover from errors during the workflow."""
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
            
            # Step 1: Simulate processing error
            video_window.processing_status = video_window.processing_status.model_copy(
                update={"state": "error", "error_message": "Test processing error"}
            )
            await pilot.pause(0.1)
            
            # Step 2: Verify error state is shown
            status_dashboard = app.query_one("#status-dashboard")
            assert status_dashboard is not None
            
            # Step 3: User should be able to retry
            process_button = app.query_one("#process-button", Button)
            # Add files to make retry possible
            test_files = [Path("/tmp/test_video.mp4")]
            video_window.add_files(test_files)
            video_window.update_submit_state()
            
            assert process_button.disabled == False, "Should be able to retry after error"
            
            # Step 4: Clear error state (simulate retry)
            video_window.processing_status = video_window.processing_status.model_copy(
                update={"state": "idle", "error_message": ""}
            )
            await pilot.pause(0.1)
            
            # Form should be usable again
            assert video_window.processing_status.state == "idle", "Should return to idle state after error recovery"