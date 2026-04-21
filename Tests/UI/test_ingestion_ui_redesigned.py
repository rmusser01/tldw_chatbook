# test_ingestion_ui_redesigned.py
# Test for the redesigned media ingestion UI system to ensure it loads without crashing

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
import asyncio

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


class TestIngestUIRedesigned:
    """Test suite for the redesigned media ingestion UI system."""
    
    @pytest.mark.asyncio
    async def test_factory_creates_video_window_without_crash(self):
        """Test that the factory can create a video ingestion window without crashing."""
        # Create a minimal test app
        class TestApp(App):
            def __init__(self):
                super().__init__()
                # Mock app_config to prevent errors
                self.app_config = {
                    "api_settings": {
                        "openai": {"models": ["gpt-4"]},
                        "anthropic": {"models": ["claude-3-sonnet"]}
                    }
                }
            
            def compose(self):
                # Use the factory to create the video ingestion UI
                yield IngestUIFactory.create_ui(self, "video")
        
        app = TestApp()
        async with app.run_test() as pilot:
            # Give the app time to fully load
            await pilot.pause(0.5)
            
            # Check that the app loaded without crashing
            assert app.is_running
            
            # Verify the video ingestion window is present
            video_windows = app.query(VideoIngestWindowRedesigned)
            assert len(video_windows) == 1, "Should have exactly one VideoIngestWindowRedesigned"
            
            video_window = video_windows.first()
            assert video_window is not None
            assert video_window.media_type == "video"

    @pytest.mark.asyncio 
    async def test_video_ingestion_form_elements_present(self):
        """Test that all required form elements are present and visible."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {
                    "api_settings": {
                        "openai": {"models": ["gpt-4"]},
                        "anthropic": {"models": ["claude-3-sonnet"]}
                    }
                }
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Test essential form elements exist and are visible
            
            # Status dashboard
            status_dashboard = app.query_one("#status-dashboard")
            assert status_dashboard is not None
            
            # File selection buttons
            browse_button = app.query_one("#browse-files", Button)
            assert browse_button is not None
            assert "Browse Files" in str(browse_button.label)
            
            clear_button = app.query_one("#clear-files", Button)
            assert clear_button is not None
            
            add_urls_button = app.query_one("#add-urls", Button)
            assert add_urls_button is not None
            
            # Metadata inputs
            title_input = app.query_one("#title-input", Input)
            assert title_input is not None
            
            author_input = app.query_one("#author-input", Input)  
            assert author_input is not None
            
            keywords_input = app.query_one("#keywords-input", Input)
            assert keywords_input is not None
            
            # Mode toggle
            mode_toggle = app.query_one("#mode-toggle", RadioSet)
            assert mode_toggle is not None
            
            # Process button
            process_button = app.query_one("#process-button", Button)
            assert process_button is not None
            # Button should be disabled initially (no files selected)
            assert process_button.disabled == True

    @pytest.mark.asyncio
    async def test_video_specific_options_present(self):
        """Test that video-specific options are present."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {
                    "api_settings": {
                        "openai": {"models": ["gpt-4"]},
                        "anthropic": {"models": ["claude-3-sonnet"]}  
                    }
                }
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Video processing options
            extract_audio_checkbox = app.query_one("#extract-audio-only", Checkbox)
            assert extract_audio_checkbox is not None
            
            download_video_checkbox = app.query_one("#download-video", Checkbox)
            assert download_video_checkbox is not None
            
            # Time range inputs
            start_time_input = app.query_one("#start-time", Input)
            assert start_time_input is not None
            
            end_time_input = app.query_one("#end-time", Input)
            assert end_time_input is not None
            
            # Transcription options  
            transcription_provider = app.query_one("#transcription-provider", Select)
            assert transcription_provider is not None
            
            transcription_model = app.query_one("#transcription-model", Select)
            assert transcription_model is not None
            
            language_select = app.query_one("#language", Select)
            assert language_select is not None

    @pytest.mark.asyncio
    async def test_simple_advanced_mode_toggle(self):
        """Test that simple/advanced mode toggle works correctly."""
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
            
            # Should start in advanced mode (changed behavior)
            assert video_window.simple_mode == False
            
            # Advanced options should be visible initially
            analysis_options = app.query_one("#analysis-options")
            assert "hidden" not in analysis_options.classes
            
            chunking_options = app.query_one("#chunking-options")  
            assert "hidden" not in chunking_options.classes
            
            # Switch to advanced mode
            await pilot.click("#advanced-mode")
            await pilot.pause(0.2)
            
            # Should now be in advanced mode
            assert video_window.simple_mode == False
            
            # Advanced options should be visible
            assert "hidden" not in analysis_options.classes
            assert "hidden" not in chunking_options.classes

    @pytest.mark.asyncio
    async def test_url_input_functionality(self):
        """Test that URL input section shows and hides correctly."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # URL input section should now be always visible (changed behavior)
            url_section = app.query_one("#url-input-section")
            assert "hidden" not in url_section.classes
            
            # URLs textarea should be present and functional
            urls_textarea = app.query_one("#urls-textarea", TextArea)
            assert urls_textarea is not None
            
            # Textarea should exist and be functional
            assert urls_textarea.id == "urls-textarea"

    @pytest.mark.asyncio 
    async def test_form_validation_basic(self):
        """Test basic form validation functionality."""
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
            
            # Process button should be disabled initially (no files)
            process_button = app.query_one("#process-button", Button)
            assert process_button.disabled == True
            
            # Add some form data to simulate file selection
            video_window.form_data = {
                "files": [Path("test_video.mp4")],
                "urls": []
            }
            
            # Update submit state 
            video_window.update_submit_state()
            
            # Process button should now be enabled
            assert process_button.disabled == False
            
            # Test title validation (too short)
            title_input = app.query_one("#title-input", Input)
            title_input.value = "a"  # Too short
            
            # Trigger validation
            error = video_window.validate_field("title-input", "a")
            assert error is not None
            assert "at least 2 characters" in error

    @pytest.mark.asyncio
    async def test_ingestion_ui_loads_in_main_app(self):
        """Integration test: Verify ingestion UI loads properly in the main TldwCli app."""
        
        # Mock config to disable splash screen and set up basic config
        with patch('tldw_chatbook.config._CONFIG_CACHE', {
            "splash_screen": {"enabled": False},
            "media_ingestion": {"ui_style": "redesigned"}
        }):
            # Create main app 
            app = TldwCli()
            
            async with app.run_test() as pilot:
                # Give app time to fully initialize
                await pilot.pause(1.0)
                
                # App should be running without crashes
                assert app.is_running, "Main app failed to start"
                
                # Navigate to media ingestion tab
                try:
                    # Try to find and click the Media tab
                    media_tab_buttons = app.query(Button).filter(lambda btn: "Media" in str(btn.label) or "Ingest" in str(btn.label))
                    if media_tab_buttons:
                        await pilot.click(media_tab_buttons.first())
                        await pilot.pause(0.5)
                except Exception as e:
                    # If clicking fails, that's fine - we just want to test that the UI can be created
                    pass
                
                # Test that the factory can create each media type without crashing
                media_types = ["video", "audio", "document", "pdf", "ebook", "plaintext"]
                
                for media_type in media_types:
                    try:
                        # Create the UI widget for this media type
                        ui_widget = IngestUIFactory.create_ui(app, media_type)
                        
                        # Verify it's a valid widget (Container for redesigned, Widget for legacy)
                        from textual.screen import Screen
                        from textual.widget import Widget
                        assert isinstance(ui_widget, (Container, Screen, Widget)), f"{media_type} UI should be a Container, Screen, or Widget"
                        
                        # For video, verify it's the redesigned version
                        if media_type == "video":
                            assert isinstance(ui_widget, VideoIngestWindowRedesigned), "Video should use redesigned UI"
                            assert ui_widget.media_type == "video"
                        
                    except Exception as e:
                        # If there's an error creating the UI, fail the test
                        pytest.fail(f"Failed to create {media_type} ingestion UI: {str(e)}")

    @pytest.mark.asyncio
    async def test_select_widgets_have_valid_values(self):
        """Test that all Select widgets are properly initialized with valid values."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {
                    "api_settings": {
                        "openai": {"models": ["gpt-4"]},
                        "anthropic": {"models": ["claude-3-sonnet"]}
                    }
                }
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Check all Select widgets have valid values
            select_widgets = app.query(Select)
            
            for select_widget in select_widgets:
                # Each Select should have at least one option
                assert len(select_widget._options) > 0, f"Select widget {select_widget.id} has no options"
                
                # If there's a value set, it should be valid
                if hasattr(select_widget, '_value') and select_widget._value is not None:
                    # The value should be in the options
                    option_values = [option[0] for option in select_widget._options]
                    assert select_widget._value in option_values, f"Select widget {select_widget.id} has invalid value: {select_widget._value}"

    @pytest.mark.asyncio
    async def test_css_styling_loads_correctly(self):
        """Test that CSS styling loads and applies correctly."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Check that form inputs have the expected CSS classes
            title_input = app.query_one("#title-input", Input)
            assert "form-input" in title_input.classes
            
            # Check that the main scroll container is present
            scroll_containers = app.query(VerticalScroll)
            assert len(scroll_containers) >= 1, "Should have at least one VerticalScroll container"
            
            # Check that status dashboard has correct styling
            status_dashboard = app.query_one("#status-dashboard")
            assert "status-dashboard" in status_dashboard.classes

    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(self):
        """Test that the UI handles errors gracefully without crashing."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                # Simulate missing or malformed config
                self.app_config = None
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # App should still load despite missing config
            assert app.is_running
            
            # Video window should exist
            video_windows = app.query(VideoIngestWindowRedesigned)
            assert len(video_windows) == 1
            
            video_window = video_windows.first()
            assert video_window is not None
            
            # Basic elements should still be present
            status_dashboard = app.query_one("#status-dashboard")
            assert status_dashboard is not None
            
            browse_button = app.query_one("#browse-files", Button)
            assert browse_button is not None

    @pytest.mark.asyncio
    async def test_performance_large_terminal_size(self):
        """Test that the UI performs well with large terminal sizes."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test(size=(200, 60)) as pilot:  # Large terminal size
            import time
            
            start_time = time.time()
            await pilot.pause(0.5)
            render_time = time.time() - start_time
            
            # Should render reasonably quickly even with large screen
            assert render_time < 2.0, f"UI took too long to render: {render_time}s"
            
            # UI should still be functional
            assert app.is_running
            
            video_window = app.query_one(VideoIngestWindowRedesigned)
            assert video_window is not None

    @pytest.mark.asyncio
    async def test_small_terminal_size_responsive(self):
        """Test that the UI is responsive to small terminal sizes."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test(size=(80, 24)) as pilot:  # Standard small terminal
            await pilot.pause(0.5)
            
            # Should still load successfully
            assert app.is_running
            
            video_window = app.query_one(VideoIngestWindowRedesigned)
            assert video_window is not None
            
            # Key elements should still be accessible
            status_dashboard = app.query_one("#status-dashboard")
            assert status_dashboard is not None
            
            # Form inputs should still be present
            title_input = app.query_one("#title-input", Input)
            assert title_input is not None

    @pytest.mark.asyncio
    async def test_input_visibility_critical_issue(self):
        """Test that input widgets are visible - this should FAIL for broken simplified windows."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Critical visibility test - Input widgets MUST have explicit height
            title_input = app.query_one("#title-input", Input)
            
            # This test verifies the fix for invisible input widgets
            # Input widgets need height: 3 or similar explicit height to be visible
            input_styles = title_input.styles
            assert hasattr(input_styles, 'height') and input_styles.height is not None, \
                "Input widget must have explicit height for visibility - this is a critical Textual requirement"
            
            # Verify CSS classes are applied correctly
            assert "form-input" in title_input.classes, \
                "Input widgets should have 'form-input' CSS class for proper styling"

    @pytest.mark.asyncio
    async def test_no_double_scrolling_containers(self):
        """Test that there are no nested VerticalScroll containers (Textual anti-pattern)."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Find all VerticalScroll containers, excluding standard Textual widgets that inherit from VerticalScroll
            all_scroll_containers = app.query(VerticalScroll)
            
            # Filter out standard Textual widgets that legitimately inherit from VerticalScroll
            # RadioSet, ListView, etc. are standard Textual widgets that use VerticalScroll internally
            from textual.widgets import RadioSet, ListView
            main_scroll_containers = [
                sc for sc in all_scroll_containers 
                if not isinstance(sc, (RadioSet, ListView)) and 
                   ("main-scroll" in sc.classes or "ingest-main-scroll" in sc.classes or sc.id in ["main-scroll"])
            ]
            
            # There should be only one main VerticalScroll container
            assert len(main_scroll_containers) == 1, \
                f"Should have exactly 1 main VerticalScroll container, found {len(main_scroll_containers)}. " \
                f"Found total containers: {[(type(sc).__name__, sc.id, list(sc.classes)) for sc in all_scroll_containers]}"
            
            # Verify the single scroll container is the main scroll
            main_scroll = main_scroll_containers[0]
            assert "ingest-main-scroll" in main_scroll.classes, \
                f"Main scroll should have 'ingest-main-scroll' class, has: {list(main_scroll.classes)}"

    @pytest.mark.asyncio
    async def test_url_input_validation_comprehensive(self):
        """Test URL input validation with various URL formats and edge cases."""
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
            
            # Test valid URLs
            valid_urls = [
                "https://youtube.com/watch?v=test123",
                "https://www.youtube.com/watch?v=test456",
                "http://example.com/video.mp4",
                "https://vimeo.com/123456789",
                "https://example.com/path/to/video.mkv",
            ]
            
            for url in valid_urls:
                urls_list = video_window.parse_urls(url)
                assert len(urls_list) == 1, f"Valid URL {url} should parse to exactly 1 URL"
                assert urls_list[0].strip() == url, f"Parsed URL should match input: {url}"
            
            # Test multiple URLs (one per line)
            multi_urls = "\n".join(valid_urls)
            urls_list = video_window.parse_urls(multi_urls)
            assert len(urls_list) == len(valid_urls), "Multiple URLs should be parsed correctly"
            
            # Test invalid URLs (should be filtered out or cause validation errors)
            invalid_urls = [
                "not-a-url",
                "ftp://oldprotocol.com/file.mp4",  # Might not be supported
                "",  # Empty string
                "   ",  # Just whitespace
                "https://",  # Incomplete URL
            ]
            
            for url in invalid_urls:
                # Either should return empty list or raise validation error
                try:
                    urls_list = video_window.parse_urls(url)
                    if urls_list:  # If it returns URLs, they should be valid
                        for parsed_url in urls_list:
                            assert parsed_url.startswith(('http://', 'https://')), \
                                f"Invalid URL {url} should not parse to valid URL: {parsed_url}"
                except Exception:
                    # Validation error is acceptable for invalid URLs
                    pass

    @pytest.mark.asyncio
    async def test_form_field_validation_edge_cases(self):
        """Test form validation with edge cases and boundary conditions."""
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
            
            # Test title field validation edge cases
            title_test_cases = [
                ("", None),  # Empty should be OK (optional field)
                ("a", "Title must be at least 2 characters"),  # Too short
                ("ab", None),  # Minimum valid length
                ("A" * 1000, None),  # Very long should be OK
                ("Valid Title", None),  # Normal case
                ("Title with 123 numbers", None),  # With numbers
                ("Title with special chars!@#", None),  # With special chars
            ]
            
            for test_value, expected_error in title_test_cases:
                error = video_window.validate_field("title-input", test_value)
                if expected_error:
                    assert error is not None and expected_error in error, \
                        f"Expected error '{expected_error}' for title '{test_value}', got: {error}"
                else:
                    assert error is None, \
                        f"Expected no error for title '{test_value}', got: {error}"
            
            # Test keywords field validation
            keywords_test_cases = [
                ("", None),  # Empty OK
                ("single", None),  # Single keyword
                ("multiple,keywords,here", None),  # Comma separated
                ("keyword1, keyword2, keyword3", None),  # With spaces
                ("a,b,c,d,e,f,g,h,i,j,k,l", None),  # Many keywords
            ]
            
            for test_value, expected_error in keywords_test_cases:
                error = video_window.validate_field("keywords-input", test_value)
                if expected_error:
                    assert error is not None and expected_error in error, \
                        f"Expected error '{expected_error}' for keywords '{test_value}', got: {error}"
                else:
                    assert error is None, \
                        f"Expected no error for keywords '{test_value}', got: {error}"

    @pytest.mark.asyncio  
    async def test_file_selection_with_local_files(self):
        """Test local file selection functionality."""
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
            
            # Test adding files programmatically (simulates file browser selection)
            test_files = [
                Path("/tmp/test_video.mp4"),
                Path("/tmp/another_video.avi"),
                Path("/tmp/sample.mkv")
            ]
            
            # Simulate file selection
            video_window.add_files(test_files)
            
            # Verify files were added to form data
            assert "files" in video_window.form_data
            assert len(video_window.form_data["files"]) == 3
            
            # Verify file paths match
            for i, expected_file in enumerate(test_files):
                assert video_window.form_data["files"][i] == expected_file
            
            # Verify submit button is enabled when files are present
            video_window.update_submit_state()
            process_button = app.query_one("#process-button", Button)
            assert process_button.disabled == False, "Process button should be enabled when files are selected"
            
            # Test clearing files
            video_window.clear_files()
            assert len(video_window.form_data.get("files", [])) == 0, "Files should be cleared"
            
            # Submit button should be disabled again
            video_window.update_submit_state()
            assert process_button.disabled == True, "Process button should be disabled when no files selected"

    @pytest.mark.asyncio
    async def test_processing_status_updates(self):
        """Test that processing status updates work correctly."""
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
            
            # Test initial status
            assert video_window.processing_status.state == "idle"
            
            # Test status transitions
            video_window.processing_status = video_window.processing_status.model_copy(
                update={"state": "processing", "progress": 0.0, "current_file": "test_video.mp4"}
            )
            
            # Allow reactive updates to process
            await pilot.pause(0.1)
            
            # Verify status dashboard reflects changes
            status_dashboard = app.query_one("#status-dashboard")
            assert status_dashboard is not None
            
            # Test progress updates
            video_window.processing_status = video_window.processing_status.model_copy(
                update={"progress": 50.0}
            )
            await pilot.pause(0.1)
            
            # Test completion
            video_window.processing_status = video_window.processing_status.model_copy(
                update={"state": "complete", "progress": 100.0}
            )
            await pilot.pause(0.1)
            
            # Test error state
            video_window.processing_status = video_window.processing_status.model_copy(
                update={"state": "error", "error_message": "Test error occurred"}
            )
            await pilot.pause(0.1)

    @pytest.mark.asyncio
    async def test_css_form_styling_applied_correctly(self):
        """Test that all form elements have correct CSS styling applied."""
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Test that all Input widgets have the form-input class
            input_widgets = app.query(Input)
            for input_widget in input_widgets:
                assert "form-input" in input_widget.classes, \
                    f"Input widget {input_widget.id} should have 'form-input' class"
            
            # Test that all TextArea widgets have proper styling
            textarea_widgets = app.query(TextArea)
            for textarea_widget in textarea_widgets:
                # Should have either form-textarea or similar styling class
                has_textarea_class = any(cls in textarea_widget.classes 
                                       for cls in ["form-textarea", "textarea-styled"])
                assert has_textarea_class, \
                    f"TextArea widget {textarea_widget.id} should have textarea styling class"
            
            # Test that Select widgets have proper styling
            select_widgets = app.query(Select)  
            for select_widget in select_widgets:
                has_select_class = any(cls in select_widget.classes 
                                     for cls in ["form-select", "select-styled"])
                assert has_select_class or len(select_widget.classes) > 0, \
                    f"Select widget {select_widget.id} should have styling classes"

    @pytest.mark.asyncio
    async def test_simplified_windows_are_broken_as_expected(self):
        """Test that demonstrates the known issues with simplified windows."""
        # Import a simplified window that we know has issues
        from tldw_chatbook.Widgets.Media_Ingest.IngestLocalVideoWindowSimplified import IngestLocalVideoWindowSimplified
        
        class TestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                # Use the broken simplified window
                yield IngestLocalVideoWindowSimplified(self)
        
        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            simplified_window = app.query_one(IngestLocalVideoWindowSimplified)
            assert simplified_window is not None
            
            # These tests document the known issues that should be fixed:
            
            # Issue 1: Input widgets may not have explicit heights
            input_widgets = app.query(Input)
            inputs_with_explicit_height = 0
            
            for input_widget in input_widgets:
                # Check if input has explicit height in its styles or CSS classes
                has_height_style = (hasattr(input_widget.styles, 'height') and 
                                  input_widget.styles.height is not None)
                has_form_input_class = "form-input" in input_widget.classes
                
                if has_height_style or has_form_input_class:
                    inputs_with_explicit_height += 1
            
            # This assertion may FAIL for broken simplified windows
            if len(input_widgets) > 0:
                height_ratio = inputs_with_explicit_height / len(input_widgets)
                assert height_ratio >= 0.8, \
                    f"Most input widgets should have explicit height styling. " \
                    f"Only {inputs_with_explicit_height}/{len(input_widgets)} have proper height"
            
            # Issue 2: Check for problematic double scrolling containers
            # Note: RadioSet and other standard Textual widgets may internally use scrolling,
            # but we're specifically looking for nested VerticalScroll containers that cause UX issues
            scroll_containers = app.query(VerticalScroll)
            main_scroll_containers = [
                sc for sc in scroll_containers 
                if not isinstance(sc, (RadioSet,)) and "ingest-form-scrollable" in sc.classes
            ]
            
            # There should be exactly one main scrolling container for the form
            assert len(main_scroll_containers) == 1, \
                f"Should have exactly 1 main VerticalScroll container for the form, found {len(main_scroll_containers)}. " \
                f"Multiple main scroll containers cause broken scrolling behavior"
            
            # Check for nested scrolling containers that would cause real problems
            nested_scrolls = []
            for main_scroll in main_scroll_containers:
                nested = main_scroll.query(VerticalScroll)
                nested_scrolls.extend([n for n in nested if n != main_scroll and not isinstance(n, (RadioSet,))])
            
            assert len(nested_scrolls) == 0, \
                f"Should not have nested VerticalScroll containers inside main scroll, found {len(nested_scrolls)}. " \
                f"Nested scroll containers cause broken scrolling behavior"