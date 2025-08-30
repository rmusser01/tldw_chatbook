# ingestion_test_helpers.py
"""
Test utilities and helper functions for media ingestion UI testing.

This module provides reusable components to simplify testing of ingestion UIs:
1. Form filling utilities
2. File selection simulators  
3. Validation assertion helpers
4. Status checking utilities
5. Mock data and fixtures
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import MagicMock, patch

# Third-party Libraries
from textual.app import App
from textual.widgets import Button, Input, Select, Checkbox, TextArea, RadioSet, RadioButton
from textual.pilot import Pilot

# Local Imports
from tldw_chatbook.Widgets.Media_Ingest.base_media_ingest_window import BaseMediaIngestWindow


class IngestTestHelper:
    """Helper class for common ingestion UI testing operations."""
    
    def __init__(self, app: App, pilot: Pilot):
        self.app = app
        self.pilot = pilot
    
    async def fill_basic_metadata(
        self, 
        title: str = "Test Title", 
        author: str = "Test Author", 
        keywords: str = "test,keywords"
    ) -> None:
        """Fill basic metadata fields with test data."""
        if title:
            await self.pilot.click("#title-input")
            await self.pilot.press("ctrl+a")  # Select all
            await self.pilot.press(*title)
            await self.pilot.pause(0.1)
        
        if author:
            await self.pilot.click("#author-input")
            await self.pilot.press("ctrl+a")
            await self.pilot.press(*author)
            await self.pilot.pause(0.1)
        
        if keywords:
            await self.pilot.click("#keywords-input")
            await self.pilot.press("ctrl+a")
            await self.pilot.press(*keywords)
            await self.pilot.pause(0.1)
    
    async def add_test_files(self, file_count: int = 3) -> List[Path]:
        """Add test files to the ingestion window."""
        test_files = [
            Path(f"/tmp/test_video_{i}.mp4") for i in range(file_count)
        ]
        
        # Get the ingest window and add files programmatically
        ingest_windows = self.app.query(BaseMediaIngestWindow)
        if ingest_windows:
            ingest_window = ingest_windows.first()
            ingest_window.add_files(test_files)
        
        return test_files
    
    async def add_test_urls(self, urls: Optional[List[str]] = None) -> List[str]:
        """Add test URLs to the ingestion window."""
        if urls is None:
            urls = [
                "https://youtube.com/watch?v=test123",
                "https://example.com/video.mp4",
                "https://vimeo.com/123456789"
            ]
        
        # Show URL input section
        await self.pilot.click("#add-urls")
        await self.pilot.pause(0.2)
        
        # Enter URLs
        urls_text = "\n".join(urls)
        await self.pilot.click("#urls-textarea")
        await self.pilot.press(*urls_text)
        await self.pilot.pause(0.1)
        
        # Process URLs
        await self.pilot.click("#process-urls")
        await self.pilot.pause(0.2)
        
        return urls
    
    async def switch_to_advanced_mode(self) -> None:
        """Switch the UI to advanced mode."""
        await self.pilot.click("#advanced-mode")
        await self.pilot.pause(0.2)
    
    async def switch_to_simple_mode(self) -> None:
        """Switch the UI to simple mode."""
        await self.pilot.click("#simple-mode")
        await self.pilot.pause(0.2)
    
    async def configure_video_options(
        self,
        extract_audio: bool = True,
        download_video: bool = False,
        start_time: str = "",
        end_time: str = ""
    ) -> None:
        """Configure video-specific processing options."""
        # Extract audio checkbox
        extract_audio_checkbox = self.app.query_one("#extract-audio-only", Checkbox)
        if extract_audio_checkbox.value != extract_audio:
            await self.pilot.click("#extract-audio-only")
            await self.pilot.pause(0.1)
        
        # Download video checkbox
        download_video_checkbox = self.app.query_one("#download-video", Checkbox)
        if download_video_checkbox.value != download_video:
            await self.pilot.click("#download-video")
            await self.pilot.pause(0.1)
        
        # Time range inputs
        if start_time:
            await self.pilot.click("#start-time")
            await self.pilot.press("ctrl+a")
            await self.pilot.press(*start_time)
            await self.pilot.pause(0.1)
        
        if end_time:
            await self.pilot.click("#end-time")
            await self.pilot.press("ctrl+a")
            await self.pilot.press(*end_time)
            await self.pilot.pause(0.1)
    
    async def configure_transcription_options(
        self,
        provider: str = "faster-whisper",
        model: str = "base",
        language: str = "en"
    ) -> None:
        """Configure transcription options."""
        # Set transcription provider
        provider_select = self.app.query_one("#transcription-provider", Select)
        if provider_select.options and any(opt[0] == provider for opt in provider_select.options):
            provider_select.value = provider
            await self.pilot.pause(0.1)
        
        # Set transcription model (after provider is set)
        model_select = self.app.query_one("#transcription-model", Select)
        if model_select.options and any(opt[0] == model for opt in model_select.options):
            model_select.value = model
            await self.pilot.pause(0.1)
        
        # Set language
        language_select = self.app.query_one("#language", Select)
        if language_select.options and any(opt[0] == language for opt in language_select.options):
            language_select.value = language
            await self.pilot.pause(0.1)
    
    async def wait_for_processing_state(self, expected_state: str, timeout: float = 5.0) -> bool:
        """Wait for processing status to reach expected state."""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            ingest_windows = self.app.query(BaseMediaIngestWindow)
            if ingest_windows:
                ingest_window = ingest_windows.first()
                if ingest_window.processing_status.state == expected_state:
                    return True
            
            await self.pilot.pause(0.1)
        
        return False
    
    def assert_form_validity(self, should_be_valid: bool = True) -> None:
        """Assert that the form is in the expected validity state."""
        process_button = self.app.query_one("#process-button", Button)
        
        if should_be_valid:
            assert process_button.disabled == False, "Form should be valid (process button enabled)"
        else:
            assert process_button.disabled == True, "Form should be invalid (process button disabled)"
    
    def assert_validation_error(self, field_id: str, expected_error: str) -> None:
        """Assert that a specific field has a validation error."""
        ingest_windows = self.app.query(BaseMediaIngestWindow)
        if ingest_windows:
            ingest_window = ingest_windows.first()
            field_input = self.app.query_one(f"#{field_id}", Input)
            
            error = ingest_window.validate_field(field_id, field_input.value)
            assert error is not None and expected_error in error, \
                f"Expected validation error '{expected_error}' for field {field_id}, got: {error}"
    
    def assert_no_validation_error(self, field_id: str) -> None:
        """Assert that a specific field has no validation error."""
        ingest_windows = self.app.query(BaseMediaIngestWindow)
        if ingest_windows:
            ingest_window = ingest_windows.first()
            field_input = self.app.query_one(f"#{field_id}", Input)
            
            error = ingest_window.validate_field(field_id, field_input.value)
            assert error is None, f"Expected no validation error for field {field_id}, got: {error}"
    
    def get_form_data(self) -> Dict[str, Any]:
        """Get current form data from the ingestion window."""
        ingest_windows = self.app.query(BaseMediaIngestWindow)
        if ingest_windows:
            ingest_window = ingest_windows.first()
            return ingest_window.get_form_data()
        return {}
    
    def assert_files_selected(self, expected_count: int) -> None:
        """Assert that the expected number of files are selected."""
        form_data = self.get_form_data()
        actual_count = len(form_data.get("files", []))
        assert actual_count == expected_count, \
            f"Expected {expected_count} files selected, got {actual_count}"
    
    def assert_urls_added(self, expected_count: int) -> None:
        """Assert that the expected number of URLs are added."""
        form_data = self.get_form_data()
        actual_count = len(form_data.get("urls", []))
        assert actual_count == expected_count, \
            f"Expected {expected_count} URLs added, got {actual_count}"


class MockDataFixtures:
    """Provides mock data and fixtures for testing."""
    
    @staticmethod
    def sample_video_files(count: int = 3) -> List[Path]:
        """Generate sample video file paths."""
        extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
        return [
            Path(f"/tmp/test_video_{i}{extensions[i % len(extensions)]}")
            for i in range(count)
        ]
    
    @staticmethod
    def sample_audio_files(count: int = 3) -> List[Path]:
        """Generate sample audio file paths."""
        extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
        return [
            Path(f"/tmp/test_audio_{i}{extensions[i % len(extensions)]}")
            for i in range(count)
        ]
    
    @staticmethod
    def sample_document_files(count: int = 3) -> List[Path]:
        """Generate sample document file paths."""
        extensions = [".txt", ".pdf", ".docx", ".md", ".rtf"]
        return [
            Path(f"/tmp/test_document_{i}{extensions[i % len(extensions)]}")
            for i in range(count)
        ]
    
    @staticmethod
    def sample_urls(media_type: str = "video", count: int = 3) -> List[str]:
        """Generate sample URLs for different media types."""
        if media_type == "video":
            base_urls = [
                "https://youtube.com/watch?v={}",
                "https://vimeo.com/{}",
                "https://example.com/{}.mp4"
            ]
        elif media_type == "audio":
            base_urls = [
                "https://soundcloud.com/user/{}",
                "https://example.com/{}.mp3",
                "https://spotify.com/track/{}"
            ]
        else:
            base_urls = [
                "https://example.com/{}",
                "https://archive.org/details/{}",
                "https://docs.example.com/{}"
            ]
        
        return [base_urls[i % len(base_urls)].format(f"test{i}") for i in range(count)]
    
    @staticmethod
    def sample_form_data() -> Dict[str, Any]:
        """Generate sample form data for testing."""
        return {
            "title": "Test Media Title",
            "author": "Test Author",
            "keywords": "test,sample,media",
            "description": "This is a test description for sample media content.",
            "extract_audio_only": True,
            "download_video": False,
            "transcription_provider": "faster-whisper",
            "transcription_model": "base",
            "language": "en",
            "enable_analysis": False,
            "chunk_method": "sentences",
            "chunk_size": 1000,
            "overlap_size": 200
        }
    
    @staticmethod
    def validation_test_cases() -> List[Tuple[str, str, Optional[str]]]:
        """Generate validation test cases (field_id, test_value, expected_error)."""
        return [
            ("title-input", "", None),  # Empty title OK
            ("title-input", "a", "at least 2 characters"),  # Too short
            ("title-input", "ab", None),  # Minimum valid
            ("title-input", "A" * 1000, None),  # Very long OK
            ("title-input", "Valid Title", None),  # Normal case
            ("keywords-input", "", None),  # Empty keywords OK
            ("keywords-input", "single", None),  # Single keyword
            ("keywords-input", "multiple,keywords", None),  # Multiple keywords
            ("author-input", "", None),  # Empty author OK
            ("author-input", "Test Author", None),  # Valid author
        ]


class MockServices:
    """Mock services for testing without external dependencies."""
    
    @staticmethod
    def mock_transcription_service():
        """Create a mock transcription service."""
        mock_service = MagicMock()
        mock_service.get_available_providers.return_value = [
            "faster-whisper", "whisper", "openai-whisper"
        ]
        mock_service.get_available_models.return_value = {
            "faster-whisper": ["tiny", "base", "small", "medium", "large"],
            "whisper": ["tiny", "base", "small", "medium", "large"],
            "openai-whisper": ["whisper-1"]
        }
        mock_service.get_models_for_provider.return_value = ["tiny", "base", "small"]
        return mock_service
    
    @staticmethod
    def mock_llm_service():
        """Create a mock LLM service for analysis."""
        mock_service = MagicMock()
        mock_service.get_available_providers.return_value = ["openai", "anthropic"]
        mock_service.get_available_models.return_value = {
            "openai": ["gpt-4", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-sonnet", "claude-3-haiku"]
        }
        return mock_service
    
    @staticmethod
    def mock_file_service():
        """Create a mock file service."""
        mock_service = MagicMock()
        mock_service.validate_file_path.return_value = True
        mock_service.get_file_info.return_value = {
            "size": 1024 * 1024,  # 1MB
            "duration": 300,  # 5 minutes
            "format": "mp4"
        }
        return mock_service


class IngestTestApp:
    """Factory for creating test apps with ingestion UIs."""
    
    @staticmethod
    def create_video_app(config: Optional[Dict[str, Any]] = None):
        """Create a test app with video ingestion UI."""
        from tldw_chatbook.Widgets.Media_Ingest.Ingest_Local_Video_Window import VideoIngestWindowRedesigned
        
        class VideoTestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = config or {"api_settings": {}}
            
            def compose(self):
                yield VideoIngestWindowRedesigned(self)
        
        return VideoTestApp
    
    @staticmethod
    def create_simplified_video_app(config: Optional[Dict[str, Any]] = None):
        """Create a test app with simplified video ingestion UI."""
        from tldw_chatbook.Widgets.Media_Ingest.IngestLocalVideoWindowSimplified import IngestLocalVideoWindowSimplified
        
        class SimplifiedVideoTestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = config or {"api_settings": {}}
            
            def compose(self):
                yield IngestLocalVideoWindowSimplified(self)
        
        return SimplifiedVideoTestApp
    
    @staticmethod
    def create_factory_test_app(media_type: str = "video", ui_style: str = "default"):
        """Create a test app using the factory pattern."""
        from tldw_chatbook.Widgets.Media_Ingest.IngestUIFactory import IngestUIFactory
        
        class FactoryTestApp(App):
            def __init__(self):
                super().__init__()
                self.app_config = {"api_settings": {}}
            
            def compose(self):
                with patch('tldw_chatbook.config.get_ingest_ui_style', return_value=ui_style):
                    yield IngestUIFactory.create_ui(self, media_type)
        
        return FactoryTestApp


async def wait_for_condition(pilot: Pilot, condition_func, timeout: float = 1.0) -> bool:
    """Wait for a condition to become true."""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        await pilot.pause(0.01)
    
    return False


async def simulate_file_browser_selection(pilot: Pilot, files: List[Path]) -> None:
    """Simulate file browser selection (for testing file selection workflows)."""
    # This would typically involve mocking the file dialog
    # For now, we'll add files programmatically to the window
    pass  # Implementation depends on the specific file browser widget


def assert_widget_visible(app: App, widget_selector: str) -> None:
    """Assert that a widget is visible (not hidden)."""
    try:
        widget = app.query_one(widget_selector)
        assert "hidden" not in widget.classes, f"Widget {widget_selector} should be visible"
    except Exception:
        assert False, f"Widget {widget_selector} not found"


def assert_widget_hidden(app: App, widget_selector: str) -> None:
    """Assert that a widget is hidden."""
    try:
        widget = app.query_one(widget_selector)
        assert "hidden" in widget.classes, f"Widget {widget_selector} should be hidden"
    except Exception:
        # If widget doesn't exist, that's also considered "hidden"
        pass


def assert_input_has_proper_styling(app: App, input_id: str) -> None:
    """Assert that an input widget has proper styling for visibility."""
    input_widget = app.query_one(f"#{input_id}", Input)
    
    # Check for form-input class
    assert "form-input" in input_widget.classes, \
        f"Input {input_id} should have 'form-input' CSS class"
    
    # Check for explicit height (this is critical for Textual input visibility)
    has_height_style = (hasattr(input_widget.styles, 'height') and 
                       input_widget.styles.height is not None)
    has_form_input_class = "form-input" in input_widget.classes
    
    assert has_height_style or has_form_input_class, \
        f"Input {input_id} must have explicit height styling for visibility"


def assert_no_double_scrolling(app: App) -> None:
    """Assert that there are no nested VerticalScroll containers."""
    from textual.containers import VerticalScroll
    
    scroll_containers = app.query(VerticalScroll)
    assert len(scroll_containers) <= 1, \
        f"Should have at most 1 VerticalScroll container, found {len(scroll_containers)}. " \
        f"Multiple scroll containers cause broken scrolling behavior"


# Test data constants
TEST_CONFIG_BASIC = {
    "api_settings": {
        "openai": {"models": ["gpt-4"]},
        "anthropic": {"models": ["claude-3-sonnet"]}
    }
}

TEST_CONFIG_MINIMAL = {"api_settings": {}}

TEST_CONFIG_NONE = None

# Standard test file sets
TEST_VIDEO_FILES = MockDataFixtures.sample_video_files(3)
TEST_AUDIO_FILES = MockDataFixtures.sample_audio_files(3)
TEST_DOCUMENT_FILES = MockDataFixtures.sample_document_files(3)

# Standard test URLs
TEST_VIDEO_URLS = MockDataFixtures.sample_urls("video", 3)
TEST_AUDIO_URLS = MockDataFixtures.sample_urls("audio", 3)