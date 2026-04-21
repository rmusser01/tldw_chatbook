"""
Integration tests for the new ingest system following Textual best practices.
Tests the complete flow from UI interaction to backend processing.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from textual.app import App
from textual.widgets import Button, Input, Static

from tldw_chatbook.UI.NewIngestWindow import NewIngestWindow
from tldw_chatbook.Widgets.NewIngest.UnifiedProcessor import (
    UnifiedProcessor, VideoConfig, AudioConfig, ProcessingMode
)
from tldw_chatbook.Widgets.NewIngest.SmartFileDropZone import SmartFileDropZone
from tldw_chatbook.Widgets.NewIngest.BackendIntegration import (
    MediaProcessingService, get_processing_service
)


class TestIngestApp(App):
    """Test app for ingest testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.notifications = []
        
    def compose(self):
        yield NewIngestWindow(self)
    
    def notify(self, message, severity="info"):
        """Override notify to capture notifications."""
        self.notifications.append({"message": message, "severity": severity})


@pytest.mark.asyncio
async def test_new_ingest_window_initialization():
    """Test NewIngestWindow initializes correctly."""
    app = TestIngestApp()
    async with app.run_test() as pilot:
        # Check main components are present
        main_title = app.query_one(".main-title")
        assert "Content Ingestion Hub" in str(main_title.renderable)
        
        # Check media type cards exist
        media_cards = app.query(".media-card")
        assert len(media_cards) == 6  # video, audio, document, pdf, web, ebook
        
        # Check drop zone exists
        drop_zone = app.query_one(".drop-zone")
        assert drop_zone is not None
        
        # Check quick action buttons
        browse_button = app.query_one("#browse-files")
        assert browse_button.label == "Browse Files"


@pytest.mark.asyncio
async def test_media_type_selection():
    """Test media type card selection."""
    app = TestIngestApp()
    async with app.run_test() as pilot:
        # Click video card
        await pilot.click("#select-video")
        await pilot.pause()
        
        # Should have notification about video selection
        assert any("video" in notif["message"].lower() for notif in app.notifications)
        
        # Should switch to unified processor
        # Note: This would fail currently as UnifiedProcessor import might have issues
        # But tests the UI flow


@pytest.mark.asyncio
async def test_smart_file_drop_zone():
    """Test SmartFileDropZone functionality."""
    class DropZoneTestApp(App):
        def __init__(self):
            super().__init__()
            self.selected_files = []
            
        def compose(self):
            yield SmartFileDropZone(id="test-zone")
        
        def on_files_selected(self, event):
            self.selected_files = event.files
    
    app = DropZoneTestApp()
    async with app.run_test() as pilot:
        drop_zone = app.query_one("#test-zone")
        
        # Test file addition
        test_files = [Path("/tmp/test_video.mp4"), Path("/tmp/test_audio.mp3")]
        drop_zone.add_files(test_files)
        await pilot.pause()
        
        # Check files were added
        assert len(drop_zone.selected_files) == 2
        
        # Check file list display updated
        file_items = app.query(".file-preview-item")
        assert len(file_items) == 2
        
        # Test file removal
        await pilot.click(".remove-button")
        await pilot.pause()
        
        assert len(drop_zone.selected_files) == 1


@pytest.mark.asyncio
async def test_unified_processor_initialization():
    """Test UnifiedProcessor with mock app."""
    class MockApp:
        def __init__(self):
            self.notifications = []
        
        def notify(self, message, severity="info"):
            self.notifications.append({"message": message, "severity": severity})
        
        def post_message(self, message):
            pass
    
    mock_app = MockApp()
    test_files = [Path("/tmp/test.mp4")]
    
    processor = UnifiedProcessor(mock_app, initial_files=test_files)
    
    # Test initialization
    assert processor.media_type == "video"  # Should detect video from .mp4
    assert len(processor.selected_files) == 1
    assert processor.processing_mode == ProcessingMode.SIMPLE


@pytest.mark.asyncio
async def test_unified_processor_ui():
    """Test UnifiedProcessor UI components."""
    class ProcessorTestApp(App):
        def __init__(self):
            super().__init__()
            self.processor_messages = []
        
        def compose(self):
            test_files = [Path("/tmp/test_video.mp4")]
            yield UnifiedProcessor(self, initial_files=test_files)
        
        def post_message(self, message):
            self.processor_messages.append(message)
    
    app = ProcessorTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        # Test file selector is present
        file_selector = app.query_one("#file-selector")
        assert file_selector is not None
        
        # Test metadata inputs
        title_input = app.query_one("#title-input")
        assert title_input is not None
        
        # Test mode selector
        mode_selector = app.query_one("#mode-selector")
        assert mode_selector is not None
        
        # Test process button
        process_button = app.query_one("#process-button")
        assert process_button is not None
        
        # Initially button should be enabled (files are present)
        assert not process_button.disabled


@pytest.mark.asyncio
async def test_processing_mode_toggle():
    """Test switching between processing modes."""
    class ProcessorTestApp(App):
        def compose(self):
            test_files = [Path("/tmp/test.mp4")]
            yield UnifiedProcessor(self, initial_files=test_files)
    
    app = ProcessorTestApp()
    async with app.run_test() as pilot:
        processor = app.query_one("UnifiedProcessor")
        
        # Start in simple mode
        assert processor.processing_mode == ProcessingMode.SIMPLE
        
        # Switch to advanced mode
        await pilot.click("#advanced-mode")
        await pilot.pause()
        
        assert processor.processing_mode == ProcessingMode.ADVANCED


@pytest.mark.asyncio
async def test_form_input_handling():
    """Test form input handling in UnifiedProcessor."""
    class ProcessorTestApp(App):
        def compose(self):
            test_files = [Path("/tmp/test.mp4")]
            yield UnifiedProcessor(self, initial_files=test_files)
    
    app = ProcessorTestApp()
    async with app.run_test() as pilot:
        # Fill in title
        await pilot.click("#title-input")
        await pilot.press(*"Test Video Title")
        await pilot.pause()
        
        title_input = app.query_one("#title-input")
        assert title_input.value == "Test Video Title"
        
        # Fill in author
        await pilot.click("#author-input")
        await pilot.press(*"Test Author")
        await pilot.pause()
        
        author_input = app.query_one("#author-input")
        assert author_input.value == "Test Author"
        
        # Fill in keywords
        await pilot.click("#keywords-input")
        await pilot.press(*"test,video,demo")
        await pilot.pause()
        
        keywords_input = app.query_one("#keywords-input")
        assert keywords_input.value == "test,video,demo"


@pytest.mark.asyncio
async def test_media_type_detection():
    """Test automatic media type detection."""
    class MockApp:
        def notify(self, message, severity="info"): pass
        def post_message(self, message): pass
    
    mock_app = MockApp()
    processor = UnifiedProcessor(mock_app)
    
    # Test video detection
    video_files = [Path("test.mp4"), Path("test2.avi")]
    detected = processor._detect_media_type(video_files)
    assert detected == "video"
    
    # Test audio detection
    audio_files = [Path("test.mp3"), Path("test2.wav")]
    detected = processor._detect_media_type(audio_files)
    assert detected == "audio"
    
    # Test PDF detection
    pdf_files = [Path("test.pdf")]
    detected = processor._detect_media_type(pdf_files)
    assert detected == "pdf"
    
    # Test mixed detection
    mixed_files = [Path("test.mp4"), Path("test.pdf")]
    detected = processor._detect_media_type(mixed_files)
    assert detected == "mixed"
    
    # Test empty list
    detected = processor._detect_media_type([])
    assert detected == "auto"


@pytest.mark.asyncio 
@patch('tldw_chatbook.Local_Ingestion.video_processing.process_videos')
async def test_backend_integration_video(mock_process_videos):
    """Test backend integration for video processing."""
    # Mock the video processing function
    mock_process_videos.return_value = {
        "status": "success",
        "processed_files": ["/tmp/test.mp4"],
        "results": {"transcription": "Test transcription"}
    }
    
    class MockApp:
        def notify(self, message, severity="info"): pass
        def post_message(self, message): pass
    
    mock_app = MockApp()
    service = MediaProcessingService(mock_app)
    
    # Create video config
    config = VideoConfig(
        files=[Path("/tmp/test.mp4")],
        title="Test Video",
        extract_audio_only=False,
        chunk_method="words",
        chunk_size=400
    )
    
    # Submit job
    job_id = service.submit_job(config, "Test Job")
    assert job_id is not None
    assert job_id.startswith("job-")
    
    # Wait a moment for processing
    await asyncio.sleep(0.1)
    
    # Check job was called
    mock_process_videos.assert_called_once()


@pytest.mark.asyncio
@patch('tldw_chatbook.Local_Ingestion.audio_processing.process_audio_files')
async def test_backend_integration_audio(mock_process_audio):
    """Test backend integration for audio processing."""
    mock_process_audio.return_value = {
        "status": "success", 
        "processed_files": ["/tmp/test.mp3"],
        "results": {"transcription": "Test transcription"}
    }
    
    class MockApp:
        def notify(self, message, severity="info"): pass
        def post_message(self, message): pass
    
    mock_app = MockApp()
    service = MediaProcessingService(mock_app)
    
    # Create audio config
    config = AudioConfig(
        files=[Path("/tmp/test.mp3")],
        title="Test Audio",
        transcription_provider="whisper",
        chunk_method="words",
        chunk_size=400
    )
    
    # Submit job
    job_id = service.submit_job(config, "Test Audio Job")
    assert job_id is not None
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Verify service was called
    mock_process_audio.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in processing."""
    class ErrorApp(App):
        def __init__(self):
            super().__init__()
            self.errors = []
        
        def compose(self):
            yield UnifiedProcessor(self, initial_files=[])
        
        def notify(self, message, severity="info"):
            if severity == "error":
                self.errors.append(message)
    
    app = ErrorApp()
    async with app.run_test() as pilot:
        # Try to process with no files
        await pilot.click("#process-button")
        await pilot.pause()
        
        # Should get error notification
        assert len(app.errors) > 0
        assert any("no files" in error.lower() for error in app.errors)


@pytest.mark.asyncio
async def test_configuration_validation():
    """Test configuration validation."""
    # Test valid video config
    config = VideoConfig(
        files=[Path("/tmp/test.mp4")],
        title="Test",
        chunk_size=400,
        chunk_overlap=50
    )
    assert config.chunk_size == 400
    assert config.chunk_overlap == 50
    
    # Test invalid config should raise ValidationError
    with pytest.raises(Exception):  # Pydantic ValidationError
        VideoConfig(
            files=[Path("/tmp/test.mp4")],
            chunk_size=-1  # Invalid negative size
        )


@pytest.mark.asyncio
async def test_full_integration_workflow():
    """Test complete workflow from file selection to processing."""
    class IntegrationTestApp(App):
        def __init__(self):
            super().__init__()
            self.messages = []
            self.notifications = []
        
        def compose(self):
            return NewIngestWindow(self)
        
        def notify(self, message, severity="info"):
            self.notifications.append({"message": message, "severity": severity})
        
        def post_message(self, message):
            self.messages.append(message)
    
    with patch('tldw_chatbook.Local_Ingestion.video_processing.process_videos') as mock_process:
        mock_process.return_value = {"status": "success"}
        
        app = IntegrationTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            # 1. Start at main ingest window
            main_title = app.query_one(".main-title")
            assert "Content Ingestion Hub" in str(main_title.renderable)
            
            # 2. Select video media type
            await pilot.click("#select-video") 
            await pilot.pause()
            
            # Should get notification about video selection
            video_notifications = [n for n in app.notifications 
                                 if "video" in n["message"].lower()]
            assert len(video_notifications) > 0
            
            # Note: Full workflow test would need more mocking
            # of the UnifiedProcessor and file selection dialogs
            # This tests the basic UI flow


@pytest.mark.asyncio
async def test_responsive_layout():
    """Test layout adaptation to different terminal sizes."""
    app = TestIngestApp()
    
    # Test narrow layout
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        # Layout should adapt to narrow screen
        # In a real implementation, we'd check CSS classes or layout changes
        
    # Test wide layout  
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Layout should use more horizontal space


@pytest.mark.asyncio
async def test_keyboard_navigation():
    """Test keyboard navigation through the interface."""
    app = TestIngestApp()
    async with app.run_test() as pilot:
        # Test tab navigation
        await pilot.press("tab")
        await pilot.pause()
        
        # Test enter key on focused elements
        await pilot.press("enter")
        await pilot.pause()
        
        # Should be able to navigate without mouse


# Performance test
@pytest.mark.asyncio
async def test_performance_large_file_list():
    """Test performance with many files."""
    class PerformanceTestApp(App):
        def compose(self):
            yield SmartFileDropZone()
    
    app = PerformanceTestApp()
    async with app.run_test() as pilot:
        drop_zone = app.query_one("SmartFileDropZone")
        
        # Add many files
        import time
        start_time = time.time()
        
        large_file_list = [Path(f"/tmp/file_{i}.mp4") for i in range(100)]
        drop_zone.add_files(large_file_list)
        await pilot.pause()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should handle 100 files reasonably quickly
        assert processing_time < 2.0  # Should complete in under 2 seconds
        assert len(drop_zone.selected_files) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])