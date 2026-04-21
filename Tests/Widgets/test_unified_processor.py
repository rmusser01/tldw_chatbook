# test_unified_processor.py
"""
Unit tests for UnifiedProcessor and related components.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from textual.app import App

from tldw_chatbook.Widgets.NewIngest.UnifiedProcessor import (
    UnifiedProcessor,
    ModeToggle,
    MediaSpecificOptions, 
    ProcessingMode,
    ProcessingStatus,
    VideoConfig,
    AudioConfig,
    DocumentConfig,
    PDFConfig,
    EbookConfig,
    WebConfig,
    ProcessingStarted,
    ProcessingComplete,
    ProcessingError
)


class TestApp(App):
    """Test app for component testing."""
    
    def compose(self):
        return UnifiedProcessor(self)


@pytest.mark.asyncio
async def test_processing_status_model():
    """Test ProcessingStatus model validation."""
    # Valid status
    status = ProcessingStatus(
        state="processing",
        progress=0.5,
        current_file="test.mp4",
        files_completed=1,
        total_files=2,
        message="Processing file",
        elapsed_time=30.0
    )
    
    assert status.state == "processing"
    assert status.progress == 0.5
    assert status.current_file == "test.mp4"
    assert status.files_completed == 1
    assert status.total_files == 2
    assert status.elapsed_time == 30.0
    
    # Test validation
    with pytest.raises(ValueError):
        ProcessingStatus(progress=1.5)  # Progress > 1.0
    
    with pytest.raises(ValueError):
        ProcessingStatus(files_completed=-1)  # Negative count


@pytest.mark.asyncio
async def test_video_config_model():
    """Test VideoConfig model validation."""
    config = VideoConfig(
        files=[Path("test.mp4")],
        extract_audio_only=True,
        start_time="00:01:30",
        end_time="00:05:00",
        transcription_provider="whisper",
        chunk_size=500,
        chunk_overlap=75
    )
    
    assert len(config.files) == 1
    assert config.extract_audio_only == True
    assert config.start_time == "00:01:30"
    assert config.chunk_size == 500
    
    # Test validation
    with pytest.raises(ValueError):
        VideoConfig(chunk_size=50)  # Below minimum
    
    with pytest.raises(ValueError):
        VideoConfig(chunk_overlap=250)  # Above maximum


@pytest.mark.asyncio
async def test_audio_config_model():
    """Test AudioConfig model validation.""" 
    config = AudioConfig(
        files=[Path("test.mp3")],
        speaker_diarization=True,
        noise_reduction=True,
        transcription_model="large"
    )
    
    assert config.speaker_diarization == True
    assert config.noise_reduction == True
    assert config.transcription_model == "large"


@pytest.mark.asyncio
async def test_document_config_model():
    """Test DocumentConfig model validation."""
    config = DocumentConfig(
        files=[Path("test.docx")],
        ocr_enabled=True,
        preserve_formatting=False,
        chunk_method="semantic"
    )
    
    assert config.ocr_enabled == True
    assert config.preserve_formatting == False
    assert config.chunk_method == "semantic"


@pytest.mark.asyncio
async def test_pdf_config_model():
    """Test PDFConfig model validation."""
    config = PDFConfig(
        files=[Path("test.pdf")],
        extract_images=True,
        preserve_layout=True,
        chunk_size=600
    )
    
    assert config.extract_images == True
    assert config.preserve_layout == True
    assert config.chunk_size == 600


@pytest.mark.asyncio
async def test_ebook_config_model():
    """Test EbookConfig model validation."""
    config = EbookConfig(
        files=[Path("test.epub")],
        extract_metadata=True,
        preserve_chapters=True,
        include_toc=True,
        chunk_method="chapter"
    )
    
    assert config.extract_metadata == True
    assert config.preserve_chapters == True
    assert config.include_toc == True
    assert config.chunk_method == "chapter"


@pytest.mark.asyncio
async def test_web_config_model():
    """Test WebConfig model validation."""
    config = WebConfig(
        files=[Path("test.html")],
        extract_links=True,
        include_images=True,
        clean_html=False
    )
    
    assert config.extract_links == True
    assert config.include_images == True
    assert config.clean_html == False


@pytest.mark.asyncio
async def test_mode_toggle_initialization():
    """Test ModeToggle initializes correctly."""
    toggle = ModeToggle()
    assert toggle.current_mode == ProcessingMode.SIMPLE


@pytest.mark.asyncio
async def test_mode_toggle_compose():
    """Test ModeToggle composes correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        toggle = ModeToggle()
        await app.mount(toggle)
        await pilot.pause()
        
        # Check components exist
        assert toggle.query(".mode-toggle")
        assert toggle.query(".mode-label")
        assert toggle.query("#mode-selector")
        assert toggle.query("#simple-mode")
        assert toggle.query("#advanced-mode")
        assert toggle.query("#expert-mode")
        assert toggle.query("#mode-description")


@pytest.mark.asyncio
async def test_mode_toggle_mode_changes():
    """Test ModeToggle handles mode changes correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        toggle = ModeToggle()
        await app.mount(toggle)
        await pilot.pause()
        
        # Click advanced mode
        await pilot.click("#advanced-mode")
        await pilot.pause()
        
        assert toggle.current_mode == ProcessingMode.ADVANCED
        
        # Click expert mode
        await pilot.click("#expert-mode")
        await pilot.pause()
        
        assert toggle.current_mode == ProcessingMode.EXPERT


@pytest.mark.asyncio
async def test_media_specific_options_initialization():
    """Test MediaSpecificOptions initializes correctly."""
    options = MediaSpecificOptions()
    assert options.media_type == "auto"
    assert options.processing_mode == ProcessingMode.SIMPLE


@pytest.mark.asyncio
async def test_media_specific_options_compose():
    """Test MediaSpecificOptions composes correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        options = MediaSpecificOptions()
        await app.mount(options)
        await pilot.pause()
        
        # Check components exist
        assert options.query(".media-options")
        assert options.query(".options-title")
        assert options.query("#options-content")


@pytest.mark.asyncio
async def test_media_specific_options_video_rebuild():
    """Test MediaSpecificOptions rebuilds correctly for video."""
    app = TestApp()
    async with app.run_test() as pilot:
        options = MediaSpecificOptions()
        await app.mount(options)
        await pilot.pause()
        
        # Change to video type
        options.media_type = "video"
        await pilot.pause()
        
        # Check video-specific options exist
        assert options.query("#extract-audio-only")
        assert options.query("#transcription-provider")


@pytest.mark.asyncio 
async def test_media_specific_options_audio_rebuild():
    """Test MediaSpecificOptions rebuilds correctly for audio."""
    app = TestApp()
    async with app.run_test() as pilot:
        options = MediaSpecificOptions()
        await app.mount(options)
        await pilot.pause()
        
        # Change to audio type
        options.media_type = "audio"
        await pilot.pause()
        
        # Check audio-specific options exist
        assert options.query("#speaker-diarization")
        assert options.query("#noise-reduction")


@pytest.mark.asyncio
async def test_media_specific_options_config_data():
    """Test MediaSpecificOptions can extract configuration data."""
    app = TestApp()
    async with app.run_test() as pilot:
        options = MediaSpecificOptions()
        await app.mount(options)
        await pilot.pause()
        
        # Set to video type to get some widgets
        options.media_type = "video"
        await pilot.pause()
        
        # Get config (should not error even with no values set)
        config = options.get_config_data()
        assert isinstance(config, dict)


@pytest.mark.asyncio
async def test_unified_processor_initialization():
    """Test UnifiedProcessor initializes correctly."""
    mock_app = Mock()
    processor = UnifiedProcessor(mock_app)
    
    assert processor.app_instance == mock_app
    assert processor.selected_files == []
    assert processor.media_type == "auto"
    assert processor.processing_mode == ProcessingMode.SIMPLE
    assert isinstance(processor.processing_status, ProcessingStatus)


@pytest.mark.asyncio
async def test_unified_processor_initialization_with_files():
    """Test UnifiedProcessor initializes correctly with initial files."""
    mock_app = Mock()
    test_files = [Path("test.mp4"), Path("test2.mp4")]
    
    processor = UnifiedProcessor(mock_app, initial_files=test_files)
    
    assert processor.selected_files == test_files
    assert processor.media_type == "video"  # Should auto-detect


@pytest.mark.asyncio
async def test_unified_processor_compose():
    """Test UnifiedProcessor composes correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        processor = app.query_one(UnifiedProcessor)
        
        # Check main components exist
        assert processor.query(".processor-title")
        assert processor.query(".processor-subtitle")
        assert processor.query(".processor-content")
        assert processor.query(".file-panel")
        assert processor.query(".options-panel")
        assert processor.query("#file-selector")
        assert processor.query("#mode-toggle")
        assert processor.query("#media-options")
        assert processor.query("#process-button")


@pytest.mark.asyncio
async def test_unified_processor_media_type_detection():
    """Test UnifiedProcessor media type detection."""
    mock_app = Mock()
    processor = UnifiedProcessor(mock_app)
    
    # Test video detection
    video_files = [Path("test.mp4"), Path("test2.avi")]
    assert processor._detect_media_type(video_files) == "video"
    
    # Test audio detection
    audio_files = [Path("test.mp3"), Path("test2.wav")]
    assert processor._detect_media_type(audio_files) == "audio"
    
    # Test PDF detection
    pdf_files = [Path("test.pdf")]
    assert processor._detect_media_type(pdf_files) == "pdf"
    
    # Test document detection
    doc_files = [Path("test.docx"), Path("test2.txt")]
    assert processor._detect_media_type(doc_files) == "document"
    
    # Test ebook detection
    ebook_files = [Path("test.epub"), Path("test2.mobi")]
    assert processor._detect_media_type(ebook_files) == "ebook"
    
    # Test web detection
    web_files = [Path("test.html"), Path("test2.xml")]
    assert processor._detect_media_type(web_files) == "web"
    
    # Test mixed types
    mixed_files = [Path("test.mp4"), Path("test.pdf")]
    assert processor._detect_media_type(mixed_files) == "mixed"
    
    # Test empty
    assert processor._detect_media_type([]) == "auto"


@pytest.mark.asyncio
async def test_unified_processor_config_models():
    """Test UnifiedProcessor returns correct config models."""
    mock_app = Mock()
    processor = UnifiedProcessor(mock_app)
    
    # Test video config
    processor.media_type = "video"
    assert processor._get_config_model() == VideoConfig
    
    # Test audio config
    processor.media_type = "audio" 
    assert processor._get_config_model() == AudioConfig
    
    # Test document config
    processor.media_type = "document"
    assert processor._get_config_model() == DocumentConfig
    
    # Test PDF config
    processor.media_type = "pdf"
    assert processor._get_config_model() == PDFConfig
    
    # Test ebook config
    processor.media_type = "ebook"
    assert processor._get_config_model() == EbookConfig
    
    # Test web config
    processor.media_type = "web"
    assert processor._get_config_model() == WebConfig


@pytest.mark.asyncio
async def test_unified_processor_configuration_extraction():
    """Test UnifiedProcessor can extract configuration from UI."""
    app = TestApp()
    async with app.run_test() as pilot:
        processor = app.query_one(UnifiedProcessor)
        
        # Set some test values
        processor.selected_files = [Path("test.mp4")]
        processor.media_type = "video"
        
        # Get configuration (should not error)
        config = processor._get_configuration()
        assert isinstance(config, dict)
        assert "files" in config
        assert config["files"] == [Path("test.mp4")]


@pytest.mark.asyncio
async def test_unified_processor_process_button_state():
    """Test UnifiedProcessor updates process button correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        processor = app.query_one(UnifiedProcessor)
        
        # Initially disabled (no files)
        button = processor.query_one("#process-button")
        assert button.disabled == True
        
        # Add files - should enable
        processor.selected_files = [Path("test.mp4")]
        await pilot.pause()
        
        assert button.disabled == False


@pytest.mark.asyncio
async def test_unified_processor_processing_simulation():
    """Test UnifiedProcessor processing simulation."""
    app = TestApp()
    async with app.run_test() as pilot:
        processor = app.query_one(UnifiedProcessor)
        
        # Setup test files
        test_files = [Path("test.mp4")]
        processor.selected_files = test_files
        processor.media_type = "video"
        
        # Mock backend processor
        async def mock_processor(file_path, config):
            return {"file": str(file_path), "status": "success"}
        
        processor._call_backend_processor = mock_processor
        
        # Create test config
        config = VideoConfig(files=test_files)
        
        # Process (should complete without error)
        results = await processor._process_media(config)
        
        assert isinstance(results, dict)
        assert "processed_files" in results
        assert "errors" in results


@pytest.mark.asyncio
async def test_processing_messages():
    """Test processing message creation."""
    config_data = {"files": [Path("test.mp4")]}
    
    # Test ProcessingStarted
    start_msg = ProcessingStarted(config_data)
    assert start_msg.config == config_data
    
    # Test ProcessingComplete
    results = {"processed": 1}
    complete_msg = ProcessingComplete(results)
    assert complete_msg.results == results
    
    # Test ProcessingError
    error_msg = ProcessingError("Test error")
    assert error_msg.error == "Test error"


@pytest.mark.asyncio
async def test_unified_processor_watchers():
    """Test UnifiedProcessor reactive watchers."""
    app = TestApp()
    async with app.run_test() as pilot:
        processor = app.query_one(UnifiedProcessor)
        
        # Test file selection watcher
        test_files = [Path("test.mp4")]
        processor.selected_files = test_files
        await pilot.pause()
        
        # Should update media type
        assert processor.media_type == "video"
        
        # Test processing mode watcher
        processor.processing_mode = ProcessingMode.ADVANCED
        await pilot.pause()
        
        # Options should update mode
        options = processor.query_one("#media-options")
        assert options.processing_mode == ProcessingMode.ADVANCED


@pytest.mark.asyncio
async def test_unified_processor_status_display():
    """Test UnifiedProcessor status display updates."""
    app = TestApp()
    async with app.run_test() as pilot:
        processor = app.query_one(UnifiedProcessor)
        
        # Update status
        status = ProcessingStatus(
            state="processing",
            progress=0.5,
            message="Processing...",
            files_completed=1,
            total_files=2
        )
        
        processor.processing_status = status
        await pilot.pause()
        
        # Should update UI (no errors)
        status_container = processor.query_one("#status-container")
        assert status_container is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])