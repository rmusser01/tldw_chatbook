# test_new_ingest_window.py
"""
Unit tests for the new modern ingest window components.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from textual.app import App

from tldw_chatbook.UI.NewIngestWindow import (
    NewIngestWindow, 
    MediaTypeCard, 
    GlobalDropZone, 
    ActivityFeed,
    MediaTypeSelected,
    FileDropped
)


class TestApp(App):
    """Test app for component testing."""
    
    def compose(self):
        yield NewIngestWindow(self)


@pytest.mark.asyncio
async def test_media_type_card_initialization():
    """Test MediaTypeCard initializes correctly."""
    card = MediaTypeCard("video", "Video Content", "Test description", "🎬")
    
    assert card.media_type == "video"
    assert card.title == "Video Content" 
    assert card.description == "Test description"
    assert card.icon == "🎬"


@pytest.mark.asyncio
async def test_media_type_card_compose():
    """Test MediaTypeCard composes correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        card = MediaTypeCard("video", "Video Content", "Test description", "🎬")
        
        # Mount the card for testing
        await app.mount(card)
        await pilot.pause()
        
        # Check components exist
        assert card.query(".media-card")
        assert card.query(".card-header")
        assert card.query(".card-icon")
        assert card.query(".card-title")
        assert card.query(".card-description")
        assert card.query(".card-button")


@pytest.mark.asyncio
async def test_media_type_card_selection():
    """Test MediaTypeCard posts correct message when selected."""
    app = TestApp()
    async with app.run_test() as pilot:
        card = MediaTypeCard("video", "Video Content", "Test description", "🎬")
        await app.mount(card)
        await pilot.pause()
        
        # Track messages
        messages = []
        def capture_message(message):
            messages.append(message)
        
        app.on_event = capture_message
        
        # Click the select button
        await pilot.click("#select-video")
        await pilot.pause()
        
        # Verify MediaTypeSelected message was posted
        assert any(isinstance(msg, MediaTypeSelected) and msg.media_type == "video" 
                  for msg in messages)


@pytest.mark.asyncio
async def test_global_drop_zone_initialization():
    """Test GlobalDropZone initializes correctly."""
    drop_zone = GlobalDropZone()
    
    assert drop_zone.is_active == False
    assert drop_zone.has_files == False
    assert drop_zone.file_count == 0


@pytest.mark.asyncio 
async def test_global_drop_zone_file_addition():
    """Test GlobalDropZone handles file addition correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        drop_zone = GlobalDropZone()
        await app.mount(drop_zone)
        await pilot.pause()
        
        # Create test files
        test_files = [Path("test1.mp4"), Path("test2.mp4")]
        
        # Add files
        drop_zone.add_files(test_files)
        await pilot.pause()
        
        # Check state
        assert drop_zone.file_count == 2
        assert drop_zone.has_files == True
        
        # Check UI updates
        file_count_widget = drop_zone.query_one("#file-count")
        assert "hidden" not in file_count_widget.classes


@pytest.mark.asyncio
async def test_global_drop_zone_reactive_updates():
    """Test GlobalDropZone reactive properties update UI correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        drop_zone = GlobalDropZone()
        await app.mount(drop_zone)
        await pilot.pause()
        
        # Test is_active watcher
        drop_zone.is_active = True
        await pilot.pause()
        assert "active" in drop_zone.classes
        
        drop_zone.is_active = False  
        await pilot.pause()
        assert "active" not in drop_zone.classes


@pytest.mark.asyncio
async def test_activity_feed_initialization():
    """Test ActivityFeed initializes correctly."""
    feed = ActivityFeed()
    
    assert feed.activities == []


@pytest.mark.asyncio
async def test_activity_feed_add_activity():
    """Test ActivityFeed can add activities correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        feed = ActivityFeed()
        await app.mount(feed)
        await pilot.pause()
        
        # Add an activity
        feed.add_activity("Test activity", "processing", 0.5)
        await pilot.pause()
        
        # Check activity was added
        assert len(feed.activities) == 1
        activity = feed.activities[0]
        assert activity["title"] == "Test activity"
        assert activity["status"] == "processing" 
        assert activity["progress"] == 0.5
        assert "time" in activity


@pytest.mark.asyncio
async def test_activity_feed_status_icons():
    """Test ActivityFeed returns correct status icons."""
    feed = ActivityFeed()
    
    assert feed._get_status_icon("completed") == "✅"
    assert feed._get_status_icon("processing") == "⚙️" 
    assert feed._get_status_icon("failed") == "❌"
    assert feed._get_status_icon("queued") == "⏳"
    assert feed._get_status_icon("unknown") == "📄"


@pytest.mark.asyncio
async def test_new_ingest_window_initialization():
    """Test NewIngestWindow initializes correctly."""
    mock_app = Mock()
    window = NewIngestWindow(mock_app)
    
    assert window.app_instance == mock_app
    assert window.selected_files == []
    assert window.current_media_type == "auto"
    assert window.processing_active == False


@pytest.mark.asyncio
async def test_new_ingest_window_compose():
    """Test NewIngestWindow composes all components correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        window = app.query_one(NewIngestWindow)
        
        # Check main components exist
        assert window.query(".main-title")
        assert window.query(".main-subtitle")
        assert window.query(".main-content")
        assert window.query(".media-selection-panel")
        assert window.query(".activity-panel")
        assert window.query(".quick-actions")
        
        # Check media type cards exist
        media_cards = window.query(MediaTypeCard)
        assert len(media_cards) == 6  # video, audio, document, pdf, web, ebook
        
        # Check specific cards
        video_card = window.query_one(MediaTypeCard).filter(lambda c: c.media_type == "video")
        assert video_card is not None
        
        # Check other components
        assert window.query_one(GlobalDropZone)
        assert window.query_one(ActivityFeed)


@pytest.mark.asyncio
async def test_new_ingest_window_media_type_detection():
    """Test media type detection from file extensions."""
    mock_app = Mock()
    window = NewIngestWindow(mock_app)
    
    # Test video detection
    video_files = [Path("test.mp4"), Path("test2.avi")]
    assert window._detect_media_type(video_files) == "video"
    
    # Test audio detection
    audio_files = [Path("test.mp3"), Path("test2.wav")]
    assert window._detect_media_type(audio_files) == "audio"
    
    # Test PDF detection
    pdf_files = [Path("test.pdf")]
    assert window._detect_media_type(pdf_files) == "pdf"
    
    # Test document detection
    doc_files = [Path("test.txt"), Path("test2.docx")]
    assert window._detect_media_type(doc_files) == "document"
    
    # Test ebook detection  
    ebook_files = [Path("test.epub"), Path("test2.mobi")]
    assert window._detect_media_type(ebook_files) == "ebook"
    
    # Test mixed types
    mixed_files = [Path("test.mp4"), Path("test.pdf")]
    assert window._detect_media_type(mixed_files) is None
    
    # Test empty
    assert window._detect_media_type([]) is None


@pytest.mark.asyncio
async def test_new_ingest_window_handles_media_type_selection():
    """Test NewIngestWindow handles MediaTypeSelected messages."""
    app = TestApp()
    async with app.run_test() as pilot:
        window = app.query_one(NewIngestWindow)
        
        # Send MediaTypeSelected message
        message = MediaTypeSelected("video")
        window.handle_media_type_selected(message)
        
        # Check state updated
        assert window.current_media_type == "video"


@pytest.mark.asyncio
async def test_new_ingest_window_handles_file_dropped():
    """Test NewIngestWindow handles FileDropped messages."""
    app = TestApp()
    async with app.run_test() as pilot:
        window = app.query_one(NewIngestWindow)
        
        # Create test files
        test_files = [Path("test1.mp4"), Path("test2.mp4")]
        
        # Send FileDropped message  
        message = FileDropped(test_files)
        window.handle_files_dropped(message)
        await pilot.pause()
        
        # Check state updated
        assert window.selected_files == test_files
        assert window.current_media_type == "video"  # Should auto-detect


@pytest.mark.asyncio
async def test_new_ingest_window_browse_files():
    """Test NewIngestWindow browse files functionality."""
    app = TestApp()
    
    with patch('tldw_chatbook.UI.NewIngestWindow.FileOpen') as mock_file_open:
        # Mock file selection
        test_files = [Path("test.mp4")]
        mock_file_open.return_value = AsyncMock()
        
        async with app.run_test() as pilot:
            # Mock push_screen_wait to return test files
            app.push_screen_wait = AsyncMock(return_value=test_files)
            
            window = app.query_one(NewIngestWindow)
            
            # Click browse button
            await pilot.click("#browse-files")
            await pilot.pause()
            
            # Check files were set
            assert window.selected_files == test_files


@pytest.mark.asyncio
async def test_file_dropped_message():
    """Test FileDropped message creation."""
    test_files = [Path("test1.mp4"), Path("test2.mp4")]
    message = FileDropped(test_files)
    
    assert message.files == test_files
    assert len(message.files) == 2


@pytest.mark.asyncio
async def test_media_type_selected_message():
    """Test MediaTypeSelected message creation."""
    message = MediaTypeSelected("video")
    
    assert message.media_type == "video"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])