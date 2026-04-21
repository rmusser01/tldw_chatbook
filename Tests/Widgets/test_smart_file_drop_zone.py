# test_smart_file_drop_zone.py
"""
Unit tests for SmartFileDropZone component.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from textual.app import App

from tldw_chatbook.Widgets.NewIngest.SmartFileDropZone import (
    SmartFileDropZone,
    FilePreviewItem, 
    FilesSelected,
    FileRemoved
)


class TestApp(App):
    """Test app for component testing."""
    
    def compose(self):
        return SmartFileDropZone()


@pytest.mark.asyncio
async def test_file_preview_item_initialization():
    """Test FilePreviewItem initializes correctly."""
    test_file = Path("test.mp4")
    
    with patch.object(Path, 'exists', return_value=True), \
         patch.object(Path, 'stat') as mock_stat:
        
        mock_stat.return_value = MagicMock()
        mock_stat.return_value.st_size = 1024000  # 1MB
        mock_stat.return_value.st_mtime = 1234567890
        
        item = FilePreviewItem(test_file)
        
        assert item.file_path == test_file
        assert item._file_info is not None


@pytest.mark.asyncio  
async def test_file_preview_item_file_analysis():
    """Test FilePreviewItem analyzes files correctly."""
    test_file = Path("test.mp4")
    
    with patch.object(Path, 'exists', return_value=True), \
         patch.object(Path, 'stat') as mock_stat:
        
        mock_stat.return_value = MagicMock()
        mock_stat.return_value.st_size = 1024000  # 1MB
        mock_stat.return_value.st_mtime = 1234567890
        
        item = FilePreviewItem(test_file)
        
        # Test video file detection
        assert item._file_info["icon"] == "🎬"
        assert "Video" in item._file_info["type"]
        assert "1.0 MB" in item._file_info["details"]


@pytest.mark.asyncio
async def test_file_preview_item_size_formatting():
    """Test file size formatting is correct."""
    test_file = Path("test.txt")
    item = FilePreviewItem(test_file)
    
    # Test different size formats
    assert item._format_file_size(512) == "512.0 B"
    assert item._format_file_size(1024) == "1.0 KB"
    assert item._format_file_size(1024 * 1024) == "1.0 MB"
    assert item._format_file_size(1024 * 1024 * 1024) == "1.0 GB"


@pytest.mark.asyncio
async def test_file_preview_item_icon_detection():
    """Test file icon detection for different types."""
    item = FilePreviewItem(Path("dummy.txt"))
    
    # Test video files
    icon, type_name = item._get_file_icon_and_type(".mp4", "video/mp4")
    assert icon == "🎬"
    assert type_name == "Video"
    
    # Test audio files
    icon, type_name = item._get_file_icon_and_type(".mp3", "audio/mp3")
    assert icon == "🎵"
    assert type_name == "Audio"
    
    # Test PDF files
    icon, type_name = item._get_file_icon_and_type(".pdf", "application/pdf")
    assert icon == "📕"
    assert type_name == "PDF"
    
    # Test document files
    icon, type_name = item._get_file_icon_and_type(".docx", None)
    assert icon == "📄"
    assert type_name == "Word Document"
    
    # Test unknown files
    icon, type_name = item._get_file_icon_and_type(".unknown", None)
    assert icon == "📄"
    assert type_name == "File"


@pytest.mark.asyncio
async def test_file_preview_item_compose():
    """Test FilePreviewItem composes correctly."""
    app = TestApp()
    test_file = Path("test.mp4")
    
    with patch.object(Path, 'exists', return_value=True), \
         patch.object(Path, 'stat') as mock_stat:
        
        mock_stat.return_value = MagicMock()
        mock_stat.return_value.st_size = 1024
        mock_stat.return_value.st_mtime = 1234567890
        
        async with app.run_test() as pilot:
            item = FilePreviewItem(test_file)
            await app.mount(item)
            await pilot.pause()
            
            # Check components exist
            assert item.query(".file-preview-item")
            assert item.query(".file-info-row")
            assert item.query(".file-icon")
            assert item.query(".file-details")
            assert item.query(".file-name")
            assert item.query(".file-metadata")
            assert item.query(".remove-button")


@pytest.mark.asyncio
async def test_file_preview_item_remove_message():
    """Test FilePreviewItem posts FileRemoved message when remove is clicked."""
    app = TestApp()
    test_file = Path("test.mp4")
    
    with patch.object(Path, 'exists', return_value=True), \
         patch.object(Path, 'stat') as mock_stat:
        
        mock_stat.return_value = MagicMock()
        mock_stat.return_value.st_size = 1024
        mock_stat.return_value.st_mtime = 1234567890
        
        async with app.run_test() as pilot:
            item = FilePreviewItem(test_file)
            await app.mount(item)
            await pilot.pause()
            
            # Track messages
            messages = []
            original_post = item.post_message
            item.post_message = lambda msg: messages.append(msg)
            
            # Click remove button
            remove_button = item.query_one(".remove-button")
            remove_button.press()
            await pilot.pause()
            
            # Check FileRemoved message was posted
            assert len(messages) == 1
            assert isinstance(messages[0], FileRemoved)
            assert messages[0].file_path == test_file


@pytest.mark.asyncio
async def test_smart_file_drop_zone_initialization():
    """Test SmartFileDropZone initializes correctly."""
    allowed_types = {'.mp4', '.mp3'}
    zone = SmartFileDropZone(allowed_types=allowed_types, max_files=50)
    
    assert zone.selected_files == []
    assert zone.is_dragging == False
    assert zone.allowed_types == allowed_types
    assert zone.max_files == 50


@pytest.mark.asyncio
async def test_smart_file_drop_zone_compose():
    """Test SmartFileDropZone composes correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        zone = app.query_one(SmartFileDropZone)
        
        # Check main components exist
        assert zone.query(".smart-drop-zone")
        assert zone.query("#drop-area")
        assert zone.query("#drop-title")
        assert zone.query("#drop-subtitle")
        assert zone.query("#browse-overlay")
        assert zone.query("#file-list-container")
        assert zone.query("#file-list")
        assert zone.query("#file-summary")
        assert zone.query(".file-actions")


@pytest.mark.asyncio
async def test_smart_file_drop_zone_add_files():
    """Test adding files to SmartFileDropZone."""
    app = TestApp()
    async with app.run_test() as pilot:
        zone = app.query_one(SmartFileDropZone)
        
        # Track messages
        messages = []
        original_post = zone.post_message
        zone.post_message = lambda msg: messages.append(msg)
        
        # Add test files
        test_files = [Path("test1.mp4"), Path("test2.mp3")]
        zone.add_files(test_files)
        await pilot.pause()
        
        # Check state updated
        assert zone.selected_files == test_files
        
        # Check FilesSelected message posted
        assert len(messages) == 1
        assert isinstance(messages[0], FilesSelected)
        assert messages[0].files == test_files


@pytest.mark.asyncio
async def test_smart_file_drop_zone_file_type_validation():
    """Test file type validation in SmartFileDropZone."""
    allowed_types = {'.mp4', '.mp3'}
    app = TestApp()
    async with app.run_test() as pilot:
        zone = SmartFileDropZone(allowed_types=allowed_types)
        await app.mount(zone)
        await pilot.pause()
        
        # Try to add allowed and disallowed files
        test_files = [Path("test.mp4"), Path("test.pdf")]  # mp4 allowed, pdf not
        zone.add_files(test_files)
        await pilot.pause()
        
        # Check only allowed file was added
        assert len(zone.selected_files) == 1
        assert zone.selected_files[0] == Path("test.mp4")


@pytest.mark.asyncio
async def test_smart_file_drop_zone_max_files_limit():
    """Test max files limit in SmartFileDropZone."""
    app = TestApp()
    async with app.run_test() as pilot:
        zone = SmartFileDropZone(max_files=2)
        await app.mount(zone)
        await pilot.pause()
        
        # Try to add more files than limit
        test_files = [Path(f"test{i}.mp4") for i in range(5)]
        zone.add_files(test_files)
        await pilot.pause()
        
        # Check only max_files were added
        assert len(zone.selected_files) == 2


@pytest.mark.asyncio
async def test_smart_file_drop_zone_duplicate_prevention():
    """Test duplicate file prevention in SmartFileDropZone."""
    app = TestApp()
    async with app.run_test() as pilot:
        zone = app.query_one(SmartFileDropZone)
        
        # Add same file twice
        test_file = Path("test.mp4")
        zone.add_files([test_file])
        zone.add_files([test_file])  # Try to add again
        await pilot.pause()
        
        # Check file only added once
        assert len(zone.selected_files) == 1
        assert zone.selected_files[0] == test_file


@pytest.mark.asyncio
async def test_smart_file_drop_zone_remove_file():
    """Test removing individual files from SmartFileDropZone."""
    app = TestApp()
    async with app.run_test() as pilot:
        zone = app.query_one(SmartFileDropZone)
        
        # Add files
        test_files = [Path("test1.mp4"), Path("test2.mp4")]
        zone.add_files(test_files)
        await pilot.pause()
        
        # Remove one file
        zone.remove_file(Path("test1.mp4"))
        await pilot.pause()
        
        # Check file was removed
        assert len(zone.selected_files) == 1
        assert zone.selected_files[0] == Path("test2.mp4")


@pytest.mark.asyncio
async def test_smart_file_drop_zone_clear_files():
    """Test clearing all files from SmartFileDropZone."""
    app = TestApp()
    async with app.run_test() as pilot:
        zone = app.query_one(SmartFileDropZone)
        
        # Add files
        test_files = [Path("test1.mp4"), Path("test2.mp4")]
        zone.add_files(test_files)
        await pilot.pause()
        
        # Clear all files
        zone.clear_files()
        await pilot.pause()
        
        # Check all files cleared
        assert len(zone.selected_files) == 0


@pytest.mark.asyncio
async def test_smart_file_drop_zone_set_allowed_types():
    """Test setting allowed types filters existing files."""
    app = TestApp()
    async with app.run_test() as pilot:
        zone = app.query_one(SmartFileDropZone)
        
        # Add mixed file types
        test_files = [Path("test.mp4"), Path("test.pdf")]
        zone.add_files(test_files)
        await pilot.pause()
        
        # Set allowed types to only video
        zone.set_allowed_types({'.mp4'})
        await pilot.pause()
        
        # Check only video file remains
        assert len(zone.selected_files) == 1
        assert zone.selected_files[0] == Path("test.mp4")


@pytest.mark.asyncio
async def test_smart_file_drop_zone_file_type_allowed():
    """Test file type checking."""
    zone = SmartFileDropZone(allowed_types={'.mp4', '.mp3'})
    
    assert zone._is_file_type_allowed(Path("test.mp4")) == True
    assert zone._is_file_type_allowed(Path("test.mp3")) == True
    assert zone._is_file_type_allowed(Path("test.pdf")) == False
    
    # Test with no restrictions
    zone_unrestricted = SmartFileDropZone()
    assert zone_unrestricted._is_file_type_allowed(Path("test.anything")) == True


@pytest.mark.asyncio
async def test_smart_file_drop_zone_create_filters():
    """Test file filter creation for dialog."""
    zone = SmartFileDropZone(allowed_types={'.mp4', '.mp3'})
    
    filters = zone._create_file_filters()
    
    # Should have "All Allowed" plus individual filters
    assert len(filters) >= 2
    assert filters[0][0] == "All Allowed Files"
    assert ".mp4" in filters[0][1] and ".mp3" in filters[0][1]


@pytest.mark.asyncio
async def test_smart_file_drop_zone_browse_files():
    """Test browse files functionality."""
    app = TestApp()
    
    with patch('tldw_chatbook.Widgets.NewIngest.SmartFileDropZone.FileOpen') as mock_file_open:
        async with app.run_test() as pilot:
            zone = app.query_one(SmartFileDropZone)
            
            # Mock file selection
            test_files = [Path("test.mp4")]
            app.push_screen_wait = Mock(return_value=test_files)
            
            # Click browse button
            await pilot.click("#browse-overlay")
            await pilot.pause()
            
            # Check files were added
            assert zone.selected_files == test_files


@pytest.mark.asyncio  
async def test_smart_file_drop_zone_clear_all_button():
    """Test clear all button functionality."""
    app = TestApp()
    async with app.run_test() as pilot:
        zone = app.query_one(SmartFileDropZone)
        
        # Add files first
        test_files = [Path("test.mp4")]
        zone.add_files(test_files)
        await pilot.pause()
        
        # Click clear all button
        await pilot.click("#clear-all")
        await pilot.pause()
        
        # Check files were cleared
        assert len(zone.selected_files) == 0


@pytest.mark.asyncio
async def test_smart_file_drop_zone_reactive_updates():
    """Test reactive property watchers update UI correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        zone = app.query_one(SmartFileDropZone)
        
        # Test is_dragging watcher
        zone.is_dragging = True
        await pilot.pause()
        
        drop_area = zone.query_one("#drop-area")
        assert "dragging" in drop_area.classes
        
        zone.is_dragging = False
        await pilot.pause()
        assert "dragging" not in drop_area.classes


@pytest.mark.asyncio
async def test_files_selected_message():
    """Test FilesSelected message creation."""
    test_files = [Path("test1.mp4"), Path("test2.mp4")]
    message = FilesSelected(test_files)
    
    assert message.files == test_files


@pytest.mark.asyncio
async def test_file_removed_message():
    """Test FileRemoved message creation."""
    test_file = Path("test.mp4")
    message = FileRemoved(test_file)
    
    assert message.file_path == test_file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])