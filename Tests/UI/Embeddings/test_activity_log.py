"""
Tests for Activity Log Widget.
Tests log entry management, filtering, export functionality, and UI behavior.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime, timedelta
from pathlib import Path

from textual.widgets import Button, Select, ListView, ListItem, Static
from textual.containers import Container, ScrollableContainer

from tldw_chatbook.Widgets.activity_log import (
    ActivityLogWidget,
    ActivityEntry,
    LogLevel
)

from .test_base import EmbeddingsTestBase, WidgetTestApp


class TestActivityEntry(EmbeddingsTestBase):
    """Test ActivityEntry dataclass functionality."""
    
    def test_log_entry_creation(self):
        """Test creating log entries."""
        # Test with minimal args
        entry = ActivityEntry(
            timestamp=datetime.now(),
            level="info",
            category="general",
            message="Test message"
        )
        
        assert entry.message == "Test message"
        assert entry.level == "info"
        assert entry.category == "general"
        assert entry.timestamp is not None
        assert isinstance(entry.timestamp, datetime)
    
    def test_log_entry_with_all_fields(self):
        """Test log entry with all fields specified."""
        timestamp = datetime.now()
        entry = ActivityEntry(
            timestamp=timestamp,
            level="error",
            category="processing",
            message="Detailed message",
            details={"error_code": 500, "file": "test.txt"}
        )
        
        assert entry.message == "Detailed message"
        assert entry.level == "error"
        assert entry.category == "processing"
        assert entry.timestamp == timestamp
        assert entry.details["error_code"] == 500
        assert entry.details["file"] == "test.txt"
    
    def test_log_entry_to_dict(self):
        """Test converting log entry to dictionary."""
        entry = ActivityEntry(
            timestamp=datetime.now(),
            level="warning",
            category="export",
            message="Export test",
            details={"count": 10}
        )
        
        data = entry.to_dict()
        assert data["message"] == "Export test"
        assert data["level"] == "warning"
        assert data["category"] == "export"
        assert "timestamp" in data
        assert data["details"]["count"] == 10
    
    def test_log_entry_to_string(self):
        """Test string representation of log entry."""
        entry = ActivityEntry(
            timestamp=datetime.now(),
            level="debug",
            category="test",
            message="String test"
        )
        
        string_repr = entry.to_string()
        assert "String test" in string_repr
        assert "DEBUG" in string_repr.upper()
        assert "test" in string_repr


class TestActivityLogWidget(EmbeddingsTestBase):
    """Test ActivityLogWidget functionality."""
    
    @pytest.mark.asyncio
    async def test_widget_creation(self):
        """Test creating activity log widget."""
        log = ActivityLogWidget(
            max_entries=100,
            show_filters=True,
            show_export=True
        )
        
        assert log.max_entries == 100
        assert log.show_filters == True
        assert log.show_export == True
        assert len(log.entries) == 0
    
    @pytest.mark.asyncio
    async def test_widget_compose(self):
        """Test widget UI composition."""
        log = ActivityLogWidget()
        
        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Check header
            header = pilot.app.query_one(".activity-log-header")
            assert header is not None
            
            # Check filter controls
            level_filter = pilot.app.query_one("#log-level-filter", Select)
            assert level_filter is not None
            
            category_filter = pilot.app.query_one("#log-category-filter", Select)
            assert category_filter is not None
            
            # Check action buttons
            clear_btn = pilot.app.query_one("#clear-log", Button)
            assert clear_btn is not None
            assert clear_btn.label == "Clear Log"
            
            export_btn = pilot.app.query_one("#export-log", Button)
            assert export_btn is not None
            assert export_btn.label == "Export"
            
            # Check log container
            log_list = pilot.app.query_one("#log-entries", ListView)
            assert log_list is not None
    
    @pytest.mark.asyncio
    async def test_add_entry(self):
        """Test adding log entries."""
        log = ActivityLogWidget()
        
        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add entries
            log.add_entry("First message", "info", "test")
            log.add_entry("Second message", "warning", "test")
            log.add_entry("Third message", "error", "processing")
            await pilot.pause()
            
            # Check entries were added
            assert len(log.entries) == 3
            assert log.entries[0].message == "Third message"  # Most recent first
            assert log.entries[1].message == "Second message"
            assert log.entries[2].message == "First message"
            
            # Check UI updated
            list_items = pilot.app.query(".log-entry")
            assert len(list_items) == 3
    
    @pytest.mark.asyncio
    async def test_max_entries_limit(self):
        """Test maximum entries limit."""
        log = ActivityLogWidget(max_entries=5)
        
        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add more than max entries
            for i in range(10):
                log.add_entry(f"Message {i}", "info")
                await pilot.pause(0.01)
            
            # Should only keep max_entries
            assert len(log.entries) == 5
            
            # Should have most recent entries
            assert log.entries[0].message == "Message 9"
            assert log.entries[4].message == "Message 5"
    
    @pytest.mark.asyncio
    async def test_level_filtering(self):
        """Test filtering by log level."""
        log = ActivityLogWidget()
        
        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add entries of different levels
            log.add_entry("Debug msg", "debug")
            log.add_entry("Info msg", "info")
            log.add_entry("Warning msg", "warning")
            log.add_entry("Error msg", "error")
            await pilot.pause()
            
            # Filter by warning level
            level_filter = pilot.app.query_one("#log-level-filter", Select)
            level_filter.value = "warning"
            await pilot.pause()
            
            # Trigger filter update
            log._apply_filters()
            await pilot.pause()
            
            # Check filtered entries
            visible_items = pilot.app.query(".log-entry:not(.hidden)")
            assert len(visible_items) == 2  # Warning and error
    
    @pytest.mark.asyncio
    async def test_category_filtering(self):
        """Test filtering by category."""
        log = ActivityLogWidget()
        
        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add entries with different categories
            log.add_entry("Model msg", "info", "model")
            log.add_entry("Processing msg", "info", "processing")
            log.add_entry("Another model msg", "info", "model")
            log.add_entry("Storage msg", "info", "storage")
            await pilot.pause()
            
            # Update categories in filter
            log._update_category_filter()
            await pilot.pause()
            
            # Filter by model category
            category_filter = pilot.app.query_one("#log-category-filter", Select)
            category_filter.value = "model"
            await pilot.pause()
            
            # Apply filters
            log._apply_filters()
            await pilot.pause()
            
            # Check filtered entries
            visible_entries = [e for e in log.entries if log._filter_entry(e)]
            assert len(visible_entries) == 2
            assert all(e.category == "model" for e in visible_entries)
    
    @pytest.mark.asyncio
    async def test_clear_log(self):
        """Test clearing the log."""
        log = ActivityLogWidget()
        
        # Track clear events
        clear_event_fired = False
        
        def on_clear(event):
            nonlocal clear_event_fired
            if isinstance(event, ActivityLogCleared):
                clear_event_fired = True
        
        app = WidgetTestApp(log)
        app.on_activity_log_cleared = on_clear
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add some entries
            for i in range(5):
                log.add_entry(f"Message {i}", "info")
            await pilot.pause()
            
            assert len(log.entries) == 5
            
            # Clear log
            await pilot.click("#clear-log")
            await pilot.pause()
            
            # Check log was cleared
            assert len(log.entries) == 0
            assert clear_event_fired == True
            
            # Check UI updated
            list_items = pilot.app.query(".log-entry")
            assert len(list_items) == 0
    
    @pytest.mark.asyncio
    async def test_export_log_json(self):
        """Test exporting log to JSON."""
        log = ActivityLogWidget()
        
        # Track export events
        export_path = None
        
        def on_export(event):
            nonlocal export_path
            if isinstance(event, ActivityLogExported):
                export_path = event.file_path
        
        app = WidgetTestApp(log)
        app.on_activity_log_exported = on_export
        
        # Mock file operations
        mock_file = MagicMock()
        
        with patch('builtins.open', mock_open()) as mock_open_func:
            mock_open_func.return_value = mock_file
            
            async with app.run_test() as pilot:
                await pilot.pause()
                
                # Add entries
                log.add_entry("Export test 1", "info", "test")
                log.add_entry("Export test 2", "warning", "test")
                await pilot.pause()
                
                # Export as JSON
                log.export_log("json")
                await pilot.pause()
                
                # Check file was written
                mock_open_func.assert_called()
                
                # Check JSON was written (through mock write calls)
                written_data = ''.join(
                    call.args[0] for call in mock_file.write.call_args_list
                )
                assert "Export test 1" in written_data
                assert "Export test 2" in written_data
    
    @pytest.mark.asyncio
    async def test_export_log_csv(self):
        """Test exporting log to CSV."""
        log = ActivityLogWidget()
        
        with patch('builtins.open', mock_open()) as mock_open_func:
            mock_file = mock_open_func.return_value
            
            app = WidgetTestApp(log)
            async with app.run_test() as pilot:
                await pilot.pause()
                
                # Add entries
                log.add_entry("CSV test", "error", "export")
                await pilot.pause()
                
                # Export as CSV
                log.export_log("csv")
                await pilot.pause()
                
                # Check CSV headers and data were written
                written_data = ''.join(
                    call.args[0] for call in mock_file.write.call_args_list
                )
                assert "timestamp,level,category,message" in written_data
                assert "CSV test" in written_data
                assert "error" in written_data
                assert "export" in written_data
    
    @pytest.mark.asyncio
    async def test_export_log_text(self):
        """Test exporting log to text."""
        log = ActivityLogWidget()
        
        with patch('builtins.open', mock_open()) as mock_open_func:
            mock_file = mock_open_func.return_value
            
            app = WidgetTestApp(log)
            async with app.run_test() as pilot:
                await pilot.pause()
                
                # Add entry with details
                log.add_entry(
                    "Text export test",
                    "info",
                    "export",
                    details={"key": "value"}
                )
                await pilot.pause()
                
                # Export as text
                log.export_log("text")
                await pilot.pause()
                
                # Check text was written
                written_data = ''.join(
                    call.args[0] for call in mock_file.write.call_args_list
                )
                assert "Text export test" in written_data
                assert "INFO" in written_data
                assert "export" in written_data
    
    @pytest.mark.asyncio
    async def test_log_entry_display(self):
        """Test log entry display formatting."""
        log = ActivityLogWidget()
        
        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add entries with different levels
            log.add_entry("Info message", "info")
            log.add_entry("Warning message", "warning")
            log.add_entry("Error message", "error")
            log.add_entry("Debug message", "debug")
            await pilot.pause()
            
            # Check entries have correct classes
            entries = pilot.app.query(".log-entry")
            
            # Should be in reverse order (most recent first)
            assert "log-debug" in entries[0].classes
            assert "log-error" in entries[1].classes
            assert "log-warning" in entries[2].classes
            assert "log-info" in entries[3].classes
    
    @pytest.mark.asyncio
    async def test_auto_scroll(self):
        """Test auto-scroll to latest entry."""
        log = ActivityLogWidget(auto_scroll=True)
        
        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add many entries to trigger scrolling
            for i in range(50):
                log.add_entry(f"Entry {i}", "info")
                await pilot.pause(0.01)
            
            # Latest entry should be visible
            # (In real test would check scroll position)
            assert len(log.entries) == 50
    
    @pytest.mark.asyncio
    async def test_entry_with_details(self):
        """Test log entries with additional details."""
        log = ActivityLogWidget()
        
        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add entry with details
            details = {
                "file": "test.txt",
                "size": 1024,
                "duration": 2.5
            }
            log.add_entry(
                "Processing complete",
                "success",
                "processing",
                details=details
            )
            await pilot.pause()
            
            # Check entry stored details
            assert log.entries[0].details == details
            
            # Details might be shown in tooltip or expanded view
            entry_items = pilot.app.query(".log-entry")
            first_entry = entry_items[0]
            
            # Could check for tooltip or expandable details
            assert first_entry is not None


class TestActivityLogIntegration(EmbeddingsTestBase):
    """Test activity log integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_real_time_updates(self):
        """Test real-time log updates from multiple sources."""
        log = ActivityLogWidget()
        
        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Simulate multiple async operations logging
            async def operation1():
                for i in range(3):
                    log.add_entry(f"Op1 step {i}", "info", "operation1")
                    await asyncio.sleep(0.05)
            
            async def operation2():
                for i in range(3):
                    log.add_entry(f"Op2 step {i}", "info", "operation2")
                    await asyncio.sleep(0.05)
            
            # Run operations concurrently
            await asyncio.gather(operation1(), operation2())
            await pilot.pause()
            
            # Should have all entries
            assert len(log.entries) == 6
            
            # Check both operations logged
            op1_entries = [e for e in log.entries if e.category == "operation1"]
            op2_entries = [e for e in log.entries if e.category == "operation2"]
            assert len(op1_entries) == 3
            assert len(op2_entries) == 3
    
    @pytest.mark.asyncio
    async def test_performance_with_many_entries(self):
        """Test performance with many log entries."""
        log = ActivityLogWidget(max_entries=1000)
        
        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add many entries quickly
            start_time = asyncio.get_event_loop().time()
            
            for i in range(100):
                log.add_entry(
                    f"Performance test {i}",
                    "info" if i % 2 == 0 else "debug",
                    f"category{i % 5}"
                )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Should complete quickly
            assert elapsed < 1.0  # Less than 1 second for 100 entries
            
            # Check entries
            assert len(log.entries) == 100
    
    @pytest.mark.asyncio
    async def test_export_filtered_entries(self):
        """Test exporting only filtered entries."""
        log = ActivityLogWidget()
        
        with patch('builtins.open', mock_open()) as mock_open_func:
            mock_file = mock_open_func.return_value
            
            app = WidgetTestApp(log)
            async with app.run_test() as pilot:
                await pilot.pause()
                
                # Add mixed entries
                log.add_entry("Keep 1", "error", "important")
                log.add_entry("Skip 1", "debug", "verbose")
                log.add_entry("Keep 2", "warning", "important")
                log.add_entry("Skip 2", "info", "verbose")
                await pilot.pause()
                
                # Apply filter
                log.level_filter = "warning"
                log._apply_filters()
                await pilot.pause()
                
                # Export (should only export visible entries)
                log.export_log("json", filtered_only=True)
                await pilot.pause()
                
                # Check only filtered entries were exported
                written_data = ''.join(
                    call.args[0] for call in mock_file.write.call_args_list
                )
                assert "Keep 1" in written_data
                assert "Keep 2" in written_data
                assert "Skip 1" not in written_data
                assert "Skip 2" not in written_data