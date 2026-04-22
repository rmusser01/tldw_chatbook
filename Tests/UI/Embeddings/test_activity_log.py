"""
Tests for the embeddings activity log widget.

This suite targets the current widget contract rather than older list-based
implementations that no longer exist in the app.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, mock_open, patch

import pytest
from textual.containers import Container
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Widgets.activity_log import (
    ActivityEntry,
    ActivityLogWidget,
)

from .test_base import EmbeddingsTestBase, WidgetTestApp


def _written_data(mock_open_func: MagicMock) -> str:
    """Collect all text written through a mocked file handle."""
    handle = mock_open_func.return_value.__enter__.return_value
    return "".join(call.args[0] for call in handle.write.call_args_list)


class TestActivityEntry(EmbeddingsTestBase):
    """Test ActivityEntry dataclass behavior."""

    def test_entry_creation(self):
        entry = ActivityEntry(
            timestamp=datetime.now(),
            level="info",
            category="general",
            message="Test message",
        )

        assert entry.message == "Test message"
        assert entry.level == "info"
        assert entry.category == "general"

    def test_entry_to_dict(self):
        entry = ActivityEntry(
            timestamp=datetime.now(),
            level="warning",
            category="export",
            message="Export test",
            details={"count": 10},
        )

        data = entry.to_dict()
        assert data["message"] == "Export test"
        assert data["level"] == "warning"
        assert data["category"] == "export"
        assert data["details"]["count"] == 10

    def test_entry_to_string(self):
        entry = ActivityEntry(
            timestamp=datetime.now(),
            level="debug",
            category="test",
            message="String test",
            details={"mode": "verbose"},
        )

        string_repr = entry.to_string()
        assert "String test" in string_repr
        assert "DEBUG" in string_repr
        assert "test" in string_repr
        assert "verbose" in string_repr


class TestActivityLogWidget(EmbeddingsTestBase):
    """Test ActivityLogWidget behavior."""

    @pytest.mark.asyncio
    async def test_widget_creation(self):
        log = ActivityLogWidget(
            max_entries=100,
            show_filters=True,
            show_export=True,
            auto_scroll=False,
        )

        assert log.max_entries == 100
        assert log.show_filters is True
        assert log.show_export is True
        assert log.auto_scroll is False
        assert len(log.entries) == 0

    @pytest.mark.asyncio
    async def test_widget_compose(self):
        log = ActivityLogWidget()

        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            assert pilot.app.query_one(".activity-log-header") is not None
            assert pilot.app.query_one("#log-level-filter", Select) is not None
            assert pilot.app.query_one("#log-category-filter", Select) is not None
            assert pilot.app.query_one("#search-logs", Input) is not None
            assert pilot.app.query_one("#clear-log", Button).label == "Clear Log"
            assert pilot.app.query_one("#export-log", Button).label == "Export"
            assert pilot.app.query_one("#log-entries", Container) is not None

    @pytest.mark.asyncio
    async def test_add_entry_displays_newest_first(self):
        log = ActivityLogWidget()

        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            log.add_entry("First message", "info", "test")
            log.add_entry("Second message", "warning", "test")
            log.add_entry("Third message", "error", "processing")
            await pilot.pause()

            assert len(log.entries) == 3
            assert log.entries[0].message == "Third message"
            assert log.entries[1].message == "Second message"
            assert log.entries[2].message == "First message"
            assert len(pilot.app.query(".log-entry")) == 3

    @pytest.mark.asyncio
    async def test_max_entries_limit(self):
        log = ActivityLogWidget(max_entries=5)

        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            for i in range(10):
                log.add_entry(f"Message {i}", "info")
                await pilot.pause(0.01)

            assert len(log.entries) == 5
            assert log.entries[0].message == "Message 9"
            assert log.entries[4].message == "Message 5"

    @pytest.mark.asyncio
    async def test_level_filtering(self):
        log = ActivityLogWidget()

        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            log.add_entry("Debug msg", "debug")
            log.add_entry("Info msg", "info")
            log.add_entry("Warning msg", "warning")
            log.add_entry("Error msg", "error")
            await pilot.pause()

            log.filter_level = "warning"
            log._apply_filters()
            await pilot.pause()

            visible_entries = log._get_filtered_entries()
            assert len(visible_entries) == 1
            assert visible_entries[0].message == "Warning msg"
            assert len(pilot.app.query(".log-entry")) == 1

    @pytest.mark.asyncio
    async def test_category_filtering(self):
        log = ActivityLogWidget()

        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            log.add_entry("Model msg", "info", "model")
            log.add_entry("Processing msg", "info", "processing")
            log.add_entry("Another model msg", "info", "model")
            log.add_entry("Storage msg", "info", "storage")
            await pilot.pause()

            log._update_category_filter()
            log.filter_category = "model"
            log._apply_filters()
            await pilot.pause()

            visible_entries = log._get_filtered_entries()
            assert len(visible_entries) == 2
            assert all(entry.category == "model" for entry in visible_entries)

    @pytest.mark.asyncio
    async def test_clear_log(self):
        log = ActivityLogWidget()

        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            for i in range(5):
                log.add_entry(f"Message {i}", "info")
            await pilot.pause()

            log.clear_log()
            await pilot.pause()

            assert len(log.entries) == 0
            assert len(pilot.app.query(".log-entry")) == 0

    @pytest.mark.asyncio
    async def test_export_log_json(self):
        log = ActivityLogWidget()

        with patch("builtins.open", mock_open()) as mocked_open:
            app = WidgetTestApp(log)
            async with app.run_test() as pilot:
                await pilot.pause()

                log.add_entry("Export test 1", "info", "test")
                log.add_entry("Export test 2", "warning", "test")
                await pilot.pause()

                log.export_log("json")
                await pilot.pause()

        written_data = _written_data(mocked_open)
        assert "Export test 1" in written_data
        assert "Export test 2" in written_data
        assert '"entries"' in written_data

    @pytest.mark.asyncio
    async def test_export_log_csv(self):
        log = ActivityLogWidget()

        with patch("builtins.open", mock_open()) as mocked_open:
            app = WidgetTestApp(log)
            async with app.run_test() as pilot:
                await pilot.pause()

                log.add_entry("CSV test", "error", "export")
                await pilot.pause()

                log.export_log("csv")
                await pilot.pause()

        written_data = _written_data(mocked_open)
        assert "timestamp,level,category,message,details" in written_data
        assert "CSV test" in written_data
        assert "error" in written_data
        assert "export" in written_data

    @pytest.mark.asyncio
    async def test_export_log_text(self):
        log = ActivityLogWidget()

        with patch("builtins.open", mock_open()) as mocked_open:
            app = WidgetTestApp(log)
            async with app.run_test() as pilot:
                await pilot.pause()

                log.add_entry(
                    "Text export test",
                    "info",
                    "export",
                    details={"key": "value"},
                )
                await pilot.pause()

                log.export_log("text")
                await pilot.pause()

        written_data = _written_data(mocked_open)
        assert "Text export test" in written_data
        assert "INFO" in written_data
        assert "export" in written_data
        assert "value" in written_data

    @pytest.mark.asyncio
    async def test_log_entry_display_classes(self):
        log = ActivityLogWidget()

        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            log.add_entry("Info message", "info")
            log.add_entry("Warning message", "warning")
            log.add_entry("Error message", "error")
            log.add_entry("Debug message", "debug")
            await pilot.pause()

            entries = list(pilot.app.query(".log-entry"))
            assert "log-debug" in entries[0].classes
            assert "log-error" in entries[1].classes
            assert "log-warning" in entries[2].classes
            assert "log-info" in entries[3].classes

    @pytest.mark.asyncio
    async def test_entry_with_details_is_stored_and_rendered(self):
        log = ActivityLogWidget()

        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            details = {"file": "test.txt", "size": 1024, "duration": 2.5}
            log.add_entry(
                "Processing complete",
                "success",
                "processing",
                details=details,
            )
            await pilot.pause()

            assert log.entries[0].details == details
            message = pilot.app.query_one(".log-message", Static)
            assert "test.txt" in str(message.render())


class TestActivityLogIntegration(EmbeddingsTestBase):
    """Integration coverage for current activity-log behavior."""

    @pytest.mark.asyncio
    async def test_real_time_updates(self):
        log = ActivityLogWidget()

        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            async def operation1():
                for i in range(3):
                    log.add_entry(f"Op1 step {i}", "info", "operation1")
                    await asyncio.sleep(0.05)

            async def operation2():
                for i in range(3):
                    log.add_entry(f"Op2 step {i}", "info", "operation2")
                    await asyncio.sleep(0.05)

            await asyncio.gather(operation1(), operation2())
            await pilot.pause()

            assert len(log.entries) == 6
            assert len([e for e in log.entries if e.category == "operation1"]) == 3
            assert len([e for e in log.entries if e.category == "operation2"]) == 3

    @pytest.mark.asyncio
    async def test_performance_with_many_entries(self):
        log = ActivityLogWidget(max_entries=1000)

        app = WidgetTestApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            start_time = asyncio.get_event_loop().time()
            for i in range(100):
                log.add_entry(
                    f"Performance test {i}",
                    "info" if i % 2 == 0 else "debug",
                    f"category{i % 5}",
                )
            elapsed = asyncio.get_event_loop().time() - start_time

            assert elapsed < 1.0
            assert len(log.entries) == 100

    @pytest.mark.asyncio
    async def test_export_filtered_entries(self):
        log = ActivityLogWidget()

        with patch("builtins.open", mock_open()) as mocked_open:
            app = WidgetTestApp(log)
            async with app.run_test() as pilot:
                await pilot.pause()

                log.add_entry("Keep 1", "error", "important")
                log.add_entry("Skip 1", "debug", "verbose")
                log.add_entry("Keep 2", "warning", "important")
                log.add_entry("Skip 2", "info", "verbose")
                await pilot.pause()

                log.filter_category = "important"
                log._apply_filters()
                await pilot.pause()

                log.export_log("json", filtered_only=True)
                await pilot.pause()

        written_data = _written_data(mocked_open)
        assert "Keep 1" in written_data
        assert "Keep 2" in written_data
        assert "Skip 1" not in written_data
        assert "Skip 2" not in written_data
