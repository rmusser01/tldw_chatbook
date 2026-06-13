"""
Tests for the detailed progress widget.

These assertions follow the current Textual widget surface instead of older
tests that relied on deprecated `Static.renderable` access.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Button, ProgressBar, Static

from tldw_chatbook.Widgets.detailed_progress import DetailedProgressBar

from .test_base import EmbeddingsTestBase, WidgetTestApp


def _static_text(widget: Static) -> str:
    """Return the rendered text for a Static widget."""
    return str(widget.render())


class TestDetailedProgressBar(EmbeddingsTestBase):
    """Test current DetailedProgressBar behavior."""

    @pytest.mark.asyncio
    async def test_progress_bar_creation(self):
        progress = DetailedProgressBar(
            stages=["Preprocessing", "Embedding", "Storing"],
            show_memory=True,
            show_speed=True,
            pausable=True,
        )

        assert len(progress.stages) == 3
        assert progress.stages[0].name == "Preprocessing"
        assert progress.show_memory is True
        assert progress.show_speed is True
        assert progress.pausable is True

    @pytest.mark.asyncio
    async def test_progress_bar_compose(self):
        progress = DetailedProgressBar(stages=["Stage 1", "Stage 2"])

        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()

            assert pilot.app.query_one("#stage-indicator", Static) is not None
            assert pilot.app.query_one("#current-item", Static) is not None
            assert pilot.app.query_one("#main-progress", ProgressBar) is not None
            assert pilot.app.query_one("#stage-progress", ProgressBar) is not None
            assert pilot.app.query_one("#pause-button", Button).label == "⏸ Pause"

    @pytest.mark.asyncio
    async def test_stage_progression(self):
        progress = DetailedProgressBar(stages=["Download", "Process", "Upload"])

        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()

            progress.start_stage(0, total_items=100)
            await pilot.pause()

            stage_indicator = pilot.app.query_one("#stage-indicator", Static)
            assert "1/3: Download" in _static_text(stage_indicator)

            progress.update_progress(items=50, current_item="file1.txt")
            await pilot.pause()

            assert progress.stages[0].current == 50

            progress.complete_stage()
            await pilot.pause()

            assert progress.stages[0].completed is True
            assert progress.current_stage == 1
            assert "2/3: Process" in _static_text(stage_indicator)

    @pytest.mark.asyncio
    async def test_pause_resume_functionality(self):
        progress = DetailedProgressBar(stages=["Processing"], pausable=True)

        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()

            progress.start_stage(0, total_items=100)
            progress.pause()
            await pilot.pause()

            assert progress.is_paused is True
            assert pilot.app.query_one("#pause-button", Button).label == "▶ Resume"

            progress.update_progress(items=10)
            assert progress.stages[0].current == 0

            progress.resume()
            await pilot.pause()

            assert progress.is_paused is False
            assert pilot.app.query_one("#pause-button", Button).label == "⏸ Pause"

            progress.update_progress(items=10)
            assert progress.stages[0].current == 10

    @pytest.mark.asyncio
    async def test_progress_calculations(self):
        progress = DetailedProgressBar(stages=["Stage1", "Stage2", "Stage3"])
        progress.stages[0].weight = 1.0
        progress.stages[1].weight = 2.0
        progress.stages[2].weight = 1.0

        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()

            progress.start_stage(0, 100)
            progress.update_progress(100)
            progress.complete_stage()
            await pilot.pause()

            main_bar = pilot.app.query_one("#main-progress", ProgressBar)
            assert main_bar.percentage == pytest.approx(0.25)

            progress.update_progress(50)
            await pilot.pause()
            assert main_bar.percentage == pytest.approx(0.50)

    @pytest.mark.asyncio
    async def test_speed_metrics(self):
        progress = DetailedProgressBar(stages=["Processing"], show_speed=True)

        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()

            progress.start_stage(0, 1000)
            progress.update_progress(items=100, bytes_count=1024 * 1024)
            time.sleep(0.1)
            progress.update_progress(items=100, bytes_count=1024 * 1024)
            progress._update_metrics()
            await pilot.pause()

            speed_metric = pilot.app.query_one("#speed-metric", Static)
            assert "items/sec" in _static_text(speed_metric)
            assert progress.items_processed == 200

    @pytest.mark.asyncio
    async def test_time_estimation(self):
        progress = DetailedProgressBar(stages=["Processing"])

        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()

            progress.start_stage(0, 1000)
            progress.update_progress(100)
            progress._update_metrics()
            await pilot.pause()

            elapsed = pilot.app.query_one("#elapsed-time", Static)
            remaining = pilot.app.query_one("#remaining-time", Static)
            assert _static_text(elapsed) != ""
            assert _static_text(remaining) != ""

    @pytest.mark.asyncio
    async def test_memory_tracking(self):
        with patch("psutil.Process") as mock_process:
            process_instance = MagicMock()
            process_instance.memory_info.return_value = MagicMock(rss=1024 * 1024 * 256)
            mock_process.return_value = process_instance

            progress = DetailedProgressBar(stages=["Processing"], show_memory=True)

            app = WidgetTestApp(progress)
            async with app.run_test() as pilot:
                await pilot.pause()

                progress._update_metrics()
                await pilot.pause()

                memory_usage = pilot.app.query_one("#memory-usage", Static)
                assert "MB" in _static_text(memory_usage)

    @pytest.mark.asyncio
    async def test_current_item_display(self):
        progress = DetailedProgressBar(stages=["Processing"])

        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()

            progress.start_stage(0, 100)
            progress.update_progress(1, current_item="file.txt")
            await pilot.pause()

            current_item = pilot.app.query_one("#current-item", Static)
            assert _static_text(current_item) == "file.txt"

            long_name = "a" * 100
            progress.update_progress(1, current_item=long_name)
            await pilot.pause()

            displayed = _static_text(current_item)
            assert len(displayed) <= 53
            assert displayed.endswith("...")

    @pytest.mark.asyncio
    async def test_multi_stage_complete(self):
        progress = DetailedProgressBar(stages=["Stage1", "Stage2", "Stage3"])

        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()

            for i in range(3):
                progress.start_stage(i, 100)
                progress.update_progress(100)
                progress.complete_stage()
                await pilot.pause()

            remaining = pilot.app.query_one("#remaining-time", Static)
            assert _static_text(remaining) == "Complete!"

    @pytest.mark.asyncio
    async def test_pause_time_tracking(self):
        progress = DetailedProgressBar(stages=["Processing"], pausable=True)

        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()

            progress.start_stage(0, 100)
            progress.update_progress(25)
            await pilot.pause()

            progress.pause()
            await asyncio.sleep(0.5)
            progress.resume()
            progress.update_progress(25)

            assert progress.total_pause_duration.total_seconds() >= 0.5

    @pytest.mark.asyncio
    async def test_error_handling(self):
        progress = DetailedProgressBar(stages=["Processing"])

        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()

            progress.update_progress(10)
            progress.start_stage(99, 100)
            progress.complete_stage()

            assert pilot.app is not None
