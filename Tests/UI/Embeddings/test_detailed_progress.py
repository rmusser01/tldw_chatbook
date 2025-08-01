"""
Tests for Detailed Progress Widget.
Tests multi-stage progress tracking, pause/resume, and performance metrics.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from textual.widgets import Button, Static, ProgressBar

from tldw_chatbook.Widgets.detailed_progress import (
    DetailedProgressBar,
    ProgressStage
)

from .test_base import EmbeddingsTestBase, WidgetTestApp


class TestDetailedProgressBar(EmbeddingsTestBase):
    """Test DetailedProgressBar functionality."""
    
    @pytest.mark.asyncio
    async def test_progress_bar_creation(self):
        """Test creating progress bar with stages."""
        stages = ["Preprocessing", "Embedding", "Storing"]
        progress = DetailedProgressBar(
            stages=stages,
            show_memory=True,
            show_speed=True,
            pausable=True
        )
        
        assert len(progress.stages) == 3
        assert progress.stages[0].name == "Preprocessing"
        assert progress.show_memory == True
        assert progress.show_speed == True
        assert progress.pausable == True
    
    @pytest.mark.asyncio
    async def test_progress_bar_compose(self):
        """Test progress bar UI composition."""
        stages = ["Stage 1", "Stage 2"]
        progress = DetailedProgressBar(stages=stages)
        
        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Check stage indicator
            stage_indicator = pilot.app.query_one("#stage-indicator", Static)
            assert stage_indicator is not None
            
            # Check current item display
            current_item = pilot.app.query_one("#current-item", Static)
            assert current_item is not None
            
            # Check progress bars
            main_progress = pilot.app.query_one("#main-progress", ProgressBar)
            assert main_progress is not None
            
            stage_progress = pilot.app.query_one("#stage-progress", ProgressBar)
            assert stage_progress is not None
            
            # Check pause button (if pausable)
            pause_button = pilot.app.query_one("#pause-button", Button)
            assert pause_button is not None
            assert pause_button.label == "⏸ Pause"
    
    @pytest.mark.asyncio
    async def test_stage_progression(self):
        """Test progressing through stages."""
        stages = ["Download", "Process", "Upload"]
        progress = DetailedProgressBar(stages=stages)
        
        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Start first stage
            progress.start_stage(0, total_items=100)
            await pilot.pause()
            
            stage_indicator = pilot.app.query_one("#stage-indicator", Static)
            assert "1/3: Download" in stage_indicator.renderable
            
            # Update progress
            progress.update_progress(items=50, current_item="file1.txt")
            await pilot.pause()
            
            assert progress.stages[0].current == 50
            
            # Complete stage and move to next
            progress.complete_stage()
            await pilot.pause()
            
            assert progress.stages[0].completed == True
            assert progress.current_stage == 1
            assert "2/3: Process" in stage_indicator.renderable
    
    @pytest.mark.asyncio
    async def test_pause_resume_functionality(self):
        """Test pause and resume functionality."""
        progress = DetailedProgressBar(
            stages=["Processing"],
            pausable=True
        )
        
        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Start processing
            progress.start_stage(0, total_items=100)
            
            # Click pause button
            await pilot.click("#pause-button")
            await pilot.pause()
            
            assert progress.is_paused == True
            pause_button = pilot.app.query_one("#pause-button", Button)
            assert pause_button.label == "▶ Resume"
            
            # Try to update progress while paused (should be ignored)
            progress.update_progress(items=10)
            assert progress.stages[0].current == 0
            
            # Resume
            await pilot.click("#pause-button")
            await pilot.pause()
            
            assert progress.is_paused == False
            assert pause_button.label == "⏸ Pause"
            
            # Now updates should work
            progress.update_progress(items=10)
            assert progress.stages[0].current == 10
    
    @pytest.mark.asyncio
    async def test_progress_calculations(self):
        """Test progress percentage calculations."""
        stages = [
            ("Stage1", 1.0),  # weight 1
            ("Stage2", 2.0),  # weight 2
            ("Stage3", 1.0),  # weight 1
        ]
        progress = DetailedProgressBar(
            stages=[s[0] for s in stages]
        )
        
        # Set weights
        for i, (_, weight) in enumerate(stages):
            progress.stages[i].weight = weight
        
        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Complete first stage (25% of total weight)
            progress.start_stage(0, 100)
            progress.update_progress(100)
            progress.complete_stage()
            await pilot.pause()
            
            # Check overall progress
            main_bar = pilot.app.query_one("#main-progress", ProgressBar)
            assert main_bar.percentage == 25.0
            
            # Half complete second stage (25% more)
            progress.update_progress(50)
            await pilot.pause()
            
            assert main_bar.percentage == 50.0
    
    @pytest.mark.asyncio
    async def test_speed_metrics(self):
        """Test speed and throughput metrics."""
        progress = DetailedProgressBar(
            stages=["Processing"],
            show_speed=True
        )
        
        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            progress.start_stage(0, 1000)
            
            # Simulate processing with known timing
            start_time = time.time()
            progress.update_progress(items=100, bytes_count=1024*1024)  # 1MB
            time.sleep(0.1)  # Simulate processing time
            progress.update_progress(items=100, bytes_count=1024*1024)  # 1MB
            
            await pilot.pause()
            
            # Check speed metrics
            speed_metric = pilot.app.query_one("#speed-metric", Static)
            # Should show items/sec and MB/s
            assert "items/sec" in speed_metric.renderable
            assert progress.items_processed == 200
    
    @pytest.mark.asyncio
    async def test_time_estimation(self):
        """Test elapsed and remaining time estimation."""
        progress = DetailedProgressBar(stages=["Processing"])
        
        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            progress.start_stage(0, 1000)
            
            # Process some items
            progress.update_progress(100)
            await pilot.pause()
            
            # Check elapsed time
            elapsed = pilot.app.query_one("#elapsed-time", Static)
            assert elapsed.renderable != ""
            
            # Check remaining time estimation
            remaining = pilot.app.query_one("#remaining-time", Static)
            # Should show time or "Calculating..."
            assert remaining.renderable != ""
    
    @pytest.mark.asyncio
    async def test_memory_tracking(self):
        """Test memory usage tracking."""
        with patch('psutil.Process') as mock_process:
            # Mock memory info
            process_instance = MagicMock()
            process_instance.memory_info.return_value = MagicMock(rss=1024*1024*256)  # 256MB
            mock_process.return_value = process_instance
            
            progress = DetailedProgressBar(
                stages=["Processing"],
                show_memory=True
            )
            
            app = WidgetTestApp(progress)
            async with app.run_test() as pilot:
                await pilot.pause()
                
                # Trigger metrics update
                progress._update_metrics()
                await pilot.pause()
                
                # Check memory display
                memory_usage = pilot.app.query_one("#memory-usage", Static)
                assert "MB" in memory_usage.renderable
    
    @pytest.mark.asyncio
    async def test_current_item_display(self):
        """Test current item display and truncation."""
        progress = DetailedProgressBar(stages=["Processing"])
        
        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            progress.start_stage(0, 100)
            
            # Short item name
            progress.update_progress(1, current_item="file.txt")
            await pilot.pause()
            
            current_item = pilot.app.query_one("#current-item", Static)
            assert current_item.renderable == "file.txt"
            
            # Long item name (should be truncated)
            long_name = "a" * 100
            progress.update_progress(1, current_item=long_name)
            await pilot.pause()
            
            assert len(current_item.renderable) <= 53  # 50 + "..."
            assert current_item.renderable.endswith("...")
    
    @pytest.mark.asyncio
    async def test_multi_stage_complete(self):
        """Test completing all stages."""
        stages = ["Stage1", "Stage2", "Stage3"]
        progress = DetailedProgressBar(stages=stages)
        
        completed = False
        
        # Mock _on_complete
        original_complete = progress._on_complete
        def mock_complete():
            nonlocal completed
            completed = True
            original_complete()
        progress._on_complete = mock_complete
        
        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Complete all stages
            for i in range(3):
                progress.start_stage(i, 100)
                progress.update_progress(100)
                progress.complete_stage()
                await pilot.pause()
            
            # Should trigger completion
            assert completed == True
            
            # Remaining time should show "Complete!"
            remaining = pilot.app.query_one("#remaining-time", Static)
            assert remaining.renderable == "Complete!"
    
    @pytest.mark.asyncio
    async def test_pause_time_tracking(self):
        """Test that pause time is excluded from calculations."""
        progress = DetailedProgressBar(
            stages=["Processing"],
            pausable=True
        )
        
        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            progress.start_stage(0, 100)
            start_time = datetime.now()
            
            # Process some items
            progress.update_progress(25)
            await pilot.pause()
            
            # Pause for a bit
            progress.pause()
            await asyncio.sleep(0.5)
            
            # Resume and process more
            progress.resume()
            progress.update_progress(25)
            
            # Total pause duration should be tracked
            assert progress.total_pause_duration.total_seconds() >= 0.5
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in progress updates."""
        progress = DetailedProgressBar(stages=["Processing"])
        
        app = WidgetTestApp(progress)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Try to update without starting stage
            progress.update_progress(10)  # Should not crash
            
            # Start invalid stage
            progress.start_stage(99, 100)  # Out of range
            
            # Complete without starting
            progress.complete_stage()  # Should handle gracefully
            
            # App should still be running
            assert pilot.app is not None