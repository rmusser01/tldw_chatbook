# tldw_chatbook/Widgets/detailed_progress.py
# Enhanced progress tracking for long operations
#
# Imports
from __future__ import annotations
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

# Third-party imports
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Static, Label, ProgressBar, Button
from textual.widget import Widget
from textual.reactive import reactive
from textual.timer import Timer
from loguru import logger

# Configure logger
logger = logger.bind(module="detailed_progress")


@dataclass
class ProgressStage:
    """Represents a stage in a multi-stage operation."""
    name: str
    weight: float = 1.0  # Relative weight for overall progress
    completed: bool = False
    current: int = 0
    total: int = 100


class DetailedProgressBar(Widget):
    """Enhanced progress bar with detailed tracking.
    
    Features:
    - Multi-stage progress tracking
    - Speed/throughput metrics
    - Time estimation
    - Memory usage tracking
    - Pause/resume capability
    """
    
    DEFAULT_CLASSES = "detailed-progress-bar"
    
    # Reactive properties
    current_stage: reactive[int] = reactive(0)
    is_paused: reactive[bool] = reactive(False)
    current_item: reactive[str] = reactive("")
    
    def __init__(
        self,
        stages: Optional[List[str]] = None,
        show_memory: bool = True,
        show_speed: bool = True,
        pausable: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stages = [ProgressStage(name=s) for s in (stages or ["Processing"])]
        self.show_memory = show_memory
        self.show_speed = show_speed
        self.pausable = pausable
        
        # Timing and metrics
        self.start_time: Optional[datetime] = None
        self.pause_time: Optional[datetime] = None
        self.total_pause_duration = timedelta()
        self.items_processed = 0
        self.bytes_processed = 0
        self.last_update_time = time.time()
        self.speed_history: List[float] = []  # Rolling average
        self.update_timer: Optional[Timer] = None
        
    def compose(self) -> ComposeResult:
        """Compose the detailed progress display."""
        with Vertical(classes="detailed-progress-container"):
            # Stage indicator
            with Horizontal(classes="progress-stage-row"):
                yield Label("Stage:", classes="progress-label")
                yield Static("", id="stage-indicator", classes="stage-indicator")
                if self.pausable:
                    yield Button(
                        "⏸ Pause" if not self.is_paused else "▶ Resume",
                        id="pause-button",
                        classes="pause-button"
                    )
            
            # Current item being processed
            with Horizontal(classes="progress-item-row"):
                yield Label("Processing:", classes="progress-label")
                yield Static("", id="current-item", classes="current-item")
            
            # Main progress bar
            yield ProgressBar(
                id="main-progress",
                show_eta=False,
                show_percentage=True
            )
            
            # Sub-progress for current stage
            yield ProgressBar(
                id="stage-progress",
                show_eta=False,
                show_percentage=True,
                classes="stage-progress"
            )
            
            # Metrics row
            with Horizontal(classes="progress-metrics-row"):
                # Speed indicator
                if self.show_speed:
                    with Vertical(classes="metric-group"):
                        yield Label("Speed:", classes="metric-label")
                        yield Static("0 items/sec", id="speed-metric", classes="metric-value")
                
                # Time elapsed/remaining
                with Vertical(classes="metric-group"):
                    yield Label("Elapsed:", classes="metric-label")
                    yield Static("00:00:00", id="elapsed-time", classes="metric-value")
                
                with Vertical(classes="metric-group"):
                    yield Label("Remaining:", classes="metric-label")
                    yield Static("Calculating...", id="remaining-time", classes="metric-value")
                
                # Memory usage
                if self.show_memory:
                    with Vertical(classes="metric-group"):
                        yield Label("Memory:", classes="metric-label")
                        yield Static("0 MB", id="memory-usage", classes="metric-value")
    
    def on_mount(self) -> None:
        """Start tracking when mounted."""
        self.start_time = datetime.now()
        self.update_timer = self.set_timer(1.0, self._update_metrics, pause=False)
    
    def start_stage(self, stage_index: int, total_items: int = 100) -> None:
        """Start a new stage of processing."""
        if 0 <= stage_index < len(self.stages):
            self.current_stage = stage_index
            stage = self.stages[stage_index]
            stage.current = 0
            stage.total = total_items
            stage.completed = False
            
            # Update stage indicator
            self._update_stage_display()
    
    def update_progress(
        self,
        items: int = 1,
        current_item: Optional[str] = None,
        bytes_count: Optional[int] = None
    ) -> None:
        """Update progress for current stage."""
        if self.is_paused:
            return
            
        stage = self.stages[self.current_stage]
        stage.current = min(stage.current + items, stage.total)
        self.items_processed += items
        
        if bytes_count:
            self.bytes_processed += bytes_count
        
        if current_item:
            self.current_item = current_item
            self._update_current_item_display()
        
        # Update progress bars
        self._update_progress_bars()
        
        # Track speed
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        if time_delta > 0:
            speed = items / time_delta
            self.speed_history.append(speed)
            # Keep last 10 measurements for rolling average
            self.speed_history = self.speed_history[-10:]
        self.last_update_time = current_time
    
    def complete_stage(self) -> None:
        """Mark current stage as complete."""
        if self.current_stage < len(self.stages):
            self.stages[self.current_stage].completed = True
            self.stages[self.current_stage].current = self.stages[self.current_stage].total
            
            # Auto-advance to next stage if available
            if self.current_stage + 1 < len(self.stages):
                self.start_stage(self.current_stage + 1)
            else:
                # All stages complete
                self._on_complete()
    
    def pause(self) -> None:
        """Pause the operation."""
        if not self.is_paused and self.pausable:
            self.is_paused = True
            self.pause_time = datetime.now()
            self._update_pause_button()
    
    def resume(self) -> None:
        """Resume the operation."""
        if self.is_paused and self.pause_time:
            self.is_paused = False
            self.total_pause_duration += datetime.now() - self.pause_time
            self.pause_time = None
            self._update_pause_button()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle pause/resume button."""
        if event.button.id == "pause-button":
            if self.is_paused:
                self.resume()
            else:
                self.pause()
    
    def _update_stage_display(self) -> None:
        """Update stage indicator."""
        try:
            indicator = self.query_one("#stage-indicator", Static)
            stage = self.stages[self.current_stage]
            stage_text = f"{self.current_stage + 1}/{len(self.stages)}: {stage.name}"
            indicator.update(stage_text)
        except Exception as e:
            logger.error(f"Error updating stage display: {e}")
    
    def _update_current_item_display(self) -> None:
        """Update current item display."""
        try:
            item_display = self.query_one("#current-item", Static)
            # Truncate long items
            display_text = self.current_item[:50] + "..." if len(self.current_item) > 50 else self.current_item
            item_display.update(display_text)
        except Exception as e:
            logger.error(f"Error updating current item: {e}")
    
    def _update_progress_bars(self) -> None:
        """Update both progress bars."""
        try:
            # Calculate overall progress
            total_weight = sum(s.weight for s in self.stages)
            completed_weight = sum(
                s.weight * (s.current / s.total if s.total > 0 else 0)
                for s in self.stages
            )
            overall_progress = (completed_weight / total_weight) * 100 if total_weight > 0 else 0
            
            # Update main progress
            main_bar = self.query_one("#main-progress", ProgressBar)
            main_bar.update(progress=overall_progress)
            
            # Update stage progress
            stage = self.stages[self.current_stage]
            stage_progress = (stage.current / stage.total * 100) if stage.total > 0 else 0
            stage_bar = self.query_one("#stage-progress", ProgressBar)
            stage_bar.update(progress=stage_progress)
            
        except Exception as e:
            logger.error(f"Error updating progress bars: {e}")
    
    def _update_pause_button(self) -> None:
        """Update pause button text."""
        try:
            button = self.query_one("#pause-button", Button)
            button.label = "▶ Resume" if self.is_paused else "⏸ Pause"
        except Exception:
            pass
    
    def _update_metrics(self) -> None:
        """Update all metrics displays."""
        if not self.start_time:
            return
            
        try:
            # Calculate elapsed time
            elapsed = datetime.now() - self.start_time - self.total_pause_duration
            elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
            self.query_one("#elapsed-time", Static).update(elapsed_str)
            
            # Calculate speed
            if self.show_speed and self.speed_history:
                avg_speed = sum(self.speed_history) / len(self.speed_history)
                speed_text = f"{avg_speed:.1f} items/sec"
                if self.bytes_processed > 0:
                    mb_per_sec = (self.bytes_processed / 1024 / 1024) / elapsed.total_seconds()
                    speed_text += f" ({mb_per_sec:.1f} MB/s)"
                self.query_one("#speed-metric", Static).update(speed_text)
            
            # Estimate remaining time
            if self.items_processed > 0:
                # Calculate total items across all stages
                total_items = sum(s.total for s in self.stages)
                total_processed = sum(s.current for s in self.stages)
                
                if total_processed > 0:
                    rate = elapsed.total_seconds() / total_processed
                    remaining_items = total_items - total_processed
                    remaining_seconds = remaining_items * rate
                    remaining_time = timedelta(seconds=int(remaining_seconds))
                    self.query_one("#remaining-time", Static).update(str(remaining_time))
            
            # Update memory usage
            if self.show_memory:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.query_one("#memory-usage", Static).update(f"{memory_mb:.1f} MB")
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _on_complete(self) -> None:
        """Handle completion of all stages."""
        if self.update_timer:
            self.update_timer.stop()
        
        # Final metrics update
        self._update_metrics()
        
        # Update remaining time to show completion
        try:
            self.query_one("#remaining-time", Static).update("Complete!")
        except:
            pass