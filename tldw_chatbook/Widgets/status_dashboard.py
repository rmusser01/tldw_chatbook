# tldw_chatbook/Widgets/status_dashboard.py
# Status dashboard widget for unified status handling in ingestion forms

from typing import Optional, List, Dict, Any
from datetime import datetime
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, ProgressBar, Label, Button
from textual.widget import Widget
from textual.reactive import reactive
from textual.timer import Timer
from loguru import logger

class StatusDashboard(Widget):
    """
    A unified status dashboard widget for tracking ingestion progress.
    
    Features:
    - Status message display
    - Progress bar with percentage
    - File counter (X of Y files)
    - Time elapsed/estimated
    - Error/warning display
    - Cancel/retry buttons
    """
    
    # Reactive properties
    status_message = reactive("Ready")
    progress_value = reactive(0.0)
    progress_total = reactive(100.0)
    current_file = reactive(0)
    total_files = reactive(0)
    is_processing = reactive(False)
    has_errors = reactive(False)
    
    def __init__(
        self, 
        id: Optional[str] = None,
        classes: Optional[str] = None,
        show_time: bool = True,
        show_file_counter: bool = True,
        show_actions: bool = True
    ):
        super().__init__(id=id, classes=classes)
        self.show_time = show_time
        self.show_file_counter = show_file_counter
        self.show_actions = show_actions
        self.start_time: Optional[datetime] = None
        self.elapsed_timer: Optional[Timer] = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def compose(self) -> ComposeResult:
        """Compose the status dashboard."""
        with Container(classes="status-dashboard-container"):
            # Main status row
            with Horizontal(classes="status-main-row"):
                # Status message
                yield Label(self.status_message, id="status-message", classes="status-message")
                
                # File counter
                if self.show_file_counter:
                    yield Label("", id="file-counter", classes="file-counter")
                
                # Time display
                if self.show_time:
                    yield Label("", id="time-display", classes="time-display")
                
                # Action buttons
                if self.show_actions:
                    with Container(id="action-buttons", classes="action-buttons hidden"):
                        yield Button("Cancel", id="cancel-button", variant="error", classes="small-button")
                        yield Button("Retry", id="retry-button", variant="warning", classes="small-button hidden")
            
            # Progress bar
            yield ProgressBar(
                id="progress-bar",
                total=self.progress_total,
                show_eta=False,
                show_percentage=True,
                classes="hidden"
            )
            
            # Error/warning container
            with Container(id="alert-container", classes="alert-container hidden"):
                yield Static("", id="alert-messages", classes="alert-messages")
    
    def watch_status_message(self, new_message: str) -> None:
        """Update status message display."""
        try:
            message_label = self.query_one("#status-message", Label)
            message_label.update(new_message)
        except Exception as e:
            logger.error(f"Error updating status message: {e}")
    
    def watch_is_processing(self, is_processing: bool) -> None:
        """React to processing state changes."""
        try:
            progress_bar = self.query_one("#progress-bar", ProgressBar)
            action_buttons = self.query_one("#action-buttons", Container)
            
            if is_processing:
                # Show progress bar and action buttons
                progress_bar.remove_class("hidden")
                if self.show_actions:
                    action_buttons.remove_class("hidden")
                
                # Start timing
                self.start_time = datetime.now()
                if self.show_time and not self.elapsed_timer:
                    self.elapsed_timer = self.set_interval(1, self._update_time_display)
            else:
                # Hide progress bar
                progress_bar.add_class("hidden")
                
                # Stop timing
                if self.elapsed_timer:
                    self.elapsed_timer.pause()
                    self.elapsed_timer = None
                
                # Update action buttons
                if self.show_actions:
                    cancel_button = self.query_one("#cancel-button", Button)
                    retry_button = self.query_one("#retry-button", Button)
                    
                    cancel_button.add_class("hidden")
                    if self.has_errors:
                        retry_button.remove_class("hidden")
                    else:
                        action_buttons.add_class("hidden")
                        
        except Exception as e:
            logger.error(f"Error updating processing state: {e}")
    
    def watch_progress_value(self, progress: float) -> None:
        """Update progress bar."""
        try:
            progress_bar = self.query_one("#progress-bar", ProgressBar)
            progress_bar.progress = progress
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    def watch_current_file(self, current: int) -> None:
        """Update file counter."""
        if self.show_file_counter:
            self._update_file_counter()
    
    def watch_total_files(self, total: int) -> None:
        """Update file counter."""
        if self.show_file_counter:
            self._update_file_counter()
    
    def watch_has_errors(self, has_errors: bool) -> None:
        """Update error display."""
        try:
            alert_container = self.query_one("#alert-container", Container)
            if has_errors and (self.errors or self.warnings):
                alert_container.remove_class("hidden")
                self._update_alert_messages()
            else:
                alert_container.add_class("hidden")
        except Exception as e:
            logger.error(f"Error updating error display: {e}")
    
    def _update_file_counter(self) -> None:
        """Update the file counter display."""
        try:
            counter = self.query_one("#file-counter", Label)
            if self.total_files > 0:
                counter.update(f"File {self.current_file}/{self.total_files}")
            else:
                counter.update("")
        except Exception as e:
            logger.error(f"Error updating file counter: {e}")
    
    def _update_time_display(self) -> None:
        """Update the time elapsed display."""
        if not self.start_time or not self.show_time:
            return
        
        try:
            elapsed = datetime.now() - self.start_time
            minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
            hours, minutes = divmod(minutes, 60)
            
            if hours > 0:
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                time_str = f"{minutes:02d}:{seconds:02d}"
            
            time_display = self.query_one("#time-display", Label)
            
            # Add ETA if we have progress info
            if self.progress_value > 0 and self.progress_total > 0 and self.is_processing:
                progress_pct = self.progress_value / self.progress_total
                if progress_pct > 0:
                    total_estimated = elapsed.total_seconds() / progress_pct
                    remaining = total_estimated - elapsed.total_seconds()
                    if remaining > 0:
                        rem_minutes, rem_seconds = divmod(int(remaining), 60)
                        rem_hours, rem_minutes = divmod(rem_minutes, 60)
                        if rem_hours > 0:
                            eta_str = f"{rem_hours:02d}:{rem_minutes:02d}:{rem_seconds:02d}"
                        else:
                            eta_str = f"{rem_minutes:02d}:{rem_seconds:02d}"
                        time_str += f" (ETA: {eta_str})"
            
            time_display.update(f"Time: {time_str}")
            
        except Exception as e:
            logger.error(f"Error updating time display: {e}")
    
    def _update_alert_messages(self) -> None:
        """Update alert messages display."""
        try:
            alert_static = self.query_one("#alert-messages", Static)
            messages = []
            
            if self.errors:
                messages.append(f"[bold red]Errors ({len(self.errors)}):[/bold red]")
                for error in self.errors[-3:]:  # Show last 3 errors
                    messages.append(f"  • {error}")
                if len(self.errors) > 3:
                    messages.append(f"  ... and {len(self.errors) - 3} more")
            
            if self.warnings:
                if messages:
                    messages.append("")  # Spacing
                messages.append(f"[bold yellow]Warnings ({len(self.warnings)}):[/bold yellow]")
                for warning in self.warnings[-2:]:  # Show last 2 warnings
                    messages.append(f"  • {warning}")
                if len(self.warnings) > 2:
                    messages.append(f"  ... and {len(self.warnings) - 2} more")
            
            alert_static.update("\n".join(messages))
        except Exception as e:
            logger.error(f"Error updating alert messages: {e}")
    
    # Public methods for controlling the dashboard
    
    def start_processing(self, total_files: int = 0, message: str = "Processing...") -> None:
        """Start processing state."""
        self.total_files = total_files
        self.current_file = 0
        self.progress_value = 0
        self.status_message = message
        self.errors.clear()
        self.warnings.clear()
        self.has_errors = False
        self.is_processing = True
    
    def update_progress(
        self, 
        current_file: Optional[int] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None
    ) -> None:
        """Update progress state."""
        if current_file is not None:
            self.current_file = current_file
        if progress is not None:
            self.progress_value = progress
        if message is not None:
            self.status_message = message
    
    def finish_processing(self, message: str = "Complete", success: bool = True) -> None:
        """Finish processing state."""
        self.is_processing = False
        self.status_message = message
        self.progress_value = self.progress_total if success else self.progress_value
        
        if not success:
            self.has_errors = True
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.has_errors = True
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def reset(self) -> None:
        """Reset the dashboard to initial state."""
        self.status_message = "Ready"
        self.progress_value = 0
        self.current_file = 0
        self.total_files = 0
        self.is_processing = False
        self.has_errors = False
        self.errors.clear()
        self.warnings.clear()
        self.start_time = None

# End of status_dashboard.py