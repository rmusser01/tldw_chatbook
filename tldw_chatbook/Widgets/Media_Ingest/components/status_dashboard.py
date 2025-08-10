# tldw_chatbook/Widgets/Media_Ingest/components/status_dashboard.py
"""
Reusable status dashboard component for media ingestion progress.
"""

from typing import Optional
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, ProgressBar
from textual.reactive import reactive
from textual.widget import Widget

from ..base_media_ingest_window import ProcessingStatus


class StatusDashboard(Container):
    """
    Status dashboard showing processing progress and current operations.
    
    Features:
    - Main status message
    - File counter
    - Progress bar 
    - Current operation display
    - Error display
    - Responsive layout
    """
    
    status = reactive(ProcessingStatus(state="idle"))
    
    def compose(self) -> ComposeResult:
        """Compose the status dashboard."""
        with Container(classes="status-dashboard-container"):
            # Main status row
            with Horizontal(classes="status-main-row"):
                yield Static("Ready to process files", id="status-message", classes="status-message")
                yield Static("", id="file-counter", classes="file-counter hidden")
                yield Static("", id="time-display", classes="time-display hidden")
            
            # Progress bar (initially hidden)
            yield ProgressBar(id="progress-bar", classes="progress-bar hidden")
            
            # Current operation (initially hidden)
            yield Static("", id="current-operation", classes="current-operation hidden")
            
            # Error display (initially hidden)  
            yield Static("", id="error-display", classes="error-display hidden")
    
    def watch_status(self, status: ProcessingStatus):
        """Update display when status changes."""
        try:
            # Update status message
            status_msg = self.query_one("#status-message")
            status_msg.update(status.message)
            
            # Update progress bar
            progress_bar = self.query_one("#progress-bar")
            if status.state == "processing":
                progress_bar.remove_class("hidden")
                progress_bar.progress = status.progress
            else:
                progress_bar.add_class("hidden")
            
            # Update file counter
            counter = self.query_one("#file-counter")
            if status.total_files > 0:
                counter.update(f"{status.files_processed}/{status.total_files} files")
                counter.remove_class("hidden")
            else:
                counter.add_class("hidden")
            
            # Update current operation
            current_op = self.query_one("#current-operation")
            if status.current_file:
                current_op.update(f"Processing: {status.current_file}")
                current_op.remove_class("hidden")
            else:
                current_op.add_class("hidden")
            
            # Update error display
            error_display = self.query_one("#error-display")
            if status.error:
                error_display.update(f"Error: {status.error}")
                error_display.remove_class("hidden")
            else:
                error_display.add_class("hidden")
                
            # Update container styling based on state
            container = self.query_one(".status-dashboard-container")
            container.remove_class("idle", "processing", "complete", "error")
            container.add_class(status.state)
            
        except Exception as e:
            # Fail silently - don't break UI for display issues
            pass
    
    def set_status(self, status: ProcessingStatus):
        """Manually set the status (alternative to reactive)."""
        self.status = status
    
    def reset(self):
        """Reset to idle state."""
        self.status = ProcessingStatus(state="idle", message="Ready to process files")
    
    # Default styling
    DEFAULT_CSS = """
    .status-dashboard-container {
        dock: top;
        height: auto;
        min-height: 3;
        background: $surface;
        border: round $primary;
        padding: 1;
        margin-bottom: 1;
    }
    
    .status-dashboard-container.processing {
        border: round $accent;
        background: $accent 10%;
    }
    
    .status-dashboard-container.complete {
        border: round $success;
        background: $success 10%;
    }
    
    .status-dashboard-container.error {
        border: round $error;
        background: $error 10%;
    }
    
    .status-main-row {
        height: 3;
        align: left middle;
    }
    
    .status-message {
        width: 1fr;
        text-style: bold;
    }
    
    .file-counter, .time-display {
        width: auto;
        margin-left: 2;
        color: $text-muted;
    }
    
    .progress-bar {
        margin-top: 1;
        height: 1;
    }
    
    .current-operation {
        margin-top: 1;
        color: $text-muted;
        text-style: italic;
    }
    
    .error-display {
        margin-top: 1;
        padding: 1;
        background: $error 20%;
        border: solid $error;
        color: $error;
        text-style: bold;
    }
    
    .hidden {
        display: none;
    }
    """