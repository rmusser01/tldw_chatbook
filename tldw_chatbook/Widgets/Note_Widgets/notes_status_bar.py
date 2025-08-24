"""Status bar widget for notes screen."""

from typing import Optional
from datetime import datetime
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static, Label
from textual.reactive import reactive


class NotesStatusBar(Horizontal):
    """
    Status bar for notes screen showing save status, word count, etc.
    Follows Textual reactive patterns.
    """
    
    DEFAULT_CSS = """
    NotesStatusBar {
        height: 3;
        padding: 0 1;
        background: $panel;
        align: center middle;
    }
    
    .status-indicator {
        width: auto;
        margin: 0 1;
    }
    
    .status-indicator.saved {
        color: $success;
    }
    
    .status-indicator.unsaved {
        color: $warning;
        text-style: bold;
    }
    
    .status-indicator.saving {
        color: $primary;
        text-style: italic;
    }
    
    .status-indicator.error {
        color: $error;
        text-style: bold;
    }
    
    .word-count {
        color: $text-muted;
        margin: 0 2;
    }
    
    .last-saved {
        color: $text-muted;
        margin: 0 2;
    }
    
    .auto-save-status {
        color: $text-muted;
        margin: 0 2;
    }
    """
    
    # Reactive attributes
    save_status: reactive[str] = reactive("ready")  # ready, saved, unsaved, saving, error
    word_count: reactive[int] = reactive(0)
    char_count: reactive[int] = reactive(0)
    last_saved_time: reactive[Optional[datetime]] = reactive(None)
    auto_save_enabled: reactive[bool] = reactive(True)
    
    def compose(self) -> ComposeResult:
        """Compose the status bar."""
        yield Label("Ready", id="status-indicator", classes="status-indicator")
        yield Label("Words: 0", id="word-count", classes="word-count")
        yield Label("Chars: 0", id="char-count", classes="char-count")
        yield Label("", id="last-saved", classes="last-saved")
        yield Label("Auto-save: On", id="auto-save-status", classes="auto-save-status")
    
    def on_mount(self) -> None:
        """Initialize the status bar."""
        logger.debug("NotesStatusBar mounted")
        self._update_status_display()
    
    def watch_save_status(self, status: str) -> None:
        """React to save status changes."""
        self._update_status_display()
    
    def watch_word_count(self, count: int) -> None:
        """React to word count changes."""
        try:
            word_label = self.query_one("#word-count", Label)
            word_label.update(f"Words: {count:,}")
        except Exception:
            pass
    
    def watch_char_count(self, count: int) -> None:
        """React to character count changes."""
        try:
            char_label = self.query_one("#char-count", Label)
            char_label.update(f"Chars: {count:,}")
        except Exception:
            pass
    
    def watch_last_saved_time(self, time: Optional[datetime]) -> None:
        """React to last saved time changes."""
        try:
            saved_label = self.query_one("#last-saved", Label)
            if time:
                # Format as relative time
                now = datetime.now()
                delta = now - time
                
                if delta.total_seconds() < 60:
                    time_str = "Just now"
                elif delta.total_seconds() < 3600:
                    minutes = int(delta.total_seconds() / 60)
                    time_str = f"{minutes}m ago"
                elif delta.total_seconds() < 86400:
                    hours = int(delta.total_seconds() / 3600)
                    time_str = f"{hours}h ago"
                else:
                    time_str = time.strftime("%b %d, %H:%M")
                
                saved_label.update(f"Saved: {time_str}")
            else:
                saved_label.update("")
        except Exception:
            pass
    
    def watch_auto_save_enabled(self, enabled: bool) -> None:
        """React to auto-save status changes."""
        try:
            auto_save_label = self.query_one("#auto-save-status", Label)
            status = "On" if enabled else "Off"
            auto_save_label.update(f"Auto-save: {status}")
        except Exception:
            pass
    
    def _update_status_display(self) -> None:
        """Update the status indicator based on current status."""
        try:
            indicator = self.query_one("#status-indicator", Label)
            
            # Remove all status classes
            indicator.remove_class("saved", "unsaved", "saving", "error")
            
            # Update based on status
            if self.save_status == "saved":
                indicator.update("✓ Saved")
                indicator.add_class("saved")
            elif self.save_status == "unsaved":
                indicator.update("● Unsaved")
                indicator.add_class("unsaved")
            elif self.save_status == "saving":
                indicator.update("⟳ Saving...")
                indicator.add_class("saving")
            elif self.save_status == "error":
                indicator.update("✗ Error")
                indicator.add_class("error")
            else:  # ready
                indicator.update("Ready")
        except Exception as e:
            logger.error(f"Error updating status display: {e}")
    
    def update_counts(self, word_count: int, char_count: int) -> None:
        """
        Update word and character counts.
        
        Args:
            word_count: Number of words
            char_count: Number of characters
        """
        self.word_count = word_count
        self.char_count = char_count
    
    def set_saving(self) -> None:
        """Set status to saving."""
        self.save_status = "saving"
    
    def set_saved(self, update_time: bool = True) -> None:
        """
        Set status to saved.
        
        Args:
            update_time: Whether to update last saved time
        """
        self.save_status = "saved"
        if update_time:
            self.last_saved_time = datetime.now()
    
    def set_unsaved(self) -> None:
        """Set status to unsaved."""
        self.save_status = "unsaved"
    
    def set_error(self, error_message: Optional[str] = None) -> None:
        """
        Set status to error.
        
        Args:
            error_message: Optional error message to display
        """
        self.save_status = "error"
        if error_message:
            logger.error(f"Status bar error: {error_message}")
    
    def set_ready(self) -> None:
        """Set status to ready."""
        self.save_status = "ready"
    
    def toggle_auto_save(self) -> bool:
        """Toggle auto-save on/off and return new state."""
        self.auto_save_enabled = not self.auto_save_enabled
        return self.auto_save_enabled