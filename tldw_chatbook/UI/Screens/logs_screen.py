"""Logs screen implementation."""

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.widgets import Button

from ..Navigation.base_app_screen import BaseAppScreen
from ..Logs_Window import LogsWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class LogsScreen(BaseAppScreen):
    """
    Logs screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "logs", **kwargs)
        self.logs_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the logs window content."""
        self.logs_window = LogsWindow(self.app_instance, classes="window")
        yield self.logs_window
    
    def on_mount(self) -> None:
        """When the logs screen is mounted, display all buffered logs."""
        super().on_mount()
        
        # Always display buffered logs when the screen is mounted
        if hasattr(self.app_instance, '_log_buffer'):
            try:
                # Find the RichLog widget
                log_widget = self.query_one("#app-log-display")
                # Display all buffered logs
                self.app_instance._display_buffered_logs(log_widget)
            except Exception as e:
                from loguru import logger
                logger.error(f"Failed to display buffered logs: {e}")
    
    def on_unmount(self) -> None:
        """When the logs screen is unmounted, clear the widget reference."""
        super().on_unmount()
        
        # Clear the current log widget reference
        if hasattr(self.app_instance, '_current_log_widget'):
            self.app_instance._current_log_widget = None
    
    def save_state(self):
        """Save logs window state."""
        state = super().save_state()
        # Add any logs-specific state here
        return state
    
    def restore_state(self, state):
        """Restore logs window state."""
        super().restore_state(state)
        # Restore any logs-specific state here
    
    @on(Button.Pressed, "#copy-logs-button")
    async def handle_copy_logs_button(self, event: Button.Pressed) -> None:
        """Handle the copy logs button press."""
        from loguru import logger
        from textual.widgets import RichLog
        
        logger.info("Copy logs button pressed in LogsScreen")
        
        try:
            # For screen navigation, we have a simpler approach
            # Just copy the buffered logs directly
            if hasattr(self.app_instance, '_log_buffer') and self.app_instance._log_buffer:
                # Join all buffered log messages
                all_log_text = "\n".join(self.app_instance._log_buffer)
                
                if all_log_text:
                    # Copy to clipboard
                    self.app.copy_to_clipboard(all_log_text)
                    self.app.notify(
                        f"Copied {len(self.app_instance._log_buffer)} log entries to clipboard!",
                        title="Clipboard",
                        severity="information",
                        timeout=4
                    )
                    logger.info(f"Copied {len(self.app_instance._log_buffer)} log entries to clipboard")
                else:
                    self.app.notify("Log is empty, nothing to copy.", title="Clipboard", severity="warning", timeout=4)
            else:
                # Fallback: try to get from the RichLog widget
                log_widget = self.query_one("#app-log-display", RichLog)
                
                if log_widget.lines:
                    # Extract text from the widget's lines
                    all_log_text_parts = []
                    for strip in log_widget.lines:
                        if hasattr(strip, 'text'):
                            all_log_text_parts.append(strip.text)
                        else:
                            all_log_text_parts.append(str(strip))
                    
                    all_log_text = "\n".join(all_log_text_parts)
                    
                    if all_log_text:
                        self.app.copy_to_clipboard(all_log_text)
                        self.app.notify(
                            f"Copied {len(log_widget.lines)} lines to clipboard!",
                            title="Clipboard",
                            severity="information",
                            timeout=4
                        )
                        logger.info(f"Copied {len(log_widget.lines)} lines to clipboard")
                    else:
                        self.app.notify("Log is empty, nothing to copy.", title="Clipboard", severity="warning", timeout=4)
                else:
                    self.app.notify("Log is empty, nothing to copy.", title="Clipboard", severity="warning", timeout=4)
                    
        except Exception as e:
            self.app.notify(f"Error copying logs: {str(e)}", title="Error", severity="error", timeout=6)
            logger.error(f"Failed to copy logs: {e}", exc_info=True)