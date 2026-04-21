"""
Chat Input Handler Module

Handles all chat input functionality including:
- Send/Stop button management
- Text input handling
- Message sending with attachments
- Button state management
"""

import asyncio
import time
from typing import TYPE_CHECKING, Optional
from loguru import logger
from textual.widgets import Button, TextArea
from textual.worker import WorkerCancelled

if TYPE_CHECKING:
    from ..Chat_Window_Enhanced import ChatWindowEnhanced

logger = logger.bind(module="ChatInputHandler")


class ChatInputHandler:
    """Handles chat input and send/stop functionality."""
    
    # Debouncing for button clicks
    DEBOUNCE_MS = 300
    
    def __init__(self, chat_window: 'ChatWindowEnhanced'):
        """Initialize the input handler.
        
        Args:
            chat_window: Parent ChatWindowEnhanced instance
        """
        self.chat_window = chat_window
        self.app_instance = chat_window.app_instance
        self._last_send_stop_click = 0
    
    async def handle_send_stop_button(self, event):
        """Unified handler for Send/Stop button with debouncing and error recovery.
        
        Args:
            event: Button.Pressed event
        """
        from ...Event_Handlers.Chat_Events import chat_events
        
        current_time = time.time() * 1000
        
        # Debounce rapid clicks
        if current_time - self._last_send_stop_click < self.DEBOUNCE_MS:
            logger.debug("Button click debounced", extra={"time_diff": current_time - self._last_send_stop_click})
            return
        self._last_send_stop_click = current_time
        
        # Disable button during operation
        button = self.chat_window._get_send_button()
        if button:
            try:
                button.disabled = True
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Could not disable send/stop button: {e}")
        
        try:
            # Check current state and route to appropriate handler
            if self.app_instance.get_current_chat_is_streaming() or (
                hasattr(self.app_instance, 'current_chat_worker') and 
                self.app_instance.current_chat_worker and 
                self.app_instance.current_chat_worker.is_running
            ):
                # Stop operation
                logger.info("Send/Stop button pressed - stopping generation", 
                          extra={"action": "stop", "is_streaming": self.app_instance.get_current_chat_is_streaming()})
                await chat_events.handle_stop_chat_generation_pressed(self.app_instance, event)
            else:
                # Send operation - use enhanced handler that includes image
                logger.info("Send/Stop button pressed - sending message", 
                          extra={"action": "send", 
                                 "has_attachment": bool(self.chat_window.pending_attachment or self.chat_window.pending_image)})
                await self.handle_enhanced_send_button(event)
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Widget access error in send/stop handler: {e}", 
                        extra={"button_state": "send" if self.chat_window.is_send_button else "stop"})
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
        except WorkerCancelled as e:
            logger.warning(f"Worker cancelled during send/stop operation: {e}")
            self.app_instance.notify("Operation cancelled", severity="warning")
        except asyncio.CancelledError as e:
            logger.warning(f"Async operation cancelled: {e}")
            self.app_instance.notify("Operation cancelled", severity="warning")
        finally:
            # Re-enable button and update state after operation
            if button:
                try:
                    button.disabled = False
                except (AttributeError, RuntimeError) as e:
                    logger.warning(f"Could not re-enable send/stop button: {e}")
            self.chat_window._update_button_state()
    
    async def handle_enhanced_send_button(self, event):
        """Enhanced send handler that includes image data.
        
        Args:
            event: Button.Pressed event
        """
        from ...Event_Handlers.Chat_Events import chat_events
        
        # First call the original handler
        await chat_events.handle_chat_send_button_pressed(self.app_instance, event)
        
        # Clear attachment states after successful send
        self.chat_window._clear_attachment_state()
    
    def update_button_state(self):
        """Update the send/stop button state based on streaming status."""
        try:
            # Determine current state
            is_streaming = self.app_instance.get_current_chat_is_streaming()
            should_be_send = not is_streaming
            
            # Update reactive property if state changed
            if self.chat_window.is_send_button != should_be_send:
                self.chat_window.is_send_button = should_be_send
                logger.debug(f"Button state updated - is_send: {should_be_send}, is_streaming: {is_streaming}")
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"Could not update button state: {e}")
    
    def get_chat_input_value(self) -> str:
        """Get the current value of the chat input.
        
        Returns:
            Current text in the chat input, or empty string if not available
        """
        chat_input = self.chat_window._get_chat_input()
        if chat_input:
            return chat_input.value
        return ""
    
    def clear_chat_input(self):
        """Clear the chat input field."""
        chat_input = self.chat_window._get_chat_input()
        if chat_input:
            try:
                chat_input.clear()
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Could not clear chat input: {e}")
    
    def focus_chat_input(self):
        """Set focus to the chat input field."""
        chat_input = self.chat_window._get_chat_input()
        if chat_input:
            try:
                chat_input.focus()
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Could not focus chat input: {e}")
    
    def insert_text_at_cursor(self, text: str):
        """Insert text at the current cursor position in the chat input.
        
        Args:
            text: Text to insert
        """
        chat_input = self.chat_window._get_chat_input()
        if not chat_input:
            logger.warning("Chat input not available")
            return
        
        try:
            current_text = chat_input.value
            cursor_pos = chat_input.cursor_location
            
            # Insert text at cursor position
            if cursor_pos:
                row, col = cursor_pos
                lines = current_text.split('\n')
                if row < len(lines):
                    line = lines[row]
                    lines[row] = line[:col] + text + line[col:]
                    new_text = '\n'.join(lines)
                else:
                    new_text = current_text + text
            else:
                new_text = current_text + text
            
            chat_input.value = new_text
            
            # Move cursor to end of inserted text
            lines = new_text.split('\n')
            last_row = len(lines) - 1
            last_col = len(lines[-1]) if lines else 0
            chat_input.cursor_location = (last_row, last_col)
            
        except (IndexError, ValueError, AttributeError) as e:
            logger.error(f"Error inserting text at cursor: {e}")
            # Fallback: just append
            if chat_input:
                chat_input.value += text