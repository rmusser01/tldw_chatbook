"""
Chat Worker Handler - Handles chat-related worker state changes.

This module manages state changes for chat API calls, including regular chat,
character chat (ccp), and respond-for-me workers.
"""

from typing import TYPE_CHECKING, Optional
from textual.worker import Worker, WorkerState
from textual.widgets import Button, Markdown
from textual.css.query import QueryError

from .base_handler import BaseWorkerHandler

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ChatWorkerHandler(BaseWorkerHandler):
    """Handles chat-related worker state changes."""
    
    def can_handle(self, worker_name: str, worker_group: Optional[str] = None) -> bool:
        """
        Check if this handler can process the given worker.
        
        Args:
            worker_name: The name attribute of the worker
            worker_group: The group attribute of the worker (unused for chat workers)
            
        Returns:
            True if this is a chat-related worker
        """
        return (isinstance(worker_name, str) and
                (worker_name.startswith("API_Call_chat") or
                 worker_name.startswith("API_Call_ccp") or
                 worker_name == "respond_for_me_worker"))
    
    async def handle(self, event: Worker.StateChanged) -> None:
        """
        Handle the chat worker state change event.
        
        Args:
            event: The worker state changed event
        """
        worker_info = self.get_worker_info(event)
        self.log_state_change(worker_info, "Chat: ")
        
        # Import here to avoid circular imports
        from tldw_chatbook.config import get_cli_setting
        from tldw_chatbook.Utils.Emoji_Handling import get_char, EMOJI_STOP, FALLBACK_STOP, EMOJI_SEND, FALLBACK_SEND
        from tldw_chatbook.Event_Handlers import worker_events
        
        # Determine which button selector to use based on which chat window is active
        use_enhanced_chat = get_cli_setting("chat_defaults", "use_enhanced_window", False)
        send_button_id = "send-stop-chat" if use_enhanced_chat else "send-chat"
        
        if worker_info['state'] == WorkerState.RUNNING:
            await self._handle_running_state(send_button_id, get_char, EMOJI_STOP, FALLBACK_STOP)
            
        elif worker_info['state'] in [WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED]:
            await self._handle_finished_state(
                event, worker_info, send_button_id, 
                get_char, EMOJI_SEND, FALLBACK_SEND, worker_events
            )
        
        else:
            self.logger.debug(f"Chat-related worker '{worker_info['name']}' in other state: {worker_info['state']}")
    
    async def _handle_running_state(self, send_button_id: str, get_char: callable, 
                                   emoji_stop: str, fallback_stop: str) -> None:
        """Handle the RUNNING state for chat workers."""
        self.logger.info(f"Chat worker is RUNNING, updating button to STOP state")
        
        try:
            send_button = self.app.query_one(f"#{send_button_id}", Button)
            send_button.label = get_char(emoji_stop, fallback_stop)
            self.logger.info(f"Button '#{send_button_id}' changed to STOP state")
        except QueryError:
            self.logger.error(f"Could not find button '#{send_button_id}' to change it to stop state")
    
    async def _handle_finished_state(self, event: Worker.StateChanged, worker_info: dict,
                                    send_button_id: str, get_char: callable,
                                    emoji_send: str, fallback_send: str,
                                    worker_events) -> None:
        """Handle the finished states (SUCCESS, ERROR, CANCELLED) for chat workers."""
        self.logger.info(f"Chat worker finished with state {worker_info['state']}")
        
        # Change stop button back to send button
        try:
            send_button = self.app.query_one(f"#{send_button_id}", Button)
            send_button.label = get_char(emoji_send, fallback_send)
            self.logger.info(f"Button '#{send_button_id}' changed back to SEND state")
        except QueryError:
            self.logger.error(f"Could not find button '#{send_button_id}' to change it back to send state")
        
        # Handle specific finished states
        if worker_info['state'] in [WorkerState.SUCCESS, WorkerState.ERROR]:
            self.logger.debug(
                f"Delegating state {worker_info['state']} for chat worker to worker_events for result processing"
            )
            # Delegate to existing handler for result processing
            await worker_events.handle_api_call_worker_state_changed(self.app, event)
            
        elif worker_info['state'] == WorkerState.CANCELLED:
            await self._handle_cancelled_state(worker_info['name'])
        
        # Clear the current chat worker reference and streaming state
        if worker_info['name'].startswith("API_Call_chat") or worker_info['name'] == "respond_for_me_worker":
            self.app.set_current_chat_worker(None)
            
            # Reset streaming state for chat workers (not for respond_for_me)
            if worker_info['name'].startswith("API_Call_chat"):
                self.app.set_current_chat_is_streaming(False)
                self.logger.debug(f"Reset current_chat_is_streaming to False after {worker_info['name']} finished")
            
            self.logger.debug(f"Cleared current_chat_worker after {worker_info['name']} finished")
    
    async def _handle_cancelled_state(self, worker_name: str) -> None:
        """Handle the CANCELLED state for chat workers."""
        self.logger.info(f"Worker '{worker_name}' was cancelled")
        
        # Update AI message widget if needed
        if (hasattr(self.app, 'current_ai_message_widget') and 
            self.app.current_ai_message_widget and 
            not self.app.current_ai_message_widget.generation_complete):
            
            self.logger.debug("Finalizing AI message widget UI due to worker CANCELLED state")
            
            try:
                markdown_widget = self.app.current_ai_message_widget.query_one(".message-text", Markdown)
                
                # Check if already updated by handle_stop_chat_generation_pressed
                if "Chat generation cancelled by user." not in self.app.current_ai_message_widget.message_text:
                    self.app.current_ai_message_widget.message_text += "\n_(Stream Cancelled)_"
                    markdown_widget.update(self.app.current_ai_message_widget.message_text)
                
                self.app.current_ai_message_widget.mark_generation_complete()
                self.app.current_ai_message_widget = None  # Clear reference
                
            except QueryError as e:
                self.logger.error(f"Error updating AI message UI on CANCELLED state: {e}")