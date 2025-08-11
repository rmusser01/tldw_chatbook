# Chat_Window.py
# Description: This file contains the UI components for the chat window
#
# Imports
from typing import TYPE_CHECKING
import time
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, TextArea
from textual.reactive import reactive
#
# Local Imports
# Check if optimized versions are available, fall back to original if not
try:
    from ..Widgets.settings_sidebar_optimized import create_settings_sidebar_optimized
    from tldw_chatbook.Widgets.Chat_Widgets.chat_right_sidebar_optimized import create_chat_right_sidebar_optimized
    USE_OPTIMIZED_SIDEBARS = True
    logger.info("Using optimized sidebars for better performance")
except ImportError:
    from ..Widgets.settings_sidebar import create_settings_sidebar
    from tldw_chatbook.Widgets.Chat_Widgets.chat_right_sidebar import create_chat_right_sidebar
    USE_OPTIMIZED_SIDEBARS = False
    logger.info("Using standard sidebars")
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from ..Constants import TAB_CHAT
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, EMOJI_SEND, FALLBACK_SEND, \
    EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON, EMOJI_STOP, FALLBACK_STOP
from ..config import get_cli_setting

# Configure logger with context
logger = logger.bind(module="Chat_Window")

#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################

#
# Functions:

class ChatWindow(Container):
    """
    Container for the Chat Tab's UI.
    """
    
    BINDINGS = [
        ("ctrl+shift+left", "resize_sidebar_shrink", "Shrink sidebar"),
        ("ctrl+shift+right", "resize_sidebar_expand", "Expand sidebar"),
        ("ctrl+e", "edit_focused_message", "Edit focused message"),
    ]
    
    # Track button state for Send/Stop functionality
    is_send_button = reactive(True)
    
    # Debouncing for button clicks
    _last_send_stop_click = 0
    DEBOUNCE_MS = 300
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.tab_container = None  # Will be set if tabs are enabled
        logger.debug("ChatWindow initialized.")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Central handler for button presses in the ChatWindow.
        Delegates to the appropriate handler in chat_events.py.
        """
        from ..Event_Handlers.Chat_Events import chat_events
        from ..Event_Handlers.Chat_Events import chat_events_sidebar
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize

        button_id = event.button.id
        if not button_id:
            logger.warning("Button pressed with no ID")
            return

        logger.debug(f"Button pressed: {button_id}")

        # Map of button IDs to their handler functions
        button_handlers = {
            "send-stop-chat": self.handle_send_stop_button,  # New unified handler
            "respond-for-me-button": chat_events.handle_respond_for_me_button_pressed,
            "toggle-chat-left-sidebar": chat_events.handle_chat_tab_sidebar_toggle,
            "toggle-chat-right-sidebar": chat_events.handle_chat_tab_sidebar_toggle,
            "chat-new-conversation-button": chat_events.handle_chat_new_conversation_button_pressed,
            "chat-save-current-chat-button": chat_events.handle_chat_save_current_chat_button_pressed,
            "chat-save-conversation-details-button": chat_events.handle_chat_save_details_button_pressed,
            "chat-conversation-load-selected-button": chat_events.handle_chat_load_selected_button_pressed,
            "chat-prompt-load-selected-button": chat_events.handle_chat_view_selected_prompt_button_pressed,
            "chat-prompt-copy-system-button": chat_events.handle_chat_copy_system_prompt_button_pressed,
            "chat-prompt-copy-user-button": chat_events.handle_chat_copy_user_prompt_button_pressed,
            "chat-load-character-button": chat_events.handle_chat_load_character_button_pressed,
            "chat-clear-active-character-button": chat_events.handle_chat_clear_active_character_button_pressed,
            "chat-apply-template-button": chat_events.handle_chat_apply_template_button_pressed,
            # Notes expand/collapse handler
            "chat-notes-expand-button": self.handle_notes_expand_button,
        }

        # Add sidebar button handlers
        button_handlers.update(chat_events_sidebar.CHAT_SIDEBAR_BUTTON_HANDLERS)
        # Add sidebar resize handlers
        button_handlers.update(chat_events_sidebar_resize.CHAT_SIDEBAR_RESIZE_HANDLERS)

        # Check if we have a handler for this button
        handler = button_handlers.get(button_id)
        if handler:
            logger.debug(f"Calling handler for button: {button_id}")
            # Call the handler with the app instance and event
            await handler(self.app_instance, event)
            # Stop the event from propagating
            event.stop()
        else:
            logger.warning(f"No handler found for button: {button_id}")


    def compose(self) -> ComposeResult:
        logger.debug("Composing ChatWindow UI")
        compose_start = time.perf_counter()
        
        # Settings Sidebar (Left)
        sidebar_start = time.perf_counter()
        if USE_OPTIMIZED_SIDEBARS:
            yield from create_settings_sidebar_optimized(TAB_CHAT, self.app_instance.app_config)
        else:
            yield from create_settings_sidebar(TAB_CHAT, self.app_instance.app_config)
        left_sidebar_time = time.perf_counter() - sidebar_start
        logger.info(f"ChatWindow: Left sidebar created in {left_sidebar_time:.3f}s (optimized={USE_OPTIMIZED_SIDEBARS})")

        # Check if tabs are enabled
        enable_tabs = get_cli_setting("chat_defaults", "enable_tabs", False)
        
        if enable_tabs:
            # Use tabbed interface
            logger.info("Chat tabs are enabled - using ChatTabContainer")
            with Container(id="chat-main-content"):
                # Store reference for compatibility
                self.tab_container = ChatTabContainer(self.app_instance)
                yield self.tab_container
        else:
            # Use original single-session interface
            logger.debug("Chat tabs are disabled - using single session interface")
            with Container(id="chat-main-content"):
                yield VerticalScroll(id="chat-log")
                with Horizontal(id="chat-input-area"):
                    yield Button(
                        get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), 
                        id="toggle-chat-left-sidebar",
                        classes="sidebar-toggle",
                        tooltip="Toggle left sidebar (Ctrl+\[)"
                    )
                    yield TextArea(id="chat-input", classes="chat-input")
                    yield Button(
                        get_char(EMOJI_SEND if self.is_send_button else EMOJI_STOP, 
                                FALLBACK_SEND if self.is_send_button else FALLBACK_STOP), 
                        id="send-stop-chat", 
                        classes="send-button",
                        tooltip="Send message" if self.is_send_button else "Stop generation"
                    )
                    yield Button(
                        "💡", 
                        id="respond-for-me-button", 
                        classes="action-button suggest-button",
                        tooltip="Suggest a response"
                    ) # Suggest button
                    logger.debug("'respond-for-me-button' composed.")
                    yield Button(
                        get_char(EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON), 
                        id="toggle-chat-right-sidebar",
                        classes="sidebar-toggle",
                        tooltip="Toggle right sidebar (Ctrl+\])"
                    )

        # Character Details Sidebar (Right)
        right_sidebar_start = time.perf_counter()
        if USE_OPTIMIZED_SIDEBARS:
            yield from create_chat_right_sidebar_optimized(
                "chat",
                initial_ephemeral_state=self.app_instance.current_chat_is_ephemeral
            )
        else:
            yield from create_chat_right_sidebar(
                "chat",
                initial_ephemeral_state=self.app_instance.current_chat_is_ephemeral
            )
        right_sidebar_time = time.perf_counter() - right_sidebar_start
        logger.info(f"ChatWindow: Right sidebar created in {right_sidebar_time:.3f}s (optimized={USE_OPTIMIZED_SIDEBARS})")
        
        total_compose_time = time.perf_counter() - compose_start
        logger.info(f"ChatWindow: Total compose time: {total_compose_time:.3f}s")
    
    async def handle_notes_expand_button(self, app, event) -> None:
        """Handle the notes expand/collapse button."""
        try:
            button = app.query_one("#chat-notes-expand-button", Button)
            textarea = app.query_one("#chat-notes-content-textarea", TextArea)
            
            # Toggle between expanded and normal states
            if "notes-textarea-expanded" in textarea.classes:
                # Collapse
                textarea.remove_class("notes-textarea-expanded")
                textarea.add_class("notes-textarea-normal")
                textarea.styles.height = 10
                button.label = "Expand Notes"
            else:
                # Expand
                textarea.remove_class("notes-textarea-normal")
                textarea.add_class("notes-textarea-expanded")
                textarea.styles.height = 25
                button.label = "Collapse Notes"
                
            # Focus the textarea after expanding
            textarea.focus()
            
        except Exception as e:
            logger.error(f"Error handling notes expand button: {e}")
    
    async def action_resize_sidebar_shrink(self) -> None:
        """Action for keyboard shortcut to shrink sidebar."""
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        await chat_events_sidebar_resize.handle_sidebar_shrink(self.app_instance, None)
    
    async def action_resize_sidebar_expand(self) -> None:
        """Action for keyboard shortcut to expand sidebar."""
        from ..Event_Handlers.Chat_Events import chat_events_sidebar_resize
        await chat_events_sidebar_resize.handle_sidebar_expand(self.app_instance, None)
    
    async def action_edit_focused_message(self) -> None:
        """Action for keyboard shortcut to edit the focused message."""
        from ..Event_Handlers.Chat_Events import chat_events
        
        try:
            # Get the chat log container
            chat_log = self.app_instance.query_one("#chat-log", VerticalScroll)
            
            # Find the focused ChatMessage widget
            focused_widget = self.app_instance.focused
            
            # Check if the focused widget is a ChatMessage or if we need to find one
            from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage
            
            if isinstance(focused_widget, ChatMessage):
                message_widget = focused_widget
            else:
                # Try to find the last message in the chat log as a fallback
                messages = chat_log.query(ChatMessage)
                if messages:
                    message_widget = messages[-1]
                    message_widget.focus()
                else:
                    logger.debug("No messages found to edit")
                    return
            
            # Find the edit button in the message widget
            try:
                edit_button = message_widget.query_one(".edit-button", Button)
                # Trigger the edit action by simulating button press
                await chat_events.handle_chat_action_button_pressed(
                    self.app_instance, 
                    edit_button, 
                    message_widget
                )
            except Exception as e:
                logger.debug(f"Could not find or click edit button: {e}")
                
        except Exception as e:
            logger.error(f"Error in edit_focused_message action: {e}")
            self.app_instance.notify("Could not enter edit mode", severity="warning")
    
    async def on_mount(self) -> None:
        """Called when the widget is mounted."""
        # Watch for streaming state changes
        self._update_button_state()
        # Set up periodic state checking (every 500ms)
        self.set_interval(0.5, self._check_streaming_state)
    
    def _update_button_state(self) -> None:
        """Update the send/stop button based on streaming state."""
        is_streaming = self.app_instance.get_current_chat_is_streaming()
        has_worker = (hasattr(self.app_instance, 'current_chat_worker') and 
                     self.app_instance.current_chat_worker and 
                     self.app_instance.current_chat_worker.is_running)
        
        # Update button state
        self.is_send_button = not (is_streaming or has_worker)
        
        # Update button appearance
        try:
            button = self.query_one("#send-stop-chat", Button)
            button.label = get_char(EMOJI_SEND if self.is_send_button else EMOJI_STOP,
                                  FALLBACK_SEND if self.is_send_button else FALLBACK_STOP)
            button.tooltip = "Send message" if self.is_send_button else "Stop generation"
            
            # Update button styling
            if self.is_send_button:
                button.remove_class("stop-state")
            else:
                button.add_class("stop-state")
        except Exception as e:
            logger.debug(f"Could not update button: {e}")
    
    def watch_is_send_button(self, is_send: bool) -> None:
        """Watch for changes to button state to update appearance."""
        self._update_button_state()
    
    def _check_streaming_state(self) -> None:
        """Periodically check streaming state and update button."""
        self._update_button_state()
    
    async def handle_send_stop_button(self, app_instance, event):
        """Unified handler for Send/Stop button with debouncing."""
        from ..Event_Handlers.Chat_Events import chat_events
        
        current_time = time.time() * 1000
        
        # Debounce rapid clicks
        if current_time - self._last_send_stop_click < self.DEBOUNCE_MS:
            logger.debug("Button click debounced")
            return
        self._last_send_stop_click = current_time
        
        # Disable button during operation
        try:
            button = self.query_one("#send-stop-chat", Button)
            button.disabled = True
        except Exception:
            pass
        
        try:
            # Check current state and route to appropriate handler
            if self.app_instance.get_current_chat_is_streaming() or (
                hasattr(self.app_instance, 'current_chat_worker') and 
                self.app_instance.current_chat_worker and 
                self.app_instance.current_chat_worker.is_running
            ):
                # Stop operation
                logger.info("Send/Stop button pressed - stopping generation")
                await chat_events.handle_stop_chat_generation_pressed(app_instance, event)
            else:
                # Send operation
                logger.info("Send/Stop button pressed - sending message")
                await chat_events.handle_chat_send_button_pressed(app_instance, event)
        finally:
            # Re-enable button and update state after operation
            try:
                button = self.query_one("#send-stop-chat", Button)
                button.disabled = False
            except Exception:
                pass
            self._update_button_state()

#
# End of Chat_Window.py
#######################################################################################################################
