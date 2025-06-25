# Chat_Window.py
# Description: This file contains the UI components for the chat window
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, TextArea
#
# Local Imports
from ..Widgets.settings_sidebar import create_settings_sidebar
from ..Widgets.chat_right_sidebar import create_chat_right_sidebar
from ..Constants import TAB_CHAT
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, EMOJI_SEND, FALLBACK_SEND, \
    EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON, EMOJI_STOP, FALLBACK_STOP

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
    ]
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
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
            "send-chat": chat_events.handle_chat_send_button_pressed,
            "respond-for-me-button": chat_events.handle_respond_for_me_button_pressed,
            "stop-chat-generation": chat_events.handle_stop_chat_generation_pressed,
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
        # Settings Sidebar (Left)
        yield from create_settings_sidebar(TAB_CHAT, self.app_instance.app_config)

        # Main Chat Content Area
        with Container(id="chat-main-content"):
            yield VerticalScroll(id="chat-log")
            with Horizontal(id="chat-input-area"):
                yield Button(
                    get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), 
                    id="toggle-chat-left-sidebar",
                    classes="sidebar-toggle",
                    tooltip="Toggle left sidebar"
                )
                yield TextArea(id="chat-input", classes="chat-input")
                yield Button(
                    get_char(EMOJI_SEND, FALLBACK_SEND), 
                    id="send-chat", 
                    classes="send-button",
                    tooltip="Send message"
                )
                yield Button(
                    "💡", 
                    id="respond-for-me-button", 
                    classes="action-button suggest-button",
                    tooltip="Suggest a response"
                ) # Suggest button
                logger.debug("'respond-for-me-button' composed.")
                yield Button(
                    get_char(EMOJI_STOP, FALLBACK_STOP), 
                    id="stop-chat-generation", 
                    classes="stop-button",
                    disabled=True,
                    tooltip="Stop generation"
                )
                yield Button(
                    get_char(EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON), 
                    id="toggle-chat-right-sidebar",
                    classes="sidebar-toggle",
                    tooltip="Toggle right sidebar"
                )

        # Character Details Sidebar (Right)
        yield from create_chat_right_sidebar(
            "chat",
            initial_ephemeral_state=self.app_instance.current_chat_is_ephemeral
        )
    
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
                button.label = "⬆ Expand"
            else:
                # Expand
                textarea.remove_class("notes-textarea-normal")
                textarea.add_class("notes-textarea-expanded")
                textarea.styles.height = 25
                button.label = "⬇ Collapse"
                
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

#
# End of Chat_Window.py
#######################################################################################################################
