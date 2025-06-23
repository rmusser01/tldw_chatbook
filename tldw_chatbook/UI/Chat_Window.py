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
        }

        # Add sidebar button handlers
        button_handlers.update(chat_events_sidebar.CHAT_SIDEBAR_BUTTON_HANDLERS)

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

    async def on_key(self, event) -> None:
        """Handle key presses in the chat window."""
        # Check if we're in the chat input area
        if event.key == "enter" and self.app_instance.focused and self.app_instance.focused.id == "chat-input":
            # Get the TextArea widget
            chat_input = self.app_instance.query_one("#chat-input", TextArea)
            # Check if it's empty or only whitespace
            if chat_input.text.strip():
                # Only send on Enter in single-line mode (when TextArea height is 1)
                if chat_input.height <= 3:  # Accounting for borders
                    # Simulate clicking the send button
                    from ..Event_Handlers.Chat_Events import chat_events
                    await chat_events.handle_chat_send_button_pressed(self.app_instance, None)
                    event.stop()

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
                    tooltip="Toggle left sidebar (Ctrl+[)"
                )
                yield TextArea(id="chat-input", classes="chat-input")
                yield Button(
                    get_char(EMOJI_SEND, FALLBACK_SEND), 
                    id="send-chat", 
                    classes="send-button",
                    tooltip="Send message (Enter)"
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
                    tooltip="Toggle right sidebar (Ctrl+])"
                )

        # Character Details Sidebar (Right)
        yield from create_chat_right_sidebar(
            "chat",
            initial_ephemeral_state=self.app_instance.current_chat_is_ephemeral
        )

#
# End of Chat_Window.py
#######################################################################################################################
