# Chat_Window_Branched.py
# Description: Enhanced Chat Window with branching support and improved UX
#
# Imports
from typing import TYPE_CHECKING, Optional
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, TextArea, Static
from textual.binding import Binding
#
# Local Imports
from ..Widgets.settings_sidebar import create_settings_sidebar
from ..Widgets.chat_right_sidebar import create_chat_right_sidebar
from ..Widgets.branch_tree_view import BranchTreeView
from ..Constants import TAB_CHAT
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, EMOJI_SEND, FALLBACK_SEND, \
    EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON, EMOJI_STOP, FALLBACK_STOP

# Configure logger with context
logger = logger.bind(module="Chat_Window_Branched")

#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################

#
# Functions:

class ChatWindowBranched(Container):
    """
    Enhanced Container for the Chat Tab's UI with branching support.
    """
    
    # CSS for improved layout
    DEFAULT_CSS = """
    ChatWindowBranched {
        layout: horizontal;
    }
    
    #chat-main-content {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }
    
    #chat-log {
        width: 100%;
        height: 1fr;
        padding: 1;
    }
    
    #chat-input-area {
        width: 100%;
        height: auto;
        layout: horizontal;
        padding: 1;
        background: $surface;
        border-top: solid $surface-lighten-1;
    }
    
    #chat-input {
        width: 1fr;
        height: 3;
        min-height: 3;
        max-height: 10;
    }
    
    .input-button-group {
        layout: horizontal;
        height: 3;
        margin: 0 1;
    }
    
    .input-button-group Button {
        min-width: 3;
        margin: 0 0 0 1;
    }
    
    #branch-indicator {
        width: 100%;
        height: 2;
        background: $accent-darken-2;
        color: $text;
        padding: 0 1;
        display: none;
    }
    
    #branch-indicator.visible {
        display: block;
    }
    
    .branch-tree-panel {
        width: 25%;
        height: 100%;
        display: none;
        border-left: solid $surface-lighten-1;
    }
    
    .branch-tree-panel.visible {
        display: block;
    }
    """
    
    # Key bindings
    BINDINGS = [
        Binding("ctrl+enter", "send_message", "Send Message"),
        Binding("ctrl+b", "toggle_branch_view", "Toggle Branches"),
        Binding("ctrl+n", "new_branch", "New Branch"),
        Binding("ctrl+k", "quick_switch", "Quick Switch"),
        Binding("escape", "cancel_operation", "Cancel"),
    ]
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.branch_tree_visible = False
        logger.debug("ChatWindowBranched initialized.")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Central handler for button presses in the ChatWindow.
        Delegates to the appropriate handler.
        """
        from ..Event_Handlers.Chat_Events import chat_events
        from ..Event_Handlers.Chat_Events import chat_events_sidebar
        from ..Event_Handlers.Chat_Events import chat_branch_events

        button_id = event.button.id
        if not button_id:
            logger.warning("Button pressed with no ID")
            return

        logger.debug(f"Button pressed: {button_id}")

        # Map of button IDs to their handler functions
        button_handlers = {
            # Original handlers
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
            
            # New branch-related handlers
            "toggle-branch-view": self.handle_toggle_branch_view,
            "create-branch": lambda app, event: chat_branch_events.handle_create_branch_from_current(app),
            "quick-action": self.handle_quick_action_menu,
        }

        # Add sidebar button handlers
        button_handlers.update(chat_events_sidebar.CHAT_SIDEBAR_BUTTON_HANDLERS)
        
        # Add branch button handlers
        button_handlers.update(chat_branch_events.CHAT_BRANCH_BUTTON_HANDLERS)

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

    async def handle_toggle_branch_view(self, app_instance, event) -> None:
        """Toggle the branch tree view panel."""
        try:
            branch_panel = self.query_one(".branch-tree-panel")
            self.branch_tree_visible = not self.branch_tree_visible
            
            if self.branch_tree_visible:
                branch_panel.add_class("visible")
                # Update branch tree data
                from ..Event_Handlers.Chat_Events.chat_branch_events import update_branch_tree_view
                await update_branch_tree_view(app_instance)
            else:
                branch_panel.remove_class("visible")
                
        except Exception as e:
            logger.error(f"Error toggling branch view: {e}")

    async def handle_quick_action_menu(self, app_instance, event) -> None:
        """Show quick action menu for grouped operations."""
        # This would show a dropdown/popup menu with options like:
        # - New Branch
        # - Switch Conversation
        # - Apply Template
        # - Insert Media
        # - Export Chat
        app_instance.notify("Quick actions menu (coming soon)", severity="information")

    def action_send_message(self) -> None:
        """Handle Ctrl+Enter to send message."""
        # Trigger the send button click
        try:
            send_button = self.query_one("#send-chat", Button)
            send_button.press()
        except Exception as e:
            logger.error(f"Error triggering send: {e}")

    def action_toggle_branch_view(self) -> None:
        """Handle Ctrl+B to toggle branch view."""
        try:
            toggle_button = self.query_one("#toggle-branch-view", Button)
            toggle_button.press()
        except Exception as e:
            logger.error(f"Error toggling branch view: {e}")

    def action_new_branch(self) -> None:
        """Handle Ctrl+N to create new branch."""
        try:
            branch_button = self.query_one("#create-branch", Button)
            branch_button.press()
        except Exception as e:
            logger.error(f"Error creating branch: {e}")

    def action_quick_switch(self) -> None:
        """Handle Ctrl+K for quick conversation switch."""
        self.app_instance.notify("Quick switch (Ctrl+K) coming soon", severity="information")

    def action_cancel_operation(self) -> None:
        """Handle Escape key."""
        # Could be used to cancel ongoing operations, close popups, etc.
        pass

    def compose(self) -> ComposeResult:
        logger.debug("Composing ChatWindowBranched UI")
        
        # Settings Sidebar (Left)
        yield from create_settings_sidebar(TAB_CHAT, self.app_instance.app_config)

        # Main Chat Content Area
        with Container(id="chat-main-content"):
            # Branch indicator bar
            yield Static(
                "🔀 Viewing branch: Main", 
                id="branch-indicator",
                classes="hidden"
            )
            
            # Chat messages area
            yield VerticalScroll(id="chat-log")
            
            # Simplified input area with grouped buttons
            with Horizontal(id="chat-input-area"):
                # Left sidebar toggle
                yield Button(
                    get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), 
                    id="toggle-chat-left-sidebar",
                    classes="sidebar-toggle",
                    tooltip="Toggle settings (F1)"
                )
                
                # Main input area
                yield TextArea(
                    id="chat-input", 
                    classes="chat-input",
                    placeholder="Type your message... (Ctrl+Enter to send)"
                )
                
                # Primary action group
                with Container(classes="input-button-group"):
                    yield Button(
                        get_char(EMOJI_SEND, FALLBACK_SEND), 
                        id="send-chat", 
                        classes="send-button",
                        variant="primary",
                        tooltip="Send message (Ctrl+Enter)"
                    )
                    yield Button(
                        get_char(EMOJI_STOP, FALLBACK_STOP), 
                        id="stop-chat-generation", 
                        classes="stop-button",
                        disabled=True,
                        variant="error",
                        tooltip="Stop generation"
                    )
                
                # Secondary action group
                with Container(classes="input-button-group"):
                    yield Button(
                        "⚡", 
                        id="quick-action",
                        classes="action-button",
                        tooltip="Quick actions"
                    )
                    yield Button(
                        "💡", 
                        id="respond-for-me-button", 
                        classes="action-button suggest-button",
                        tooltip="Suggest response"
                    )
                    yield Button(
                        "🔀", 
                        id="toggle-branch-view",
                        classes="action-button",
                        tooltip="Toggle branches (Ctrl+B)"
                    )
                
                # Right sidebar toggle
                yield Button(
                    get_char(EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON), 
                    id="toggle-chat-right-sidebar",
                    classes="sidebar-toggle",
                    tooltip="Toggle character/session (F2)"
                )

        # Branch Tree Panel (hidden by default)
        with Container(classes="branch-tree-panel"):
            yield BranchTreeView(id="chat-branch-tree")

        # Character Details Sidebar (Right)
        yield from create_chat_right_sidebar(
            "chat",
            initial_ephemeral_state=self.app_instance.current_chat_is_ephemeral
        )

#
# End of Chat_Window_Branched.py
#######################################################################################################################