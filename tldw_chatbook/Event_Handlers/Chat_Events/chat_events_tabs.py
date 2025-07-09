# chat_events_tabs.py
# Description: Tab-aware versions of chat event handlers
#
# Imports
from typing import TYPE_CHECKING, Optional
#
# 3rd-Party Imports
from loguru import logger
from textual.widgets import Button
#
# Local Imports
from ...Chat.chat_models import ChatSessionData
from ...config import get_cli_setting
from . import chat_events
#
if TYPE_CHECKING:
    from ...app import TldwCli
#
#######################################################################################################################
#
# Functions:

def get_active_session_data(app: 'TldwCli') -> Optional[ChatSessionData]:
    """
    Get the active chat session data.
    
    Returns None if tabs are disabled or no active session.
    """
    if not get_cli_setting("chat_defaults", "enable_tabs", False):
        # Tabs disabled - return a default session data for compatibility
        return ChatSessionData(tab_id="default", title="Chat", is_ephemeral=app.current_chat_is_ephemeral)
    
    # Tabs enabled - get from tab container
    try:
        chat_window = app.query_one("#chat-window")
        if hasattr(chat_window, 'tab_container') and chat_window.tab_container:
            active_session = chat_window.tab_container.get_active_session()
            if active_session:
                return active_session.session_data
    except Exception as e:
        logger.error(f"Error getting active session: {e}")
    
    return None

def get_widget_id_for_session(base_id: str, session_data: Optional[ChatSessionData]) -> str:
    """
    Get the appropriate widget ID for a session.
    
    Args:
        base_id: Base widget ID (e.g., "chat-input")
        session_data: Session data containing tab_id
        
    Returns:
        Full widget ID (e.g., "chat-input-abc123" for tabs, "chat-input" for non-tabs)
    """
    if not session_data or not get_cli_setting("chat_defaults", "enable_tabs", False):
        # No tabs - use base ID
        return base_id
    
    # Tabs enabled - append tab ID
    return f"{base_id}-{session_data.tab_id}"

async def handle_chat_send_button_pressed_with_tabs(app: 'TldwCli', event: Button.Pressed, session_data: Optional[ChatSessionData] = None) -> None:
    """
    Tab-aware version of handle_chat_send_button_pressed.
    
    This wrapper modifies the event handler to work with tabbed sessions.
    """
    # Get session data if not provided
    if not session_data:
        session_data = get_active_session_data(app)
    
    if not session_data:
        logger.error("No active session found for send button")
        app.notify("No active chat session", severity="error")
        return
    
    # Store original query_one method
    original_query_one = app.query_one
    
    # Create a wrapper that modifies widget IDs for tab context
    def tab_aware_query_one(selector, widget_type=None):
        # Check if this is a chat-related widget
        if selector.startswith("#chat-") and not selector.startswith("#chat-window"):
            # Extract base ID
            base_id = selector[1:]  # Remove '#'
            # Get tab-specific ID
            tab_specific_id = f"#{get_widget_id_for_session(base_id, session_data)}"
            logger.debug(f"Redirecting query from {selector} to {tab_specific_id}")
            selector = tab_specific_id
        
        # Call original query_one
        if widget_type:
            return original_query_one(selector, widget_type)
        else:
            return original_query_one(selector)
    
    # Temporarily replace query_one
    app.query_one = tab_aware_query_one
    
    try:
        # Update session-specific state before calling handler
        if session_data:
            # Update current conversation ID to match this session
            app.current_chat_conversation_id = session_data.conversation_id
            app.current_chat_is_ephemeral = session_data.is_ephemeral
            
            # Update worker and streaming state
            app.current_chat_worker = session_data.current_worker
            app.current_ai_message_widget = session_data.current_ai_message_widget
            app.set_current_chat_is_streaming(session_data.is_streaming)
        
        # Call the original handler
        await chat_events.handle_chat_send_button_pressed(app, event)
        
        # Update session data after handler completes
        if session_data:
            session_data.is_streaming = app.get_current_chat_is_streaming()
            session_data.current_worker = app.current_chat_worker
            session_data.current_ai_message_widget = app.current_ai_message_widget
            
    finally:
        # Restore original query_one
        app.query_one = original_query_one

async def handle_stop_chat_generation_pressed_with_tabs(app: 'TldwCli', event: Button.Pressed, session_data: Optional[ChatSessionData] = None) -> None:
    """
    Tab-aware version of handle_stop_chat_generation_pressed.
    """
    # Get session data if not provided
    if not session_data:
        session_data = get_active_session_data(app)
    
    if not session_data:
        logger.error("No active session found for stop button")
        return
    
    # Update app state to match this session before stopping
    if session_data.current_worker:
        app.current_chat_worker = session_data.current_worker
    
    # Call original handler
    await chat_events.handle_stop_chat_generation_pressed(app, event)
    
    # Clear session-specific state
    session_data.is_streaming = False
    session_data.current_worker = None

async def handle_respond_for_me_button_pressed_with_tabs(app: 'TldwCli', event: Button.Pressed, session_data: Optional[ChatSessionData] = None) -> None:
    """
    Tab-aware version of handle_respond_for_me_button_pressed.
    """
    # Get session data if not provided
    if not session_data:
        session_data = get_active_session_data(app)
    
    if not session_data:
        logger.error("No active session found for suggest button")
        app.notify("No active chat session", severity="error")
        return
    
    # Store original query_one method
    original_query_one = app.query_one
    
    # Create tab-aware query wrapper
    def tab_aware_query_one(selector, widget_type=None):
        if selector.startswith("#chat-") and not selector.startswith("#chat-window"):
            base_id = selector[1:]
            tab_specific_id = f"#{get_widget_id_for_session(base_id, session_data)}"
            logger.debug(f"Redirecting query from {selector} to {tab_specific_id}")
            selector = tab_specific_id
        
        if widget_type:
            return original_query_one(selector, widget_type)
        else:
            return original_query_one(selector)
    
    # Temporarily replace query_one
    app.query_one = tab_aware_query_one
    
    try:
        # Update app state for this session
        if session_data:
            app.current_chat_conversation_id = session_data.conversation_id
            app.current_chat_is_ephemeral = session_data.is_ephemeral
        
        # Call original handler
        await chat_events.handle_respond_for_me_button_pressed(app, event)
        
    finally:
        # Restore original query_one
        app.query_one = original_query_one

# Additional tab-aware handlers for other chat functionality
async def handle_chat_conversation_search_changed_with_tabs(app: 'TldwCli', event_value: str, session_data: Optional[ChatSessionData] = None) -> None:
    """
    Tab-aware version of conversation search handler.
    """
    # Get session data if not provided
    if not session_data:
        session_data = get_active_session_data(app)
    
    if not session_data:
        logger.error("No active session found for conversation search")
        return
    
    # Update app state for this session
    if session_data:
        app.current_chat_conversation_id = session_data.conversation_id
        app.current_chat_is_ephemeral = session_data.is_ephemeral
    
    # Call original handler
    await chat_events.handle_chat_conversation_search_bar_changed(app, event_value)

async def display_conversation_in_chat_tab_ui_with_tabs(app: 'TldwCli', conversation_id: str, session_data: Optional[ChatSessionData] = None) -> None:
    """
    Tab-aware version of display_conversation_in_chat_tab_ui.
    
    This loads a conversation into a specific tab session.
    """
    # Get session data if not provided
    if not session_data:
        session_data = get_active_session_data(app)
    
    if not session_data:
        logger.error("No active session found for displaying conversation")
        app.notify("No active chat session", severity="error")
        return
    
    # Store original query_one method
    original_query_one = app.query_one
    
    # Create tab-aware query wrapper
    def tab_aware_query_one(selector, widget_type=None):
        # Map widget IDs to tab-specific versions
        widget_mappings = {
            "#chat-log": f"#chat-log-{session_data.tab_id}",
            "#chat-input": f"#chat-input-{session_data.tab_id}",
            "#chat-conversation-title-input": f"#chat-conversation-title-input",  # Keep global
            "#chat-conversation-keywords-input": f"#chat-conversation-keywords-input",  # Keep global
            "#chat-conversation-uuid-display": f"#chat-conversation-uuid-display",  # Keep global
            "#chat-system-prompt": f"#chat-system-prompt",  # Keep global for now
        }
        
        # Check if we need to redirect this selector
        if selector in widget_mappings:
            selector = widget_mappings[selector]
            logger.debug(f"Redirecting query to {selector}")
        
        if widget_type:
            return original_query_one(selector, widget_type)
        else:
            return original_query_one(selector)
    
    # Temporarily replace query_one
    app.query_one = tab_aware_query_one
    
    try:
        # Update session data before calling handler
        session_data.conversation_id = conversation_id
        session_data.is_ephemeral = False
        
        # Update app state
        app.current_chat_conversation_id = conversation_id
        app.current_chat_is_ephemeral = False
        
        # Call the original handler
        await chat_events.display_conversation_in_chat_tab_ui(app, conversation_id)
        
        # Update session title based on loaded conversation
        try:
            conv_title_input = app.query_one("#chat-conversation-title-input")
            if conv_title_input and conv_title_input.value:
                session_data.title = conv_title_input.value
                # If we have access to the tab container, update the tab label
                chat_window = app.query_one("#chat-window")
                if hasattr(chat_window, 'tab_container') and chat_window.tab_container:
                    chat_window.tab_container.update_tab_title(session_data.tab_id, session_data.title)
        except Exception as e:
            logger.debug(f"Could not update tab title: {e}")
        
    finally:
        # Restore original query_one
        app.query_one = original_query_one

# Export a function to patch the ChatSession widget
def setup_tab_aware_handlers(session_widget):
    """
    Set up tab-aware event handlers for a ChatSession widget.
    """
    # The ChatSession widget should override its button handlers to use these tab-aware versions
    session_widget.handle_send_stop_button = lambda event: handle_chat_send_button_pressed_with_tabs(
        session_widget.app_instance, event, session_widget.session_data
    )
    
    session_widget.handle_suggest_button = lambda event: handle_respond_for_me_button_pressed_with_tabs(
        session_widget.app_instance, event, session_widget.session_data
    )

#
# End of chat_events_tabs.py
#######################################################################################################################