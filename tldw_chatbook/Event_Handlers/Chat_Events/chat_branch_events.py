# chat_branch_events.py
# Description: Event handlers for chat branching functionality
#
# Imports
from typing import TYPE_CHECKING, Optional, Dict, Any
#
# 3rd-Party Imports
from loguru import logger
from textual.widgets import Button, Input, TextArea, ListView, ListItem, Label
from textual.containers import VerticalScroll
from textual.css.query import QueryError
#
# Local Imports
from tldw_chatbook.Chat.Chat_Branching import (
    create_conversation_branch,
    get_conversation_branches,
    get_branch_info,
    navigate_to_branch,
    get_message_branches
)
from tldw_chatbook.Widgets.chat_message import ChatMessage
from tldw_chatbook.Widgets.branch_tree_view import BranchTreeView, CompactBranchIndicator
from tldw_chatbook.Utils.input_validation import validate_text_input

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

# Configure logger
logger = logger.bind(module="chat_branch_events")

#
########################################################################################################################
#
# Branch Event Handlers:

async def handle_create_branch_from_message(app: 'TldwCli', message_id: str, message_widget: ChatMessage) -> None:
    """
    Create a new conversation branch from a specific message.
    
    Args:
        app: Application instance
        message_id: ID of the message to branch from
        message_widget: The ChatMessage widget instance
    """
    logger.info(f"Creating branch from message {message_id}")
    
    if not app.chachanotes_db:
        logger.error("Database not available")
        app.notify("Database error: Cannot create branch", severity="error")
        return
    
    if not app.current_chat_conversation_id:
        logger.warning("No active conversation to branch from")
        app.notify("No active conversation to branch from", severity="warning")
        return
    
    try:
        # Get current character data
        character_id = None
        if app.current_chat_active_character_data:
            character_id = app.current_chat_active_character_data.get('id')
        
        if not character_id:
            # Use default character ID
            from tldw_chatbook.Character_Chat.Character_Chat_Lib import DEFAULT_CHARACTER_ID
            character_id = DEFAULT_CHARACTER_ID
        
        # Create the branch
        new_branch_id = create_conversation_branch(
            app.chachanotes_db,
            app.current_chat_conversation_id,
            message_id,
            character_id,
            branch_title=None,  # Will auto-generate
            client_id=app.chachanotes_db.client_id
        )
        
        if new_branch_id:
            logger.info(f"Successfully created branch {new_branch_id}")
            app.notify(f"Created new branch from message", severity="information")
            
            # Update UI to show branch indicator
            await update_message_branch_indicators(app, app.current_chat_conversation_id)
            
            # Optionally switch to the new branch
            if await confirm_switch_to_branch(app):
                await switch_to_branch(app, new_branch_id)
        else:
            logger.error("Failed to create branch")
            app.notify("Failed to create branch", severity="error")
            
    except Exception as e:
        logger.error(f"Error creating branch: {e}", exc_info=True)
        app.notify(f"Error creating branch: {str(e)}", severity="error")


async def handle_switch_branch(app: 'TldwCli', target_branch_id: str) -> None:
    """
    Switch to a different conversation branch.
    
    Args:
        app: Application instance
        target_branch_id: ID of the branch to switch to
    """
    logger.info(f"Switching to branch {target_branch_id}")
    
    if not app.chachanotes_db:
        logger.error("Database not available")
        app.notify("Database error: Cannot switch branch", severity="error")
        return
    
    try:
        # Validate the switch
        if app.current_chat_conversation_id:
            success, error = navigate_to_branch(
                app.chachanotes_db,
                app.current_chat_conversation_id,
                target_branch_id
            )
            
            if not success:
                logger.warning(f"Cannot switch to branch: {error}")
                app.notify(f"Cannot switch to branch: {error}", severity="warning")
                return
        
        # Load the target branch conversation
        await switch_to_branch(app, target_branch_id)
        
    except Exception as e:
        logger.error(f"Error switching branch: {e}", exc_info=True)
        app.notify(f"Error switching branch: {str(e)}", severity="error")


async def handle_show_branch_tree(app: 'TldwCli') -> None:
    """
    Show the branch tree view for the current conversation.
    
    Args:
        app: Application instance
    """
    logger.debug("Showing branch tree view")
    
    if not app.current_chat_conversation_id:
        app.notify("No active conversation", severity="warning")
        return
    
    if not app.chachanotes_db:
        logger.error("Database not available")
        return
    
    try:
        # Get branch information
        branch_info = get_branch_info(
            app.chachanotes_db,
            app.current_chat_conversation_id
        )
        
        if not branch_info:
            logger.warning("No branch information available")
            app.notify("No branch information available", severity="warning")
            return
        
        # Check if we have a branch tree widget in the UI
        try:
            branch_tree = app.query_one("#chat-branch-tree", BranchTreeView)
            branch_tree.update_branch_tree(branch_info)
            
            # Make sure it's visible
            branch_tree.remove_class("hidden")
            
        except QueryError:
            logger.warning("Branch tree widget not found in UI")
            # Could create a modal or popup here
            app.notify("Branch view not available in current UI", severity="warning")
            
    except Exception as e:
        logger.error(f"Error showing branch tree: {e}", exc_info=True)
        app.notify("Error displaying branch tree", severity="error")


async def handle_navigate_message_branches(app: 'TldwCli', message_id: str, direction: str) -> None:
    """
    Navigate between alternative message branches.
    
    Args:
        app: Application instance
        message_id: Parent message ID
        direction: 'next' or 'previous'
    """
    logger.debug(f"Navigating message branches for {message_id} ({direction})")
    
    if not app.chachanotes_db or not app.current_chat_conversation_id:
        return
    
    try:
        # Get all branches for this message
        branches = get_message_branches(
            app.chachanotes_db,
            app.current_chat_conversation_id,
            message_id
        )
        
        if len(branches) <= 1:
            logger.debug("No alternative branches for this message")
            return
        
        # Find current message in branches
        current_idx = -1
        chat_log = app.query_one("#chat-log", VerticalScroll)
        
        # This is simplified - in reality we'd need to track which branch is currently shown
        # and navigate accordingly
        
        app.notify(f"Message has {len(branches)} alternatives", severity="information")
        
    except Exception as e:
        logger.error(f"Error navigating message branches: {e}", exc_info=True)


#
# Helper Functions:

async def switch_to_branch(app: 'TldwCli', branch_id: str) -> None:
    """
    Switch the UI to display a different conversation branch.
    
    Args:
        app: Application instance
        branch_id: ID of the branch to switch to
    """
    # Import the display function from chat_events
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import display_conversation_in_chat_tab_ui
    
    # Use the existing conversation display function
    await display_conversation_in_chat_tab_ui(app, branch_id)
    
    # Update branch indicators
    await update_branch_tree_view(app)


async def update_message_branch_indicators(app: 'TldwCli', conversation_id: str) -> None:
    """
    Update branch indicators on messages that have branches.
    
    Args:
        app: Application instance
        conversation_id: Current conversation ID
    """
    try:
        chat_log = app.query_one("#chat-log", VerticalScroll)
        
        # Get all messages in the conversation
        if not app.chachanotes_db:
            return
            
        messages = app.chachanotes_db.get_messages_for_conversation(conversation_id)
        
        # Build a map of messages that have branches
        message_branch_count = {}
        for msg in messages:
            parent_id = msg.get('parent_message_id')
            if parent_id:
                message_branch_count[parent_id] = message_branch_count.get(parent_id, 0) + 1
        
        # Update UI indicators
        # This would require modifying ChatMessage widgets to show branch indicators
        # For now, just log the information
        for msg_id, count in message_branch_count.items():
            if count > 1:
                logger.debug(f"Message {msg_id} has {count} branches")
                
    except Exception as e:
        logger.error(f"Error updating branch indicators: {e}", exc_info=True)


async def update_branch_tree_view(app: 'TldwCli') -> None:
    """
    Update the branch tree view if it exists.
    
    Args:
        app: Application instance
    """
    try:
        branch_tree = app.query_one("#chat-branch-tree", BranchTreeView)
        
        if app.current_chat_conversation_id and app.chachanotes_db:
            branch_info = get_branch_info(
                app.chachanotes_db,
                app.current_chat_conversation_id
            )
            
            if branch_info:
                branch_tree.update_branch_tree(branch_info)
                
    except QueryError:
        # Branch tree not in current UI
        pass
    except Exception as e:
        logger.error(f"Error updating branch tree: {e}", exc_info=True)


async def confirm_switch_to_branch(app: 'TldwCli') -> bool:
    """
    Ask user to confirm switching to a new branch.
    
    Args:
        app: Application instance
        
    Returns:
        True if user confirms, False otherwise
    """
    # For now, return True automatically
    # In a full implementation, this would show a confirmation dialog
    return True


#
# Button Handler Mappings:

CHAT_BRANCH_BUTTON_HANDLERS = {
    "chat-create-branch": lambda app, event: handle_create_branch_from_current(app),
    "chat-show-branches": lambda app, event: handle_show_branch_tree(app),
    "chat-branch-previous": lambda app, event: handle_navigate_branches(app, "previous"),
    "chat-branch-next": lambda app, event: handle_navigate_branches(app, "next"),
}


async def handle_create_branch_from_current(app: 'TldwCli') -> None:
    """Create a branch from the current conversation state."""
    if not app.current_chat_conversation_id:
        app.notify("No active conversation to branch from", severity="warning")
        return
    
    # Get the last message ID
    try:
        chat_log = app.query_one("#chat-log", VerticalScroll)
        messages = list(chat_log.query(ChatMessage))
        
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'message_id') and last_message.message_id:
                await handle_create_branch_from_message(
                    app, 
                    last_message.message_id,
                    last_message
                )
            else:
                app.notify("Cannot determine last message ID", severity="error")
        else:
            app.notify("No messages in conversation", severity="warning")
            
    except Exception as e:
        logger.error(f"Error creating branch: {e}", exc_info=True)
        app.notify("Error creating branch", severity="error")


async def handle_navigate_branches(app: 'TldwCli', direction: str) -> None:
    """Navigate between conversation branches."""
    if not app.current_chat_conversation_id or not app.chachanotes_db:
        return
    
    try:
        # Get branch info
        branch_info = get_branch_info(
            app.chachanotes_db,
            app.current_chat_conversation_id
        )
        
        siblings = branch_info.get('siblings', [])
        if not siblings:
            app.notify("No sibling branches available", severity="information")
            return
        
        # For simplicity, just notify about siblings
        # Full implementation would track current position and navigate
        app.notify(f"Found {len(siblings)} sibling branches", severity="information")
        
    except Exception as e:
        logger.error(f"Error navigating branches: {e}", exc_info=True)


#
# End of chat_branch_events.py
########################################################################################################################