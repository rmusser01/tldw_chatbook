# Chat_Branching.py
# Description: Core functionality for managing chat conversation branches
"""
This module provides functions for creating, navigating, and managing branches
in chat conversations. It supports both message-level and conversation-level
branching operations.
"""
#
# Imports
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
#
# 3rd-party Libraries
from loguru import logger
#
# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, InputError, ConflictError

# Configure logger with context
logger = logger.bind(module="Chat_Branching")

#
####################################################################################################
#
# Functions:

def create_conversation_branch(
    db: CharactersRAGDB,
    parent_conversation_id: str,
    forked_from_message_id: str,
    character_id: int,
    branch_title: Optional[str] = None,
    client_id: Optional[str] = None
) -> Optional[str]:
    """
    Create a new conversation branch from a specific message in an existing conversation.
    
    Args:
        db: Database instance
        parent_conversation_id: ID of the conversation to branch from
        forked_from_message_id: ID of the message to branch from
        character_id: Character ID for the new branch
        branch_title: Optional title for the branch
        client_id: Optional client ID, uses db.client_id if not provided
        
    Returns:
        The ID of the newly created branch conversation, or None on error
    """
    try:
        # Get parent conversation details
        parent_conv = db.get_conversation_by_id(parent_conversation_id)
        if not parent_conv:
            logger.error(f"Parent conversation {parent_conversation_id} not found")
            return None
            
        # Get the root_id from parent (branches share the same root)
        root_id = parent_conv.get('root_id', parent_conversation_id)
        
        # Generate title if not provided
        if not branch_title:
            parent_title = parent_conv.get('title', 'Untitled')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            branch_title = f"{parent_title} - Branch at {timestamp}"
        
        # Create the new branch conversation
        conv_data = {
            'root_id': root_id,
            'forked_from_message_id': forked_from_message_id,
            'parent_conversation_id': parent_conversation_id,
            'character_id': character_id,
            'title': branch_title,
            'client_id': client_id or db.client_id
        }
        
        new_conv_id = db.add_conversation(conv_data)
        logger.info(f"Created branch conversation {new_conv_id} from parent {parent_conversation_id}")
        
        # Copy messages up to the fork point
        copy_success = copy_messages_to_branch(
            db, 
            parent_conversation_id, 
            new_conv_id, 
            forked_from_message_id
        )
        
        if not copy_success:
            logger.warning(f"Failed to copy messages to branch {new_conv_id}")
            # Could optionally delete the branch here
            
        return new_conv_id
        
    except CharactersRAGDBError as e:
        logger.error(f"Database error creating branch: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating branch: {e}", exc_info=True)
        return None


def copy_messages_to_branch(
    db: CharactersRAGDB,
    source_conv_id: str,
    target_conv_id: str,
    up_to_message_id: str
) -> bool:
    """
    Copy messages from source conversation to target up to a specific message.
    
    Args:
        db: Database instance
        source_conv_id: Source conversation ID
        target_conv_id: Target conversation ID (the branch)
        up_to_message_id: Copy messages up to and including this message ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get all messages from source conversation
        messages = db.get_messages_for_conversation(source_conv_id)
        if not messages:
            logger.warning(f"No messages found in source conversation {source_conv_id}")
            return True  # Not an error, just nothing to copy
            
        # Build message tree to find path to fork point
        message_tree = build_message_tree(messages)
        messages_to_copy = get_messages_up_to(messages, up_to_message_id, message_tree)
        
        # Copy each message to the new conversation
        for msg in messages_to_copy:
            new_msg_data = {
                'conversation_id': target_conv_id,
                'parent_message_id': msg.get('parent_message_id'),
                'sender': msg['sender'],
                'content': msg['content'],
                'image_data': msg.get('image_data'),
                'image_mime_type': msg.get('image_mime_type'),
                'ranking': msg.get('ranking'),
                'client_id': msg.get('client_id', db.client_id)
            }
            
            # Use the same ID to preserve parent-child relationships
            new_msg_data['id'] = msg['id']
            
            db.add_message(new_msg_data)
            
        logger.info(f"Copied {len(messages_to_copy)} messages to branch {target_conv_id}")
        return True
        
    except CharactersRAGDBError as e:
        logger.error(f"Database error copying messages: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error copying messages: {e}", exc_info=True)
        return False


def get_conversation_branches(
    db: CharactersRAGDB,
    root_id: str
) -> List[Dict[str, Any]]:
    """
    Get all conversation branches for a given root conversation.
    
    Args:
        db: Database instance
        root_id: Root conversation ID
        
    Returns:
        List of conversation dictionaries that share the same root
    """
    try:
        # Use the new database method
        branches = db.get_conversation_branches(root_id)
        logger.info(f"Found {len(branches)} branches for root {root_id}")
        return branches
        
    except CharactersRAGDBError as e:
        logger.error(f"Database error getting branches: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting branches: {e}", exc_info=True)
        return []


def get_message_branches(
    db: CharactersRAGDB,
    conversation_id: str,
    parent_message_id: str
) -> List[Dict[str, Any]]:
    """
    Get all alternative message branches for a given parent message.
    
    Args:
        db: Database instance
        conversation_id: Conversation ID
        parent_message_id: Parent message ID
        
    Returns:
        List of messages that are children of the parent message
    """
    try:
        messages = db.get_messages_for_conversation(conversation_id)
        
        # Filter messages that have the specified parent
        branches = [
            msg for msg in messages 
            if msg.get('parent_message_id') == parent_message_id
        ]
        
        # Sort by ranking (preferred) or timestamp
        branches.sort(key=lambda x: (x.get('ranking', 0), x.get('timestamp', '')))
        
        logger.debug(f"Found {len(branches)} message branches for parent {parent_message_id}")
        return branches
        
    except CharactersRAGDBError as e:
        logger.error(f"Database error getting message branches: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting message branches: {e}", exc_info=True)
        return []


def build_message_tree(messages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build a tree structure of message parent-child relationships.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dictionary mapping parent_message_id to list of child message IDs
    """
    tree = {}
    
    for msg in messages:
        parent_id = msg.get('parent_message_id')
        msg_id = msg['id']
        
        if parent_id not in tree:
            tree[parent_id] = []
        tree[parent_id].append(msg_id)
        
    return tree


def get_messages_up_to(
    messages: List[Dict[str, Any]],
    target_message_id: str,
    message_tree: Optional[Dict[str, List[str]]] = None
) -> List[Dict[str, Any]]:
    """
    Get all messages in the conversation up to and including the target message.
    This follows the conversation path from the beginning to the target.
    
    Args:
        messages: All messages in the conversation
        target_message_id: The message to stop at (inclusive)
        message_tree: Optional pre-built message tree
        
    Returns:
        List of messages in chronological order up to target
    """
    # Create message lookup
    msg_lookup = {msg['id']: msg for msg in messages}
    
    if target_message_id not in msg_lookup:
        logger.warning(f"Target message {target_message_id} not found")
        return []
    
    # Trace back from target to root
    path_messages = []
    current_msg = msg_lookup[target_message_id]
    
    while current_msg:
        path_messages.append(current_msg)
        parent_id = current_msg.get('parent_message_id')
        current_msg = msg_lookup.get(parent_id) if parent_id else None
    
    # Reverse to get chronological order
    path_messages.reverse()
    
    return path_messages


def get_branch_info(
    db: CharactersRAGDB,
    conversation_id: str
) -> Dict[str, Any]:
    """
    Get detailed branch information for a conversation.
    
    Args:
        db: Database instance
        conversation_id: Conversation ID
        
    Returns:
        Dictionary with branch details including siblings, parent, children
    """
    try:
        conv = db.get_conversation_by_id(conversation_id)
        if not conv:
            return {}
            
        root_id = conv.get('root_id', conversation_id)
        parent_id = conv.get('parent_conversation_id')
        fork_msg_id = conv.get('forked_from_message_id')
        
        # Get all branches with same root
        all_branches = get_conversation_branches(db, root_id)
        
        # Find siblings (same parent)
        siblings = [
            b for b in all_branches 
            if b.get('parent_conversation_id') == parent_id and b['id'] != conversation_id
        ] if parent_id else []
        
        # Find children using the new DB method
        children = db.get_child_conversations(conversation_id)
        
        # Get message branch info
        message_branches = db.get_messages_with_branches(conversation_id)
        
        return {
            'conversation_id': conversation_id,
            'root_id': root_id,
            'parent_id': parent_id,
            'forked_from_message_id': fork_msg_id,
            'siblings': siblings,
            'children': children,
            'all_branches': all_branches,
            'message_branches': message_branches,
            'total_branches': len(all_branches),
            'is_root': conversation_id == root_id,
            'is_branch': parent_id is not None
        }
        
    except Exception as e:
        logger.error(f"Error getting branch info: {e}", exc_info=True)
        return {}


def navigate_to_branch(
    db: CharactersRAGDB,
    current_conv_id: str,
    target_branch_id: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate navigation from current conversation to target branch.
    
    Args:
        db: Database instance
        current_conv_id: Current conversation ID
        target_branch_id: Target branch conversation ID
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        current = db.get_conversation_by_id(current_conv_id)
        target = db.get_conversation_by_id(target_branch_id)
        
        if not current or not target:
            return False, "Conversation not found"
            
        # Check if they share the same root
        if current.get('root_id', current_conv_id) != target.get('root_id', target_branch_id):
            return False, "Conversations are not related"
            
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating branch navigation: {e}")
        return False, str(e)


def merge_branches(
    db: CharactersRAGDB,
    source_branch_id: str,
    target_branch_id: str,
    merge_point_message_id: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Merge messages from one branch into another.
    This is a complex operation that should be used carefully.
    
    Args:
        db: Database instance
        source_branch_id: Branch to merge from
        target_branch_id: Branch to merge into
        merge_point_message_id: Optional specific message to merge at
        
    Returns:
        Tuple of (success, error_message)
    """
    # This is a placeholder for a more complex merge operation
    # Full implementation would handle:
    # - Conflict resolution
    # - Message reordering
    # - Parent-child relationship updates
    # - User confirmation UI
    
    logger.warning("Branch merging not fully implemented yet")
    return False, "Branch merging coming soon"


#
# End of Chat_Branching.py
####################################################################################################