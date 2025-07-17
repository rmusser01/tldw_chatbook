"""
Event handlers for chat dictionary UI interactions in the chat sidebar.
"""

import logging
from typing import TYPE_CHECKING, List, Dict, Any, Optional
from loguru import logger as loguru_logger
from textual.widgets import ListItem, Input, ListView, TextArea, Button, Label, Checkbox
from textual.css.query import QueryError

from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import ChatDictionary
from tldw_chatbook.config import get_cli_setting

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


async def handle_dictionary_search_input(app: 'TldwCli', search_term: str) -> None:
    """
    Handle chat dictionary search input changes.
    
    Args:
        app: The application instance
        search_term: The search term entered by the user
    """
    loguru_logger.debug(f"Searching chat dictionaries with term: '{search_term}'")
    
    try:
        available_list = app.query_one("#chat-dictionary-available-listview", ListView)
        await available_list.clear()
        
        if not app.db:
            loguru_logger.error("Database not initialized")
            app.notify("Database not initialized", severity="error")
            return
            
        dict_lib = ChatDictionary(app.db)
        
        # Search dictionaries
        if search_term.strip():
            dictionaries = dict_lib.search_dictionaries(search_term)
        else:
            # Get all dictionaries if no search term
            dictionaries = dict_lib.list_all_dictionaries()
        
        # Populate the list
        if not dictionaries:
            await available_list.append(ListItem(Label("No dictionaries found.")))
        else:
            for dictionary in dictionaries:
                # Get entry count for display
                entries = dict_lib.get_dictionary_entries(dictionary['id'])
                entry_count = len(entries) if entries else 0
                
                display_label = f"{dictionary['name']} ({entry_count} entries)"
                if dictionary.get('description'):
                    display_label += f" - {dictionary['description'][:30]}..."
                
                list_item = ListItem(Label(display_label))
                # Store the dictionary data on the list item for later use
                setattr(list_item, 'dictionary_data', dictionary)
                await available_list.append(list_item)
                
    except QueryError as e:
        loguru_logger.error(f"Error querying dictionary UI elements: {e}")
        app.notify(f"Error accessing dictionary UI: {e}", severity="error")
    except Exception as e:
        loguru_logger.error(f"Exception during dictionary search: {e}", exc_info=True)
        app.notify(f"Error during dictionary search: {e}", severity="error")


async def handle_dictionary_add_button(app: 'TldwCli') -> None:
    """
    Handle adding a dictionary to the current conversation.
    """
    loguru_logger.debug("Adding dictionary to conversation")
    
    try:
        # Get selected dictionary from available list
        available_list = app.query_one("#chat-dictionary-available-listview", ListView)
        selected_item = available_list.highlighted_child
        
        if not selected_item or not hasattr(selected_item, 'dictionary_data'):
            app.notify("Please select a dictionary to add", severity="warning")
            return
            
        dictionary_data = getattr(selected_item, 'dictionary_data')
        
        # Get the current conversation ID
        conv_id = getattr(app, 'current_chat_conversation_id', None)
        if not conv_id:
            app.notify("No active conversation. Please start or load a conversation first.", severity="warning")
            return
            
        if not app.db:
            loguru_logger.error("Database not initialized")
            app.notify("Database not initialized", severity="error")
            return
            
        # Add association
        dict_lib = ChatDictionary(app.db)
        success = dict_lib.link_dictionary_to_conversation(
            dictionary_id=dictionary_data['id'],
            conversation_id=conv_id
        )
        
        if success:
            app.notify(f"Added '{dictionary_data['name']}' to conversation", severity="information")
            # Refresh the active dictionaries list
            await refresh_active_dictionaries(app)
        else:
            app.notify("Failed to add dictionary to conversation", severity="error")
            
    except Exception as e:
        loguru_logger.error(f"Error adding dictionary: {e}", exc_info=True)
        app.notify(f"Error adding dictionary: {e}", severity="error")


async def handle_dictionary_remove_button(app: 'TldwCli') -> None:
    """
    Handle removing a dictionary from the current conversation.
    """
    loguru_logger.debug("Removing dictionary from conversation")
    
    try:
        # Get selected dictionary from active list
        active_list = app.query_one("#chat-dictionary-active-listview", ListView)
        selected_item = active_list.highlighted_child
        
        if not selected_item or not hasattr(selected_item, 'dictionary_data'):
            app.notify("Please select a dictionary to remove", severity="warning")
            return
            
        dictionary_data = getattr(selected_item, 'dictionary_data')
        
        # Get the current conversation ID
        conv_id = getattr(app, 'current_chat_conversation_id', None)
        if not conv_id:
            app.notify("No active conversation", severity="warning")
            return
            
        if not app.db:
            loguru_logger.error("Database not initialized")
            app.notify("Database not initialized", severity="error")
            return
            
        # Remove association
        dict_lib = ChatDictionary(app.db)
        success = dict_lib.unlink_dictionary_from_conversation(
            dictionary_id=dictionary_data['id'],
            conversation_id=conv_id
        )
        
        if success:
            app.notify(f"Removed '{dictionary_data['name']}' from conversation", severity="information")
            # Refresh the active dictionaries list
            await refresh_active_dictionaries(app)
        else:
            app.notify("Failed to remove dictionary from conversation", severity="error")
            
    except Exception as e:
        loguru_logger.error(f"Error removing dictionary: {e}", exc_info=True)
        app.notify(f"Error removing dictionary: {e}", severity="error")


async def refresh_active_dictionaries(app: 'TldwCli') -> None:
    """
    Refresh the list of active dictionaries for the current conversation.
    """
    loguru_logger.debug("Refreshing active dictionaries list")
    
    try:
        active_list = app.query_one("#chat-dictionary-active-listview", ListView)
        await active_list.clear()
        
        # Get the current conversation ID
        conv_id = getattr(app, 'current_chat_conversation_id', None)
        if not conv_id:
            await active_list.append(ListItem(Label("No active conversation")))
            return
            
        if not app.db:
            loguru_logger.error("Database not initialized")
            await active_list.append(ListItem(Label("Database not initialized")))
            return
            
        dict_lib = ChatDictionary(app.db)
        
        # Get dictionaries for this conversation
        active_dicts = dict_lib.get_conversation_dictionaries(conversation_id=conv_id)
        
        if not active_dicts:
            await active_list.append(ListItem(Label("No dictionaries associated")))
        else:
            for dictionary in active_dicts:
                # Get entry count
                entries = dict_lib.get_dictionary_entries(dictionary['id'])
                entry_count = len(entries) if entries else 0
                
                display_label = f"{dictionary['name']} ({entry_count} entries)"
                
                list_item = ListItem(Label(display_label))
                # Store the dictionary data on the list item
                setattr(list_item, 'dictionary_data', dictionary)
                await active_list.append(list_item)
                
    except QueryError as e:
        loguru_logger.error(f"Error querying active dictionaries UI: {e}")
    except Exception as e:
        loguru_logger.error(f"Error refreshing active dictionaries: {e}", exc_info=True)


async def handle_dictionary_selection(app: 'TldwCli', list_view_id: str) -> None:
    """
    Handle selection of a dictionary in either available or active list.
    Updates the details display and enables/disables appropriate buttons.
    """
    loguru_logger.debug(f"Handling dictionary selection from {list_view_id}")
    
    try:
        list_view = app.query_one(f"#{list_view_id}", ListView)
        selected_item = list_view.highlighted_child
        
        # Update button states based on which list was selected
        add_button = app.query_one("#chat-dictionary-add-button", Button)
        remove_button = app.query_one("#chat-dictionary-remove-button", Button)
        
        if list_view_id == "chat-dictionary-available-listview":
            add_button.disabled = selected_item is None or not hasattr(selected_item, 'dictionary_data')
            remove_button.disabled = True
        else:  # active list
            add_button.disabled = True
            remove_button.disabled = selected_item is None or not hasattr(selected_item, 'dictionary_data')
        
        # Update details display
        details_display = app.query_one("#chat-dictionary-details-display", TextArea)
        
        if not selected_item or not hasattr(selected_item, 'dictionary_data'):
            details_display.clear()
            return
            
        dictionary_data = getattr(selected_item, 'dictionary_data')
        
        if not app.db:
            details_display.text = "Database not initialized"
            return
            
        dict_lib = ChatDictionary(app.db)
        
        # Get dictionary entries for more details
        entries = dict_lib.get_dictionary_entries(dictionary_data['id'])
        entry_count = len(entries) if entries else 0
        
        # Count different entry types
        pre_count = sum(1 for e in entries if e.get('entry_type') == 'preprocessing')
        post_count = sum(1 for e in entries if e.get('entry_type') == 'postprocessing')
        regex_count = sum(1 for e in entries if e.get('use_regex', False))
        
        # Format dictionary details
        details_text = f"Name: {dictionary_data['name']}\n"
        details_text += f"ID: {dictionary_data['id']}\n"
        
        if dictionary_data.get('description'):
            details_text += f"Description: {dictionary_data['description']}\n"
            
        details_text += f"\nStatistics:\n"
        details_text += f"  Total Entries: {entry_count}\n"
        details_text += f"  Pre-processing: {pre_count}\n"
        details_text += f"  Post-processing: {post_count}\n"
        details_text += f"  Regex Patterns: {regex_count}\n"
        
        # Show a few example entries
        if entries and len(entries) > 0:
            details_text += f"\nExample Entries (first 3):\n"
            for i, entry in enumerate(entries[:3]):
                pattern = entry.get('pattern', '')
                replacement = entry.get('replacement', '')
                if len(pattern) > 20:
                    pattern = pattern[:17] + "..."
                if len(replacement) > 20:
                    replacement = replacement[:17] + "..."
                details_text += f"  '{pattern}' → '{replacement}'\n"
        
        details_display.text = details_text
        
    except QueryError as e:
        loguru_logger.error(f"Error handling dictionary selection: {e}")
    except Exception as e:
        loguru_logger.error(f"Error displaying dictionary details: {e}", exc_info=True)


async def handle_dictionary_enable_checkbox(app: 'TldwCli', enabled: bool) -> None:
    """
    Handle the dictionary processing enable/disable checkbox.
    This updates the config setting that controls whether dictionaries are processed at all.
    """
    loguru_logger.debug(f"Setting dictionary processing enabled to: {enabled}")
    
    try:
        # This would typically update a config setting
        # For now, we'll just store it as an app attribute
        app.dictionary_processing_enabled = enabled
        
        status = "enabled" if enabled else "disabled"
        app.notify(f"Dictionary processing {status}", severity="information")
        
    except Exception as e:
        loguru_logger.error(f"Error updating dictionary setting: {e}", exc_info=True)
        app.notify(f"Error updating dictionary setting: {e}", severity="error")


# --- Button Handler Map ---
CHAT_DICTIONARY_BUTTON_HANDLERS = {
    "chat-dictionary-add-button": handle_dictionary_add_button,
    "chat-dictionary-remove-button": handle_dictionary_remove_button,
}