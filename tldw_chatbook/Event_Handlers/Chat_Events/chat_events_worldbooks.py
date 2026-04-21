"""
Event handlers for world book UI interactions in the chat sidebar.
"""

import logging
from typing import TYPE_CHECKING, List, Dict, Any, Optional
from loguru import logger as loguru_logger
from textual.widgets import ListItem, Input, ListView, TextArea, Button, Label, Select, Checkbox
from textual.css.query import QueryError

from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.config import get_cli_setting

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


async def handle_worldbook_search_input(app: 'TldwCli', search_term: str) -> None:
    """
    Handle world book search input changes.
    
    Args:
        app: The application instance
        search_term: The search term entered by the user
    """
    loguru_logger.debug(f"Searching world books with term: '{search_term}'")
    
    try:
        available_list = app.query_one("#chat-worldbook-available-listview", ListView)
        await available_list.clear()
        
        if not app.db:
            loguru_logger.error("Database not initialized")
            app.notify("Database not initialized", severity="error")
            return
            
        wb_manager = WorldBookManager(app.db)
        
        # Get all world books
        all_books = wb_manager.list_world_books(include_disabled=False)
        
        # Filter by search term if provided
        if search_term.strip():
            search_lower = search_term.lower()
            filtered_books = [
                book for book in all_books
                if search_lower in book['name'].lower() or 
                   (book.get('description') and search_lower in book['description'].lower())
            ]
        else:
            filtered_books = all_books
        
        # Populate the list
        if not filtered_books:
            await available_list.append(ListItem(Label("No world books found.")))
        else:
            for book in filtered_books:
                display_label = f"{book['name']} (ID: {book['id']})"
                if book.get('description'):
                    display_label += f" - {book['description'][:50]}..."
                
                list_item = ListItem(Label(display_label))
                # Store the book data on the list item for later use
                setattr(list_item, 'worldbook_data', book)
                await available_list.append(list_item)
                
    except QueryError as e:
        loguru_logger.error(f"Error querying world book UI elements: {e}")
        app.notify(f"Error accessing world book UI: {e}", severity="error")
    except Exception as e:
        loguru_logger.error(f"Exception during world book search: {e}", exc_info=True)
        app.notify(f"Error during world book search: {e}", severity="error")


async def handle_worldbook_add_button(app: 'TldwCli') -> None:
    """
    Handle adding a world book to the current conversation.
    """
    loguru_logger.debug("Adding world book to conversation")
    
    try:
        # Get selected world book from available list
        available_list = app.query_one("#chat-worldbook-available-listview", ListView)
        selected_item = available_list.highlighted_child
        
        if not selected_item or not hasattr(selected_item, 'worldbook_data'):
            app.notify("Please select a world book to add", severity="warning")
            return
            
        worldbook_data = getattr(selected_item, 'worldbook_data')
        
        # Get the current conversation ID
        conv_id = getattr(app, 'active_conversation_id', None)
        if not conv_id:
            app.notify("No active conversation. Please start or load a conversation first.", severity="warning")
            return
            
        # Get priority from select widget
        priority_select = app.query_one("#chat-worldbook-priority-select", Select)
        priority = int(priority_select.value)
        
        if not app.db:
            loguru_logger.error("Database not initialized")
            app.notify("Database not initialized", severity="error")
            return
            
        # Add association
        wb_manager = WorldBookManager(app.db)
        success = wb_manager.associate_world_book_with_conversation(
            conversation_id=conv_id,
            world_book_id=worldbook_data['id'],
            priority=priority
        )
        
        if success:
            app.notify(f"Added '{worldbook_data['name']}' to conversation", severity="information")
            # Refresh the active world books list
            await refresh_active_worldbooks(app)
        else:
            app.notify("Failed to add world book to conversation", severity="error")
            
    except Exception as e:
        loguru_logger.error(f"Error adding world book: {e}", exc_info=True)
        app.notify(f"Error adding world book: {e}", severity="error")


async def handle_worldbook_remove_button(app: 'TldwCli') -> None:
    """
    Handle removing a world book from the current conversation.
    """
    loguru_logger.debug("Removing world book from conversation")
    
    try:
        # Get selected world book from active list
        active_list = app.query_one("#chat-worldbook-active-listview", ListView)
        selected_item = active_list.highlighted_child
        
        if not selected_item or not hasattr(selected_item, 'worldbook_data'):
            app.notify("Please select a world book to remove", severity="warning")
            return
            
        worldbook_data = getattr(selected_item, 'worldbook_data')
        
        # Get the current conversation ID
        conv_id = getattr(app, 'active_conversation_id', None)
        if not conv_id:
            app.notify("No active conversation", severity="warning")
            return
            
        if not app.db:
            loguru_logger.error("Database not initialized")
            app.notify("Database not initialized", severity="error")
            return
            
        # Remove association
        wb_manager = WorldBookManager(app.db)
        success = wb_manager.disassociate_world_book_from_conversation(
            conversation_id=conv_id,
            world_book_id=worldbook_data['id']
        )
        
        if success:
            app.notify(f"Removed '{worldbook_data['name']}' from conversation", severity="information")
            # Refresh the active world books list
            await refresh_active_worldbooks(app)
        else:
            app.notify("Failed to remove world book from conversation", severity="error")
            
    except Exception as e:
        loguru_logger.error(f"Error removing world book: {e}", exc_info=True)
        app.notify(f"Error removing world book: {e}", severity="error")


async def refresh_active_worldbooks(app: 'TldwCli') -> None:
    """
    Refresh the list of active world books for the current conversation.
    """
    loguru_logger.debug("Refreshing active world books list")
    
    try:
        # Try to find the listview in the current screen/chat window context
        try:
            active_list = app.screen.query_one("#chat-worldbook-active-listview", ListView)
        except QueryError:
            try:
                chat_window = app.screen.query_one("#chat-window")
                active_list = chat_window.query_one("#chat-worldbook-active-listview", ListView)
            except QueryError:
                active_list = app.query_one("#chat-worldbook-active-listview", ListView)
        await active_list.clear()
        
        # Get the current conversation ID
        conv_id = getattr(app, 'active_conversation_id', None)
        if not conv_id:
            await active_list.append(ListItem(Label("No active conversation")))
            return
            
        if not app.db:
            loguru_logger.error("Database not initialized")
            await active_list.append(ListItem(Label("Database not initialized")))
            return
            
        wb_manager = WorldBookManager(app.db)
        
        # Get world books for this conversation
        active_books = wb_manager.get_world_books_for_conversation(
            conversation_id=conv_id,
            enabled_only=False  # Show all, including disabled
        )
        
        if not active_books:
            await active_list.append(ListItem(Label("No world books associated")))
        else:
            for book in active_books:
                status = "✓" if book['enabled'] else "✗"
                display_label = f"{status} {book['name']} (Priority: {book['priority']})"
                
                list_item = ListItem(Label(display_label))
                # Store the book data on the list item
                setattr(list_item, 'worldbook_data', book)
                await active_list.append(list_item)
                
    except QueryError as e:
        loguru_logger.error(f"Error querying active world books UI: {e}")
    except Exception as e:
        loguru_logger.error(f"Error refreshing active world books: {e}", exc_info=True)


async def handle_worldbook_selection(app: 'TldwCli', list_view_id: str) -> None:
    """
    Handle selection of a world book in either available or active list.
    Updates the details display and enables/disables appropriate buttons.
    """
    loguru_logger.debug(f"Handling world book selection from {list_view_id}")
    
    try:
        list_view = app.query_one(f"#{list_view_id}", ListView)
        selected_item = list_view.highlighted_child
        
        # Update button states based on which list was selected
        add_button = app.query_one("#chat-worldbook-add-button", Button)
        remove_button = app.query_one("#chat-worldbook-remove-button", Button)
        
        if list_view_id == "chat-worldbook-available-listview":
            add_button.disabled = selected_item is None or not hasattr(selected_item, 'worldbook_data')
            remove_button.disabled = True
        else:  # active list
            add_button.disabled = True
            remove_button.disabled = selected_item is None or not hasattr(selected_item, 'worldbook_data')
        
        # Update details display
        details_display = app.query_one("#chat-worldbook-details-display", TextArea)
        
        if not selected_item or not hasattr(selected_item, 'worldbook_data'):
            details_display.clear()
            return
            
        worldbook_data = getattr(selected_item, 'worldbook_data')
        
        # Format world book details
        details_text = f"Name: {worldbook_data['name']}\n"
        details_text += f"ID: {worldbook_data['id']}\n"
        
        if worldbook_data.get('description'):
            details_text += f"Description: {worldbook_data['description']}\n"
            
        details_text += f"\nSettings:\n"
        details_text += f"  Scan Depth: {worldbook_data.get('scan_depth', 3)}\n"
        details_text += f"  Token Budget: {worldbook_data.get('token_budget', 500)}\n"
        details_text += f"  Recursive: {'Yes' if worldbook_data.get('recursive_scanning') else 'No'}\n"
        details_text += f"  Enabled: {'Yes' if worldbook_data.get('enabled', True) else 'No'}\n"
        
        # If this book has entries, show count
        if 'entries' in worldbook_data:
            entry_count = len(worldbook_data['entries'])
            details_text += f"\nEntries: {entry_count}"
        
        details_display.text = details_text
        
    except QueryError as e:
        loguru_logger.error(f"Error handling world book selection: {e}")
    except Exception as e:
        loguru_logger.error(f"Error displaying world book details: {e}", exc_info=True)


async def handle_worldbook_enable_checkbox(app: 'TldwCli', enabled: bool) -> None:
    """
    Handle the world info processing enable/disable checkbox.
    This updates the config setting that controls whether world info is processed at all.
    """
    loguru_logger.debug(f"Setting world info processing enabled to: {enabled}")
    
    try:
        # This would typically update a config setting
        # For now, we'll just store it as an app attribute
        app.worldinfo_processing_enabled = enabled
        
        status = "enabled" if enabled else "disabled"
        app.notify(f"World info processing {status}", severity="information")
        
    except Exception as e:
        loguru_logger.error(f"Error updating world info setting: {e}", exc_info=True)
        app.notify(f"Error updating world info setting: {e}", severity="error")


# --- Button Handler Map ---
CHAT_WORLDBOOK_BUTTON_HANDLERS = {
    "chat-worldbook-add-button": handle_worldbook_add_button,
    "chat-worldbook-remove-button": handle_worldbook_remove_button,
}