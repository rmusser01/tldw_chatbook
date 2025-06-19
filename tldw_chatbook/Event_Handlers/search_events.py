# tldw_chatbook/Event_Handlers/search_events.py
#
# Event handlers for Search tab functionality
#
# Imports
from typing import TYPE_CHECKING
from loguru import logger
from textual.widgets import Button
#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli

# Configure logger with context
logger = logger.bind(module="search_events")

# Button handler functions
async def handle_search_button_pressed(app: "TldwCli", button: Button) -> None:
    """
    Handle button presses within the Search tab.
    This function delegates to the SearchWindow's button handler.
    """
    button_id = button.id
    if not button_id:
        return
    
    logger.debug(f"Search button pressed: {button_id}")
    
    # Get the SearchWindow instance and delegate the button handling to it
    try:
        search_window = app.query_one("#search-window")
        if hasattr(search_window, 'on_button_pressed'):
            # Create a mock event object that SearchWindow expects
            class MockButtonEvent:
                def __init__(self, button):
                    self.button = button
                def stop(self):
                    pass
            
            mock_event = MockButtonEvent(button)
            search_window.on_button_pressed(mock_event)
    except Exception as e:
        logger.error(f"Error handling search button press: {e}", exc_info=True)
        app.notify(f"Error handling search button: {str(e)}", severity="error")

# Export button handlers map
SEARCH_BUTTON_HANDLERS = {
    # These button IDs are handled by SearchWindow's internal @on decorators
    # We just need to ensure button events get routed to SearchWindow
    "creation-create-embeddings-button": handle_search_button_pressed,
    "creation-refresh-list-button": handle_search_button_pressed,
    "mgmt-refresh-list-button": handle_search_button_pressed,
    "mgmt-delete-item-embeddings-button": handle_search_button_pressed,
    "mgmt-delete-collection-button": handle_search_button_pressed,
    "web-search-button": handle_search_button_pressed,
    # Add handlers for SearchRAGWindow buttons
    "rag-search-btn": handle_search_button_pressed,
    "index-content-btn": handle_search_button_pressed,
    "clear-cache-btn": handle_search_button_pressed,
}