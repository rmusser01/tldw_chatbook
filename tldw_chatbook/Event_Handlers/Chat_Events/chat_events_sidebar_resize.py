# chat_events_sidebar_resize.py
# Description: Handlers for sidebar resize functionality in the chat tab
#
# Imports
import logging
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from loguru import logger as loguru_logger
from textual.widgets import Button
#
# Local Imports
from ...config import save_setting_to_cli_config
#
if TYPE_CHECKING:
    from ...app import TldwCli
#
########################################################################################################################
#
# Functions:

async def handle_sidebar_shrink(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle shrinking the right sidebar width."""
    logger = getattr(app, 'loguru_logger', logging)
    try:
        current_width = app.chat_right_sidebar_width
        new_width = max(15, current_width - 5)  # Minimum 15% width
        app.chat_right_sidebar_width = new_width
        
        # Apply the new width to the sidebar
        try:
            sidebar = app.query_one("#chat-right-sidebar")
            sidebar.styles.width = f"{new_width}%"
        except Exception as query_error:
            logger.error(f"Error querying sidebar: {query_error}")
            raise
        
        # Save the width preference
        save_setting_to_cli_config("chat_defaults", "right_sidebar_width", new_width)
        
        logger.debug(f"Sidebar width decreased to {new_width}%")
    except Exception as e:
        logger.error(f"Error shrinking sidebar: {e}", exc_info=True)


async def handle_sidebar_expand(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle expanding the right sidebar width."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.debug(f"handle_sidebar_expand called from file: {__file__}")
    try:
        current_width = app.chat_right_sidebar_width
        new_width = min(50, current_width + 5)  # Maximum 50% width
        app.chat_right_sidebar_width = new_width
        
        # Apply the new width to the sidebar
        try:
            sidebar = app.query_one("#chat-right-sidebar")
            sidebar.styles.width = f"{new_width}%"
        except Exception as query_error:
            logger.error(f"Error querying sidebar: {query_error}")
            logger.error(f"Error type: {type(query_error)}")
            logger.error(f"Error args: {query_error.args}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        
        # Save the width preference
        save_setting_to_cli_config("chat_defaults", "right_sidebar_width", new_width)
        
        logger.debug(f"Sidebar width increased to {new_width}%")
    except Exception as e:
        logger.error(f"Error expanding sidebar: {e}", exc_info=True)
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception occurred in file: {__file__}")


# Handler map for sidebar resize buttons
CHAT_SIDEBAR_RESIZE_HANDLERS = {
    "chat-sidebar-shrink": handle_sidebar_shrink,
    "chat-sidebar-expand": handle_sidebar_expand,
}

#
# End of chat_events_sidebar_resize.py
########################################################################################################################