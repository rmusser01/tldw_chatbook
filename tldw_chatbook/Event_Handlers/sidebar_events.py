# side_bar_events.py
# Description:
#
# Imports
import logging
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:

async def handle_sidebar_toggle_button_pressed(app: 'TldwCli', button_id: str) -> None:
    """Handles all sidebar toggle button presses."""
    # --- Chat Sidebar ---
    if button_id == "toggle-chat-left-sidebar":
        app.chat_sidebar_collapsed = not app.chat_sidebar_collapsed
        logging.debug("Chat sidebar now %s", "collapsed" if app.chat_sidebar_collapsed else "expanded")
    elif button_id == "toggle-chat-right-sidebar":
        app.chat_right_sidebar_collapsed = not app.chat_right_sidebar_collapsed
        logging.debug("Character sidebar now %s", "collapsed" if app.chat_right_sidebar_collapsed else "expanded")
    # --- Notes Sidebars ---
    elif button_id == "toggle-notes-sidebar-left":
        app.notes_sidebar_left_collapsed = not app.notes_sidebar_left_collapsed
        logging.debug("Notes left sidebar now %s", "collapsed" if app.notes_sidebar_left_collapsed else "expanded")
    elif button_id == "toggle-notes-sidebar-right":
        app.notes_sidebar_right_collapsed = not app.notes_sidebar_right_collapsed
        logging.debug("Notes right sidebar now %s", "collapsed" if app.notes_sidebar_right_collapsed else "expanded")
    elif button_id == "toggle-notes-sidebar-right":
        app.notes_sidebar_right_collapsed = not app.notes_sidebar_right_collapsed
        logging.debug(f"Notes right sidebar collapsed state: {app.notes_sidebar_right_collapsed}")
    # --- Conversation Character Sidebars ---
    elif button_id == "toggle-conv-char-left-sidebar":
        app.conv_char_sidebar_left_collapsed = not app.conv_char_sidebar_left_collapsed
        logging.debug("CCP left sidebar now %s", "collapsed" if app.conv_char_sidebar_left_collapsed else "expanded")
    elif button_id == "toggle-conv-char-right-sidebar":
        app.conv_char_sidebar_right_collapsed = not app.conv_char_sidebar_right_collapsed
        logging.debug("CCP right sidebar now %s", "collapsed" if app.conv_char_sidebar_right_collapsed else "expanded")
    else:
        logging.warning(f"Unhandled sidebar toggle button ID: {button_id}")

# --- Button Handler Map ---
SIDEBAR_BUTTON_HANDLERS = {
    # Chat
    "toggle-chat-left-sidebar": handle_sidebar_toggle_button_pressed,
    "toggle-chat-right-sidebar": handle_sidebar_toggle_button_pressed,
    # Notes
    "toggle-notes-sidebar-left": handle_sidebar_toggle_button_pressed,
    "toggle-notes-sidebar-right": handle_sidebar_toggle_button_pressed,
    # CCP
    "toggle-conv-char-left-sidebar": handle_sidebar_toggle_button_pressed,
    "toggle-conv-char-right-sidebar": handle_sidebar_toggle_button_pressed,
}


#
# End of side_bar_events.py
########################################################################################################################
