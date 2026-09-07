# tab_events.py
# Description:
#
# Imports
import logging
from typing import TYPE_CHECKING

from textual.widgets import Button

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


async def handle_tab_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Handles tab switching button presses."""
    button_id = event.button.id
    new_tab_id = button_id.replace("tab-", "")
    logging.info(f"Tab button {button_id} pressed. Requesting switch to '{new_tab_id}'")
    if new_tab_id != app.current_tab:
        app.current_tab = (
            new_tab_id  # This will trigger the watch_current_tab method in TldwCli
        )
    else:
        logging.debug(f"Already on tab '{new_tab_id}'. Ignoring.")


# --- Button Handler Map ---
TAB_BUTTON_HANDLERS = {
    "tab-chat": handle_tab_button_pressed,
    "tab-notes": handle_tab_button_pressed,
    "tab-ccp": handle_tab_button_pressed,
    "tab-media": handle_tab_button_pressed,
    "tab-llm-management": handle_tab_button_pressed,
    "tab-ingest": handle_tab_button_pressed,
    "tab-app-log": handle_tab_button_pressed,
}

#
# End of tab_events.py
########################################################################################################################
