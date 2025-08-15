# Logs_Window.py
# Description: This file contains the UI functions for the Logs_Window tab
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container  # Use Container as the base for the window
from textual.widgets import RichLog, Button
#
# Local Imports
# from ..Constants import TAB_LOGS # Not strictly needed
if TYPE_CHECKING:
    from ..app import TldwCli
#
#########################################################################################################################
#
# Functions:

class LogsWindow(Container): # Inherit from Container
    """
    Container for the Logs Tab's UI.
    """
    DEFAULT_CSS = """
    LogsWindow {
        layout: vertical; /* Ensure the window itself has a vertical layout */
    }
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance # Not strictly used in compose below

    def compose(self) -> ComposeResult:
        # Create RichLog with max_lines for performance (keep last 10000 lines)
        yield RichLog(
            id="app-log-display", 
            wrap=True, 
            highlight=True, 
            markup=True, 
            auto_scroll=True,
            max_lines=10000  # Limit for performance while still keeping plenty of history
        )
        yield Button("Copy All Logs to Clipboard", id="copy-logs-button", classes="logs-action-button")

#
# End of Logs_Window.py
#######################################################################################################################
