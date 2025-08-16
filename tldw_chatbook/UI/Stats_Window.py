# tldw_chatbook/UI/Stats_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# Third-party Imports
from textual.app import ComposeResult
from textual.containers import Container # Use Container as the base for the window
#
# Local Imports
from tldw_chatbook.UI.Screens.stats_screen import StatsScreen # Import the actual screen content
# from ..Constants import TAB_STATS # Not strictly needed
if TYPE_CHECKING:
    pass
#
#######################################################################################################################
#
# Functions:

class StatsWindow(Container): # Inherit from Container
    """
    Container for the Stats Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance # Not strictly used in compose below

    def compose(self) -> ComposeResult:
        yield StatsScreen(id="stats_screen_content")

#
# End of Stats_Window.py
#######################################################################################################################
