# Tab_Bar.py
# Description: This file contains the UI functions for the tab bar
#
# Imports
from typing import TYPE_CHECKING, List
#
# Third-Party Imports
from textual.app import ComposeResult
from textual.containers import Horizontal, HorizontalScroll
from textual.widgets import Button
#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli  # Not strictly needed for compose but good for context

from ..Constants import TAB_CCP, TAB_TOOLS_SETTINGS, TAB_INGEST, TAB_LLM, TAB_EVALS, TAB_CODING, TAB_STTS # Added import
#
#######################################################################################################################
#
# Functions:

class TabBar(Horizontal):  # The outer container for the tab bar
    """
    A custom widget for the application's tab bar.
    """

    def __init__(self, tab_ids: List[str], initial_active_tab: str, **kwargs):
        super().__init__(**kwargs)
        self.tab_ids = tab_ids
        self.initial_active_tab = initial_active_tab
        self.id = "tabs-outer-container"  # Matches CSS

    def compose(self) -> ComposeResult:
        with HorizontalScroll(id="tabs"):  # Inner scrollable area
            for tab_id_loop in self.tab_ids:
                # Determine label based on tab_id (matches logic in app.py)
                if tab_id_loop == TAB_CCP:
                    label_text = "CCP"
                elif tab_id_loop == TAB_TOOLS_SETTINGS:
                    label_text = "Settings"
                elif tab_id_loop == TAB_INGEST:
                    label_text = "Ingest Content"
                elif tab_id_loop == TAB_LLM:
                    label_text = "LLM Management"
                elif tab_id_loop == TAB_EVALS:
                    label_text = "Evals"
                elif tab_id_loop == TAB_CODING:
                    label_text = "Coding"
                elif tab_id_loop == TAB_STTS:
                    label_text = "S/TT/S"
                else:
                    label_text = tab_id_loop.replace('_', ' ').capitalize()

                yield Button(
                    label_text,
                    id=f"tab-{tab_id_loop}",
                    classes="-active" if tab_id_loop == self.initial_active_tab else ""
                )

#
# End of Tab_Bar.py
#######################################################################################################################
