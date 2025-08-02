# Tab_Dropdown.py
# Description: Dropdown navigation widget to replace horizontal tab bar
#
# Imports
from typing import TYPE_CHECKING, List, Tuple
#
# Third-Party Imports
from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Select
from textual.message import Message
#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli

from ..Constants import (
    TAB_CCP, TAB_TOOLS_SETTINGS, TAB_INGEST, TAB_LLM, TAB_EVALS, 
    TAB_CODING, TAB_STTS, TAB_STUDY, TAB_CHATBOOKS, TAB_CHAT,
    TAB_NOTES, TAB_MEDIA, TAB_SEARCH, TAB_LOGS, TAB_STATS,
    TAB_SUBSCRIPTIONS
)
#
#######################################################################################################################
#
# Classes:

class TabChanged(Message):
    """Message emitted when dropdown selection changes"""
    def __init__(self, tab_id: str) -> None:
        self.tab_id = tab_id
        super().__init__()


class TabDropdown(Container):
    """
    A dropdown navigation widget for tab switching.
    Replaces the horizontal tab bar with a compact dropdown selector.
    """

    def __init__(self, tab_ids: List[str], initial_active_tab: str, **kwargs):
        super().__init__(**kwargs)
        self.tab_ids = tab_ids
        self.current_tab = initial_active_tab
        self.id = "tab-dropdown-container"
        # Store the select widget reference for later updates
        self._select_widget = None

    def _get_tab_label(self, tab_id: str) -> str:
        """
        Convert tab ID to user-friendly label.
        Mirrors the logic from TabBar for consistency.
        """
        label_map = {
            TAB_CHAT: "Chat",
            TAB_CCP: "CCP",
            TAB_NOTES: "Notes",
            TAB_MEDIA: "Media",
            TAB_SEARCH: "Search",
            TAB_TOOLS_SETTINGS: "Settings",
            TAB_INGEST: "Ingest Content",
            TAB_LLM: "LLM Management",
            TAB_EVALS: "Evals",
            TAB_CODING: "Coding",
            TAB_STTS: "S/TT/S",
            TAB_STUDY: "Study",
            TAB_CHATBOOKS: "Chatbooks",
            TAB_LOGS: "Logs",
            TAB_STATS: "Stats",
            TAB_SUBSCRIPTIONS: "Subscriptions"
        }
        
        # Return mapped label or format the ID if not in map
        return label_map.get(tab_id, tab_id.replace('_', ' ').title())

    def compose(self) -> ComposeResult:
        """Compose the dropdown widget"""
        # Create options list with (label, value) tuples
        options = [(self._get_tab_label(tab_id), tab_id) for tab_id in self.tab_ids]
        
        # Create the Select widget
        self._select_widget = Select(
            options=options,
            value=self.current_tab,
            id="tab-dropdown-select",
            prompt="Navigate to...",
            allow_blank=False  # Always have a selection
        )
        
        yield self._select_widget

    @on(Select.Changed, "#tab-dropdown-select")
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle dropdown selection changes"""
        if event.value and event.value != self.current_tab:
            self.current_tab = event.value
            # Post a custom message for the app to handle
            self.post_message(TabChanged(event.value))

    def update_active_tab(self, tab_id: str) -> None:
        """
        Update the dropdown to reflect the current active tab.
        This method can be called externally when tabs are switched via other means.
        """
        if self._select_widget and tab_id in self.tab_ids:
            self.current_tab = tab_id
            self._select_widget.value = tab_id

#
# End of Tab_Dropdown.py
#######################################################################################################################