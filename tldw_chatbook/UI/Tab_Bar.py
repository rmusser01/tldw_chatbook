# Tab_Bar.py
# Description: Tab bar with visual grouping via separator widgets
#
# Imports
from typing import TYPE_CHECKING, List
#
# Third-Party Imports
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, HorizontalScroll
from textual.widgets import Button, Static
#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli

from ..Constants import (
    TAB_CCP, TAB_TOOLS_SETTINGS, TAB_INGEST, TAB_LLM, TAB_EVALS,
    TAB_CODING, TAB_STTS, TAB_STUDY, TAB_CHATBOOKS, TAB_GROUPS,
)
from ..UI.Navigation.main_navigation import NavigateToScreen
#
#######################################################################################################################
#
# Functions:

# Label overrides for tab display names
_TAB_LABELS = {
    TAB_CCP: "Library",
    TAB_TOOLS_SETTINGS: "Settings",
    TAB_INGEST: "Ingest",
    TAB_LLM: "Models",
    TAB_EVALS: "Evals",
    TAB_CODING: "Coding",
    TAB_STTS: "Speech",
    TAB_STUDY: "Study",
    TAB_CHATBOOKS: "Chatbooks",
}


def _get_tab_label(tab_id: str) -> str:
    """Return the display label for a tab ID."""
    return _TAB_LABELS.get(tab_id, tab_id.replace('_', ' ').capitalize())


class TabBar(Horizontal):
    """
    A custom widget for the application's tab bar with visual grouping.
    Uses screen-based navigation.
    """

    def __init__(self, tab_ids: List[str], initial_active_tab: str, **kwargs):
        super().__init__(**kwargs)
        self.tab_ids = tab_ids
        self.initial_active_tab = initial_active_tab
        self.current_active_tab = initial_active_tab
        self.id = "tabs-outer-container"

    def compose(self) -> ComposeResult:
        with HorizontalScroll(id="tabs"):
            group_index = 0
            for group_name, group_tab_ids in TAB_GROUPS.items():
                visible_tabs = [t for t in group_tab_ids if t in self.tab_ids]
                if not visible_tabs:
                    continue

                # Add group divider between groups
                if group_index > 0:
                    yield Static("  ┃  ", classes="tab-group-separator")

                for tab_id_loop in visible_tabs:
                    label_text = _get_tab_label(tab_id_loop)
                    yield Button(
                        label_text,
                        id=f"tab-{tab_id_loop}",
                        classes="-active" if tab_id_loop == self.initial_active_tab else "",
                    )

                group_index += 1

            # Render any ungrouped tabs
            grouped_tabs = {t for tabs in TAB_GROUPS.values() for t in tabs}
            ungrouped = [t for t in self.tab_ids if t not in grouped_tabs]
            if ungrouped:
                yield Static("  ┃  ", classes="tab-group-separator")
                for tab_id_loop in ungrouped:
                    label_text = _get_tab_label(tab_id_loop)
                    yield Button(
                        label_text,
                        id=f"tab-{tab_id_loop}",
                        classes="-active" if tab_id_loop == self.initial_active_tab else "",
                    )

    @on(Button.Pressed, "Button")
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle tab button presses and navigate to the corresponding screen."""
        button = event.button
        if button.id and button.id.startswith("tab-"):
            tab_id = button.id[4:]
            self._update_active_tab(tab_id)
            screen_name = 'ccp' if tab_id == TAB_CCP else tab_id
            self.post_message(NavigateToScreen(screen_name=screen_name))

    def _update_active_tab(self, new_tab_id: str) -> None:
        """Update the visual state of tab buttons."""
        for button in self.query("Button"):
            button.remove_class("-active")
        try:
            new_button = self.query_one(f"#tab-{new_tab_id}", Button)
            new_button.add_class("-active")
        except Exception:
            pass
        self.current_active_tab = new_tab_id

#
# End of Tab_Bar.py
#######################################################################################################################
