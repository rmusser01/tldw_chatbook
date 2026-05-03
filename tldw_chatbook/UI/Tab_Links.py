# Tab_Links.py
# Description: Single-line clickable tab links navigation with visual grouping
#
# Imports
from typing import TYPE_CHECKING, List
#
# Third-Party Imports
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.widgets import Static
#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli

from ..Constants import (
    TAB_CCP, TAB_GROUPS, TAB_MCP, TAB_TOOLS_SETTINGS, get_tab_display_label,
)
from ..UI.Navigation.main_navigation import NavigateToScreen
#
#######################################################################################################################
#
# Functions:


def _get_tab_label(tab_id: str) -> str:
    """Return the display label for a tab ID."""
    return get_tab_display_label(tab_id)


class TabLinks(ScrollableContainer):
    """
    A single-line navigation with clickable tab titles, visually grouped.
    """

    def __init__(self, tab_ids: List[str], initial_active_tab: str, **kwargs):
        super().__init__(**kwargs)
        self.tab_ids = tab_ids
        self.initial_active_tab = initial_active_tab
        self.id = "tab-links-container"
        self.can_focus = False

    def compose(self) -> ComposeResult:
        """Create clickable tab links organized by group with | dividers between groups."""
        with Horizontal(id="tab-links-inner"):
            group_index = 0
            for group_name, group_tab_ids in TAB_GROUPS.items():
                # Only include tabs that exist in the provided tab_ids list
                visible_tabs = [t for t in group_tab_ids if t in self.tab_ids]
                if not visible_tabs:
                    continue

                # Add group divider between groups (not before the first)
                if group_index > 0:
                    yield Static("  ┃  ", classes="tab-group-separator")

                for i, tab_id in enumerate(visible_tabs):
                    label_text = _get_tab_label(tab_id)

                    classes = "tab-link"
                    if tab_id == self.initial_active_tab:
                        classes += " -active"

                    yield Static(
                        label_text,
                        id=f"tab-link-{tab_id}",
                        classes=classes,
                    )

                    # Add thin separator between tabs within the same group
                    if i < len(visible_tabs) - 1:
                        yield Static(" · ", classes="tab-separator")

                group_index += 1

            # Render any tabs not in any group (safety net)
            grouped_tabs = {t for tabs in TAB_GROUPS.values() for t in tabs}
            ungrouped = [t for t in self.tab_ids if t not in grouped_tabs]
            if ungrouped:
                yield Static("  ┃  ", classes="tab-group-separator")
                for i, tab_id in enumerate(ungrouped):
                    label_text = _get_tab_label(tab_id)
                    classes = "tab-link"
                    if tab_id == self.initial_active_tab:
                        classes += " -active"
                    yield Static(
                        label_text,
                        id=f"tab-link-{tab_id}",
                        classes=classes,
                    )
                    if i < len(ungrouped) - 1:
                        yield Static(" · ", classes="tab-separator")

    async def on_click(self, event) -> None:
        """Handle clicks on the container to detect which link was clicked."""
        clicked_widget = self.app.get_widget_at(event.screen_x, event.screen_y)[0]

        if clicked_widget and hasattr(clicked_widget, 'id') and clicked_widget.id:
            widget_id = clicked_widget.id
            if widget_id.startswith("tab-link-"):
                new_tab_id = widget_id.replace("tab-link-", "")
                self._update_active_link(new_tab_id)
                screen_name = 'ccp' if new_tab_id == TAB_CCP else new_tab_id
                self.post_message(NavigateToScreen(screen_name=screen_name))

    def set_active_tab(self, tab_id: str) -> None:
        """Public API: update the visual active state to the given tab."""
        self._update_active_link(tab_id)

    def _update_active_link(self, new_tab_id: str) -> None:
        """Update the visual state of tab links."""
        if new_tab_id == TAB_MCP and TAB_TOOLS_SETTINGS in self.tab_ids:
            new_tab_id = TAB_TOOLS_SETTINGS
        for link in self.query(".tab-link"):
            link.remove_class("-active")
        try:
            active_link = self.query_one(f"#tab-link-{new_tab_id}")
            active_link.add_class("-active")
        except Exception:
            pass

#
# End of Tab_Links.py
#######################################################################################################################
