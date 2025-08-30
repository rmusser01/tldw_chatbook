# Tab_Links.py
# Description: Single-line clickable tab links navigation
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

from ..Constants import TAB_CCP, TAB_TOOLS_SETTINGS, TAB_INGEST, TAB_LLM, TAB_EVALS, TAB_CODING, TAB_STTS, TAB_STUDY, TAB_CHATBOOKS, TAB_CUSTOMIZE
from ..UI.Navigation.main_navigation import NavigateToScreen
#
#######################################################################################################################
#
# Functions:

class TabLinks(ScrollableContainer):
    """
    A single-line navigation with clickable tab titles.
    """

    def __init__(self, tab_ids: List[str], initial_active_tab: str, **kwargs):
        super().__init__(**kwargs)
        self.tab_ids = tab_ids
        self.initial_active_tab = initial_active_tab
        self.id = "tab-links-container"
        self.can_focus = False
        
    def compose(self) -> ComposeResult:
        """Create clickable tab links with separators in a horizontal container."""
        with Horizontal(id="tab-links-inner"):
            for i, tab_id in enumerate(self.tab_ids):
                # Determine label based on tab_id
                if tab_id == TAB_CCP:
                    label_text = "CCP"
                elif tab_id == TAB_TOOLS_SETTINGS:
                    label_text = "Settings"
                elif tab_id == TAB_INGEST:
                    label_text = "Ingest"
                elif tab_id == TAB_LLM:
                    label_text = "LLM"
                elif tab_id == TAB_EVALS:
                    label_text = "Evals"
                elif tab_id == TAB_CODING:
                    label_text = "Coding"
                elif tab_id == TAB_STTS:
                    label_text = "S/TT/S"
                elif tab_id == TAB_STUDY:
                    label_text = "Study"
                elif tab_id == TAB_CHATBOOKS:
                    label_text = "Chatbooks"
                elif tab_id == TAB_CUSTOMIZE:
                    label_text = "Customize"
                else:
                    # Default: capitalize first letter of each word
                    label_text = tab_id.replace('_', ' ').title()
                
                # Create the clickable link
                classes = "tab-link"
                if tab_id == self.initial_active_tab:
                    classes += " -active"
                
                yield Static(
                    label_text,
                    id=f"tab-link-{tab_id}",
                    classes=classes
                )
                
                # Add separator except for the last item
                if i < len(self.tab_ids) - 1:
                    yield Static(" | ", classes="tab-separator")
    
    async def on_click(self, event) -> None:
        """Handle clicks on the container to detect which link was clicked."""
        # Get the widget that was clicked
        clicked_widget = self.app.get_widget_at(event.screen_x, event.screen_y)[0]
        
        # Check if it's a tab link
        if clicked_widget and hasattr(clicked_widget, 'id') and clicked_widget.id:
            widget_id = clicked_widget.id
            if widget_id.startswith("tab-link-"):
                new_tab_id = widget_id.replace("tab-link-", "")
                
                # Update visual state
                self._update_active_link(new_tab_id)
                
                # Map special tab IDs to screen names
                screen_name = 'ccp' if new_tab_id == TAB_CCP else new_tab_id
                
                # Post navigation message to app for screen-based navigation
                self.post_message(NavigateToScreen(screen_name=screen_name))
    
    def _update_active_link(self, new_tab_id: str) -> None:
        """Update the visual state of tab links."""
        # Remove active class from all links
        for link in self.query(".tab-link"):
            link.remove_class("-active")
        
        # Add active class to the new link
        try:
            active_link = self.query_one(f"#tab-link-{new_tab_id}")
            active_link.add_class("-active")
        except:
            pass  # Tab might not exist

#
# End of Tab_Links.py
#######################################################################################################################