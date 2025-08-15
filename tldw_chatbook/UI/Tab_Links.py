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
                
                # Check if the app is ready for tab switching
                from ..app import TldwCli
                app = self.app
                if isinstance(app, TldwCli):
                    # Only proceed if UI is ready
                    if not getattr(app, '_ui_ready', False):
                        # UI not ready yet, ignore click
                        return
                    
                    # Don't switch if already on this tab
                    if new_tab_id == app.current_tab:
                        return
                
                # Update active state visually
                for link in self.query(".tab-link"):
                    link.remove_class("-active")
                clicked_widget.add_class("-active")
                
                # Create a fake button with the appropriate ID to trigger the app's button handler
                from textual.widgets import Button
                fake_button = Button("", id=f"tab-{new_tab_id}")
                
                # Post a Button.Pressed event to trigger the app's tab switching logic
                button_pressed_event = Button.Pressed(fake_button)
                self.app.post_message(button_pressed_event)
    
    def set_active_tab(self, tab_id: str) -> None:
        """Update the visual active state of tabs. Called by app when tab changes."""
        # Update active state visually
        for link in self.query(".tab-link"):
            link.remove_class("-active")
        
        # Add active class to the new tab
        try:
            active_link = self.query_one(f"#tab-link-{tab_id}")
            active_link.add_class("-active")
        except:
            pass  # Tab might not exist

#
# End of Tab_Links.py
#######################################################################################################################