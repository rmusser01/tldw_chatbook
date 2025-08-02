"""
MediaNavigationPanel - Sidebar navigation for media types.

This component provides:
- List of media types from database
- Collapsible sidebar functionality
- Media type selection
"""

from typing import TYPE_CHECKING, List, Optional
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static, Button
from textual.message import Message
from loguru import logger

from ...Utils.text import slugify

if TYPE_CHECKING:
    from ...app import TldwCli


class MediaTypeSelectedEvent(Message):
    """Event fired when a media type is selected."""
    
    def __init__(self, type_slug: str, display_name: str) -> None:
        super().__init__()
        self.type_slug = type_slug
        self.display_name = display_name


class MediaNavigationPanel(Container):
    """
    Navigation panel for media types with collapse functionality.
    
    This panel displays available media types and allows users to select
    which type to browse. It can be collapsed to save screen space.
    """
    
    DEFAULT_CSS = """
    MediaNavigationPanel {
        width: 20%;
        min-width: 20;
        max-width: 40;
        height: 100%;
        background: $boost;
        layout: vertical;
        border-right: thick $background-darken-1;
    }
    
    MediaNavigationPanel.collapsed {
        width: 3;
        min-width: 3;
    }
    
    MediaNavigationPanel.collapsed .nav-content {
        display: none;
    }
    
    MediaNavigationPanel .nav-content {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    MediaNavigationPanel .nav-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    MediaNavigationPanel .media-type-button {
        width: 100%;
        margin-bottom: 1;
        border: none;
        height: 3;
        text-align: left;
        padding: 0 1;
    }
    
    MediaNavigationPanel .media-type-button:hover {
        background: $accent 75%;
    }
    
    MediaNavigationPanel .media-type-button.active {
        background: $accent;
        text-style: bold;
    }
    """
    
    # Reactive properties
    collapsed: reactive[bool] = reactive(False)
    selected_type: reactive[Optional[str]] = reactive(None)
    
    def __init__(self, app_instance: 'TldwCli', media_types: List[str], **kwargs):
        """
        Initialize the navigation panel.
        
        Args:
            app_instance: Reference to the main app
            media_types: List of media type names from database
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_types = media_types
        
    def compose(self) -> ComposeResult:
        """Compose the navigation panel UI."""
        # Navigation content - hideable
        with VerticalScroll(classes="nav-content"):
            yield Static("Media Types", classes="nav-title")
            
            # Check for error states
            if not self.media_types or (
                len(self.media_types) == 1 and self.media_types[0] in [
                    "Error Loading Types", "DB Error", "Service Error",
                    "DB Error or No Media in DB", "No media types loaded."
                ]
            ):
                error_message = self.media_types[0] if self.media_types else "No media types loaded."
                yield Static(error_message, classes="error-message")
            else:
                # Regular media type buttons
                for media_type in self.media_types:
                    type_slug = slugify(media_type)
                    yield Button(
                        media_type,
                        id=f"media-nav-{type_slug}",
                        classes="media-type-button"
                    )
                
                # Special sections
                yield Button(
                    "Analysis Review",
                    id="media-nav-analysis-review",
                    classes="media-type-button"
                )
                yield Button(
                    "Collections/Tags",
                    id="media-nav-collections-tags",
                    classes="media-type-button"
                )
                yield Button(
                    "Multi-Item Review",
                    id="media-nav-multi-item-review",
                    classes="media-type-button"
                )
    
    def watch_collapsed(self, collapsed: bool) -> None:
        """React to collapse state changes."""
        self.set_class(collapsed, "collapsed")
        
    def watch_selected_type(self, old_type: Optional[str], new_type: Optional[str]) -> None:
        """React to selected type changes."""
        # Update button states
        if old_type:
            try:
                old_button = self.query_one(f"#media-nav-{old_type}")
                old_button.remove_class("active")
            except:
                pass
                
        if new_type:
            try:
                new_button = self.query_one(f"#media-nav-{new_type}")
                new_button.add_class("active")
            except:
                pass
    
    @on(Button.Pressed, ".media-type-button")
    def handle_type_selection(self, event: Button.Pressed) -> None:
        """Handle media type button press."""
        if event.button.id:
            type_slug = event.button.id.replace("media-nav-", "")
            self.selected_type = type_slug
            
            # Get display name from button label
            display_name = str(event.button.label)
            
            # Post event for MediaWindow to handle
            self.post_message(MediaTypeSelectedEvent(type_slug, display_name))