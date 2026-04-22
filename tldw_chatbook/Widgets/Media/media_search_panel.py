"""
MediaSearchPanel - Search and filter controls for media items.

This component provides:
- Search input field
- Keyword filter
- Show deleted items checkbox
- Active filter display
"""

from typing import TYPE_CHECKING, Optional
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import Input, Label, Checkbox, Button, Static, Collapsible, Select
from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from ...app import TldwCli


class MediaSearchEvent(Message):
    """Event fired when search criteria change."""
    
    def __init__(self, search_term: str, keyword_filter: str, show_deleted: bool) -> None:
        super().__init__()
        self.search_term = search_term
        self.keyword_filter = keyword_filter
        self.show_deleted = show_deleted


class MediaBrowseSubviewChangedEvent(Message):
    """Event fired when the browse subview changes."""

    def __init__(self, subview: str) -> None:
        super().__init__()
        self.subview = subview


class MediaSearchPanel(Container):
    """
    Search panel for media items with filters and options.
    
    Provides search input, keyword filtering, and display options
    for browsing media items.
    """
    
    DEFAULT_CSS = """
    MediaSearchPanel {
        height: auto;
        padding: 1;
        background: $boost;
        border-bottom: thick $background-darken-1;
        width: 100%;
        layout: vertical;
    }
    
    MediaSearchPanel .header-row {
        layout: horizontal;
        height: auto;
        width: 100%;
        align-vertical: top;
    }
    
    MediaSearchPanel .media-sidebar-toggle {
        height: 3;
        width: auto;
        min-width: 10;
        margin-right: 1;
        background: $boost;
        border: solid $primary;
    }
    
    MediaSearchPanel .search-collapsible {
        width: 1fr;
    }
    
    MediaSearchPanel .search-row {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    MediaSearchPanel .search-input {
        width: 1fr;
        height: 3;
        margin-right: 1;
    }
    
    MediaSearchPanel .keyword-input {
        height: 3;
        margin: 0 1;
    }
    
    MediaSearchPanel .search-button {
        width: auto;
        min-width: 10;
        height: 3;
    }
    
    MediaSearchPanel .filter-row {
        layout: grid;
        grid-size: 3 1;
        grid-columns: auto 2fr 1fr;
        height: 3;
        align-vertical: middle;
        grid-gutter: 0;
    }
    
    MediaSearchPanel .filter-label {
        width: auto;
        margin-right: 1;
        padding: 0 1;
    }
    
    MediaSearchPanel .checkbox-container {
        height: 3;
        width: 100%;
        layout: vertical;
    }
    
    MediaSearchPanel .show-deleted-checkbox {
        height: 3;
        width: 100%;
    }

    MediaSearchPanel .saved-view-row {
        layout: horizontal;
        height: 3;
        align-vertical: middle;
        margin-bottom: 1;
    }

    MediaSearchPanel .saved-view-select {
        width: 24;
        margin-left: 1;
    }
    
    MediaSearchPanel Collapsible {
        margin-top: 0;
    }
    
    MediaSearchPanel Collapsible Collapsible {
        margin-top: 1;
    }
    
    MediaSearchPanel .active-filters {
        width: 100%;
        height: 1;
        text-align: right;
        color: $text-muted;
        padding: 0 1;
        margin: 0;
    }
    """
    
    # Reactive properties
    search_term: reactive[str] = reactive("")
    keyword_filter: reactive[str] = reactive("")
    show_deleted: reactive[bool] = reactive(False)
    active_type: reactive[Optional[str]] = reactive(None)
    browse_subview: reactive[str] = reactive("all")
    saved_view_enabled: reactive[bool] = reactive(True)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the search panel."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._synchronizing_browse_subview = False
        
    def compose(self) -> ComposeResult:
        """Compose the search panel UI."""
        # Import here to avoid circular imports
        from ...Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
        
        # Create a horizontal container for the sidebar toggle and collapsible header
        with Horizontal(classes="header-row"):
            # Sidebar toggle button
            yield Button(
                get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                id="media-sidebar-toggle",
                classes="media-sidebar-toggle"
            )
            
            # Wrap all search functionality in a collapsible
            with Collapsible(title="Search & Filters", collapsed=False, classes="search-collapsible"):
                # Main search row
                with Horizontal(classes="search-row"):
                    yield Input(
                        placeholder="Search media items...",
                        id="search-input",
                        classes="search-input"
                    )
                    yield Button("Search", id="search-button", classes="search-button", variant="primary")
                
                # Additional options in nested collapsible
                with Collapsible(title="Additional Options", collapsed=True):
                    # Filter row with grid layout for better control
                    with Container(classes="filter-row"):
                        yield Label("Keywords:", classes="filter-label")
                        yield Input(
                            placeholder="Enter keywords separated by commas",
                            id="keyword-input",
                            classes="keyword-input"
                        )
                        with Container(classes="checkbox-container"):
                            yield Checkbox(
                                "Show deleted",
                                id="show-deleted-checkbox",
                                classes="show-deleted-checkbox",
                                value=False
                            )

                    with Horizontal(classes="saved-view-row"):
                        yield Label("Browse:", classes="filter-label")
                        yield Select(
                            [
                                ("All media", "all"),
                                ("Read-it-later", "read-it-later"),
                            ],
                            id="browse-subview-select",
                            classes="saved-view-select",
                            value="all",
                        )
                    
                    # Active filters display on separate line
                    yield Static("", id="active-filters", classes="active-filters")
    
    def watch_search_term(self, search_term: str) -> None:
        """Update search input when search term changes."""
        try:
            search_input = self.query_one("#search-input", Input)
            if search_input.value != search_term:
                search_input.value = search_term
        except:
            pass
    
    def watch_keyword_filter(self, keyword_filter: str) -> None:
        """Update keyword input when filter changes."""
        try:
            keyword_input = self.query_one("#keyword-input", Input)
            if keyword_input.value != keyword_filter:
                keyword_input.value = keyword_filter
        except:
            pass
    
    def watch_show_deleted(self, show_deleted: bool) -> None:
        """Update checkbox when show deleted changes."""
        try:
            checkbox = self.query_one("#show-deleted-checkbox", Checkbox)
            if checkbox.value != show_deleted:
                checkbox.value = show_deleted
        except:
            pass
        self._update_active_filters()
    
    def watch_active_type(self, active_type: Optional[str]) -> None:
        """Update active filters display when type changes."""
        self._update_active_filters()

    def watch_browse_subview(self, browse_subview: str) -> None:
        """Update browse subview select when state changes."""
        try:
            browse_select = self.query_one("#browse-subview-select", Select)
            if browse_select.value != browse_subview:
                self._synchronizing_browse_subview = True
                try:
                    browse_select.value = browse_subview
                finally:
                    self._synchronizing_browse_subview = False
        except Exception:
            pass
        self._update_active_filters()

    def watch_saved_view_enabled(self, saved_view_enabled: bool) -> None:
        """Enable or disable the saved-view selector for current context."""
        try:
            browse_select = self.query_one("#browse-subview-select", Select)
            browse_select.disabled = not saved_view_enabled
        except Exception:
            pass
    
    def _update_active_filters(self) -> None:
        """Update the active filters display."""
        filters = []
        
        if self.active_type:
            filters.append(f"Type: {self.active_type}")

        if self.browse_subview == "read-it-later":
            filters.append("Browse: Read-it-later")
            
        if self.search_term:
            filters.append(f"Search: '{self.search_term}'")
            
        if self.keyword_filter:
            filters.append(f"Keywords: {self.keyword_filter}")
            
        if self.show_deleted:
            filters.append("Showing deleted")
        
        try:
            active_filters = self.query_one("#active-filters", Static)
            active_filters.update(" | ".join(filters) if filters else "No active filters")
        except:
            pass
    
    @on(Input.Changed, "#search-input")
    def handle_search_input(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        self.search_term = event.value
    
    @on(Input.Changed, "#keyword-input") 
    def handle_keyword_input(self, event: Input.Changed) -> None:
        """Handle keyword input changes."""
        self.keyword_filter = event.value
    
    @on(Checkbox.Changed, "#show-deleted-checkbox")
    def handle_show_deleted(self, event: Checkbox.Changed) -> None:
        """Handle show deleted checkbox changes."""
        self.show_deleted = event.value

    @on(Select.Changed, "#browse-subview-select")
    def handle_browse_subview_changed(self, event: Select.Changed) -> None:
        """Handle browse subview changes."""
        new_subview = str(event.value or "all")
        if self._synchronizing_browse_subview or new_subview == self.browse_subview:
            return
        self.browse_subview = new_subview
        self.post_message(MediaBrowseSubviewChangedEvent(self.browse_subview))
    
    @on(Button.Pressed, "#search-button")
    def handle_search_button(self) -> None:
        """Handle search button press."""
        self.perform_search()
    
    @on(Input.Submitted)
    def handle_input_submit(self) -> None:
        """Handle Enter key in input fields."""
        self.perform_search()
    
    @on(Button.Pressed, "#media-sidebar-toggle")
    def handle_sidebar_toggle(self) -> None:
        """Handle sidebar toggle button press."""
        # Post event to toggle sidebar
        from ...Event_Handlers.media_events import SidebarCollapseEvent
        self.post_message(SidebarCollapseEvent())
    
    def perform_search(self) -> None:
        """Trigger a search with current criteria."""
        self.post_message(MediaSearchEvent(
            self.search_term,
            self.keyword_filter,
            self.show_deleted
        ))
    
    def set_type_filter(self, type_slug: str, display_name: str) -> None:
        """Set the active media type filter."""
        self.active_type = display_name
        self._update_active_filters()

    def set_browse_subview(self, subview: str) -> None:
        """Set the active browse subview."""
        self.browse_subview = str(subview or "all")

    def set_saved_view_enabled(self, enabled: bool) -> None:
        """Toggle saved-view browsing for the current context."""
        self.saved_view_enabled = bool(enabled)
    
    def clear_filters(self) -> None:
        """Clear all search filters."""
        self.search_term = ""
        self.keyword_filter = ""
        self.show_deleted = False
        self.active_type = None
