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
from textual.widgets import Input, Label, Checkbox, Button, Static
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


class MediaSearchPanel(Container):
    """
    Search panel for media items with filters and options.
    
    Provides search input, keyword filtering, and display options
    for browsing media items.
    """
    
    DEFAULT_CSS = """
    MediaSearchPanel {
        dock: top;
        height: auto;
        padding: 1;
        background: $boost;
        border-bottom: thick $background-darken-1;
    }
    
    MediaSearchPanel .search-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    
    MediaSearchPanel .search-input {
        width: 1fr;
        margin-right: 1;
    }
    
    MediaSearchPanel .keyword-input {
        width: 1fr;
        margin-right: 1;
    }
    
    MediaSearchPanel .search-button {
        width: auto;
        min-width: 10;
    }
    
    MediaSearchPanel .filter-row {
        layout: horizontal;
        height: auto;
    }
    
    MediaSearchPanel .filter-label {
        width: auto;
        margin-right: 1;
        padding: 0 1;
    }
    
    MediaSearchPanel .show-deleted-checkbox {
        width: auto;
        margin-right: 2;
    }
    
    MediaSearchPanel .active-filters {
        width: 1fr;
        text-align: right;
        color: $text-muted;
    }
    """
    
    # Reactive properties
    search_term: reactive[str] = reactive("")
    keyword_filter: reactive[str] = reactive("")
    show_deleted: reactive[bool] = reactive(False)
    active_type: reactive[Optional[str]] = reactive(None)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the search panel."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
    def compose(self) -> ComposeResult:
        """Compose the search panel UI."""
        # Search row
        with Horizontal(classes="search-row"):
            yield Input(
                placeholder="Search media items...",
                id="search-input",
                classes="search-input"
            )
            yield Button("Search", id="search-button", classes="search-button", variant="primary")
        
        # Filter row
        with Horizontal(classes="filter-row"):
            yield Label("Keywords:", classes="filter-label")
            yield Input(
                placeholder="Enter keywords separated by commas",
                id="keyword-input",
                classes="keyword-input"
            )
            yield Checkbox(
                "Show deleted items",
                id="show-deleted-checkbox",
                classes="show-deleted-checkbox",
                value=False
            )
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
    
    def _update_active_filters(self) -> None:
        """Update the active filters display."""
        filters = []
        
        if self.active_type:
            filters.append(f"Type: {self.active_type}")
            
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
    
    @on(Button.Pressed, "#search-button")
    def handle_search_button(self) -> None:
        """Handle search button press."""
        self.perform_search()
    
    @on(Input.Submitted)
    def handle_input_submit(self) -> None:
        """Handle Enter key in input fields."""
        self.perform_search()
    
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
    
    def clear_filters(self) -> None:
        """Clear all search filters."""
        self.search_term = ""
        self.keyword_filter = ""
        self.show_deleted = False
        self.active_type = None