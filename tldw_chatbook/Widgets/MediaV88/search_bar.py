"""
Search Bar component for Media UI V88.

Provides collapsible search interface with quick search and advanced filtering options.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, Checkbox, Collapsible, Static
from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from ...app import TldwCli

from ...UI.MediaWindowV88 import MediaSearchEventV88


class SearchBar(Container):
    """
    Collapsible search bar with quick search and advanced filtering.
    
    Features:
    - Collapsible design to save space
    - Quick search input with debouncing
    - Keyword tags input
    - Advanced filters (expandable)
    - Sort options
    - Show deleted toggle
    """
    
    DEFAULT_CSS = """
    SearchBar {
        height: auto;
        layout: vertical;
        background: $boost;
        border-bottom: solid $primary-lighten-2;
        padding: 1;
    }
    
    SearchBar.collapsed {
        height: 3;
        padding: 0;
    }
    
    SearchBar.collapsed .search-content {
        display: none;
    }
    
    .search-header {
        layout: horizontal;
        height: 3;
        align-vertical: middle;
    }
    
    #search-toggle {
        width: auto;
        min-width: 5;
        height: 3;
        margin-right: 1;
        background: $primary;
    }
    
    .search-title {
        width: 1fr;
        text-style: bold;
        padding: 0 1;
        content-align: left middle;
    }
    
    .search-content {
        layout: vertical;
        margin-top: 1;
        height: auto;
    }
    
    .search-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
        align-vertical: middle;
    }
    
    .search-label {
        width: 12;
        text-align: right;
        padding-right: 1;
        color: $text-muted;
    }
    
    #search-input {
        width: 1fr;
        height: 3;
    }
    
    #keywords-input {
        width: 1fr;
        height: 3;
    }
    
    .search-buttons {
        layout: horizontal;
        height: 3;
        margin-left: 12;
    }
    
    .search-buttons Button {
        width: auto;
        min-width: 10;
        margin-right: 1;
    }
    
    #search-button {
        background: $success;
    }
    
    #clear-button {
        background: $warning;
    }
    
    .advanced-filters {
        margin-top: 1;
        padding: 1;
        background: $surface;
        border: solid $primary-lighten-3;
    }
    
    .filter-row {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
        align-vertical: middle;
    }
    
    .filter-label {
        width: 12;
        text-align: right;
        padding-right: 1;
        color: $text-muted;
    }
    
    .sort-options {
        layout: horizontal;
        height: auto;
    }
    
    .sort-option {
        margin-right: 2;
    }
    
    #show-deleted {
        margin-left: 12;
    }
    
    .current-filters {
        margin-top: 1;
        padding: 1;
        background: $primary-background;
        border: round $primary-lighten-3;
    }
    
    .filter-tags {
        layout: horizontal;
        height: auto;
    }
    
    .filter-tag {
        padding: 0 1;
        margin: 0 1 1 0;
        background: $accent;
        border: solid $accent-lighten-1;
        height: 3;
    }
    """
    
    # Reactive properties
    collapsed: reactive[bool] = reactive(True)
    search_term: reactive[str] = reactive("")
    keyword_filter: reactive[str] = reactive("")
    show_deleted: reactive[bool] = reactive(False)
    show_advanced: reactive[bool] = reactive(False)
    active_type_filter: reactive[Optional[str]] = reactive(None)
    
    # Debounce timer
    _search_timer: Optional[object] = None
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the search bar."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.active_filters: Dict[str, Any] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the search bar UI."""
        # Header with toggle button
        with Horizontal(classes="search-header"):
            yield Button("▼ Search", id="search-toggle", variant="primary")
            yield Static("Search & Filter Media", classes="search-title")
        
        # Main search content (hidden when collapsed)
        with Container(classes="search-content"):
            # Quick search row
            with Horizontal(classes="search-row"):
                yield Label("Search:", classes="search-label")
                yield Input(
                    placeholder="Enter search terms...",
                    id="search-input"
                )
            
            # Keywords row
            with Horizontal(classes="search-row"):
                yield Label("Keywords:", classes="search-label")
                yield Input(
                    placeholder="Enter keywords separated by commas...",
                    id="keywords-input"
                )
            
            # Action buttons
            with Horizontal(classes="search-buttons"):
                yield Button("Search", id="search-button", variant="success")
                yield Button("Clear", id="clear-button", variant="warning")
                yield Button("Advanced ▼", id="advanced-toggle")
            
            # Advanced filters (collapsible)
            with Collapsible(
                title="Advanced Filters",
                collapsed=True,
                id="advanced-filters-collapsible",
                classes="advanced-filters"
            ):
                # Show deleted checkbox
                yield Checkbox(
                    "Show deleted items",
                    id="show-deleted",
                    value=False
                )
            
            # Current filters display
            with Container(id="current-filters-container", classes="current-filters"):
                yield Label("Active Filters:", classes="filter-label")
                yield Container(id="filter-tags", classes="filter-tags")
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        logger.info("SearchBar mounted")
        
        # Hide current filters initially
        self._update_filter_display()
    
    def watch_collapsed(self, collapsed: bool) -> None:
        """React to collapse state changes."""
        if collapsed:
            self.add_class("collapsed")
            # Update toggle button
            try:
                toggle = self.query_one("#search-toggle", Button)
                toggle.label = "▶ Search"
            except Exception:
                pass
        else:
            self.remove_class("collapsed")
            # Update toggle button
            try:
                toggle = self.query_one("#search-toggle", Button)
                toggle.label = "▼ Search"
            except Exception:
                pass
    
    @on(Button.Pressed, "#search-toggle")
    def handle_toggle(self, event: Button.Pressed) -> None:
        """Toggle the search bar collapse state."""
        self.collapsed = not self.collapsed
    
    @on(Button.Pressed, "#search-button")
    def handle_search(self, event: Button.Pressed) -> None:
        """Execute search with current parameters."""
        self._execute_search()
    
    @on(Button.Pressed, "#clear-button")
    def handle_clear(self, event: Button.Pressed) -> None:
        """Clear all search parameters."""
        # Clear inputs
        try:
            search_input = self.query_one("#search-input", Input)
            search_input.value = ""
            
            keywords_input = self.query_one("#keywords-input", Input)
            keywords_input.value = ""
            
            show_deleted = self.query_one("#show-deleted", Checkbox)
            show_deleted.value = False
        except Exception as e:
            logger.debug(f"Error clearing inputs: {e}")
        
        # Clear state
        self.search_term = ""
        self.keyword_filter = ""
        self.show_deleted = False
        self.active_filters = {}
        
        # Update display
        self._update_filter_display()
        
        # Execute empty search
        self._execute_search()
    
    @on(Button.Pressed, "#advanced-toggle")
    def handle_advanced_toggle(self, event: Button.Pressed) -> None:
        """Toggle advanced filters visibility."""
        try:
            collapsible = self.query_one("#advanced-filters-collapsible", Collapsible)
            collapsible.collapsed = not collapsible.collapsed
            
            # Update button text
            button = self.query_one("#advanced-toggle", Button)
            if collapsible.collapsed:
                button.label = "Advanced ▼"
            else:
                button.label = "Advanced ▲"
        except Exception as e:
            logger.debug(f"Error toggling advanced filters: {e}")
    
    @on(Input.Changed, "#search-input")
    def handle_search_input(self, event: Input.Changed) -> None:
        """Handle search input changes with debouncing."""
        self.search_term = event.value
        
        # Cancel previous timer if exists
        if self._search_timer:
            self.set_timer(0, self._search_timer.stop)
        
        # Set new timer for debounced search (300ms)
        self._search_timer = self.set_timer(0.3, self._execute_search)
    
    @on(Input.Submitted, "#search-input")
    def handle_search_submit(self, event: Input.Submitted) -> None:
        """Handle Enter key in search input."""
        # Cancel debounce timer and search immediately
        if self._search_timer:
            self.set_timer(0, self._search_timer.stop)
        self._execute_search()
    
    @on(Input.Changed, "#keywords-input")
    def handle_keywords_input(self, event: Input.Changed) -> None:
        """Handle keywords input changes."""
        self.keyword_filter = event.value
    
    @on(Input.Submitted, "#keywords-input")
    def handle_keywords_submit(self, event: Input.Submitted) -> None:
        """Handle Enter key in keywords input."""
        self._execute_search()
    
    @on(Checkbox.Changed, "#show-deleted")
    def handle_show_deleted(self, event: Checkbox.Changed) -> None:
        """Handle show deleted checkbox change."""
        self.show_deleted = event.value
        self._execute_search()
    
    def _execute_search(self) -> None:
        """Execute search with current parameters."""
        # Parse keywords
        keywords = []
        if self.keyword_filter:
            keywords = [k.strip() for k in self.keyword_filter.split(',') if k.strip()]
        
        # Build filters
        filters = {
            'show_deleted': self.show_deleted,
            'type_filter': self.active_type_filter,
        }
        
        # Update active filters
        self.active_filters = {
            'search': self.search_term if self.search_term else None,
            'keywords': keywords if keywords else None,
            'show_deleted': self.show_deleted if self.show_deleted else None,
            'type': self.active_type_filter if self.active_type_filter else None,
        }
        
        # Remove None values
        self.active_filters = {k: v for k, v in self.active_filters.items() if v is not None}
        
        # Update filter display
        self._update_filter_display()
        
        # Emit search event
        logger.info(f"Executing search: term='{self.search_term}', keywords={keywords}")
        self.post_message(MediaSearchEventV88(self.search_term, keywords, filters))
    
    def _update_filter_display(self) -> None:
        """Update the active filters display."""
        try:
            container = self.query_one("#current-filters-container", Container)
            tags_container = self.query_one("#filter-tags", Container)
            
            # Clear existing tags
            tags_container.remove_children()
            
            if not self.active_filters:
                # Hide container if no filters
                container.styles.display = "none"
            else:
                # Show container
                container.styles.display = "block"
                
                # Add filter tags
                for key, value in self.active_filters.items():
                    if key == 'search' and value:
                        tag = Static(f"Search: {value[:20]}...", classes="filter-tag")
                        tags_container.mount(tag)
                    elif key == 'keywords' and value:
                        for kw in value:
                            tag = Static(f"Keyword: {kw}", classes="filter-tag")
                            tags_container.mount(tag)
                    elif key == 'show_deleted' and value:
                        tag = Static("Show Deleted", classes="filter-tag")
                        tags_container.mount(tag)
                    elif key == 'type' and value:
                        tag = Static(f"Type: {value}", classes="filter-tag")
                        tags_container.mount(tag)
                        
        except Exception as e:
            logger.debug(f"Error updating filter display: {e}")
    
    def set_type_filter(self, type_slug: str, display_name: str) -> None:
        """Set the active media type filter."""
        self.active_type_filter = type_slug if type_slug != "all-media" else None
        
        # Update filter display
        if self.active_type_filter:
            self.active_filters['type'] = display_name
        else:
            self.active_filters.pop('type', None)
        
        self._update_filter_display()
    
    def clear_filters(self) -> None:
        """Clear all filters and search terms."""
        self.handle_clear(None)