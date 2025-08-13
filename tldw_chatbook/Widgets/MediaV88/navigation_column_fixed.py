"""
Navigation Column for Media UI V88 - Fixed version with working dropdowns.

Provides media type selection via buttons and a paginated list of media items.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional, Tuple
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.reactive import reactive
from textual.widgets import ListView, ListItem, Button, Label, Static
from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from ...app import TldwCli

from ...UI.MediaWindowV88 import MediaItemSelectedEventV88, MediaTypeSelectedEventV88


class NavigationColumn(Container):
    """
    Left navigation column containing media type selector and item list.
    
    Features:
    - Button-based media type selection (replacing broken dropdowns)
    - Scrollable list of media items
    - Pagination controls
    - Loading states
    """
    
    DEFAULT_CSS = """
    NavigationColumn {
        layout: vertical;
        height: 100%;
        background: $surface;
    }
    
    .nav-header {
        height: auto;
        padding: 1;
        background: $boost;
        border-bottom: solid $primary-lighten-1;
    }
    
    .nav-header Label {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .type-selector-container {
        layout: vertical;
        padding: 1;
        background: $panel;
        border-bottom: solid $primary-lighten-2;
    }
    
    .type-selector-label {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .type-buttons {
        layout: vertical;
        height: auto;
    }
    
    .type-button {
        width: 100%;
        height: 3;
        margin-bottom: 1;
        text-align: left;
        padding: 0 1;
    }
    
    .type-button.active {
        background: $accent;
        text-style: bold;
    }
    
    .media-list-container {
        height: 1fr;
        min-height: 0;
        layout: vertical;
        overflow: hidden;
    }
    
    #media-items-list {
        height: 1fr;
        min-height: 0;
        overflow-y: auto;
        border: none;
        background: $primary-background;
    }
    
    .media-list-item {
        padding: 1;
        height: auto;
        min-height: 3;
        max-height: 5;
        border-bottom: solid $primary-lighten-3;
        width: 100%;
    }
    
    .media-list-item:hover {
        background: $accent 30%;
    }
    
    .media-list-item.selected {
        background: $accent 50%;
        text-style: bold;
    }
    
    .item-title {
        text-style: bold;
        color: $text;
        text-overflow: ellipsis;
    }
    
    .item-meta {
        color: $text-muted;
        text-style: italic;
    }
    
    .pagination-controls {
        height: auto;
        padding: 1;
        background: $boost;
        border-top: solid $primary-lighten-1;
        layout: horizontal;
        align-horizontal: center;
    }
    
    .pagination-controls Button {
        width: auto;
        min-width: 8;
        height: 3;
        margin: 0 1;
    }
    
    .page-info {
        width: auto;
        padding: 0 1;
        content-align: center middle;
    }
    
    .loading-indicator {
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    
    .no-items {
        text-align: center;
        color: $text-muted;
        padding: 4;
    }
    """
    
    # Reactive properties
    selected_type: reactive[Optional[str]] = reactive("all-media")
    selected_item_id: reactive[Optional[int]] = reactive(None)
    loading: reactive[bool] = reactive(False)
    current_page: reactive[int] = reactive(1)
    total_pages: reactive[int] = reactive(1)
    
    def __init__(
        self,
        app_instance: 'TldwCli',
        media_types: List[str],
        **kwargs
    ):
        """Initialize the navigation column."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_types = media_types
        self.media_items: List[Dict[str, Any]] = []
        self._type_options = self._build_type_options()
    
    def _build_type_options(self) -> List[Tuple[str, str]]:
        """Build options for the media type selector."""
        from ...Utils.text import slugify
        
        options = []
        for media_type in self.media_types:
            slug = slugify(media_type)
            # Special handling for known types
            if media_type.lower() == "all media":
                slug = "all-media"
            options.append((media_type, slug))
        
        return options
    
    def compose(self) -> ComposeResult:
        """Compose the navigation column UI."""
        # Header
        with Container(classes="nav-header"):
            yield Label("Media Library")
        
        # Media type selector using buttons
        with Container(classes="type-selector-container"):
            yield Label("Media Type:", classes="type-selector-label")
            with Container(classes="type-buttons"):
                for name, value in self._type_options:
                    btn = Button(
                        name,
                        id=f"type-btn-{value}",
                        classes="type-button"
                    )
                    if value == "all-media":
                        btn.add_class("active")
                    yield btn
        
        # Media items list
        with Container(classes="media-list-container"):
            yield ListView(id="media-items-list")
        
        # Pagination controls
        with Horizontal(classes="pagination-controls"):
            yield Button("◀ Prev", id="prev-page", disabled=True)
            yield Static("Page 1 / 1", id="page-info", classes="page-info")
            yield Button("Next ▶", id="next-page", disabled=True)
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        logger.info("NavigationColumn (fixed) mounted")
    
    @on(Button.Pressed)
    def handle_button_press(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        # Handle type selection buttons
        if button_id and button_id.startswith("type-btn-"):
            type_value = button_id.replace("type-btn-", "")
            
            # Find display name for the selected value
            display_name = next(
                (name for name, value in self._type_options if value == type_value),
                type_value
            )
            
            logger.info(f"Media type selected: {type_value} ({display_name})")
            
            # Update active button styling
            for name, value in self._type_options:
                btn = self.query_one(f"#type-btn-{value}", Button)
                if value == type_value:
                    btn.add_class("active")
                else:
                    btn.remove_class("active")
            
            self.selected_type = type_value
            
            # Emit event for parent to handle
            self.post_message(MediaTypeSelectedEventV88(type_value, display_name))
        
        # Handle pagination buttons
        elif button_id == "prev-page" and self.current_page > 1:
            self.current_page -= 1
            self.request_page_change(self.current_page)
        elif button_id == "next-page" and self.current_page < self.total_pages:
            self.current_page += 1
            self.request_page_change(self.current_page)
    
    @on(ListView.Selected)
    def handle_item_selection(self, event: ListView.Selected) -> None:
        """Handle media item selection from list."""
        if event.control.id == "media-items-list" and event.item:
            # Extract media ID from the list item
            item_id = event.item.id
            if item_id and item_id.startswith("media-item-"):
                try:
                    media_id = int(item_id.replace("media-item-", ""))
                    self.selected_item_id = media_id
                    
                    # Find the media data
                    media_data = next(
                        (item for item in self.media_items if item.get('id') == media_id),
                        None
                    )
                    
                    if media_data:
                        logger.info(f"Media item selected: {media_id} - {media_data.get('title', 'Unknown')}")
                        # Emit event for parent to handle
                        self.post_message(MediaItemSelectedEventV88(media_id, media_data))
                    else:
                        logger.warning(f"Media data not found for ID: {media_id}")
                        
                except ValueError:
                    logger.error(f"Invalid media item ID: {item_id}")
    
    def set_media_type(self, type_slug: str, display_name: str) -> None:
        """Set the active media type."""
        self.selected_type = type_slug
        
        # Update button styling
        for name, value in self._type_options:
            try:
                btn = self.query_one(f"#type-btn-{value}", Button)
                if value == type_slug:
                    btn.add_class("active")
                else:
                    btn.remove_class("active")
            except Exception as e:
                logger.debug(f"Could not update type button: {e}")
    
    def load_items(
        self,
        items: List[Dict[str, Any]],
        page: int = 1,
        total_pages: int = 1
    ) -> None:
        """Load media items into the list."""
        logger.info(f"Loading {len(items)} items (page {page}/{total_pages})")
        
        self.media_items = items
        self.current_page = page
        self.total_pages = total_pages
        
        # Update list view
        list_view = self.query_one("#media-items-list", ListView)
        
        # Clear the list completely including any loading indicators
        list_view.clear()
        
        if not items:
            # Show no items message
            list_view.append(
                ListItem(
                    Static("No media items found", classes="no-items"),
                    id="no-items"
                )
            )
        else:
            # Add items to list
            for item in items:
                list_item = self._create_list_item(item)
                list_view.append(list_item)
        
        # Update pagination controls
        self._update_pagination()
    
    def _create_list_item(self, media_data: Dict[str, Any]) -> ListItem:
        """Create a list item widget for media data."""
        media_id = media_data.get('id', 0)
        title = media_data.get('title', 'Untitled')
        media_type = media_data.get('type', 'Unknown')
        author = media_data.get('author', '')
        
        # Truncate title for narrow column
        max_title_len = 25
        if len(title) > max_title_len:
            title = title[:max_title_len-3] + "..."
        
        # Build compact metadata
        if author and len(author) > 15:
            author = author[:12] + "..."
        
        meta_text = f"{media_type}"
        if author:
            meta_text = f"{author} • {media_type}"
        
        # Create compact content container
        content = Vertical(
            Static(title, classes="item-title"),
            Static(meta_text, classes="item-meta") if meta_text else Static(""),
        )
        
        list_item = ListItem(
            content,
            id=f"media-item-{media_id}",
            classes="media-list-item"
        )
        
        # Mark as selected if it matches current selection
        if media_id == self.selected_item_id:
            list_item.add_class("selected")
        
        return list_item
    
    def _update_pagination(self) -> None:
        """Update pagination controls based on current state."""
        try:
            # Update page info
            page_info = self.query_one("#page-info", Static)
            page_info.update(f"Page {self.current_page} / {self.total_pages}")
            
            # Update button states
            prev_btn = self.query_one("#prev-page", Button)
            next_btn = self.query_one("#next-page", Button)
            
            prev_btn.disabled = self.current_page <= 1
            next_btn.disabled = self.current_page >= self.total_pages
            
        except Exception as e:
            logger.debug(f"Could not update pagination: {e}")
    
    def set_loading(self, loading: bool) -> None:
        """Set the loading state of the list."""
        self.loading = loading
        
        if loading:
            # Show loading indicator
            list_view = self.query_one("#media-items-list", ListView)
            
            # Check if loading item already exists
            try:
                existing_loading = list_view.query_one("#loading-item")
                # Already has loading indicator, don't add another
                return
            except:
                # No loading indicator, add one
                list_view.clear()
                list_view.append(
                    ListItem(
                        Static("Loading media items...", classes="loading-indicator"),
                        id="loading-item"
                    )
                )
    
    def request_page_change(self, page: int) -> None:
        """Request a page change from the parent."""
        # The parent should handle the actual search/load
        if hasattr(self.parent, 'handle_page_change'):
            self.parent.handle_page_change(page)
    
    def clear_selection(self) -> None:
        """Clear the current item selection."""
        self.selected_item_id = None
        
        # Remove selected class from all items
        list_view = self.query_one("#media-items-list", ListView)
        for item in list_view.children:
            if isinstance(item, ListItem):
                item.remove_class("selected")