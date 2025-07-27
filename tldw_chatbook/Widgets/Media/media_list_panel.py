"""
MediaListPanel - List view for media items with pagination.

This component provides:
- ListView for media items
- Pagination controls
- Item selection handling
- Result count display
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import ListView, ListItem, Button, Label, Static
from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from ...app import TldwCli


class MediaItemSelectedEvent(Message):
    """Event fired when a media item is selected."""
    
    def __init__(self, media_id: int, media_data: Dict[str, Any]) -> None:
        super().__init__()
        self.media_id = media_id
        self.media_data = media_data


class MediaListPanel(Container):
    """
    List panel for displaying media items with pagination.
    
    Shows search results in a scrollable list with pagination controls.
    """
    
    DEFAULT_CSS = """
    MediaListPanel {
        width: 35%;
        height: 100%;
        border-right: solid $primary;
        layout: vertical;
    }
    
    MediaListPanel .list-header {
        dock: top;
        height: 3;
        padding: 0 1;
        background: $boost;
        border-bottom: solid $background-darken-1;
    }
    
    MediaListPanel .list-title {
        text-align: center;
        text-style: bold;
    }
    
    MediaListPanel .media-list {
        height: 1fr;
        border: round $primary-lighten-2;
        background: $primary-background;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    MediaListPanel .media-item {
        padding: 1;
        margin-bottom: 1;
    }
    
    MediaListPanel .media-item.deleted {
        opacity: 0.6;
    }
    
    MediaListPanel .media-item:hover {
        background: $accent 50%;
    }
    
    MediaListPanel .media-item.selected {
        background: $accent;
        text-style: bold;
    }
    
    MediaListPanel .item-title {
        text-style: bold;
        color: $text;
    }
    
    MediaListPanel .item-title.deleted {
        color: $error;
        text-style: strike;
    }
    
    MediaListPanel .item-meta {
        color: $text-muted;
        text-style: italic;
    }
    
    MediaListPanel .item-snippet {
        color: $text-muted;
        max-height: 2;
        overflow-y: hidden;
    }
    
    MediaListPanel .pagination-bar {
        dock: bottom;
        height: 3;
        layout: horizontal;
        align-horizontal: center;
        padding: 0 1;
        background: $boost;
        border-top: solid $background-darken-1;
    }
    
    MediaListPanel .pagination-bar Button {
        width: auto;
        min-width: 10;
        margin: 0 1;
    }
    
    MediaListPanel .page-label {
        width: auto;
        text-align: center;
        padding: 0 1;
    }
    
    MediaListPanel .no-results {
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    """
    
    # Reactive properties
    items: reactive[List[Dict[str, Any]]] = reactive([])
    current_page: reactive[int] = reactive(1)
    total_pages: reactive[int] = reactive(1)
    selected_id: reactive[Optional[int]] = reactive(None)
    loading: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the list panel."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.items_per_page = 20
        
    def compose(self) -> ComposeResult:
        """Compose the list panel UI."""
        # Header
        with Container(classes="list-header"):
            yield Label("Media Items", id="list-title", classes="list-title")
        
        # List view
        yield ListView(id="media-list", classes="media-list")
        
        # Pagination
        with Horizontal(classes="pagination-bar"):
            yield Button("Previous", id="prev-button", disabled=True)
            yield Label("Page 1 / 1", id="page-label", classes="page-label")
            yield Button("Next", id="next-button", disabled=True)
    
    def watch_items(self, items: List[Dict[str, Any]]) -> None:
        """Update the list when items change."""
        self.refresh_list()
    
    def watch_current_page(self, page: int) -> None:
        """Update pagination controls when page changes."""
        self.update_pagination()
    
    def watch_total_pages(self, total: int) -> None:
        """Update pagination controls when total pages change."""
        self.update_pagination()
    
    def watch_selected_id(self, media_id: Optional[int]) -> None:
        """Update list selection when selected ID changes."""
        if media_id is None:
            return
            
        # Update visual selection
        list_view = self.query_one("#media-list", ListView)
        for item in list_view.children:
            if isinstance(item, ListItem) and item.id == f"media-item-{media_id}":
                item.add_class("selected")
            else:
                item.remove_class("selected")
    
    def refresh_list(self) -> None:
        """Refresh the list view with current items."""
        list_view = self.query_one("#media-list", ListView)
        list_view.clear()
        
        if not self.items:
            list_view.append(ListItem(
                Static("No media items found", classes="no-results")
            ))
            return
        
        for item in self.items:
            # Create list item content
            with Vertical() as item_content:
                # Title
                title_classes = "item-title"
                if item.get("is_deleted"):
                    title_classes += " deleted"
                    
                Static(
                    item.get("title", "Untitled"),
                    classes=title_classes
                )
                
                # Metadata
                meta_parts = []
                if item.get("media_type"):
                    meta_parts.append(f"Type: {item['media_type']}")
                if item.get("author"):
                    meta_parts.append(f"Author: {item['author']}")
                    
                if meta_parts:
                    Static(" | ".join(meta_parts), classes="item-meta")
                
                # Snippet
                if item.get("content"):
                    snippet = str(item["content"])[:150] + "..." if len(str(item["content"])) > 150 else str(item["content"])
                    Static(snippet, classes="item-snippet")
            
            # Create list item
            list_item_classes = "media-item"
            if item.get("is_deleted"):
                list_item_classes += " deleted"
                
            list_item = ListItem(
                item_content,
                id=f"media-item-{item['id']}",
                classes=list_item_classes
            )
            
            list_view.append(list_item)
    
    def update_pagination(self) -> None:
        """Update pagination controls."""
        try:
            prev_button = self.query_one("#prev-button", Button)
            next_button = self.query_one("#next-button", Button)
            page_label = self.query_one("#page-label", Label)
            
            prev_button.disabled = self.current_page <= 1
            next_button.disabled = self.current_page >= self.total_pages
            
            page_label.update(f"Page {self.current_page} / {self.total_pages}")
        except:
            pass
    
    @on(ListView.Selected, "#media-list")
    def handle_item_selection(self, event: ListView.Selected) -> None:
        """Handle list item selection."""
        if event.item and event.item.id:
            try:
                media_id = int(event.item.id.replace("media-item-", ""))
                
                # Find the media data
                media_data = None
                for item in self.items:
                    if item["id"] == media_id:
                        media_data = item
                        break
                
                if media_data:
                    self.selected_id = media_id
                    self.post_message(MediaItemSelectedEvent(media_id, media_data))
            except Exception as e:
                logger.error(f"Error handling item selection: {e}")
    
    @on(Button.Pressed, "#prev-button")
    def handle_prev_page(self) -> None:
        """Handle previous page button."""
        if self.current_page > 1:
            self.current_page -= 1
            self.post_message(MediaSearchEvent("", "", False))  # Trigger search with new page
    
    @on(Button.Pressed, "#next-button")
    def handle_next_page(self) -> None:
        """Handle next page button."""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.post_message(MediaSearchEvent("", "", False))  # Trigger search with new page
    
    def set_loading(self, loading: bool) -> None:
        """Set loading state."""
        self.loading = loading
        
        try:
            list_title = self.query_one("#list-title", Label)
            if loading:
                list_title.update("Loading...")
            else:
                item_count = len(self.items)
                list_title.update(f"Media Items ({item_count})")
        except:
            pass
    
    def load_items(self, items: List[Dict[str, Any]], page: int, total_pages: int) -> None:
        """Load new items into the list."""
        self.items = items
        self.current_page = page
        self.total_pages = total_pages
        self.set_loading(False)


# Import at end to avoid circular dependency
from .media_search_panel import MediaSearchEvent