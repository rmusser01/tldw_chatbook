# lazy_widgets.py
# Performance-optimized widgets that defer content creation until needed

from typing import Callable, Optional, List, Any
from textual.app import ComposeResult
from textual.widgets import Collapsible, Static, ListView, ListItem
from textual.containers import Container
from textual import work
from loguru import logger
import asyncio

class LazyCollapsible(Collapsible):
    """A Collapsible that defers content creation until first expanded.
    
    This significantly improves startup performance by not creating
    widgets that are initially hidden.
    """
    
    def __init__(
        self, 
        title: str,
        *,
        content_factory: Optional[Callable[[], ComposeResult]] = None,
        collapsed: bool = True,
        **kwargs
    ):
        """Initialize LazyCollapsible.
        
        Args:
            title: The title of the collapsible
            content_factory: A callable that yields widgets when called
            collapsed: Whether to start collapsed (default True)
            **kwargs: Additional arguments for Collapsible
        """
        super().__init__(title=title, collapsed=collapsed, **kwargs)
        self._content_factory = content_factory
        self._content_loaded = False
        self._loading_message = None
        
    def compose(self) -> ComposeResult:
        """Compose with placeholder if collapsed."""
        if self.collapsed and self._content_factory:
            # Show a lightweight placeholder
            self._loading_message = Static("Content will load when expanded...", 
                                          classes="lazy-placeholder")
            yield self._loading_message
        elif self._content_factory and not self._content_loaded:
            # If starting expanded, load content immediately
            yield from self._load_content()
        
    def watch_collapsed(self, collapsed: bool) -> None:
        """Handle expansion/collapse events."""
        super().watch_collapsed(collapsed)
        
        if not collapsed and not self._content_loaded and self._content_factory:
            # First time expanding - load content
            self.call_after_refresh(self._async_load_content)
    
    def _load_content(self) -> ComposeResult:
        """Load the actual content."""
        if self._content_factory:
            logger.debug(f"LazyCollapsible '{self.title}' loading content")
            try:
                # Remove placeholder if it exists
                if self._loading_message:
                    self._loading_message.remove()
                    self._loading_message = None
                
                # Generate and yield the actual content
                yield from self._content_factory()
                self._content_loaded = True
                logger.debug(f"LazyCollapsible '{self.title}' content loaded")
            except Exception as e:
                logger.error(f"Error loading content for LazyCollapsible '{self.title}': {e}")
                yield Static(f"Error loading content: {e}", classes="error-message")
    
    async def _async_load_content(self) -> None:
        """Asynchronously load content after expansion."""
        if self._content_loaded or not self._content_factory:
            return
            
        try:
            # Remove placeholder
            if self._loading_message:
                await self._loading_message.remove()
                self._loading_message = None
            
            # Mount the actual content
            widgets = list(self._content_factory())
            if widgets:
                await self.mount(*widgets)
                self._content_loaded = True
                logger.debug(f"LazyCollapsible '{self.title}' async content loaded")
        except Exception as e:
            logger.error(f"Error async loading content for LazyCollapsible '{self.title}': {e}")
            await self.mount(Static(f"Error loading content: {e}", classes="error-message"))


class VirtualListView(ListView):
    """A ListView that only renders visible items for better performance.
    
    This is especially useful for long lists where creating all items
    upfront would be expensive.
    """
    
    def __init__(
        self,
        *children: ListItem,
        initial_items: Optional[List[Any]] = None,
        item_factory: Optional[Callable[[Any], ListItem]] = None,
        virtual_size: int = 50,  # Number of items to keep rendered
        **kwargs
    ):
        """Initialize VirtualListView.
        
        Args:
            *children: Initial children (if any)
            initial_items: Data items to display
            item_factory: Function to create ListItem from data
            virtual_size: Number of items to keep in DOM
            **kwargs: Additional ListView arguments
        """
        super().__init__(*children, **kwargs)
        self._items = initial_items or []
        self._item_factory = item_factory or self._default_item_factory
        self._virtual_size = virtual_size
        self._rendered_range = (0, min(virtual_size, len(self._items)))
        self._pending_update = False
        
    def _default_item_factory(self, item: Any) -> ListItem:
        """Default factory for creating list items."""
        return ListItem(Static(str(item)))
    
    def compose(self) -> ComposeResult:
        """Compose only the initially visible items."""
        if self._items and self._item_factory:
            # Only render the first batch of items
            end_idx = min(self._virtual_size, len(self._items))
            for item in self._items[:end_idx]:
                yield self._item_factory(item)
            logger.debug(f"VirtualListView rendered {end_idx} of {len(self._items)} items")
    
    def set_items(self, items: List[Any]) -> None:
        """Update the list with new items."""
        self._items = items
        self._rendered_range = (0, min(self._virtual_size, len(items)))
        self._update_visible_items()
    
    def add_item(self, item: Any) -> None:
        """Add a single item to the list."""
        self._items.append(item)
        # Only render if within visible range
        if len(self._items) <= self._rendered_range[1]:
            self.mount(self._item_factory(item))
    
    @work(exclusive=True)
    async def _update_visible_items(self) -> None:
        """Update which items are rendered based on scroll position."""
        if self._pending_update:
            return
        
        self._pending_update = True
        try:
            # Clear existing items
            await self.clear()
            
            # Render visible range
            start, end = self._rendered_range
            for item in self._items[start:end]:
                await self.mount(self._item_factory(item))
                
            logger.debug(f"VirtualListView updated range {start}-{end}")
        finally:
            self._pending_update = False
    
    def on_scroll(self, event) -> None:
        """Handle scroll events to update visible items."""
        super().on_scroll(event)
        
        # Calculate which items should be visible
        # This is simplified - a real implementation would calculate based on
        # actual scroll position and item heights
        visible_start = int(self.scroll_y / 2)  # Assuming ~2 lines per item
        visible_start = max(0, visible_start)
        visible_end = min(len(self._items), visible_start + self._virtual_size)
        
        # Update if range changed significantly
        if abs(visible_start - self._rendered_range[0]) > self._virtual_size // 4:
            self._rendered_range = (visible_start, visible_end)
            self._update_visible_items()


class LazyContainer(Container):
    """A Container that defers child creation until visible.
    
    Useful for complex layouts where sections might not be immediately visible.
    """
    
    def __init__(
        self,
        *children,
        content_factory: Optional[Callable[[], ComposeResult]] = None,
        load_on_mount: bool = False,
        **kwargs
    ):
        """Initialize LazyContainer.
        
        Args:
            *children: Initial children (if any)
            content_factory: Factory for creating content
            load_on_mount: Whether to load content on mount
            **kwargs: Additional Container arguments
        """
        super().__init__(*children, **kwargs)
        self._content_factory = content_factory
        self._content_loaded = False
        self._load_on_mount = load_on_mount
        
    def compose(self) -> ComposeResult:
        """Compose with placeholder or content."""
        if self._load_on_mount and self._content_factory:
            yield from self._load_content()
        else:
            # Start with placeholder
            yield Static("Loading...", classes="lazy-placeholder")
    
    def on_mount(self) -> None:
        """Handle mount event."""
        if self._load_on_mount and not self._content_loaded:
            self.load_content()
    
    def load_content(self) -> None:
        """Trigger content loading."""
        if not self._content_loaded and self._content_factory:
            self.call_after_refresh(self._async_load_content)
    
    def _load_content(self) -> ComposeResult:
        """Load the actual content."""
        if self._content_factory:
            try:
                yield from self._content_factory()
                self._content_loaded = True
            except Exception as e:
                logger.error(f"Error loading LazyContainer content: {e}")
                yield Static(f"Error: {e}", classes="error-message")
    
    async def _async_load_content(self) -> None:
        """Asynchronously load content."""
        if self._content_loaded or not self._content_factory:
            return
        
        try:
            # Clear placeholder
            await self.clear()
            
            # Mount actual content
            widgets = list(self._content_factory())
            if widgets:
                await self.mount(*widgets)
                self._content_loaded = True
        except Exception as e:
            logger.error(f"Error async loading LazyContainer: {e}")
            await self.mount(Static(f"Error: {e}", classes="error-message"))