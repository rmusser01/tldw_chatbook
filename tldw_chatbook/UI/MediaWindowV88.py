"""
MediaWindow V88 - Complete rebuild following Textual best practices.

This is the main orchestrator for the media browsing interface, providing:
- Left navigation column with type selector and item list
- Right content area with search, metadata, and content viewing
- Reactive state management
- Event-driven component communication
"""

from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.message import Message
from textual.widgets import Button, Label
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli


class MediaItemSelectedEventV88(Message):
    """Event fired when a media item is selected from the list."""
    
    def __init__(self, media_id: int, media_data: Dict[str, Any]) -> None:
        super().__init__()
        self.media_id = media_id
        self.media_data = media_data


class MediaSearchEventV88(Message):
    """Event fired when search parameters change."""
    
    def __init__(
        self,
        search_term: str,
        keywords: List[str],
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.search_term = search_term
        self.keywords = keywords
        self.filters = filters or {}


class MediaTypeSelectedEventV88(Message):
    """Event fired when media type is selected from dropdown."""
    
    def __init__(self, type_slug: str, display_name: str) -> None:
        super().__init__()
        self.type_slug = type_slug
        self.display_name = display_name


class MediaWindowV88(Container):
    """
    Main orchestrator for the Media UI.
    
    Manages the layout and coordination between navigation, search, 
    metadata display, and content viewing components.
    """
    
    DEFAULT_CSS = """
    MediaWindowV88 {
        layout: horizontal;
        height: 100%;
        width: 100%;
    }
    
    #media-nav-column {
        width: 20%;
        min-width: 25;
        max-width: 40;
        height: 100%;
        border-right: solid $primary;
        background: $surface;
    }
    
    #media-nav-column.collapsed {
        display: none;
    }
    
    #media-content-area {
        width: 1fr;
        height: 100%;
        layout: vertical;
        overflow: hidden;
    }
    
    .placeholder-message {
        text-align: center;
        color: $text-muted;
        padding: 4;
        margin-top: 8;
    }
    """
    
    # Reactive state properties
    active_media_type: reactive[Optional[str]] = reactive("all-media")
    selected_media_id: reactive[Optional[int]] = reactive(None)
    current_media_data: reactive[Optional[Dict[str, Any]]] = reactive(None)
    navigation_collapsed: reactive[bool] = reactive(False)
    search_collapsed: reactive[bool] = reactive(True)
    
    # Search state
    search_term: reactive[str] = reactive("")
    search_keywords: reactive[List[str]] = reactive([])
    search_filters: reactive[Dict[str, Any]] = reactive({})
    
    # Pagination state
    current_page: reactive[int] = reactive(1)
    total_pages: reactive[int] = reactive(1)
    items_per_page: int = 20
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the MediaWindowV88."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_types = self._get_available_media_types()
        self._media_cache: Dict[int, Dict[str, Any]] = {}
        self._search_results_cache: Optional[Tuple[List[Dict], int]] = None
        
    def _get_available_media_types(self) -> List[str]:
        """Get available media types from the app configuration."""
        default_types = [
            "All Media",
            "Article", 
            "Video",
            "Audio", 
            "Document",
            "Book",
            "Podcast",
            "Website"
        ]
        return getattr(self.app_instance, '_media_types_for_ui', default_types)
    
    def compose(self) -> ComposeResult:
        """Compose the Media Window UI structure."""
        # Import all components
        from ..Widgets.MediaV88 import (
            NavigationColumn,
            SearchBar,
            MetadataPanel,
            ContentViewerTabs
        )
        
        # Navigation column
        self.nav_column = NavigationColumn(
            self.app_instance,
            self.media_types,
            id="media-nav-column"
        )
        yield self.nav_column
        
        # Main content area
        with Container(id="media-content-area"):
            # Search bar
            self.search_bar = SearchBar(
                self.app_instance,
                id="media-search-bar"
            )
            yield self.search_bar
            
            # Metadata panel
            self.metadata_panel = MetadataPanel(
                self.app_instance,
                id="media-metadata-panel"
            )
            yield self.metadata_panel
            
            # Content viewer tabs
            self.content_viewer = ContentViewerTabs(
                self.app_instance,
                id="media-content-viewer"
            )
            yield self.content_viewer
    
    def on_mount(self) -> None:
        """Initialize the window when mounted."""
        logger.info("MediaWindowV88 mounted")
        
        # Set initial media type if available
        if self.media_types and not self.active_media_type:
            # Default to "All Media" if available
            if "All Media" in self.media_types:
                # Don't trigger search on initial mount in tests
                self.active_media_type = "all-media"
            else:
                # Use first available type
                first_type = self.media_types[0]
                from ..Utils.text import slugify
                self.active_media_type = slugify(first_type)
    
    def watch_navigation_collapsed(self, collapsed: bool) -> None:
        """React to navigation collapse state changes."""
        if hasattr(self, 'nav_column'):
            if collapsed:
                self.nav_column.add_class("collapsed")
            else:
                self.nav_column.remove_class("collapsed")
    
    def watch_active_media_type(self, media_type: Optional[str]) -> None:
        """React to media type changes."""
        if media_type:
            logger.info(f"Active media type changed to: {media_type}")
            # Don't auto-trigger search on initial set
            pass
    
    def watch_selected_media_id(self, media_id: Optional[int]) -> None:
        """React to media selection changes."""
        if media_id:
            logger.info(f"Media item selected: {media_id}")
            self.load_media_details(media_id)
    
    @on(MediaTypeSelectedEventV88)
    def handle_media_type_selected(self, event: MediaTypeSelectedEventV88) -> None:
        """Handle media type selection from navigation."""
        self.activate_media_type(event.type_slug, event.display_name)
    
    @on(MediaItemSelectedEventV88)
    def handle_media_item_selected(self, event: MediaItemSelectedEventV88) -> None:
        """Handle media item selection from list."""
        logger.info(f"Media item selected: {event.media_id}")
        self.selected_media_id = event.media_id
        
        # Load full media details from database (this will update metadata and content panels)
        self.load_media_details(event.media_id)
    
    @on(MediaSearchEventV88)
    def handle_media_search(self, event: MediaSearchEventV88) -> None:
        """Handle search event from search bar."""
        logger.info(f"Search triggered: '{event.search_term}' with keywords: {event.keywords}")
        
        self.search_term = event.search_term
        self.search_keywords = event.keywords
        self.search_filters = event.filters
        self.current_page = 1  # Reset to first page on new search
        
        self.perform_search()
    
    def activate_media_type(self, type_slug: str, display_name: str) -> None:
        """Activate a specific media type and refresh the view."""
        logger.info(f"Activating media type: {type_slug} ({display_name})")
        
        self.active_media_type = type_slug
        self.current_page = 1
        self.selected_media_id = None
        self.current_media_data = None
        
        # Update navigation if it exists
        if hasattr(self, 'nav_column'):
            try:
                self.nav_column.set_media_type(type_slug, display_name)
            except AttributeError:
                pass  # Component method not implemented yet
        
        # Update search bar if it exists
        if hasattr(self, 'search_bar'):
            try:
                self.search_bar.set_type_filter(type_slug, display_name)
            except AttributeError:
                pass  # Component method not implemented yet
        
        # Clear displays
        if hasattr(self, 'metadata_panel'):
            try:
                self.metadata_panel.clear_display()
            except AttributeError:
                pass
        
        if hasattr(self, 'content_viewer'):
            try:
                self.content_viewer.clear_display()
            except AttributeError:
                pass
        
        # Perform initial search
        self.perform_search()
    
    @work(exclusive=True, exit_on_error=False)
    async def perform_search(self) -> None:
        """Execute media search with current parameters."""
        logger.info(f"Performing search: type={self.active_media_type}, term='{self.search_term}'")
        
        try:
            if not self.app_instance.media_db:
                logger.error("Media database not available")
                return
            
            # Set loading state if navigation exists
            if hasattr(self, 'nav_column'):
                try:
                    self.nav_column.set_loading(True)
                except AttributeError:
                    pass
            
            # Prepare search parameters
            media_types_filter = None
            if self.active_media_type and self.active_media_type != "all-media":
                # Convert slug to database type
                db_media_type = self.active_media_type.replace('-', '_')
                media_types_filter = [db_media_type]
            
            # Execute search
            results, total_matches = await self.search_media_async(
                query=self.search_term if self.search_term else None,
                media_types=media_types_filter,
                keywords=self.search_keywords if self.search_keywords else None,
                page=self.current_page,
                per_page=self.items_per_page
            )
            
            # Calculate total pages
            self.total_pages = max(1, (total_matches + self.items_per_page - 1) // self.items_per_page)
            
            # Update navigation list if it exists
            if hasattr(self, 'nav_column'):
                try:
                    self.nav_column.load_items(results, self.current_page, self.total_pages)
                except AttributeError:
                    pass
            
            logger.info(f"Search complete: {len(results)} results (page {self.current_page}/{self.total_pages})")
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            self.app_instance.notify(f"Search error: {str(e)[:100]}", severity="error")
        finally:
            # Clear loading state
            if hasattr(self, 'nav_column'):
                try:
                    self.nav_column.set_loading(False)
                except AttributeError:
                    pass
    
    async def search_media_async(
        self,
        query: Optional[str] = None,
        media_types: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Async wrapper for database search."""
        import asyncio
        
        # Run synchronous database operation in thread pool
        def search_sync():
            return self.app_instance.media_db.search_media_db(
                search_query=query,
                media_types=media_types,
                search_fields=['title', 'content', 'author', 'url', 'type'],
                must_have_keywords=keywords,
                sort_by="last_modified_desc",
                page=page,
                results_per_page=per_page,
                include_trash=False,
                include_deleted=False
            )
        
        return await asyncio.to_thread(search_sync)
    
    @work(exclusive=True)
    async def load_media_details(self, media_id: int) -> None:
        """Load full media details including content."""
        logger.info(f"Loading details for media ID: {media_id}")
        
        try:
            # Check cache first
            if media_id in self._media_cache:
                full_data = self._media_cache[media_id]
            else:
                # Fetch from database
                if not self.app_instance.media_db:
                    logger.error("Media database not available")
                    return
                
                import asyncio
                full_data = await asyncio.to_thread(
                    self.app_instance.media_db.get_media_by_id,
                    media_id,
                    include_trash=True
                )
                
                if full_data:
                    # Cache the result (limit cache size)
                    if len(self._media_cache) > 100:
                        # Remove oldest entries
                        self._media_cache = dict(list(self._media_cache.items())[-50:])
                    self._media_cache[media_id] = full_data
            
            if full_data:
                self.current_media_data = full_data
                
                # Must update UI from main thread when in a worker
                def update_ui():
                    self.metadata_panel.load_media(full_data)
                    self.content_viewer.load_media(full_data)
                
                # Schedule the UI update in the main thread
                self.app.call_after_refresh(update_ui)
                
                logger.info(f"Loaded media details: {full_data.get('title', 'Unknown')}")
            else:
                logger.warning(f"No data found for media ID: {media_id}")
                self.app_instance.notify("Media item not found", severity="warning")
                
        except Exception as e:
            logger.error(f"Failed to load media details: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to load media: {str(e)[:100]}", severity="error")
    
    def handle_page_change(self, page: int) -> None:
        """Handle pagination changes."""
        if page != self.current_page and 1 <= page <= self.total_pages:
            self.current_page = page
            self.perform_search()
    
    def toggle_navigation(self) -> None:
        """Toggle the navigation column visibility."""
        self.navigation_collapsed = not self.navigation_collapsed
    
    def refresh_current_view(self) -> None:
        """Refresh the current view (e.g., after an update)."""
        self.perform_search()
        if self.selected_media_id:
            self.load_media_details(self.selected_media_id)
    
    def activate_initial_view(self) -> None:
        """Activate the initial view when the tab is first shown."""
        logger.info("Activating initial media view")
        if not self.active_media_type and self.media_types:
            # Default to "All Media" if available
            if "All Media" in self.media_types:
                self.activate_media_type("all-media", "All Media")
            else:
                # Use first available type
                first_type = self.media_types[0]
                from ..Utils.text import slugify
                self.activate_media_type(slugify(first_type), first_type)