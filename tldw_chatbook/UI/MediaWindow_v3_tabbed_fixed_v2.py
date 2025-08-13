"""
MediaWindow v3 Fixed v2 - Properly working tabbed interface for media browsing.
"""

from typing import TYPE_CHECKING, List, Optional, Dict, Any
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import (
    Button, Static, TabbedContent, TabPane,
    DataTable, Label, Input, Select
)
from textual.binding import Binding
from loguru import logger

# Import media components
from ..Widgets.Media import (
    MediaNavigationPanel,
    MediaSearchPanel,
    MediaSearchEvent,
    MediaListPanel,
    MediaItemSelectedEvent,
    MediaViewerPanel
)

# Import events
from ..Widgets.Media.media_navigation_panel import MediaTypeSelectedEvent
from ..Event_Handlers.media_events import (
    MediaMetadataUpdateEvent,
    MediaDeleteConfirmationEvent,
    MediaUndeleteEvent,
    MediaListCollapseEvent,
    SidebarCollapseEvent
)

if TYPE_CHECKING:
    from ..app import TldwCli

logger = logger.bind(module="MediaWindow_v3")


class DetailedMediaView(Container):
    """Container for the Detailed Media View tab content."""
    
    def __init__(self, app_instance: 'TldwCli', media_types: List[str], **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_types = media_types
        
    def compose(self) -> ComposeResult:
        """Compose the detailed view layout."""
        with Container(classes="tab-content-container"):
            # Navigation sidebar
            yield MediaNavigationPanel(
                self.app_instance,
                self.media_types,
                id="media-nav-panel",
                classes="media-sidebar"
            )
            
            # Main content area
            with Container(classes="media-main-content"):
                # Search panel
                yield MediaSearchPanel(
                    self.app_instance,
                    id="media-search-panel",
                    classes="media-search-bar"
                )
                
                # Content area with list and viewer
                with Container(classes="tab-content-container"):
                    # Media list
                    yield MediaListPanel(
                        self.app_instance,
                        id="media-list-panel",
                        classes="media-list-container"
                    )
                    
                    # Media viewer
                    yield MediaViewerPanel(
                        self.app_instance,
                        id="media-viewer-panel",
                        classes="media-viewer-container"
                    )


class AnalysisReviewView(Container):
    """Container for the Analysis Review tab content."""
    
    def compose(self) -> ComposeResult:
        """Compose the analysis review layout."""
        with Container(classes="tab-content-container"):
            # Analysis sidebar
            with VerticalScroll(classes="media-sidebar"):
                yield Static("📊 Analysis Filters", classes="section-title")
                
                yield Label("Type:")
                yield Select(
                    [("All", "all"), ("Summary", "summary"), ("Transcript", "transcript")],
                    id="analysis-type-filter",
                    value="all"
                )
                
                yield Label("Status:")
                yield Select(
                    [("All", "all"), ("Complete", "complete"), ("Pending", "pending")],
                    id="analysis-status-filter",
                    value="all"
                )
                
                yield Button("🔄 Refresh", id="refresh-analyses-btn", classes="action-button")
            
            # Main analysis content
            with Container(classes="media-main-content"):
                with Container(classes="media-search-bar"):
                    yield Input(placeholder="Search analyses...", id="analysis-search-input")
                
                # Analysis content area
                with VerticalScroll():
                    yield Static(
                        "No analyses found. Select media items to generate analyses.",
                        classes="placeholder-text"
                    )


class MultiItemReviewView(Container):
    """Container for the Multi-Item Review tab content."""
    
    def compose(self) -> ComposeResult:
        """Compose the multi-item review layout."""
        with Container(classes="tab-content-container"):
            # Multi-item sidebar
            with VerticalScroll(classes="media-sidebar"):
                yield Static("🎯 Bulk Actions", classes="section-title")
                
                yield Button("Select All", id="select-all-btn", classes="action-button")
                yield Button("Clear Selection", id="clear-selection-btn", classes="action-button")
                yield Button("📝 Bulk Analyze", id="bulk-analyze-btn", classes="action-button")
                yield Button("📤 Bulk Export", id="bulk-export-btn", classes="action-button")
                yield Button("🗑️ Bulk Delete", id="bulk-delete-btn", classes="action-button")
            
            # Main multi-item content
            with Container(classes="media-main-content"):
                yield DataTable(
                    id="multi-item-table",
                    show_header=True,
                    zebra_stripes=True,
                    cursor_type="row"
                )


class MediaWindow(Container):
    """
    Tabbed Media Window with three view modes.
    """
    
    BINDINGS = [
        Binding("ctrl+f", "focus_search", "Focus Search"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+t", "toggle_sidebar", "Toggle Sidebar"),
    ]
    
    DEFAULT_CSS = """
    MediaWindow {
        layout: vertical;
        height: 100%;
        width: 100%;
    }
    
    #media-header {
        height: 3;
        width: 100%;
        background: $surface;
        padding: 0 1;
        border-bottom: solid $primary;
    }
    
    .media-title {
        text-style: bold;
        color: $primary;
    }
    
    .media-stats {
        color: $text-muted;
        text-align: right;
    }
    
    #media-tabs {
        height: 1fr;
        width: 100%;
    }
    
    .tab-content-container {
        height: 100%;
        width: 100%;
        layout: horizontal;
    }
    
    .media-sidebar {
        width: 20%;
        min-width: 20;
        height: 100%;
        border-right: solid $primary;
        background: $panel;
    }
    
    .media-sidebar.collapsed {
        display: none;
    }
    
    .media-main-content {
        layout: vertical;
        width: 1fr;
        height: 100%;
        overflow: hidden;
    }
    
    .media-search-bar {
        height: auto;
        padding: 1;
        background: $surface;
        border-bottom: solid $secondary;
    }
    
    .media-list-container {
        width: 35%;
        height: 100%;
        border-right: solid $secondary;
        background: $panel;
    }
    
    .media-list-container.collapsed {
        display: none;
    }
    
    .media-viewer-container {
        width: 1fr;
        height: 100%;
        overflow: auto;
    }
    
    .placeholder-text {
        align: center middle;
        color: $text-muted;
        text-style: italic;
    }
    
    .section-title {
        text-style: bold;
        color: $primary;
        margin: 1 0;
    }
    
    .action-button {
        width: 100%;
        margin: 0 0;
    }
    """
    
    # Reactive properties
    active_media_type: reactive[Optional[str]] = reactive(None)
    selected_media_id: reactive[Optional[int]] = reactive(None)
    sidebar_collapsed: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the MediaWindow."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_types = self._get_media_types()
        
    def _get_media_types(self) -> List[str]:
        """Get media types from the app instance."""
        return getattr(self.app_instance, '_media_types_for_ui', [])
    
    def compose(self) -> ComposeResult:
        """Compose the MediaWindow UI with tabs."""
        # Header with stats
        with Horizontal(id="media-header"):
            yield Static("📚 Media Library", classes="media-title")
            yield Static(
                f"[bold cyan]Items:[/bold cyan] 0 | "
                f"[bold green]Selected:[/bold green] 0",
                id="media-stats",
                classes="media-stats"
            )
        
        # Main tabbed interface using context manager pattern
        with TabbedContent(initial="detailed-view", id="media-tabs"):
            with TabPane("📖 Detailed View", id="detailed-view"):
                yield DetailedMediaView(self.app_instance, self.media_types)
            
            with TabPane("🔍 Analysis Review", id="analysis-review"):
                yield AnalysisReviewView()
            
            with TabPane("📋 Multi-Item Review", id="multi-item-review"):
                yield MultiItemReviewView()
    
    def on_mount(self) -> None:
        """Initialize the window when mounted."""
        logger.info("MediaWindow mounted")
        # Delay table setup to ensure it's mounted
        self.call_after_refresh(self._setup_multi_item_table)
        self._update_stats()
    
    def _setup_multi_item_table(self) -> None:
        """Set up the multi-item review table."""
        try:
            table = self.query_one("#multi-item-table", DataTable)
            table.add_column("✓", width=3, key="selected")
            table.add_column("Title", width=40, key="title")
            table.add_column("Type", width=15, key="type")
            table.add_column("Date", width=20, key="date")
            logger.info("Multi-item table configured")
        except QueryError:
            logger.debug("Multi-item table not found (expected if not on that tab)")
    
    def _update_stats(self) -> None:
        """Update the header statistics."""
        try:
            stats = self.query_one("#media-stats", Static)
            stats.update(
                f"[bold cyan]Items:[/bold cyan] 0 | "
                f"[bold green]Selected:[/bold green] 0"
            )
        except QueryError:
            pass
    
    # Tab navigation handling
    @on(TabbedContent.TabActivated)
    def on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Handle tab activation."""
        logger.info(f"Tab activated: {event.tab.label}")
        
        if "Analysis" in event.tab.label:
            self._load_analyses()
        elif "Multi-Item" in event.tab.label:
            self._refresh_multi_item_list()
            self.call_after_refresh(self._setup_multi_item_table)
    
    def _load_analyses(self) -> None:
        """Load analyses for the analysis review tab."""
        logger.info("Loading analyses")
        # Implementation would load from database
    
    def _refresh_multi_item_list(self) -> None:
        """Refresh the multi-item list."""
        logger.info("Refreshing multi-item list")
        # Implementation would load from database
    
    # Action handlers
    def action_focus_search(self) -> None:
        """Focus the search input."""
        try:
            # Try to focus the appropriate search based on active tab
            tabs = self.query_one("#media-tabs", TabbedContent)
            active_tab = tabs.active_tab
            if active_tab and "Detailed" in active_tab.label:
                self.query_one("#media-search-panel").focus()
            elif active_tab and "Analysis" in active_tab.label:
                self.query_one("#analysis-search-input").focus()
        except QueryError:
            pass
    
    def action_refresh(self) -> None:
        """Refresh the current view."""
        logger.info("Refreshing media view")
        self._update_stats()
    
    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        self.sidebar_collapsed = not self.sidebar_collapsed
        
        for sidebar in self.query(".media-sidebar"):
            if self.sidebar_collapsed:
                sidebar.add_class("collapsed")
            else:
                sidebar.remove_class("collapsed")
    
    # Event handlers
    @on(MediaItemSelectedEvent)
    def handle_media_selected(self, event: MediaItemSelectedEvent) -> None:
        """Handle media item selection."""
        self.selected_media_id = event.media_id
        logger.info(f"Media item selected: {event.media_id}")
        self._update_stats()
    
    @on(MediaTypeSelectedEvent)
    def handle_media_type_selected(self, event: MediaTypeSelectedEvent) -> None:
        """Handle media type selection."""
        self.active_media_type = event.media_type
        logger.info(f"Media type selected: {event.media_type}")
    
    @on(Button.Pressed, "#select-all-btn")
    def handle_select_all(self) -> None:
        """Select all items."""
        logger.info("Selecting all items")
        # Implementation would select all items in table
    
    @on(Button.Pressed, "#clear-selection-btn") 
    def handle_clear_selection(self) -> None:
        """Clear selection."""
        logger.info("Clearing selection")
        # Implementation would clear selections
    
    @on(Button.Pressed, "#bulk-analyze-btn")
    def handle_bulk_analyze(self) -> None:
        """Start bulk analysis."""
        logger.info("Starting bulk analysis")
        self.app_instance.notify("Bulk analysis started", severity="information")
    
    @on(Button.Pressed, "#refresh-analyses-btn")
    def handle_refresh_analyses(self) -> None:
        """Refresh analyses."""
        self._load_analyses()
    
    @on(Select.Changed)
    def handle_filter_changed(self, event: Select.Changed) -> None:
        """Handle filter changes."""
        if event.select.id in ["analysis-type-filter", "analysis-status-filter"]:
            logger.info(f"Filter changed: {event.select.id} = {event.value}")
            self._load_analyses()