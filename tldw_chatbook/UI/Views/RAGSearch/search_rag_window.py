"""
Main RAG Search Window Component

The primary window for RAG search functionality
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple
import asyncio
from datetime import datetime
from pathlib import Path
import json

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll, Grid
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, ListView, ListItem,
    DataTable, Markdown, Label, TabbedContent, TabPane,
    LoadingIndicator, ProgressBar, Collapsible
)
from textual.binding import Binding
from textual.reactive import reactive
from textual.css.query import NoMatches
from rich.markup import escape
from rich.text import Text
from loguru import logger

# Local imports
from .search_history_dropdown import SearchHistoryDropdown
from .search_result import SearchResult
from .saved_searches_panel import SavedSearchesPanel
from .constants import (
    DEFAULT_TOP_K, DEFAULT_TEMPERATURE, DEFAULT_PARENT_SIZE,
    MAX_CONCURRENT_SEARCHES, SEARCH_MODES, PARENT_STRATEGIES
)

from ....Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ....DB.search_history_db import SearchHistoryDB
from ....Utils.paths import get_user_data_dir

# Conditionally import web search functionality
WEB_SEARCH_AVAILABLE = DEPENDENCIES_AVAILABLE.get('websearch', False)
if WEB_SEARCH_AVAILABLE:
    try:
        from ....Web_Scraping.WebSearch_APIs import search_web_bing, parse_bing_results
        logger.info("✅ Web Search dependencies found. Feature is enabled.")
    except (ImportError, ModuleNotFoundError) as e:
        WEB_SEARCH_AVAILABLE = False
        logger.warning(f"⚠️ Web Search dependencies not found, feature will be disabled. Reason: {e}")
        # Define placeholders
        def search_web_bing(*args, **kwargs):
            raise ImportError("Web search not available - missing dependencies")
        def parse_bing_results(*args, **kwargs):
            raise ImportError("Web search not available - missing dependencies")
else:
    # Define placeholders
    def search_web_bing(*args, **kwargs):
        raise ImportError("Web search not available - missing dependencies")
    def parse_bing_results(*args, **kwargs):
        raise ImportError("Web search not available - missing dependencies")

# Conditionally import RAG-related modules
try:
    from ....Event_Handlers.Chat_Events.chat_rag_events import (
        perform_plain_rag_search, perform_full_rag_pipeline, perform_hybrid_rag_search
    )
    RAG_EVENTS_AVAILABLE = True
    
    # Try to import pipeline integration
    try:
        from ....RAG_Search.pipeline_integration import get_pipeline_manager
        PIPELINE_INTEGRATION_AVAILABLE = True
    except ImportError:
        logger.info("Pipeline integration not available")
        PIPELINE_INTEGRATION_AVAILABLE = False
        
except ImportError as e:
    logger.warning(f"RAG event handlers not available: {e}")
    RAG_EVENTS_AVAILABLE = False
    PIPELINE_INTEGRATION_AVAILABLE = False
    # Create placeholder functions
    async def perform_plain_rag_search(*args, **kwargs):
        raise ImportError("RAG search not available - missing dependencies")
    async def perform_full_rag_pipeline(*args, **kwargs):
        raise ImportError("RAG pipeline not available - missing dependencies")
    async def perform_hybrid_rag_search(*args, **kwargs):
        raise ImportError("Hybrid RAG search not available - missing dependencies")

try:
    from tldw_chatbook.RAG_Search.simplified import (
        RAGService, create_config_for_collection, RAGConfig, IndexingResult,
        create_rag_service, get_available_profiles
    )
    RAG_SERVICES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Simplified RAG services not available: {e}")
    RAG_SERVICES_AVAILABLE = False
    # Create placeholder classes
    class RAGService:
        def __init__(self, *args, **kwargs):
            raise ImportError("RAGService not available - missing dependencies")
    def create_config_for_collection(*args, **kwargs):
        raise ImportError("RAG configuration not available - missing dependencies")
    class RAGConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("RAGConfig not available - missing dependencies")
    class IndexingResult:
        pass
    def create_rag_service(*args, **kwargs):
        raise ImportError("RAG service factory not available - missing dependencies")
    def get_available_profiles():
        return []

try:
    from tldw_chatbook.RAG_Search.Services.embeddings_service import EmbeddingsService
    EMBEDDINGS_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"EmbeddingsService not available: {e}")
    EMBEDDINGS_SERVICE_AVAILABLE = False
    # Create placeholder class
    class EmbeddingsService:
        def __init__(self, *args, **kwargs):
            raise ImportError("EmbeddingsService not available - missing dependencies")

if TYPE_CHECKING:
    from ....app import TldwCli

logger = logger.bind(module="SearchRAGWindow")

# Import event handler mixin
from .search_event_handlers import SearchEventHandlersMixin


class SearchRAGWindow(SearchEventHandlersMixin, Container):
    """Enhanced RAG search window with improved visual design and UX"""
    
    BINDINGS = [
        Binding("ctrl+s", "save_search", "Save Search"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+e", "export", "Export Results"),
        Binding("ctrl+c", "clear", "Clear Results"),
        Binding("ctrl+i", "index", "Index Content"),
        Binding("f", "focus_search", "Focus Search", priority=True),
    ]
    
    # Reactive attributes for state management
    is_searching = reactive(False)
    current_search_mode = reactive("plain")
    enable_parent_docs = reactive(False)
    parent_retrieval_strategy = reactive("full")
    parent_retrieval_size = reactive(DEFAULT_PARENT_SIZE)
    
    def __init__(self, app_instance: "TldwCli", id: str = None):
        super().__init__(id=id)
        self.app_instance = app_instance
        self.search_results: List[Dict[str, Any]] = []
        history_db_path = get_user_data_dir() / "search_history.db"
        self.search_history_db = SearchHistoryDB(history_db_path)
        self.active_searches = 0
        self.current_page = 1
        self.results_per_page = 10
        self.total_results = 0
        self.rag_service: Optional[RAGService] = None
        self.available_collections: List[str] = []
        self.last_search_config: Optional[Dict[str, Any]] = None
        
        # Performance tracking
        self.last_search_time = 0.0
        self.search_metrics: Dict[str, Any] = {
            "total_searches": 0,
            "avg_search_time": 0.0,
            "fastest_search": float('inf'),
            "slowest_search": 0.0
        }
        
    def compose(self) -> ComposeResult:
        """Create the UI layout"""
        with Container(id="search-rag-container", classes="search-rag-container-enhanced"):
            # Header with branding and quick stats
            with Horizontal(id="search-header-enhanced", classes="search-header-enhanced"):
                yield Static("🔍 RAG Search Engine", classes="search-title-enhanced")
                yield Static(
                    f"[bold cyan]Collections:[/bold cyan] {len(self.available_collections)} | "
                    f"[bold green]History:[/bold green] 0",
                    id="search-stats",
                    classes="search-stats"
                )
            
            # Main content area with tabbed interface
            with TabbedContent(id="search-tabs", initial="search-tab"):
                # Search Tab
                with TabPane("🔍 Search", id="search-tab"):
                    with Container(classes="search-tab-content"):
                        # Search controls section
                        with Collapsible(title="Search Configuration", collapsed=False, id="search-config-section"):
                            # Query input with history dropdown
                            with Container(classes="search-input-wrapper-enhanced"):
                                search_input = Input(
                                    placeholder="Enter your search query...",
                                    id="search-query-input",
                                    classes="search-query-input-enhanced"
                                )
                                yield search_input
                                yield SearchHistoryDropdown(self.search_history_db)
                                
                            # Search mode and collection selection
                            with Grid(classes="search-options-grid-enhanced"):
                                # Search mode selection
                                with Vertical(classes="option-group"):
                                    yield Label("Search Mode", classes="option-label")
                                    yield Select(
                                        [(label, mode) for mode, label in SEARCH_MODES.items()],
                                        value="plain",
                                        id="search-mode-select",
                                        classes="search-mode-select"
                                    )
                                
                                # Collection selection
                                with Vertical(classes="option-group"):
                                    yield Label("Collection", classes="option-label")
                                    yield Select(
                                        [("All Collections", "all")] + [(c, c) for c in self.available_collections],
                                        value="all",
                                        id="collection-select",
                                        classes="collection-select"
                                    )
                                
                                # Top K results
                                with Vertical(classes="option-group"):
                                    yield Label("Results Count", classes="option-label")
                                    yield Input(
                                        value=str(DEFAULT_TOP_K),
                                        type="integer",
                                        id="top-k-input",
                                        classes="numeric-input"
                                    )
                                
                                # Temperature (for contextual search)
                                with Vertical(classes="option-group"):
                                    yield Label("Temperature", classes="option-label")
                                    yield Input(
                                        value=str(DEFAULT_TEMPERATURE),
                                        type="number",
                                        id="temperature-input",
                                        classes="numeric-input",
                                        disabled=True
                                    )
                            
                            # Advanced options
                            with Collapsible(title="Advanced Options", collapsed=True, id="advanced-options"):
                                # Parent document retrieval
                                yield Checkbox(
                                    "Enable Parent Document Retrieval",
                                    id="parent-docs-checkbox",
                                    value=False,
                                    classes="advanced-checkbox"
                                )
                                
                                with Container(id="parent-docs-options", classes="parent-docs-options disabled"):
                                    with Grid(classes="parent-options-grid"):
                                        with Vertical(classes="option-group"):
                                            yield Label("Strategy", classes="option-label")
                                            yield Select(
                                                [(label, value) for value, label in PARENT_STRATEGIES],
                                                value="full",
                                                id="parent-strategy-select",
                                                classes="parent-strategy-select"
                                            )
                                        
                                        with Vertical(classes="option-group"):
                                            yield Label("Window Size", classes="option-label")
                                            yield Input(
                                                value=str(DEFAULT_PARENT_SIZE),
                                                type="integer",
                                                id="parent-size-input",
                                                classes="numeric-input"
                                            )
                                    
                                    # Parent inclusion preview
                                    yield Container(
                                        Static(
                                            "[dim]Parent document retrieval will include surrounding context[/dim]",
                                            id="parent-preview-text"
                                        ),
                                        id="parent-preview",
                                        classes="parent-preview"
                                    )
                                
                                # Source filtering
                                with Container(classes="source-filters"):
                                    yield Label("Filter by Source", classes="option-label")
                                    with Horizontal(classes="source-checkboxes"):
                                        yield Checkbox("Media", id="filter-media", value=True)
                                        yield Checkbox("Conversations", id="filter-conversations", value=True)
                                        yield Checkbox("Notes", id="filter-notes", value=True)
                                
                                # Web search integration
                                if WEB_SEARCH_AVAILABLE:
                                    yield Checkbox(
                                        "Include Web Search Results",
                                        id="include-web-search",
                                        value=False,
                                        classes="web-search-checkbox"
                                    )
                        
                        # Search buttons
                        with Horizontal(classes="search-buttons-enhanced"):
                            yield Button(
                                "🔍 Search",
                                id="search-button",
                                variant="primary",
                                classes="search-button-primary"
                            )
                            yield Button(
                                "🗑️ Clear",
                                id="clear-search-button",
                                variant="default",
                                classes="search-button-secondary"
                            )
                            yield Button(
                                "💾 Save Search",
                                id="save-search-button",
                                variant="default",
                                classes="search-button-secondary",
                                disabled=True
                            )
                        
                        # Search status and progress
                        with Container(id="search-status-container", classes="search-status-container hidden"):
                            yield LoadingIndicator(id="search-loading")
                            yield Static("", id="search-status-text", classes="search-status-text")
                            yield ProgressBar(id="search-progress", total=100, show_eta=False)
                        
                        # Results section
                        with Container(id="results-container-enhanced", classes="results-container-enhanced"):
                            # Results header
                            with Horizontal(id="results-header-enhanced", classes="results-header-enhanced hidden"):
                                yield Static("", id="results-count", classes="results-count")
                                yield Static("", id="search-time", classes="search-time")
                                with Horizontal(classes="results-actions"):
                                    yield Button("📤 Export", id="export-results", classes="results-action-button")
                                    yield Button("🔄 Refresh", id="refresh-results", classes="results-action-button")
                            
                            # Results list
                            yield VerticalScroll(id="results-list-enhanced", classes="results-list-enhanced")
                            
                            # Pagination
                            with Horizontal(id="pagination-enhanced", classes="pagination-enhanced hidden"):
                                yield Button("← Previous", id="prev-page", classes="pagination-button")
                                yield Static("", id="page-info", classes="page-info")
                                yield Button("Next →", id="next-page", classes="pagination-button")
                
                # Saved Searches Tab
                with TabPane("💾 Saved", id="saved-tab"):
                    yield SavedSearchesPanel()
                
                # History Tab
                with TabPane("📊 History", id="history-tab"):
                    with Container(classes="history-tab-content"):
                        # History controls
                        with Horizontal(classes="history-controls"):
                            yield Label("Time Range:", classes="history-label")
                            yield Select(
                                [
                                    ("Last 7 days", "7"),
                                    ("Last 30 days", "30"),
                                    ("Last 90 days", "90"),
                                    ("All time", "all")
                                ],
                                value="30",
                                id="history-range-select"
                            )
                            yield Button("🔄 Refresh", id="refresh-history", classes="history-refresh-button")
                        
                        # Search history table
                        yield DataTable(id="search-history-table", classes="search-history-table-enhanced")
                        
                        # Analytics summary
                        with Container(id="search-analytics", classes="search-analytics-enhanced"):
                            yield Static("📊 Search Analytics", classes="analytics-title")
                            with Grid(classes="analytics-grid"):
                                yield Static("", id="total-searches-stat", classes="analytics-stat")
                                yield Static("", id="avg-results-stat", classes="analytics-stat")
                                yield Static("", id="popular-queries-stat", classes="analytics-stat")
                                yield Static("", id="search-trends-stat", classes="analytics-stat")
                
                # Maintenance Tab (if RAG services available)
                if RAG_SERVICES_AVAILABLE:
                    with TabPane("🔧 Maintenance", id="maintenance-tab"):
                        with Container(classes="maintenance-tab-content"):
                            yield Static("🔧 Index Management", classes="maintenance-title")
                            
                            # Collection management
                            with Container(classes="collection-management"):
                                yield Label("Available Collections:", classes="maintenance-label")
                                yield ListView(id="collections-list", classes="collections-list")
                                
                                with Horizontal(classes="collection-actions"):
                                    yield Button("➕ Create Collection", id="create-collection", classes="maintenance-button")
                                    yield Button("🗑️ Delete Collection", id="delete-collection", classes="maintenance-button danger")
                                    yield Button("🔄 Refresh Collections", id="refresh-collections", classes="maintenance-button")
                            
                            # Indexing controls
                            with Container(classes="indexing-controls"):
                                yield Static("📥 Index New Content", classes="indexing-title")
                                
                                yield Select(
                                    [
                                        ("Index Media", "media"),
                                        ("Index Conversations", "conversations"),
                                        ("Index Notes", "notes"),
                                        ("Index All Content", "all")
                                    ],
                                    value="all",
                                    id="index-source-select",
                                    classes="index-source-select"
                                )
                                
                                yield Button(
                                    "🚀 Start Indexing",
                                    id="start-indexing",
                                    variant="primary",
                                    classes="indexing-button"
                                )
                                
                                # Indexing status
                                with Container(id="indexing-status", classes="indexing-status hidden"):
                                    yield LoadingIndicator()
                                    yield Static("", id="indexing-status-text")
                                    yield ProgressBar(id="indexing-progress", total=100)
                            
                            # Index statistics
                            with Container(classes="index-statistics"):
                                yield Static("📊 Index Statistics", classes="statistics-title")
                                yield DataTable(id="index-stats-table", classes="index-stats-table")
        

    def on_mount(self) -> None:
        """Called when the widget is mounted"""
        # Check if embeddings/RAG dependencies are available
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            from ....Utils.widget_helpers import alert_embeddings_not_available
            # Show alert after a short delay to ensure UI is ready
            self.set_timer(0.1, lambda: alert_embeddings_not_available(self))
            # Disable search functionality
            self.is_searching = True  # Prevent searches
            try:
                search_input = self.query_one("#search-query-input", Input)
                search_input.disabled = True
                search_input.placeholder = "Embeddings not available - install dependencies"
            except NoMatches:
                pass
        
        # Setup UI components after all widgets are created
        self._setup_history_table()
        self._setup_analytics()
        self._setup_collections_list()
        self._setup_index_stats()
    
    # Setup methods implementation
    def _setup_history_table(self) -> None:
        """Setup the search history table"""
        table = self.query_one("#search-history-table", DataTable)
        table.add_columns("Query", "Type", "Results", "Time", "Date")
        table.cursor_type = "row"
        self._load_recent_search_history()
    
    def _setup_analytics(self) -> None:
        """Setup analytics display"""
        analytics = self.get_search_analytics(days_back=30)
        
        # Update stats
        self.query_one("#total-searches-stat").update(
            f"Total Searches\n[bold cyan]{analytics['total_searches']}[/bold cyan]"
        )
        self.query_one("#avg-results-stat").update(
            f"Avg Results\n[bold green]{analytics['avg_result_count']:.1f}[/bold green]"
        )
        
        # Popular queries
        popular_queries = "\n".join(
            f"• {q['query']} ({q['count']})" 
            for q in analytics['popular_queries'][:3]
        )
        self.query_one("#popular-queries-stat").update(
            f"Popular Queries\n[dim]{popular_queries}[/dim]"
        )
        
        # Search trends
        trend = analytics.get('search_trend', 'stable')
        trend_icon = "📈" if trend == 'increasing' else "📉" if trend == 'decreasing' else "➡️"
        self.query_one("#search-trends-stat").update(
            f"Search Trend\n{trend_icon} {trend.title()}"
        )
    
    def _setup_collections_list(self) -> None:
        """Setup collections list for maintenance"""
        if RAG_SERVICES_AVAILABLE:
            collections_list = self.query_one("#collections-list", ListView)
            self._refresh_collections_list()
    
    def _setup_index_stats(self) -> None:
        """Setup index statistics table"""
        if RAG_SERVICES_AVAILABLE:
            table = self.query_one("#index-stats-table", DataTable)
            table.add_columns("Collection", "Documents", "Chunks", "Size", "Last Updated")
            self._refresh_index_stats()
    
    @work(thread=True)
    async def _refresh_collections_list(self) -> None:
        """Refresh the list of available collections"""
        try:
            # Get available collections
            self.available_collections = get_available_profiles()
            
            # Update UI
            collections_list = self.query_one("#collections-list", ListView)
            await collections_list.clear()
            
            for collection in self.available_collections:
                item = ListItem(Static(collection))
                await collections_list.append(item)
                
            # Update collection select
            collection_select = self.query_one("#collection-select", Select)
            collection_select.set_options(
                [("all", "All Collections")] + [(c, c) for c in self.available_collections]
            )
        except Exception as e:
            logger.error(f"Error refreshing collections: {e}")
    
    @work(thread=True)
    async def _refresh_index_stats(self) -> None:
        """Refresh index statistics"""
        try:
            if not self.rag_service:
                return
                
            # Get stats for each collection
            table = self.query_one("#index-stats-table", DataTable)
            table.clear()
            
            for collection in self.available_collections:
                # Get collection stats (placeholder - implement actual stats retrieval)
                stats = {
                    "documents": 0,
                    "chunks": 0,
                    "size": "0 MB",
                    "last_updated": "N/A"
                }
                
                table.add_row(
                    collection,
                    str(stats["documents"]),
                    str(stats["chunks"]),
                    stats["size"],
                    stats["last_updated"]
                )
        except Exception as e:
            logger.error(f"Error refreshing index stats: {e}")
    
    # Search implementation methods
    async def _perform_plain_search(self, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform plain RAG search"""
        if not RAG_EVENTS_AVAILABLE:
            raise ImportError("RAG search not available")
            
        results = await perform_plain_rag_search(
            query=query,
            api_choice=self.app_instance.api_endpoint,
            filters=config.get('filters', {}),
            top_k=config.get('top_k', DEFAULT_TOP_K),
            collection_name=config.get('collection', 'all')
        )
        
        return self._format_search_results(results, "plain")
    
    async def _perform_contextual_search(self, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform contextual RAG search"""
        if not RAG_EVENTS_AVAILABLE:
            raise ImportError("RAG pipeline not available")
            
        results = await perform_full_rag_pipeline(
            query=query,
            api_choice=self.app_instance.api_endpoint,
            filters=config.get('filters', {}),
            top_k=config.get('top_k', DEFAULT_TOP_K),
            temperature=config.get('temperature', DEFAULT_TEMPERATURE),
            collection_name=config.get('collection', 'all')
        )
        
        return self._format_search_results(results, "contextual")
    
    async def _perform_hybrid_search(self, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform hybrid RAG search"""
        if not RAG_EVENTS_AVAILABLE:
            raise ImportError("Hybrid search not available")
            
        results = await perform_hybrid_rag_search(
            query=query,
            api_choice=self.app_instance.api_endpoint,
            filters=config.get('filters', {}),
            top_k=config.get('top_k', DEFAULT_TOP_K),
            collection_name=config.get('collection', 'all')
        )
        
        return self._format_search_results(results, "hybrid")
    
    async def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search"""
        if not WEB_SEARCH_AVAILABLE:
            return []
            
        try:
            web_results = await search_web_bing(query)
            parsed_results = parse_bing_results(web_results)
            
            # Format web results to match our structure
            formatted_results = []
            for idx, result in enumerate(parsed_results[:5]):  # Limit to 5 web results
                formatted_results.append({
                    "title": result.get("name", "Web Result"),
                    "content": result.get("snippet", ""),
                    "source": "web",
                    "score": 0.5,  # Fixed score for web results
                    "metadata": {
                        "url": result.get("url", ""),
                        "displayUrl": result.get("displayUrl", ""),
                        "datePublished": result.get("datePublished", "")
                    }
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def _format_search_results(self, raw_results: Any, search_type: str) -> List[Dict[str, Any]]:
        """Format raw search results into consistent structure"""
        formatted_results = []
        
        # Handle different result formats based on search type
        if isinstance(raw_results, list):
            for idx, result in enumerate(raw_results):
                if isinstance(result, dict):
                    formatted_results.append(result)
                elif hasattr(result, '__dict__'):
                    # Convert object to dict
                    formatted_results.append(vars(result))
                else:
                    # Wrap simple results
                    formatted_results.append({
                        "title": f"Result {idx + 1}",
                        "content": str(result),
                        "source": "unknown",
                        "score": 0.0
                    })
        
        return formatted_results
    
    async def _display_results(self) -> None:
        """Display search results with pagination"""
        results_list = self.query_one("#results-list-enhanced")
        await results_list.remove_children()
        
        # Calculate pagination
        start_idx = (self.current_page - 1) * self.results_per_page
        end_idx = start_idx + self.results_per_page
        page_results = self.search_results[start_idx:end_idx]
        
        # Display results
        for idx, result in enumerate(page_results):
            result_widget = SearchResult(result, start_idx + idx)
            await results_list.mount(result_widget)
            
            # Add expand/collapse handler
            expand_button = result_widget.query_one(f"#expand-{start_idx + idx}")
            expand_button.on_click = lambda idx=start_idx + idx: self._toggle_result_expansion(idx)
        
        # Update pagination
        self._update_pagination()
    
    def _toggle_result_expansion(self, index: int) -> None:
        """Toggle expanded view of a result"""
        try:
            result_widget = self.query_one(f"#result-{index}", SearchResult)
            expanded_content = result_widget.query_one(f"#expanded-{index}")
            expand_button = result_widget.query_one(f"#expand-{index}", Button)
            
            result_widget.expanded = not result_widget.expanded
            
            if result_widget.expanded:
                expanded_content.remove_class("hidden")
                expand_button.label = "🔼 Hide"
            else:
                expanded_content.add_class("hidden")
                expand_button.label = "🔽 View"
        except NoMatches:
            pass
    
    def _update_pagination(self) -> None:
        """Update pagination controls"""
        if self.total_results <= self.results_per_page:
            self.query_one("#pagination-enhanced").add_class("hidden")
            return
            
        self.query_one("#pagination-enhanced").remove_class("hidden")
        
        total_pages = (self.total_results + self.results_per_page - 1) // self.results_per_page
        self.query_one("#page-info").update(
            f"Page {self.current_page} of {total_pages} ({self.total_results} results)"
        )
        
        self.query_one("#prev-page", Button).disabled = self.current_page <= 1
        self.query_one("#next-page", Button).disabled = self.current_page >= total_pages
    
    def _load_recent_search_history(self, limit: int = 20):
        """Load recent search history into the table"""
        table = self.query_one("#search-history-table", DataTable)
        table.clear()
        
        history = self.search_history_db.get_search_history(limit=limit)
        for item in history:
            table.add_row(
                item['query'],
                item.get('search_type', 'plain'),
                str(item.get('results_count', 0)),
                f"{item.get('search_time', 0):.2f}s",
                datetime.fromisoformat(item['timestamp']).strftime("%Y-%m-%d %H:%M")
            )
    
    def _record_search_to_history(
        self,
        query: str,
        search_type: str,
        filters: Dict[str, Any],
        results_count: int
    ) -> None:
        """Record search to history database"""
        try:
            self.search_history_db.add_search(
                query=query,
                search_type=search_type,
                results_count=results_count,
                search_time=self.last_search_time,
                filters=filters
            )
        except Exception as e:
            logger.error(f"Error recording search to history: {e}")
    
    def _update_last_search_results_count(self, count: int) -> None:
        """Update the results count for the last search"""
        try:
            # Get the last search ID
            history = self.search_history_db.get_search_history(limit=1)
            if history:
                search_id = history[0]['id']
                # Update results count
                self.search_history_db.update_search_results_count(search_id, count)
        except Exception as e:
            logger.error(f"Error updating search results count: {e}")
    
    @work(thread=True)
    def _update_history_table_async(self) -> None:
        """Update history table in background"""
        self._load_recent_search_history()
    
    # Action methods
    def action_focus_search(self) -> None:
        """Focus the search input"""
        self.query_one("#search-query-input", Input).focus()
    
    def action_save_search(self) -> None:
        """Save current search configuration"""
        if not self.last_search_config:
            self.app_instance.notify("No search to save", severity="warning")
            return
            
        # TODO: Implement save dialog
        self.app_instance.notify("Save search dialog not yet implemented", severity="information")
    
    def action_refresh(self) -> None:
        """Refresh current view"""
        active_tab = self.query_one("#search-tabs", TabbedContent).active
        
        if active_tab == "history-tab":
            self._update_history_table_async()
        elif active_tab == "maintenance-tab":
            self._refresh_collections_list()
            self._refresh_index_stats()
    
    def action_export(self) -> None:
        """Export search results"""
        if not self.search_results:
            self.app_instance.notify("No results to export", severity="warning")
            return
            
        # Export results to JSON
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = Path.home() / f"rag_search_results_{timestamp}.json"
            
            export_data = {
                "query": self.query_one("#search-query-input", Input).value,
                "timestamp": datetime.now().isoformat(),
                "config": self.last_search_config,
                "results": self.search_results
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            self.app_instance.notify(
                f"Results exported to {export_path}",
                severity="information"
            )
        except Exception as e:
            logger.error(f"Export error: {e}")
            self.app_instance.notify(
                f"Export failed: {str(e)}",
                severity="error"
            )
    
    def action_clear(self) -> None:
        """Clear search results"""
        self.call_later(self.handle_clear_search, Button.Pressed(self.query_one("#clear-search-button")))
    
    def action_index(self) -> None:
        """Open indexing tab"""
        tabs = self.query_one("#search-tabs", TabbedContent)
        tabs.active = "maintenance-tab"
    
    def get_search_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get search analytics for the specified period"""
        try:
            return self.search_history_db.get_search_analytics(days_back)
        except Exception as e:
            logger.error(f"Error getting search analytics: {e}")
            return {
                "total_searches": 0,
                "avg_result_count": 0,
                "popular_queries": [],
                "search_trend": "stable"
            }