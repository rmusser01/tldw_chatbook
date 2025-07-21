# SearchRAGWindow.py
# Description: Improved RAG search interface with better UX
#
# Imports
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

# Local Imports
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ..DB.search_history_db import SearchHistoryDB
from ..Utils.paths import get_user_data_dir

# Conditionally import RAG-related modules
try:
    from ..Event_Handlers.Chat_Events.chat_rag_events import (
        perform_plain_rag_search, perform_full_rag_pipeline, perform_hybrid_rag_search
    )
    RAG_EVENTS_AVAILABLE = True
    
    # Try to import pipeline integration
    try:
        from ..RAG_Search.pipeline_integration import get_pipeline_manager
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
    from ..app import TldwCli

logger = logger.bind(module="SearchRAGWindow")

# Source type icons and colors
SOURCE_ICONS = {
    "media": "🎬",
    "conversations": "💬", 
    "notes": "📝"
}

SOURCE_COLORS = {
    "media": "cyan",
    "conversations": "green",
    "notes": "yellow"
}

class SearchHistoryDropdown(Container):
    """Dropdown for search history with auto-complete functionality"""
    
    def __init__(self, search_history_db: SearchHistoryDB):
        super().__init__(id="search-history-dropdown", classes="search-history-dropdown hidden")
        self.search_history_db = search_history_db
        self.history_items: List[str] = []
        
    def compose(self) -> ComposeResult:
        yield ListView(id="search-history-list", classes="search-history-list")
            
    async def show_history(self, current_query: str = "") -> None:
        """Show search history filtered by current query"""
        list_view = self.query_one("#search-history-list", ListView)
        await list_view.clear()
        
        # Get recent searches
        history = self.search_history_db.get_search_history(limit=10, days_back=30)
        self.history_items = []
        
        for item in history:
            query = item['query']
            if current_query.lower() in query.lower() or not current_query:
                self.history_items.append(query)
                list_item = ListItem(Static(query, classes="history-item-text"))
                await list_view.append(list_item)
        
        if self.history_items:
            self.remove_class("hidden")
        else:
            self.add_class("hidden")
            
    def hide(self) -> None:
        """Hide the dropdown"""
        self.add_class("hidden")

class SearchResult(Container):
    """Enhanced container for displaying a single search result with better visual design"""
    
    def __init__(self, result: Dict[str, Any], index: int):
        super().__init__(id=f"result-{index}", classes="search-result-card-enhanced")
        self.result = result
        self.index = index
        self.expanded = False
        
    def compose(self) -> ComposeResult:
        """Create the enhanced result display"""
        source = self.result.get('source', 'unknown')
        source_icon = SOURCE_ICONS.get(source, "📄")
        source_color = SOURCE_COLORS.get(source, "white")
        
        with Container(classes="result-card-wrapper"):
            # Left side - Source indicator
            with Vertical(classes="result-source-column"):
                yield Static(source_icon, classes=f"source-icon source-{source}")
                yield Static(source.upper(), classes=f"source-label source-{source}")
            
            # Main content area
            with Vertical(classes="result-content-column"):
                # Header with title and score
                with Horizontal(classes="result-header-enhanced"):
                    yield Static(
                        f"[bold]{self.result['title']}[/bold]",
                        classes="result-title-enhanced"
                    )
                    # Score visualization
                    score = self.result.get('score', 0)
                    yield Container(
                        Static(f"{score:.1%}", classes="score-text"),
                        classes=f"score-indicator score-{'high' if score > 0.7 else 'medium' if score > 0.4 else 'low'}"
                    )
                
                # Content preview with better formatting
                content = self.result.get('content', '')
                content_preview = content[:250] + "..." if len(content) > 250 else content
                yield Static(content_preview, classes="result-preview-enhanced")
                
                # Metadata pills
                if self.result.get('metadata'):
                    with Horizontal(classes="metadata-pills"):
                        for key, value in list(self.result['metadata'].items())[:3]:
                            if value and str(value).strip():
                                yield Static(
                                    f"{key}: {str(value)[:30]}{'...' if len(str(value)) > 30 else ''}",
                                    classes="metadata-pill"
                                )
                        if len(self.result['metadata']) > 3:
                            yield Static(
                                f"+{len(self.result['metadata']) - 3} more",
                                classes="metadata-pill more"
                            )
                
                # Expanded content (initially hidden)
                with Container(id=f"expanded-{self.index}", classes="result-expanded-content hidden"):
                    # Full content
                    yield Static("[bold]Full Content:[/bold]", classes="expanded-section-title")
                    yield Static(content, classes="result-full-content")
                    
                    # Full metadata
                    if self.result.get('metadata'):
                        yield Static("[bold]All Metadata:[/bold]", classes="expanded-section-title")
                        for key, value in self.result['metadata'].items():
                            yield Static(f"• {key}: {value}", classes="metadata-full-item")
                
                # Action bar
                with Horizontal(classes="result-actions-enhanced"):
                    yield Button(
                        "🔽 View" if not self.expanded else "🔼 Hide",
                        id=f"expand-{self.index}",
                        classes="result-button view-button"
                    )
                    yield Button("📋 Copy", id=f"copy-{self.index}", classes="result-button")
                    yield Button("📝 Note", id=f"add-note-{self.index}", classes="result-button")
                    yield Button("📤 Export", id=f"export-{self.index}", classes="result-button")

class SavedSearchesPanel(Container):
    """Enhanced panel for managing saved searches"""
    
    def __init__(self):
        super().__init__(id="saved-searches-panel", classes="saved-searches-panel-enhanced")
        self.saved_searches: Dict[str, Dict[str, Any]] = self._load_saved_searches()
        self.selected_search_name: Optional[str] = None
        
    def compose(self) -> ComposeResult:
        with Container(classes="saved-searches-wrapper"):
            with Horizontal(classes="saved-searches-header"):
                yield Static("💾 Saved Searches", classes="saved-searches-title")
                yield Button("+", id="new-saved-search", classes="new-search-button", tooltip="Save current search")
            
            if self.saved_searches:
                yield ListView(id="saved-searches-list", classes="saved-searches-list-enhanced")
            else:
                yield Static(
                    "No saved searches yet.\nPerform a search and click 'Save Search' to store it.",
                    classes="empty-saved-searches"
                )
            
            with Horizontal(classes="saved-search-actions-enhanced"):
                yield Button("📥 Load", id="load-saved-search", classes="saved-action-button", disabled=True)
                yield Button("🗑️ Delete", id="delete-saved-search", classes="saved-action-button danger", disabled=True)
    
    def _load_saved_searches(self) -> Dict[str, Dict[str, Any]]:
        """Load saved searches from user data"""
        saved_searches_path = get_user_data_dir() / "saved_searches.json"
        if saved_searches_path.exists():
            try:
                with open(saved_searches_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading saved searches: {e}")
        return {}
    
    def save_search(self, name: str, config: Dict[str, Any]) -> None:
        """Save a search configuration"""
        self.saved_searches[name] = {
            "config": config,
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        }
        self._persist_saved_searches()
        self.refresh_list()
    
    def _persist_saved_searches(self) -> None:
        """Save searches to disk"""
        saved_searches_path = get_user_data_dir() / "saved_searches.json"
        saved_searches_path.parent.mkdir(parents=True, exist_ok=True)
        with open(saved_searches_path, 'w') as f:
            json.dump(self.saved_searches, f, indent=2)
    
    async def refresh_list(self) -> None:
        """Refresh the saved searches list"""
        # Check if we have a list view or need to show empty state
        try:
            list_view = self.query_one("#saved-searches-list", ListView)
            await list_view.clear()
            
            for name, data in self.saved_searches.items():
                created = datetime.fromisoformat(data['created_at']).strftime("%Y-%m-%d %H:%M")
                list_item = ListItem(
                    Static(f"{name}\n[dim]{created}[/dim]", classes="saved-search-item")
                )
                await list_view.append(list_item)
                
            # Enable/disable action buttons based on selection
            self.query_one("#load-saved-search").disabled = True
            self.query_one("#delete-saved-search").disabled = True
        except NoMatches:
            # List view doesn't exist, we're showing empty state
            logger.debug("Saved searches list view not found, showing empty state")
            pass

class SearchRAGWindow(Container):
    """Enhanced RAG search interface window with improved UX"""
    
    BINDINGS = [
        Binding("ctrl+s", "save_search", "Save Search"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+e", "export", "Export Results"),
        Binding("ctrl+i", "index", "Index Content"),
        Binding("escape", "clear", "Clear Search"),
        Binding("ctrl+k", "focus_search", "Focus Search", priority=True),
    ]
    
    # Reactive attributes for state management
    is_searching = reactive(False)
    current_page = reactive(1)
    results_per_page = reactive(20)
    total_results = reactive(0)
    
    def __init__(self, app_instance: "TldwCli", id: str = None):
        super().__init__(id=id)
        self.app_instance = app_instance
        self.current_results: List[Dict[str, Any]] = []
        self.all_results: List[Dict[str, Any]] = []  # Store all results for pagination
        self.search_history: List[str] = []
        self.current_search_id: Optional[int] = None
        
        # Initialize components
        history_db_path = get_user_data_dir() / "search_history.db"
        self.search_history_db = SearchHistoryDB(history_db_path)
        self._load_recent_search_history()
        
        # Check dependencies
        self.embeddings_available = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False) and RAG_SERVICES_AVAILABLE
        self.flashrank_available = DEPENDENCIES_AVAILABLE.get('flashrank', False)
        self.rag_search_available = RAG_EVENTS_AVAILABLE and RAG_SERVICES_AVAILABLE
        
        # Search configuration state
        self.current_search_config: Dict[str, Any] = {}
        
    def compose(self) -> ComposeResult:
        """Create the enhanced UI layout"""
        with Container(classes="rag-search-main-wrapper"):
            with VerticalScroll(classes="rag-search-container"):
                # Header Section
                with Container(classes="search-header-section"):
                    yield Static("🔍 RAG Search & Discovery", classes="rag-title-enhanced")
                    if not self.rag_search_available:
                        yield Static(
                            "⚠️ RAG search functionality is limited - install dependencies with: pip install -e '.[embeddings_rag]'",
                            classes="rag-warning"
                        )
                    else:
                        yield Static("Search across your media, conversations, and notes with semantic understanding", classes="rag-subtitle")
                
                # Search Section with visual prominence
                with Container(classes="search-section"):
                    with Container(classes="search-input-wrapper"):
                        with Horizontal(classes="search-bar-enhanced"):
                            self.search_input = Input(
                                placeholder="🔎 Enter your search query...",
                                id="rag-search-input",
                                classes="search-input-enhanced"
                            )
                            self.search_input.tooltip = "Press Ctrl+K to focus • Esc to clear"
                            yield self.search_input
                            yield Button("×", id="clear-search-btn", classes="clear-button hidden", variant="default")
                            yield Button("Search", id="rag-search-btn", classes="primary search-button-enhanced", variant="primary")
                        yield LoadingIndicator(id="search-loading", classes="search-loading-indicator hidden")
                        
                        # Search history dropdown
                        yield SearchHistoryDropdown(self.search_history_db)
                
                # Settings Section with better organization
                with Container(classes="settings-section"):
                    # Quick Settings Row
                    with Container(classes="quick-settings-container"):
                        yield Static("⚙️ Search Configuration", classes="settings-title")
                        
                        with Horizontal(classes="settings-grid"):
                            # Search Mode Selection
                            with Vertical(classes="setting-group"):
                                yield Label("Search Mode:", classes="setting-label")
                                
                                # Build pipeline options dynamically
                                pipeline_options = [
                                    ("📊 Plain RAG (Fast)", "plain"),
                                    ("🧠 Semantic" if self.embeddings_available else "🧠 Semantic (Unavailable)", "full"),
                                    ("🔀 Hybrid" if self.embeddings_available else "🔀 Hybrid (Unavailable)", "hybrid")
                                ]
                                
                                # Add custom pipelines if available
                                if PIPELINE_INTEGRATION_AVAILABLE:
                                    try:
                                        pipeline_manager = get_pipeline_manager()
                                        custom_pipelines = pipeline_manager.list_available_pipelines()
                                        
                                        # Add separator if we have custom pipelines
                                        if custom_pipelines:
                                            pipeline_options.append(("─" * 20, "separator"))
                                            
                                        # Add custom pipelines
                                        for pipeline in custom_pipelines:
                                            if pipeline["enabled"] and pipeline["id"] not in ["plain", "semantic", "full", "hybrid"]:
                                                # Use emoji based on type
                                                emoji = "🔧"  # Default
                                                if "technical" in pipeline.get("tags", []):
                                                    emoji = "🛠️"
                                                elif "support" in pipeline.get("tags", []):
                                                    emoji = "💬"
                                                elif "medical" in pipeline.get("tags", []):
                                                    emoji = "🏥"
                                                elif "legal" in pipeline.get("tags", []):
                                                    emoji = "⚖️"
                                                
                                                label = f"{emoji} {pipeline['name']}"
                                                pipeline_options.append((label, pipeline["id"]))
                                    except Exception as e:
                                        logger.warning(f"Failed to load custom pipelines: {e}")
                                
                                yield Select(
                                    options=pipeline_options,
                                    value="plain",
                                    id="search-mode-select",
                                    classes="mode-select"
                                )
                        
                            # Source Selection
                            with Vertical(classes="setting-group"):
                                yield Label("Search Sources:", classes="setting-label")
                                with Horizontal(classes="source-checkboxes-enhanced"):
                                    yield Checkbox("🎬 Media", value=True, id="source-media", classes="source-checkbox")
                                    yield Checkbox("💬 Chats", value=True, id="source-conversations", classes="source-checkbox")
                                    yield Checkbox("📝 Notes", value=True, id="source-notes", classes="source-checkbox")
                
                # Saved searches panel - moved here for better flow
                yield SavedSearchesPanel()
            
                # Advanced settings with better organization
                with Collapsible(title="🔧 Advanced Settings", collapsed=True, id="advanced-settings-collapsible", classes="advanced-collapsible"):
                    with Container(classes="advanced-settings-wrapper"):
                        # Search parameters section
                        with Container(classes="advanced-section"):
                            yield Static("📊 Result Parameters", classes="advanced-section-title")
                            with Grid(classes="parameter-grid-enhanced"):
                                with Vertical(classes="param-group"):
                                    yield Label("Results to Return:", classes="param-label")
                                    yield Input(value="10", id="top-k-input", type="integer", classes="param-input", tooltip="Number of top results to display")
                                with Vertical(classes="param-group"):
                                    yield Label("Context Length:", classes="param-label")
                                    yield Input(value="10000", id="max-context-input", type="integer", classes="param-input", tooltip="Maximum characters for context window")
                        
                        # Re-ranking section
                        with Container(classes="advanced-section"):
                            yield Static("🎯 Result Optimization", classes="advanced-section-title")
                            yield Checkbox(
                                "Enable Smart Re-ranking" if self.flashrank_available else "Enable Smart Re-ranking (Not Available)",
                                value=self.flashrank_available,
                                id="enable-rerank",
                                disabled=not self.flashrank_available,
                                classes="rerank-checkbox"
                            )
                        
                        # Chunking options (for semantic mode)
                        with Container(classes="advanced-section chunking-section hidden", id="chunking-options"):
                            yield Static("📄 Document Chunking", classes="advanced-section-title")
                            with Grid(classes="parameter-grid-enhanced"):
                                with Vertical(classes="param-group"):
                                    yield Label("Chunk Size:", classes="param-label")
                                    yield Input(value="400", id="chunk-size-input", type="integer", classes="param-input", tooltip="Size of text chunks for processing")
                                with Vertical(classes="param-group"):
                                    yield Label("Overlap:", classes="param-label")
                                    yield Input(value="100", id="chunk-overlap-input", type="integer", classes="param-input", tooltip="Overlap between chunks")
                        
                        # Parent document inclusion settings
                        with Collapsible(title="📚 Parent Document Inclusion", collapsed=True, id="parent-doc-settings", classes="parent-doc-section"):
                            yield Checkbox(
                                "Include parent documents when relevant",
                                value=False,
                                id="include-parent-docs",
                                classes="parent-doc-checkbox",
                                tooltip="Include full parent documents when they're small enough"
                            )
                            
                            with Container(classes="parent-doc-options", id="parent-doc-options"):
                                with Grid(classes="parameter-grid-enhanced"):
                                    with Vertical(classes="param-group"):
                                        yield Label("Parent Size Threshold (chars):", classes="param-label")
                                        yield Input(
                                            value="5000", 
                                            id="parent-size-threshold", 
                                            type="integer", 
                                            classes="param-input",
                                            tooltip="Maximum size for including parent documents"
                                        )
                                    
                                    with Vertical(classes="param-group"):
                                        yield Label("Inclusion Strategy:", classes="param-label")
                                        yield Select(
                                            [
                                                ("Size Based", "size_based"),
                                                ("Always Include", "always"),
                                                ("Never Include", "never")
                                            ],
                                            id="parent-strategy",
                                            classes="param-select",
                                            value="size_based"
                                        )
                                
                                yield Static(
                                    "ℹ️ Parent documents provide additional context for better understanding",
                                    id="parent-inclusion-preview",
                                    classes="parent-doc-info"
                                )
            
            # Status and progress area with better visibility
            with Container(id="status-container", classes="status-container-enhanced"):
                with Horizontal(classes="status-bar"):
                    yield Static("🟢 Ready to search", id="search-status", classes="search-status-enhanced")
                    yield Static("", id="search-stats", classes="search-stats")
                yield ProgressBar(id="search-progress", classes="search-progress-bar hidden", total=100)
            
            # Results area with enhanced tabs and better visual design
            with Container(classes="results-section"):
                yield Static("📋 Search Results", classes="results-title")
                
                with TabbedContent(id="results-tabs", classes="results-tabs-enhanced"):
                    with TabPane("🔍 Results", id="results-tab"):
                        # Results header with summary and controls
                        with Container(classes="results-header"):
                            with Horizontal(classes="results-header-bar"):
                                yield Static(
                                    "💡 Enter a search query to discover relevant content",
                                    id="results-summary",
                                    classes="results-summary-enhanced"
                                )
                                # Pagination controls
                                with Horizontal(classes="pagination-controls-enhanced hidden", id="pagination-controls"):
                                    yield Button("◀", id="prev-page-btn", classes="page-button", disabled=True, tooltip="Previous page")
                                    yield Static("Page 1 of 1", id="page-info", classes="page-info-enhanced")
                                    yield Button("▶", id="next-page-btn", classes="page-button", disabled=True, tooltip="Next page")
                        
                        # Results container with improved styling
                        yield VerticalScroll(id="results-container", classes="results-container-enhanced")
                
                    with TabPane("📄 Context", id="context-tab"):
                        yield Markdown(
                            "*No search performed yet*\n\nThe combined context from your search results will appear here.",
                            id="context-preview",
                            classes="context-preview-enhanced"
                        )
                    
                    with TabPane("🕒 History", id="history-tab"):
                        yield DataTable(id="search-history-table", classes="history-table-enhanced")
                    
                    with TabPane("📊 Analytics", id="analytics-tab"):
                        yield Markdown(
                            "# 📊 Search Analytics\n\n*No analytics data available yet*\n\nPerform searches to see insights about your search patterns and results.",
                            id="analytics-content",
                            classes="analytics-content-enhanced"
                        )
            
            # Action buttons with better organization
            with Container(classes="actions-section"):
                with Horizontal(classes="action-buttons-bar-enhanced"):
                    # Primary actions
                    with Horizontal(classes="primary-actions"):
                        yield Button("💾 Save Search", id="save-search-btn", classes="action-button save-button")
                        yield Button("📤 Export", id="export-results-btn", classes="action-button export-button", disabled=True)
                    
                    # Spacer
                    yield Static("", classes="action-spacer")
                    
                    # Maintenance dropdown
                    with Horizontal(classes="maintenance-actions"):
                        yield Button("⚙️ Maintenance ▼", id="maintenance-menu-btn", classes="action-button maintenance-button")
                        # Hidden menu items
                        with Container(id="maintenance-menu", classes="maintenance-menu hidden"):
                            yield Button("🔄 Index Content", id="index-content-btn", classes="menu-item")
                            yield Button("🗑️ Clear Cache", id="clear-cache-btn", classes="menu-item")
    
    async def on_mount(self) -> None:
        """Initialize the window when mounted"""
        # Set up search history table
        history_table = self.query_one("#search-history-table", DataTable)
        history_table.add_columns("Time", "Query", "Mode", "Results", "Duration")
        history_table.zebra_stripes = True
        
        # Set up saved searches
        saved_searches_panel = self.query_one(SavedSearchesPanel)
        await saved_searches_panel.refresh_list()
        
        # Focus search input
        self.search_input.focus()
        
        # Set up ARIA labels
        self._setup_aria_labels()
        
        # Check indexing status
        await self._check_index_status()
    
    def _setup_aria_labels(self) -> None:
        """Set up ARIA labels for accessibility"""
        self.search_input.aria_label = "Search query input"
        self.query_one("#rag-search-btn").aria_label = "Execute search"
        self.query_one("#search-mode-select").aria_label = "Select search mode"
        self.query_one("#source-media").aria_label = "Include media items in search"
        self.query_one("#source-conversations").aria_label = "Include conversations in search"
        self.query_one("#source-notes").aria_label = "Include notes in search"
        self.query_one("#save-search-btn").aria_label = "Save current search configuration"
        self.query_one("#export-results-btn").aria_label = "Export search results"
    
    def watch_is_searching(self, is_searching: bool) -> None:
        """React to search state changes"""
        # Check if the widgets exist before trying to query them
        # This can be called before compose() completes
        try:
            loading = self.query_one("#search-loading")
            search_btn = self.query_one("#rag-search-btn")
            
            if is_searching:
                loading.remove_class("hidden")
                search_btn.disabled = True
                search_btn.label = "Searching..."
            else:
                loading.add_class("hidden")
                search_btn.disabled = False
                search_btn.label = "Search"
        except NoMatches:
            # Widgets not yet created, ignore
            logger.debug("Search loading/button widgets not yet created")
            pass
    
    @on(Input.Changed, "#rag-search-input")
    async def handle_search_input_change(self, event: Input.Changed) -> None:
        """Handle search input changes for history dropdown and clear button"""
        query = event.value.strip()
        history_dropdown = self.query_one(SearchHistoryDropdown)
        clear_button = self.query_one("#clear-search-btn")
        
        # Show/hide clear button
        if query:
            clear_button.remove_class("hidden")
            await history_dropdown.show_history(query)
        else:
            clear_button.add_class("hidden")
            history_dropdown.hide()
    
    @on(Button.Pressed, "#clear-search-btn")
    def handle_clear_search(self, event: Button.Pressed) -> None:
        """Clear the search input"""
        self.search_input.value = ""
        self.search_input.focus()
        event.stop()
    
    @on(Checkbox.Changed, "#include-parent-docs")
    def handle_parent_docs_toggle(self, event: Checkbox.Changed) -> None:
        """Handle parent documents inclusion toggle"""
        parent_options = self.query_one("#parent-doc-options", Container)
        if event.value:
            parent_options.remove_class("hidden")
            # Update preview
            self._update_parent_inclusion_preview()
        else:
            parent_options.add_class("hidden")
    
    @on(Select.Changed, "#parent-strategy")
    def handle_parent_strategy_change(self, event: Select.Changed) -> None:
        """Handle parent inclusion strategy change"""
        self._update_parent_inclusion_preview()
    
    @on(Input.Changed, "#parent-size-threshold")
    def handle_parent_size_change(self, event: Input.Changed) -> None:
        """Handle parent size threshold change"""
        self._update_parent_inclusion_preview()
    
    def _update_parent_inclusion_preview(self) -> None:
        """Update the parent inclusion preview message"""
        try:
            preview = self.query_one("#parent-inclusion-preview", Static)
            strategy = self.query_one("#parent-strategy", Select).value
            threshold = self.query_one("#parent-size-threshold", Input).value
            
            if strategy == "always":
                preview.update("📚 All parent documents will be included regardless of size")
            elif strategy == "never":
                preview.update("❌ Parent documents will not be included")
            else:  # size_based
                try:
                    size = int(threshold)
                    preview.update(f"📏 Parent documents smaller than {size:,} characters will be included")
                except ValueError:
                    preview.update("⚠️ Please enter a valid size threshold")
        except Exception as e:
            logger.debug(f"Error updating parent inclusion preview: {e}")
    
    @on(ListView.Selected, "#search-history-list")
    async def handle_history_selection(self, event: ListView.Selected) -> None:
        """Handle selection from search history"""
        history_dropdown = self.query_one(SearchHistoryDropdown)
        if event.item and history_dropdown.history_items:
            index = event.list_view.index
            if 0 <= index < len(history_dropdown.history_items):
                self.search_input.value = history_dropdown.history_items[index]
                history_dropdown.hide()
                self.search_input.focus()
    
    @on(Button.Pressed, "#rag-search-btn")
    async def handle_search(self, event: Button.Pressed) -> None:
        """Handle search button press"""
        if self.is_searching:
            return
            
        query = self.search_input.value.strip()
        if not query:
            self.app_instance.notify("🔍 Please enter a search query", severity="warning")
            self.search_input.focus()
            return
        
        await self._perform_search(query)
    
    @on(Input.Submitted, "#rag-search-input")
    async def handle_search_submit(self, event: Input.Submitted) -> None:
        """Handle Enter key in search input"""
        await self.handle_search(Button.Pressed(self.query_one("#rag-search-btn")))
    
    @on(Select.Changed, "#search-mode-select")
    def handle_search_mode_change(self, event: Select.Changed) -> None:
        """Show/hide chunking options based on search mode"""
        chunking_options = self.query_one("#chunking-options")
        mode_name = "Semantic" if event.value == "full" else "Hybrid" if event.value == "hybrid" else "Plain"
        
        if event.value == "full":
            chunking_options.remove_class("hidden")
            if not self.embeddings_available:
                self.app_instance.notify(
                    "🔒 Semantic search requires embeddings dependencies",
                    severity="warning",
                    timeout=5
                )
        else:
            chunking_options.add_class("hidden")
        
        # Update status to reflect mode change
        status = self.query_one("#search-status")
        status.update(f"🔄 Switched to {mode_name} search mode")
    
    @work(exclusive=True)
    async def _perform_search(self, query: str) -> None:
        """Perform the actual search with streaming results"""
        if not self.rag_search_available:
            self.app_instance.notify(
                "RAG search functionality is not available. Please install required dependencies.",
                severity="error",
                timeout=5
            )
            return
            
        self.is_searching = True
        start_time = datetime.now()
        
        # Update status
        status_elem = self.query_one("#search-status")
        progress_bar = self.query_one("#search-progress", ProgressBar)
        progress_bar.remove_class("hidden")
        progress_bar.update(progress=0)
        
        # Clear previous results
        self.all_results = []
        self.current_results = []
        self.current_page = 1
        results_container = self.query_one("#results-container", Container)
        await results_container.remove_children()
        
        try:
            # Get search parameters
            search_mode = self.query_one("#search-mode-select", Select).value
            sources = {
                'media': self.query_one("#source-media", Checkbox).value,
                'conversations': self.query_one("#source-conversations", Checkbox).value,
                'notes': self.query_one("#source-notes", Checkbox).value
            }
            
            if not any(sources.values()):
                self.app_instance.notify("Please select at least one source", severity="warning")
                return
            
            # Store current search configuration
            self.current_search_config = {
                "query": query,
                "mode": search_mode,
                "sources": sources,
                "top_k": int(self.query_one("#top-k-input", Input).value or "10"),
                "max_context": int(self.query_one("#max-context-input", Input).value or "10000"),
                "include_parent_docs": self.query_one("#include-parent-docs", Checkbox).value,
                "parent_size_threshold": int(self.query_one("#parent-size-threshold", Input).value or "5000"),
                "parent_inclusion_strategy": self.query_one("#parent-strategy", Select).value,
                "enable_rerank": self.query_one("#enable-rerank", Checkbox).value
            }
            
            # Update status with more descriptive message
            mode_names = {"plain": "Plain", "full": "Semantic", "hybrid": "Hybrid"}
            mode_display = mode_names.get(search_mode, search_mode)
            active_sources = [k for k, v in sources.items() if v]
            sources_display = ", ".join(s.capitalize() for s in active_sources)
            
            await status_elem.update(f"🔍 Searching {sources_display} using {mode_display} mode...")
            await progress_bar.update(progress=20)
            
            # Perform search based on mode
            chunk_size = int(self.query_one("#chunk-size-input", Input).value or "400")
            chunk_overlap = int(self.query_one("#chunk-overlap-input", Input).value or "100")
            
            # Check if pipeline integration is available and try to use it
            if PIPELINE_INTEGRATION_AVAILABLE:
                pipeline_manager = get_pipeline_manager()
                
                # Check if the search_mode is a pipeline ID
                if pipeline_manager.validate_pipeline_id(search_mode):
                    logger.info(f"Using pipeline '{search_mode}' from TOML configuration")
                    
                    # Prepare kwargs for pipeline execution
                    pipeline_kwargs = {
                        "top_k": self.current_search_config["top_k"],
                        "max_context_length": self.current_search_config["max_context"],
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "include_metadata": True,
                        "enable_rerank": self.current_search_config["enable_rerank"],
                        "reranker_model": "flashrank",
                        "bm25_weight": 0.5,
                        "vector_weight": 0.5,
                        "include_parent_docs": self.current_search_config["include_parent_docs"],
                        "parent_size_threshold": self.current_search_config["parent_size_threshold"],
                        "parent_inclusion_strategy": self.current_search_config["parent_inclusion_strategy"]
                    }
                    
                    # Get pipeline default parameters and merge
                    default_params = pipeline_manager.get_pipeline_parameters(search_mode)
                    pipeline_kwargs.update(default_params)
                    
                    # Execute pipeline
                    results, context = await pipeline_manager.execute_pipeline(
                        search_mode, self.app_instance, query, sources, **pipeline_kwargs
                    )
                else:
                    logger.info(f"Pipeline '{search_mode}' not found in TOML, falling back to legacy mode")
                    # Fall through to legacy implementation below
                    results = None
                    context = None
            else:
                results = None
                context = None
                
            # Legacy implementation
            if results is None:
                if search_mode == "plain":
                    results, context = await perform_plain_rag_search(
                        self.app_instance,
                        query,
                        sources,
                        self.current_search_config["top_k"],
                        self.current_search_config["max_context"],
                        self.current_search_config["enable_rerank"],
                        "flashrank"
                    )
                elif search_mode == "full":
                    if not self.embeddings_available:
                        self.app_instance.notify("Embeddings not available, using plain search", severity="info")
                        results, context = await perform_plain_rag_search(
                            self.app_instance,
                            query,
                            sources,
                            self.current_search_config["top_k"],
                            self.current_search_config["max_context"],
                            self.current_search_config["enable_rerank"],
                            "flashrank"
                        )
                    else:
                        results, context = await perform_full_rag_pipeline(
                            self.app_instance,
                            query,
                            sources,
                            self.current_search_config["top_k"],
                            self.current_search_config["max_context"],
                            chunk_size,
                            chunk_overlap,
                            True,  # include_metadata
                            self.current_search_config["enable_rerank"],
                            "flashrank"
                        )
                else:  # hybrid
                    if not self.embeddings_available:
                        self.app_instance.notify("Embeddings not available for hybrid search, using plain search", severity="info")
                        results, context = await perform_plain_rag_search(
                            self.app_instance,
                            query,
                            sources,
                            self.current_search_config["top_k"],
                            self.current_search_config["max_context"],
                            self.current_search_config["enable_rerank"],
                            "flashrank"
                        )
                    else:
                        results, context = await perform_hybrid_rag_search(
                            self.app_instance,
                            query,
                            sources,
                            self.current_search_config["top_k"],
                            self.current_search_config["max_context"],
                            self.current_search_config["enable_rerank"],
                            "flashrank",
                            chunk_size,
                            chunk_overlap,
                            0.5,  # BM25 weight
                            0.5   # Vector weight
                        )
            
            await progress_bar.update(progress=80)
            
            # Store results
            self.all_results = results
            self.total_results = len(results)
            
            # Display results with pagination
            await self._display_results_page(context)
            
            # Enable export button if we have results
            if results:
                self.query_one("#export-results-btn").disabled = False
            
            # Record search to history
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.current_search_id = self._record_search_to_history(
                query=query,
                search_type=search_mode,
                results=results,
                execution_time_ms=duration_ms,
                search_params=self.current_search_config
            )
            
            # Update history dropdown
            history_dropdown = self.query_one(SearchHistoryDropdown)
            history_dropdown.hide()
            
            # Update final status
            if len(results) > 0:
                await status_elem.update(f"✅ Found {len(results)} results in {duration_ms/1000:.2f}s")
            else:
                await status_elem.update(f"🔍 No results found in {duration_ms/1000:.2f}s - try different keywords")
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            error_msg = str(e)
            if "embeddings" in error_msg.lower():
                await status_elem.update("❌ Embeddings not available - using plain search instead")
                # Fallback to plain search
                if search_mode != "plain":
                    self.query_one("#search-mode-select").value = "plain"
                    await self._perform_search(query)
                    return
            else:
                await status_elem.update(f"❌ Search error: {error_msg[:100]}...")
            self.app_instance.notify(f"Search error: {error_msg[:100]}...", severity="error")
            
        finally:
            self.is_searching = False
            progress_bar.add_class("hidden")
            # Ensure maintenance menu is hidden
            try:
                self.query_one("#maintenance-menu").add_class("hidden")
            except NoMatches:
                logger.debug("Maintenance menu not found, likely not created yet")
                pass
    
    async def _display_results_page(self, context: str) -> None:
        """Display current page of results"""
        results_container = self.query_one("#results-container", VerticalScroll)
        await results_container.remove_children()
        
        # Calculate pagination
        start_idx = (self.current_page - 1) * self.results_per_page
        end_idx = start_idx + self.results_per_page
        page_results = self.all_results[start_idx:end_idx]
        
        # Update summary
        summary = self.query_one("#results-summary", Static)
        total_pages = max(1, (self.total_results + self.results_per_page - 1) // self.results_per_page)
        
        if self.total_results == 0:
            await summary.update("💭 No results found. Try adjusting your search query or filters.")
        else:
            await summary.update(
                f"🎯 Found {self.total_results} results • Showing {start_idx + 1}-{min(end_idx, self.total_results)}"
            )
        
        # Update stats
        stats = self.query_one("#search-stats", Static)
        if self.total_results > 0:
            avg_score = sum(r.get('score', 0) for r in self.all_results) / len(self.all_results)
            await stats.update(f"Avg. relevance: {avg_score:.1%}")
        else:
            await stats.update("")
        
        # Show/update pagination controls
        if self.total_results > self.results_per_page:
            pagination = self.query_one("#pagination-controls")
            pagination.remove_class("hidden")
            
            prev_btn = self.query_one("#prev-page-btn", Button)
            next_btn = self.query_one("#next-page-btn", Button)
            page_info = self.query_one("#page-info", Static)
            
            prev_btn.disabled = self.current_page <= 1
            next_btn.disabled = self.current_page >= total_pages
            await page_info.update(f"Page {self.current_page} of {total_pages}")
        else:
            self.query_one("#pagination-controls").add_class("hidden")
        
        # Display results for current page
        if page_results:
            for i, result in enumerate(page_results, start=start_idx):
                result_widget = SearchResult(result, i)
                await results_container.mount(result_widget)
        else:
            # Show empty state
            empty_msg = Static(
                "No results to display. Try a different search query or check your filters.",
                classes="empty-results-message"
            )
            await results_container.mount(empty_msg)
        
        # Update context preview
        context_preview = self.query_one("#context-preview", Markdown)
        if context:
            await context_preview.update(f"## 📝 Combined Context\n\n```\n{context}\n```")
        else:
            await context_preview.update("*No context available yet*\n\nContext will be generated from your search results.")
        
        # Update analytics
        await self._update_analytics()
    
    @on(Button.Pressed, "#prev-page-btn")
    async def handle_prev_page(self, event: Button.Pressed) -> None:
        """Handle previous page button"""
        if self.current_page > 1:
            self.current_page -= 1
            await self._display_results_page("")
    
    @on(Button.Pressed, "#next-page-btn")
    async def handle_next_page(self, event: Button.Pressed) -> None:
        """Handle next page button"""
        total_pages = max(1, (self.total_results + self.results_per_page - 1) // self.results_per_page)
        if self.current_page < total_pages:
            self.current_page += 1
            await self._display_results_page("")
    
    @on(Button.Pressed, "#save-search-btn")
    async def handle_save_search(self, event: Button.Pressed) -> None:
        """Save current search configuration"""
        if not self.current_search_config:
            self.app_instance.notify("⚠️ No search to save", severity="warning")
            return
        
        # Generate a more descriptive name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_name = f"{self.current_search_config['query'][:30]} - {timestamp}"
        
        saved_searches = self.query_one(SavedSearchesPanel)
        saved_searches.save_search(save_name, self.current_search_config)
        self.app_instance.notify(f"✅ Search saved successfully", severity="success")
    
    @on(Button.Pressed, "#new-saved-search")
    async def handle_new_saved_search(self, event: Button.Pressed) -> None:
        """Alias for save search from the saved searches panel"""
        await self.handle_save_search(event)
    
    @on(Button.Pressed, "#maintenance-menu-btn")
    def handle_maintenance_menu(self, event: Button.Pressed) -> None:
        """Toggle maintenance menu dropdown"""
        menu = self.query_one("#maintenance-menu")
        if "hidden" in menu.classes:
            menu.remove_class("hidden")
        else:
            menu.add_class("hidden")
        event.stop()
    
    @on(Button.Pressed, "#index-content-btn")
    async def handle_index_content(self, event: Button.Pressed) -> None:
        """Handle index content button with progress indicator"""
        # Hide the maintenance menu
        self.query_one("#maintenance-menu").add_class("hidden")
        await self._index_all_content()
    
    @on(Button.Pressed, "#clear-cache-btn")
    async def handle_clear_cache(self, event: Button.Pressed) -> None:
        """Handle clear cache button"""
        # Hide the maintenance menu
        self.query_one("#maintenance-menu").add_class("hidden")
        
        try:
            # Clear the embeddings cache in the simplified RAG service
            # We'll need to get or create a RAG service instance
            config = create_config_for_collection(
                "media",
                persist_dir=Path.home() / ".local" / "share" / "tldw_cli" / "chromadb"
            )
            
            # Get profile from config
            settings = self.app_instance.load_settings()
            rag_config = settings.get('rag_search', {})
            service_config = rag_config.get('service', {})
            profile_name = service_config.get('profile', 'hybrid_basic')
            
            # Create RAG service with profile
            rag_service = create_rag_service(
                profile_name=profile_name,
                config=config
            )
            rag_service.clear_cache()
            rag_service.close()
            self.app_instance.notify("✅ Cache cleared successfully", severity="success")
        except ImportError:
            logger.warning("RAG service not available - dependencies missing")
            self.app_instance.notify("❌ RAG service not available - please install RAG dependencies", severity="error")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            self.app_instance.notify(f"❌ Error clearing cache: {str(e)}", severity="error")
    
    @work(exclusive=True)
    async def _index_all_content(self) -> None:
        """Index all content with background progress indicator"""
        if not self.embeddings_available or not RAG_SERVICES_AVAILABLE:
            self.app_instance.notify(
                "Embeddings dependencies not available. Install with: pip install -e '.[embeddings_rag]'",
                severity="warning"
            )
            return
        
        # Update status
        status_elem = self.query_one("#search-status")
        progress_bar = self.query_one("#search-progress", ProgressBar)
        progress_bar.remove_class("hidden")
        
        try:
            await status_elem.update("🔄 Indexing content...")
            
            # Initialize RAG service
            config = create_config_for_collection(
                "media",  # Default collection, will be used for all types
                persist_dir=Path.home() / ".local" / "share" / "tldw_cli" / "chromadb"
            )
            # Get profile from config
            settings = self.app_instance.load_settings()
            rag_config = settings.get('rag_search', {})
            service_config = rag_config.get('service', {})
            profile_name = service_config.get('profile', 'hybrid_basic')
            
            # Create RAG service with profile
            rag_service = create_rag_service(
                profile_name=profile_name,
                config=config
            )
            
            # Index with progress updates
            total_steps = 3
            await progress_bar.update(progress=0, total=total_steps * 100)
            
            async def update_progress(content_type: str, current: int, total: int):
                step = {"media": 0, "conversations": 1, "notes": 2}.get(content_type, 0)
                progress = (step * 100) + int((current / max(total, 1)) * 100)
                await progress_bar.update(progress=progress)
                await status_elem.update(f"🔄 Indexing {content_type}: {current}/{total}")
            
            # Index each content type
            results = {"media": 0, "conversations": 0, "notes": 0}
            
            # Index media content
            if self.app_instance.media_db:
                await update_progress("media", 0, 100)
                media_docs = await self._get_media_documents()
                for i, doc in enumerate(media_docs):
                    await update_progress("media", i, len(media_docs))
                    result = await rag_service.index_document(
                        doc_id=doc["id"],
                        content=doc["content"],
                        title=doc["title"],
                        metadata=doc["metadata"]
                    )
                    if result.success:
                        results["media"] += 1
            
            # Index conversations
            if self.app_instance.chachanotes_db:
                await update_progress("conversations", 0, 100)
                conv_docs = await self._get_conversation_documents()
                for i, doc in enumerate(conv_docs):
                    await update_progress("conversations", i, len(conv_docs))
                    result = await rag_service.index_document(
                        doc_id=doc["id"],
                        content=doc["content"],
                        title=doc["title"],
                        metadata=doc["metadata"]
                    )
                    if result.success:
                        results["conversations"] += 1
            
            # Index notes
            await update_progress("notes", 0, 100)
            notes_docs = await self._get_notes_documents()
            for i, doc in enumerate(notes_docs):
                await update_progress("notes", i, len(notes_docs))
                result = await rag_service.index_document(
                    doc_id=doc["id"],
                    content=doc["content"],
                    title=doc["title"],
                    metadata=doc["metadata"]
                )
                if result.success:
                    results["notes"] += 1
            
            total_indexed = sum(results.values())
            await status_elem.update(
                f"✅ Indexed {total_indexed} items (Media: {results['media']}, "
                f"Conversations: {results['conversations']}, Notes: {results['notes']})"
            )
            
        except Exception as e:
            logger.error(f"Indexing error: {e}", exc_info=True)
            await status_elem.update(f"❌ Indexing error: {str(e)}")
            self.app_instance.notify(f"Indexing error: {str(e)}", severity="error")
        
        finally:
            await asyncio.sleep(2)  # Show completion status briefly
            progress_bar.add_class("hidden")
            await status_elem.update("Ready to search")
    
    @on(Button.Pressed)
    async def handle_result_button(self, event: Button.Pressed) -> None:
        """Handle button presses for search results"""
        button_id = event.button.id
        if not button_id:
            return
        
        if button_id.startswith("expand-"):
            index = int(button_id.split("-")[1])
            await self._toggle_result_expansion(index)
        elif button_id.startswith("copy-"):
            index = int(button_id.split("-")[1])
            await self._copy_result(index)
        elif button_id.startswith("export-"):
            index = int(button_id.split("-")[1])
            await self._export_result(index)
        elif button_id.startswith("add-note-"):
            index = int(button_id.split("-")[1])
            await self._add_result_to_notes(index)
    
    async def _toggle_result_expansion(self, index: int) -> None:
        """Toggle expanded view of a result"""
        try:
            result_container = self.query_one(f"#result-{index}", SearchResult)
            expanded_content = result_container.query_one(f"#expanded-{index}")
            expand_btn = result_container.query_one(f"#expand-{index}", Button)
            
            # Toggle expansion
            if "hidden" in expanded_content.classes:
                expanded_content.remove_class("hidden")
                expand_btn.label = "🔼 Hide"
                result_container.expanded = True
                result_container.add_class("expanded")
            else:
                expanded_content.add_class("hidden")
                expand_btn.label = "🔽 View"
                result_container.expanded = False
                result_container.remove_class("expanded")
                
        except Exception as e:
            logger.error(f"Error toggling result expansion: {e}")
    
    async def _add_result_to_notes(self, index: int) -> None:
        """Add result to notes"""
        if index <= len(self.all_results):
            result = self.all_results[index - 1]
            # Implementation would add to notes
            self.app_instance.notify("Added to notes", severity="success")
    
    async def _update_analytics(self) -> None:
        """Update search analytics display"""
        analytics = self.query_one("#analytics-content", Markdown)
        
        # Calculate analytics
        if self.all_results:
            avg_score = sum(r.get('score', 0) for r in self.all_results) / len(self.all_results)
            source_dist = {}
            for r in self.all_results:
                source = r.get('source', 'unknown')
                source_dist[source] = source_dist.get(source, 0) + 1
        else:
            avg_score = 0
            source_dist = {}
        
        # Get search history analytics
        search_analytics = self.get_search_analytics(days_back=7)
        
        # Format analytics
        analytics_text = f"""# Search Analytics

## Current Search
- Total Results: {len(self.all_results)}
- Average Relevance Score: {avg_score:.3f}

### Source Distribution
"""
        
        for source, count in source_dist.items():
            percentage = (count / len(self.all_results)) * 100 if self.all_results else 0
            icon = SOURCE_ICONS.get(source, "📄")
            analytics_text += f"- {icon} {source.capitalize()}: {count} ({percentage:.1f}%)\n"
        
        analytics_text += f"""
## Search History (Last 7 Days)
- Total Searches: {search_analytics.get('total_searches', 0)}
- Average Results: {search_analytics.get('avg_results', 0):.1f}
- Success Rate: {search_analytics.get('success_rate', 0):.1f}%

### Popular Queries
"""
        
        for query_info in search_analytics.get('popular_queries', [])[:5]:
            analytics_text += f"- {query_info['query']} ({query_info['count']} times)\n"
        
        await analytics.update(analytics_text)
    
    @on(ListView.Selected, "#saved-searches-list")
    async def handle_saved_search_selection(self, event: ListView.Selected) -> None:
        """Handle selection of a saved search"""
        if event.item:
            index = event.list_view.index
            saved_names = list(self.saved_searches.keys())
            if 0 <= index < len(saved_names):
                saved_searches = self.query_one(SavedSearchesPanel)
                saved_searches.selected_search_name = saved_names[index]
                self.query_one("#load-saved-search").disabled = False
                self.query_one("#delete-saved-search").disabled = False
    
    @on(Button.Pressed, "#load-saved-search")
    async def handle_load_saved_search(self, event: Button.Pressed) -> None:
        """Load a saved search configuration"""
        saved_searches = self.query_one(SavedSearchesPanel)
        if saved_searches.selected_search_name and saved_searches.selected_search_name in saved_searches.saved_searches:
            config = saved_searches.saved_searches[saved_searches.selected_search_name]['config']
            
            # Apply the saved configuration
            self.search_input.value = config.get('query', '')
            self.query_one("#search-mode-select").value = config.get('mode', 'plain')
            
            sources = config.get('sources', {})
            self.query_one("#source-media").value = sources.get('media', True)
            self.query_one("#source-conversations").value = sources.get('conversations', True)
            self.query_one("#source-notes").value = sources.get('notes', True)
            
            self.query_one("#top-k-input").value = str(config.get('top_k', 10))
            self.query_one("#max-context-input").value = str(config.get('max_context', 10000))
            self.query_one("#enable-rerank").value = config.get('enable_rerank', False)
            
            self.app_instance.notify("📥 Loaded saved search configuration", severity="success")
            self.search_input.focus()
    
    @on(Button.Pressed, "#delete-saved-search")
    async def handle_delete_saved_search(self, event: Button.Pressed) -> None:
        """Delete a saved search"""
        saved_searches = self.query_one(SavedSearchesPanel)
        if saved_searches.selected_search_name and saved_searches.selected_search_name in saved_searches.saved_searches:
            del saved_searches.saved_searches[saved_searches.selected_search_name]
            saved_searches._persist_saved_searches()
            await saved_searches.refresh_list()
            self.app_instance.notify("🗑️ Deleted saved search", severity="success")
    
    # Action methods
    def action_focus_search(self) -> None:
        """Focus the search input (Ctrl+K)"""
        self.search_input.focus()
        self.search_input.cursor_position = len(self.search_input.value)
    
    def action_save_search(self) -> None:
        """Save current search (Ctrl+S)"""
        self.query_one("#save-search-btn").press()
    
    def action_refresh(self) -> None:
        """Refresh action (Ctrl+R)"""
        self.search_input.focus()
    
    def action_export(self) -> None:
        """Export results (Ctrl+E)"""
        if not self.all_results:
            self.app_instance.notify("No results to export", severity="warning")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_results_{timestamp}.json"
            
            export_data = {
                "query": self.search_input.value,
                "timestamp": timestamp,
                "config": self.current_search_config,
                "results": self.all_results
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.app_instance.notify(f"Results exported to {filename}", severity="success")
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            self.app_instance.notify(f"Export error: {str(e)}", severity="error")
    
    def action_index(self) -> None:
        """Trigger content indexing (Ctrl+I)"""
        self.query_one("#index-content-btn").press()
    
    def action_clear(self) -> None:
        """Clear search (Escape)"""
        # Only clear if we're not in a dropdown or dialog
        if self.search_input.has_focus:
            self.search_input.value = ""
            self.all_results = []
            self.current_results = []
            self.current_search_id = None
            self.current_page = 1
            
            results_container = self.query_one("#results-container")
            self.run_worker(results_container.remove_children())
            
            summary = self.query_one("#results-summary")
            self.run_worker(summary.update("💡 Enter a search query to discover relevant content"))
            
            stats = self.query_one("#search-stats")
            self.run_worker(stats.update(""))
            
            status = self.query_one("#search-status")
            self.run_worker(status.update("🟢 Ready to search"))
            
            self.query_one("#export-results-btn").disabled = True
            
            # Hide search history dropdown
            history_dropdown = self.query_one(SearchHistoryDropdown)
            history_dropdown.hide()
    
    async def _check_index_status(self) -> None:
        """Check the status of vector indices"""
        if not self.embeddings_available or not EMBEDDINGS_SERVICE_AVAILABLE:
            return
            
        try:
            embeddings_dir = Path.home() / ".local" / "share" / "tldw_cli" / "chromadb"
            embeddings_service = EmbeddingsService(embeddings_dir)
            
            collections = embeddings_service.list_collections()
            if collections:
                total_docs = 0
                for collection in collections:
                    info = embeddings_service.get_collection_info(collection)
                    if info:
                        total_docs += info.get('count', 0)
                
                if total_docs > 0:
                    status_elem = self.query_one("#search-status")
                    await status_elem.update(f"Ready to search ({total_docs} documents indexed)")
            
        except Exception as e:
            logger.debug(f"Could not check index status: {e}")
    
    def _load_recent_search_history(self, limit: int = 20):
        """Load recent search history from database."""
        try:
            history = self.search_history_db.get_search_history(limit=limit, days_back=7)
            self.search_history = [item['query'] for item in history if item['success']]
            logger.debug(f"Loaded {len(self.search_history)} recent search queries")
        except Exception as e:
            logger.error(f"Error loading search history: {e}")
            self.search_history = []
    
    def _record_search_to_history(
        self,
        query: str,
        search_type: str,
        results: List[Dict[str, Any]],
        execution_time_ms: int,
        search_params: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> int:
        """Record a search to the history database."""
        try:
            search_id = self.search_history_db.record_search(
                query=query,
                search_type=search_type,
                results=results,
                execution_time_ms=execution_time_ms,
                search_params=search_params,
                error_message=error_message
            )
            
            # Update in-memory history
            if query not in self.search_history:
                self.search_history.insert(0, query)
                self.search_history = self.search_history[:20]
            
            # Update history table
            self._update_history_table_async()
            
            return search_id
        except Exception as e:
            logger.error(f"Error recording search to history: {e}")
            return -1
    
    def _update_history_table_async(self) -> None:
        """Update the history table asynchronously"""
        self.run_worker(self._update_history_table())
    
    async def _update_history_table(self) -> None:
        """Update the history table with recent searches"""
        history_table = self.query_one("#search-history-table", DataTable)
        history_table.clear()
        
        # Get recent history
        history = self.search_history_db.get_search_history(limit=50, days_back=30)
        
        for item in history:
            time_str = datetime.fromisoformat(item['timestamp']).strftime("%H:%M:%S")
            query_preview = item['query'][:50] + "..." if len(item['query']) > 50 else item['query']
            history_table.add_row(
                time_str,
                query_preview,
                item['search_type'],
                str(item['result_count']),
                f"{item['execution_time_ms']/1000:.2f}s"
            )
    
    async def _copy_result(self, index: int) -> None:
        """Copy result content to clipboard"""
        if index <= len(self.all_results):
            result = self.all_results[index - 1]
            content = f"[{result['source'].upper()}] {result['title']}\n\n{result['content']}"
            
            try:
                import pyperclip
                pyperclip.copy(content)
                self.app_instance.notify("Result copied to clipboard", severity="success")
            except ImportError:
                self.app_instance.notify("pyperclip not available - cannot copy to clipboard", severity="warning")
            except Exception as e:
                self.app_instance.notify(f"Copy failed: {str(e)}", severity="error")
    
    async def _export_result(self, index: int) -> None:
        """Export single result to file"""
        if index <= len(self.all_results):
            result = self.all_results[index - 1]
            
            # Create export content
            export_content = f"""# Search Result Export
Source: {result['source'].upper()}
Title: {result['title']}
Score: {result.get('score', 0):.3f}

## Content
{result['content']}

## Metadata
"""
            for key, value in result.get('metadata', {}).items():
                export_content += f"- {key}: {value}\n"
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_result_{timestamp}_{index}.md"
            filepath = Path.home() / "Downloads" / filename
            
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(export_content, encoding='utf-8')
                self.app_instance.notify(f"Result exported to {filepath}", severity="success")
            except Exception as e:
                self.app_instance.notify(f"Export failed: {str(e)}", severity="error")
    
    async def _get_media_documents(self) -> List[Dict[str, Any]]:
        """Get documents from media database"""
        documents = []
        try:
            if self.app_instance.media_db:
                # Get all media entries
                media_items = self.app_instance.media_db.get_all_media()
                for item in media_items:
                    doc = {
                        "id": f"media_{item['id']}",
                        "title": item.get('title', 'Untitled Media'),
                        "content": item.get('transcription', '') or item.get('content', ''),
                        "metadata": {
                            "source": "media",
                            "media_type": item.get('type', 'unknown'),
                            "url": item.get('url', ''),
                            "author": item.get('author', ''),
                            "ingested_at": item.get('ingested_at', '')
                        }
                    }
                    if doc["content"]:  # Only add if there's content
                        documents.append(doc)
        except Exception as e:
            logger.error(f"Error getting media documents: {e}")
        return documents
    
    async def _get_conversation_documents(self) -> List[Dict[str, Any]]:
        """Get documents from conversations database"""
        documents = []
        try:
            if self.app_instance.chachanotes_db:
                # Get all conversations
                conversations = self.app_instance.chachanotes_db.get_all_conversations()
                for conv in conversations:
                    # Get messages for this conversation
                    messages = self.app_instance.chachanotes_db.get_messages_for_conversation(conv['id'])
                    if messages:
                        # Combine messages into conversation content
                        content = "\n".join([
                            f"{msg['role']}: {msg['content']}" 
                            for msg in messages
                        ])
                        doc = {
                            "id": f"conv_{conv['id']}",
                            "title": conv.get('name', f"Conversation {conv['id']}"),
                            "content": content,
                            "metadata": {
                                "source": "conversations",
                                "conversation_id": conv['id'],
                                "character_id": conv.get('character_id'),
                                "created_at": conv.get('created_at', ''),
                                "message_count": len(messages)
                            }
                        }
                        documents.append(doc)
        except Exception as e:
            logger.error(f"Error getting conversation documents: {e}")
        return documents
    
    async def _get_notes_documents(self) -> List[Dict[str, Any]]:
        """Get documents from notes"""
        documents = []
        try:
            if self.app_instance.chachanotes_db:
                # Get all notes
                notes = self.app_instance.chachanotes_db.get_all_notes()
                for note in notes:
                    doc = {
                        "id": f"note_{note['id']}",
                        "title": note.get('title', 'Untitled Note'),
                        "content": note.get('content', ''),
                        "metadata": {
                            "source": "notes",
                            "note_id": note['id'],
                            "tags": note.get('tags', ''),
                            "created_at": note.get('created_at', ''),
                            "updated_at": note.get('updated_at', '')
                        }
                    }
                    if doc["content"]:  # Only add if there's content
                        documents.append(doc)
        except Exception as e:
            logger.error(f"Error getting notes documents: {e}")
        return documents

    def get_search_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get search analytics from the history database."""
        try:
            return self.search_history_db.get_search_analytics(days_back=days_back)
        except Exception as e:
            logger.error(f"Error getting search analytics: {e}")
            return {}
