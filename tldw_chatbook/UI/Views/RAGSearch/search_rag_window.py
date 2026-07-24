"""
Main RAG Search Window Component

The primary window for RAG search functionality
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple
import asyncio
from datetime import datetime
from html import escape as html_escape
from inspect import isawaitable
from pathlib import Path
import json

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll, Grid
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, ListView, ListItem,
    DataTable, Label, TabbedContent, TabPane,
    LoadingIndicator, ProgressBar, Collapsible
)
from textual.binding import Binding
from textual.reactive import reactive
from textual.css.query import NoMatches
from rich.markup import escape
from loguru import logger

# Local imports
from ...destination_recovery import optional_dependency_recovery_state
from .search_history_dropdown import SearchHistoryDropdown
from .search_result import SearchResult
from .saved_searches_panel import SavedSearchesPanel
from .search_handoff import build_search_chat_handoff_payload
from .constants import (
    DEFAULT_TOP_K, DEFAULT_TEMPERATURE, DEFAULT_PARENT_SIZE,
    SEARCH_MODES, PARENT_STRATEGIES
)

from ....Chat.chat_handoff_messages import (
    USE_IN_CHAT_UNAVAILABLE_RECOVERY,
    build_handoff_policy_blocking_message,
)
from ....Chat.chat_handoff_models import ChatHandoffPayload
from ....Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ....RAG_Search.ingestion_indexing import (
    ITEM_TYPE_CONVERSATION,
    ITEM_TYPE_MEDIA,
    ITEM_TYPE_NOTE,
    backfill_semantic_index,
    get_shared_rag_service,
    peek_shared_rag_service,
    semantic_indexing_available,
)
from ....RAG_Search.semantic_availability import (
    SEMANTIC_DIAGNOSTICS_KEY,
    SEMANTIC_EMPTY_INDEX_MESSAGE,
    SEMANTIC_REASON_INIT_FAILED,
    SEMANTIC_STATUS_EMPTY_INDEX,
    SEMANTIC_STATUS_UNAVAILABLE,
    SEMANTIC_UNAVAILABLE_MESSAGES,
    trustworthy_collection_count,
)
from ....DB.search_history_db import SearchHistoryDB
from ....Utils.input_validation import sanitize_string
from ....Utils.paths import get_user_data_dir

# Conditionally import web search functionality
WEB_SEARCH_AVAILABLE = DEPENDENCIES_AVAILABLE.get('websearch', False)
USE_IN_CONSOLE_UNAVAILABLE_RECOVERY = (
    "Use in Console is unavailable because the Console live-work surface is not mounted. "
    "Open Console from the navigation, then try again."
)

#: Map of the #index-source-select options onto the backfill item-type
#: contract from RAG_Search.ingestion_indexing (task-251).
INDEX_SOURCE_ITEM_TYPES: Dict[str, Tuple[str, ...]] = {
    "media": (ITEM_TYPE_MEDIA,),
    "conversations": (ITEM_TYPE_CONVERSATION,),
    "notes": (ITEM_TYPE_NOTE,),
    "all": (ITEM_TYPE_MEDIA, ITEM_TYPE_NOTE, ITEM_TYPE_CONVERSATION),
}

INDEXING_ALREADY_RUNNING_MESSAGE = (
    "Semantic indexing is already running. Wait for the current run to finish."
)
INDEXING_UNAVAILABLE_MESSAGE = (
    "semantic indexing is unavailable. Install embeddings support "
    "(pip install 'tldw_chatbook[embeddings_rag]') and ensure "
    "[AppRAGSearchConfig.rag.indexing].enabled is not false."
)
INDEXING_NO_DATABASE_MESSAGE = (
    "Cannot index: no source database is available in this session for the "
    "selected content type."
)


def _sanitize_console_text(value: Any, *, max_length: int = 1000) -> str:
    """Return text safe for Rich/Textual status-row rendering."""
    cleaned = sanitize_string(str(value or ""), max_length=max_length)
    return escape(html_escape(cleaned, quote=False))


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
except ImportError as e:
    logger.warning(f"RAG event handlers not available: {e}")
    RAG_EVENTS_AVAILABLE = False
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
from .search_event_handlers import SearchEventHandlersMixin  # noqa: E402


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
        # Semantic-leg availability diagnostics from the last pipeline run
        # (task-250); read by _semantic_leg_notice to report honest states.
        self.last_search_diagnostics: Dict[str, Any] = {}
        # Single-flight guard for the Maintenance-tab bulk index run (task-251).
        self._indexing_in_flight = False
        
        # Performance tracking
        self.last_search_time = 0.0
        self.search_metrics: Dict[str, Any] = {
            "total_searches": 0,
            "avg_search_time": 0.0,
            "fastest_search": float('inf'),
            "slowest_search": 0.0
        }

    def _search_empty_state_text(self, *, no_results: bool = False) -> str:
        if no_results:
            return (
                "No results found.\n"
                "Try Plain Search, switch Collection to All Collections, or use Ingest to add/index content.\n"
                "When results appear, Use in Chat turns a selected result into Chat context."
            )
        return (
            "Start with a query.\n"
            "Plain Search scans indexed text quickly; contextual and hybrid RAG use collections for deeper retrieval.\n"
            "Use All Collections to search broadly, or choose a collection to narrow scope.\n"
            "Each result can be sent to Chat with Use in Chat, turning search evidence into Chat context."
        )

    async def _mount_search_empty_state(self, *, no_results: bool = False) -> None:
        results_list = self.query_one("#results-list-enhanced")
        await results_list.mount(
            Static(
                self._search_empty_state_text(no_results=no_results),
                id="search-empty-state",
                classes="search-empty-state",
            )
        )
        
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

                            # Keep the primary action in the first viewport; advanced
                            # controls can extend below it without hiding Search.
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
                                    # Refresh was removed (task-251): re-running
                                    # the query is exactly the Search button.
                                    yield Button("📤 Export", id="export-results", classes="results-action-button")
                            
                            # Results list
                            with VerticalScroll(id="results-list-enhanced", classes="results-list-enhanced"):
                                yield Static(
                                    self._search_empty_state_text(),
                                    id="search-empty-state",
                                    classes="search-empty-state",
                                )
                            
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
                            
                            # Collection management (display + refresh only:
                            # the simplified RAG service manages one
                            # collection per profile, so create/delete had no
                            # real backing and were removed, task-251)
                            with Container(classes="collection-management"):
                                yield Label("Available Collections:", classes="maintenance-label")
                                yield ListView(id="collections-list", classes="collections-list")

                                with Horizontal(classes="collection-actions"):
                                    yield Button("🔄 Refresh", id="refresh-collections", classes="maintenance-button")

                            # Indexing controls (wired to the real bulk
                            # backfill path from task-247, task-251)
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

                                # Indexing status (no progress bar: the
                                # backfill total is unknown up front, so a
                                # percentage would be fake -- real per-batch
                                # counts go into the status text instead)
                                with Container(id="indexing-status", classes="indexing-status hidden"):
                                    yield LoadingIndicator()
                                    yield Static("", id="indexing-status-text")

                            # Index statistics
                            with Container(classes="index-statistics"):
                                yield Static("📊 Index Statistics", classes="statistics-title")
                                yield DataTable(id="index-stats-table", classes="index-stats-table")
        

    def _missing_embeddings_recovery_state(self):
        from ....Utils.optional_deps import get_optional_feature_info

        feature = get_optional_feature_info("embeddings_rag")
        return optional_dependency_recovery_state(
            unavailable_what=feature.unavailable_what,
            missing_dependencies=(feature.extra,),
            install_targets=(
                feature.source_install_command,
                feature.package_install_command,
            ),
            stable_selector="search-rag-dependency-missing",
            recovery_action=feature.recovery_action,
            authority_owner=feature.owner,
        )

    async def on_mount(self) -> None:
        """Called when the widget is mounted"""
        parent_on_mount = getattr(super(), "on_mount", None)
        if callable(parent_on_mount):
            result = parent_on_mount()
            if isawaitable(result):
                await result

        # Check if embeddings/RAG dependencies are available
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            from ....Utils.widget_helpers import alert_embeddings_not_available
            recovery_state = self._missing_embeddings_recovery_state()
            # Show alert after a short delay to ensure UI is ready
            self.set_timer(0.1, lambda: alert_embeddings_not_available(self))
            try:
                results_list = self.query_one("#results-list-enhanced")
                await results_list.remove_children()
                await results_list.mount(
                    Static(
                        recovery_state.visible_copy,
                        id=recovery_state.stable_selector,
                        classes="search-empty-state search-recovery-state",
                    )
                )
                search_input = self.query_one("#search-query-input", Input)
                search_input.disabled = True
                search_input.placeholder = "Embeddings not available - install dependencies"
                search_input.tooltip = recovery_state.disabled_tooltip
                search_button = self.query_one("#search-button", Button)
                search_button.disabled = True
                search_button.tooltip = recovery_state.disabled_tooltip
            except NoMatches:
                pass
            # The Maintenance-tab indexing controls need the same runtime the
            # searches do; disable them with the same recovery copy (task-251).
            try:
                start_indexing = self.query_one("#start-indexing", Button)
                start_indexing.disabled = True
                start_indexing.tooltip = recovery_state.disabled_tooltip
                index_source = self.query_one("#index-source-select", Select)
                index_source.disabled = True
                index_source.tooltip = recovery_state.disabled_tooltip
            except NoMatches:
                pass

        # Setup UI components after all widgets are created
        self._setup_history_table()
        self._setup_analytics()
        self._setup_collections_list()
        self._setup_index_stats()

    def _authoritative_runtime_backend(self) -> str:
        get_source = getattr(self.app_instance, "get_authoritative_runtime_source", None)
        backend = get_source() if callable(get_source) else "local"
        backend = str(backend or "local").strip().lower()
        return backend if backend in {"local", "server"} else "local"

    def _build_search_chat_handoff_payload(self, result: Dict[str, Any]):
        return build_search_chat_handoff_payload(
            dict(result),
            runtime_backend=self._authoritative_runtime_backend(),
        )

    def _build_search_console_launch(self, result: Dict[str, Any]) -> dict[str, Any]:
        handoff_payload = self._build_search_chat_handoff_payload(result)
        metadata = dict(handoff_payload.metadata or {})
        source_id = str(handoff_payload.source_id or "").strip()
        content_ref = str(handoff_payload.content_ref or "").strip()
        target_id = content_ref or (
            f"{handoff_payload.source}:{source_id}" if source_id else handoff_payload.source
        )
        is_web = handoff_payload.source == "search-web"
        launch_payload = {
            "target_id": _sanitize_console_text(target_id),
            "source_id": _sanitize_console_text(source_id),
            "content_ref": _sanitize_console_text(content_ref),
            "runtime_backend": handoff_payload.runtime_backend or self._authoritative_runtime_backend(),
            "source": _sanitize_console_text(
                str(result.get("source") or "unknown").strip() or "unknown",
                max_length=120,
            ),
            "score": metadata.get("score"),
            "display_summary": _sanitize_console_text(handoff_payload.display_summary),
            "suggested_prompt": _sanitize_console_text(handoff_payload.suggested_prompt),
        }
        return {
            "source": "Web Search" if is_web else "RAG",
            "title": _sanitize_console_text(handoff_payload.title, max_length=500),
            "payload": {
                key: value
                for key, value in launch_payload.items()
                if value is not None and str(value).strip()
            },
            "status": "ready",
            "recovery": (
                "Use this web result as Console context, or return to Search/RAG to adjust the query."
                if is_web
                else "Use this retrieved RAG result as Console context, or return to Search/RAG to adjust the query."
            ),
            "action_label": "Ask from web result" if is_web else "Ask from RAG result",
        }

    @staticmethod
    def _rag_handoff_runtime_action_id(payload: ChatHandoffPayload) -> str | None:
        if payload.source != "search-rag":
            return None
        backend = str(payload.runtime_backend or "").strip().lower()
        if backend != "server":
            return None
        return "rag.media_embeddings.search.server"

    def _rag_handoff_policy_blocking_message(self, payload: ChatHandoffPayload) -> str:
        action_id = self._rag_handoff_runtime_action_id(payload)
        return build_handoff_policy_blocking_message(
            self.app_instance,
            action_id=action_id,
            fallback_message="This RAG search action is blocked by runtime policy.",
        )

    @on(SearchResult.UseInChatRequested)
    def handle_search_result_use_in_chat(self, event: SearchResult.UseInChatRequested) -> None:
        event.stop()
        payload = self._build_search_chat_handoff_payload(event.result)
        policy_message = self._rag_handoff_policy_blocking_message(payload)
        if policy_message:
            self.app_instance.notify(policy_message, severity="warning")
            return
        open_chat = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat):
            self.app_instance.notify(USE_IN_CHAT_UNAVAILABLE_RECOVERY, severity="warning")
            return
        open_chat(payload)

    @on(SearchResult.UseInConsoleRequested)
    def handle_search_result_use_in_console(self, event: SearchResult.UseInConsoleRequested) -> None:
        """Stage a selected Search/RAG result as Console live-work context.

        Args:
            event: Result-card event carrying the selected search result.
        """
        event.stop()
        payload = self._build_search_chat_handoff_payload(event.result)
        policy_message = self._rag_handoff_policy_blocking_message(payload)
        if policy_message:
            self.app_instance.notify(policy_message, severity="warning")
            return
        open_console = getattr(self.app_instance, "open_console_for_live_work", None)
        if not callable(open_console):
            self.app_instance.notify(USE_IN_CONSOLE_UNAVAILABLE_RECOVERY, severity="warning")
            return
        open_console(**self._build_search_console_launch(event.result))

    @on(SavedSearchesPanel.LoadRequested)
    def handle_saved_search_load_requested(self, event: SavedSearchesPanel.LoadRequested) -> None:
        """Apply a selected saved search to the active Search/RAG controls."""
        event.stop()
        config = dict(event.config)

        self.query_one("#search-query-input", Input).value = str(config.get("query") or "")
        if "mode" in config:
            self.query_one("#search-mode-select", Select).value = config["mode"]
        if "collection" in config:
            self.query_one("#collection-select", Select).value = config["collection"]
        if "top_k" in config:
            self.query_one("#top-k-input", Input).value = str(config["top_k"] if config["top_k"] is not None else "")
        if "temperature" in config:
            self.query_one("#temperature-input", Input).value = str(config["temperature"] if config["temperature"] is not None else "")

        filters = config.get("filters") or config.get("sources") or {}
        if filters:
            self.query_one("#filter-media", Checkbox).value = bool(filters.get("media", True))
            self.query_one("#filter-conversations", Checkbox).value = bool(filters.get("conversations", True))
            self.query_one("#filter-notes", Checkbox).value = bool(filters.get("notes", True))

        if "enable_parent_docs" in config:
            parent_docs_enabled = bool(config["enable_parent_docs"])
            self.enable_parent_docs = parent_docs_enabled
            self.query_one("#parent-docs-checkbox", Checkbox).value = parent_docs_enabled
        if "parent_strategy" in config:
            self.query_one("#parent-strategy-select", Select).value = config["parent_strategy"]
        if "parent_size" in config:
            self.query_one("#parent-size-input", Input).value = str(config["parent_size"] if config["parent_size"] is not None else "")

        self.last_search_config = config
        self.query_one("#search-query-input", Input).focus()
        self.app_instance.notify(
            f"Loaded saved search '{event.name}' into Search.",
            severity="information",
        )
    
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
            self.query_one("#collections-list", ListView)
            self._refresh_collections_list()
    
    def _setup_index_stats(self) -> None:
        """Setup index statistics table"""
        if RAG_SERVICES_AVAILABLE:
            table = self.query_one("#index-stats-table", DataTable)
            table.add_columns("Collection", "Chunks", "Status")
            self._refresh_index_stats()

    def _load_available_collections(self) -> list[str]:
        """Load available collection names without touching Textual widgets."""
        return list(get_available_profiles())

    async def _apply_available_collections(self, collections: list[str]) -> None:
        """Apply loaded collection names to mounted Textual widgets."""
        self.available_collections = list(collections)

        try:
            collections_list = self.query_one("#collections-list", ListView)
            await collections_list.clear()

            for collection in self.available_collections:
                await collections_list.append(ListItem(Static(collection)))

            collection_select = self.query_one("#collection-select", Select)
            collection_select.set_options(
                [("All Collections", "all")] + [(c, c) for c in self.available_collections]
            )
        except NoMatches:
            # The window (or its Maintenance tab) went away while the
            # collections loader thread was in flight; nothing to update.
            return
    
    @work(thread=True)
    def _refresh_collections_list(self) -> None:
        """Refresh the list of available collections"""
        try:
            collections = self._load_available_collections()
            # Dedicated group: an exclusive worker in the DEFAULT group would
            # cancel unrelated default-group workers (e.g. an in-flight
            # search) whenever a collections refresh lands (task-251).
            self.app.call_from_thread(
                lambda: self.run_worker(
                    self._apply_available_collections(collections),
                    exclusive=True,
                    group="rag-collections-apply",
                )
            )
        except Exception as e:
            logger.error(f"Error refreshing collections: {e}")
    
    def _load_index_stats(self) -> Optional[Dict[str, Any]]:
        """Read collection stats from an ALREADY-initialized RAG runtime.

        Never constructs the runtime: rendering statistics must not load an
        embedding model as a side effect. Safe to run off the UI thread
        (ChromaDB-backed stats can touch disk).

        Returns:
            The raw stats dict, an ``{"error": ...}`` dict when the read
            failed, or None when no runtime exists yet.
        """
        service = getattr(self.app_instance, "_rag_service", None)
        if service is None:
            service = peek_shared_rag_service()
        get_stats = getattr(getattr(service, "vector_store", None), "get_collection_stats", None)
        if not callable(get_stats):
            return None
        try:
            stats = get_stats()
        except Exception as e:
            logger.error(
                f"Vector-store stats read failed "
                f"(operation=vector_store.get_collection_stats, "
                f"service={type(service).__name__}): {e}"
            )
            return {"error": str(e)}
        if not isinstance(stats, dict):
            return {"error": "vector store returned malformed statistics"}
        return stats

    @work(thread=True, exclusive=True, group="rag-index-stats", exit_on_error=False)
    def _refresh_index_stats(self) -> None:
        """Refresh index statistics off the UI thread, then apply on it (task-251)."""
        try:
            stats = self._load_index_stats()
            self.app.call_from_thread(self._apply_index_stats, stats)
        except Exception as e:
            logger.error(
                f"Error refreshing index stats "
                f"(operation=refresh_index_stats, worker_group=rag-index-stats): {e}"
            )

    def _apply_index_stats(self, stats: Optional[Dict[str, Any]]) -> None:
        """Render index statistics honestly: real counts or the actual state.

        Args:
            stats: Result of :meth:`_load_index_stats` (raw stats dict,
                ``{"error": ...}``, or None when no runtime exists yet).
        """
        try:
            table = self.query_one("#index-stats-table", DataTable)
        except NoMatches:
            return
        table.clear()
        if stats is None:
            table.add_row(
                "semantic index",
                "—",
                "not initialized — run Start Indexing or a semantic search",
            )
            return
        name = str(stats.get("name") or "semantic index")
        error = stats.get("error")
        if error:
            table.add_row(name, "—", f"statistics unavailable: {error}")
            return
        count = trustworthy_collection_count(stats)
        if count is None:
            table.add_row(name, "—", "statistics unavailable: untrustworthy chunk count")
            return
        table.add_row(
            name,
            str(count),
            "empty — no content indexed yet" if count == 0 else "ready",
        )

    # Bulk indexing (task-251): #start-indexing runs the real backfill path
    def _selected_index_item_types(self) -> Tuple[str, ...]:
        """Map the #index-source-select value onto backfill item types."""
        try:
            raw = self.query_one("#index-source-select", Select).value
        except NoMatches:
            raw = "all"
        return INDEX_SOURCE_ITEM_TYPES.get(str(raw), INDEX_SOURCE_ITEM_TYPES["all"])

    def _set_indexing_ui_running(self, running: bool, status: str = "") -> None:
        """Toggle the Start Indexing button and status row for an active run."""
        try:
            self.query_one("#start-indexing", Button).disabled = running
            status_container = self.query_one("#indexing-status")
            if running:
                status_container.remove_class("hidden")
            else:
                status_container.add_class("hidden")
            self.query_one("#indexing-status-text", Static).update(status)
        except NoMatches:
            pass

    def _start_indexing_run(self) -> None:
        """Validate preconditions and launch the bulk semantic-index worker.

        Honest by construction: refuses (with WHY) when embeddings support is
        missing, a run is already in flight, or no source database backs the
        selected content type -- it never pretends to index.
        """
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            self.app_instance.notify(
                f"Indexing did not run: {INDEXING_UNAVAILABLE_MESSAGE}",
                severity="warning",
            )
            return
        if self._indexing_in_flight:
            self.app_instance.notify(INDEXING_ALREADY_RUNNING_MESSAGE, severity="warning")
            return

        item_types = self._selected_index_item_types()
        media_db = (
            getattr(self.app_instance, "media_db", None)
            if ITEM_TYPE_MEDIA in item_types else None
        )
        chachanotes_db = (
            getattr(self.app_instance, "chachanotes_db", None)
            if (ITEM_TYPE_NOTE in item_types or ITEM_TYPE_CONVERSATION in item_types)
            else None
        )
        if media_db is None and chachanotes_db is None:
            self.app_instance.notify(INDEXING_NO_DATABASE_MESSAGE, severity="error")
            return

        self._indexing_in_flight = True
        self._set_indexing_ui_running(True, "Starting semantic indexing…")
        self._run_index_backfill(item_types, media_db, chachanotes_db)

    @work(thread=True, exclusive=True, group="rag-index-backfill", exit_on_error=False)
    def _run_index_backfill(
        self,
        item_types: Tuple[str, ...],
        media_db: Optional[Any],
        chachanotes_db: Optional[Any],
    ) -> None:
        """Thread worker: run the real bulk backfill; exceptions never escape.

        Runs ``backfill_semantic_index`` inside its own event loop on this
        worker thread (exactly like the CLI entry point), so the source-DB
        pagination and embedding work never touch the UI event loop. Both DB
        classes use thread-local sqlite connections, so cross-thread use is
        safe. Progress and completion are marshalled back with
        ``call_from_thread``.

        The shared RAG service is pre-resolved here, OUTSIDE the transient
        ``asyncio.run`` loop, so first-time service construction can never
        happen inside a loop that closes when this run finishes (PR #700
        review). Verified no current component binds a loop at construction
        (plain ThreadPoolExecutor, threading.Lock circuit breaker, cache's
        asyncio.Lock binds on first acquisition, local embeddings with no
        HTTP client) -- pre-resolving keeps that true by construction.
        """
        def _progress(update: Dict[str, Any]) -> None:
            try:
                self.app.call_from_thread(self._update_indexing_progress, dict(update))
            except Exception as e:
                logger.debug(
                    f"Indexing progress update failed "
                    f"(operation=update_indexing_progress, "
                    f"item_type={update.get('item_type', '?')}): {e}"
                )

        rag_service = None
        indexing_available = False
        try:
            indexing_available = semantic_indexing_available()
            if indexing_available:
                rag_service = get_shared_rag_service()
        except Exception as e:
            logger.error(
                f"Shared RAG service pre-resolution failed "
                f"(operation=get_shared_rag_service, item_types={item_types}): {e}"
            )
        if indexing_available and rag_service is None:
            # Do not enter the transient loop at all: backfill would retry
            # service construction inside it, which is exactly what
            # pre-resolution exists to prevent. Same outcome/copy as
            # backfill's own service-unavailable path.
            unavailable_summary = {
                "status": "unavailable",
                "indexed": 0,
                "skipped": 0,
                "failed": 0,
                "errors": ["RAG service could not be created"],
            }
            try:
                self.app.call_from_thread(self._finish_indexing_run, unavailable_summary)
            except Exception as e:
                logger.error(
                    f"Could not deliver indexing completion to the UI "
                    f"(operation=finish_indexing_run, item_types={item_types}, "
                    f"summary_status=unavailable): {e}"
                )
                self._indexing_in_flight = False
            return

        try:
            summary = asyncio.run(
                backfill_semantic_index(
                    media_db=media_db,
                    chachanotes_db=chachanotes_db,
                    rag_service=rag_service,
                    item_types=item_types,
                    progress_callback=_progress,
                )
            )
        except Exception as e:
            logger.opt(exception=True).error(
                f"Semantic index backfill crashed "
                f"(operation=backfill_semantic_index, item_types={item_types}): {e}"
            )
            summary = {
                "status": "error",
                "indexed": 0,
                "skipped": 0,
                "failed": 0,
                "errors": [str(e)],
            }
        try:
            self.app.call_from_thread(self._finish_indexing_run, summary)
        except Exception as e:
            logger.error(
                f"Could not deliver indexing completion to the UI "
                f"(operation=finish_indexing_run, item_types={item_types}, "
                f"summary_status={summary.get('status')}): {e}"
            )
            self._indexing_in_flight = False

    def _update_indexing_progress(self, update: Dict[str, Any]) -> None:
        """Show real per-batch backfill counts in the indexing status row."""
        try:
            status_text = self.query_one("#indexing-status-text", Static)
        except NoMatches:
            return
        status_text.update(
            f"Indexing {update.get('item_type', '?')}: "
            f"{update.get('indexed', 0)} indexed, "
            f"{update.get('skipped', 0)} up-to-date, "
            f"{update.get('failed', 0)} failed"
        )

    def _finish_indexing_run(self, summary: Dict[str, Any]) -> None:
        """Report the backfill outcome honestly and refresh the real stats."""
        self._indexing_in_flight = False
        self._set_indexing_ui_running(False)

        status = summary.get("status")
        errors = summary.get("errors") or []
        last_error = str(errors[-1]) if errors else None
        if status == "unavailable":
            self.app_instance.notify(
                f"Indexing did not run: {last_error or INDEXING_UNAVAILABLE_MESSAGE}",
                severity="warning",
            )
        elif status == "error" or summary.get("failed") or errors:
            detail = f" Last error: {last_error}" if last_error else ""
            self.app_instance.notify(
                f"Indexing finished with problems: {summary.get('indexed', 0)} indexed, "
                f"{summary.get('failed', 0)} failed.{detail}",
                severity="error" if status == "error" else "warning",
            )
        else:
            self.app_instance.notify(
                f"Indexing complete: {summary.get('indexed', 0)} indexed, "
                f"{summary.get('skipped', 0)} already up-to-date.",
                severity="information",
            )
        self._refresh_index_stats()

    # Search implementation methods
    @staticmethod
    def _sources_from_config(config: Dict[str, Any]) -> Dict[str, bool]:
        """Map the filter checkboxes onto the pipeline sources contract."""
        filters = config.get('filters') or {}
        return {
            'media': bool(filters.get('media', True)),
            'conversations': bool(filters.get('conversations', True)),
            'notes': bool(filters.get('notes', True)),
        }

    async def _perform_plain_search(self, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform plain RAG search"""
        if not RAG_EVENTS_AVAILABLE:
            raise ImportError("RAG search not available")

        self.last_search_diagnostics = {}
        results, _context = await perform_plain_rag_search(
            self.app_instance,
            query,
            self._sources_from_config(config),
            top_k=config.get('top_k', DEFAULT_TOP_K),
        )

        return self._format_search_results(results, "plain")

    async def _perform_contextual_search(self, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform contextual (semantic) RAG search"""
        if not RAG_EVENTS_AVAILABLE:
            raise ImportError("RAG pipeline not available")

        diagnostics: Dict[str, Any] = {}
        results, _context = await perform_full_rag_pipeline(
            self.app_instance,
            query,
            self._sources_from_config(config),
            top_k=config.get('top_k', DEFAULT_TOP_K),
            diagnostics=diagnostics,
        )
        self.last_search_diagnostics = diagnostics

        return self._format_search_results(results, "contextual")

    async def _perform_hybrid_search(self, query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform hybrid RAG search"""
        if not RAG_EVENTS_AVAILABLE:
            raise ImportError("Hybrid search not available")

        diagnostics: Dict[str, Any] = {}
        results, _context = await perform_hybrid_rag_search(
            self.app_instance,
            query,
            self._sources_from_config(config),
            top_k=config.get('top_k', DEFAULT_TOP_K),
            diagnostics=diagnostics,
        )
        self.last_search_diagnostics = diagnostics

        return self._format_search_results(results, "hybrid")

    def _semantic_leg_notice(self, search_mode: str) -> Optional[Tuple[str, str]]:
        """(short marker, full message) when the semantic leg didn't contribute.

        Reads the diagnostics recorded by the last pipeline run (task-250).
        Returns None when the semantic leg ran fine or the mode has no
        semantic leg (plain).

        Args:
            search_mode: The mode of the search the diagnostics belong to.

        Returns:
            Tuple of a short results-header marker and the full user-facing
            reason message, or None when there is nothing to report.
        """
        semantic_state = (getattr(self, 'last_search_diagnostics', None) or {}).get(
            SEMANTIC_DIAGNOSTICS_KEY
        ) or {}
        status = semantic_state.get('status')
        if status == SEMANTIC_STATUS_UNAVAILABLE:
            message = semantic_state.get('message') or SEMANTIC_UNAVAILABLE_MESSAGES[
                SEMANTIC_REASON_INIT_FAILED
            ]
            if search_mode == "hybrid":
                return (
                    "keyword-only (semantic unavailable)",
                    f"Hybrid results are keyword-only (FTS): {message}",
                )
            return ("semantic unavailable", message)
        if status == SEMANTIC_STATUS_EMPTY_INDEX:
            message = semantic_state.get('message') or SEMANTIC_EMPTY_INDEX_MESSAGE
            if search_mode == "hybrid":
                return (
                    "keyword-only (semantic index empty)",
                    f"Hybrid results are keyword-only (FTS): {message}",
                )
            return ("semantic index empty", message)
        return None
    
    async def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search"""
        if not WEB_SEARCH_AVAILABLE:
            return []
            
        try:
            web_results = await asyncio.to_thread(search_web_bing, query)
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
        if not page_results:
            await self._mount_search_empty_state(no_results=True)
            self._update_pagination()
            return
        
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
    
    def _selected_history_days_back(self) -> Optional[int]:
        """Read the #history-range-select value as days-back (None = all time)."""
        try:
            raw = self.query_one("#history-range-select", Select).value
        except NoMatches:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def _refresh_history_view(self) -> None:
        """Reload the history table for the selected time range (task-251)."""
        self._load_recent_search_history(days_back=self._selected_history_days_back())

    def _load_recent_search_history(self, limit: int = 20, days_back: Optional[int] = None):
        """Load recent search history into the table"""
        table = self.query_one("#search-history-table", DataTable)
        table.clear()

        history = self.search_history_db.get_search_history(limit=limit, days_back=days_back)
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
