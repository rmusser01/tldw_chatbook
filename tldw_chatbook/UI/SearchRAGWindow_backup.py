# SearchRAGWindow.py
# Description: Dedicated RAG search interface for tldw_chatbook
#
# Imports
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple
import asyncio
from datetime import datetime
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll, Grid
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, 
    DataTable, Markdown, Label, TabbedContent, TabPane,
    LoadingIndicator, ProgressBar
)
from textual.binding import Binding
from rich.markup import escape
from loguru import logger

# Local Imports
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Conditional imports for RAG functionality
try:
    from ..Event_Handlers.Chat_Events.chat_rag_events import (
        perform_plain_rag_search, perform_full_rag_pipeline, perform_hybrid_rag_search
    )
    RAG_EVENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG event handlers not available: {e}")
    RAG_EVENTS_AVAILABLE = False
    
    # Create placeholder functions
    async def perform_plain_rag_search(*args, **kwargs):
        raise ImportError("RAG search not available. Please install RAG dependencies: pip install tldw_chatbook[embeddings_rag]")
    
    async def perform_full_rag_pipeline(*args, **kwargs):
        raise ImportError("RAG pipeline not available. Please install RAG dependencies: pip install tldw_chatbook[embeddings_rag]")
    
    async def perform_hybrid_rag_search(*args, **kwargs):
        raise ImportError("Hybrid RAG search not available. Please install RAG dependencies: pip install tldw_chatbook[embeddings_rag]")

try:
    from ..RAG_Search.Services import EmbeddingsService, ChunkingService, IndexingService
    RAG_SERVICES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG services not available: {e}")
    RAG_SERVICES_AVAILABLE = False
    
    # Create placeholder classes
    class EmbeddingsService:
        def __init__(self, *args, **kwargs):
            raise ImportError("EmbeddingsService not available. Please install RAG dependencies: pip install tldw_chatbook[embeddings_rag]")
    
    class ChunkingService:
        def __init__(self, *args, **kwargs):
            raise ImportError("ChunkingService not available. Please install RAG dependencies: pip install tldw_chatbook[embeddings_rag]")
    
    class IndexingService:
        def __init__(self, *args, **kwargs):
            raise ImportError("IndexingService not available. Please install RAG dependencies: pip install tldw_chatbook[embeddings_rag]")
from ..DB.search_history_db import SearchHistoryDB
from ..Utils.paths import get_user_data_dir

if TYPE_CHECKING:
    from ..app import TldwCli

logger = logger.bind(module="SearchRAGWindow")

class SearchResult(Container):
    """Container for displaying a single search result"""
    
    def __init__(self, result: Dict[str, Any], index: int):
        super().__init__(id=f"result-{index}")
        self.result = result
        self.index = index
        self.expanded = False
        
    def compose(self) -> ComposeResult:
        """Create the result display"""
        with Vertical(classes="search-result"):
            # Header with title and score
            with Horizontal(classes="result-header"):
                yield Static(
                    f"[bold]{self.index}. [{self.result['source'].upper()}] {self.result['title']}[/bold]",
                    classes="result-title"
                )
                yield Static(
                    f"Score: {self.result.get('score', 0):.3f}",
                    classes="result-score"
                )
            
            # Preview of content
            content_preview = self.result['content'][:200] + "..." if len(self.result['content']) > 200 else self.result['content']
            yield Static(content_preview, classes="result-preview")
            
            # Metadata (initially hidden)
            if self.result.get('metadata'):
                metadata_text = "\n".join([f"{k}: {v}" for k, v in self.result['metadata'].items()])
                yield Static(metadata_text, classes="result-metadata hidden")
            
            # Actions
            with Horizontal(classes="result-actions"):
                yield Button("Expand", id=f"expand-{self.index}", classes="mini")
                yield Button("Copy", id=f"copy-{self.index}", classes="mini")
                yield Button("Export", id=f"export-{self.index}", classes="mini")

class SearchRAGWindow(Container):
    """Main RAG search interface window"""
    
    BINDINGS = [
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+e", "export", "Export Results"),
        Binding("ctrl+i", "index", "Index Content"),
        Binding("escape", "clear", "Clear Search"),
    ]
    
    def __init__(self, app_instance: "TldwCli", id: str = None):
        super().__init__(id=id)
        self.app_instance = app_instance
        self.current_results: List[Dict[str, Any]] = []
        self.search_history: List[str] = []  # In-memory for quick access
        self.is_searching = False
        self.current_search_id: Optional[int] = None
        
        # Initialize search history database
        history_db_path = get_user_data_dir() / "search_history.db"
        self.search_history_db = SearchHistoryDB(history_db_path)
        
        # Load recent search history for quick access
        self._load_recent_search_history()
        
        # Check dependencies
        self.embeddings_available = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False)
        self.flashrank_available = DEPENDENCIES_AVAILABLE.get('flashrank', False)
        
    def compose(self) -> ComposeResult:
        """Create the UI layout"""
        with VerticalScroll(classes="rag-search-container"):
            yield Static("[bold]RAG Search[/bold]", classes="rag-title")
            
            # Search bar
            with Horizontal(classes="search-bar"):
                self.search_input = Input(
                    placeholder="Enter your search query...",
                    id="rag-search-input"
                )
                yield self.search_input
                yield Button("Search", id="rag-search-btn", classes="primary")
                yield LoadingIndicator(id="search-loading", classes="hidden")
            
            # Search options in collapsible sections
            with Vertical(classes="search-options"):
                # Search mode selection
                with Vertical(classes="option-section"):
                    yield Static("[bold]Search Mode[/bold]", classes="section-title")
                    yield Select(
                        options=[
                            ("Plain RAG (BM25)", "plain"),
                            ("Full RAG (Embeddings)" if self.embeddings_available else "Full RAG (Requires Dependencies)", "full"),
                            ("Hybrid Search" if self.embeddings_available else "Hybrid (Requires Dependencies)", "hybrid")
                        ],
                        value="plain",
                        id="search-mode-select"
                    )
                
                # Source selection
                with Vertical(classes="option-section"):
                    yield Static("[bold]Sources[/bold]", classes="section-title")
                    with Horizontal(classes="source-checkboxes"):
                        yield Checkbox("Media Items", value=True, id="source-media")
                        yield Checkbox("Conversations", value=True, id="source-conversations")
                        yield Checkbox("Notes", value=True, id="source-notes")
                
                # Search parameters
                with Vertical(classes="option-section collapsible"):
                    yield Static("[bold]Parameters[/bold]", classes="section-title")
                    with Horizontal(classes="parameter-inputs"):
                        with Vertical(classes="parameter-group"):
                            yield Label("Top K Results:")
                            yield Input(value="10", id="top-k-input", type="integer")
                        with Vertical(classes="parameter-group"):
                            yield Label("Max Context Length:")
                            yield Input(value="10000", id="max-context-input", type="integer")
                    
                    # Re-ranking options
                    yield Checkbox(
                        "Enable Re-ranking",
                        value=self.flashrank_available,
                        id="enable-rerank",
                        disabled=not self.flashrank_available
                    )
                
                # Advanced options
                with Vertical(classes="option-section collapsible"):
                    yield Static("[bold]Advanced Options[/bold]", classes="section-title")
                    with Horizontal(classes="parameter-inputs"):
                        with Vertical(classes="parameter-group"):
                            yield Label("Chunk Size:")
                            yield Input(value="400", id="chunk-size-input", type="integer")
                        with Vertical(classes="parameter-group"):
                            yield Label("Chunk Overlap:")
                            yield Input(value="100", id="chunk-overlap-input", type="integer")
                    yield Checkbox("Include Metadata", value=True, id="include-metadata")
                
                # Actions
                with Horizontal(classes="action-buttons"):
                    yield Button("Index Content", id="index-content-btn", classes="primary")
                    yield Button("Clear Cache", id="clear-cache-btn")
            
            # Results area with tabs
            with TabbedContent(id="results-tabs", classes="results-tabs"):
                with TabPane("Results", id="results-tab"):
                    # Results summary
                    yield Static(
                        "Enter a search query to begin",
                        id="results-summary",
                        classes="results-summary"
                    )
                    
                    # Results container
                    yield Container(id="results-container", classes="results-container")
                
                with TabPane("Context", id="context-tab"):
                    # Show the formatted context that would be sent to LLM
                    yield Markdown(
                        "No search performed yet",
                        id="context-preview"
                    )
                
                with TabPane("History", id="history-tab"):
                    # Search history
                    yield DataTable(id="search-history-table")
                
                with TabPane("Analytics", id="analytics-tab"):
                    # Search analytics and metrics
                    yield Markdown(
                        "# Search Analytics\n\nNo data available yet",
                        id="analytics-content"
                    )
    
    async def on_mount(self) -> None:
        """Initialize the window when mounted"""
        # Set up search history table
        history_table = self.query_one("#search-history-table", DataTable)
        history_table.add_columns("Time", "Query", "Mode", "Results", "Duration")
        
        # Focus search input
        self.search_input.focus()
        
        # Check indexing status
        await self._check_index_status()
    
    @on(Button.Pressed, "#rag-search-btn")
    async def handle_search(self, event: Button.Pressed) -> None:
        """Handle search button press"""
        if self.is_searching:
            return
            
        query = self.search_input.value.strip()
        if not query:
            self.app_instance.notify("Please enter a search query", severity="warning")
            return
        
        await self._perform_search(query)
    
    @on(Input.Submitted, "#rag-search-input")
    async def handle_search_submit(self, event: Input.Submitted) -> None:
        """Handle enter key in search input"""
        await self._perform_search(event.value)
    
    async def _perform_search(self, query: str) -> None:
        """Perform the actual search"""
        if self.is_searching:
            return
            
        self.is_searching = True
        start_time = datetime.now()
        search_type = ""
        results = []
        error_message = None
        
        # Show loading indicator
        loading = self.query_one("#search-loading")
        loading.remove_class("hidden")
        
        # Clear previous results
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
            
            # Check if any sources are selected
            if not any(sources.values()):
                self.app_instance.notify("Please select at least one source", severity="warning")
                return
            
            top_k = int(self.query_one("#top-k-input", Input).value or "10")
            max_context = int(self.query_one("#max-context-input", Input).value or "10000")
            enable_rerank = self.query_one("#enable-rerank", Checkbox).value
            chunk_size = int(self.query_one("#chunk-size-input", Input).value or "400")
            chunk_overlap = int(self.query_one("#chunk-overlap-input", Input).value or "100")
            include_metadata = self.query_one("#include-metadata", Checkbox).value
            
            # Perform search based on mode
            search_type = search_mode
            if search_mode == "plain":
                results, context = await perform_plain_rag_search(
                    self.app_instance,
                    query,
                    sources,
                    top_k,
                    max_context,
                    enable_rerank,
                    "flashrank"  # Will be checked inside the function
                )
            elif search_mode == "full":
                if not self.embeddings_available:
                    # Fall back to plain search
                    self.app_instance.notify("Embeddings not available, using plain search", severity="info")
                    results, context = await perform_plain_rag_search(
                        self.app_instance,
                        query,
                        sources,
                        top_k,
                        max_context,
                        enable_rerank,
                        "flashrank"
                    )
                else:
                    results, context = await perform_full_rag_pipeline(
                        self.app_instance,
                        query,
                        sources,
                        top_k,
                        max_context,
                        chunk_size,
                        chunk_overlap,
                        include_metadata,
                        enable_rerank,
                        "flashrank"  # or "cohere" based on user preference
                    )
            elif search_mode == "hybrid":
                if not self.embeddings_available:
                    # Fall back to plain search
                    self.app_instance.notify("Embeddings not available for hybrid search, using plain search", severity="info")
                    results, context = await perform_plain_rag_search(
                        self.app_instance,
                        query,
                        sources,
                        top_k,
                        max_context,
                        enable_rerank,
                        "flashrank"
                    )
                else:
                    # Perform true hybrid search
                    results, context = await perform_hybrid_rag_search(
                        self.app_instance,
                        query,
                        sources,
                        top_k,
                        max_context,
                        enable_rerank,
                        "flashrank",  # or "cohere" based on user preference
                        chunk_size,
                        chunk_overlap,
                        0.5,  # BM25 weight
                        0.5   # Vector weight
                    )
            
            # Store results
            self.current_results = results
            
            # Update UI with results
            await self._display_results(results, context)
            
            # Record search to history database
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            search_params = {
                'sources': sources,
                'top_k': top_k,
                'max_context': max_context,
                'enable_rerank': enable_rerank,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
            
            self.current_search_id = self._record_search_to_history(
                query=query,
                search_type=search_mode,
                results=results,
                execution_time_ms=duration_ms,
                search_params=search_params
            )
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            error_message = str(e)
            self.app_instance.notify(f"Search error: {error_message}", severity="error")
            
            # Record failed search to history
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self._record_search_to_history(
                query=query,
                search_type=search_type or "unknown",
                results=[],
                execution_time_ms=duration_ms,
                error_message=error_message
            )
            
        finally:
            self.is_searching = False
            loading.add_class("hidden")
    
    async def _display_results(self, results: List[Dict[str, Any]], context: str) -> None:
        """Display search results in the UI"""
        results_container = self.query_one("#results-container", Container)
        
        # Update summary
        summary = self.query_one("#results-summary", Static)
        await summary.update(f"Found {len(results)} results")
        
        # Display each result
        for i, result in enumerate(results, 1):
            result_widget = SearchResult(result, i)
            await results_container.mount(result_widget)
        
        # Update context preview
        context_preview = self.query_one("#context-preview", Markdown)
        await context_preview.update(f"```\n{context}\n```")
    
    async def _add_to_history(self, query: str, mode: str, results: int, duration: float) -> None:
        """Add search to history"""
        history_table = self.query_one("#search-history-table", DataTable)
        history_table.add_row(
            datetime.now().strftime("%H:%M:%S"),
            query[:50] + "..." if len(query) > 50 else query,
            mode,
            str(results),
            f"{duration:.2f}s"
        )
        
        # Keep history limited
        if len(self.search_history) >= 50:
            self.search_history.pop(0)
        self.search_history.append(query)
    
    @on(Button.Pressed, "#index-content-btn")
    async def handle_index_content(self, event: Button.Pressed) -> None:
        """Handle index content button"""
        await self._index_all_content()
    
    async def _index_all_content(self) -> None:
        """Index all content for embeddings-based search"""
        if not self.embeddings_available:
            self.app_instance.notify(
                "Embeddings dependencies not available. Install with: pip install -e '.[embeddings_rag]'",
                severity="warning"
            )
            return
        
        # Disable index button during indexing
        index_btn = self.query_one("#index-content-btn", Button)
        index_btn.disabled = True
        
        # Create and show progress container
        progress_container = Container(
            Static("Indexing Progress", classes="progress-title"),
            ProgressBar(id="index-progress-bar", total=100),
            Static("Preparing...", id="index-progress-label"),
            classes="index-progress-container"
        )
        
        # Insert progress container into the search options area
        options_area = self.query_one(".search-options", Vertical)
        await options_area.mount(progress_container, before=0)
        
        try:
            # Initialize services
            embeddings_dir = Path.home() / ".local" / "share" / "tldw_cli" / "chromadb"
            embeddings_service = EmbeddingsService(embeddings_dir)
            chunking_service = ChunkingService()
            indexing_service = IndexingService(embeddings_service, chunking_service)
            
            # Progress tracking
            progress_data = {
                'media': {'current': 0, 'total': 0},
                'conversations': {'current': 0, 'total': 0},
                'notes': {'current': 0, 'total': 0}
            }
            
            def update_progress(content_type: str, current: int, total: int):
                """Update progress display"""
                progress_data[content_type]['current'] = current
                progress_data[content_type]['total'] = total
                
                # Calculate overall progress
                total_items = sum(p['total'] for p in progress_data.values())
                current_items = sum(p['current'] for p in progress_data.values())
                
                if total_items > 0:
                    progress_percent = int((current_items / total_items) * 100)
                    progress_bar = self.query_one("#index-progress-bar", ProgressBar)
                    progress_bar.update(progress=progress_percent)
                    
                    # Update label
                    label = self.query_one("#index-progress-label", Static)
                    label.update(f"Indexing {content_type}: {current}/{total} (Overall: {current_items}/{total_items})")
            
            # Show initial progress
            self.app_instance.notify("Starting content indexing...", severity="info")
            
            # Index all content with progress callback
            results = await indexing_service.index_all(
                media_db=self.app_instance.media_db,
                chachanotes_db=self.app_instance.chachanotes_db,
                progress_callback=update_progress
            )
            
            # Show results
            total_indexed = sum(results.values())
            self.app_instance.notify(
                f"Indexing complete: {total_indexed} items indexed "
                f"(Media: {results['media']}, Conversations: {results['conversations']}, Notes: {results['notes']})",
                severity="success"
            )
            
            # Update final progress
            progress_bar = self.query_one("#index-progress-bar", ProgressBar)
            progress_bar.update(progress=100)
            label = self.query_one("#index-progress-label", Static)
            label.update("Indexing complete!")
            
            # Wait a moment before removing progress
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Indexing error: {e}", exc_info=True)
            self.app_instance.notify(f"Indexing error: {str(e)}", severity="error")
            
        finally:
            # Remove progress container and re-enable button
            await progress_container.remove()
            index_btn.disabled = False
    
    @on(Button.Pressed, "#clear-cache-btn")
    async def handle_clear_cache(self, event: Button.Pressed) -> None:
        """Handle clear cache button"""
        # Clear any caches (embeddings cache, result cache, etc.)
        self.app_instance.notify("Cache cleared", severity="info")
    
    async def _check_index_status(self) -> None:
        """Check the status of vector indices"""
        if not self.embeddings_available:
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
                    logger.info(f"Found {total_docs} indexed documents across {len(collections)} collections")
            
        except Exception as e:
            logger.debug(f"Could not check index status: {e}")
    
    async def _update_analytics(self) -> None:
        """Update search analytics display"""
        analytics = self.query_one("#analytics-content", Markdown)
        
        # Calculate analytics
        total_searches = len(self.search_history)
        unique_queries = len(set(self.search_history))
        
        if self.current_results:
            avg_score = sum(r.get('score', 0) for r in self.current_results) / len(self.current_results)
            source_dist = {}
            for r in self.current_results:
                source = r.get('source', 'unknown')
                source_dist[source] = source_dist.get(source, 0) + 1
        else:
            avg_score = 0
            source_dist = {}
        
        # Format analytics
        analytics_text = f"""# Search Analytics

## Session Statistics
- Total Searches: {total_searches}
- Unique Queries: {unique_queries}
- Current Results: {len(self.current_results)}

## Current Search Metrics
- Average Relevance Score: {avg_score:.3f}
- Source Distribution:
"""
        
        for source, count in source_dist.items():
            percentage = (count / len(self.current_results)) * 100 if self.current_results else 0
            analytics_text += f"  - {source.capitalize()}: {count} ({percentage:.1f}%)\n"
        
        await analytics.update(analytics_text)
    
    @on(Button.Pressed)
    async def handle_result_button(self, event: Button.Pressed) -> None:
        """Handle button presses for search results"""
        button_id = event.button.id
        
        if button_id and button_id.startswith("expand-"):
            # Handle expand button
            index = int(button_id.split("-")[1])
            await self._toggle_result_expansion(index)
            
        elif button_id and button_id.startswith("copy-"):
            # Handle copy button
            index = int(button_id.split("-")[1])
            await self._copy_result(index)
            
        elif button_id and button_id.startswith("export-"):
            # Handle export button
            index = int(button_id.split("-")[1])
            await self._export_result(index)
    
    async def _toggle_result_expansion(self, index: int) -> None:
        """Toggle expanded view of a result"""
        try:
            result_container = self.query_one(f"#result-{index}", SearchResult)
            metadata_widget = result_container.query_one(".result-metadata")
            
            if "hidden" in metadata_widget.classes:
                metadata_widget.remove_class("hidden")
                # Update full content
                if index <= len(self.current_results):
                    result = self.current_results[index - 1]
                    full_content = result.get('content', '')
                    preview_widget = result_container.query_one(".result-preview", Static)
                    await preview_widget.update(full_content)
            else:
                metadata_widget.add_class("hidden")
                # Restore preview
                if index <= len(self.current_results):
                    result = self.current_results[index - 1]
                    content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                    preview_widget = result_container.query_one(".result-preview", Static)
                    await preview_widget.update(content_preview)
                    
        except Exception as e:
            logger.error(f"Error toggling result expansion: {e}")
    
    async def _copy_result(self, index: int) -> None:
        """Copy result content to clipboard"""
        if index <= len(self.current_results):
            result = self.current_results[index - 1]
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
        """Export result to file"""
        if index <= len(self.current_results):
            result = self.current_results[index - 1]
            
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
    
    
    def action_refresh(self) -> None:
        """Refresh action"""
        self.search_input.focus()
    
    def action_export(self) -> None:
        """Export all results"""
        if not self.current_results:
            self.app_instance.notify("No results to export", severity="warning")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_results_{timestamp}.json"
            
            import json
            export_data = {
                "query": self.search_input.value,
                "timestamp": timestamp,
                "results": self.current_results
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.app_instance.notify(f"Results exported to {filename}", severity="success")
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            self.app_instance.notify(f"Export error: {str(e)}", severity="error")
    
    def action_index(self) -> None:
        """Trigger content indexing"""
        asyncio.create_task(self._index_all_content())
    
    def action_clear(self) -> None:
        """Clear search"""
        self.search_input.value = ""
        self.current_results = []
        self.current_search_id = None
        self.query_one("#results-container").remove_children()
        self.query_one("#results-summary").update("Enter a search query to begin")
        self.search_input.focus()
    
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
                # Keep only recent queries in memory
                self.search_history = self.search_history[:20]
            
            return search_id
        except Exception as e:
            logger.error(f"Error recording search to history: {e}")
            return -1
    
    def _record_result_interaction(self, result_index: int, clicked: bool = True):
        """Record user interaction with a search result."""
        if self.current_search_id and 0 <= result_index < len(self.current_results):
            try:
                self.search_history_db.record_result_feedback(
                    search_id=self.current_search_id,
                    result_index=result_index,
                    clicked=clicked
                )
            except Exception as e:
                logger.error(f"Error recording result interaction: {e}")
    
    def get_search_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get search analytics from the history database."""
        try:
            return self.search_history_db.get_search_analytics(days_back=days_back)
        except Exception as e:
            logger.error(f"Error getting search analytics: {e}")
            return {}
    
    def export_search_history(self, output_path: Optional[Path] = None, days_back: int = 30) -> bool:
        """Export search history to a file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path.home() / "Downloads" / f"rag_search_history_{timestamp}.json"
        
        try:
            success = self.search_history_db.export_search_data(output_path, days_back=days_back)
            if success:
                self.app_instance.notify(f"Search history exported to {output_path}", severity="success")
            else:
                self.app_instance.notify("Failed to export search history", severity="error")
            return success
        except Exception as e:
            logger.error(f"Error exporting search history: {e}")
            self.app_instance.notify(f"Export error: {str(e)}", severity="error")
            return False
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get popular search queries."""
        try:
            return self.search_history_db.get_popular_queries(limit=limit)
        except Exception as e:
            logger.error(f"Error getting popular queries: {e}")
            return []