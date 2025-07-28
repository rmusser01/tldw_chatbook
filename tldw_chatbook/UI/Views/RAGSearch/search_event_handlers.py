"""
Event Handlers for RAG Search Window

This module contains all the event handling logic separated from the main window
"""

from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import json
import tempfile
from pathlib import Path

from textual import on, work
from textual.widgets import Button, Select, Checkbox, Input, ListView, ListItem, DataTable
from textual.css.query import NoMatches
from rich.text import Text
from loguru import logger

from .search_result import SearchResult
from .constants import MAX_CONCURRENT_SEARCHES, DEFAULT_TOP_K, DEFAULT_TEMPERATURE

# Import required dependencies conditionally
from ....Utils.optional_deps import DEPENDENCIES_AVAILABLE

WEB_SEARCH_AVAILABLE = DEPENDENCIES_AVAILABLE.get('websearch', False)
if WEB_SEARCH_AVAILABLE:
    try:
        from ....Web_Scraping.WebSearch_APIs import search_web_bing, parse_bing_results
    except (ImportError, ModuleNotFoundError):
        WEB_SEARCH_AVAILABLE = False

try:
    from ....Event_Handlers.Chat_Events.chat_rag_events import (
        perform_plain_rag_search, perform_full_rag_pipeline, perform_hybrid_rag_search
    )
    RAG_EVENTS_AVAILABLE = True
except ImportError:
    RAG_EVENTS_AVAILABLE = False


class SearchEventHandlersMixin:
    """Mixin class containing all event handlers for SearchRAGWindow"""
    
    @on(Button.Pressed, "#search-button")
    @work(exclusive=True, thread=True)
    async def handle_search(self, event: Button.Pressed) -> None:
        """Handle search button press with improved error handling and feedback"""
        if self.is_searching or self.active_searches >= MAX_CONCURRENT_SEARCHES:
            self.app_instance.notify(
                f"Search in progress. Maximum {MAX_CONCURRENT_SEARCHES} concurrent searches allowed.",
                severity="warning"
            )
            return
        
        # Get search query
        query_input = self.query_one("#search-query-input", Input)
        query = query_input.value.strip()
        
        if not query:
            self.app_instance.notify("Please enter a search query", severity="warning")
            query_input.focus()
            return
        
        # Start search
        self.is_searching = True
        self.active_searches += 1
        search_start_time = asyncio.get_event_loop().time()
        
        # Show search status
        status_container = self.query_one("#search-status-container")
        status_container.remove_class("hidden")
        status_text = self.query_one("#search-status-text")
        status_text.update("🔍 Searching...")
        
        # Get search configuration
        search_config = self._get_search_config()
        self.last_search_config = search_config
        
        # Record search to history
        self._record_search_to_history(
            query=query,
            search_type=search_config['mode'],
            filters=search_config.get('filters', {}),
            results_count=0  # Will update later
        )
        
        try:
            # Hide search history dropdown
            history_dropdown = self.query_one(".search-history-dropdown")
            history_dropdown.hide()
            
            # Clear previous results
            await self._clear_results()
            
            # Perform search based on mode
            search_mode = search_config['mode']
            results = []
            
            if search_mode == "plain":
                status_text.update("🔍 Performing plain RAG search...")
                results = await self._perform_plain_search(query, search_config)
            elif search_mode == "contextual":
                status_text.update("🧠 Performing contextual search...")
                results = await self._perform_contextual_search(query, search_config)
            elif search_mode == "hybrid":
                status_text.update("🔄 Performing hybrid search...")
                results = await self._perform_hybrid_search(query, search_config)
            
            # Include web search if enabled
            if search_config.get('include_web_search', False) and WEB_SEARCH_AVAILABLE:
                status_text.update("🌐 Searching the web...")
                web_results = await self._perform_web_search(query)
                results.extend(web_results)
            
            # Process and display results
            self.search_results = results
            self.total_results = len(results)
            
            # Update search time
            search_end_time = asyncio.get_event_loop().time()
            self.last_search_time = search_end_time - search_start_time
            self._update_search_metrics(self.last_search_time)
            
            # Display results
            await self._display_results()
            
            # Update UI
            self.query_one("#save-search-button").disabled = False
            
            # Show results header
            results_header = self.query_one("#results-header-enhanced")
            results_header.remove_class("hidden")
            
            # Update results count and time
            self.query_one("#results-count").update(
                f"Found [bold cyan]{self.total_results}[/bold cyan] results"
            )
            self.query_one("#search-time").update(
                f"Search completed in [bold green]{self.last_search_time:.2f}s[/bold green]"
            )
            
            # Show success notification
            self.app_instance.notify(
                f"Search completed: {self.total_results} results found",
                severity="information"
            )
            
            # Update history with results count
            self._update_last_search_results_count(self.total_results)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            self.app_instance.notify(
                f"Search failed: {str(e)}",
                severity="error"
            )
            # Show error in results area
            results_list = self.query_one("#results-list-enhanced")
            error_widget = Static(
                f"[bold red]Search Error[/bold red]\n\n{str(e)}",
                classes="search-error"
            )
            await results_list.mount(error_widget)
            
        finally:
            self.is_searching = False
            self.active_searches -= 1
            status_container.add_class("hidden")
    
    @on(Button.Pressed, "#clear-search-button")
    async def handle_clear_search(self, event: Button.Pressed) -> None:
        """Clear search results and reset form"""
        await self._clear_results()
        self.query_one("#search-query-input", Input).value = ""
        self.query_one("#search-query-input").focus()
        self.search_results = []
        self.total_results = 0
        self.current_page = 1
        
        # Hide results header
        self.query_one("#results-header-enhanced").add_class("hidden")
        self.query_one("#pagination-enhanced").add_class("hidden")
        
        # Disable save button
        self.query_one("#save-search-button").disabled = True
        
        self.app_instance.notify("Search cleared", severity="information")
    
    @on(Checkbox.Changed, "#parent-docs-checkbox")
    async def handle_parent_docs_toggle(self, event: Checkbox.Changed) -> None:
        """Handle parent document retrieval toggle"""
        self.enable_parent_docs = event.value
        parent_options = self.query_one("#parent-docs-options")
        
        if event.value:
            parent_options.remove_class("disabled")
            self._update_parent_inclusion_preview()
        else:
            parent_options.add_class("disabled")
    
    @on(Select.Changed, "#parent-strategy-select")
    async def handle_parent_strategy_change(self, event: Select.Changed) -> None:
        """Handle parent retrieval strategy change"""
        self.parent_retrieval_strategy = event.value
        self._update_parent_inclusion_preview()
    
    @on(Input.Changed, "#parent-size-input")
    async def handle_parent_size_change(self, event: Input.Changed) -> None:
        """Handle parent size input change"""
        try:
            self.parent_retrieval_size = int(event.value)
            self._update_parent_inclusion_preview()
        except ValueError:
            pass
    
    @on(Select.Changed, "#search-mode-select")
    async def handle_search_mode_change(self, event: Select.Changed) -> None:
        """Handle search mode change"""
        self.current_search_mode = event.value
        
        # Enable/disable temperature input based on mode
        temp_input = self.query_one("#temperature-input", Input)
        temp_input.disabled = event.value == "plain"
        
        # Update UI hints based on mode
        if event.value == "contextual":
            self.app_instance.notify(
                "Contextual search uses LLM to understand query intent",
                severity="information"
            )
        elif event.value == "hybrid":
            self.app_instance.notify(
                "Hybrid search combines keyword and semantic search",
                severity="information"
            )
    
    # Helper methods for search functionality
    def _get_search_config(self) -> Dict[str, Any]:
        """Get current search configuration from UI"""
        config = {
            "mode": self.query_one("#search-mode-select", Select).value,
            "collection": self.query_one("#collection-select", Select).value,
            "top_k": int(self.query_one("#top-k-input", Input).value or DEFAULT_TOP_K),
            "temperature": float(self.query_one("#temperature-input", Input).value or DEFAULT_TEMPERATURE),
            "enable_parent_docs": self.enable_parent_docs,
            "parent_strategy": self.parent_retrieval_strategy,
            "parent_size": self.parent_retrieval_size,
            "filters": {
                "media": self.query_one("#filter-media", Checkbox).value,
                "conversations": self.query_one("#filter-conversations", Checkbox).value,
                "notes": self.query_one("#filter-notes", Checkbox).value,
            }
        }
        
        if WEB_SEARCH_AVAILABLE:
            config["include_web_search"] = self.query_one("#include-web-search", Checkbox).value
        
        return config
    
    async def _clear_results(self) -> None:
        """Clear the results display"""
        results_list = self.query_one("#results-list-enhanced")
        await results_list.remove_children()
    
    def _update_parent_inclusion_preview(self) -> None:
        """Update the parent inclusion preview text"""
        preview_text = self.query_one("#parent-preview-text")
        
        if self.parent_retrieval_strategy == "full":
            preview_text.update(
                f"[dim]Will retrieve full parent documents[/dim]"
            )
        elif self.parent_retrieval_strategy == "sentence_window":
            preview_text.update(
                f"[dim]Will retrieve {self.parent_retrieval_size} character window around matches[/dim]"
            )
        elif self.parent_retrieval_strategy == "auto_merging":
            preview_text.update(
                f"[dim]Will automatically merge adjacent chunks up to {self.parent_retrieval_size} characters[/dim]"
            )
    
    def _update_search_metrics(self, search_time: float) -> None:
        """Update search performance metrics"""
        self.search_metrics["total_searches"] += 1
        
        # Update average
        total = self.search_metrics["total_searches"]
        current_avg = self.search_metrics["avg_search_time"]
        self.search_metrics["avg_search_time"] = (
            (current_avg * (total - 1) + search_time) / total
        )
        
        # Update extremes
        self.search_metrics["fastest_search"] = min(
            self.search_metrics["fastest_search"],
            search_time
        )
        self.search_metrics["slowest_search"] = max(
            self.search_metrics["slowest_search"],
            search_time
        )