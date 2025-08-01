# tldw_chatbook/Widgets/HuggingFace/model_search_widget.py
"""
Search widget for HuggingFace GGUF models.
"""

from typing import Optional, List, Dict, Any
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, Button, Select, Label, ListView, ListItem, Static
from textual.message import Message
from textual.reactive import reactive
from textual import work
from loguru import logger


class ModelSelectedEvent(Message):
    """Event fired when a model is selected from search results."""
    
    def __init__(self, model_info: Dict[str, Any]) -> None:
        super().__init__()
        self.model_info = model_info


class ModelSearchWidget(Container):
    """Widget for searching and browsing HuggingFace models."""
    
    DEFAULT_CSS = """
    ModelSearchWidget {
        height: 100%;
        layout: vertical;
    }
    
    ModelSearchWidget .search-container {
        height: auto;
        padding: 1;
        background: $boost;
        border-bottom: solid $primary;
    }
    
    ModelSearchWidget .search-row {
        height: 3;
        margin-bottom: 1;
    }
    
    ModelSearchWidget .search-input {
        width: 1fr;
        margin-right: 1;
    }
    
    ModelSearchWidget .filter-row {
        height: 3;
    }
    
    ModelSearchWidget .filter-select {
        width: 1fr;
        margin-right: 1;
    }
    
    ModelSearchWidget #results-list {
        height: 1fr;
        overflow-y: auto;
        border: solid $primary;
        background: $background;
        padding: 0;
    }
    
    ModelSearchWidget .model-item {
        padding: 1;
        margin-bottom: 1;
        background: $surface;
        border: solid $primary-background-darken-1;
    }
    
    ModelSearchWidget .model-item:hover {
        background: $surface-lighten-1;
    }
    
    ModelSearchWidget .model-title {
        text-style: bold;
        color: $primary;
    }
    
    ModelSearchWidget .model-meta {
        color: $text-muted;
        margin-top: 0;
    }
    
    ModelSearchWidget .model-tags {
        margin-top: 0;
    }
    
    ModelSearchWidget .loading-message {
        text-align: center;
        padding: 2;
        color: $text-muted;
    }
    
    ModelSearchWidget .error-message {
        text-align: center;
        padding: 2;
        color: $error;
    }
    """
    
    # Reactive properties
    is_loading: reactive[bool] = reactive(False)
    error_message: reactive[Optional[str]] = reactive(None)
    results: reactive[List[Dict[str, Any]]] = reactive([])
    
    SORT_OPTIONS = [
        ("Most Downloads", "downloads"),
        ("Most Likes", "likes"),
        ("Recently Updated", "lastModified"),
    ]
    
    def compose(self) -> ComposeResult:
        """Compose the search widget UI."""
        with Container(classes="search-container"):
            # Search input row
            with Horizontal(classes="search-row"):
                yield Input(
                    placeholder="Search for GGUF models...",
                    id="model-search-input",
                    classes="search-input"
                )
                yield Button("Search", id="search-button", variant="primary")
            
            # Filter row
            with Horizontal(classes="filter-row"):
                yield Select(
                    self.SORT_OPTIONS,
                    prompt="Sort by",
                    id="sort-select",
                    classes="filter-select"
                )
                yield Button("Recent", id="browse-recent", variant="default")
                yield Button("Popular", id="browse-popular", variant="default")
        
        # Results list directly without container wrapper
        yield ListView(id="results-list", classes="results-list")
    
    def on_mount(self) -> None:
        """Initialize with popular models on mount."""
        # Defer initial browse to allow Select to fully initialize
        self.call_after_refresh(self._initial_browse)
    
    def _initial_browse(self) -> None:
        """Perform initial browse after widget is ready."""
        # Just perform search without setting select value
        self.browsing_mode = True
        self.perform_search()
    
    def watch_is_loading(self, is_loading: bool) -> None:
        """Update UI when loading state changes."""
        if is_loading:
            self.call_later(self._show_loading_message)
    
    def watch_error_message(self, error: Optional[str]) -> None:
        """Display error message if any."""
        if error:
            self.call_later(self._show_error_message, error)
    
    def watch_results(self, results: List[Dict[str, Any]]) -> None:
        """Update results list when search results change."""
        logger.info(f"watch_results called with {len(results)} results")
        
        # Schedule async update
        self.call_later(self._update_results_list, results)
    
    def _create_model_item(self, model: Dict[str, Any]) -> ListItem:
        """Create a list item for a model."""
        model_id = model.get("id", "Unknown")
        author = model.get("author", "Unknown")
        downloads = model.get("downloads", 0)
        likes = model.get("likes", 0)
        last_modified = model.get("lastModified", "Unknown")
        tags = model.get("tags", [])[:5]  # Show first 5 tags
        
        # Format numbers
        downloads_str = self._format_number(downloads)
        likes_str = self._format_number(likes)
        
        # Handle lastModified formatting
        if isinstance(last_modified, str) and len(last_modified) >= 10:
            last_modified_str = last_modified[:10]
        else:
            last_modified_str = "Unknown"
        
        # Create content as a single formatted string
        content = f"{model_id}\nby {author} • ⬇ {downloads_str} • ★ {likes_str} • Updated: {last_modified_str}\nTags: {', '.join(tags) if tags else 'No tags'}"
        
        # Create item with a single Static widget
        item = ListItem(
            Static(content, classes="model-content"),
            classes="model-item"
        )
        
        # Store model data on the item
        item.model_data = model
        return item
    
    def _format_number(self, num: int) -> str:
        """Format large numbers with K/M suffix."""
        if num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        else:
            return str(num)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "search-button":
            self.perform_search()
        elif event.button.id == "browse-recent":
            self.browse_recent()
        elif event.button.id == "browse-popular":
            self.browse_popular()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key in search input."""
        if event.input.id == "model-search-input":
            self.perform_search()
    
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle model selection from list."""
        if hasattr(event.item, "model_data"):
            self.post_message(ModelSelectedEvent(event.item.model_data))
    
    def perform_search(self) -> None:
        """Trigger model search in background."""
        # Use run_worker with an async coroutine
        async def _perform_search():
            search_input = self.query_one("#model-search-input", Input)
            query = search_input.value.strip()
            
            if not query and not self.browsing_mode:
                return
            
            sort_select = self.query_one("#sort-select", Select)
            sort_display = sort_select.value or "Most Downloads"
            
            # Map display value to API value
            sort_map = {opt[0]: opt[1] for opt in self.SORT_OPTIONS}
            sort_by = sort_map.get(sort_display, "downloads")
            
            self.is_loading = True
            self.error_message = None
            
            try:
                # Import here to avoid circular imports
                from ...LLM_Calls.huggingface_api import HuggingFaceAPI
                
                api = HuggingFaceAPI()
                results = await api.search_models(
                    query=query,
                    sort=sort_by,
                    limit=50
                )
                
                logger.info(f"Search returned {len(results)} results")
                logger.debug(f"First 3 results: {results[:3] if results else 'None'}")
                
                # Update results - this will trigger the watcher
                self.results = results
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                self.error_message = str(e)
            finally:
                self.is_loading = False
                self.browsing_mode = False
        
        # Run the worker
        self.run_worker(_perform_search(), exclusive=True)
    
    def browse_recent(self) -> None:
        """Browse recently updated models."""
        self.browsing_mode = True
        try:
            sort_select = self.query_one("#sort-select", Select)
            # Only set value if Select is ready
            if sort_select._options:
                sort_select.value = "Recently Updated"
        except Exception:
            pass
        search_input = self.query_one("#model-search-input", Input)
        search_input.value = ""
        self.perform_search()
    
    def browse_popular(self) -> None:
        """Browse most popular models."""
        self.browsing_mode = True
        try:
            sort_select = self.query_one("#sort-select", Select)
            # Only set value if Select is ready
            if sort_select._options:
                sort_select.value = "Most Downloads"
        except Exception:
            pass
        search_input = self.query_one("#model-search-input", Input)
        search_input.value = ""
        self.perform_search()
    
    def __init__(self, **kwargs):
        """Initialize the search widget."""
        super().__init__(**kwargs)
        self.browsing_mode = False
    
    
    async def _update_results_list(self, results: List[Dict[str, Any]]) -> None:
        """Update the ListView with search results."""
        results_list = self.query_one("#results-list", ListView)
        await results_list.clear()
        
        if not results and not self.is_loading:
            await results_list.append(
                ListItem(Static("No models found", classes="loading-message"))
            )
            return
        
        for model in results:
            item = self._create_model_item(model)
            await results_list.append(item)
        
        logger.info(f"Added {len(results)} items to results list")
    
    async def _show_loading_message(self) -> None:
        """Show loading message in the ListView."""
        results_list = self.query_one("#results-list", ListView)
        await results_list.clear()
        await results_list.append(
            ListItem(Static("Loading models...", classes="loading-message"))
        )
    
    async def _show_error_message(self, error: str) -> None:
        """Show error message in the ListView."""
        results_list = self.query_one("#results-list", ListView)
        await results_list.clear()
        await results_list.append(
            ListItem(Static(f"Error: {error}", classes="error-message"))
        )