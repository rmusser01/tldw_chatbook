# Embeddings_Management_Window.py
# Description: Embeddings Management interface with dual-pane layout
#
# Imports
from __future__ import annotations

from typing import Optional, List, Dict, Any
from loguru import logger

# Third-party imports
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button, Input, Label, ListView, ListItem, Select, TextArea,
    Collapsible, LoadingIndicator, Markdown, Static
)

# Local imports
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ..DB.ChaChaNotes_DB import CharactersRAGDB

# Check if embeddings dependencies are available
if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
    from ..Embeddings.Embeddings_Lib import EmbeddingFactory
    from ..Embeddings.Chroma_Lib import ChromaDBManager
else:
    EmbeddingFactory = None
    ChromaDBManager = None

logger = logger.bind(name="Embeddings_Management_Window")

########################################################################################################################
#
# Embeddings Management Window
#
########################################################################################################################

class EmbeddingsManagementWindow(Widget):
    """Embeddings Management window with dual-pane layout."""
    
    DEFAULT_CSS = """
    EmbeddingsManagementWindow {
        layout: horizontal;
        height: 100%;
        width: 100%;
    }
    
    .embeddings-left-pane {
        width: 40%;
        min-width: 30;
        max-width: 60;
        height: 100%;
        background: $boost;
        padding: 1;
        border-right: thick $background-darken-1;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    .embeddings-left-pane.collapsed {
        width: 0 !important;
        min-width: 0 !important;
        border-right: none !important;
        padding: 0 !important;
        overflow: hidden !important;
        display: none !important;
    }
    
    .embeddings-right-pane {
        width: 1fr;
        height: 100%;
        background: $surface;
        padding: 1 2;
        overflow-y: auto;
    }
    
    .embeddings-section-title {
        text-style: bold underline;
        margin-bottom: 1;
        text-align: center;
        width: 100%;
    }
    
    .embeddings-list-item {
        width: 100%;
        padding: 1;
        margin-bottom: 1;
        border: round $surface;
        background: $panel;
    }
    
    .embeddings-list-item:hover {
        background: $panel-lighten-1;
        border: round $accent;
    }
    
    .embeddings-list-item.-selected {
        background: $accent-darken-1;
        border: round $accent;
    }
    
    .embeddings-info-label {
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }
    
    .embeddings-info-value {
        margin-bottom: 1;
        padding: 1;
        background: $surface-darken-1;
        border: round $primary-darken-1;
    }
    
    .embeddings-action-button {
        width: 100%;
        margin-top: 1;
    }
    
    .embeddings-test-area {
        margin-top: 2;
        padding: 1;
        border: round $primary;
        background: $surface;
    }
    
    .embeddings-loading {
        align: center middle;
        height: 100%;
    }
    
    .embeddings-error {
        color: $error;
        text-style: bold;
        padding: 1;
        border: round $error;
        margin: 1;
    }
    
    .embeddings-toggle-button {
        width: 5;
        height: 100%;
        min-width: 0;
        border: none;
        background: $surface-darken-1;
        color: $text;
    }
    
    .embeddings-toggle-button:hover {
        background: $surface;
    }
    """
    
    # Reactive attributes
    selected_model: reactive[Optional[str]] = reactive(None)
    selected_collection: reactive[Optional[str]] = reactive(None)
    left_pane_collapsed: reactive[bool] = reactive(False)
    is_loading: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: Any, **kwargs):
        """Initialize the Embeddings Management Window.
        
        Args:
            app_instance: The main app instance
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.chachanotes_db = app_instance.chachanotes_db if hasattr(app_instance, 'chachanotes_db') else None
        self.embedding_factory: Optional[EmbeddingFactory] = None
        self.chroma_manager: Optional[ChromaDBManager] = None
        self.available_models: List[str] = []
        self.collections: List[Dict[str, Any]] = []
        
        # Check dependencies
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            logger.warning("Embeddings/RAG dependencies not available")
    
    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        with Horizontal():
            # Left pane - Model and Collection Selection
            with Container(classes="embeddings-left-pane", id="embeddings-left-pane"):
                with VerticalScroll():
                    yield Label("Embedding Models", classes="embeddings-section-title")
                    
                    # Search input for models
                    yield Input(
                        placeholder="Search models...",
                        id="embeddings-model-search"
                    )
                    
                    # Model list
                    yield ListView(
                        id="embeddings-model-list",
                        classes="sidebar-listview"
                    )
                    
                    yield Label("Vector Collections", classes="embeddings-section-title")
                    
                    # Search input for collections
                    yield Input(
                        placeholder="Search collections...",
                        id="embeddings-collection-search"
                    )
                    
                    # Collection list
                    yield ListView(
                        id="embeddings-collection-list",
                        classes="sidebar-listview"
                    )
                    
                    # Refresh button
                    yield Button(
                        "Refresh Lists",
                        id="embeddings-refresh-lists",
                        classes="embeddings-action-button"
                    )
            
            # Toggle button
            yield Button(
                "☰",
                id="toggle-embeddings-pane",
                classes="embeddings-toggle-button"
            )
            
            # Right pane - Details and Actions
            with Container(classes="embeddings-right-pane", id="embeddings-right-pane"):
                with VerticalScroll():
                    # Model information section
                    with Collapsible(title="Model Information", id="embeddings-model-info-collapsible"):
                        yield Label("Provider:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-model-provider", classes="embeddings-info-value")
                        
                        yield Label("Dimension:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-model-dimension", classes="embeddings-info-value")
                        
                        yield Label("Status:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-model-status", classes="embeddings-info-value")
                        
                        yield Label("Cache Status:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-model-cache-status", classes="embeddings-info-value")
                        
                        # Model actions
                        with Horizontal():
                            yield Button("Load Model", id="embeddings-load-model", classes="embeddings-action-button")
                            yield Button("Unload Model", id="embeddings-unload-model", classes="embeddings-action-button")
                    
                    # Collection information section
                    with Collapsible(title="Collection Information", id="embeddings-collection-info-collapsible"):
                        yield Label("Name:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-collection-name", classes="embeddings-info-value")
                        
                        yield Label("Document Count:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-collection-count", classes="embeddings-info-value")
                        
                        yield Label("Metadata:", classes="embeddings-info-label")
                        yield TextArea(
                            "",
                            id="embeddings-collection-metadata",
                            read_only=True,
                            classes="embeddings-info-value"
                        )
                        
                        # Collection actions
                        with Horizontal():
                            yield Button("Delete Collection", id="embeddings-delete-collection", classes="embeddings-action-button")
                            yield Button("Export Collection", id="embeddings-export-collection", classes="embeddings-action-button")
                    
                    # Testing section
                    with Collapsible(title="Test Embeddings", id="embeddings-test-collapsible"):
                        yield Label("Test Text:", classes="embeddings-info-label")
                        yield TextArea(
                            "Enter text to test embeddings...",
                            id="embeddings-test-input",
                            classes="embeddings-test-area"
                        )
                        
                        yield Button("Generate Embedding", id="embeddings-test-generate", classes="embeddings-action-button")
                        
                        yield Label("Result:", classes="embeddings-info-label")
                        yield TextArea(
                            "",
                            id="embeddings-test-result",
                            read_only=True,
                            classes="embeddings-info-value"
                        )
                    
                    # Performance metrics section
                    with Collapsible(title="Performance Metrics", id="embeddings-metrics-collapsible"):
                        yield Label("Memory Usage:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-memory-usage", classes="embeddings-info-value")
                        
                        yield Label("Average Processing Time:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-avg-time", classes="embeddings-info-value")
                        
                        yield Label("Total Embeddings Generated:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-total-generated", classes="embeddings-info-value")
    
    async def on_mount(self) -> None:
        """Handle mount event - initialize embeddings components."""
        await self._initialize_embeddings()
        await self._load_models_list()
        await self._load_collections_list()
    
    async def _initialize_embeddings(self) -> None:
        """Initialize embedding factory and ChromaDB manager."""
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            self._show_error("Embeddings dependencies not installed. Install with: pip install tldw_chatbook[embeddings_rag]")
            return
        
        try:
            # Load embedding configuration
            from ..config import get_cli_setting, load_settings
            
            # Get the full settings and extract embedding_config
            settings = load_settings()
            embedding_config = settings.get('embedding_config', {})
            
            # Create default config that EmbeddingFactory can use directly
            default_factory_config = {
                'default_model_id': 'e5-small-v2',
                'models': {
                    'e5-small-v2': {
                        'provider': 'huggingface',
                        'model_name_or_path': 'intfloat/e5-small-v2',
                        'dimension': 384
                    }
                }
            }
            
            # The EmbeddingFactory constructor will handle validation internally
            # We pass the raw config and let it do the validation
            if embedding_config and embedding_config.get('models'):
                try:
                    self.embedding_factory = EmbeddingFactory(
                        embedding_config,
                        max_cached=2,
                        idle_seconds=900
                    )
                    logger.info("Initialized embedding factory with user configuration")
                except Exception as e:
                    logger.error(f"Failed to initialize with user config: {e}, using defaults")
                    self.embedding_factory = EmbeddingFactory(
                        default_factory_config,
                        max_cached=2,
                        idle_seconds=900
                    )
                    logger.info("Initialized embedding factory with default configuration")
            else:
                logger.warning("No embedding configuration found, using defaults")
                self.embedding_factory = EmbeddingFactory(
                    default_factory_config,
                    max_cached=2,
                    idle_seconds=900
                )
                
            # Initialize ChromaDB manager
            user_id = settings.get('USERS_NAME', 'default_user')
            # ChromaDBManager expects a full config structure with embedding_config inside it
            chroma_config = {
                'embedding_config': embedding_config if embedding_config else default_factory_config,
                'database': settings.get('database', {}),
                'USER_DB_BASE_DIR': settings.get('database', {}).get('USER_DB_BASE_DIR', '~/.local/share/tldw_cli')
            }
            self.chroma_manager = ChromaDBManager(user_id, chroma_config)
            logger.info("Initialized ChromaDB manager")
            
        except Exception as e:
            logger.error(f"(EMW) Failed to initialize embeddings: {e}")
            self._show_error(f"(EMW) Failed to initialize embeddings: {str(e)}")
    
    async def _load_models_list(self) -> None:
        """Load the list of available embedding models."""
        if not self.embedding_factory:
            return
        
        try:
            model_list = self.query_one("#embeddings-model-list", ListView)
            await model_list.clear()
            
            # Get models from configuration
            config = self.embedding_factory.config
            if config and hasattr(config, 'models'):
                self.available_models = list(config.models.keys())
                
                for model_id in self.available_models:
                    await model_list.append(ListItem(Label(model_id), id=f"model-{model_id}"))
                
                logger.info(f"Loaded {len(self.available_models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load models list: {e}")
    
    async def _load_collections_list(self) -> None:
        """Load the list of vector collections."""
        if not self.chroma_manager:
            return
        
        try:
            collection_list = self.query_one("#embeddings-collection-list", ListView)
            await collection_list.clear()
            
            # Get collections from ChromaDB
            # Note: This is a simplified version - actual implementation would need
            # to interface with ChromaDB's collection listing API
            self.collections = []  # Placeholder for actual collection data
            
            logger.info("Collections list loaded")
            
        except Exception as e:
            logger.error(f"Failed to load collections list: {e}")
    
    # Event handlers
    @on(ListView.Selected, "#embeddings-model-list")
    async def on_model_selected(self, event: ListView.Selected) -> None:
        """Handle model selection."""
        if event.item and event.item.id:
            model_id = event.item.id.replace("model-", "")
            self.selected_model = model_id
            await self._update_model_info(model_id)
    
    @on(ListView.Selected, "#embeddings-collection-list")
    async def on_collection_selected(self, event: ListView.Selected) -> None:
        """Handle collection selection."""
        if event.item and event.item.id:
            collection_name = event.item.id.replace("collection-", "")
            self.selected_collection = collection_name
            await self._update_collection_info(collection_name)
    
    @on(Button.Pressed, "#toggle-embeddings-pane")
    def on_toggle_pane(self) -> None:
        """Toggle the left pane visibility."""
        self.left_pane_collapsed = not self.left_pane_collapsed
        left_pane = self.query_one("#embeddings-left-pane")
        if self.left_pane_collapsed:
            left_pane.add_class("collapsed")
        else:
            left_pane.remove_class("collapsed")
    
    @on(Button.Pressed, "#embeddings-refresh-lists")
    async def on_refresh_lists(self) -> None:
        """Refresh both models and collections lists."""
        await self._load_models_list()
        await self._load_collections_list()
        self.notify("Lists refreshed", severity="information")
    
    @on(Button.Pressed, "#embeddings-load-model")
    async def on_load_model(self) -> None:
        """Load the selected model."""
        if not self.selected_model:
            self.notify("Please select a model first", severity="warning")
            return
        
        if not self.embedding_factory:
            self.notify("Embedding factory not initialized", severity="error")
            return
        
        try:
            self.is_loading = True
            # Prefetch the model to load it into cache
            self.embedding_factory.prefetch([self.selected_model])
            self.notify(f"Model {self.selected_model} loaded successfully", severity="information")
            await self._update_model_info(self.selected_model)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.notify(f"Failed to load model: {str(e)}", severity="error")
        finally:
            self.is_loading = False
    
    @on(Button.Pressed, "#embeddings-test-generate")
    async def on_test_generate(self) -> None:
        """Generate test embeddings."""
        if not self.selected_model:
            self.notify("Please select a model first", severity="warning")
            return
        
        if not self.embedding_factory:
            self.notify("Embedding factory not initialized", severity="error")
            return
        
        test_input = self.query_one("#embeddings-test-input", TextArea)
        test_result = self.query_one("#embeddings-test-result", TextArea)
        
        text = test_input.text.strip()
        if not text:
            self.notify("Please enter test text", severity="warning")
            return
        
        try:
            self.is_loading = True
            # Generate embedding
            embedding = await self.embedding_factory.async_embed(
                [text],
                model_id=self.selected_model,
                as_list=True
            )
            
            # Display result
            result_text = f"Model: {self.selected_model}\n"
            result_text += f"Dimension: {len(embedding[0]) if embedding else 0}\n"
            result_text += f"First 10 values: {embedding[0][:10] if embedding else []}\n"
            result_text += f"Norm: {sum(x**2 for x in embedding[0])**0.5 if embedding else 0:.4f}"
            
            test_result.text = result_text
            
        except Exception as e:
            logger.error(f"Failed to generate test embedding: {e}")
            test_result.text = f"Error: {str(e)}"
        finally:
            self.is_loading = False
    
    async def _update_model_info(self, model_id: str) -> None:
        """Update the model information display."""
        if not self.embedding_factory:
            return
        
        try:
            config = self.embedding_factory.config
            if not config or not hasattr(config, 'models'):
                return
            
            model_spec = config.models.get(model_id)
            if not model_spec:
                return
            
            # Update provider
            provider_widget = self.query_one("#embeddings-model-provider", Static)
            provider_widget.update(str(model_spec.provider))
            
            # Update dimension
            dimension_widget = self.query_one("#embeddings-model-dimension", Static)
            dimension = getattr(model_spec, 'dimension', 'Unknown')
            dimension_widget.update(str(dimension))
            
            # Update status
            status_widget = self.query_one("#embeddings-model-status", Static)
            # Check if model is in cache
            is_cached = model_id in getattr(self.embedding_factory, '_cache', {})
            status_widget.update("Loaded" if is_cached else "Not Loaded")
            
            # Update cache status
            cache_widget = self.query_one("#embeddings-model-cache-status", Static)
            cache_info = f"Models in cache: {len(getattr(self.embedding_factory, '_cache', {}))}"
            cache_widget.update(cache_info)
            
        except Exception as e:
            logger.error(f"Failed to update model info: {e}")
    
    async def _update_collection_info(self, collection_name: str) -> None:
        """Update the collection information display."""
        # This would be implemented when we have actual ChromaDB collection data
        pass
    
    def _show_error(self, message: str) -> None:
        """Show an error message in the UI."""
        # This could be implemented as a modal or notification
        self.notify(message, severity="error")

# End of Embeddings_Management_Window.py
########################################################################################################################