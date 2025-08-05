# Embeddings_Management_Window.py
# Description: Embeddings Management interface with dual-pane layout
#
# Imports
from __future__ import annotations

from typing import Optional, List, Dict, Any
from loguru import logger
import os
import time

# Third-party imports
from textual import events, on, work
from textual.app import ComposeResult
from textual.message import Message
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button, Input, Label, ListView, ListItem, Select, TextArea,
    Collapsible, LoadingIndicator, Markdown, Static, Checkbox
)
from textual.screen import ModalScreen

# Local widget imports
from ..Widgets.embeddings_list_items import ModelListItem, CollectionListItem
from ..Widgets.empty_state import ModelsEmptyState, CollectionsEmptyState
from ..Widgets.activity_log import ActivityLogWidget
from ..Widgets.performance_metrics import PerformanceMetricsWidget

# Local imports
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ..Utils.model_preferences import ModelPreferencesManager
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

class DownloadStatusMessage(Message):
    """Message to update download status."""
    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__()

class SetLoadingMessage(Message):
    """Message to set loading state."""
    def __init__(self, loading: bool) -> None:
        self.loading = loading
        super().__init__()

class EmbeddingsManagementWindow(Widget):
    """Embeddings Management window with dual-pane layout."""
    
    # Reactive attributes
    selected_model: reactive[Optional[str]] = reactive(None)
    selected_collection: reactive[Optional[str]] = reactive(None)
    left_pane_collapsed: reactive[bool] = reactive(False)
    is_loading: reactive[bool] = reactive(False)
    loading_message: reactive[str] = reactive("Processing...")
    
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
        self.model_preferences = ModelPreferencesManager()
        self.model_filter: str = "all"  # all, favorites, recent, most_used
        self.batch_mode_enabled: bool = False
        self.selected_models: set[str] = set()
        self.selected_collections: set[str] = set()
        self.activity_log: Optional[ActivityLogWidget] = None
        self.performance_metrics: Optional[PerformanceMetricsWidget] = None
        
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
                    
                    # Model filter dropdown
                    yield Select(
                        [("All Models", "all"), ("Favorites", "favorites"), 
                         ("Recent", "recent"), ("Most Used", "most_used")],
                        id="embeddings-model-filter",
                        value="all"
                    )
                    
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
                    
                    # Batch selection buttons for models (hidden by default)
                    with Horizontal(id="batch-model-controls", classes="batch-controls hidden"):
                        yield Button("Select All", id="select-all-models", classes="batch-button")
                        yield Button("Select None", id="select-none-models", classes="batch-button")
                        yield Button("Delete Selected", id="delete-selected-models", classes="batch-button", variant="error")
                    
                    # Batch selection buttons for collections (hidden by default)
                    with Horizontal(id="batch-collection-controls", classes="batch-controls hidden"):
                        yield Button("Select All", id="select-all-collections", classes="batch-button")
                        yield Button("Select None", id="select-none-collections", classes="batch-button")
                        yield Button("Delete Selected", id="delete-selected-collections", classes="batch-button", variant="error")
                    
                    # Refresh button
                    yield Button(
                        "Refresh Lists",
                        id="embeddings-refresh-lists",
                        classes="embeddings-action-button"
                    )
                    
                    # Toggle batch mode button
                    yield Button(
                        "Batch Mode",
                        id="toggle-batch-mode",
                        classes="embeddings-action-button"
                    )
            
            # Toggle button with better visibility
            yield Button(
                "◀ Hide Sidebar" if not self.left_pane_collapsed else "▶ Show Sidebar",
                id="toggle-embeddings-pane",
                classes="embeddings-toggle-button-enhanced"
            )
            
            # Right pane - Details and Actions
            with Container(classes="embeddings-right-pane", id="embeddings-right-pane"):
                # Loading overlay container
                with Container(id="embeddings-loading-overlay", classes="embeddings-loading-overlay hidden"):
                    yield LoadingIndicator(id="embeddings-loading-indicator")
                    yield Label("Processing...", id="embeddings-loading-label", classes="embeddings-loading-label")
                
                with VerticalScroll():
                    # Model information section - expanded by default
                    with Collapsible(title="Model Information", id="embeddings-model-info-collapsible", collapsed=False):
                        yield Label("Provider:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-model-provider", classes="embeddings-info-value")
                        
                        yield Label("Model Path:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-model-path", classes="embeddings-info-value")
                        
                        yield Label("Dimension:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-model-dimension", classes="embeddings-info-value")
                        
                        yield Label("Download Status:", classes="embeddings-info-label")
                        yield Static("Not Downloaded", id="embeddings-model-download-status", classes="embeddings-info-value")
                        
                        yield Label("Model Size:", classes="embeddings-info-label")
                        yield Static("Unknown", id="embeddings-model-size", classes="embeddings-info-value")
                        
                        yield Label("Memory Status:", classes="embeddings-info-label")
                        yield Static("Not Loaded", id="embeddings-model-status", classes="embeddings-info-value")
                        
                        yield Label("Cache Location:", classes="embeddings-info-label")
                        yield Static("", id="embeddings-model-cache-location", classes="embeddings-info-value")
                        
                        # Model actions
                        with Horizontal():
                            yield Button("⭐ Favorite", id="embeddings-favorite-model", classes="embeddings-action-button")
                            yield Button("Download", id="embeddings-download-model", classes="embeddings-action-button")
                            yield Button("Load Model", id="embeddings-load-model", classes="embeddings-action-button")
                            yield Button("Unload Model", id="embeddings-unload-model", classes="embeddings-action-button")
                            yield Button("Delete Model", id="embeddings-delete-model", classes="embeddings-action-button")
                    
                    # Collection information section - expanded by default
                    with Collapsible(title="Collection Information", id="embeddings-collection-info-collapsible", collapsed=False):
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
                    
                    # Testing section - expanded by default
                    with Collapsible(title="Test Embeddings", id="embeddings-test-collapsible", collapsed=False):
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
                    
                    # Performance metrics section - expanded by default
                    with Collapsible(title="Performance Metrics", id="embeddings-metrics-collapsible", collapsed=False):
                        yield PerformanceMetricsWidget(
                            id="embeddings-performance-metrics",
                            show_charts=True,
                            show_alerts=True,
                            compact=False
                        )
                    
                    # Activity log section
                    with Collapsible(title="Activity Log", id="embeddings-activity-collapsible", collapsed=False):
                        yield ActivityLogWidget(
                            id="embeddings-activity-log",
                            show_filters=True,
                            show_search=True,
                            show_actions=True,
                            max_entries=500
                        )
    
    async def on_mount(self) -> None:
        """Handle mount event - initialize embeddings components."""
        # Get activity log widget
        try:
            self.activity_log = self.query_one("#embeddings-activity-log", ActivityLogWidget)
        except:
            logger.warning("Activity log widget not found")
        
        # Get performance metrics widget
        try:
            self.performance_metrics = self.query_one("#embeddings-performance-metrics", PerformanceMetricsWidget)
        except:
            logger.warning("Performance metrics widget not found")
        
        await self._initialize_embeddings()
        await self._load_models_list()
        await self._load_collections_list()
        
        # Log initialization
        if self.activity_log:
            self.activity_log.log_info("Embeddings Management Window initialized", "system")
    
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
                
                # Filter models based on current filter
                display_models = self._filter_models(self.available_models)
                
                for model_id in display_models:
                    # Get model info from config
                    model_spec = config.models.get(model_id, {})
                    model_info = {
                        'provider': getattr(model_spec, 'provider', 'unknown'),
                        'is_downloaded': self._check_model_downloaded(model_id),
                        'is_loaded': model_id in getattr(self.embedding_factory, '_cache', {}),
                        'dimension': getattr(model_spec, 'dimension', None),
                        'is_favorite': self.model_preferences.is_favorite(model_id),
                        'usage_stats': self.model_preferences.get_model_stats(model_id)
                    }
                    
                    item = ModelListItem(
                        model_id,
                        model_info,
                        show_selection=self.batch_mode_enabled,
                        id=f"model-{model_id}"
                    )
                    await model_list.append(item)
                
                logger.info(f"Loaded {len(self.available_models)} models")
            
            # Show empty state if no models
            if not display_models:
                empty_state = ModelsEmptyState()
                await model_list.append(empty_state)
            
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
            
            # Show empty state if no collections
            if not self.collections:
                empty_state = CollectionsEmptyState()
                await collection_list.append(empty_state)
            else:
                for collection_name in self.collections:
                    # Create collection info (placeholder)
                    collection_info = {
                        'document_count': 0,
                        'last_modified': None,
                        'status': 'ready'
                    }
                    
                    item = CollectionListItem(
                        collection_name,
                        collection_info,
                        show_selection=self.batch_mode_enabled,
                        id=f"collection-{collection_name}"
                    )
                    await collection_list.append(item)
            
            logger.info("Collections list loaded")
            
        except Exception as e:
            logger.error(f"Failed to load collections list: {e}")
    
    def _filter_models(self, models: List[str]) -> List[str]:
        """Filter models based on current filter setting."""
        if self.model_filter == "all":
            return models
        elif self.model_filter == "favorites":
            favorites = self.model_preferences.get_favorite_models()
            return [m for m in models if m in favorites]
        elif self.model_filter == "recent":
            recent = self.model_preferences.get_recent_models(limit=10)
            return [m for m in models if m in recent]
        elif self.model_filter == "most_used":
            most_used = self.model_preferences.get_most_used_models(limit=10)
            most_used_ids = [m[0] for m in most_used]
            return [m for m in models if m in most_used_ids]
        return models
    
    # Event handlers
    @on(Select.Changed, "#embeddings-model-filter")
    async def on_filter_changed(self, event: Select.Changed) -> None:
        """Handle model filter change."""
        self.model_filter = event.value
        await self._load_models_list()
    
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
    
    def watch_left_pane_collapsed(self, is_collapsed: bool) -> None:
        """Update toggle button text when pane state changes."""
        try:
            toggle_button = self.query_one("#toggle-embeddings-pane", Button)
            toggle_button.label = "▶ Show Sidebar" if is_collapsed else "◀ Hide Sidebar"
        except Exception as e:
            logger.warning(f"Could not update toggle button text: {e}")
    
    def watch_is_loading(self, is_loading: bool) -> None:
        """Show/hide loading overlay when loading state changes."""
        try:
            overlay = self.query_one("#embeddings-loading-overlay")
            loading_label = self.query_one("#embeddings-loading-label", Label)
            
            if is_loading:
                overlay.remove_class("hidden")
                loading_label.update(self.loading_message)
            else:
                overlay.add_class("hidden")
        except Exception as e:
            logger.warning(f"Could not toggle loading overlay: {e}")
    
    @on(Button.Pressed, "#toggle-embeddings-pane")
    def on_toggle_pane(self, event: Button.Pressed) -> None:
        """Toggle the left pane visibility."""
        event.stop()  # Stop event propagation
        self.left_pane_collapsed = not self.left_pane_collapsed
        left_pane = self.query_one("#embeddings-left-pane")
        if self.left_pane_collapsed:
            left_pane.add_class("collapsed")
        else:
            left_pane.remove_class("collapsed")
    
    @on(Button.Pressed, "#embeddings-refresh-lists")
    async def on_refresh_lists(self, event: Button.Pressed) -> None:
        """Refresh both models and collections lists."""
        event.stop()  # Stop event propagation
        await self._load_models_list()
        await self._load_collections_list()
        self.notify("Lists refreshed", severity="information")
    
    @on(Button.Pressed, "#embeddings-load-model")
    async def on_load_model(self, event: Button.Pressed) -> None:
        """Load the selected model."""
        event.stop()  # Stop event propagation
        
        if not self.selected_model:
            self.notify("Please select a model first", severity="warning")
            return
        
        if not self.embedding_factory:
            self.notify("Embedding factory not initialized", severity="error")
            return
        
        try:
            self.loading_message = f"Loading model {self.selected_model}..."
            self.is_loading = True
            # Prefetch the model to load it into cache
            self.embedding_factory.prefetch([self.selected_model])
            # Record model usage
            self.model_preferences.record_model_use(self.selected_model)
            self.notify(f"Model {self.selected_model} loaded successfully", severity="information")
            
            # Log to activity log
            if self.activity_log:
                self.activity_log.log_success(f"Loaded model: {self.selected_model}", "models")
            
            await self._update_model_info(self.selected_model)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.notify(f"Failed to load model: {str(e)}", severity="error")
            if self.activity_log:
                self.activity_log.log_error(f"Failed to load model {self.selected_model}: {str(e)}", "models")
        finally:
            self.is_loading = False
    
    @on(Button.Pressed, "#embeddings-test-generate")
    async def on_test_generate(self, event: Button.Pressed) -> None:
        """Generate test embeddings."""
        event.stop()  # Stop event propagation
        
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
            self.loading_message = "Generating embedding..."
            self.is_loading = True
            
            # Track performance
            start_time = time.time()
            
            # Generate embedding
            embedding = await self.embedding_factory.async_embed(
                [text],
                model_id=self.selected_model,
                as_list=True
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Display result
            result_text = f"Model: {self.selected_model}\n"
            result_text += f"Dimension: {len(embedding[0]) if embedding else 0}\n"
            result_text += f"First 10 values: {embedding[0][:10] if embedding else []}\n"
            result_text += f"Norm: {sum(x**2 for x in embedding[0])**0.5 if embedding else 0:.4f}\n"
            result_text += f"Processing time: {processing_time_ms:.1f}ms"
            
            test_result.text = result_text
            
            # Update performance metrics
            if self.performance_metrics:
                self.performance_metrics.record_embedding_processed(1)
                self.performance_metrics.record_chunk_processed(processing_time_ms)
            
            # Log success
            if self.activity_log:
                self.activity_log.log_success(
                    f"Generated test embedding in {processing_time_ms:.1f}ms",
                    "embeddings"
                )
            
        except Exception as e:
            logger.error(f"Failed to generate test embedding: {e}")
            test_result.text = f"Error: {str(e)}"
        finally:
            self.is_loading = False
    
    @on(Button.Pressed, "#embeddings-download-model")
    async def on_download_model(self, event: Button.Pressed) -> None:
        """Download the selected model from HuggingFace."""
        event.stop()  # Stop event propagation to prevent app-level warning
        
        if not self.selected_model:
            self.notify("Please select a model first", severity="warning")
            return
        
        if not self.embedding_factory:
            self.notify("Embedding factory not initialized", severity="error")
            return
        
        try:
            config = self.embedding_factory.config
            if not config or not hasattr(config, 'models'):
                self.notify("No model configuration found", severity="error")
                return
            
            model_spec = config.models.get(self.selected_model)
            if not model_spec:
                self.notify("Model configuration not found", severity="error")
                return
            
            # Only download HuggingFace models
            if model_spec.provider != "huggingface":
                self.notify("Only HuggingFace models can be downloaded", severity="warning")
                return
            
            self.loading_message = f"Downloading model {self.selected_model}..."
            self.is_loading = True
            download_status = self.query_one("#embeddings-model-download-status", Static)
            download_status.update("Downloading...")
            
            # Just use the worker thread directly
            self.run_worker(
                self._download_model_worker,
                thread=True,
                name=f"download_{self.selected_model}"
            )
            
        except Exception as e:
            logger.error(f"Failed to start model download: {e}")
            self.notify(f"Failed to start download: {str(e)}", severity="error")
            self.is_loading = False
    
    @on(Button.Pressed, "#embeddings-unload-model")
    async def on_unload_model(self, event: Button.Pressed) -> None:
        """Unload the selected model from memory."""
        event.stop()  # Stop event propagation
        
        if not self.selected_model:
            self.notify("Please select a model first", severity="warning")
            return
        
        if not self.embedding_factory:
            self.notify("Embedding factory not initialized", severity="error")
            return
        
        try:
            # Check if model is in cache
            cache = getattr(self.embedding_factory, '_cache', {})
            if self.selected_model not in cache:
                self.notify("Model is not loaded", severity="warning")
                return
            
            self.loading_message = f"Unloading model {self.selected_model}..."
            self.is_loading = True
            
            # Remove from cache
            if hasattr(self.embedding_factory, '_cache') and self.selected_model in self.embedding_factory._cache:
                del self.embedding_factory._cache[self.selected_model]
                self.notify(f"Model {self.selected_model} unloaded from memory", severity="information")
                await self._update_model_info(self.selected_model)
            
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            self.notify(f"Failed to unload model: {str(e)}", severity="error")
        finally:
            self.is_loading = False
    
    @on(Button.Pressed, "#embeddings-delete-model")
    async def on_delete_model(self, event: Button.Pressed) -> None:
        """Delete the downloaded model files."""
        event.stop()  # Stop event propagation
        
        if not self.selected_model:
            self.notify("Please select a model first", severity="warning")
            return
        
        # Create a simple confirmation dialog
        class DeleteModelDialog(ModalScreen):
            """Modal dialog for confirming model deletion."""
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name
                super().__init__()
            
            def compose(self) -> ComposeResult:
                with Container(id="dialog-container"):
                    yield Label(f"Delete model '{self.model_name}'?", id="dialog-title")
                    yield Label("This will remove the downloaded model files.", id="dialog-message")
                    with Horizontal(id="dialog-buttons"):
                        yield Button("Cancel", id="cancel", variant="default")
                        yield Button("Delete", id="confirm", variant="error")
            
            @on(Button.Pressed)
            async def on_button_pressed(self, event: Button.Pressed) -> None:
                self.dismiss(event.button.id == "confirm")
        
        # Show confirmation dialog
        confirm = await self.app.push_screen(DeleteModelDialog(self.selected_model), wait_for_dismiss=True)
        
        if confirm:
            # TODO: Implement actual model deletion
            # Remove from preferences when deleted
            self.model_preferences.remove_model(self.selected_model)
            self.notify(f"Model deletion for '{self.selected_model}' not yet implemented", severity="information")
    
    
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
            
            # Update model path
            path_widget = self.query_one("#embeddings-model-path", Static)
            model_path = getattr(model_spec, 'model_name_or_path', 'Unknown')
            path_widget.update(str(model_path))
            
            # Update dimension
            dimension_widget = self.query_one("#embeddings-model-dimension", Static)
            dimension = getattr(model_spec, 'dimension', 'Unknown')
            dimension_widget.update(str(dimension))
            
            # Update download status
            download_widget = self.query_one("#embeddings-model-download-status", Static)
            if model_spec.provider == "huggingface":
                # Check if model exists in cache
                try:
                    from huggingface_hub import cached_download
                    cache_dir = getattr(model_spec, 'cache_dir', None)
                    if cache_dir:
                        cache_dir = os.path.expanduser(cache_dir)
                    # TODO: Properly check if model is downloaded
                    download_widget.update("Check Manually")
                except (ImportError, AttributeError, OSError) as e:
                    logger.warning(f"Failed to check model download status: {e}")
                    download_widget.update("Unknown")
            else:
                download_widget.update("N/A (Not HuggingFace)")
            
            # Update memory status
            status_widget = self.query_one("#embeddings-model-status", Static)
            # Check if model is in cache
            is_cached = model_id in getattr(self.embedding_factory, '_cache', {})
            status_widget.update("Loaded in Memory" if is_cached else "Not Loaded")
            
            # Update favorite button text
            favorite_button = self.query_one("#embeddings-favorite-model", Button)
            is_favorite = self.model_preferences.is_favorite(model_id)
            favorite_button.label = "★ Favorited" if is_favorite else "⭐ Favorite"
            
            # Update cache location
            cache_location_widget = self.query_one("#embeddings-model-cache-location", Static)
            cache_dir = getattr(model_spec, 'cache_dir', None)
            if cache_dir:
                cache_location_widget.update(os.path.expanduser(cache_dir))
            else:
                cache_location_widget.update("Default HuggingFace Cache")
        except Exception as e:
            logger.error(f"Failed to update model info: {e}")
    
    def _check_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded."""
        # This is a simplified check - actual implementation would
        # check the model cache directory
        return False
    
    async def _update_collection_info(self, collection_name: str) -> None:
        """Update the collection information display."""
        # This would be implemented when we have actual ChromaDB collection data
        pass
    
    def _show_error(self, message: str) -> None:
        """Show an error message in the UI."""
        # This could be implemented as a modal or notification
        self.notify(message, severity="error")
    
    async def _async_download_model(self) -> None:
        """Async method to download model."""
        import asyncio
        import os
        
        try:
            # Set environment variables to help with HuggingFace downloads
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            
            # Run the prefetch in a thread to avoid blocking
            # This will trigger HuggingFace's automatic download if needed
            await asyncio.to_thread(
                self.embedding_factory.prefetch, 
                [self.selected_model]
            )
            
            # Update UI on success
            download_status = self.query_one("#embeddings-model-download-status", Static)
            download_status.update("Download Complete")
            self.notify(f"Model {self.selected_model} downloaded successfully", severity="success")
            
            # Update model info
            await self._update_model_info(self.selected_model)
            
        except Exception as e:
            logger.error(f"Async download failed: {e}", exc_info=True)
            download_status = self.query_one("#embeddings-model-download-status", Static)
            download_status.update(f"Download Failed: {str(e)}")
            self.notify(f"Download failed: {str(e)}", severity="error")
            raise
        finally:
            self.is_loading = False
    
    def _download_model_worker(self) -> None:
        """Worker to download the selected model."""
        try:
            if not self.selected_model or not self.embedding_factory:
                return
            
            model_spec = self.embedding_factory.config.models.get(self.selected_model)
            if not model_spec or model_spec.provider != "huggingface":
                return
            
            # Update status before starting
            self.post_message(DownloadStatusMessage("Downloading..."))
            
            # Just call prefetch - the file descriptor protection is now handled
            # inside the EmbeddingFactory when it loads models
            logger.info(f"Downloading model: {self.selected_model}")
            self.embedding_factory.prefetch([self.selected_model])
            
            # Post message to update UI
            self.post_message(DownloadStatusMessage("Download Complete"))
            
            logger.info(f"Model {self.selected_model} downloaded successfully")
            
            # Log to activity log
            if self.activity_log:
                self.call_from_thread(
                    self.activity_log.log_success,
                    f"Downloaded model: {self.selected_model}",
                    "models"
                )
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}", exc_info=True)
            # Post message to update UI
            self.post_message(DownloadStatusMessage(f"Download Failed: {str(e)}"))
        finally:
            # Post message to set loading state
            self.post_message(SetLoadingMessage(False))
    
    def _update_download_status(self, status: str) -> None:
        """Update the download status in the UI."""
        try:
            download_status = self.query_one("#embeddings-model-download-status", Static)
            download_status.update(status)
        except Exception as e:
            logger.error(f"Failed to update download status: {e}")
    
    def _set_loading(self, loading: bool) -> None:
        """Set the loading state."""
        self.is_loading = loading
    
    @on(DownloadStatusMessage)
    def handle_download_status(self, message: DownloadStatusMessage) -> None:
        """Handle download status message from worker."""
        self._update_download_status(message.status)
    
    @on(SetLoadingMessage)
    def handle_set_loading(self, message: SetLoadingMessage) -> None:
        """Handle set loading message from worker."""
        self._set_loading(message.loading)
    
    @on(Button.Pressed, "#embeddings-delete-collection")
    async def on_delete_collection(self, event: Button.Pressed) -> None:
        """Delete the selected collection."""
        event.stop()  # Stop event propagation
        
        if not self.selected_collection:
            self.notify("Please select a collection first", severity="warning")
            return
        
        # Show confirmation dialog using our consistent DeleteConfirmationDialog
        from ..Widgets.delete_confirmation_dialog import create_delete_confirmation
        dialog = create_delete_confirmation(
            item_type="Collection",
            item_name=self.selected_collection,
            additional_warning="All embeddings in this collection will be permanently deleted.",
            permanent=True
        )
        
        confirm = await self.app.push_screen_wait(dialog)
        
        if confirm:
            # TODO: Implement actual collection deletion
            self.notify(f"Collection deletion for '{self.selected_collection}' not yet implemented", severity="information")
    
    @on(Button.Pressed, "#embeddings-favorite-model")
    async def on_favorite_model(self, event: Button.Pressed) -> None:
        """Toggle favorite status for the selected model."""
        event.stop()  # Stop event propagation
        
        if not self.selected_model:
            self.notify("Please select a model first", severity="warning")
            return
        
        # Toggle favorite status
        is_favorite = self.model_preferences.toggle_favorite(self.selected_model)
        
        # Update button text
        favorite_button = self.query_one("#embeddings-favorite-model", Button)
        favorite_button.label = "★ Favorited" if is_favorite else "⭐ Favorite"
        
        # Notify user
        if is_favorite:
            self.notify(f"Added {self.selected_model} to favorites", severity="success")
        else:
            self.notify(f"Removed {self.selected_model} from favorites", severity="information")
        
        # Refresh list if we're viewing favorites
        if self.model_filter == "favorites":
            await self._load_models_list()
    
    @on(Button.Pressed, "#toggle-batch-mode")
    async def on_toggle_batch_mode(self, event: Button.Pressed) -> None:
        """Toggle batch mode for selection."""
        event.stop()
        
        self.batch_mode_enabled = not self.batch_mode_enabled
        
        # Update button text
        toggle_button = self.query_one("#toggle-batch-mode", Button)
        toggle_button.label = "Exit Batch Mode" if self.batch_mode_enabled else "Batch Mode"
        
        # Show/hide batch controls
        model_controls = self.query_one("#batch-model-controls")
        collection_controls = self.query_one("#batch-collection-controls")
        
        if self.batch_mode_enabled:
            model_controls.remove_class("hidden")
            collection_controls.remove_class("hidden")
        else:
            model_controls.add_class("hidden")
            collection_controls.add_class("hidden")
            # Clear selections
            self.selected_models.clear()
            self.selected_collections.clear()
        
        # Reload lists to show/hide checkboxes
        await self._load_models_list()
        await self._load_collections_list()
    
    @on(Button.Pressed, "#select-all-models")
    async def on_select_all_models(self, event: Button.Pressed) -> None:
        """Select all models."""
        event.stop()
        
        model_list = self.query_one("#embeddings-model-list", ListView)
        for item in model_list.children:
            if isinstance(item, ModelListItem):
                item.is_selected = True
                self.selected_models.add(item.model_id)
    
    @on(Button.Pressed, "#select-none-models")
    async def on_select_none_models(self, event: Button.Pressed) -> None:
        """Deselect all models."""
        event.stop()
        
        model_list = self.query_one("#embeddings-model-list", ListView)
        for item in model_list.children:
            if isinstance(item, ModelListItem):
                item.is_selected = False
        self.selected_models.clear()
    
    @on(Button.Pressed, "#delete-selected-models")
    async def on_delete_selected_models(self, event: Button.Pressed) -> None:
        """Delete selected models."""
        event.stop()
        
        if not self.selected_models:
            self.notify("No models selected", severity="warning")
            return
        
        # Create confirmation dialog
        class BatchDeleteDialog(ModalScreen):
            """Modal dialog for confirming batch deletion."""
            def __init__(self, count: int) -> None:
                self.count = count
                super().__init__()
            
            def compose(self) -> ComposeResult:
                with Container(id="dialog-container"):
                    yield Label(f"Delete {self.count} models?", id="dialog-title")
                    yield Label("This will remove the downloaded model files.", id="dialog-message")
                    with Horizontal(id="dialog-buttons"):
                        yield Button("Cancel", id="cancel", variant="default")
                        yield Button("Delete", id="confirm", variant="error")
            
            @on(Button.Pressed)
            async def on_button_pressed(self, event: Button.Pressed) -> None:
                self.dismiss(event.button.id == "confirm")
        
        # Show confirmation dialog
        confirm = await self.app.push_screen(BatchDeleteDialog(len(self.selected_models)), wait_for_dismiss=True)
        
        if confirm:
            # TODO: Implement batch model deletion
            self.notify(f"Batch deletion of {len(self.selected_models)} models not yet implemented", severity="information")
            # Clear selections after operation
            self.selected_models.clear()
            await self._load_models_list()
    
    @on(Button.Pressed, "#select-all-collections")
    async def on_select_all_collections(self, event: Button.Pressed) -> None:
        """Select all collections."""
        event.stop()
        
        collection_list = self.query_one("#embeddings-collection-list", ListView)
        for item in collection_list.children:
            if isinstance(item, CollectionListItem):
                item.is_selected = True
                self.selected_collections.add(item.collection_name)
    
    @on(Button.Pressed, "#select-none-collections")
    async def on_select_none_collections(self, event: Button.Pressed) -> None:
        """Deselect all collections."""
        event.stop()
        
        collection_list = self.query_one("#embeddings-collection-list", ListView)
        for item in collection_list.children:
            if isinstance(item, CollectionListItem):
                item.is_selected = False
        self.selected_collections.clear()
    
    @on(Button.Pressed, "#delete-selected-collections")
    async def on_delete_selected_collections(self, event: Button.Pressed) -> None:
        """Delete selected collections."""
        event.stop()
        
        if not self.selected_collections:
            self.notify("No collections selected", severity="warning")
            return
        
        # Show confirmation dialog using our consistent DeleteConfirmationDialog
        from ..Widgets.delete_confirmation_dialog import create_delete_confirmation
        dialog = create_delete_confirmation(
            item_type="Collections",
            item_name=f"{len(self.selected_collections)} selected collections",
            additional_warning="All embeddings in these collections will be permanently deleted.",
            permanent=True
        )
        
        confirm = await self.app.push_screen_wait(dialog)
        
        if confirm:
            # TODO: Implement batch collection deletion
            self.notify(f"Batch deletion of {len(self.selected_collections)} collections not yet implemented", severity="information")
            # Clear selections after operation
            self.selected_collections.clear()
            await self._load_collections_list()
    
    @on(Checkbox.Changed)
    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes in list items."""
        checkbox_id = event.checkbox.id
        
        if checkbox_id and checkbox_id.startswith("select-"):
            # Extract the item type and id
            parts = checkbox_id.split("-", 2)
            if len(parts) >= 3:
                item_type = parts[1]
                item_id = parts[2]
                
                if item_type == "model":
                    if event.value:
                        self.selected_models.add(item_id)
                    else:
                        self.selected_models.discard(item_id)
                elif item_type == "collection":
                    if event.value:
                        self.selected_collections.add(item_id)
                    else:
                        self.selected_collections.discard(item_id)

# End of Embeddings_Management_Window.py
########################################################################################################################