# Embeddings_Management_Window.py
# Description: Embeddings Management interface with dual-pane layout
#
# Imports
from __future__ import annotations

from typing import Optional, List, Dict, Any
from loguru import logger
import os

# Third-party imports
from textual import events, on, work
from textual.app import ComposeResult
from textual.message import Message
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
                            yield Button("Download", id="embeddings-download-model", classes="embeddings-action-button")
                            yield Button("Load Model", id="embeddings-load-model", classes="embeddings-action-button")
                            yield Button("Unload Model", id="embeddings-unload-model", classes="embeddings-action-button")
                            yield Button("Delete Model", id="embeddings-delete-model", classes="embeddings-action-button")
                    
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
            
            # Remove from cache
            if hasattr(self.embedding_factory, '_cache') and self.selected_model in self.embedding_factory._cache:
                del self.embedding_factory._cache[self.selected_model]
                self.notify(f"Model {self.selected_model} unloaded from memory", severity="information")
                await self._update_model_info(self.selected_model)
            
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            self.notify(f"Failed to unload model: {str(e)}", severity="error")
    
    @on(Button.Pressed, "#embeddings-delete-model")
    async def on_delete_model(self, event: Button.Pressed) -> None:
        """Delete the downloaded model files."""
        event.stop()  # Stop event propagation
        
        if not self.selected_model:
            self.notify("Please select a model first", severity="warning")
            return
        
        # TODO: Implement model deletion from cache directory
        self.notify("Model deletion not yet implemented", severity="information")
    
    
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
                    import os
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
            
            # Update cache location
            cache_location_widget = self.query_one("#embeddings-model-cache-location", Static)
            cache_dir = getattr(model_spec, 'cache_dir', None)
            if cache_dir:
                cache_location_widget.update(os.path.expanduser(cache_dir))
            else:
                cache_location_widget.update("Default HuggingFace Cache")
            
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

# End of Embeddings_Management_Window.py
########################################################################################################################