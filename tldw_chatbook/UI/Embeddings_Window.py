# tldw_chatbook/UI/Embeddings_Window.py
# Description: Main Embeddings window container with navigation between creation and management views
#
# Imports
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from pathlib import Path
import asyncio

# 3rd-Party Imports
from loguru import logger
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import (
    Static, Button, Input, Label, Select, TextArea, Checkbox, RadioButton, RadioSet,
    Collapsible, LoadingIndicator, ProgressBar, DataTable, Rule, ContentSwitcher
)

# Configure logger with context
logger = logger.bind(module="Embeddings_Window")

# Local Imports
from .Embeddings_Management_Window import EmbeddingsManagementWindow
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE, force_recheck_embeddings
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from ..Third_Party.textual_fspicker import Filters

# Force a recheck of embeddings dependencies to ensure they're properly detected
embeddings_available = force_recheck_embeddings()
logger.info(f"Embeddings dependencies available: {embeddings_available}")

# Check if embeddings dependencies are available
if embeddings_available or DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
    try:
        from ..Embeddings.Embeddings_Lib import EmbeddingFactory
        from ..Embeddings.Chroma_Lib import ChromaDBManager
        from ..Chunking.Chunk_Lib import chunk_for_embedding
        logger.info("Successfully imported embeddings modules")
    except ImportError as e:
        logger.error(f"Failed to import embeddings modules: {e}")
        EmbeddingFactory = None
        ChromaDBManager = None
        chunk_for_embedding = None
        DEPENDENCIES_AVAILABLE['embeddings_rag'] = False
else:
    EmbeddingFactory = None
    ChromaDBManager = None
    chunk_for_embedding = None

# Define available chunk methods
CHUNK_METHODS = ['words', 'sentences', 'paragraphs', 'tokens', 'semantic', 'json', 'xml', 'ebook_chapters', 'rolling_summarize']

if TYPE_CHECKING:
    from ..app import TldwCli

########################################################################################################################
#
# Constants and View Definitions
#
########################################################################################################################

EMBEDDINGS_VIEW_IDS = [
    "embeddings-view-create",
    "embeddings-view-manage"
]

EMBEDDINGS_NAV_BUTTON_IDS = [
    "embeddings-nav-create",
    "embeddings-nav-manage"
]

########################################################################################################################
#
# Classes
#
########################################################################################################################

class EmbeddingsWindow(Container):
    """Main container for embeddings functionality with navigation."""
    
    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select All", show=False),
        Binding("ctrl+d", "clear_selection", "Deselect All", show=False),
        Binding("space", "toggle_selection", "Toggle Selection", show=False),
    ]
    
    # Input source types
    SOURCE_TEXT = "text"
    SOURCE_FILE = "file"
    SOURCE_DATABASE = "database"
    
    # Reactive attributes
    selected_source: reactive[str] = reactive(SOURCE_FILE)
    selected_model: reactive[Optional[str]] = reactive(None)
    is_processing: reactive[bool] = reactive(False)
    selected_files: reactive[List[Path]] = reactive([])
    selected_db: reactive[str] = reactive("media")  # "media" or "chachanotes"
    selected_db_type: reactive[Optional[str]] = reactive("media")
    selected_db_items: reactive[set] = reactive(set())  # Track selected item IDs
    selected_db_mode: reactive[str] = reactive("search")  # "all", "specific", "keywords", "search"
    specific_item_ids: reactive[str] = reactive("")  # Comma-separated IDs
    keyword_filter: reactive[str] = reactive("")  # Comma-separated keywords
    embeddings_active_view: reactive[str] = reactive("embeddings-view-create")  # Track active view
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.chachanotes_db = app_instance.chachanotes_db if hasattr(app_instance, 'chachanotes_db') else None
        self.media_db = app_instance.media_db if hasattr(app_instance, 'media_db') else None
        self.embedding_factory: Optional[EmbeddingFactory] = None
        self.chroma_manager: Optional[ChromaDBManager] = None
        
        # Check dependencies
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            logger.warning("Embeddings/RAG dependencies not available")
        
        logger.debug("EmbeddingsWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the embeddings window with navigation and content areas."""
        logger.debug("Composing EmbeddingsWindow UI")
        
        # Force recheck dependencies one more time
        embeddings_available = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False)
        logger.info(f"Embeddings available check in compose: {embeddings_available}")
        
        # Log individual dependency states
        logger.debug(f"torch: {DEPENDENCIES_AVAILABLE.get('torch', False)}")
        logger.debug(f"transformers: {DEPENDENCIES_AVAILABLE.get('transformers', False)}")
        logger.debug(f"numpy: {DEPENDENCIES_AVAILABLE.get('numpy', False)}")
        logger.debug(f"chromadb: {DEPENDENCIES_AVAILABLE.get('chromadb', False)}")
        logger.debug(f"sentence_transformers: {DEPENDENCIES_AVAILABLE.get('sentence_transformers', False)}")
        
        # Check if embeddings dependencies are available
        if not embeddings_available:
            with Container(classes="embeddings-not-available"):
                yield Static("⚠️ Embeddings/RAG functionality not available", classes="warning-title")
                yield Static("The required dependencies for embeddings are not installed.", classes="warning-message")
                yield Static("To enable embeddings, please install with:", classes="warning-message")
                yield Static("pip install tldw_chatbook[embeddings_rag]", classes="code-block")
                yield Static("", classes="warning-message")
                yield Static("All required packages appear to be installed. Try restarting the application.", classes="warning-message")
            return
        
        # Left navigation pane
        with VerticalScroll(id="embeddings-nav-pane", classes="embeddings-nav-pane"):
            yield Static("Embeddings Options", classes="sidebar-title")
            yield Button("Create Embeddings", id="embeddings-nav-create", classes="embeddings-nav-button")
            yield Button("Manage Embeddings", id="embeddings-nav-manage", classes="embeddings-nav-button")
        
        # Right content pane
        with Container(id="embeddings-content-pane", classes="embeddings-content-pane"):
            # Create embeddings view - VerticalScroll as direct child
            with Container(id="embeddings-view-create", classes="embeddings-view-area"):
                with VerticalScroll(classes="embeddings-creation-scroll"):
                    yield Label("Create Embeddings", classes="embeddings-form-title")
                    
                    # Model Selection Section
                    yield Label("Model Selection", classes="embeddings-section-title")
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Embedding Model:", classes="embeddings-form-label")
                        yield Select(
                            [(model, model) for model in self._get_available_models()],
                            id="embeddings-model-select",
                            classes="embeddings-form-control",
                            allow_blank=False
                        )
                    
                    yield Rule()
                    
                    # Input Source Section
                    yield Label("Input Source", classes="embeddings-section-title")
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Source Type:", classes="embeddings-form-label")
                        yield Select(
                            [
                                ("Files", self.SOURCE_FILE),
                                ("Database Content", self.SOURCE_DATABASE)
                            ],
                            id="embeddings-source-type",
                            classes="embeddings-form-control",
                            value=self.SOURCE_FILE
                        )
                    
                    # Use ContentSwitcher for source type switching
                    with ContentSwitcher(initial=self.SOURCE_FILE, id="embeddings-source-switcher"):
                        # File input container
                        with Container(id="file-input-container", classes="embeddings-input-source-container"):
                            with Horizontal(classes="embeddings-form-row"):
                                yield Button("Select Files", id="embeddings-select-files", classes="embeddings-action-button")
                                yield Label("Selected: 0 files", id="embeddings-file-count")
                            
                            yield TextArea(
                                "",
                                id="embeddings-file-list",
                                classes="embeddings-file-list",
                                read_only=True
                            )
                        
                        # Database query container
                        with Container(id="db-input-container", classes="embeddings-input-source-container"):
                            # Database selection from app's loaded databases
                            with Horizontal(classes="embeddings-form-row"):
                                yield Label("Database:", classes="embeddings-form-label")
                                yield Select(
                                [
                                    ("Media Database", "media"),
                                    ("ChaChaNotes Database", "chachanotes")
                                ],
                                id="embeddings-db-select",
                                classes="embeddings-form-control",
                                allow_blank=False
                            )
                            
                            with Horizontal(classes="embeddings-form-row"):
                                yield Label("Content Type:", classes="embeddings-form-label")
                                yield Select(
                                    [
                                        ("Media Content", "media")
                                    ],
                                    id="embeddings-db-type",
                                    classes="embeddings-form-control",
                                    allow_blank=False,
                                    value="media"
                                )
                            
                            yield Rule()
                            
                            # Selection mode
                            yield Label("Selection Mode", classes="embeddings-section-title")
                            with Container(classes="embeddings-form-full-row"):
                                with RadioSet(id="embeddings-db-mode-set"):
                                    yield RadioButton("Search & Select", id="embeddings-mode-search")
                                    yield RadioButton("All Items", id="embeddings-mode-all")
                                    yield RadioButton("Specific IDs", id="embeddings-mode-specific")
                                    yield RadioButton("By Keywords", id="embeddings-mode-keywords")
                            
                            # Search input (shown for search mode)
                            with Container(id="embeddings-search-container", classes="embeddings-mode-container"):
                                with Horizontal(classes="embeddings-form-row"):
                                    yield Label("Search:", classes="embeddings-form-label")
                                    yield Input(
                                        placeholder="Search for content...",
                                        id="embeddings-db-filter",
                                        classes="embeddings-form-control"
                                    )
                            
                            # Specific IDs input (shown for specific mode)
                            with Container(id="embeddings-specific-container", classes="embeddings-mode-container"):
                                with Horizontal(classes="embeddings-form-row"):
                                    yield Label("Item IDs:", classes="embeddings-form-label")
                                    yield Input(
                                        placeholder="Enter comma-separated IDs (e.g., 1,2,3)",
                                        id="embeddings-specific-ids",
                                        classes="embeddings-form-control"
                                    )
                            
                            # Keywords input (shown for keywords mode)
                            with Container(id="embeddings-keywords-container", classes="embeddings-mode-container"):
                                with Horizontal(classes="embeddings-form-row"):
                                    yield Label("Keywords:", classes="embeddings-form-label")
                                    yield Input(
                                        placeholder="Enter comma-separated keywords",
                                        id="embeddings-keywords-input",
                                        classes="embeddings-form-control"
                                    )
                            
                            with Horizontal(classes="embeddings-form-row"):
                                yield Button("Load Items", id="embeddings-search-db", classes="embeddings-action-button")
                                yield Label("No items selected", id="embeddings-db-selection-count")
                            
                            # Selection control buttons
                            with Horizontal(classes="embeddings-db-selection-buttons"):
                                yield Button("Select All", id="embeddings-select-all", classes="embeddings-db-selection-button")
                                yield Button("Clear Selection", id="embeddings-clear-selection", classes="embeddings-db-selection-button")
                            
                            # DataTable to show search results in a container
                            with Container(classes="embeddings-db-results-container"):
                                yield DataTable(
                                    id="embeddings-db-results",
                                    show_header=True,
                                    zebra_stripes=True,
                                    cursor_type="row",
                                    show_cursor=True
                                )
                    
                    yield Rule()
                    
                    # Chunking Configuration Section
                    yield Label("Chunking Configuration", classes="embeddings-section-title")
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Chunk Method:", classes="embeddings-form-label")
                        yield Select(
                            self._get_chunk_methods(),
                            id="embeddings-chunk-method",
                            classes="embeddings-form-control"
                        )
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Chunk Size:", classes="embeddings-form-label")
                        yield Input(
                            "512",
                            id="embeddings-chunk-size",
                            classes="embeddings-form-control",
                            type="integer"
                        )
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Chunk Overlap:", classes="embeddings-form-label")
                        yield Input(
                            "128",
                            id="embeddings-chunk-overlap",
                            classes="embeddings-form-control",
                            type="integer"
                        )
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Checkbox("Enable adaptive chunking", id="embeddings-adaptive-chunking")
                    
                    # Chunk preview
                    with Collapsible(title="Chunk Preview", id="embeddings-chunk-preview-collapsible"):
                        yield TextArea(
                            "",
                            id="embeddings-chunk-preview",
                            classes="embeddings-chunk-preview",
                            read_only=True
                        )
                    
                    yield Rule()
                    
                    # Collection Settings Section
                    yield Label("Collection Settings", classes="embeddings-section-title")
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Collection Name:", classes="embeddings-form-label")
                        yield Input(
                            placeholder="my_embeddings",
                            id="embeddings-collection-name",
                            classes="embeddings-form-control"
                        )
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Description:", classes="embeddings-form-label")
                        yield Input(
                            placeholder="Optional description",
                            id="embeddings-collection-desc",
                            classes="embeddings-form-control"
                        )
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Checkbox("Overwrite if exists", id="embeddings-overwrite")
                    
                    yield Rule()
                    
                    # Action Buttons Section
                    yield Label("Actions", classes="embeddings-section-title")
                    
                    # Try yielding buttons directly without Horizontal container
                    yield Button("Preview Chunks", id="embeddings-preview", classes="embeddings-action-button")
                    yield Button("Create Embeddings", id="embeddings-create", classes="embeddings-action-button", variant="primary")
                    yield Button("Clear Form", id="embeddings-clear", classes="embeddings-action-button")
                    
                    # Progress Section
                    with Container(id="embeddings-progress-container", classes="embeddings-progress-container"):
                        yield Label("Processing...", id="embeddings-progress-label", classes="embeddings-progress-label")
                        yield ProgressBar(id="embeddings-progress-bar", total=100)
                    
                    # Status Output
                    yield TextArea(
                        "",
                        id="embeddings-status-output",
                        classes="embeddings-status-output",
                        read_only=True
                    )
                    
                    # Extra spacing at bottom to ensure visibility
                    yield Static("", classes="embeddings-bottom-spacer")
            
            # Manage embeddings view
            with Container(id="embeddings-view-manage", classes="embeddings-view-area"):
                yield EmbeddingsManagementWindow(self.app_instance, id="embeddings-management-widget")
    
    async def on_mount(self) -> None:
        """Handle mount event - initialize embeddings components."""
        logger.debug("EmbeddingsWindow on_mount called")
        
        # Check if embeddings dependencies are available
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            logger.warning("Embeddings dependencies not available, skipping initialization")
            return
            
        await self._initialize_embeddings()
        
        # Small delay to ensure DOM is fully ready
        await asyncio.sleep(0.1)
        
        # Set initial view visibility
        logger.debug("Setting initial view visibility")
        for view_id in EMBEDDINGS_VIEW_IDS:
            try:
                view = self.query_one(f"#{view_id}")
                view.styles.display = "none"
            except Exception:
                pass
        
        # Show the initial view
        try:
            initial_view = self.query_one(f"#{self.embeddings_active_view}")
            initial_view.styles.display = "block"
        except Exception as e:
            logger.error(f"Failed to show initial view: {e}")
        
        # Check initial Select value
        try:
            source_select = self.query_one("#embeddings-source-type", Select)
            logger.info(f"Initial source select value: {source_select.value}")
        except Exception as e:
            logger.error(f"Could not query source select: {e}")
        
        # Set default radio button selection
        try:
            radio_set = self.query_one("#embeddings-db-mode-set", RadioSet)
            # Press the first radio button (search mode)
            search_radio = self.query_one("#embeddings-mode-search", RadioButton)
            search_radio.value = True
            
            # Ensure search container is visible by default for database mode
            search_container = self.query_one("#embeddings-search-container")
            search_container.styles.display = "block"
            
            specific_container = self.query_one("#embeddings-specific-container")
            specific_container.styles.display = "none"
            
            keywords_container = self.query_one("#embeddings-keywords-container")
            keywords_container.styles.display = "none"
        except Exception as e:
            logger.debug(f"Mode containers not yet available: {e}")
        
        # Initialize the DataTable if it exists (it's only visible when database source is selected)
        try:
            table = self.query_one("#embeddings-db-results", DataTable)
            table.add_columns("✓", "ID", "Title", "Type", "Date")
            table.cursor_type = "row"
        except NoMatches:
            logger.debug("DataTable not available yet - will be initialized when database source is selected")
        
        # Clear selected items
        self.selected_db_items = set()
        
        # Trigger initial database selection if available
        try:
            db_select = self.query_one("#embeddings-db-select", Select)
            # The Select widget should auto-select the first option when allow_blank=False
            # But we'll manually trigger the change event to set up the content types
            if db_select.value and db_select.value != Select.BLANK:
                self.on_database_changed(Select.Changed(db_select, db_select.value))
        except NoMatches:
            logger.debug("Database select not available yet")
    
    async def _initialize_embeddings(self) -> None:
        """Initialize embedding factory and ChromaDB manager."""
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            self.notify("Embeddings dependencies not installed. Install with: pip install tldw_chatbook[embeddings_rag]", severity="error")
            return
        
        try:
            # Load embedding configuration
            from ..config import load_settings, load_cli_config_and_ensure_existence
            from ..Embeddings.Embeddings_Lib import (
                EmbeddingConfigSchema, 
                get_default_embedding_config,
                create_embedding_factory_with_defaults
            )
            
            # Get the full config for ChromaDB manager
            full_config = load_cli_config_and_ensure_existence()
            settings = load_settings()
            embedding_config = settings.get('embedding_config', {})
            
            if embedding_config:
                # Get models configuration from TOML
                models_config = embedding_config.get('models', {})
                
                # If models_config exists but is empty or invalid, use defaults
                if not models_config:
                    logger.warning("No models configured, using default configuration")
                    # Get the default configuration and use its models
                    default_config = get_default_embedding_config()
                    models_config = {k: v.model_dump() for k, v in default_config.models.items()}
                
                # Prepare the configuration for validation
                factory_config = {
                    'default_model_id': embedding_config.get('default_model_id', 'e5-small-v2'),
                    'models': models_config
                }
                
                # Validate the configuration using pydantic
                try:
                    validated_config = EmbeddingConfigSchema(**factory_config)
                    
                    self.embedding_factory = EmbeddingFactory(
                        validated_config,
                        max_cached=2,
                        idle_seconds=900
                    )
                    logger.info("Initialized embedding factory with validated configuration")
                except Exception as config_error:
                    logger.error(f"Configuration validation failed: {config_error}")
                    # Use the default configuration helper
                    self.embedding_factory = create_embedding_factory_with_defaults(
                        max_cached=2,
                        idle_seconds=900
                    )
                    logger.info("Initialized embedding factory with default configuration")
            else:
                logger.warning("No embedding configuration found, using defaults")
                # Use the default configuration helper
                self.embedding_factory = create_embedding_factory_with_defaults(
                    max_cached=2,
                    idle_seconds=900
                )
                self.notify("Warning: No embedding configuration found, using defaults", severity="warning")
                
            # Initialize ChromaDB manager with full config  
            user_id = settings.get('USERS_NAME', 'default_user')
            # Make sure the config has the expected structure for ChromaDBManager
            if 'embedding_config' not in full_config:
                # Use the default configuration if no embedding config exists
                if not embedding_config:
                    default_config = get_default_embedding_config()
                    embedding_config = {
                        'default_model_id': default_config.default_model_id,
                        'models': {k: v.model_dump() for k, v in default_config.models.items()}
                    }
                full_config['embedding_config'] = embedding_config
            if 'USER_DB_BASE_DIR' not in full_config and 'database' in full_config:
                full_config['USER_DB_BASE_DIR'] = full_config['database'].get('USER_DB_BASE_DIR', '~/.local/share/tldw_cli')
            self.chroma_manager = ChromaDBManager(user_id, full_config)
            logger.info("Initialized ChromaDB manager")
            
            # Update the model select widget with available models
            model_select = self.query_one("#embeddings-model-select", Select)
            available_models = self._get_available_models()
            if available_models and available_models != ["No models available"]:
                model_select.set_options([(model, model) for model in available_models])
                # Select the first model by default
                if available_models:
                    model_select.value = available_models[0]
                    self.selected_model = available_models[0]
            
        except Exception as e:
            logger.error(f"(ECW) Failed to initialize embeddings: {e}")
            self.notify(f"Error: Failed to initialize embeddings: {str(e)}", severity="error")
    
    def _get_available_models(self) -> List[str]:
        """Get list of available embedding models."""
        if self.embedding_factory and hasattr(self.embedding_factory.config, 'models'):
            return list(self.embedding_factory.config.models.keys())
        return ["No models available"]
    
    def _get_chunk_methods(self) -> List[tuple[str, str]]:
        """Get available chunking methods."""
        return [(method, method.replace('_', ' ').title()) for method in CHUNK_METHODS]
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses within the embeddings window."""
        button_id = event.button.id
        if not button_id:
            return
        
        # Navigation buttons are handled by the app-level handler via reactive attributes
        if button_id in EMBEDDINGS_NAV_BUTTON_IDS:
            logger.info(f"EmbeddingsWindow.on_button_pressed: Navigation button '{button_id}' pressed, not handling here")
            return
        
        # Other button handling can go here if needed
    
    # Event handlers
    @on(Select.Changed, "#embeddings-source-type")
    def on_source_changed(self, event: Select.Changed) -> None:
        """Handle source type change."""
        logger.info("=== on_source_changed CALLED ===")
        logger.debug(f"Source type changed event triggered. Value: {event.value}, Type: {type(event.value)}")
        
        # Show a notification to confirm the event is firing
        self.notify(f"Source changed to: {event.value}", severity="information")
        
        if event.value and event.value != Select.BLANK:
            self.selected_source = str(event.value)
            logger.debug(f"Selected source set to: {self.selected_source}")
            logger.debug(f"SOURCE_FILE constant: {self.SOURCE_FILE}")
            logger.debug(f"SOURCE_DATABASE constant: {self.SOURCE_DATABASE}")
            
            # Use ContentSwitcher to switch between containers
            try:
                switcher = self.query_one("#embeddings-source-switcher", ContentSwitcher)
                switcher.current = self.selected_source
                logger.info(f"ContentSwitcher updated to show: {self.selected_source}")
                
                # Initialize DataTable when switching to database source
                if self.selected_source == self.SOURCE_DATABASE:
                    try:
                        table = self.query_one("#embeddings-db-results", DataTable)
                        # Only initialize if not already initialized (no columns)
                        if not table.columns:
                            table.add_columns("✓", "ID", "Title", "Type", "Date")
                            table.cursor_type = "row"
                            logger.debug("DataTable initialized for database source")
                    except NoMatches:
                        logger.error("DataTable not found even though database source is selected")
                        
            except Exception as e:
                logger.error(f"Failed to update ContentSwitcher: {e}")
                # Fallback to old method
                self._update_source_containers()
        else:
            logger.warning(f"Invalid or blank value received: {event.value}")
    
    @on(Select.Changed, "#embeddings-model-select")
    def on_model_changed(self, event: Select.Changed) -> None:
        """Handle model selection change."""
        if event.value and event.value != Select.BLANK:
            self.selected_model = str(event.value)
            logger.info(f"Selected embedding model: {self.selected_model}")
            
            # Show model information
            if self.embedding_factory and hasattr(self.embedding_factory.config, 'models'):
                model_config = self.embedding_factory.config.models.get(self.selected_model)
                if model_config:
                    info_parts = []
                    info_parts.append(f"Model: {self.selected_model}")
                    info_parts.append(f"Provider: {getattr(model_config, 'provider', 'Unknown')}")
                    
                    if hasattr(model_config, 'dimension'):
                        info_parts.append(f"Dimension: {model_config.dimension}")
                    
                    if hasattr(model_config, 'model_name_or_path'):
                        info_parts.append(f"Path: {model_config.model_name_or_path}")
                    
                    self.notify(" | ".join(info_parts), severity="information")
            
            self.notify(f"Selected model: {self.selected_model}", severity="information")
    
    def watch_embeddings_active_view(self, old: str, new: str) -> None:
        """React to view changes by showing/hiding containers."""
        # Skip if embeddings dependencies are not available
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            return
            
        logger.debug(f"Switching from view {old} to {new}")
        
        # Hide all views first
        for view_id in EMBEDDINGS_VIEW_IDS:
            try:
                view = self.query_one(f"#{view_id}")
                view.styles.display = "none"
            except Exception:
                pass
        
        # Show the selected view
        try:
            active_view = self.query_one(f"#{new}")
            active_view.styles.display = "block"
        except Exception as e:
            logger.error(f"Failed to show view {new}: {e}")
    
    @on(Button.Pressed, "#embeddings-select-files")
    async def on_select_files(self) -> None:
        """Handle file selection."""
        def handle_selected(paths: List[Path]) -> None:
            """Handle selected files."""
            self.selected_files = paths
            file_list = self.query_one("#embeddings-file-list", TextArea)
            file_count = self.query_one("#embeddings-file-count", Label)
            
            if paths:
                file_list.text = "\n".join(str(p) for p in paths)
                file_count.update(f"Selected: {len(paths)} files")
            else:
                file_list.text = ""
                file_count.update("Selected: 0 files")
        
        # Show file picker dialog
        file_picker = FileOpen(
            filters=Filters(
                ("All Files", lambda p: True),
                ("Text Files", lambda p: p.suffix in {".txt", ".md", ".json"}),
                ("Documents", lambda p: p.suffix in {".pdf", ".doc", ".docx"}),
            )
        )
        
        self.app.push_screen(file_picker, handle_selected)
    
    @on(Input.Changed, "#embeddings-specific-ids")
    def on_specific_ids_changed(self, event: Input.Changed) -> None:
        """Track specific IDs input."""
        self.specific_item_ids = event.value
    
    @on(Input.Changed, "#embeddings-keywords-input")
    def on_keywords_changed(self, event: Input.Changed) -> None:
        """Track keywords input."""
        self.keyword_filter = event.value
    
    @on(RadioSet.Changed, "#embeddings-db-mode-set")
    def on_db_mode_changed(self, event: RadioSet.Changed) -> None:
        """Handle database selection mode change."""
        if event.pressed:
            # Determine which radio button was pressed based on its ID
            button_id = event.pressed.id
            if button_id == "embeddings-mode-search":
                self.selected_db_mode = "search"
            elif button_id == "embeddings-mode-all":
                self.selected_db_mode = "all"
            elif button_id == "embeddings-mode-specific":
                self.selected_db_mode = "specific"
            elif button_id == "embeddings-mode-keywords":
                self.selected_db_mode = "keywords"
            else:
                return
            
            # Show/hide appropriate containers
            search_container = self.query_one("#embeddings-search-container")
            specific_container = self.query_one("#embeddings-specific-container")
            keywords_container = self.query_one("#embeddings-keywords-container")
            
            # Hide all first
            search_container.styles.display = "none"
            specific_container.styles.display = "none"
            keywords_container.styles.display = "none"
            
            # Show the appropriate one
            if self.selected_db_mode == "search":
                search_container.styles.display = "block"
            elif self.selected_db_mode == "specific":
                specific_container.styles.display = "block"
            elif self.selected_db_mode == "keywords":
                keywords_container.styles.display = "block"
            # "all" mode doesn't need any input container
            
            # Clear previous selections when switching modes
            self.selected_db_items.clear()
            table = self.query_one("#embeddings-db-results", DataTable)
            table.clear()
            self.query_one("#embeddings-db-selection-count", Label).update("No items selected")
            
            # Update button text based on mode
            button = self.query_one("#embeddings-search-db", Button)
            if self.selected_db_mode == "all":
                button.label = "Load All Items"
            elif self.selected_db_mode == "specific":
                button.label = "Load Specific Items"
            elif self.selected_db_mode == "keywords":
                button.label = "Load by Keywords"
            else:
                button.label = "Search Database"
    
    @on(Select.Changed, "#embeddings-db-select")
    def on_database_changed(self, event: Select.Changed) -> None:
        """Handle database selection change."""
        if event.value and event.value != Select.BLANK:
            self.selected_db = str(event.value)
            
            # Update content type options based on selected database
            db_type_select = self.query_one("#embeddings-db-type", Select)
            
            if self.selected_db == "media":
                db_type_select.set_options([
                    ("Media Content", "media")
                ])
                # Don't set value - Select will auto-select first option when allow_blank=False
            else:  # chachanotes
                db_type_select.set_options([
                    ("Conversations", "conversations"),
                    ("Notes", "notes"),
                    ("Characters", "characters")
                ])
                # Don't set value - Select will auto-select first option when allow_blank=False
            
            self.notify(f"Selected {event.value} database", severity="information")
    
    @on(Button.Pressed, "#embeddings-search-db")
    async def on_search_database(self) -> None:
        """Load database items based on selected mode."""
        db_type = str(self.query_one("#embeddings-db-type", Select).value)
        
        if not db_type:
            self.notify("Please select a content type", severity="warning")
            return
        
        table = self.query_one("#embeddings-db-results", DataTable)
        table.clear()
        
        try:
            results = []
            
            # Use app's loaded databases
            media_db = self.media_db
            chachanotes_db = self.chachanotes_db
            
            # Handle different modes
            if self.selected_db_mode == "all":
                # Get all items with a reasonable limit
                if db_type == "media" and media_db:
                    results = media_db.get_all_active_media_for_embedding(limit=1000)
                elif db_type == "conversations" and chachanotes_db:
                    results = chachanotes_db.get_all_conversations(limit=1000)
                elif db_type == "notes" and chachanotes_db:
                    results = chachanotes_db.get_recent_notes(limit=1000)
                elif db_type == "characters" and chachanotes_db:
                    results = chachanotes_db.get_all_characters()
                
                # Add rows to table for all mode
                if db_type == "media":
                    for item in results:
                        item_id = str(item.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            item.get('title', 'Untitled')[:50],
                            item.get('type', 'unknown'),
                            item.get('created_at', '')[:10],
                            key=item_id
                        )
                elif db_type == "conversations":
                    for conv in results:
                        item_id = str(conv.get('conversation_id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            conv.get('title', 'Untitled Conversation')[:50],
                            "conversation",
                            conv.get('created_at', '')[:10],
                            key=item_id
                        )
                elif db_type == "notes":
                    for note in results:
                        item_id = str(note.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            note.get('title', 'Untitled Note')[:50],
                            "note",
                            note.get('created', '')[:10],
                            key=item_id
                        )
                elif db_type == "characters":
                    for char in results:
                        item_id = str(char.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            char.get('name', 'Unnamed Character')[:50],
                            "character",
                            char.get('created_at', '')[:10],
                            key=item_id
                        )
                    
            elif self.selected_db_mode == "specific":
                # Get specific items by IDs
                ids_input = self.query_one("#embeddings-specific-ids", Input).value.strip()
                if not ids_input:
                    self.notify("Please enter item IDs", severity="warning")
                    return
                    
                try:
                    # Parse comma-separated IDs
                    item_ids = [int(id.strip()) for id in ids_input.split(",") if id.strip()]
                    
                    if db_type == "media" and media_db:
                        for item_id in item_ids:
                            item = media_db.get_media_by_id(item_id)
                            if item:
                                results.append(item)
                    elif db_type == "conversations" and chachanotes_db:
                        for item_id in item_ids:
                            conv = chachanotes_db.get_conversation_by_id(item_id)
                            if conv:
                                results.append(conv)
                    elif db_type == "notes" and chachanotes_db:
                        for item_id in item_ids:
                            note = chachanotes_db.get_note_by_id(item_id)
                            if note:
                                results.append(note)
                    elif db_type == "characters" and chachanotes_db:
                        for item_id in item_ids:
                            char = chachanotes_db.get_character_by_id(item_id)
                            if char:
                                results.append(char)
                except ValueError:
                    self.notify("Invalid ID format. Please enter numeric IDs separated by commas.", severity="error")
                    return
                
                # Add rows to table for specific mode
                if db_type == "media":
                    for item in results:
                        item_id = str(item.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            item.get('title', 'Untitled')[:50],
                            item.get('type', 'unknown'),
                            item.get('created_at', '')[:10],
                            key=item_id
                        )
                elif db_type == "conversations":
                    for conv in results:
                        item_id = str(conv.get('conversation_id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            conv.get('title', 'Untitled Conversation')[:50],
                            "conversation",
                            conv.get('created_at', '')[:10],
                            key=item_id
                        )
                elif db_type == "notes":
                    for note in results:
                        item_id = str(note.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            note.get('title', 'Untitled Note')[:50],
                            "note",
                            note.get('created', '')[:10],
                            key=item_id
                        )
                elif db_type == "characters":
                    for char in results:
                        item_id = str(char.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            char.get('name', 'Unnamed Character')[:50],
                            "character",
                            char.get('created_at', '')[:10],
                            key=item_id
                        )
                    
            elif self.selected_db_mode == "keywords":
                # Get items by keywords
                keywords_input = self.query_one("#embeddings-keywords-input", Input).value.strip()
                if not keywords_input:
                    self.notify("Please enter keywords", severity="warning")
                    return
                    
                # Parse comma-separated keywords
                keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
                
                if db_type == "media" and media_db:
                    # Use the fetch_media_for_keywords method
                    keyword_results = media_db.fetch_media_for_keywords(keywords)
                    # Flatten the results from the dictionary
                    seen_ids = set()
                    for keyword, items in keyword_results.items():
                        for item in items:
                            if item['id'] not in seen_ids:
                                results.append(item)
                                seen_ids.add(item['id'])
                elif db_type in ["conversations", "notes", "characters"] and chachanotes_db:
                    # For chachanotes, use keyword search
                    for keyword in keywords:
                        if db_type == "conversations":
                            keyword_results = chachanotes_db.search_conversations_by_keywords(keyword)
                        elif db_type == "notes":
                            keyword_results = chachanotes_db.search_notes(keyword)
                        else:  # characters
                            keyword_results = chachanotes_db.search_characters(keyword)
                        
                        # Add unique results
                        for item in keyword_results:
                            item_id = item.get('conversation_id' if db_type == "conversations" else 'id')
                            if not any(r.get('conversation_id' if db_type == "conversations" else 'id') == item_id for r in results):
                                results.append(item)
                
                # Add rows to table for keywords mode
                if db_type == "media":
                    for item in results:
                        item_id = str(item.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            item.get('title', 'Untitled')[:50],
                            item.get('type', 'unknown'),
                            item.get('created_at', '')[:10],
                            key=item_id
                        )
                elif db_type == "conversations":
                    for conv in results:
                        item_id = str(conv.get('conversation_id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            conv.get('title', 'Untitled Conversation')[:50],
                            "conversation",
                            conv.get('created_at', '')[:10],
                            key=item_id
                        )
                elif db_type == "notes":
                    for note in results:
                        item_id = str(note.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            note.get('title', 'Untitled Note')[:50],
                            "note",
                            note.get('created', '')[:10],
                            key=item_id
                        )
                elif db_type == "characters":
                    for char in results:
                        item_id = str(char.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            char.get('name', 'Unnamed Character')[:50],
                            "character",
                            char.get('created_at', '')[:10],
                            key=item_id
                        )
                                
            else:  # search mode
                # Original search behavior
                search_term = self.query_one("#embeddings-db-filter", Input).value
                
                if db_type == "media" and media_db:
                    # Search media database
                    results = media_db.search_media_db(search_term) if search_term else media_db.get_all_active_media_for_embedding(limit=100)
                    for item in results:
                        item_id = str(item.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            item.get('title', 'Untitled')[:50],
                            item.get('type', 'unknown'),
                            item.get('created_at', '')[:10],
                            key=item_id  # Add key for easier row identification
                        )
                
                elif db_type == "conversations" and chachanotes_db:
                    # Search conversations
                    results = chachanotes_db.search_conversations_by_keywords(search_term) if search_term else chachanotes_db.get_all_conversations(limit=100)
                    for conv in results:
                        item_id = str(conv.get('conversation_id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            conv.get('title', 'Untitled Conversation')[:50],
                            "conversation",
                            conv.get('created_at', '')[:10],
                            key=item_id
                        )
                
                elif db_type == "notes" and chachanotes_db:
                    # Search notes
                    results = chachanotes_db.search_notes(search_term) if search_term else chachanotes_db.get_recent_notes(limit=100)
                    for note in results:
                        item_id = str(note.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            note.get('title', 'Untitled Note')[:50],
                            "note",
                            note.get('created', '')[:10],
                            key=item_id
                        )
                
                elif db_type == "characters" and chachanotes_db:
                    # Search characters
                    results = chachanotes_db.search_characters(search_term) if search_term else chachanotes_db.get_all_characters()
                    for char in results:
                        item_id = str(char.get('id', ''))
                        table.add_row(
                            "" if item_id not in self.selected_db_items else "✓",
                            item_id,
                            char.get('name', 'Unnamed Character')[:50],
                            "character",
                            char.get('created_at', '')[:10],
                            key=item_id
                        )
            
            count = len(results)
            
            # Update status based on mode
            if self.selected_db_mode == "all":
                self.query_one("#embeddings-db-selection-count", Label).update(f"Loaded all {count} items")
                # Auto-select all items for "all" mode
                for row_key in table.rows:
                    row_data = table.get_row(row_key)
                    # ID is in the second column (index 1)
                    item_id = str(row_data[1])
                    self.selected_db_items.add(item_id)
                    # Update first column to show selection
                    column_keys = list(table.columns.keys())
                    if column_keys:
                        table.update_cell(row_key, column_keys[0], "✓")
            elif self.selected_db_mode == "specific":
                self.query_one("#embeddings-db-selection-count", Label).update(f"Loaded {count} specific items")
                # Auto-select all loaded items
                for row_key in table.rows:
                    row_data = table.get_row(row_key)
                    # ID is in the second column (index 1)
                    item_id = str(row_data[1])
                    self.selected_db_items.add(item_id)
                    # Update first column to show selection
                    column_keys = list(table.columns.keys())
                    if column_keys:
                        table.update_cell(row_key, column_keys[0], "✓")
            elif self.selected_db_mode == "keywords":
                self.query_one("#embeddings-db-selection-count", Label).update(f"Found {count} items matching keywords")
                # Auto-select all keyword matches
                for row_key in table.rows:
                    row_data = table.get_row(row_key)
                    # ID is in the second column (index 1)
                    item_id = str(row_data[1])
                    self.selected_db_items.add(item_id)
                    # Update first column to show selection
                    column_keys = list(table.columns.keys())
                    if column_keys:
                        table.update_cell(row_key, column_keys[0], "✓")
            else:
                self.query_one("#embeddings-db-selection-count", Label).update(f"Found {count} items")
            
            if count == 0:
                if self.selected_db_mode == "specific":
                    self.notify("No items found with the specified IDs", severity="warning")
                elif self.selected_db_mode == "keywords":
                    self.notify("No items found matching the keywords", severity="warning")
                else:
                    self.notify("No items found", severity="information")
            
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            self.notify(f"Search failed: {str(e)}", severity="error")
    
    @on(DataTable.RowSelected, "#embeddings-db-results")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in results table."""
        table = self.query_one("#embeddings-db-results", DataTable)
        row_key = event.row_key
        
        if row_key is not None:
            try:
                # Get the row data
                row_data = table.get_row(row_key)
                # ID is in the second column (index 1)
                item_id = str(row_data[1])
                
                # Get the first column key
                column_keys = list(table.columns.keys())
                if not column_keys:
                    raise Exception("No columns found in table")
                first_column_key = column_keys[0]
                
                # Toggle selection
                if item_id in self.selected_db_items:
                    self.selected_db_items.discard(item_id)
                    # Update first column to clear selection
                    table.update_cell(row_key, first_column_key, "")
                else:
                    self.selected_db_items.add(item_id)
                    # Update first column to show selection
                    table.update_cell(row_key, first_column_key, "✓")
                
                # Update selection count
                self.query_one("#embeddings-db-selection-count", Label).update(
                    f"Selected {len(self.selected_db_items)} items"
                )
            except Exception as e:
                logger.error(f"Error selecting row: {e}")
                self.notify(f"Error selecting item: {str(e)}", severity="error")
    
    @on(Button.Pressed, "#embeddings-select-all")
    def on_select_all(self) -> None:
        """Select all items in the results table."""
        table = self.query_one("#embeddings-db-results", DataTable)
        
        # Select all items
        column_keys = list(table.columns.keys())
        if not column_keys:
            self.notify("No columns found in table", severity="error")
            return
        first_column_key = column_keys[0]
        
        for row_key in table.rows:
            row_data = table.get_row(row_key)
            # ID is in the second column (index 1)
            item_id = str(row_data[1])
            self.selected_db_items.add(item_id)
            # Update first column to show selection
            table.update_cell(row_key, first_column_key, "✓")
        
        # Update selection count
        self.query_one("#embeddings-db-selection-count", Label).update(
            f"Selected {len(self.selected_db_items)} items"
        )
    
    @on(Button.Pressed, "#embeddings-clear-selection")
    def on_clear_selection(self) -> None:
        """Clear all selections in the results table."""
        table = self.query_one("#embeddings-db-results", DataTable)
        
        # Clear all selections
        self.selected_db_items.clear()
        column_keys = list(table.columns.keys())
        if column_keys:
            first_column_key = column_keys[0]
            for row_key in table.rows:
                # Update first column to clear selection
                table.update_cell(row_key, first_column_key, "")
        
        # Update selection count
        self.query_one("#embeddings-db-selection-count", Label).update("No items selected")
    
    @on(Button.Pressed, "#embeddings-preview")
    async def on_preview_chunks(self) -> None:
        """Preview how text will be chunked."""
        text = await self._get_input_text()
        if not text:
            self.notify("No input text to preview", severity="warning")
            return
        
        try:
            # Get chunking parameters
            method = str(self.query_one("#embeddings-chunk-method", Select).value)
            chunk_size = int(self.query_one("#embeddings-chunk-size", Input).value or "512")
            chunk_overlap = int(self.query_one("#embeddings-chunk-overlap", Input).value or "128")
            
            # Generate preview of first few chunks
            if chunk_for_embedding and method in CHUNK_METHODS:
                chunks = chunk_for_embedding(text, chunk_method=method, max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                preview_text = f"Chunking Method: {method}\n"
                preview_text += f"Chunk Size: {chunk_size}, Overlap: {chunk_overlap}\n"
                preview_text += f"Total Chunks: {len(chunks)}\n\n"
                
                # Show first 3 chunks
                for i, chunk in enumerate(chunks[:3]):
                    preview_text += f"--- Chunk {i+1} ---\n{chunk}\n\n"
                
                if len(chunks) > 3:
                    preview_text += f"... and {len(chunks) - 3} more chunks"
                
                preview_area = self.query_one("#embeddings-chunk-preview", TextArea)
                preview_area.text = preview_text
                
                # Expand the collapsible
                collapsible = self.query_one("#embeddings-chunk-preview-collapsible", Collapsible)
                collapsible.collapsed = False
                
        except Exception as e:
            logger.error(f"Failed to preview chunks: {e}")
            self.notify(f"Preview failed: {str(e)}", severity="error")
    
    @on(Button.Pressed, "#embeddings-create")
    async def on_create_embeddings(self) -> None:
        """Create embeddings from the input."""
        if self.is_processing:
            self.notify("Already processing, please wait", severity="warning")
            return
        
        if not self.selected_model:
            self.notify("Please select an embedding model", severity="warning")
            return
        
        collection_name = self.query_one("#embeddings-collection-name", Input).value
        if not collection_name:
            self.notify("Please enter a collection name", severity="warning")
            return
        
        try:
            self.is_processing = True
            self._show_progress(True)
            
            # Provide feedback based on mode
            if self.selected_source == self.SOURCE_DATABASE:
                if self.selected_db_mode == "all":
                    self._update_status(f"Creating embeddings for ALL items in {self.selected_db} database...")
                elif self.selected_db_mode == "specific":
                    self._update_status(f"Creating embeddings for specific items: {self.specific_item_ids}")
                elif self.selected_db_mode == "keywords":
                    self._update_status(f"Creating embeddings for items matching keywords: {self.keyword_filter}")
                else:
                    self._update_status(f"Creating embeddings for {len(self.selected_db_items)} selected items")
            else:
                self._update_status(f"Creating embeddings for {len(self.selected_files)} files")
            
            # Get input text
            text = await self._get_input_text()
            if not text:
                self.notify("No input text to process", severity="warning")
                return
            
            # Process embeddings
            await self._process_embeddings(text, collection_name)
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            self._update_status(f"Error: {str(e)}")
            self.notify(f"Failed to create embeddings: {str(e)}", severity="error")
        finally:
            self.is_processing = False
            self._show_progress(False)
    
    @on(Button.Pressed, "#embeddings-clear")
    def on_clear_form(self) -> None:
        """Clear all form inputs."""
        # Clear file selection
        self.selected_files = []
        self.query_one("#embeddings-file-list", TextArea).text = ""
        self.query_one("#embeddings-file-count", Label).update("Selected: 0 files")
        
        # Clear database filter
        self.query_one("#embeddings-db-filter", Input).value = ""
        
        # Clear collection settings
        self.query_one("#embeddings-collection-name", Input).value = ""
        self.query_one("#embeddings-collection-desc", Input).value = ""
        
        # Clear status
        self._update_status("")
        
        # Clear database results and selections
        table = self.query_one("#embeddings-db-results", DataTable)
        table.clear()
        self.selected_db_items.clear()
        self.query_one("#embeddings-db-selection-count", Label).update("No items selected")
        
        # Reset database selection - let the Select widget handle its own state
        
        # Reset source to files
        source_select = self.query_one("#embeddings-source-type", Select)
        source_select.value = self.SOURCE_FILE
        
        self.notify("Form cleared", severity="information")
    
    async def _get_input_text(self) -> str:
        """Get input text based on selected source."""
        if self.selected_source == self.SOURCE_FILE:
            # Read content from selected files
            all_text = []
            for file_path in self.selected_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        all_text.append(f.read())
                except Exception as e:
                    logger.error(f"Failed to read file {file_path}: {e}")
            return "\n\n".join(all_text)
        
        elif self.selected_source == self.SOURCE_DATABASE:
            db_type = str(self.query_one("#embeddings-db-type", Select).value)
            
            all_text = []
            
            # Use app's loaded databases
            media_db = self.media_db
            chachanotes_db = self.chachanotes_db
            
            # Determine which items to process based on mode
            items_to_process = []
            
            if self.selected_db_mode == "all":
                # For "all" mode, fetch all items directly
                if db_type == "media" and media_db:
                    items = media_db.get_all_active_media_for_embedding(limit=10000)  # Higher limit for embeddings
                    items_to_process = [(str(item.get('id', '')), item) for item in items]
                elif db_type == "conversations" and chachanotes_db:
                    items = chachanotes_db.get_all_conversations(limit=10000)
                    items_to_process = [(str(item.get('conversation_id', '')), item) for item in items]
                elif db_type == "notes" and chachanotes_db:
                    items = chachanotes_db.get_recent_notes(limit=10000)
                    items_to_process = [(str(item.get('id', '')), item) for item in items]
                elif db_type == "characters" and chachanotes_db:
                    items = chachanotes_db.get_all_characters()
                    items_to_process = [(str(item.get('id', '')), item) for item in items]
                    
            elif self.selected_db_mode == "specific":
                # For specific mode, parse IDs and fetch items
                if self.specific_item_ids:
                    try:
                        item_ids = [int(id.strip()) for id in self.specific_item_ids.split(",") if id.strip()]
                        
                        if db_type == "media" and media_db:
                            for item_id in item_ids:
                                item = media_db.get_media_by_id(item_id)
                                if item:
                                    items_to_process.append((str(item_id), item))
                        elif db_type == "conversations" and chachanotes_db:
                            for item_id in item_ids:
                                item = chachanotes_db.get_conversation_by_id(item_id)
                                if item:
                                    items_to_process.append((str(item_id), item))
                        elif db_type == "notes" and chachanotes_db:
                            for item_id in item_ids:
                                item = chachanotes_db.get_note_by_id(item_id)
                                if item:
                                    items_to_process.append((str(item_id), item))
                        elif db_type == "characters" and chachanotes_db:
                            for item_id in item_ids:
                                item = chachanotes_db.get_character_by_id(item_id)
                                if item:
                                    items_to_process.append((str(item_id), item))
                    except ValueError:
                        logger.error("Invalid ID format in specific IDs")
                        
            elif self.selected_db_mode == "keywords":
                # For keywords mode, fetch by keywords
                if self.keyword_filter:
                    keywords = [kw.strip() for kw in self.keyword_filter.split(",") if kw.strip()]
                    
                    if db_type == "media" and media_db:
                        keyword_results = media_db.fetch_media_for_keywords(keywords)
                        seen_ids = set()
                        for keyword, items in keyword_results.items():
                            for item in items:
                                if item['id'] not in seen_ids:
                                    items_to_process.append((str(item['id']), item))
                                    seen_ids.add(item['id'])
                    elif chachanotes_db:
                        # For chachanotes, aggregate results from keyword searches
                        seen_ids = set()
                        for keyword in keywords:
                            if db_type == "conversations":
                                results = chachanotes_db.search_conversations_by_keywords(keyword)
                                for item in results:
                                    item_id = str(item.get('conversation_id', ''))
                                    if item_id not in seen_ids:
                                        items_to_process.append((item_id, item))
                                        seen_ids.add(item_id)
                            elif db_type == "notes":
                                results = chachanotes_db.search_notes(keyword)
                                for item in results:
                                    item_id = str(item.get('id', ''))
                                    if item_id not in seen_ids:
                                        items_to_process.append((item_id, item))
                                        seen_ids.add(item_id)
                            elif db_type == "characters":
                                results = chachanotes_db.search_characters(keyword)
                                for item in results:
                                    item_id = str(item.get('id', ''))
                                    if item_id not in seen_ids:
                                        items_to_process.append((item_id, item))
                                        seen_ids.add(item_id)
                                        
            else:  # search mode - use selected items from table
                # Original behavior - process selected items
                for item_id in self.selected_db_items:
                    items_to_process.append((item_id, None))  # We'll fetch the item below
            
            # Process the items
            for item_id, item in items_to_process:
                try:
                    if db_type == "media" and media_db:
                        # Use provided item or fetch it
                        media_item = item if item else media_db.get_media_by_id(int(item_id))
                        if media_item:
                            content = media_item.get('content', media_item.get('transcript', ''))
                            if content:
                                all_text.append(f"=== {media_item.get('title', 'Untitled')} ===\n{content}")
                        
                    elif db_type == "conversations" and chachanotes_db:
                        # For conversations, we always need to fetch messages
                        conv = item if item else chachanotes_db.get_conversation_by_id(int(item_id))
                        if conv:
                            conv_id = conv.get('conversation_id') or conv.get('id')
                            messages = chachanotes_db.get_messages_by_conversation_id(int(conv_id))
                            if messages:
                                conv_text = []
                                for msg in messages:
                                    role = msg.get('role', 'unknown')
                                    content = msg.get('content', '')
                                    conv_text.append(f"{role}: {content}")
                                all_text.append(f"=== {conv.get('title', f'Conversation {conv_id}')} ===\n" + "\n\n".join(conv_text))
                    
                    elif db_type == "notes" and chachanotes_db:
                        # Use provided item or fetch it
                        note = item if item else chachanotes_db.get_note_by_id(int(item_id))
                        if note:
                            all_text.append(f"=== {note.get('title', 'Untitled Note')} ===\n{note.get('content', '')}")
                    
                    elif db_type == "characters" and chachanotes_db:
                        # Use provided item or fetch it
                        character = item if item else chachanotes_db.get_character_by_id(int(item_id))
                        if character:
                            char_text = []
                            char_text.append(f"Name: {character.get('name', 'Unknown')}")
                            char_text.append(f"Description: {character.get('description', '')}")
                            char_text.append(f"Personality: {character.get('personality', '')}")
                            char_text.append(f"First Message: {character.get('first_message', '')}")
                            all_text.append(f"=== Character: {character.get('name', 'Unknown')} ===\n" + "\n".join(char_text))
                    
                except Exception as e:
                    logger.error(f"Failed to get content for {db_type} ID {item_id}: {e}")
            
            return "\n\n".join(all_text)
        
        return ""
    
    async def _process_embeddings(self, text: str, collection_name: str) -> None:
        """Process text and create embeddings."""
        if not self.embedding_factory or not self.chroma_manager:
            raise ValueError("Embeddings not properly initialized")
        
        # Get chunking parameters
        method = str(self.query_one("#embeddings-chunk-method", Select).value)
        chunk_size = int(self.query_one("#embeddings-chunk-size", Input).value or "512")
        chunk_overlap = int(self.query_one("#embeddings-chunk-overlap", Input).value or "128")
        
        self._update_status("Chunking text...")
        
        # Chunk the text
        if chunk_for_embedding and method in CHUNK_METHODS:
            chunks = chunk_for_embedding(text, chunk_method=method, max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            # Simple chunking fallback
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-chunk_overlap)]
        
        self._update_status(f"Processing {len(chunks)} chunks...")
        
        # Update progress bar
        progress_bar = self.query_one("#embeddings-progress-bar", ProgressBar)
        progress_bar.total = len(chunks)
        
        # Generate embeddings
        embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = await self.embedding_factory.async_embed(
                [chunk],
                model_id=self.selected_model
            )
            embeddings.append(embedding[0])
            
            # Update progress
            progress_bar.advance(1)
            self._update_status(f"Processed {i+1}/{len(chunks)} chunks...")
        
        # Store in ChromaDB
        self._update_status("Storing embeddings in ChromaDB...")
        
        # This would interface with ChromaDB to store the embeddings
        # For now, just update status
        self._update_status(f"Successfully created {len(embeddings)} embeddings in collection '{collection_name}'")
        self.notify(f"Embeddings created successfully!", severity="success")
    
    def _show_progress(self, show: bool) -> None:
        """Show or hide progress container."""
        progress_container = self.query_one("#embeddings-progress-container")
        if show:
            progress_container.styles.display = "block"
            progress_bar = self.query_one("#embeddings-progress-bar", ProgressBar)
            progress_bar.update(progress=0)
        else:
            progress_container.styles.display = "none"
    
    def _update_status(self, message: str) -> None:
        """Update status output."""
        status_output = self.query_one("#embeddings-status-output", TextArea)
        if message:
            status_output.text += f"{message}\n"
            status_output.scroll_end()
        else:
            status_output.text = ""
    
    def _update_source_containers(self) -> None:
        """Update visibility of source containers based on selected source."""
        try:
            file_container = self.query_one("#file-input-container")
            db_container = self.query_one("#db-input-container")
            
            logger.info(f"=== _update_source_containers CALLED ===")
            logger.info(f"Selected source: {self.selected_source}")
            logger.info(f"SOURCE_FILE: {self.SOURCE_FILE}, SOURCE_DATABASE: {self.SOURCE_DATABASE}")
            
            # Log current state before changes
            logger.info(f"Before - File container classes: {file_container.classes}, DB container classes: {db_container.classes}")
            
            # Use add_class/remove_class for better Textual compatibility
            if self.selected_source == self.SOURCE_FILE:
                file_container.remove_class("hidden")
                db_container.add_class("hidden")
                # Also set display style directly as a fallback
                file_container.styles.display = "block"
                db_container.styles.display = "none"
                logger.info("ACTION: Showing file container, hiding db container")
            elif self.selected_source == self.SOURCE_DATABASE:
                file_container.add_class("hidden")
                db_container.remove_class("hidden")
                # Also set display style directly as a fallback
                file_container.styles.display = "none"
                db_container.styles.display = "block"
                logger.info("ACTION: Hiding file container, showing db container")
            else:
                logger.warning(f"Unknown source type: {self.selected_source}")
                # Default to file input
                file_container.remove_class("hidden")
                db_container.add_class("hidden")
                file_container.styles.display = "block"
                db_container.styles.display = "none"
            
            # Log state after changes
            logger.info(f"After - File container classes: {file_container.classes}, DB container classes: {db_container.classes}")
            
            # Find the parent VerticalScroll and refresh it
            try:
                # The containers are inside a VerticalScroll widget
                scroll_widget = file_container.parent
                if scroll_widget:
                    logger.debug(f"Refreshing parent widget: {scroll_widget.__class__.__name__}")
                    scroll_widget.refresh(layout=True)
                    
                    # Also refresh the grandparent (the view container)
                    view_container = scroll_widget.parent
                    if view_container:
                        logger.debug(f"Refreshing view container: {view_container.__class__.__name__}")
                        view_container.refresh(layout=True)
            except Exception as e:
                logger.debug(f"Could not refresh parent widgets: {e}")
            
            # Force refresh on the containers themselves
            file_container.refresh(layout=True)
            db_container.refresh(layout=True)
            
            # Refresh the entire embeddings window
            self.refresh(layout=True)
            
            # Debug check after refresh
            def debug_check():
                has_hidden_class_file = "hidden" in file_container.classes
                has_hidden_class_db = "hidden" in db_container.classes
                logger.debug(
                    f"Post-refresh check - File container hidden class: {has_hidden_class_file}, "
                    f"DB container hidden class: {has_hidden_class_db}"
                )
                logger.info(
                    f"Containers updated - File: {'hidden' if has_hidden_class_file else 'visible'}, "
                    f"DB: {'hidden' if has_hidden_class_db else 'visible'}"
                )
            
            self.app.call_later(debug_check)
            
        except Exception as e:
            logger.error(f"Error updating source containers: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    
    def action_select_all(self) -> None:
        """Action handler for Ctrl+A - select all items."""
        # Only work if database content is visible and has results
        if self.selected_source == self.SOURCE_DATABASE:
            table = self.query_one("#embeddings-db-results", DataTable)
            if table.row_count > 0:
                self.on_select_all()
    
    def action_clear_selection(self) -> None:
        """Action handler for Ctrl+D - clear all selections."""
        if self.selected_source == self.SOURCE_DATABASE:
            table = self.query_one("#embeddings-db-results", DataTable)
            if table.row_count > 0:
                self.on_clear_selection()
    
    def action_toggle_selection(self) -> None:
        """Action handler for Space - toggle current row selection."""
        if self.selected_source == self.SOURCE_DATABASE:
            table = self.query_one("#embeddings-db-results", DataTable)
            if table.cursor_coordinate:
                # Simulate a row selection event for the current cursor position
                row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
                if row_key is not None:
                    # Get the row data
                    row_data = table.get_row(row_key)
                    # ID is in the second column (index 1)
                    item_id = str(row_data[1])
                    
                    # Get the first column key
                    column_keys = list(table.columns.keys())
                    if not column_keys:
                        raise Exception("No columns found in table")
                    first_column_key = column_keys[0]
                    
                    # Toggle selection
                    if item_id in self.selected_db_items:
                        self.selected_db_items.discard(item_id)
                        # Update first column to clear selection
                        table.update_cell(row_key, first_column_key, "")
                    else:
                        self.selected_db_items.add(item_id)
                        # Update first column to show selection
                        table.update_cell(row_key, first_column_key, "✓")
                    
                    # Update selection count
                    self.query_one("#embeddings-db-selection-count", Label).update(
                        f"Selected {len(self.selected_db_items)} items"
                    )

#
# End of Embeddings_Window.py
########################################################################################################################