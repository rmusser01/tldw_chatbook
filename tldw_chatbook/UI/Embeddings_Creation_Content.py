# tldw_chatbook/UI/Embeddings_Creation_Content.py
# Description: Embeddings creation content for use within Search tab
#
# This is a simplified version of the EmbeddingsWindow that only contains
# the content generation logic without the navigation pane

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

# 3rd-Party Imports
from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.reactive import reactive
from textual.widgets import (
    Static, Button, Input, Label, Select, TextArea, Checkbox, RadioButton, RadioSet,
    Collapsible, ProgressBar, Rule, ContentSwitcher, TabbedContent, TabPane
)

# Local widget imports
from ..Widgets.tooltip import HelpIcon
from ..Widgets.chunk_preview import ChunkPreview
from ..Widgets.embedding_template_selector import EmbeddingTemplateQuickSelect, EmbeddingTemplateSelected

# Configure logger with context
logger = logger.bind(module="Embeddings_Creation_Content")

# Local Imports
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from ..Third_Party.textual_fspicker import Filters

# Check if embeddings dependencies are available
if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
    try:
        from ..Embeddings.Embeddings_Lib import EmbeddingFactory
        from ..Embeddings.Chroma_Lib import ChromaDBManager
        from ..Chunking.Chunk_Lib import chunk_for_embedding
        logger.info("Successfully imported embeddings modules in EmbeddingsCreationContent")
    except ImportError as e:
        logger.error(f"Failed to import embeddings modules: {e}")
        EmbeddingFactory = None
        ChromaDBManager = None
        chunk_for_embedding = None
else:
    logger.warning("Embeddings dependencies not available according to DEPENDENCIES_AVAILABLE")
    EmbeddingFactory = None
    ChromaDBManager = None
    chunk_for_embedding = None

# Define available chunk methods
CHUNK_METHODS = ['words', 'sentences', 'paragraphs', 'tokens', 'semantic', 'json', 'xml', 'ebook_chapters', 'rolling_summarize']

if TYPE_CHECKING:
    from ..app import TldwCli

class EmbeddingsCreationContent(Container):
    """Content container for creating embeddings, designed to work within the Search tab."""
    
    DEFAULT_CSS = """
    EmbeddingsCreationContent {
        layout: vertical;
        height: 100%;
        width: 100%;
        background: $surface;
        padding: 1;
        display: block !important;
    }
    
    /* Ensure TabbedContent is visible */
    EmbeddingsCreationContent TabbedContent {
        height: 1fr;
        width: 100%;
        display: block !important;
    }
    
    /* Ensure TabPane content is visible */
    EmbeddingsCreationContent TabPane {
        height: 100%;
        width: 100%;
        display: block !important;
    }
    
    /* Ensure VerticalScroll is visible */
    EmbeddingsCreationContent VerticalScroll {
        height: 100%;
        width: 100%;
        display: block !important;
    }
    
    /* Override any parent display:none */
    EmbeddingsCreationContent * {
        display: block !important;
    }
    """
    
    # Input source types
    SOURCE_FILE = "file"
    SOURCE_DATABASE = "database"
    
    # Reactive attributes
    selected_source: reactive[str] = reactive(SOURCE_FILE)
    selected_model: reactive[Optional[str]] = reactive(None)
    is_processing: reactive[bool] = reactive(False)
    selected_files: reactive[List[Path]] = reactive([])
    selected_db: reactive[str] = reactive("media")
    selected_db_type: reactive[Optional[str]] = reactive("media")
    selected_db_items: reactive[set] = reactive(set())
    selected_db_mode: reactive[str] = reactive("search")
    specific_item_ids: reactive[str] = reactive("")
    keyword_filter: reactive[str] = reactive("")
    
    # Validation states
    validation_errors: reactive[dict] = reactive({})
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.chachanotes_db = app_instance.chachanotes_db if hasattr(app_instance, 'chachanotes_db') else None
        self.media_db = app_instance.media_db if hasattr(app_instance, 'media_db') else None
        self.embedding_factory: Optional[EmbeddingFactory] = None
        self.chroma_manager: Optional[ChromaDBManager] = None
        
    def compose(self) -> ComposeResult:
        """Compose the embeddings creation content."""
        logger.info("EmbeddingsCreationContent.compose() called")
        
        # Check if embeddings dependencies are available
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            logger.warning("Embeddings dependencies not available")
            with Container(classes="embeddings-not-available"):
                yield Static("⚠️ Embeddings/RAG functionality not available", classes="warning-title")
                yield Static("The required dependencies for embeddings are not installed.", classes="warning-message")
                yield Static("To enable embeddings, please install with:", classes="warning-message")
                yield Static("pip install tldw_chatbook[embeddings_rag]", classes="code-block")
            return
        
        logger.info("Embeddings dependencies are available, proceeding with UI")
        
        # Create the tabbed interface
        with TabbedContent(id="embeddings-tabs"):
            with TabPane("Source & Model", id="tab-source-model"):
                yield from self._compose_source_model_tab()
            
            with TabPane("Processing", id="tab-processing"):
                yield from self._compose_processing_tab()
            
            with TabPane("Output", id="tab-output"):
                # Yield content directly without wrapper
                yield Label("Collection Settings", classes="embeddings-section-title")
                
                with Container(classes="embeddings-form-section"):
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Collection Name:", classes="embeddings-form-label")
                        yield Input(
                            placeholder="Enter collection name",
                            id="embeddings-collection-name",
                            classes="embeddings-form-control"
                        )
                    
                    yield Static("Choose a descriptive name for your collection", classes="embeddings-help-text")
                    yield Static("", id="error-collection-name", classes="error-message hidden")
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Description:", classes="embeddings-form-label")
                        yield TextArea(
                            "",
                            id="embeddings-collection-description",
                            classes="embeddings-form-control"
                        )
                    
                    yield Checkbox("Add timestamps to metadata", id="embeddings-add-timestamps", value=True)
                    yield Checkbox("Include source information", id="embeddings-include-source", value=True)
    
    def _compose_source_model_tab(self) -> ComposeResult:
        """Compose the Source & Model tab content."""
        with VerticalScroll(classes="embeddings-tab-scroll"):
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
            
            # Error display for model selection
            yield Static("", id="error-model", classes="error-message hidden")
            
            # Template selector - temporarily disabled to test basic functionality
            # yield EmbeddingTemplateQuickSelect(id="embeddings-template-selector")
            yield Static("Template selector will be here", classes="embeddings-form-full-row")
            
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
                    
                    # Error display for file selection
                    yield Static("", id="error-files", classes="error-message hidden")
                
                # Database query container
                with Container(id="db-input-container", classes="embeddings-input-source-container"):
                    # Database selection
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
                                ("Single Media Item", "single_media"),
                                ("All Media Items", "all_media"),
                                ("Media by Keywords", "media_keywords")
                            ],
                            id="embeddings-db-type",
                            classes="embeddings-form-control",
                            allow_blank=False,
                            value="single_media"
                        )
                    
                    yield Rule()
                    
                    # Selection mode - simplified
                    yield Label("Selection Mode", classes="embeddings-section-title")
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Mode:", classes="embeddings-form-label")
                        yield Select(
                            [
                                ("Search & Select", "search"),
                                ("All Items", "all"),
                                ("Specific IDs", "specific"),
                                ("By Keywords", "keywords")
                            ],
                            id="embeddings-db-mode-select",
                            classes="embeddings-form-control",
                            value="search"
                        )
                    
                    # Mode-specific input containers
                    with ContentSwitcher(initial="mode-search", id="embeddings-mode-switcher"):
                        # Search mode container
                        with Container(id="mode-search", classes="embeddings-mode-container"):
                            with Horizontal(classes="embeddings-form-row"):
                                yield Label("Search:", classes="embeddings-form-label")
                                yield Input(
                                    placeholder="Search for content...",
                                    id="embeddings-db-filter",
                                    classes="embeddings-form-control"
                                )
                            yield TextArea(
                                "Search results will appear here...",
                                id="embeddings-db-results",
                                classes="embeddings-db-results",
                                read_only=True
                            )
                        
                        # Specific ID mode container
                        with Container(id="mode-specific", classes="embeddings-mode-container"):
                            with Horizontal(classes="embeddings-form-row"):
                                yield Label("Item ID:", classes="embeddings-form-label")
                                yield Input(
                                    placeholder="Enter specific item ID",
                                    id="embeddings-specific-id",
                                    classes="embeddings-form-control"
                                )
                            yield Static(
                                "Enter the ID of the specific item to create embeddings for.",
                                classes="embeddings-help-text"
                            )
                        
                        # All items mode container
                        with Container(id="mode-all", classes="embeddings-mode-container"):
                            yield Static(
                                "⚠️ This will process ALL items in the selected database.",
                                classes="embeddings-warning"
                            )
                            yield Static(
                                "",
                                id="embeddings-all-count",
                                classes="embeddings-info"
                            )
                        
                        # Keywords mode container
                        with Container(id="mode-keywords", classes="embeddings-mode-container"):
                            with Horizontal(classes="embeddings-form-row"):
                                yield Label("Keywords:", classes="embeddings-form-label")
                                yield Input(
                                    placeholder="Enter keywords (comma-separated)",
                                    id="embeddings-keywords",
                                    classes="embeddings-form-control"
                                )
                            with Horizontal(classes="embeddings-form-row"):
                                yield Label("Match:", classes="embeddings-form-label")
                                yield RadioSet(
                                    RadioButton("Match ANY keyword", value=True, id="match-any"),
                                    RadioButton("Match ALL keywords", id="match-all"),
                                    id="embeddings-keyword-match",
                                    classes="embeddings-form-control"
                                )
                    
                    # Error display for database selection
                    yield Static("", id="error-database", classes="error-message hidden")
    
    def _compose_processing_tab(self) -> ComposeResult:
        """Compose the Processing tab content."""
        with VerticalScroll(classes="embeddings-tab-scroll"):
            # Chunking Configuration Section
            with Container(classes="embeddings-form-section"):
                yield Label("Chunking Configuration", classes="embeddings-section-title")
                
                with Horizontal(classes="embeddings-form-row"):
                    yield Label("Chunk Method:", classes="embeddings-form-label")
                    yield Select(
                        [(method.replace('_', ' ').title(), method) for method in CHUNK_METHODS],
                        id="embeddings-chunk-method",
                        classes="embeddings-form-control",
                        value="words"
                    )
                    yield HelpIcon(
                        "Choose how to split your text:\n\n"
                        "• Words: Split by word count (good for general text)\n"
                        "• Sentences: Split by sentences (preserves meaning)\n"
                        "• Paragraphs: Split by paragraphs (maintains context)\n"
                        "• Tokens: Split by model tokens (precise control)\n"
                        "• Semantic: Smart splitting based on content\n"
                        "• JSON/XML: Structure-aware splitting",
                        classes="help-icon-inline"
                    )
                
                with Horizontal(classes="embeddings-form-row"):
                    yield Label("Chunk Size:", classes="embeddings-form-label")
                    yield Input(
                        "512",
                        id="embeddings-chunk-size",
                        classes="embeddings-form-control",
                        type="integer"
                    )
                    yield HelpIcon(
                        "Number of units per chunk:\n\n"
                        "• For 'words': 100-500 recommended\n"
                        "• For 'tokens': 256-512 for most models\n"
                        "• Larger chunks = more context\n"
                        "• Smaller chunks = better precision\n\n"
                        "OpenAI models: 512-1024 tokens\n"
                        "Sentence transformers: 256-512 tokens",
                        classes="help-icon-inline"
                    )
                
                # Error display for chunk size
                yield Static("", id="error-chunk-size", classes="error-message hidden")
                
                with Horizontal(classes="embeddings-form-row"):
                    yield Label("Chunk Overlap:", classes="embeddings-form-label")
                    yield Input(
                        "128",
                        id="embeddings-chunk-overlap",
                        classes="embeddings-form-control",
                        type="integer"
                    )
                    yield HelpIcon(
                        "Overlap between consecutive chunks:\n\n"
                        "• Prevents context loss at boundaries\n"
                        "• Typically 10-25% of chunk size\n"
                        "• Higher overlap = better continuity\n"
                        "• Lower overlap = less redundancy\n\n"
                        "Example: Size=512, Overlap=128\n"
                        "Chunk 1: 0-512, Chunk 2: 384-896",
                        classes="help-icon-inline"
                    )
                
                # Error display for chunk overlap
                yield Static("", id="error-chunk-overlap", classes="error-message hidden")
            
            # Advanced Options Section
            with Collapsible(title="Advanced Options", id="embeddings-advanced-options", classes="embeddings-form-section"):
                with Horizontal(classes="embeddings-form-row"):
                    yield Label("Min Chunk Size:", classes="embeddings-form-label")
                    yield Input(
                        "100",
                        id="embeddings-min-chunk-size",
                        classes="embeddings-form-control",
                        type="integer"
                    )
                
                with Horizontal(classes="embeddings-form-row"):
                    yield Label("Max Chunk Size:", classes="embeddings-form-label")
                    yield Input(
                        "2048",
                        id="embeddings-max-chunk-size",
                        classes="embeddings-form-control",
                        type="integer"
                    )
                
                with Horizontal(classes="embeddings-form-row"):
                    yield Label("Language:", classes="embeddings-form-label")
                    yield Select(
                        [
                            ("Auto-detect", "auto"),
                            ("English", "en"),
                            ("Chinese", "zh"),
                            ("Japanese", "ja"),
                            ("Spanish", "es"),
                            ("French", "fr"),
                            ("German", "de"),
                            ("Russian", "ru"),
                            ("Arabic", "ar"),
                            ("Portuguese", "pt"),
                            ("Italian", "it"),
                            ("Korean", "ko"),
                            ("Dutch", "nl"),
                            ("Turkish", "tr"),
                            ("Polish", "pl"),
                            ("Swedish", "sv"),
                            ("Indonesian", "id"),
                            ("Vietnamese", "vi"),
                            ("Thai", "th"),
                            ("Hindi", "hi")
                        ],
                        id="embeddings-language",
                        classes="embeddings-form-control",
                        value="auto"
                    )
                
                yield Checkbox("Strip formatting before chunking", id="embeddings-strip-formatting", value=True)
                yield Checkbox("Remove stop words", id="embeddings-remove-stopwords", value=False)
                yield Checkbox("Apply stemming/lemmatization", id="embeddings-stemming", value=False)
            
            # Preview Section
            yield Label("Chunk Preview", classes="embeddings-section-title")
            yield ChunkPreview(id="embeddings-chunk-preview", classes="embeddings-chunk-preview")
    
    def _compose_output_tab(self) -> ComposeResult:
        """Compose the Output tab content."""
        # Collection Settings Section
        yield Label("Collection Settings", classes="embeddings-section-title")
        
        with Container(classes="embeddings-form-section"):
            with Horizontal(classes="embeddings-form-row"):
                yield Label("Collection Name:", classes="embeddings-form-label")
                yield Input(
                    placeholder="Enter collection name",
                    id="embeddings-collection-name",
                    classes="embeddings-form-control"
                )
            
            # Simplified help text
            yield Static("Choose a descriptive name for your collection", classes="embeddings-help-text")
            
            # Error display for collection name
            yield Static("", id="error-collection-name", classes="error-message hidden")
            
            with Horizontal(classes="embeddings-form-row"):
                yield Label("Description:", classes="embeddings-form-label")
                yield TextArea(
                    "",
                    id="embeddings-collection-description",
                    classes="embeddings-form-control"
                )
            
            yield Checkbox("Add timestamps to metadata", id="embeddings-add-timestamps", value=True)
            yield Checkbox("Include source information", id="embeddings-include-source", value=True)
        
        # Action Buttons
        yield Label("Actions", classes="embeddings-section-title")
        with Container(classes="embeddings-action-container"):
            with Horizontal(classes="embeddings-button-row"):
                yield Button("Clear Form", id="embeddings-clear", classes="embeddings-action-button")
                yield Button("Preview Chunks", id="embeddings-preview", classes="embeddings-action-button", variant="warning")
                yield Button("Create Embeddings", id="embeddings-create", classes="embeddings-action-button", variant="primary")
        
        # Progress Section (hidden by default)
        with Container(id="embeddings-progress-container", classes="embeddings-progress-container hidden"):
            yield Label("Processing...", id="embeddings-progress-label", classes="embeddings-progress-label")
            yield ProgressBar(id="embeddings-progress-bar", total=100)
            yield Static("", id="embeddings-progress-status", classes="embeddings-status-output")
        
        # Error display
        yield Static("", id="error-general", classes="error-message error-general hidden")
    
    def _get_available_models(self) -> List[str]:
        """Get list of available embedding models."""
        models = []
        
        # Add OpenAI models
        models.extend(["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"])
        
        # Get models from embedding config
        try:
            from ..config import load_settings
            settings = load_settings()
            embedding_config = settings.get('embedding_config', {})
            
            # Add configured models
            if embedding_config and embedding_config.get('models'):
                for model_id in embedding_config['models'].keys():
                    if model_id not in models:
                        models.append(model_id)
        except Exception as e:
            logger.error(f"Error loading embedding models from config: {e}")
        
        # Add some common models if none configured
        if len(models) == 3:  # Only OpenAI models
            models.extend([
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5"
            ])
        
        return models

    
    # --- Event Handlers ---
    
    @on(Select.Changed, "#embeddings-source-type")
    def on_source_type_changed(self, event: Select.Changed) -> None:
        """Handle source type selection change."""
        self.selected_source = event.value
        # Update content switcher
        switcher = self.query_one("#embeddings-source-switcher", ContentSwitcher)
        switcher.current = "file-input-container" if event.value == self.SOURCE_FILE else "db-input-container"
    
    @on(Select.Changed, "#embeddings-model-select")
    def on_model_selected(self, event: Select.Changed) -> None:
        """Handle model selection."""
        self.selected_model = event.value
        # Clear model validation error if any
        if "model" in self.validation_errors:
            del self.validation_errors["model"]
            self._display_errors(self.validation_errors)
        
        # Update smart defaults based on model
        self._update_smart_defaults()
    
    @on(Select.Changed, "#embeddings-db-select")
    def on_database_changed(self, event: Select.Changed) -> None:
        """Handle database selection change to update content types."""
        db_type_select = self.query_one("#embeddings-db-type", Select)
        
        if event.value == "media":
            # Media database content types
            db_type_select.set_options([
                ("Single Media Item", "single_media"),
                ("All Media Items", "all_media"),
                ("Media by Keywords", "media_keywords")
            ])
            db_type_select.value = "single_media"
        else:  # chachanotes
            # ChaChaNotes database content types
            db_type_select.set_options([
                ("Single Note", "single_note"),
                ("All Notes", "all_notes"),
                ("Notes by Keywords", "notes_keywords"),
                ("Single Conversation", "single_conversation"),
                ("All Conversations (Non-Character)", "all_conversations"),
                ("Conversations by Keywords", "conversations_keywords")
            ])
            db_type_select.value = "single_note"
        
        # Update the UI based on new content type
        self._update_selection_mode_ui()
    
    @on(Select.Changed, "#embeddings-db-type")
    def on_content_type_changed(self, event: Select.Changed) -> None:
        """Handle content type selection change to update UI."""
        self.selected_db_type = event.value
        self._update_selection_mode_ui()
    
    @on(Select.Changed, "#embeddings-db-mode-select")
    def on_mode_changed(self, event: Select.Changed) -> None:
        """Handle selection mode change."""
        self.selected_db_mode = event.value
        self._update_mode_containers()
    
    @on(Input.Changed, "#embeddings-chunk-size")
    def on_chunk_size_changed(self, event: Input.Changed) -> None:
        """Validate chunk size on change."""
        # Clear previous error
        if "chunk-size" in self.validation_errors:
            del self.validation_errors["chunk-size"]
        
        # Validate
        try:
            size = int(event.value) if event.value else 0
            if size < 50:
                self.validation_errors["chunk-size"] = "Chunk size must be at least 50"
            elif size > 10000:
                self.validation_errors["chunk-size"] = "Chunk size cannot exceed 10000"
        except ValueError:
            self.validation_errors["chunk-size"] = "Chunk size must be a number"
        
        self._display_errors(self.validation_errors)
    
    @on(Input.Changed, "#embeddings-chunk-overlap")
    def on_chunk_overlap_changed(self, event: Input.Changed) -> None:
        """Validate chunk overlap on change."""
        # Clear previous error
        if "chunk-overlap" in self.validation_errors:
            del self.validation_errors["chunk-overlap"]
        
        # Validate
        try:
            overlap = int(event.value) if event.value else 0
            chunk_size_input = self.query_one("#embeddings-chunk-size", Input)
            chunk_size = int(chunk_size_input.value) if chunk_size_input.value else 512
            
            if overlap < 0:
                self.validation_errors["chunk-overlap"] = "Overlap cannot be negative"
            elif overlap >= chunk_size:
                self.validation_errors["chunk-overlap"] = "Overlap must be less than chunk size"
        except ValueError:
            self.validation_errors["chunk-overlap"] = "Overlap must be a number"
        
        self._display_errors(self.validation_errors)
    
    @on(Input.Changed, "#embeddings-collection-name")
    def on_collection_name_changed(self, event: Input.Changed) -> None:
        """Validate collection name on change."""
        # Clear previous error
        if "collection-name" in self.validation_errors:
            del self.validation_errors["collection-name"]
        
        # Validate if provided
        if event.value:
            import re
            if not re.match(r'^[a-z0-9_]+$', event.value):
                self.validation_errors["collection-name"] = "Use only lowercase letters, numbers, and underscores"
        
        self._display_errors(self.validation_errors)
    
    @on(Button.Pressed, "#embeddings-clear")
    def on_clear_pressed(self, event: Button.Pressed) -> None:
        """Clear the form."""
        # Reset all inputs
        self.query_one("#embeddings-model-select", Select).value = None
        self.query_one("#embeddings-source-type", Select).value = self.SOURCE_FILE
        self.query_one("#embeddings-chunk-method", Select).value = "words"
        self.query_one("#embeddings-chunk-size", Input).value = "512"
        self.query_one("#embeddings-chunk-overlap", Input).value = "128"
        self.query_one("#embeddings-collection-name", Input).value = ""
        self.query_one("#embeddings-collection-description", TextArea).text = ""
        
        # Clear selections
        self.selected_files = []
        self.selected_db_items = set()
        self.validation_errors = {}
        
        # Update UI
        self.query_one("#embeddings-file-count", Label).update("Selected: 0 files")
        self.query_one("#embeddings-file-list", TextArea).text = ""
        self._display_errors({})
    
    @on(Button.Pressed, "#embeddings-preview")
    async def on_preview_pressed(self, event: Button.Pressed) -> None:
        """Preview chunks before creating embeddings."""
        if not self._validate_form():
            return
        
        # TODO: Implement chunk preview
        self.notify("Chunk preview not yet implemented", severity="warning")
    
    @on(Button.Pressed, "#embeddings-create")
    async def on_create_pressed(self, event: Button.Pressed) -> None:
        """Create embeddings."""
        if not self._validate_form():
            return
        
        if self.is_processing:
            self.notify("Already processing embeddings", severity="warning")
            return
        
        # Start processing
        self.is_processing = True
        self.run_worker(self._create_embeddings_worker)
    
    @on(Button.Pressed, "#embeddings-select-files")
    async def on_select_files_pressed(self, event: Button.Pressed) -> None:
        """Open file picker."""
        def handle_selected(paths: List[Path]) -> None:
            self.selected_files = paths
            self.query_one("#embeddings-file-count", Label).update(f"Selected: {len(paths)} files")
            
            # Update file list display
            file_list = self.query_one("#embeddings-file-list", TextArea)
            file_list.text = "\n".join(str(p) for p in paths)
        
        # Show file picker
        file_picker = FileOpen(
            title="Select Files for Embedding",
            filters=Filters(
                ("All Files", lambda p: True),
                ("Text Files", lambda p: p.suffix in [".txt", ".md", ".rst"]),
                ("Documents", lambda p: p.suffix in [".pdf", ".docx", ".doc"]),
                ("Code", lambda p: p.suffix in [".py", ".js", ".java", ".cpp", ".c", ".h"])
            ),
            select_multiple=True
        )
        
        self.app.push_screen(file_picker, handle_selected)
    
    # --- Helper Methods ---
    
    def _update_smart_defaults(self) -> None:
        """Update form defaults based on selected model."""
        if not self.selected_model:
            return
            
        # Smart defaults based on model type
        chunk_size_input = self.query_one("#embeddings-chunk-size", Input)
        chunk_overlap_input = self.query_one("#embeddings-chunk-overlap", Input)
        
        if "ada" in self.selected_model.lower() or "text-embedding" in self.selected_model:
            # OpenAI models - larger chunks
            chunk_size_input.value = "1024"
            chunk_overlap_input.value = "256"
        elif "sentence-transformers" in self.selected_model or "MiniLM" in self.selected_model:
            # Sentence transformers - smaller chunks
            chunk_size_input.value = "256"
            chunk_overlap_input.value = "64"
        elif "e5" in self.selected_model.lower():
            # E5 models - medium chunks
            chunk_size_input.value = "512"
            chunk_overlap_input.value = "128"
        else:
            # Default
            chunk_size_input.value = "512"
            chunk_overlap_input.value = "128"
    
    def _update_selection_mode_ui(self) -> None:
        """Update the UI based on the selected content type."""
        content_type = self.selected_db_type
        
        # First, update the selection mode options based on content type
        mode_select = self.query_one("#embeddings-db-mode-select", Select)
        
        if content_type in ["single_media", "single_note", "single_conversation"]:
            # Single item selection
            mode_select.set_options([
                ("Specific ID", "specific"),
                ("Search & Select", "search")
            ])
            mode_select.value = "specific"
        elif content_type in ["all_media", "all_notes", "all_conversations"]:
            # All items selection
            mode_select.set_options([
                ("All Items", "all"),
                ("Search & Filter", "search")
            ])
            mode_select.value = "all"
        else:  # Keywords-based selection
            mode_select.set_options([
                ("By Keywords", "keywords"),
                ("Search & Select", "search")
            ])
            mode_select.value = "keywords"
        
        # Update visibility of input containers based on mode
        self._update_mode_containers()
    
    def _update_mode_containers(self) -> None:
        """Update the visible mode container based on selection."""
        try:
            switcher = self.query_one("#embeddings-mode-switcher", ContentSwitcher)
            mode = self.selected_db_mode
            
            # Map mode to container ID
            mode_mapping = {
                "search": "mode-search",
                "specific": "mode-specific",
                "all": "mode-all",
                "keywords": "mode-keywords"
            }
            
            container_id = mode_mapping.get(mode, "mode-search")
            switcher.current = container_id
            
            # Update count for "all" mode
            if mode == "all":
                self._update_all_items_count()
        except Exception as e:
            logger.error(f"Error updating mode containers: {e}")
    
    def _update_all_items_count(self) -> None:
        """Update the count display for 'all items' mode."""
        try:
            count_static = self.query_one("#embeddings-all-count", Static)
            db_type = self.selected_db
            content_type = self.selected_db_type
            
            # Placeholder counts - these would be fetched from actual database
            if db_type == "media":
                count_static.update("Total items to process: [Loading...]")
                # TODO: Query actual count from MediaDatabase
            else:  # chachanotes
                if "note" in content_type:
                    count_static.update("Total notes to process: [Loading...]")
                    # TODO: Query actual count from CharactersRAGDB
                elif "conversation" in content_type:
                    count_static.update("Total conversations to process: [Loading...]")
                    # TODO: Query actual count from CharactersRAGDB
        except Exception as e:
            logger.error(f"Error updating item count: {e}")
    
    def _validate_form(self) -> bool:
        """Validate the entire form."""
        errors = {}
        
        # Check model selection
        if not self.selected_model:
            errors["model"] = "Please select an embedding model"
        
        # Check source selection
        if self.selected_source == self.SOURCE_FILE:
            if not self.selected_files:
                errors["files"] = "Please select at least one file"
        else:
            # Database source
            if self.selected_db_mode == "specific":
                specific_id = self.query_one("#embeddings-specific-id", Input).value
                if not specific_id:
                    errors["database"] = "Please enter an item ID"
            elif self.selected_db_mode == "keywords":
                keywords = self.query_one("#embeddings-keywords", Input).value
                if not keywords:
                    errors["database"] = "Please enter at least one keyword"
        
        # Check collection name (optional, but validate if provided)
        collection_name = self.query_one("#embeddings-collection-name", Input).value
        if collection_name:
            import re
            if not re.match(r'^[a-z0-9_]+$', collection_name):
                errors["collection-name"] = "Use only lowercase letters, numbers, and underscores"
        
        self.validation_errors = errors
        self._display_errors(errors)
        return len(errors) == 0
    
    def _display_errors(self, errors: dict) -> None:
        """Display validation errors."""
        # Hide all error messages first
        for error_id in ["error-model", "error-files", "error-database", 
                        "error-chunk-size", "error-chunk-overlap", "error-collection-name"]:
            try:
                error_widget = self.query_one(f"#{error_id}", Static)
                error_widget.update("")
                error_widget.add_class("hidden")
            except:
                pass
        
        # Show specific errors
        for field, message in errors.items():
            try:
                error_widget = self.query_one(f"#error-{field}", Static)
                error_widget.update(f"❌ {message}")
                error_widget.remove_class("hidden")
            except:
                pass
        
        # Show general error if any
        if errors:
            try:
                general_error = self.query_one("#error-general", Static)
                general_error.update("Please fix the errors above before proceeding")
                general_error.remove_class("hidden")
            except:
                pass
        else:
            try:
                general_error = self.query_one("#error-general", Static)
                general_error.update("")
                general_error.add_class("hidden")
            except:
                pass
    
    @work(thread=True)
    def _create_embeddings_worker(self) -> None:
        """Worker to create embeddings."""
        try:
            # TODO: Implement actual embedding creation
            import time
            time.sleep(2)  # Simulate processing
            
            self.call_from_thread(self.notify, "Embeddings created successfully!", severity="information")
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            self.call_from_thread(self.notify, f"Error: {str(e)}", severity="error")
        finally:
            self.is_processing = False
    
    def watch_is_processing(self, is_processing: bool) -> None:
        """React to processing state changes."""
        progress_container = self.query_one("#embeddings-progress-container", Container)
        if is_processing:
            progress_container.display = True
        else:
            progress_container.display = False
