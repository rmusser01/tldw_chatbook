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
    from ..Embeddings.Embeddings_Lib import EmbeddingFactory
    from ..Embeddings.Chroma_Lib import ChromaDBManager
    from ..Chunking.Chunk_Lib import chunk_for_embedding
else:
    EmbeddingFactory = None
    ChromaDBManager = None
    chunk_for_embedding = None

# Define available chunk methods
CHUNK_METHODS = ['words', 'sentences', 'paragraphs', 'tokens', 'semantic', 'json', 'xml', 'ebook_chapters', 'rolling_summarize']

if TYPE_CHECKING:
    from ..app import TldwCli

class EmbeddingsCreationContent(Container):
    """Content container for creating embeddings, designed to work within the Search tab."""
    
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
        
        # Check if embeddings dependencies are available
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            with Container(classes="embeddings-not-available"):
                yield Static("⚠️ Embeddings/RAG functionality not available", classes="warning-title")
                yield Static("The required dependencies for embeddings are not installed.", classes="warning-message")
                yield Static("To enable embeddings, please install with:", classes="warning-message")
                yield Static("pip install tldw_chatbook[embeddings_rag]", classes="code-block")
            return
            
        yield Label("Create Embeddings", classes="embeddings-form-title")
        
        # Main content in tabs
        with TabbedContent(id="embeddings-tabs"):
            # Tab 1: Source & Model
            with TabPane("Source & Model", id="tab-source-model"):
                with VerticalScroll():
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
                    
                    # Template selector
                    yield EmbeddingTemplateQuickSelect(id="embeddings-template-selector")
                    
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
                                        ("Media Content", "media")
                                    ],
                                    id="embeddings-db-type",
                                    classes="embeddings-form-control",
                                    allow_blank=False,
                                    value="media"
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
                            
                            # Search input (shown for search mode)
                            with Container(id="embeddings-search-container", classes="embeddings-mode-container"):
                                with Horizontal(classes="embeddings-form-row"):
                                    yield Label("Search:", classes="embeddings-form-label")
                                    yield Input(
                                        placeholder="Search for content...",
                                        id="embeddings-db-filter",
                                        classes="embeddings-form-control"
                                    )
                            
                            # Show results placeholder
                            yield TextArea(
                                "",
                                id="embeddings-db-results",
                                classes="embeddings-db-results",
                                read_only=True
                            )
                            
                            # Error display for database selection
                            yield Static("", id="error-database", classes="error-message hidden")
            
            # Tab 2: Processing
            with TabPane("Processing", id="tab-processing"):
                with VerticalScroll():
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
                    
                        # Advanced options collapsible
                        with Collapsible(title="Advanced Options", id="embeddings-advanced-options"):
                            with Horizontal(classes="embeddings-form-row"):
                                yield Label("Adaptive Chunking:", classes="embeddings-form-label")
                                yield Checkbox("Enable adaptive chunking", id="embeddings-adaptive-chunking", classes="embeddings-form-control")
                                yield HelpIcon(
                                    "Adaptive chunking adjusts chunk sizes\n"
                                    "based on content structure:\n\n"
                                    "• Respects sentence boundaries\n"
                                    "• Avoids splitting entities\n"
                                    "• May vary chunk sizes slightly\n\n"
                                    "Recommended for natural text",
                                    classes="help-icon-inline"
                                )
                    
                    # Chunk preview
                    with Collapsible(title="Chunk Preview", id="embeddings-chunk-preview-collapsible"):
                        yield ChunkPreview(
                            id="embeddings-chunk-preview",
                            max_chunks=5
                        )
            
            # Tab 3: Output
            with TabPane("Output", id="tab-output"):
                with VerticalScroll():
                    # Collection Settings Section
                    with Container(classes="embeddings-form-section"):
                        yield Label("Collection Settings", classes="embeddings-section-title")
                    
                        with Horizontal(classes="embeddings-form-row"):
                            yield Label("Collection Name:", classes="embeddings-form-label")
                            yield Input(
                                placeholder="my_embeddings",
                                id="embeddings-collection-name",
                                classes="embeddings-form-control"
                            )
                            yield HelpIcon(
                                "Collection naming rules:\n\n"
                                "• Use letters, numbers, underscore (_)\n"
                                "• Use hyphen (-) for readability\n"
                                "• No spaces allowed\n"
                                "• Must start with letter\n\n"
                                "Examples:\n"
                                "• product_docs_v2\n"
                                "• customer-feedback-2024\n"
                                "• research_papers",
                                classes="help-icon-inline"
                            )
                        
                        # Error display for collection name
                        yield Static("", id="error-collection-name", classes="error-message hidden")
                    
                        with Horizontal(classes="embeddings-form-row"):
                            yield Label("Description:", classes="embeddings-form-label")
                            yield Input(
                                placeholder="Optional description",
                                id="embeddings-collection-desc",
                                classes="embeddings-form-control"
                            )
                    
                        with Horizontal(classes="embeddings-form-row"):
                            yield Label("Overwrite:", classes="embeddings-form-label")
                            yield Checkbox("Overwrite if exists", id="embeddings-overwrite", classes="embeddings-form-control")
        
        # Progress Section - initially hidden
        with Container(id="embeddings-progress-section", classes="embeddings-progress-section hidden"):
            yield Rule()
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
        
        # Sticky footer with action buttons
        with Container(id="embeddings-footer", classes="embeddings-sticky-footer"):
            yield Rule()
            with Horizontal(classes="embeddings-button-group"):
                yield Button("Clear Form", id="embeddings-clear", classes="embeddings-action-button", variant="default")
                yield Button("Preview Chunks", id="embeddings-preview", classes="embeddings-action-button", variant="warning")
                yield Button("Create Embeddings", id="embeddings-create", classes="embeddings-action-button", variant="primary")
            
            # General error display at bottom
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
            logger.warning(f"Could not load models from config: {e}")
        
        # Scan user's models directory for embedding models
        try:
            # Get model directory from settings
            database_settings = settings.get('database', {})
            user_base_dir = Path(database_settings.get('USER_DB_BASE_DIR', '~/.local/share/tldw_cli')).expanduser()
            models_dir = user_base_dir / "models" / "embeddings"
            
            if models_dir.exists():
                # Look for downloaded embedding models
                for model_path in models_dir.glob("*"):
                    if model_path.is_dir():
                        model_name = f"local/{model_path.name}"
                        if model_name not in models:
                            models.append(model_name)
        except Exception as e:
            logger.warning(f"Could not scan models directory: {e}")
        
        # Add default local models if dependencies available
        if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False) and not models:
            models.extend([
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5"
            ])
        
        return models
    
    def watch_is_processing(self, is_processing: bool) -> None:
        """Show/hide progress section based on processing state."""
        try:
            progress_section = self.query_one("#embeddings-progress-section")
            if is_processing:
                progress_section.remove_class("hidden")
            else:
                progress_section.add_class("hidden")
        except Exception as e:
            logger.warning(f"Could not toggle progress section: {e}")
    
    def _validate_form(self) -> bool:
        """Validate all form inputs and return True if valid."""
        errors = {}
        
        # Validate model selection
        model_select = self.query_one("#embeddings-model-select", Select)
        if not model_select.value:
            errors["model"] = "Please select an embedding model"
        
        # Validate source selection
        if self.selected_source == self.SOURCE_FILE:
            if not self.selected_files:
                errors["files"] = "Please select at least one file"
        else:
            # Database source validation
            if self.selected_db_mode == "search" and not self.selected_db_items:
                errors["database"] = "Please select items from the database"
            elif self.selected_db_mode == "specific" and not self.specific_item_ids.strip():
                errors["database"] = "Please enter specific item IDs"
            elif self.selected_db_mode == "keywords" and not self.keyword_filter.strip():
                errors["database"] = "Please enter keywords to filter"
        
        # Validate chunk settings
        try:
            chunk_size_input = self.query_one("#embeddings-chunk-size", Input)
            chunk_size = int(chunk_size_input.value)
            if chunk_size < 50 or chunk_size > 10000:
                errors["chunk-size"] = "Chunk size must be between 50 and 10000"
        except (ValueError, AttributeError):
            errors["chunk-size"] = "Chunk size must be a valid number"
        
        try:
            overlap_input = self.query_one("#embeddings-chunk-overlap", Input)
            overlap = int(overlap_input.value)
            if overlap < 0:
                errors["chunk-overlap"] = "Overlap cannot be negative"
            elif 'chunk_size' in locals() and overlap >= chunk_size:
                errors["chunk-overlap"] = "Overlap must be less than chunk size"
        except (ValueError, AttributeError):
            errors["chunk-overlap"] = "Overlap must be a valid number"
        
        # Validate collection name
        collection_input = self.query_one("#embeddings-collection-name", Input)
        collection_name = collection_input.value.strip()
        if not collection_name:
            errors["collection-name"] = "Collection name is required"
        elif " " in collection_name:
            errors["collection-name"] = "Collection name cannot contain spaces"
        elif not collection_name.replace("_", "").replace("-", "").isalnum():
            errors["collection-name"] = "Collection name can only contain letters, numbers, underscores, and hyphens"
        
        # Update error displays
        self._display_errors(errors)
        self.validation_errors = errors
        
        return len(errors) == 0
    
    def _display_errors(self, errors: dict) -> None:
        """Display validation errors in the UI."""
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
                logger.warning(f"Could not display error for field: {field}")
        
        # Show general error if any validation failed
        if errors:
            try:
                general_error = self.query_one("#error-general", Static)
                general_error.update("⚠️ Please fix the errors above before proceeding")
                general_error.remove_class("hidden")
            except:
                pass
        else:
            try:
                general_error = self.query_one("#error-general", Static)
                general_error.add_class("hidden")
            except:
                pass
    
    # Event handlers
    @on(EmbeddingTemplateSelected)
    async def on_template_selected(self, event: EmbeddingTemplateSelected) -> None:
        """Handle template selection."""
        template = event.template
        config = template.config
        
        # Apply template configuration to form fields
        try:
            # Update chunk size
            if "chunk_size" in config:
                chunk_size_input = self.query_one("#embeddings-chunk-size", Input)
                chunk_size_input.value = str(config["chunk_size"])
            
            # Update chunk overlap
            if "chunk_overlap" in config:
                overlap_input = self.query_one("#embeddings-chunk-overlap", Input)
                overlap_input.value = str(config["chunk_overlap"])
            
            # Update batch size
            if "batch_size" in config:
                batch_input = self.query_one("#embeddings-batch-size", Input)
                batch_input.value = str(config["batch_size"])
            
            # Update model selection if specified
            if "model_id" in config:
                model_select = self.query_one("#embeddings-model-select", Select)
                # Check if model exists in options
                for option in model_select._options:
                    if option[0] == config["model_id"]:
                        model_select.value = config["model_id"]
                        break
            
            # Update normalize embeddings
            if "normalize_embeddings" in config:
                normalize_cb = self.query_one("#embeddings-normalize", Checkbox)
                normalize_cb.value = config["normalize_embeddings"]
            
            # Notify user
            self.notify(f"Applied template: {template.name}", severity="success")
            
            logger.info(f"Applied embedding template: {template.name}")
            
        except Exception as e:
            logger.error(f"Error applying template: {e}")
            self.notify(f"Error applying template: {str(e)}", severity="error")
    
    @on(Button.Pressed, "#embeddings-create")
    async def on_create_embeddings(self, event: Button.Pressed) -> None:
        """Handle create embeddings button press."""
        event.stop()
        
        # Validate form first
        if not self._validate_form():
            self.app_instance.notify("Please fix validation errors before proceeding", severity="error")
            return
        
        # Start processing
        self.is_processing = True
        self.run_worker(self._create_embeddings_worker, thread=True)
    
    @on(Button.Pressed, "#embeddings-preview")
    async def on_preview_chunks(self, event: Button.Pressed) -> None:
        """Handle preview chunks button press."""
        event.stop()
        
        # Basic validation for preview
        if self.selected_source == self.SOURCE_FILE and not self.selected_files:
            self.app_instance.notify("Please select files first", severity="warning")
            return
        
        # Get chunking parameters
        chunk_method = self.query_one("#embeddings-chunk-method", Select).value
        chunk_size = int(self.query_one("#embeddings-chunk-size", Input).value or "512")
        chunk_overlap = int(self.query_one("#embeddings-chunk-overlap", Input).value or "128")
        
        # Simple preview implementation
        preview_widget = self.query_one("#embeddings-chunk-preview", ChunkPreview)
        
        # Create sample chunks (simplified)
        sample_text = "This is a sample text for chunk preview. " * 50  # Sample text
        chunks = []
        
        if chunk_method == "words":
            words = sample_text.split()
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                chunks.append((chunk_text, i, min(i + chunk_size, len(words))))
                if len(chunks) >= 5:  # Limit preview
                    break
        else:
            # Simple character-based chunking for other methods
            for i in range(0, len(sample_text), chunk_size - chunk_overlap):
                chunk_text = sample_text[i:i + chunk_size]
                chunks.append((chunk_text, i, min(i + chunk_size, len(sample_text))))
                if len(chunks) >= 5:
                    break
        
        # Update preview
        preview_widget.update_chunks(chunks, chunk_overlap)
        self.app_instance.notify("Chunk preview updated", severity="information")
    
    @on(Button.Pressed, "#embeddings-clear")
    async def on_clear_form(self, event: Button.Pressed) -> None:
        """Handle clear form button press."""
        event.stop()
        
        # Reset all form fields
        try:
            self.query_one("#embeddings-model-select", Select).value = None
            self.query_one("#embeddings-source-type", Select).value = self.SOURCE_FILE
            self.query_one("#embeddings-chunk-method", Select).value = "words"
            self.query_one("#embeddings-chunk-size", Input).value = "512"
            self.query_one("#embeddings-chunk-overlap", Input).value = "128"
            self.query_one("#embeddings-adaptive-chunking", Checkbox).value = False
            self.query_one("#embeddings-collection-name", Input).value = ""
            self.query_one("#embeddings-collection-desc", Input).value = ""
            self.query_one("#embeddings-overwrite", Checkbox).value = False
            self.query_one("#embeddings-file-list", TextArea).text = ""
            self.query_one("#embeddings-db-filter", Input).value = ""
            self.query_one("#embeddings-db-results", TextArea).text = ""
            
            # Reset reactive attributes
            self.selected_files = []
            self.selected_db_items = set()
            self.validation_errors = {}
            
            # Clear all error messages
            self._display_errors({})
            
            self.app_instance.notify("Form cleared", severity="information")
        except Exception as e:
            logger.error(f"Error clearing form: {e}")
            self.app_instance.notify("Error clearing form", severity="error")
    
    @on(Button.Pressed, "#embeddings-select-files")
    async def on_select_files(self, event: Button.Pressed) -> None:
        """Handle file selection button press."""
        event.stop()
        
        # Use file picker to select files
        from ..Third_Party.textual_fspicker import Filters
        from pathlib import Path
        
        # TODO: Implement file picker callback
        await self.app_instance.push_screen(
            FileOpen(
                location=str(Path.home()),
                title="Select Files for Embeddings",
                filters=Filters(("All Files", lambda p: p.is_file())),
                multiple=True
            ),
            callback=self._handle_file_selection
        )
    
    async def _handle_file_selection(self, paths: Optional[List[Path]]) -> None:
        """Handle file selection from file picker."""
        if paths:
            self.selected_files = paths
            # Update file list display
            file_list = self.query_one("#embeddings-file-list", TextArea)
            file_list.text = "\n".join(str(p) for p in paths)
            
            # Update file count
            file_count = self.query_one("#embeddings-file-count", Label)
            file_count.update(f"Selected: {len(paths)} files")
    
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
    
    @on(Input.Changed, "#embeddings-chunk-size")
    def on_chunk_size_changed(self, event: Input.Changed) -> None:
        """Validate chunk size on change."""
        # Clear previous error
        if "chunk-size" in self.validation_errors:
            del self.validation_errors["chunk-size"]
            self._display_errors(self.validation_errors)
    
    @on(Input.Changed, "#embeddings-chunk-overlap")
    def on_chunk_overlap_changed(self, event: Input.Changed) -> None:
        """Validate chunk overlap on change."""
        # Clear previous error
        if "chunk-overlap" in self.validation_errors:
            del self.validation_errors["chunk-overlap"]
            self._display_errors(self.validation_errors)
    
    @on(Input.Changed, "#embeddings-collection-name")
    def on_collection_name_changed(self, event: Input.Changed) -> None:
        """Validate collection name on change."""
        # Clear previous error
        if "collection-name" in self.validation_errors:
            del self.validation_errors["collection-name"]
            self._display_errors(self.validation_errors)
    
    @work(thread=True)
    def _create_embeddings_worker(self) -> None:
        """Worker to create embeddings in background."""
        try:
            # TODO: Implement actual embedding creation
            # This is a placeholder
            import time
            time.sleep(2)  # Simulate work
            
            self.call_from_thread(self.app_instance.notify, 
                                "Embeddings creation not yet implemented", 
                                severity="information")
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            self.call_from_thread(self.app_instance.notify, 
                                f"Error creating embeddings: {str(e)}", 
                                severity="error")
        finally:
            self.call_from_thread(setattr, self, "is_processing", False)
    
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