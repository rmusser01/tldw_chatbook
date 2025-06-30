# Embeddings_Creation_Window.py
# Description: Embeddings Creation interface with single-pane form layout
#
# Imports
from __future__ import annotations

from typing import Optional, List, Dict, Any
from pathlib import Path
from loguru import logger

# Third-party imports
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button, Input, Label, Select, TextArea, Checkbox, RadioButton, RadioSet,
    Collapsible, LoadingIndicator, ProgressBar, Static, DataTable, Rule
)

# Local imports
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Third_Party.textual_fspicker import FileOpen, Filters

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

logger = logger.bind(name="Embeddings_Creation_Window")

########################################################################################################################
#
# Embeddings Creation Window
#
########################################################################################################################

class EmbeddingsCreationWindow(Widget):
    """Embeddings Creation window with single-pane form layout."""
    
    DEFAULT_CSS = """
    EmbeddingsCreationWindow {
        layout: vertical;
        height: 100%;
        width: 100%;
    }
    
    .embeddings-creation-scroll {
        width: 100%;
        height: 1fr;
        padding: 1 2;
    }
    
    .embeddings-form-container {
        width: 100%;
        max-width: 80;
        margin: 0;
        align: center middle;
    }
    
    .embeddings-form-title {
        text-style: bold;
        text-align: center;
        background: $accent-darken-1;
        color: $text;
        padding: 1;
        margin-bottom: 2;
        border: round $accent;
    }
    
    .embeddings-section-title {
        text-style: bold;
        margin-top: 2;
        margin-bottom: 1;
        background: $primary-background-lighten-1;
        padding: 0 1;
        border-left: thick $accent;
    }
    
    .embeddings-form-row {
        layout: horizontal;
        margin-bottom: 1;
        height: 3;
        align: left middle;
    }
    
    .embeddings-form-label {
        width: 30%;
        padding-right: 1;
        text-align: right;
    }
    
    .embeddings-form-control {
        width: 70%;
    }
    
    .embeddings-form-full-row {
        width: 100%;
        margin-bottom: 1;
    }
    
    .embeddings-input-source-container {
        padding: 1;
        border: round $surface;
        margin-bottom: 1;
    }
    
    .embeddings-file-list {
        height: 10;
        border: round $primary;
        background: $surface;
        padding: 1;
        margin-bottom: 1;
    }
    
    .embeddings-chunk-preview {
        height: 15;
        border: round $primary-lighten-1;
        background: $surface-darken-1;
        padding: 1;
        margin-top: 1;
    }
    
    .embeddings-progress-container {
        margin-top: 2;
        padding: 1;
        border: round $primary;
        background: $surface;
    }
    
    .embeddings-progress-label {
        text-align: center;
        margin-bottom: 1;
    }
    
    .embeddings-action-buttons {
        layout: horizontal;
        margin-top: 2;
        align-horizontal: center;
    }
    
    .embeddings-action-button {
        margin: 0 1;
        min-width: 15;
    }
    
    .embeddings-status-output {
        margin-top: 2;
        height: 10;
        border: round $primary-background-lighten-2;
        padding: 1;
        background: $surface;
    }
    
    .embeddings-radio-group {
        layout: horizontal;
        align: left middle;
    }
    
    .embeddings-radio-group RadioButton {
        margin-right: 2;
    }
    
    /* Initially hide file and database input containers */
    #file-input-container {
        display: none;
    }
    
    #db-input-container {
        display: none;
    }
    
    #embeddings-progress-container {
        display: none;
    }
    """
    
    # Input source types
    SOURCE_TEXT = "text"
    SOURCE_FILE = "file"
    SOURCE_DATABASE = "database"
    
    # Reactive attributes
    selected_source: reactive[str] = reactive(SOURCE_TEXT)
    selected_model: reactive[Optional[str]] = reactive(None)
    is_processing: reactive[bool] = reactive(False)
    selected_files: reactive[List[Path]] = reactive([])
    
    def __init__(self, app_instance: Any, **kwargs):
        """Initialize the Embeddings Creation Window.
        
        Args:
            app_instance: The main app instance
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.chachanotes_db = app_instance.chachanotes_db if hasattr(app_instance, 'chachanotes_db') else None
        self.media_db = app_instance.media_db if hasattr(app_instance, 'media_db') else None
        self.embedding_factory: Optional[EmbeddingFactory] = None
        self.chroma_manager: Optional[ChromaDBManager] = None
        
        # Check dependencies
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            logger.warning("Embeddings/RAG dependencies not available")
    
    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        with VerticalScroll(classes="embeddings-creation-scroll"):
            with Container(classes="embeddings-form-container"):
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
                
                with RadioSet(id="embeddings-source-type", classes="embeddings-radio-group"):
                    yield RadioButton("Direct Text", value=self.SOURCE_TEXT, id="source-text-radio")
                    yield RadioButton("Files", value=self.SOURCE_FILE, id="source-file-radio")
                    yield RadioButton("Database Query", value=self.SOURCE_DATABASE, id="source-db-radio")
                
                # Text input container
                with Container(id="text-input-container", classes="embeddings-input-source-container"):
                    yield TextArea(
                        "Enter text to embed...",
                        id="embeddings-text-input",
                        classes="embeddings-form-full-row"
                    )
                
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
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Content Type:", classes="embeddings-form-label")
                        yield Select(
                            [
                                ("media", "Media Content"),
                                ("conversations", "Conversations"),
                                ("notes", "Notes"),
                                ("characters", "Characters")
                            ],
                            id="embeddings-db-type",
                            classes="embeddings-form-control"
                        )
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Filter (optional):", classes="embeddings-form-label")
                        yield Input(
                            placeholder="e.g., keyword search",
                            id="embeddings-db-filter",
                            classes="embeddings-form-control"
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
                
                # Action Buttons
                with Horizontal(classes="embeddings-action-buttons"):
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
    
    async def on_mount(self) -> None:
        """Handle mount event - initialize embeddings components."""
        await self._initialize_embeddings()
        
        # Ensure text input container is visible by default
        text_container = self.query_one("#text-input-container")
        text_container.styles.display = "block"
    
    async def _initialize_embeddings(self) -> None:
        """Initialize embedding factory and ChromaDB manager."""
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            self._update_status("Embeddings dependencies not installed. Install with: pip install tldw_chatbook[embeddings_rag]")
            return
        
        try:
            # Load embedding configuration
            from ..config import get_cli_setting
            embedding_config = get_cli_setting('embedding_config', {})
            
            if embedding_config:
                self.embedding_factory = EmbeddingFactory(
                    embedding_config,
                    max_cached=2,
                    idle_seconds=900
                )
                logger.info("Initialized embedding factory")
            else:
                logger.warning("No embedding configuration found")
                self._update_status("Warning: No embedding configuration found")
                
            # Initialize ChromaDB manager
            user_id = get_cli_setting('users_name', 'default_user')
            self.chroma_manager = ChromaDBManager(user_id, embedding_config)
            logger.info("Initialized ChromaDB manager")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self._update_status(f"Error: Failed to initialize embeddings: {str(e)}")
    
    def _get_available_models(self) -> List[str]:
        """Get list of available embedding models."""
        if self.embedding_factory and hasattr(self.embedding_factory.config, 'models'):
            return list(self.embedding_factory.config.models.keys())
        return ["No models available"]
    
    def _get_chunk_methods(self) -> List[tuple[str, str]]:
        """Get available chunking methods."""
        return [(method, method.replace('_', ' ').title()) for method in CHUNK_METHODS]
    
    # Event handlers
    @on(RadioSet.Changed, "#embeddings-source-type")
    def on_source_changed(self, event: RadioSet.Changed) -> None:
        """Handle source type change."""
        self.selected_source = str(event.value)
        
        # Show/hide appropriate containers
        text_container = self.query_one("#text-input-container")
        file_container = self.query_one("#file-input-container")
        db_container = self.query_one("#db-input-container")
        
        text_container.styles.display = "block" if self.selected_source == self.SOURCE_TEXT else "none"
        file_container.styles.display = "block" if self.selected_source == self.SOURCE_FILE else "none"
        db_container.styles.display = "block" if self.selected_source == self.SOURCE_DATABASE else "none"
    
    @on(Select.Changed, "#embeddings-model-select")
    def on_model_changed(self, event: Select.Changed) -> None:
        """Handle model selection change."""
        if event.value and event.value != Select.BLANK:
            self.selected_model = str(event.value)
            self._update_status(f"Selected model: {self.selected_model}")
    
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
        # Clear text input
        self.query_one("#embeddings-text-input", TextArea).text = "Enter text to embed..."
        
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
        
        # Reset source to text
        radio_set = self.query_one("#embeddings-source-type", RadioSet)
        radio_set.pressed_index = 0
        
        self.notify("Form cleared", severity="information")
    
    async def _get_input_text(self) -> str:
        """Get input text based on selected source."""
        if self.selected_source == self.SOURCE_TEXT:
            text_area = self.query_one("#embeddings-text-input", TextArea)
            return text_area.text.strip()
        
        elif self.selected_source == self.SOURCE_FILE:
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
            # Query database based on type and filter
            db_type = str(self.query_one("#embeddings-db-type", Select).value)
            filter_text = self.query_one("#embeddings-db-filter", Input).value
            
            # This would be implemented based on actual database queries
            # For now, return empty string
            return ""
        
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

# End of Embeddings_Creation_Window.py
########################################################################################################################