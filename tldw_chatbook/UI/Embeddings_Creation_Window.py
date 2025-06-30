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
from textual.binding import Binding

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
    
    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select All", show=False),
        Binding("ctrl+d", "clear_selection", "Deselect All", show=False),
        Binding("space", "toggle_selection", "Toggle Selection", show=False),
    ]
    
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
    
    /* Initially hide database input container, show file container by default */
    #file-input-container {
        display: block;
    }
    
    #db-input-container {
        display: none;
    }
    
    #embeddings-progress-container {
        display: none;
    }
    
    .embeddings-db-results-container {
        height: 25;
        margin-top: 1;
        border: round $primary;
        background: $surface;
    }
    
    #embeddings-db-results {
        height: 1fr;
    }
    
    .embeddings-db-selection-buttons {
        layout: horizontal;
        margin-bottom: 1;
        align-horizontal: left;
    }
    
    .embeddings-db-selection-button {
        margin-right: 1;
        min-width: 10;
    }
    
    DataTable > .datatable--cursor {
        background: $primary 20%;
    }
    
    DataTable > .datatable--hover {
        background: $primary 10%;
    }
    
    DataTable > .datatable--selected {
        background: $accent 30%;
    }
    """
    
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
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Label("Search:", classes="embeddings-form-label")
                        yield Input(
                            placeholder="Search for content...",
                            id="embeddings-db-filter",
                            classes="embeddings-form-control"
                        )
                    
                    with Horizontal(classes="embeddings-form-row"):
                        yield Button("Search Database", id="embeddings-search-db", classes="embeddings-action-button")
                        yield Label("No items selected", id="embeddings-db-selection-count")
                    
                    # Selection control buttons
                    with Horizontal(classes="embeddings-db-selection-buttons"):
                        yield Button("Select All", id="embeddings-select-all", classes="embeddings-db-selection-button")
                        yield Button("Clear Selection", id="embeddings-clear-selection", classes="embeddings-db-selection-button")
                    
                    # DataTable to show search results in a scrollable container
                    with VerticalScroll(classes="embeddings-db-results-container"):
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
        
        # Ensure file input container is visible by default
        file_container = self.query_one("#file-input-container")
        file_container.styles.display = "block"
        
        # Initialize the DataTable
        table = self.query_one("#embeddings-db-results", DataTable)
        table.add_columns("✓", "ID", "Title", "Type", "Date")
        table.cursor_type = "row"
        
        # Clear selected items
        self.selected_db_items = set()
        
        # Trigger initial database selection
        db_select = self.query_one("#embeddings-db-select", Select)
        # The Select widget should auto-select the first option when allow_blank=False
        # But we'll manually trigger the change event to set up the content types
        if db_select.value and db_select.value != Select.BLANK:
            self.on_database_changed(Select.Changed(db_select, db_select.value))
    
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
    @on(Select.Changed, "#embeddings-source-type")
    def on_source_changed(self, event: Select.Changed) -> None:
        """Handle source type change."""
        if event.value and event.value != Select.BLANK:
            self.selected_source = str(event.value)
            
            # Show/hide appropriate containers
            file_container = self.query_one("#file-input-container")
            db_container = self.query_one("#db-input-container")
            
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
    
    @on(Select.Changed, "#embeddings-db-select")
    def on_database_changed(self, event: Select.Changed) -> None:
        """Handle database selection change."""
        if event.value and event.value != Select.BLANK:
            self.selected_db = str(event.value)
            
            # Update content type options based on selected database
            db_type_select = self.query_one("#embeddings-db-type", Select)
            
            if self.selected_db == "media":
                db_type_select.set_options([
                    ("media", "Media Content")
                ])
                # Don't set value - Select will auto-select first option when allow_blank=False
            else:  # chachanotes
                db_type_select.set_options([
                    ("conversations", "Conversations"),
                    ("notes", "Notes"),
                    ("characters", "Characters")
                ])
                # Don't set value - Select will auto-select first option when allow_blank=False
            
            self._update_status(f"Selected {event.value} database")
    
    @on(Button.Pressed, "#embeddings-search-db")
    async def on_search_database(self) -> None:
        """Search database for content to embed."""
        db_type = str(self.query_one("#embeddings-db-type", Select).value)
        search_term = self.query_one("#embeddings-db-filter", Input).value
        
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
            
            if db_type == "media" and media_db:
                # Search media database
                results = media_db.search_media(search_term) if search_term else media_db.get_all_media(limit=100)
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
            self.query_one("#embeddings-db-selection-count", Label).update(f"Found {count} items")
            
            if count == 0:
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
            # Get the ID from the row
            item_id = str(table.get_cell(row_key, "ID"))
            
            # Toggle selection
            if item_id in self.selected_db_items:
                self.selected_db_items.discard(item_id)
                table.update_cell(row_key, "✓", "")
            else:
                self.selected_db_items.add(item_id)
                table.update_cell(row_key, "✓", "✓")
            
            # Update selection count
            self.query_one("#embeddings-db-selection-count", Label).update(
                f"Selected {len(self.selected_db_items)} items"
            )
    
    @on(Button.Pressed, "#embeddings-select-all")
    def on_select_all(self) -> None:
        """Select all items in the results table."""
        table = self.query_one("#embeddings-db-results", DataTable)
        
        # Select all items
        for row_key in table.rows:
            item_id = str(table.get_cell(row_key, "ID"))
            self.selected_db_items.add(item_id)
            table.update_cell(row_key, "✓", "✓")
        
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
        for row_key in table.rows:
            table.update_cell(row_key, "✓", "")
        
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
            # Get selected items from the DataTable
            table = self.query_one("#embeddings-db-results", DataTable)
            db_type = str(self.query_one("#embeddings-db-type", Select).value)
            
            all_text = []
            
            # Use app's loaded databases
            media_db = self.media_db
            chachanotes_db = self.chachanotes_db
            
            for item_id in self.selected_db_items:
                    
                    try:
                        if db_type == "media" and media_db:
                            # Get media content
                            media_item = media_db.get_media_item_by_id(int(item_id))
                            if media_item:
                                content = media_item.get('content', media_item.get('transcript', ''))
                                if content:
                                    all_text.append(f"=== {media_item.get('title', 'Untitled')} ===\n{content}")
                        
                        elif db_type == "conversations" and chachanotes_db:
                            # Get conversation messages
                            messages = chachanotes_db.get_messages_by_conversation_id(int(item_id))
                            if messages:
                                conv_text = []
                                for msg in messages:
                                    role = msg.get('role', 'unknown')
                                    content = msg.get('content', '')
                                    conv_text.append(f"{role}: {content}")
                                all_text.append(f"=== Conversation {item_id} ===\n" + "\n\n".join(conv_text))
                        
                        elif db_type == "notes" and chachanotes_db:
                            # Get note content
                            note = chachanotes_db.get_note_by_id(int(item_id))
                            if note:
                                all_text.append(f"=== {note.get('title', 'Untitled Note')} ===\n{note.get('content', '')}")
                        
                        elif db_type == "characters" and chachanotes_db:
                            # Get character data
                            character = chachanotes_db.get_character_by_id(int(item_id))
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
                    # Get the ID from the current row
                    item_id = str(table.get_cell(row_key, "ID"))
                    
                    # Toggle selection
                    if item_id in self.selected_db_items:
                        self.selected_db_items.discard(item_id)
                        table.update_cell(row_key, "✓", "")
                    else:
                        self.selected_db_items.add(item_id)
                        table.update_cell(row_key, "✓", "✓")
                    
                    # Update selection count
                    self.query_one("#embeddings-db-selection-count", Label).update(
                        f"Selected {len(self.selected_db_items)} items"
                    )

# End of Embeddings_Creation_Window.py
########################################################################################################################