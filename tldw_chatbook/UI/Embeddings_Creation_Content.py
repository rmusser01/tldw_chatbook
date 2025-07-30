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
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.reactive import reactive
from textual.widgets import (
    Static, Button, Input, Label, Select, TextArea, Checkbox, RadioButton, RadioSet,
    Collapsible, ProgressBar, Rule, ContentSwitcher
)

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
                    
                    # Show results placeholder
                    yield TextArea(
                        "",
                        id="embeddings-db-results",
                        classes="embeddings-db-results",
                        read_only=True
                    )
            
            yield Rule()
            
            # Chunking Configuration Section
            yield Label("Chunking Configuration", classes="embeddings-section-title")
            
            with Horizontal(classes="embeddings-form-row"):
                yield Label("Chunk Method:", classes="embeddings-form-label")
                yield Select(
                    [(method.replace('_', ' ').title(), method) for method in CHUNK_METHODS],
                    id="embeddings-chunk-method",
                    classes="embeddings-form-control",
                    value="words"
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
            
            # Action buttons
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
    
    def _get_available_models(self) -> List[str]:
        """Get list of available embedding models."""
        models = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
        
        # Add local models if available
        if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            models.extend([
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5"
            ])
        
        return models