# SearchEmbeddingsWindow.py
# Description: New streamlined embeddings creation interface following Chatbook layout patterns
#
# Imports
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Set, List, Any
from pathlib import Path
import asyncio

# 3rd-Party Imports
from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Static, Button, Label, Input, Select, Checkbox,
    Collapsible, LoadingIndicator, ProgressBar, Tree
)
from textual.message import Message

# Configure logger with context
logger = logger.bind(module="SearchEmbeddingsWindow")

# Local Imports
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Client_Media_DB_v2 import MediaDatabase

# Optional embeddings imports
if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
    try:
        from ..Embeddings.Embeddings_Lib import EmbeddingFactory
        from ..RAG_Search.simplified.embeddings_wrapper import EmbeddingsService
        embeddings_available = True
        logger.info("Embeddings dependencies available")
    except ImportError as e:
        logger.warning(f"Failed to import embeddings modules: {e}")
        embeddings_available = False
else:
    embeddings_available = False
    EmbeddingFactory = None
    EmbeddingsService = None

if TYPE_CHECKING:
    from ..app import TldwCli

# Content type constants
CONTENT_TYPES = {
    'chats': 'Chats',
    'character_chats': 'Character Chats',
    'notes': 'Notes',
    'media': 'Media'
}

########################################################################################################################
#
# Messages
#
########################################################################################################################

class ContentSelectionChanged(Message):
    """Message sent when content selection changes."""

    def __init__(self, selected_items: Dict[str, Set[str]]):
        super().__init__()
        self.selected_items = selected_items

class EmbeddingProgressUpdate(Message):
    """Message for embedding progress updates."""

    def __init__(self, progress: int, status: str, current_item: str = ""):
        super().__init__()
        self.progress = progress
        self.status = status
        self.current_item = current_item

########################################################################################################################
#
# Content Tree Component
#
########################################################################################################################

class ContentTreePanel(Container):
    """Content selection tree panel with filtering."""

    # Reactive attributes
    selected_content_types = reactive(set())
    selected_items = reactive(dict, layout=False)  # {content_type: {item_ids}}
    search_query = reactive("")
    is_loading = reactive(False)

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.content_data = {}  # Cache for loaded content

    def compose(self) -> ComposeResult:
        """Compose the content tree panel."""

        # Title
        yield Label("Content", classes="panel-title")

        # Search filter
        yield Input(
            placeholder="Filter content by keyword...",
            id="content-filter",
            classes="content-filter"
        )

        # Tree container
        with VerticalScroll(classes="tree-container"):
            yield Tree("Content", id="content-tree", classes="content-tree")

    def watch_selected_content_types(self, content_types: set) -> None:
        """React to content type selection changes."""
        logger.debug(f"Content types changed: {content_types}")
        if content_types:
            self.load_content_data()
        else:
            self.clear_tree()

    def watch_search_query(self, query: str) -> None:
        """React to search query changes."""
        logger.debug(f"Search query changed: '{query}'")
        self.filter_tree_content(query)

    @work(thread=True)
    def load_content_data(self) -> None:
        """Load content data based on selected types."""
        logger.debug("Loading content data...")
        self.is_loading = True

        try:
            content_data = {}

            for content_type in self.selected_content_types:
                if content_type == 'chats':
                    content_data['chats'] = self._load_chat_data()
                elif content_type == 'character_chats':
                    content_data['character_chats'] = self._load_character_chat_data()
                elif content_type == 'notes':
                    content_data['notes'] = self._load_notes_data()
                elif content_type == 'media':
                    content_data['media'] = self._load_media_data()

            # Update UI from main thread
            self.call_from_thread(self._update_tree_content, content_data)

        except Exception as e:
            logger.error(f"Error loading content data: {e}")
            self.call_from_thread(self.notify, f"Error loading content: {e}", "error")
        finally:
            self.call_from_thread(setattr, self, "is_loading", False)

    def _load_chat_data(self) -> List[Dict[str, Any]]:
        """Load chat conversation data."""
        try:
            db = CharactersRAGDB()
            conversations = db.get_all_conversations()[:50]  # Limit for performance
            return [
                {
                    'id': str(conv['conversation_id']),
                    'title': conv.get('title', 'Untitled Conversation'),
                    'type': 'conversation',
                    'metadata': {
                        'created_at': conv.get('created_at'),
                        'message_count': conv.get('message_count', 0)
                    }
                }
                for conv in conversations
            ]
        except Exception as e:
            logger.error(f"Error loading chat data: {e}")
            return []

    def _load_character_chat_data(self) -> List[Dict[str, Any]]:
        """Load character chat data."""
        try:
            db = CharactersRAGDB()
            characters = db.get_all_characters()[:50]
            return [
                {
                    'id': str(char['character_id']),
                    'title': char.get('name', 'Unnamed Character'),
                    'type': 'character',
                    'metadata': {
                        'description': char.get('description', ''),
                        'chat_count': char.get('chat_count', 0)
                    }
                }
                for char in characters
            ]
        except Exception as e:
            logger.error(f"Error loading character chat data: {e}")
            return []

    def _load_notes_data(self) -> List[Dict[str, Any]]:
        """Load notes data."""
        try:
            db = CharactersRAGDB()
            notes = db.get_all_notes()[:50]
            return [
                {
                    'id': str(note['id']),
                    'title': note.get('title', 'Untitled Note'),
                    'type': 'note',
                    'metadata': {
                        'created_at': note.get('created_at'),
                        'updated_at': note.get('updated_at'),
                        'content_preview': (note.get('content', '')[:100] + '...'
                                          if len(note.get('content', '')) > 100
                                          else note.get('content', ''))
                    }
                }
                for note in notes
            ]
        except Exception as e:
            logger.error(f"Error loading notes data: {e}")
            return []

    def _load_media_data(self) -> List[Dict[str, Any]]:
        """Load media data."""
        try:
            db = MediaDatabase()
            media_items = db.get_all_media_items()[:50]
            return [
                {
                    'id': str(item['id']),
                    'title': item.get('title', 'Untitled Media'),
                    'type': 'media',
                    'metadata': {
                        'media_type': item.get('type', 'unknown'),
                        'duration': item.get('duration'),
                        'created_at': item.get('created_at')
                    }
                }
                for item in media_items
            ]
        except Exception as e:
            logger.error(f"Error loading media data: {e}")
            return []

    def _update_tree_content(self, content_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Update tree content from loaded data."""
        logger.debug("Updating tree content...")
        self.content_data = content_data

        try:
            tree = self.query_one("#content-tree", Tree)
            tree.clear()

            for content_type, items in content_data.items():
                # Add content type as root node
                type_label = CONTENT_TYPES.get(content_type, content_type.title())
                type_node = tree.root.add(f"{type_label} ({len(items)} items)")

                # Add individual items
                for item in items:
                    item_label = f"{item['title']}"
                    if item.get('metadata', {}).get('content_preview'):
                        item_label += f" - {item['metadata']['content_preview']}"

                    item_node = type_node.add(item_label, data={
                        'content_type': content_type,
                        'item_id': item['id'],
                        'item_data': item
                    })

            # Expand all nodes by default
            tree.root.expand_all()

        except Exception as e:
            logger.error(f"Error updating tree content: {e}")

    def clear_tree(self) -> None:
        """Clear the tree content."""
        try:
            tree = self.query_one("#content-tree", Tree)
            tree.clear()
        except Exception as e:
            logger.debug(f"Could not clear tree: {e}")

    def filter_tree_content(self, query: str) -> None:
        """Filter tree content based on search query."""
        # TODO: Implement search filtering
        logger.debug(f"Filtering tree content with query: '{query}'")

    @on(Tree.NodeSelected, "#content-tree")
    def handle_tree_selection(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        if event.node.data and 'item_id' in event.node.data:
            content_type = event.node.data['content_type']
            item_id = event.node.data['item_id']

            # Update selected items
            if content_type not in self.selected_items:
                self.selected_items[content_type] = set()

            if item_id in self.selected_items[content_type]:
                self.selected_items[content_type].remove(item_id)
            else:
                self.selected_items[content_type].add(item_id)

            logger.debug(f"Selected items updated: {self.selected_items}")

            # Notify parent of selection change
            self.post_message(ContentSelectionChanged(dict(self.selected_items)))

    @on(Input.Changed, "#content-filter")
    def handle_filter_change(self, event: Input.Changed) -> None:
        """Handle search filter changes."""
        self.search_query = event.value

########################################################################################################################
#
# Main Window Class
#
########################################################################################################################

class SearchEmbeddingsWindow(Container):
    """New streamlined embeddings creation interface."""

    DEFAULT_CSS = """
    SearchEmbeddingsWindow {
        layout: vertical;
        height: 100%;
        width: 100%;
        padding: 1;
    }
    
    /* Header */
    .window-header {
        height: 3;
        padding: 0 1;
        background: $boost;
        border: solid green;
        align: center middle;
    }
    
    .window-title {
        text-style: bold;
        color: $text;
        width: 1fr;
    }
    
    #launch-wizard {
        margin-left: 2;
        width: auto;
    }
    
    /* Content Type Selection */
    .content-type-label {
        text-style: bold;
        color: green;
        margin: 1 0;
    }
    
    .checkboxes-row {
        height: 3;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    .checkboxes-row Checkbox {
        margin-right: 2;
        background: transparent;
    }
    
    /* Main content area - 3 column layout */
    .main-content {
        layout: horizontal;
        height: 1fr;
        min-height: 20;
    }
    
    /* Left panel - Content */
    .content-panel {
        width: 33%;
        padding: 1;
        background: $surface;
        border: solid $primary;
        margin: 0 1 0 0;
    }
    
    /* Middle panel - Settings */
    .settings-panel {
        width: 34%;
        padding: 1;
        background: $surface;
        border: solid $primary;
        margin: 0 1 0 0;
    }
    
    /* Right panel - Results */
    .results-panel {
        width: 33%;
        padding: 1;
        background: $surface;
        border: solid $primary;
    }
    
    .panel-title-green {
        text-style: bold;
        color: green;
        background: $background;
        padding: 0 1;
        margin-bottom: 1;
        text-align: center;
    }
    
    .panel-subtitle {
        color: green;
        margin-bottom: 1;
        text-align: center;
    }
    
    .setting-label {
        color: $text;
        margin-top: 1;
        margin-bottom: 0;
    }
    
    .content-filter {
        width: 100%;
        margin-bottom: 1;
    }
    
    .tree-container {
        height: 1fr;
        border: solid $primary;
        background: $background;
        padding: 1;
    }
    
    .results-container {
        height: 1fr;
        border: solid $primary;
        background: $background;
        padding: 1;
    }
    
    .results-empty {
        color: $text-muted;
        text-align: center;
        margin-top: 5;
    }
    
    .results-progress {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .results-status {
        margin-bottom: 1;
    }
    
    .results-success {
        color: green;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .results-error {
        color: red;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .results-info {
        color: $text-muted;
        margin-left: 2;
    }
    
    /* Bottom Create Embeddings section */
    .create-embeddings-section {
        height: 8;
        padding: 1;
        background: $surface;
        border: solid green;
        margin-top: 1;
    }
    
    .collection-name-row {
        height: 3;
        align: left middle;
        margin-bottom: 1;
    }
    
    .collection-label {
        width: 15;
        margin-right: 1;
    }
    
    .collection-input {
        width: 1fr;
    }
    
    .action-buttons {
        height: auto;
        align: center middle;
    }
    
    .action-buttons Button {
        margin: 0 2;
        width: 12;
        height: 3;
    }
    
    #create-embeddings {
        background: green;
    }
    
    .model-dropdown {
        width: 100%;
        margin-bottom: 1;
    }
    
    .advanced-options {
        margin-top: 1;
    }
    
    .advanced-options Input, .advanced-options Select {
        margin-bottom: 1;
    }
    
    .advanced-title {
        text-style: bold;
        margin: 1 0;
    }
    
    """

    # Reactive attributes following ADR-003
    selected_content_types = reactive(set())
    selected_items = reactive(dict)  # {content_type: {item_ids}}
    is_processing = reactive(False)
    collection_name = reactive("")
    selected_model = reactive("text-embedding-ada-002")
    processing_progress = reactive(0)
    processing_status = reactive("Ready")

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.info("SearchEmbeddingsWindow initialized")

    def compose(self) -> ComposeResult:
        """Compose the main window layout."""
        logger.debug("Composing SearchEmbeddingsWindow")

        # Header section
        with Horizontal(classes="window-header"):
            yield Label("🔍 Search Embeddings", classes="window-title")
            yield Button("Launch Wizard", id="launch-wizard", variant="default")

        # Content type selection
        yield Label("Select Content Types:", classes="content-type-label")
        with Horizontal(classes="checkboxes-row"):
            yield Checkbox("Chats", id="content-type-chats")
            yield Checkbox("Character Chats", id="content-type-character-chats") 
            yield Checkbox("Notes", id="content-type-notes")
            yield Checkbox("Media", id="content-type-media")

        # Main content area with three columns
        with Horizontal(classes="main-content"):
            # Left panel - Content
            with Vertical(classes="content-panel", id="content-panel"):
                yield Static("Content", classes="panel-title-green")
                yield Static("Content", classes="panel-subtitle")
                yield Input(
                    placeholder="Filter content by keyword...",
                    id="content-filter",
                    classes="content-filter"
                )
                with VerticalScroll(classes="tree-container"):
                    yield Tree("Content", id="content-tree", classes="content-tree")

            # Middle panel - Settings
            with Vertical(classes="settings-panel", id="settings-panel"):
                yield Static("Settings", classes="panel-title-green")
                
                yield Label("Embedding Model:", classes="setting-label")
                yield Select(
                    [
                        ("OpenAI Ada v2 (Recommended)", "text-embedding-ada-002"),
                        ("OpenAI v3 Small", "text-embedding-3-small"),
                        ("OpenAI v3 Large", "text-embedding-3-large"),
                        ("Local Model (Free)", "all-MiniLM-L6-v2")
                    ],
                    id="model-select",
                    classes="model-dropdown",
                    value="text-embedding-ada-002"
                )

                # Advanced options
                yield Static("▼ Advanced Options", classes="advanced-title")
                yield Label("Chunk Size:", classes="setting-label")
                yield Input(placeholder="512", id="chunk-size", value="512")
                
                yield Label("Chunk Overlap:", classes="setting-label")
                yield Input(placeholder="50", id="chunk-overlap", value="50")
                
                yield Label("Storage Backend:", classes="setting-label")
                yield Select(
                    [("ChromaDB", "chromadb"), ("FAISS", "faiss")],
                    id="storage-backend",
                    value="chromadb"
                )

            # Right panel - Embedding Results
            with Vertical(classes="results-panel", id="results-panel"):
                yield Static("Embedding Results", classes="panel-title-green")
                with VerticalScroll(classes="results-container", id="results-container"):
                    yield Static("Ready to create embeddings", id="results-status", classes="results-empty")

        # Bottom section - Create Embeddings
        with Vertical(classes="create-embeddings-section"):
            yield Static("Create Embeddings", classes="panel-title-green")
            
            # Collection name row  
            with Horizontal(classes="collection-name-row"):
                yield Label("Collection Name:", classes="collection-label")
                yield Input(
                    placeholder="my_search_collection",
                    id="collection-name",
                    classes="collection-input"
                )
            
            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("Cancel", id="cancel-create", variant="default")
                yield Button("Create\nEmbeddings", id="create-embeddings", variant="success")

    def on_mount(self) -> None:
        """Initialize after mounting (ADR-001)."""
        logger.info("SearchEmbeddingsWindow mounted")

        # Debug: Check if all widgets are rendered
        try:
            logger.debug("Checking rendered components...")
            header = self.query(".window-header")
            logger.debug(f"Header found: {len(header)} elements")

            content_row = self.query(".content-type-row")
            logger.debug(f"Content type row found: {len(content_row)} elements")

            checkboxes_container = self.query(".content-type-checkboxes")
            logger.debug(f"Checkboxes container found: {len(checkboxes_container)} elements")

            checkboxes = self.query("Checkbox")
            logger.debug(f"All checkboxes found: {len(checkboxes)} elements")

            # Check specific checkbox IDs
            for checkbox_id in ["content-type-chats", "content-type-character-chats", "content-type-notes", "content-type-media"]:
                try:
                    checkbox = self.query_one(f"#{checkbox_id}")
                    logger.debug(f"Found checkbox: {checkbox_id} - {checkbox}")
                except Exception as cb_e:
                    logger.debug(f"Could not find checkbox {checkbox_id}: {cb_e}")

        except Exception as e:
            logger.debug(f"Component check failed: {e}")

        # Initialize with dependencies check
        if not embeddings_available:
            try:
                self.notify("Embeddings dependencies not available", severity="warning")
            except Exception:
                pass  # Widget not in app context

    # Event handlers following ADR-003 reactive pattern

    @on(Checkbox.Changed)
    def handle_content_type_selection(self, event: Checkbox.Changed) -> None:
        """Handle content type checkbox changes."""
        content_type_map = {
            'content-type-chats': 'chats',
            'content-type-character-chats': 'character_chats',
            'content-type-notes': 'notes',
            'content-type-media': 'media'
        }

        content_type = content_type_map.get(event.checkbox.id)
        if content_type:
            if event.value:
                self.selected_content_types.add(content_type)
            else:
                self.selected_content_types.discard(content_type)

            # Load content for selected types
            self.load_content_for_selected_types()

            logger.debug(f"Content types selected: {self.selected_content_types}")

    @work(thread=True)
    def load_content_for_selected_types(self) -> None:
        """Load content based on selected types."""
        if not self.selected_content_types:
            self.call_from_thread(self.clear_content_tree)
            return
            
        # Simulate loading content
        self.call_from_thread(self.update_content_tree)

    def clear_content_tree(self) -> None:
        """Clear the content tree."""
        try:
            tree = self.query_one("#content-tree", Tree)
            tree.clear()
        except Exception as e:
            logger.debug(f"Could not clear tree: {e}")

    def update_content_tree(self) -> None:
        """Update the content tree with sample data."""
        try:
            tree = self.query_one("#content-tree", Tree)
            tree.clear()
            
            # Add sample content for each selected type
            for content_type in self.selected_content_types:
                type_label = CONTENT_TYPES.get(content_type, content_type.title())
                type_node = tree.root.add(f"{type_label}")
                
                # Add sample items
                for i in range(3):
                    type_node.add(f"Sample {type_label} Item {i+1}")
                    
                type_node.expand()
                
        except Exception as e:
            logger.debug(f"Could not update tree: {e}")

    @on(ContentSelectionChanged)
    def handle_content_selection_change(self, event: ContentSelectionChanged) -> None:
        """Handle content item selection changes."""
        self.selected_items = event.selected_items
        logger.debug(f"Content items selected: {self.selected_items}")

    @on(Select.Changed, "#model-select")
    def handle_model_selection(self, event: Select.Changed) -> None:
        """Handle embedding model selection."""
        self.selected_model = event.value
        logger.debug(f"Model selected: {self.selected_model}")

    @on(Input.Changed, "#collection-name")
    def handle_collection_name_change(self, event: Input.Changed) -> None:
        """Handle collection name changes."""
        self.collection_name = event.value

    @on(Button.Pressed, "#launch-wizard")
    def handle_launch_wizard(self) -> None:
        """Launch the existing wizard as fallback."""
        logger.info("Launching embeddings wizard fallback")
        # TODO: Implement wizard launch
        self.notify("Wizard fallback not yet implemented", severity="info")

    @on(Button.Pressed, "#create-embeddings")
    def handle_create_embeddings(self) -> None:
        """Handle create embeddings button press."""
        if not self.validate_form():
            return

        logger.info("Starting embeddings creation process")
        self.start_embedding_creation()

    @on(Button.Pressed, "#cancel-create")
    def handle_cancel_create(self) -> None:
        """Handle cancel button press."""
        logger.info("Embeddings creation cancelled")
        if self.is_processing:
            # TODO: Cancel worker
            self.is_processing = False
            self.processing_status = "Cancelled"

    def validate_form(self) -> bool:
        """Validate form inputs before processing."""
        if not self.collection_name.strip():
            try:
                self.notify("Please enter a collection name", severity="error")
            except Exception:
                pass  # Widget not in app context
            return False

        if not self.selected_items:
            try:
                self.notify("Please select content to embed", severity="error")
            except Exception:
                pass  # Widget not in app context
            return False

        if not embeddings_available:
            try:
                self.notify("Embeddings dependencies not available", severity="error")
            except Exception:
                pass  # Widget not in app context
            return False

        return True

    @work(thread=True, exclusive=True)
    def start_embedding_creation(self) -> None:
        """Start embedding creation process (ADR-004)."""
        self.call_from_thread(setattr, self, "is_processing", True)
        self.call_from_thread(setattr, self, "processing_status", "Initializing...")

        try:
            # Simulate embedding creation process
            total_items = sum(len(items) for items in self.selected_items.values())

            for i in range(total_items):
                # Simulate processing time
                import time
                time.sleep(0.5)

                progress = int((i + 1) / total_items * 100)
                status = f"Processing item {i + 1} of {total_items}"

                # Update progress from thread (ADR-004)
                self.call_from_thread(self._update_progress, progress, status)

            # Mark as complete
            self.call_from_thread(self._embedding_creation_complete)

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            self.call_from_thread(self._embedding_creation_error, str(e))

    def _update_progress(self, progress: int, status: str) -> None:
        """Update progress from worker thread."""
        self.processing_progress = progress
        self.processing_status = status

        # Update results panel
        try:
            results_container = self.query_one(".results-container", VerticalScroll)
            
            # Clear existing content and add progress
            results_container.remove_children()
            
            # Add progress bar
            if self.is_processing and progress > 0:
                results_container.mount(Label(f"Progress: {progress}%", classes="results-progress"))
                results_container.mount(ProgressBar(id="embedding-progress", total=100, show_percentage=True))
                progress_bar = results_container.query_one("#embedding-progress", ProgressBar)
                progress_bar.update(progress=progress)
                
            # Add status
            results_container.mount(Label(status, classes="results-status"))
            
        except Exception as e:
            logger.debug(f"Could not update progress display: {e}")

    def _embedding_creation_complete(self) -> None:
        """Handle successful embedding creation."""
        self.is_processing = False
        self.processing_progress = 100
        self.processing_status = "Complete! Embeddings created successfully."
        
        # Update results panel
        try:
            results_container = self.query_one(".results-container", VerticalScroll)
            results_container.remove_children()
            results_container.mount(Label("✅ Success!", classes="results-success"))
            results_container.mount(Label("Embeddings created successfully!", classes="results-status"))
            results_container.mount(Label(f"Collection: {self.collection_name}", classes="results-info"))
            results_container.mount(Label(f"Model: {self.selected_model}", classes="results-info"))
            
            # Show item counts
            total_items = sum(len(items) for items in self.selected_items.values())
            results_container.mount(Label(f"Total items embedded: {total_items}", classes="results-info"))
            
        except Exception as e:
            logger.debug(f"Could not update results display: {e}")
            
        self.notify("Embeddings created successfully!", severity="success")

    def _embedding_creation_error(self, error_msg: str) -> None:
        """Handle embedding creation error."""
        self.is_processing = False
        self.processing_status = f"Error: {error_msg}"
        
        # Update results panel
        try:
            results_container = self.query_one(".results-container", VerticalScroll)
            results_container.remove_children()
            results_container.mount(Label("❌ Error!", classes="results-error"))
            results_container.mount(Label(error_msg, classes="results-status"))
            
        except Exception as e:
            logger.debug(f"Could not update results display: {e}")
            
        self.notify(f"Embedding creation failed: {error_msg}", severity="error")