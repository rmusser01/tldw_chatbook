# tldw_chatbook/UI/Embeddings_Window.py
# Description: Unified Embeddings window with tabbed interface for all embeddings functionality
#
# Imports
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any

# 3rd-Party Imports
from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    TabbedContent, TabPane, Label, Button, Static,
    LoadingIndicator, Header
)
from textual.message import Message
from textual.screen import ModalScreen

# Configure logger with context
logger = logger.bind(module="Embeddings_Window")

# Local Imports
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
from ..Widgets.empty_state import EmptyState

# Import wizard components
from .Wizards.EmbeddingsWizard import SimpleEmbeddingsWizard

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

########################################################################################################################
#
# Tab Content Components
#
########################################################################################################################

class CreateCollectionTab(Container):
    """Tab for creating new embedding collections."""
    
    def __init__(self, app_instance: 'TldwCli'):
        super().__init__()
        self.app_instance = app_instance
        
    def compose(self) -> ComposeResult:
        """Compose the creation tab content."""
        if not embeddings_available:
            yield EmptyState(
                icon="⚠️",
                title="Embeddings Dependencies Not Available",
                message="To use embeddings, install with:",
                action_text="pip install tldw_chatbook[embeddings_rag]",
                show_action_button=False
            )
        else:
            # Header with quick info
            with Container(classes="tab-header"):
                yield Label("Create Search Collection", classes="tab-title")
                yield Label(
                    "Create searchable indexes of your content using AI-powered semantic search",
                    classes="tab-subtitle"
                )
            
            # Main wizard content
            yield SimpleEmbeddingsWizard()
            
class ManageCollectionsTab(Container):
    """Tab for managing existing collections."""
    
    collections = reactive([])
    loading = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli'):
        super().__init__()
        self.app_instance = app_instance
        
    def compose(self) -> ComposeResult:
        """Compose the management tab content."""
        # Header
        with Container(classes="tab-header"):
            yield Label("Manage Collections", classes="tab-title")
            yield Label(
                "View, edit, and manage your existing search collections",
                classes="tab-subtitle"
            )
            
        # Quick actions
        with Horizontal(classes="quick-actions"):
            yield Button("🔄 Refresh", id="refresh-collections", variant="default")
            yield Button("📊 Statistics", id="view-stats", variant="default")
            yield Button("🗑️ Cleanup", id="cleanup-collections", variant="default")
            
        # Collections list area
        with Container(classes="collections-container"):
            if self.loading:
                yield LoadingIndicator("Loading collections...")
            elif not self.collections:
                yield EmptyState(
                    icon="📂",
                    title="No Collections Found",
                    message="You haven't created any search collections yet.",
                    action_text="Create Your First Collection",
                    action_id="create-first-collection"
                )
            else:
                # Collections will be loaded here
                yield Label("Collections will appear here", classes="placeholder")
                
    def on_mount(self) -> None:
        """Load collections when tab mounts."""
        self.load_collections()
        
    @work(thread=True)
    def load_collections(self):
        """Load collections in background."""
        self.loading = True
        try:
            # TODO: Load actual collections from database/service
            # For now, simulate loading
            import time
            time.sleep(1)  # Simulate loading time
            self.collections = []  # Will be populated with real data
        except Exception as e:
            logger.error(f"Failed to load collections: {e}")
        finally:
            self.loading = False
            
    @on(Button.Pressed, "#refresh-collections")
    def handle_refresh(self):
        """Refresh collections list."""
        self.load_collections()
        
    @on(Button.Pressed, "#create-first-collection")
    def handle_create_first(self):
        """Switch to create tab."""
        # Get parent tabbed content and switch to create tab
        parent_tabs = self.ancestors_with_self.filter(".embeddings-tabs").first()
        if parent_tabs:
            parent_tabs.active = "create"
            
class ModelSettingsTab(Container):
    """Tab for model configuration and preferences."""
    
    def __init__(self, app_instance: 'TldwCli'):
        super().__init__()
        self.app_instance = app_instance
        
    def compose(self) -> ComposeResult:
        """Compose the settings tab content."""
        # Header
        with Container(classes="tab-header"):
            yield Label("Model Settings", classes="tab-title")
            yield Label(
                "Configure embedding models and global preferences",
                classes="tab-subtitle"
            )
            
        # Settings content
        with Container(classes="settings-container"):
            yield Label("Model configuration options will appear here", classes="placeholder")
            
            # Quick model status
            with Container(classes="model-status"):
                yield Label("🤖 Active Models", classes="section-title")
                if embeddings_available:
                    yield Label("✅ Embeddings service available", classes="status-good")
                else:
                    yield Label("❌ Embeddings service unavailable", classes="status-error")

########################################################################################################################
#
# Main Window Class
#
########################################################################################################################

class EmbeddingsWindow(Container):
    """Unified embeddings window with tabbed interface."""
    
    DEFAULT_CSS = """
    EmbeddingsWindow {
        layout: vertical;
        height: 100%;
        width: 100%;
    }
    
    .embeddings-header {
        dock: top;
        height: auto;
        padding: 1;
        background: $boost;
        border-bottom: solid $primary;
    }
    
    .embeddings-title {
        text-style: bold;
        color: $text;
        margin-bottom: 0;
    }
    
    .embeddings-subtitle {
        color: $text-muted;
        margin-top: 0;
    }
    
    .embeddings-tabs {
        height: 1fr;
        margin: 1;
    }
    
    .tab-header {
        margin-bottom: 2;
        padding: 1;
        background: $surface;
        border-radius: 4px;
    }
    
    .tab-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .tab-subtitle {
        color: $text-muted;
        margin-bottom: 0;
    }
    
    .quick-actions {
        margin-bottom: 2;
        height: auto;
    }
    
    .quick-actions Button {
        margin-right: 1;
    }
    
    .collections-container, .settings-container {
        height: 1fr;
        padding: 1;
        background: $surface;
        border-radius: 4px;
    }
    
    .placeholder {
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    
    .section-title {
        text-style: bold;
        margin: 1 0;
    }
    
    .status-good {
        color: $success;
    }
    
    .status-error {
        color: $error;
    }
    
    .model-status {
        margin-top: 2;
        padding: 1;
        background: $boost;
        border-radius: 4px;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.info("EmbeddingsWindow initialized with unified tabbed interface")
    
    def compose(self) -> ComposeResult:
        """Compose the unified embeddings window."""
        logger.debug("Composing unified EmbeddingsWindow with tabs")
        
        # Main header
        with Container(classes="embeddings-header"):
            yield Label("🔍 AI Search Collections", classes="embeddings-title")
            yield Label(
                "Create and manage searchable collections of your content",
                classes="embeddings-subtitle"
            )
        
        # Tabbed interface
        with TabbedContent(initial="create", classes="embeddings-tabs"):
            with TabPane("Create Collection", id="create"):
                yield CreateCollectionTab(self.app_instance)
                
            with TabPane("Manage Collections", id="manage"):
                yield ManageCollectionsTab(self.app_instance)
                
            with TabPane("Model Settings", id="settings"):
                yield ModelSettingsTab(self.app_instance)
                
    def on_mount(self) -> None:
        """Handle window mount."""
        logger.info("EmbeddingsWindow mounted successfully")