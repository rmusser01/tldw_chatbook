# tldw_chatbook/UI/Embeddings_Window.py
# Description: Main Embeddings window container with navigation between creation and management views
#
# Imports
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from pathlib import Path

# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.widgets import Static, Button

# Configure logger with context
logger = logger.bind(module="Embeddings_Window")

# Local Imports
from .Embeddings_Creation_Window import EmbeddingsCreationWindow
from .Embeddings_Management_Window import EmbeddingsManagementWindow

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
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.debug("EmbeddingsWindow initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the embeddings window with navigation and content areas."""
        logger.debug("Composing EmbeddingsWindow UI")
        
        # Left navigation pane
        with VerticalScroll(id="embeddings-nav-pane", classes="embeddings-nav-pane"):
            yield Static("Embeddings Options", classes="sidebar-title")
            yield Button("Create Embeddings", id="embeddings-nav-create", classes="embeddings-nav-button")
            yield Button("Manage Embeddings", id="embeddings-nav-manage", classes="embeddings-nav-button")
        
        # Right content pane
        with Container(id="embeddings-content-pane", classes="embeddings-content-pane"):
            # Create embeddings view
            with Container(id="embeddings-view-create", classes="embeddings-view-area"):
                yield EmbeddingsCreationWindow(self.app_instance, id="embeddings-creation-widget")
            
            # Manage embeddings view
            with Container(id="embeddings-view-manage", classes="embeddings-view-area"):
                yield EmbeddingsManagementWindow(self.app_instance, id="embeddings-management-widget")
    
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

#
# End of Embeddings_Window.py
########################################################################################################################