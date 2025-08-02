# tldw_chatbook/UI/Embeddings_Window.py
# Description: Main Embeddings window - now using the new wizard UI
#
# Imports
from __future__ import annotations
from typing import TYPE_CHECKING

# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container

# Configure logger with context
logger = logger.bind(module="Embeddings_Window")

# Local Imports
from .Wizards.EmbeddingsWizard import SimpleEmbeddingsWizard

if TYPE_CHECKING:
    from ..app import TldwCli

########################################################################################################################
#
# Main Window Class
#
########################################################################################################################

class EmbeddingsWindow(Container):
    """Main container for embeddings functionality - uses the new wizard UI."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.debug("EmbeddingsWindow initialized with wizard UI")
    
    def compose(self) -> ComposeResult:
        """Compose the embeddings window with the wizard UI."""
        logger.debug("Composing EmbeddingsWindow - Wizard UI")
        
        # Use the wizard UI
        yield SimpleEmbeddingsWizard()