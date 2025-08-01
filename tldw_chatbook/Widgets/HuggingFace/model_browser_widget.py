# tldw_chatbook/Widgets/HuggingFace/model_browser_widget.py
"""
Main HuggingFace GGUF model browser widget.
"""

from typing import Optional, TYPE_CHECKING
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Label, TabbedContent, TabPane
from textual.reactive import reactive
from textual import on
from loguru import logger

from .model_search_widget import ModelSearchWidget, ModelSelectedEvent
from .model_card_viewer import ModelCardViewer, DownloadRequestEvent
from .download_manager import DownloadManager

if TYPE_CHECKING:
    from ...app import TldwCli


class HuggingFaceModelBrowser(Container):
    """Main widget for browsing and downloading HuggingFace GGUF models."""
    
    DEFAULT_CSS = """
    HuggingFaceModelBrowser {
        height: 100%;
        layout: vertical;
        background: $surface;
    }
    
    HuggingFaceModelBrowser .header {
        height: 3;
        padding: 1;
        background: $primary;
        color: $text;
        text-align: center;
    }
    
    HuggingFaceModelBrowser .header-title {
        text-style: bold;
    }
    
    HuggingFaceModelBrowser .content-area {
        height: 1fr;
        layout: horizontal;
    }
    
    HuggingFaceModelBrowser .left-panel {
        width: 40%;
        min-width: 30;
        border-right: solid $primary;
    }
    
    HuggingFaceModelBrowser .right-panel {
        width: 1fr;
    }
    
    HuggingFaceModelBrowser TabbedContent {
        height: 1fr;
    }
    
    HuggingFaceModelBrowser TabPane {
        padding: 1;
        height: 1fr;
    }
    
    HuggingFaceModelBrowser ModelCardViewer {
        height: 1fr;
    }
    
    HuggingFaceModelBrowser DownloadManager {
        height: 1fr;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the model browser."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
        # Get download directory from config
        config = self.app_instance.app_config
        self.download_dir = Path(
            config.get("llm_management", {}).get("model_download_dir", "~/Downloads/tldw_models")
        ).expanduser()
    
    def compose(self) -> ComposeResult:
        """Compose the model browser UI."""
        # Header
        with Container(classes="header"):
            yield Label("🤗 HuggingFace GGUF Model Browser", classes="header-title")
        
        # Main content area
        with Horizontal(classes="content-area"):
            # Left panel - Search
            with Container(classes="left-panel"):
                yield ModelSearchWidget(id="model-search")
            
            # Right panel - Tabbed content
            with Container(classes="right-panel"):
                with TabbedContent():
                    with TabPane("Model Details", id="details-tab"):
                        yield ModelCardViewer(id="model-card")
                    
                    with TabPane("Downloads", id="downloads-tab"):
                        yield DownloadManager(
                            download_dir=self.download_dir,
                            id="download-manager"
                        )
    
    @on(ModelSelectedEvent)
    def handle_model_selected(self, event: ModelSelectedEvent) -> None:
        """Handle model selection from search."""
        logger.info(f"Model selected: {event.model_info.get('id')}")
        
        # Update model card viewer
        model_card = self.query_one("#model-card", ModelCardViewer)
        model_card.model_info = event.model_info
        
        # Switch to details tab
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "details-tab"
    
    @on(DownloadRequestEvent)
    def handle_download_request(self, event: DownloadRequestEvent) -> None:
        """Handle download request from model card."""
        logger.info(f"Download requested for {len(event.files)} files from {event.repo_id}")
        
        # Get download manager
        download_manager = self.query_one("#download-manager", DownloadManager)
        
        # Add files to download queue
        for file_info in event.files:
            download_id = download_manager.add_download(event.repo_id, file_info)
            logger.info(f"Queued download: {download_id}")
        
        # Switch to downloads tab
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "downloads-tab"
        
        # Notify user
        self.app_instance.notify(
            f"Added {len(event.files)} file(s) to download queue",
            severity="information"
        )