"""Media Ingestion Window wrapper for the MediaIngestScreen."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import TabbedContent, TabPane, Static

from .MediaIngest.video import VideoIngestTab

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class PlaceholderTab(Container):
    """Placeholder for tabs not yet implemented."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message
        self.add_class("placeholder-tab")
    
    def compose(self) -> ComposeResult:
        yield Static(self.message)


class MediaIngestWindow(Container):
    """
    Window wrapper that provides the media ingestion UI.
    
    Since we can't directly use a Screen within the tab system,
    we implement the UI directly in this Container.
    """
    
    DEFAULT_CSS = """
    MediaIngestWindow {
        width: 100%;
        height: 100%;
        layout: vertical;
        background: $surface;
    }
    
    #ingest-container {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    
    #ingest-header {
        height: 3;
        width: 100%;
        padding: 1;
        background: $primary;
        content-align: center middle;
        margin-bottom: 1;
    }
    
    .header-title {
        text-style: bold;
        color: $text;
    }
    
    TabbedContent {
        width: 100%;
        height: 1fr;
    }
    
    TabPane {
        padding: 0;
        width: 100%;
        height: 100%;
    }
    
    .placeholder-tab {
        width: 100%;
        height: 100%;
        content-align: center middle;
        padding: 2;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.debug("MediaIngestWindow initialized")
    
    def compose(self) -> ComposeResult:
        """Build the media ingestion interface."""
        with Container(id="ingest-container"):
            # Header
            with Container(id="ingest-header"):
                yield Static("Media Ingestion", classes="header-title")
            
            # Tabbed content for different media types
            with TabbedContent(initial="video"):
                # Video tab
                with TabPane("Video", id="video"):
                    yield VideoIngestTab(self.app_instance, "video")
                
                # Audio tab
                with TabPane("Audio", id="audio"):
                    yield PlaceholderTab("Audio ingestion - Coming soon")
                
                # PDF tab
                with TabPane("PDF", id="pdf"):
                    yield PlaceholderTab("PDF ingestion - Coming soon")
                
                # Documents tab
                with TabPane("Documents", id="documents"):
                    yield PlaceholderTab("Document ingestion - Coming soon")
                
                # Ebooks tab
                with TabPane("Ebooks", id="ebooks"):
                    yield PlaceholderTab("Ebook ingestion - Coming soon")
                
                # Web tab
                with TabPane("Web", id="web"):
                    yield PlaceholderTab("Web article ingestion - Coming soon")
    
    def on_mount(self):
        """Handle mount event."""
        logger.info("MediaIngestWindow mounted")
        # Focus on the first input in the active tab
        self.set_focus(None)