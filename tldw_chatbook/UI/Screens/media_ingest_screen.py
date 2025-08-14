"""Media Ingestion screen implementation."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Button

from ..Navigation.base_app_screen import BaseAppScreen
from ..MediaIngest.panels import (
    VideoIngestPanel,
    AudioIngestPanel,
    PDFIngestPanel,
    DocumentIngestPanel,
    EbookIngestPanel,
    WebIngestPanel,
)
from textual import on

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class MediaIngestScreen(BaseAppScreen):
    """
    Media Ingestion screen with sub-navigation for different media types.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "ingest", **kwargs)
        self.current_media_type = "video"
        self.current_panel = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the media ingestion content."""
        # Container for the media type panels
        with Container(id="media-ingest-container"):
            # Start with video panel
            self.current_panel = VideoIngestPanel(self.app_instance, "video")
            yield self.current_panel
    
    async def switch_media_type(self, media_type: str) -> None:
        """Switch to a different media type panel."""
        if media_type == self.current_media_type:
            return
        
        # Get the appropriate panel class
        panel_class = self.get_panel_class(media_type)
        if not panel_class:
            logger.error(f"Unknown media type: {media_type}")
            return
        
        # Create the new panel
        new_panel = panel_class(self.app_instance, media_type)
        
        # Replace the current panel
        container = self.query_one("#media-ingest-container")
        await container.remove_children()
        await container.mount(new_panel)
        
        # Update tracking
        self.current_panel = new_panel
        self.current_media_type = media_type
        
        logger.info(f"Switched to {media_type} ingestion panel")
    
    def get_panel_class(self, media_type: str):
        """Get the panel class for a media type."""
        panel_map = {
            "video": VideoIngestPanel,
            "audio": AudioIngestPanel,
            "pdf": PDFIngestPanel,
            "document": DocumentIngestPanel,
            "ebook": EbookIngestPanel,
            "web": WebIngestPanel,
        }
        return panel_map.get(media_type)
    
    @on(Button.Pressed, ".media-nav-button")
    async def handle_media_navigation(self, event: Button.Pressed) -> None:
        """Handle navigation between media types."""
        button_id = event.button.id
        if button_id and button_id.startswith("nav-"):
            media_type = button_id.replace("nav-", "")
            await self.switch_media_type(media_type)
    
    def save_state(self):
        """Save media ingestion state."""
        state = super().save_state()
        state['current_media_type'] = self.current_media_type
        if self.current_panel:
            self.current_panel.save_state()
        return state
    
    def restore_state(self, state):
        """Restore media ingestion state."""
        super().restore_state(state)
        if 'current_media_type' in state:
            # Switch to the saved media type
            self.app_instance.call_after_refresh(
                lambda: self.switch_media_type(state['current_media_type'])
            )