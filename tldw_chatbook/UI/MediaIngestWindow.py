"""Media Ingestion Window - Screen dispatcher for media ingestion."""

from typing import TYPE_CHECKING, Optional
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static
from textual import on

from .MediaIngest.screens import (
    VideoIngestScreen,
    AudioIngestScreen,
    PDFIngestScreen,
    DocumentIngestScreen,
    EbookIngestScreen,
    WebIngestScreen,
    NavigateToMediaType
)
from .ScreenNavigation.navigation_system import ScreenManager

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class MediaIngestWindow(Container):
    """
    Media ingestion window that manages screen-based navigation.
    
    This acts as a dispatcher for the different media type screens,
    allowing navigation between Video, Audio, PDF, Documents, Ebooks, and Web.
    """
    
    DEFAULT_CSS = """
    MediaIngestWindow {
        width: 100%;
        height: 100%;
        layout: vertical;
        background: $surface;
    }
    
    #ingest-screen-container {
        width: 100%;
        height: 100%;
    }
    
    .loading-message {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.screen_manager = ScreenManager(app_instance)
        self.current_media_type = "video"
        self.current_screen: Optional[Container] = None
        self._initialized = False
        
        logger.debug("MediaIngestWindow initialized")
    
    def compose(self) -> ComposeResult:
        """Initial composition - show loading state."""
        with Container(id="ingest-screen-container"):
            yield Static("Initializing media ingestion...", classes="loading-message")
    
    async def on_mount(self) -> None:
        """Handle mount event - load the initial screen."""
        logger.info("MediaIngestWindow mounted")
        
        if not self._initialized:
            await self.load_media_screen("video")
            self._initialized = True
    
    async def load_media_screen(self, media_type: str) -> None:
        """Load a specific media ingestion screen."""
        try:
            # Save current screen state if there is one
            if self.current_screen:
                self.screen_manager.save_current_state()
            
            # Get the appropriate screen class
            screen_class = self.get_screen_class(media_type)
            if not screen_class:
                logger.error(f"Unknown media type: {media_type}")
                return
            
            # Create the new screen
            new_screen = screen_class(self.app_instance, media_type)
            
            # Clear the container and add the new screen
            container = self.query_one("#ingest-screen-container")
            await container.remove_children()
            await container.mount(new_screen)
            
            # Update tracking
            self.current_screen = new_screen
            self.current_media_type = media_type
            self.screen_manager.current_screen = f"media_ingest_{media_type}"
            
            # Restore state if available
            state_key = f"media_ingest_{media_type}"
            if state_key in self.screen_manager.screen_states:
                # Schedule state restoration after the screen is fully mounted
                self.app_instance.call_after_refresh(
                    lambda: self.screen_manager.screen_states[state_key].restore_to_screen(new_screen)
                )
            
            logger.info(f"Loaded {media_type} ingestion screen")
            
        except Exception as e:
            logger.error(f"Error loading media screen: {e}")
    
    def get_screen_class(self, media_type: str):
        """Get the screen class for a media type."""
        screen_map = {
            "video": VideoIngestScreen,
            "audio": AudioIngestScreen,
            "pdf": PDFIngestScreen,
            "document": DocumentIngestScreen,
            "ebook": EbookIngestScreen,
            "web": WebIngestScreen,
        }
        return screen_map.get(media_type)
    
    @on(NavigateToMediaType)
    async def handle_media_navigation(self, message: NavigateToMediaType) -> None:
        """Handle navigation between media types."""
        media_type = message.media_type
        
        if media_type != self.current_media_type:
            await self.load_media_screen(media_type)
    
    def on_unmount(self) -> None:
        """Save state when unmounting."""
        if self.current_screen:
            self.screen_manager.save_current_state()