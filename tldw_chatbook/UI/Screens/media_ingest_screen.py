"""Media Ingestion screen implementation - now using the rebuilt window."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..MediaIngestWindowRebuilt import MediaIngestWindowRebuilt
from .media_runtime_state import MediaRuntimeState

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class MediaIngestScreen(BaseAppScreen):
    """
    Media Ingestion screen that wraps the rebuilt MediaIngestWindowRebuilt.
    This provides compatibility with the screen-based navigation system.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "ingest", **kwargs)
        self.media_runtime_state: MediaRuntimeState = app_instance.media_runtime_state
        self.media_ingest_window = None
        logger.info("MediaIngestScreen initialized with rebuilt window")
    
    def compose_content(self) -> ComposeResult:
        """Compose the media ingestion content using the rebuilt window."""
        # Use the rebuilt media ingestion window
        self.media_ingest_window = MediaIngestWindowRebuilt(self.app_instance, id="media-ingest-window")
        self.media_ingest_window.runtime_state = self.media_runtime_state
        yield self.media_ingest_window
    
    def on_mount(self) -> None:
        """Handle mount event."""
        super().on_mount()
        logger.info("MediaIngestScreen mounted with rebuilt MediaIngestWindowRebuilt")
