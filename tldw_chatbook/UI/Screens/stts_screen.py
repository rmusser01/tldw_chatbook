"""
STTS (Speech-to-Text/Text-to-Speech) Screen
Screen wrapper for STTS functionality in screen-based navigation.
"""

from textual.app import ComposeResult
from textual.reactive import reactive
from typing import Optional, TYPE_CHECKING
from loguru import logger

from ..Navigation.base_app_screen import BaseAppScreen
from ..STTS_Window import STTSWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class STTSScreen(BaseAppScreen):
    """Screen wrapper for Speech-to-Text/Text-to-Speech functionality."""
    
    # Screen-specific state
    current_model: reactive[str] = reactive("")
    is_processing: reactive[bool] = reactive(False)
    audio_file_path: reactive[Optional[str]] = reactive(None)

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "stts", **kwargs)
        self.stts_window: Optional[STTSWindow] = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the STTS screen with the STTS window."""
        logger.info("Composing STTS screen")
        self.stts_window = STTSWindow(self.app_instance, classes="window")
        yield self.stts_window
    
    async def on_mount(self) -> None:
        """Initialize STTS services when screen is mounted."""
        logger.info("STTS screen mounted")
        
        # Get the STTS window
        stts_window = self.stts_window or self.query_one(STTSWindow)
        
        # Initialize any services if needed
        if hasattr(stts_window, 'initialize'):
            await stts_window.initialize()
    
    async def on_screen_suspend(self) -> None:
        """Clean up when screen is suspended (navigated away)."""
        logger.debug("STTS screen suspended")
        
        # Stop any ongoing audio processing
        if self.is_processing:
            stts_window = self.stts_window or self.query_one(STTSWindow)
            if hasattr(stts_window, 'stop_processing'):
                await stts_window.stop_processing()
            self.is_processing = False
    
    async def on_screen_resume(self) -> None:
        """Restore state when screen is resumed."""
        logger.debug("STTS screen resumed")
        
        # Restore any necessary state
        stts_window = self.stts_window or self.query_one(STTSWindow)
        if hasattr(stts_window, 'restore_state'):
            await stts_window.restore_state()
