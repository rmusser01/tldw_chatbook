"""
STTS (Speech-to-Text/Text-to-Speech) Screen
Screen wrapper for STTS functionality in screen-based navigation.
"""

from textual.screen import Screen
from textual.app import ComposeResult
from textual.reactive import reactive
from typing import Optional
from loguru import logger

from ..STTS_Window import STTSWindow


class STTSScreen(Screen):
    """Screen wrapper for Speech-to-Text/Text-to-Speech functionality."""
    
    # Screen-specific state
    current_model: reactive[str] = reactive("")
    is_processing: reactive[bool] = reactive(False)
    audio_file_path: reactive[Optional[str]] = reactive(None)
    
    def compose(self) -> ComposeResult:
        """Compose the STTS screen with the STTS window."""
        logger.info("Composing STTS screen")
        yield STTSWindow()
    
    async def on_mount(self) -> None:
        """Initialize STTS services when screen is mounted."""
        logger.info("STTS screen mounted")
        
        # Get the STTS window
        stts_window = self.query_one(STTSWindow)
        
        # Initialize any services if needed
        if hasattr(stts_window, 'initialize'):
            await stts_window.initialize()
    
    async def on_screen_suspend(self) -> None:
        """Clean up when screen is suspended (navigated away)."""
        logger.debug("STTS screen suspended")
        
        # Stop any ongoing audio processing
        if self.is_processing:
            stts_window = self.query_one(STTSWindow)
            if hasattr(stts_window, 'stop_processing'):
                await stts_window.stop_processing()
            self.is_processing = False
    
    async def on_screen_resume(self) -> None:
        """Restore state when screen is resumed."""
        logger.debug("STTS screen resumed")
        
        # Restore any necessary state
        stts_window = self.query_one(STTSWindow)
        if hasattr(stts_window, 'restore_state'):
            await stts_window.restore_state()