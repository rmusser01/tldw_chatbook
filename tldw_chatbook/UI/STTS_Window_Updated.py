# STTS_Window_Updated.py
# Description: Updated STTS Window that maintains backward compatibility while using improved components
#
# This file provides a migration path from the old STTS_Window to the new Speech Services architecture

from typing import Optional
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Label
from textual.reactive import reactive
from loguru import logger

# Import the new improved window
from .Speech_Services_Window import SpeechServicesWindow


class STTSWindow(Container):
    """
    Backward-compatible wrapper for the new Speech Services window.
    
    This maintains the same interface as the old STTS_Window while delegating
    to the improved SpeechServicesWindow implementation.
    """
    
    DEFAULT_CSS = """
    STTSWindow {
        height: 100%;
        width: 100%;
    }
    
    .migration-notice {
        background: $boost;
        padding: 1;
        margin: 0 0 1 0;
        border: round $primary;
    }
    
    .notice-text {
        color: $text;
    }
    """
    
    # Maintain compatibility with existing reactive attribute
    current_view = reactive("playground")
    
    def __init__(self, app_instance, **kwargs):
        """Initialize the wrapper with backward compatibility."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._speech_services_window = None
        
        # Log migration
        logger.info("STTSWindow: Using updated Speech Services implementation")
    
    def compose(self) -> ComposeResult:
        """Compose UI with migration notice and new implementation."""
        # Optional: Show a subtle migration notice
        if self.app_instance.app_config.get('show_migration_notices', False):
            with Container(classes="migration-notice"):
                yield Label(
                    "ℹ️ Speech Services has been improved with better organization and privacy features",
                    classes="notice-text"
                )
        
        # Use the new Speech Services window
        self._speech_services_window = SpeechServicesWindow(self.app_instance)
        yield self._speech_services_window
    
    def on_mount(self):
        """Ensure compatibility mappings on mount."""
        # Map old view names to new structure
        view_mapping = {
            "playground": ("tts", "playground"),
            "settings": ("tts", "settings"),
            "audiobook": ("tts", "audiobook"),
            "dictation": ("dictation", "main")
        }
        
        if self.current_view in view_mapping:
            service, view = view_mapping[self.current_view]
            if self._speech_services_window:
                self._speech_services_window.current_service = service
                if service == "tts":
                    self._speech_services_window.tts_view = view
                else:
                    self._speech_services_window.dictation_view = view
    
    def watch_current_view(self, old_view: str, new_view: str) -> None:
        """Maintain compatibility with old view watching."""
        # Map to new structure
        view_mapping = {
            "playground": ("tts", "playground"),
            "settings": ("tts", "settings"),
            "audiobook": ("tts", "audiobook"),
            "dictation": ("dictation", "main")
        }
        
        if new_view in view_mapping and self._speech_services_window:
            service, view = view_mapping[new_view]
            self._speech_services_window.current_service = service
            if service == "tts":
                self._speech_services_window._switch_to_tts_view(view)
            else:
                self._speech_services_window._switch_to_dictation_view(view)
    
    def on_button_pressed(self, event) -> None:
        """Delegate button events to the speech services window."""
        if self._speech_services_window:
            self._speech_services_window.on_button_pressed(event)


# For drop-in replacement, export the same widgets that were in original STTS_Window
from .STTS_Window import (
    TTSPlaygroundWidget,
    TTSSettingsWidget,
    AudioBookGenerationWidget
)

# Also export improved dictation window
from .Dictation_Window_Improved import ImprovedDictationWindow as DictationWindow

__all__ = [
    'STTSWindow',
    'TTSPlaygroundWidget',
    'TTSSettingsWidget',
    'AudioBookGenerationWidget',
    'DictationWindow'
]