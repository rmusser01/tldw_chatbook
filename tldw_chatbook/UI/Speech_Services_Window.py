# Speech_Services_Window.py
# Description: Improved Speech Services tab with clear separation between TTS and Dictation
#
# This is a new implementation that will replace STTS_Window.py to provide better
# organization and clarity between Text-to-Speech and Speech-to-Text functionality.

from typing import Optional, Dict, Any, List
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Label, Button, Static, Rule
from textual.widget import Widget
from textual.reactive import reactive
from textual.binding import Binding
from loguru import logger

# Import existing widgets
from tldw_chatbook.UI.STTS_Window import (
    TTSPlaygroundWidget,
    TTSSettingsWidget,
    AudioBookGenerationWidget
)
from tldw_chatbook.UI.Dictation_Window import DictationWindow

class SpeechServicesWindow(Container):
    """
    Main Speech Services window with clear separation between TTS and Dictation.
    
    Replaces the ambiguous STTS naming with a clearer structure that separates
    Text-to-Speech (TTS) and Speech-to-Text (Dictation) functionality.
    """
    
    DEFAULT_CSS = """
    SpeechServicesWindow {
        layout: horizontal;
        height: 100%;
    }
    
    .services-sidebar {
        width: 25;
        height: 100%;
        border-right: solid $surface;
        padding: 1;
        background: $panel;
    }
    
    .services-content {
        width: 1fr;
        height: 100%;
        padding: 1;
    }
    
    .service-section {
        margin-bottom: 2;
    }
    
    .section-header {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .section-description {
        color: $text-muted;
        text-style: italic;
        margin-bottom: 1;
    }
    
    .sidebar-button {
        width: 100%;
        margin-bottom: 1;
    }
    
    .service-divider {
        margin: 2 0;
        color: $surface;
    }
    
    .current-mode {
        background: $boost;
        padding: 1;
        border: solid $primary;
        margin-bottom: 1;
    }
    
    .mode-label {
        text-style: bold;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+t", "switch_to_tts", "Switch to TTS"),
        Binding("ctrl+d", "switch_to_dictation", "Switch to Dictation"),
        Binding("f1", "show_help", "Show Help"),
    ]
    
    # Service modes
    SERVICE_TTS = "tts"
    SERVICE_DICTATION = "dictation"
    
    # View states within each service
    current_service = reactive(SERVICE_TTS)
    tts_view = reactive("playground")
    dictation_view = reactive("main")
    
    def __init__(self, app_instance, **kwargs):
        """Initialize the Speech Services window."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
    def compose(self) -> ComposeResult:
        """Compose the UI with improved organization."""
        with Horizontal():
            # Sidebar with clear sections
            with Vertical(classes="services-sidebar"):
                yield Label("🗣️ Speech Services", classes="section-title")
                
                # Current mode indicator
                with Container(classes="current-mode"):
                    yield Label("Current Mode:", classes="mode-label")
                    yield Label("Text-to-Speech", id="current-mode-text")
                
                # TTS Section
                with Container(classes="service-section"):
                    yield Label("📢 Text-to-Speech (TTS)", classes="section-header")
                    yield Static(
                        "Convert text into natural speech",
                        classes="section-description"
                    )
                    yield Button(
                        "🎮 Playground",
                        id="tts-playground-btn",
                        classes="sidebar-button",
                        variant="primary"
                    )
                    yield Button(
                        "⚙️ Settings",
                        id="tts-settings-btn",
                        classes="sidebar-button"
                    )
                    yield Button(
                        "📚 AudioBook Generator",
                        id="tts-audiobook-btn",
                        classes="sidebar-button"
                    )
                    yield Button(
                        "🎭 Voice Cloning",
                        id="tts-voice-cloning-btn",
                        classes="sidebar-button"
                    )
                
                yield Rule(classes="service-divider")
                
                # Dictation Section
                with Container(classes="service-section"):
                    yield Label("🎤 Dictation (STT)", classes="section-header")
                    yield Static(
                        "Convert speech into text",
                        classes="section-description"
                    )
                    yield Button(
                        "🎙️ Live Dictation",
                        id="dictation-main-btn",
                        classes="sidebar-button"
                    )
                    yield Button(
                        "📝 Transcription History",
                        id="dictation-history-btn",
                        classes="sidebar-button",
                        disabled=True  # Future feature
                    )
                    yield Button(
                        "🎯 Voice Commands",
                        id="dictation-commands-btn",
                        classes="sidebar-button",
                        disabled=True  # Future feature
                    )
                
                yield Rule(classes="service-divider")
                
                # Additional Features (Future)
                with Container(classes="service-section"):
                    yield Label("🚀 Advanced Features", classes="section-header")
                    yield Button(
                        "🔊 Audio Effects",
                        id="audio-effects-btn",
                        classes="sidebar-button",
                        disabled=True
                    )
                    yield Button(
                        "🔄 Speech Translation",
                        id="translation-btn",
                        classes="sidebar-button",
                        disabled=True
                    )
            
            # Content area
            with Container(classes="services-content"):
                # Show TTS playground by default
                yield TTSPlaygroundWidget()
    
    def watch_current_service(self, old_service: str, new_service: str) -> None:
        """Handle service mode changes between TTS and Dictation."""
        # Update mode indicator
        mode_text = self.query_one("#current-mode-text", Label)
        if new_service == self.SERVICE_TTS:
            mode_text.update("Text-to-Speech")
            # Switch to last TTS view
            self._switch_to_tts_view(self.tts_view)
        else:
            mode_text.update("Speech-to-Text")
            # Switch to last dictation view
            self._switch_to_dictation_view(self.dictation_view)
        
        # Update button states
        self._update_button_states()
    
    def _switch_to_tts_view(self, view: str) -> None:
        """Switch to a specific TTS view."""
        content_container = self.query_one(".services-content", Container)
        
        # Clean up existing widgets
        for widget in content_container.children:
            if hasattr(widget, 'cleanup'):
                widget.cleanup()
        content_container.remove_children()
        
        # Mount appropriate widget
        if view == "playground":
            content_container.mount(TTSPlaygroundWidget())
        elif view == "settings":
            content_container.mount(TTSSettingsWidget())
        elif view == "audiobook":
            content_container.mount(AudioBookGenerationWidget())
        
        self.tts_view = view
    
    def _switch_to_dictation_view(self, view: str) -> None:
        """Switch to a specific dictation view."""
        content_container = self.query_one(".services-content", Container)
        
        # Clean up existing widgets
        for widget in content_container.children:
            if hasattr(widget, 'cleanup'):
                widget.cleanup()
        content_container.remove_children()
        
        # Mount appropriate widget
        if view == "main":
            content_container.mount(DictationWindow())
        # Future: Add history and commands views
        
        self.dictation_view = view
    
    def _update_button_states(self) -> None:
        """Update button visual states based on current service and view."""
        # Reset all buttons
        for btn in self.query(".sidebar-button").results(Button):
            btn.variant = "default"
        
        # Highlight active button
        if self.current_service == self.SERVICE_TTS:
            if self.tts_view == "playground":
                self.query_one("#tts-playground-btn", Button).variant = "primary"
            elif self.tts_view == "settings":
                self.query_one("#tts-settings-btn", Button).variant = "primary"
            elif self.tts_view == "audiobook":
                self.query_one("#tts-audiobook-btn", Button).variant = "primary"
        else:
            if self.dictation_view == "main":
                self.query_one("#dictation-main-btn", Button).variant = "primary"
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar button presses."""
        button_id = event.button.id
        
        # TTS buttons
        if button_id == "tts-playground-btn":
            self.current_service = self.SERVICE_TTS
            self._switch_to_tts_view("playground")
        elif button_id == "tts-settings-btn":
            self.current_service = self.SERVICE_TTS
            self._switch_to_tts_view("settings")
        elif button_id == "tts-audiobook-btn":
            self.current_service = self.SERVICE_TTS
            self._switch_to_tts_view("audiobook")
        elif button_id == "tts-voice-cloning-btn":
            # Keep existing behavior - open in new screen
            from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow
            self.app.push_screen(VoiceCloningWindow())
        
        # Dictation buttons
        elif button_id == "dictation-main-btn":
            self.current_service = self.SERVICE_DICTATION
            self._switch_to_dictation_view("main")
        elif button_id == "dictation-history-btn":
            self.app.notify("Transcription History coming soon!", severity="information")
        elif button_id == "dictation-commands-btn":
            self.app.notify("Voice Commands configuration coming soon!", severity="information")
        
        # Future features
        elif button_id == "audio-effects-btn":
            self.app.notify("Audio Effects coming soon!", severity="information")
        elif button_id == "translation-btn":
            self.app.notify("Speech Translation coming soon!", severity="information")
        
        # Update button states
        self._update_button_states()
    
    def action_switch_to_tts(self) -> None:
        """Keyboard shortcut to switch to TTS mode."""
        self.current_service = self.SERVICE_TTS
    
    def action_switch_to_dictation(self) -> None:
        """Keyboard shortcut to switch to Dictation mode."""
        self.current_service = self.SERVICE_DICTATION
    
    def action_show_help(self) -> None:
        """Show help for speech services."""
        help_text = """
Speech Services Help:

Text-to-Speech (TTS):
- Playground: Test different voices and settings
- Settings: Configure default TTS options
- AudioBook: Generate long-form audio content
- Voice Cloning: Create custom voices

Dictation (STT):
- Live Dictation: Real-time speech-to-text
- History: Review past transcriptions
- Commands: Configure voice commands

Shortcuts:
- Ctrl+T: Switch to TTS mode
- Ctrl+D: Switch to Dictation mode
- F1: Show this help
        """
        self.app.notify(help_text.strip(), title="Speech Services Help", timeout=10)