# Dictation_Window_Improved.py
"""
Improved dictation interface with privacy settings and better error handling.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.widgets import (
    Label, Button, TextArea, Select, Input, Static, 
    RichLog, Switch, Collapsible, Rule, ListView, ListItem,
    LoadingIndicator
)
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual import work
from textual.binding import Binding
from loguru import logger
import json

# Local imports
from ..config import get_cli_setting, save_setting_to_cli_config
from ..Audio.dictation_service_lazy import (
    LazyLiveDictationService, 
    AudioInitializationError,
    TranscriptionInitializationError,
    DictationState
)
from ..Event_Handlers.Audio_Events import (
    DictationStartedEvent, DictationStoppedEvent,
    PartialTranscriptEvent, FinalTranscriptEvent,
    VoiceCommandEvent
)
from ..Utils.input_validation import validate_text_input
from ..Widgets.audio_troubleshooting_dialog import AudioTroubleshootingDialog


class ImprovedDictationWindow(Widget):
    """
    Improved dictation interface with:
    - Privacy-first settings
    - Better error handling
    - Lazy initialization
    - Resource optimization
    """
    
    BINDINGS = [
        Binding("ctrl+d", "toggle_dictation", "Start/Stop Dictation"),
        Binding("ctrl+p", "pause_dictation", "Pause/Resume"),
        Binding("ctrl+e", "export_transcript", "Export Transcript"),
        Binding("ctrl+c", "copy_transcript", "Copy to Clipboard"),
        Binding("ctrl+shift+c", "clear_transcript", "Clear Transcript"),
        Binding("f1", "show_help", "Help"),
    ]
    
    DEFAULT_CSS = """
    ImprovedDictationWindow {
        height: 100%;
        width: 100%;
    }
    
    .dictation-container {
        height: 100%;
        layout: vertical;
    }
    
    .dictation-header {
        height: auto;
        padding: 1;
        border-bottom: solid $surface;
    }
    
    .dictation-content {
        height: 1fr;
        layout: horizontal;
    }
    
    .transcript-area {
        width: 2fr;
        padding: 1;
        border-right: solid $surface;
    }
    
    .transcript-display {
        height: 1fr;
        padding: 1;
        border: round $surface;
        background: $boost;
    }
    
    .dictation-sidebar {
        width: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    .control-section {
        margin-bottom: 2;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .privacy-notice {
        background: $warning;
        padding: 1;
        margin: 1 0;
        border: round $warning-darken-1;
    }
    
    .privacy-enabled {
        background: $success;
        border: round $success-darken-1;
    }
    
    .error-message {
        background: $error;
        padding: 1;
        margin: 1 0;
        border: round $error-darken-1;
    }
    
    .initialization-status {
        padding: 1;
        margin: 1 0;
        border: round $primary;
    }
    
    .status-ready {
        background: $success;
        border-color: $success-darken-1;
    }
    
    .status-error {
        background: $error;
        border-color: $error-darken-1;
    }
    
    .buffer-config {
        margin-top: 1;
    }
    
    .help-text {
        color: $text-muted;
        text-style: italic;
    }
    """
    
    # Reactive attributes
    is_dictating = reactive(False)
    is_initialized = reactive(False)
    initialization_error = reactive("")
    transcript_text = reactive("")
    word_count = reactive(0)
    duration = reactive(0.0)
    dictation_state = reactive(DictationState.IDLE)
    
    def __init__(self):
        """Initialize improved dictation window."""
        super().__init__()
        
        # Lazy service - not initialized until needed
        self.dictation_service = None
        
        # Transcript management
        self.transcript_segments = []
        self.transcript_history = []
        
        # Settings
        self.settings = self._load_settings()
        
        # Start time tracking
        self._start_time = None
    
    def compose(self) -> ComposeResult:
        """Compose the improved dictation UI."""
        with Vertical(classes="dictation-container"):
            # Header
            with Horizontal(classes="dictation-header"):
                yield Label("🎤 Live Dictation", classes="section-title")
                yield Label("Press Ctrl+D to start/stop • F1 for help", classes="help-text")
            
            # Status/Error area
            yield Container(id="status-container")
            
            # Main content area
            with Horizontal(classes="dictation-content"):
                # Transcript area
                with Vertical(classes="transcript-area"):
                    yield Label("Transcript", classes="section-title")
                    yield TextArea(
                        id="transcript-display",
                        classes="transcript-display",
                        read_only=True
                    )
                    
                    # Control buttons
                    with Horizontal():
                        yield Button(
                            "🎤 Start Dictation",
                            id="dictation-toggle-btn",
                            variant="primary"
                        )
                        yield Button(
                            "⏸️ Pause",
                            id="dictation-pause-btn",
                            disabled=True
                        )
                        yield Button(
                            "🗑️ Clear",
                            id="dictation-clear-btn"
                        )
                        yield Button(
                            "🔧 Troubleshoot",
                            id="troubleshoot-btn"
                        )
                
                # Sidebar
                with ScrollableContainer(classes="dictation-sidebar"):
                    # Privacy Settings
                    with Collapsible(title="🔒 Privacy Settings", collapsed=False):
                        with Vertical(classes="control-section"):
                            # Privacy status
                            privacy_class = "privacy-enabled" if self.settings['privacy']['local_only'] else ""
                            with Container(classes=f"privacy-notice {privacy_class}", id="privacy-status"):
                                yield Static(self._get_privacy_status_text())
                            
                            # Privacy switches
                            yield Switch(
                                value=self.settings['privacy']['save_history'],
                                id="save-history-switch"
                            )
                            yield Label("Save transcription history")
                            
                            yield Switch(
                                value=self.settings['privacy']['local_only'],
                                id="local-only-switch"
                            )
                            yield Label("Local processing only (privacy mode)")
                            
                            yield Switch(
                                value=self.settings['privacy']['auto_clear_buffer'],
                                id="auto-clear-switch"
                            )
                            yield Label("Auto-clear audio buffer")
                    
                    # Transcription Settings
                    with Collapsible(title="⚙️ Transcription Settings"):
                        with Vertical(classes="control-section"):
                            # Language selection
                            yield Label("Language:")
                            yield Select(
                                options=[
                                    ("English", "en"),
                                    ("Spanish", "es"),
                                    ("French", "fr"),
                                    ("German", "de"),
                                    ("Italian", "it"),
                                    ("Portuguese", "pt"),
                                    ("Russian", "ru"),
                                    ("Chinese", "zh"),
                                    ("Japanese", "ja"),
                                    ("Korean", "ko"),
                                ],
                                value=self.settings.get('language', 'en'),
                                id="language-select"
                            )
                            
                            # Provider selection (filtered by privacy mode)
                            yield Label("Provider:")
                            yield Select(
                                options=self._get_provider_options(),
                                value=self.settings.get('provider', 'auto'),
                                id="provider-select"
                            )
                            
                            # Options
                            yield Switch(
                                value=self.settings.get('punctuation', True),
                                id="punctuation-switch"
                            )
                            yield Label("Auto punctuation")
                            
                            yield Switch(
                                value=self.settings.get('commands', True),
                                id="commands-switch"
                            )
                            yield Label("Voice commands")
                    
                    # Performance Settings
                    with Collapsible(title="🚀 Performance", collapsed=True):
                        with Vertical(classes="control-section"):
                            yield Label("Buffer Duration (ms):")
                            yield Input(
                                value=str(self.settings.get('buffer_duration_ms', 500)),
                                id="buffer-duration-input",
                                type="integer",
                                min="100",
                                max="2000"
                            )
                            yield Static(
                                "Lower = more responsive, Higher = more stable",
                                classes="help-text"
                            )
                    
                    # Statistics
                    with Vertical(classes="control-section"):
                        yield Label("Statistics", classes="section-title")
                        with Vertical(id="stats-display", classes="stats-display"):
                            yield Static("Words: 0", id="word-count")
                            yield Static("Duration: 0:00", id="duration-display")
                            yield Static("Speed: 0 WPM", id="speed-display")
                            yield Static("State: Idle", id="state-display")
                    
                    # Export section
                    with Vertical(classes="control-section"):
                        yield Label("Export", classes="section-title")
                        yield Button("📋 Copy to Clipboard", id="copy-button")
                        yield Button("💾 Save as Text", id="save-text-button")
                        yield Button("📝 Save as Markdown", id="save-md-button")
                    
                    # History section (only if enabled)
                    if self.settings['privacy']['save_history']:
                        with Vertical(classes="control-section"):
                            yield Label("History", classes="section-title")
                            yield ListView(
                                id="history-list",
                                classes="history-list"
                            )
                            yield Button("Clear History", id="clear-history-button")
    
    def on_mount(self):
        """Initialize on mount."""
        # Update UI based on settings
        self._update_privacy_ui()
        
        # Load history if enabled
        if self.settings['privacy']['save_history']:
            self._load_history()
    
    def _get_privacy_status_text(self) -> str:
        """Get privacy status description."""
        if self.settings['privacy']['local_only']:
            return "🔒 Privacy Mode: All processing happens locally"
        else:
            return "⚠️ Standard Mode: May use cloud services"
    
    def _get_provider_options(self) -> List[tuple]:
        """Get provider options based on privacy settings."""
        if self.settings['privacy']['local_only']:
            # Only local providers in privacy mode
            return [
                ("Auto (Local)", "auto"),
                ("Parakeet MLX", "parakeet-mlx"),
                ("Faster Whisper", "faster-whisper"),
                ("Lightning Whisper", "lightning-whisper"),
            ]
        else:
            # All providers available
            return [
                ("Auto", "auto"),
                ("Parakeet MLX", "parakeet-mlx"),
                ("Faster Whisper", "faster-whisper"),
                ("Lightning Whisper", "lightning-whisper"),
                ("OpenAI Whisper", "openai-whisper"),
                ("Google Speech", "google-speech"),
            ]
    
    def _initialize_service(self) -> bool:
        """Initialize dictation service lazily."""
        if self.dictation_service is not None:
            return True
        
        try:
            # Show initialization status
            self._show_status("Initializing dictation service...", "info")
            
            # Create service with current settings
            self.dictation_service = LazyLiveDictationService(
                transcription_provider=self.settings.get('provider', 'auto'),
                transcription_model=self.settings.get('model'),
                language=self.settings.get('language', 'en'),
                enable_punctuation=self.settings.get('punctuation', True),
                enable_commands=self.settings.get('commands', True)
            )
            
            # Apply privacy settings
            self.dictation_service.update_privacy_settings(self.settings['privacy'])
            
            # Set buffer duration
            buffer_ms = self.settings.get('buffer_duration_ms', 500)
            self.dictation_service.set_buffer_duration(buffer_ms)
            
            self.is_initialized = True
            self._show_status("Dictation service ready", "success")
            return True
            
        except Exception as e:
            self.is_initialized = False
            self.initialization_error = str(e)
            self._show_status(f"Initialization failed: {e}", "error")
            logger.error(f"Failed to initialize dictation service: {e}")
            return False
    
    def _show_status(self, message: str, level: str = "info"):
        """Show status message in UI."""
        status_container = self.query_one("#status-container", Container)
        status_container.remove_children()
        
        if level == "error":
            status_container.mount(
                Static(message, classes="error-message")
            )
        elif level == "success":
            status_container.mount(
                Static(message, classes="initialization-status status-ready")
            )
        else:
            status_container.mount(
                Static(message, classes="initialization-status")
            )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "dictation-toggle-btn":
            self.action_toggle_dictation()
        elif button_id == "dictation-pause-btn":
            self.action_pause_dictation()
        elif button_id == "dictation-clear-btn":
            self.action_clear_transcript()
        elif button_id == "troubleshoot-btn":
            self._show_troubleshooting()
        elif button_id == "copy-button":
            self.action_copy_transcript()
        elif button_id == "save-text-button":
            self._export_as_text()
        elif button_id == "save-md-button":
            self._export_as_markdown()
        elif button_id == "clear-history-button":
            self._clear_history()
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        if event.switch.id == "save-history-switch":
            self.settings['privacy']['save_history'] = event.value
            self._save_settings()
            self._update_privacy_ui()
        elif event.switch.id == "local-only-switch":
            self.settings['privacy']['local_only'] = event.value
            self._save_settings()
            self._update_privacy_ui()
            # Update provider options
            provider_select = self.query_one("#provider-select", Select)
            provider_select.set_options(self._get_provider_options())
        elif event.switch.id == "auto-clear-switch":
            self.settings['privacy']['auto_clear_buffer'] = event.value
            self._save_settings()
            if self.dictation_service:
                self.dictation_service.update_privacy_settings(self.settings['privacy'])
        elif event.switch.id == "punctuation-switch":
            self.settings['punctuation'] = event.value
            self._save_settings()
        elif event.switch.id == "commands-switch":
            self.settings['commands'] = event.value
            self._save_settings()
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id == "buffer-duration-input":
            try:
                duration = int(event.value)
                if 100 <= duration <= 2000:
                    self.settings['buffer_duration_ms'] = duration
                    self._save_settings()
                    if self.dictation_service:
                        self.dictation_service.set_buffer_duration(duration)
            except ValueError:
                pass
    
    def action_toggle_dictation(self) -> None:
        """Toggle dictation on/off with lazy initialization."""
        if self.is_dictating:
            self._stop_dictation()
        else:
            self._start_dictation()
    
    def _start_dictation(self):
        """Start dictation with initialization if needed."""
        # Initialize service if not already done
        if not self._initialize_service():
            return
        
        # Update UI
        toggle_btn = self.query_one("#dictation-toggle-btn", Button)
        toggle_btn.label = "🛑 Stop Dictation"
        toggle_btn.variant = "error"
        
        pause_btn = self.query_one("#dictation-pause-btn", Button)
        pause_btn.disabled = False
        
        # Start dictation
        success = self.dictation_service.start_dictation(
            on_partial_transcript=self._on_partial_transcript,
            on_final_transcript=self._on_final_transcript,
            on_state_change=self._on_state_change,
            on_error=self._on_error,
            on_command=self._on_command,
            save_audio=False  # Respect privacy settings
        )
        
        if success:
            self.is_dictating = True
            self._start_time = datetime.now()
            self._show_status("Listening...", "success")
        else:
            self._show_status("Failed to start dictation", "error")
            # Reset UI
            toggle_btn.label = "🎤 Start Dictation"
            toggle_btn.variant = "primary"
            pause_btn.disabled = True
    
    def _stop_dictation(self):
        """Stop dictation."""
        if not self.dictation_service:
            return
        
        result = self.dictation_service.stop_dictation()
        
        # Update UI
        toggle_btn = self.query_one("#dictation-toggle-btn", Button)
        toggle_btn.label = "🎤 Start Dictation"
        toggle_btn.variant = "primary"
        
        pause_btn = self.query_one("#dictation-pause-btn", Button)
        pause_btn.disabled = True
        pause_btn.label = "⏸️ Pause"
        
        self.is_dictating = False
        self.duration = result.duration
        self.word_count = len(result.transcript.split())
        self._update_stats()
        
        # Save to history if enabled
        if result.transcript and self.settings['privacy']['save_history']:
            self._add_to_history(result.transcript)
        
        self._show_status("Dictation stopped", "info")
    
    def _on_partial_transcript(self, text: str):
        """Handle partial transcript updates."""
        self._update_transcript_display(text, is_partial=True)
    
    def _on_final_transcript(self, text: str):
        """Handle final transcript segments."""
        self._add_transcript_segment(text)
    
    def _on_state_change(self, state: str):
        """Handle dictation state changes."""
        self.dictation_state = state
        state_display = self.query_one("#state-display", Static)
        state_display.update(f"State: {state}")
    
    def _on_error(self, error: Exception):
        """Handle dictation errors."""
        self._show_status(f"Error: {error}", "error")
        logger.error(f"Dictation error: {error}")
    
    def _on_command(self, command: str):
        """Handle voice commands."""
        self.post_message(VoiceCommandEvent(command=command))
        logger.info(f"Voice command detected: {command}")
    
    def _update_privacy_ui(self):
        """Update UI based on privacy settings."""
        privacy_status = self.query_one("#privacy-status", Container)
        privacy_status.remove_children()
        privacy_status.mount(Static(self._get_privacy_status_text()))
        
        if self.settings['privacy']['local_only']:
            privacy_status.add_class("privacy-enabled")
        else:
            privacy_status.remove_class("privacy-enabled")
    
    def _show_troubleshooting(self):
        """Show audio troubleshooting dialog."""
        self.app.push_screen(
            AudioTroubleshootingDialog(),
            callback=self._on_troubleshooting_complete
        )
    
    def _on_troubleshooting_complete(self, result: bool):
        """Handle troubleshooting dialog completion."""
        if result:
            # Reload settings in case device preference changed
            self.settings = self._load_settings()
            
            # Reset service to pick up new settings
            if self.dictation_service:
                if self.is_dictating:
                    self._stop_dictation()
                self.dictation_service = None
            
            self._show_status("Audio settings updated", "info")
    
    def action_show_help(self):
        """Show help dialog."""
        help_text = """
Dictation Help:

Keyboard Shortcuts:
• Ctrl+D: Start/Stop dictation
• Ctrl+P: Pause/Resume
• Ctrl+C: Copy transcript
• Ctrl+E: Export transcript
• Ctrl+Shift+C: Clear transcript

Voice Commands (when enabled):
• "new paragraph": Insert paragraph break
• "new line": Insert line break
• "comma/period/question mark": Insert punctuation
• "clear all": Clear transcript
• "stop dictation": Stop recording

Privacy Settings:
• Local Only: All processing on your device
• Save History: Keep transcripts between sessions
• Auto-clear Buffer: Remove audio data after processing

Performance Tips:
• Lower buffer duration for faster response
• Higher buffer duration for stability
• Use local providers for privacy
        """
        self.app.notify(help_text.strip(), title="Dictation Help", timeout=15)
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load dictation settings with privacy defaults."""
        settings = {
            'provider': get_cli_setting('dictation.provider', 'auto'),
            'model': get_cli_setting('dictation.model', None),
            'language': get_cli_setting('dictation.language', 'en'),
            'punctuation': get_cli_setting('dictation.punctuation', True),
            'commands': get_cli_setting('dictation.commands', True),
            'buffer_duration_ms': get_cli_setting('dictation.buffer_duration_ms', 500),
            'privacy': {
                'save_history': get_cli_setting('dictation.privacy.save_history', False),
                'local_only': get_cli_setting('dictation.privacy.local_only', True),
                'auto_clear_buffer': get_cli_setting('dictation.privacy.auto_clear_buffer', True),
            }
        }
        return settings
    
    def _save_settings(self):
        """Save dictation settings."""
        save_setting_to_cli_config('dictation', 'provider', self.settings['provider'])
        save_setting_to_cli_config('dictation', 'model', self.settings.get('model'))
        save_setting_to_cli_config('dictation', 'language', self.settings['language'])
        save_setting_to_cli_config('dictation', 'punctuation', self.settings['punctuation'])
        save_setting_to_cli_config('dictation', 'commands', self.settings['commands'])
        save_setting_to_cli_config('dictation', 'buffer_duration_ms', self.settings['buffer_duration_ms'])
        
        # Save privacy settings
        for key, value in self.settings['privacy'].items():
            save_setting_to_cli_config('dictation.privacy', key, value)
    
    # Include other helper methods from original implementation...
    # (transcript management, export, etc. remain similar)