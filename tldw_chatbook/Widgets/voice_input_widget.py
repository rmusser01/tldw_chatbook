# voice_input_widget.py
"""
Reusable voice input widget for speech recording and live dictation.
Provides visual feedback and controls for voice input across the application.
"""

from typing import Optional, Callable, List, Dict, Any
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static, Select, ProgressBar
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual import work
from textual.worker import Worker, WorkerState
import asyncio
from loguru import logger

# Local imports
from ..Audio import LiveDictationService, DictationState
from ..Event_Handlers.Audio_Events import (
    DictationStartedEvent, DictationStoppedEvent,
    PartialTranscriptEvent, FinalTranscriptEvent,
    VoiceCommandEvent, DictationErrorEvent,
    AudioLevelUpdateEvent
)
from ..Utils.Emoji_Handling import get_char


class VoiceInputMessage(Message):
    """Message sent when voice input produces text."""
    
    def __init__(self, text: str, is_final: bool = True):
        super().__init__()
        self.text = text
        self.is_final = is_final


class VoiceInputWidget(Widget):
    """
    Reusable voice input widget with visual feedback.
    
    Features:
    - Record button with state indication
    - Audio level visualization
    - Device selection
    - Transcription preview
    - Error handling
    """
    
    DEFAULT_CSS = """
    VoiceInputWidget {
        height: auto;
        width: 100%;
    }
    
    .voice-input-container {
        height: auto;
        padding: 1;
        border: round $surface;
    }
    
    .voice-controls {
        height: auto;
        align: center middle;
        margin-bottom: 1;
    }
    
    .record-button {
        width: 12;
        height: 3;
        margin: 0 1;
    }
    
    .record-button.recording {
        background: $error;
        color: white;
    }
    
    .record-button.paused {
        background: $warning;
        color: black;
    }
    
    .audio-level-container {
        width: 20;
        height: 1;
        border: round $surface;
        margin: 0 1;
    }
    
    .audio-level-bar {
        background: $success;
        height: 100%;
    }
    
    .device-selector {
        width: 30;
        margin: 0 1;
    }
    
    .transcript-preview {
        min-height: 3;
        max-height: 6;
        padding: 1;
        border: round $surface;
        background: $boost;
        margin-top: 1;
    }
    
    .voice-status {
        text-align: center;
        margin-top: 1;
        text-style: italic;
    }
    
    .voice-error {
        color: $error;
        text-align: center;
        margin-top: 1;
    }
    """
    
    # Widget state
    state = reactive(DictationState.IDLE)
    audio_level = reactive(0.0)
    current_transcript = reactive("")
    error_message = reactive("")
    
    def __init__(
        self,
        show_device_selector: bool = True,
        show_transcript_preview: bool = True,
        transcription_provider: str = 'auto',
        transcription_model: Optional[str] = None,
        language: str = 'en',
        placeholder: str = "Click the microphone to start dictation..."
    ):
        """
        Initialize voice input widget.
        
        Args:
            show_device_selector: Whether to show device selection dropdown
            show_transcript_preview: Whether to show live transcript preview
            transcription_provider: Provider for transcription
            transcription_model: Model to use for transcription
            language: Language code
            placeholder: Placeholder text when not recording
        """
        super().__init__()
        self.show_device_selector = show_device_selector
        self.show_transcript_preview = show_transcript_preview
        self.transcription_provider = transcription_provider
        self.transcription_model = transcription_model
        self.language = language
        self.placeholder = placeholder
        
        # Services
        self.dictation_service = None
        
        # Workers
        self.level_monitor_worker = None
        
        # Audio devices
        self.audio_devices = []
        self.selected_device_id = None
    
    def compose(self) -> ComposeResult:
        """Compose the widget UI."""
        with Vertical(classes="voice-input-container"):
            # Controls row
            with Horizontal(classes="voice-controls"):
                yield Button(
                    self._get_record_button_label(),
                    id="record-button",
                    classes=f"record-button {self._get_button_class()}"
                )
                
                # Audio level indicator
                with Horizontal(classes="audio-level-container"):
                    yield Static("", id="audio-level-bar", classes="audio-level-bar")
                
                # Device selector
                if self.show_device_selector:
                    yield Select(
                        options=[],
                        id="device-selector",
                        classes="device-selector",
                        prompt="Select microphone..."
                    )
            
            # Transcript preview
            if self.show_transcript_preview:
                yield Static(
                    self.placeholder,
                    id="transcript-preview",
                    classes="transcript-preview"
                )
            
            # Status message
            yield Static("", id="voice-status", classes="voice-status")
            
            # Error message
            yield Static("", id="voice-error", classes="voice-error")
    
    async def on_mount(self):
        """Initialize widget on mount."""
        try:
            # Initialize dictation service
            self.dictation_service = LiveDictationService(
                transcription_provider=self.transcription_provider,
                transcription_model=self.transcription_model,
                language=self.language
            )
            
            # Load audio devices
            await self._load_audio_devices()
            
        except Exception as e:
            logger.error(f"Failed to initialize voice input: {e}")
            self.error_message = f"Voice input unavailable: {str(e)}"
    
    async def _load_audio_devices(self):
        """Load available audio devices."""
        if not self.dictation_service:
            return
        
        try:
            devices = self.dictation_service.get_audio_devices()
            self.audio_devices = devices
            
            if self.show_device_selector:
                device_selector = self.query_one("#device-selector", Select)
                
                # Format device options
                options = []
                for device in devices:
                    label = device['name']
                    if device.get('is_default'):
                        label += " (Default)"
                    options.append((label, device['id']))
                
                device_selector.set_options(options)
                
                # Select default device
                default_device = next((d for d in devices if d.get('is_default')), None)
                if default_device:
                    device_selector.value = default_device['id']
                    self.selected_device_id = default_device['id']
        
        except Exception as e:
            logger.error(f"Failed to load audio devices: {e}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "record-button":
            if self.state == DictationState.IDLE:
                self.start_recording()
            elif self.state == DictationState.LISTENING:
                self.stop_recording()
            elif self.state == DictationState.PAUSED:
                self.resume_recording()
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle device selection change."""
        if event.select.id == "device-selector" and event.value is not None:
            self.selected_device_id = event.value
            if self.dictation_service:
                self.dictation_service.set_audio_device(event.value)
                logger.info(f"Changed audio device to: {event.value}")
    
    @work(exclusive=True, thread=True)
    def start_recording(self):
        """Start voice recording and dictation."""
        try:
            # Initialize service if needed
            if not self.dictation_service:
                self._initialize_dictation_service()
                
            if not self.dictation_service:
                self.app.call_from_thread(self._set_error, "Voice input not initialized")
                return
            # Clear previous state
            self.app.call_from_thread(self._clear_state)
            
            # Set device if selected
            if self.selected_device_id is not None:
                self.dictation_service.set_audio_device(self.selected_device_id)
            
            # Start dictation
            success = self.dictation_service.start_dictation(
                on_partial_transcript=self._on_partial_transcript,
                on_final_transcript=self._on_final_transcript,
                on_state_change=self._on_state_change,
                on_error=self._on_error,
                on_command=self._on_command
            )
            
            if success:
                self.app.call_from_thread(self._set_state, DictationState.LISTENING)
                self.post_message(DictationStartedEvent(
                    provider=self.transcription_provider,
                    model=self.transcription_model
                ))
                
                # Start audio level monitoring
                self.app.call_from_thread(self._start_level_monitoring)
            else:
                self.app.call_from_thread(self._set_error, "Failed to start recording")
        
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.app.call_from_thread(self._set_error, f"Recording error: {str(e)}")
    
    @work(exclusive=True, thread=True)
    def stop_recording(self):
        """Stop voice recording and get final transcript."""
        if not self.dictation_service:
            return
        
        try:
            # Stop level monitoring
            self._stop_level_monitoring()
            
            # Stop dictation
            result = self.dictation_service.stop_dictation()
            
            # Update state
            self.app.call_from_thread(self._set_state, DictationState.IDLE)
            self.app.call_from_thread(self._set_audio_level, 0.0)
            
            # Send final transcript
            if result.transcript:
                self.post_message(VoiceInputMessage(result.transcript, is_final=True))
                self.post_message(DictationStoppedEvent(
                    transcript=result.transcript,
                    duration=result.duration,
                    word_count=len(result.transcript.split())
                ))
            
            # Clear transcript preview
            if self.show_transcript_preview:
                self.app.call_from_thread(self._update_preview, self.placeholder)
        
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            self.app.call_from_thread(self._set_error, f"Stop error: {str(e)}")
    
    def pause_recording(self):
        """Pause recording."""
        if self.dictation_service and self.state == DictationState.LISTENING:
            if self.dictation_service.pause_dictation():
                self.state = DictationState.PAUSED
    
    def resume_recording(self):
        """Resume paused recording."""
        if self.dictation_service and self.state == DictationState.PAUSED:
            if self.dictation_service.resume_dictation():
                self.state = DictationState.LISTENING
    
    def _on_partial_transcript(self, text: str):
        """Handle partial transcript update."""
        self.current_transcript = text
        self.app.call_from_thread(self._update_transcript_preview)
        self.post_message(PartialTranscriptEvent(text))
        self.post_message(VoiceInputMessage(text, is_final=False))
    
    def _on_final_transcript(self, text: str):
        """Handle final transcript segment."""
        self.post_message(FinalTranscriptEvent(text, 0))
    
    def _on_state_change(self, new_state: str):
        """Handle dictation state change."""
        old_state = self.state
        self.state = new_state
        self.app.call_from_thread(self._update_ui_state)
    
    def _on_error(self, error: Exception):
        """Handle dictation error."""
        self.error_message = str(error)
        self.post_message(DictationErrorEvent(error))
    
    def _on_command(self, command: str):
        """Handle voice command detection."""
        self.post_message(VoiceCommandEvent(command, ""))
    
    def _update_transcript_preview(self):
        """Update transcript preview display."""
        if self.show_transcript_preview:
            preview = self.query_one("#transcript-preview", Static)
            if self.current_transcript:
                preview.update(self.current_transcript)
            else:
                preview.update(self.placeholder)
    
    def _update_ui_state(self):
        """Update UI based on current state."""
        # Update button
        button = self.query_one("#record-button", Button)
        button.label = self._get_record_button_label()
        button.classes = f"record-button {self._get_button_class()}"
        
        # Update status
        status = self.query_one("#voice-status", Static)
        status.update(self._get_status_text())
    
    def _get_record_button_label(self) -> str:
        """Get appropriate button label for current state."""
        if self.state == DictationState.IDLE:
            return f"{get_char('🎤', '⚫')} Record"
        elif self.state == DictationState.LISTENING:
            return f"{get_char('⏹️', '⬜')} Stop"
        elif self.state == DictationState.PAUSED:
            return f"{get_char('▶️', '►')} Resume"
        else:
            return "Record"
    
    def _get_button_class(self) -> str:
        """Get button CSS class for current state."""
        if self.state == DictationState.LISTENING:
            return "recording"
        elif self.state == DictationState.PAUSED:
            return "paused"
        return ""
    
    def _get_status_text(self) -> str:
        """Get status text for current state."""
        if self.state == DictationState.LISTENING:
            return "Listening..."
        elif self.state == DictationState.PAUSED:
            return "Paused"
        elif self.state == DictationState.PROCESSING:
            return "Processing..."
        return ""
    
    @work(exclusive=True, thread=True)
    def _start_level_monitoring(self):
        """Start monitoring audio levels."""
        self.level_monitor_worker = self.run_worker(
            self._monitor_audio_levels,
            exclusive=True,
            thread=True
        )
    
    def _stop_level_monitoring(self):
        """Stop monitoring audio levels."""
        if self.level_monitor_worker:
            self.level_monitor_worker.cancel()
            self.level_monitor_worker = None
    
    async def _monitor_audio_levels(self):
        """Monitor and update audio levels."""
        while self.state == DictationState.LISTENING:
            if self.dictation_service:
                level = self.dictation_service.get_audio_level()
                self.audio_level = level
                self._update_level_display(level)
                self.post_message(AudioLevelUpdateEvent(level))
            
            await asyncio.sleep(0.1)
    
    def _update_level_display(self, level: float):
        """Update audio level visual indicator."""
        level_bar = self.query_one("#audio-level-bar", Static)
        # Update width based on level (0.0 to 1.0)
        width_percent = int(level * 100)
        level_bar.styles.width = f"{width_percent}%"
    
    def watch_error_message(self, error: str):
        """React to error message changes."""
        try:
            error_display = self.query_one("#voice-error", Static)
            error_display.update(error)
        except Exception:
            # Widget not ready yet
            pass
    
    def get_transcript(self) -> str:
        """Get the current full transcript."""
        if self.dictation_service:
            return self.dictation_service.get_full_transcript()
        return ""
    
    def _initialize_dictation_service(self):
        """Initialize the dictation service."""
        try:
            from ..config import get_cli_setting
            
            self.dictation_service = LazyLiveDictationService(
                transcription_provider=self.transcription_provider,
                transcription_model=self.transcription_model,
                language=self.language,
                enable_punctuation=True,
                enable_commands=False
            )
            logger.info("Dictation service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize dictation service: {e}")
            self.dictation_service = None
    
    def _clear_state(self):
        """Clear state - safe to call from any thread."""
        self.error_message = ""
        self.current_transcript = ""
    
    def _set_state(self, state: str):
        """Set state - safe to call from any thread."""
        self.state = state
    
    def _set_error(self, error: str):
        """Set error - safe to call from any thread."""
        self.error_message = error
    
    def _show_status(self, status: str):
        """Show status - safe to call from any thread."""
        try:
            status_widget = self.query_one("#voice-status", Static)
            status_widget.update(status)
        except Exception:
            pass
    
    def _start_level_monitoring(self):
        """Start audio level monitoring."""
        if not hasattr(self, 'level_monitor_worker') or not self.level_monitor_worker:
            self.level_monitor_worker = self.run_worker(
                self._monitor_audio_levels,
                exclusive=True
            )
    
    def _set_audio_level(self, level: float):
        """Set audio level - safe to call from any thread."""
        self.audio_level = level
    
    def _update_preview(self, text: str):
        """Update preview text - safe to call from any thread."""
        try:
            preview = self.query_one("#transcript-preview", Static)
            preview.update(text)
        except Exception:
            pass
    
    def _update_transcript_preview(self):
        """Update transcript preview from current transcript."""
        if self.show_transcript_preview:
            try:
                preview = self.query_one("#transcript-preview", Static)
                preview.update(self.current_transcript or self.placeholder)
            except Exception:
                pass
    
    def _update_ui_state(self):
        """Update UI based on current state."""
        try:
            # Update button
            button = self.query_one("#record-button", Button)
            button.label = self._get_record_button_label()
            button.classes = f"record-button {self._get_button_class()}"
            
            # Update status
            status = self.query_one("#voice-status", Static)
            status_text = {
                DictationState.IDLE: "",
                DictationState.LISTENING: "Listening...",
                DictationState.PROCESSING: "Processing...",
                DictationState.PAUSED: "Paused"
            }.get(self.state, "")
            status.update(status_text)
        except Exception:
            pass
    
    def clear_transcript(self):
        """Clear the current transcript."""
        self.current_transcript = ""
        if self.show_transcript_preview:
            preview = self.query_one("#transcript-preview", Static)
            preview.update(self.placeholder)