# voice_input_button.py
"""
Reusable voice input button widget that can be added to any text input area.
Provides quick access to dictation functionality from anywhere in the app.
"""

from typing import Optional, Callable, Literal
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Button, Static, LoadingIndicator
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual import work
from loguru import logger

from ..Audio.dictation_service_lazy import LazyLiveDictationService, DictationState
from ..Event_Handlers.Audio_Events.dictation_integration_events import (
    VoiceInputRequestEvent,
    VoiceInputResponseEvent,
    InsertDictationTextEvent
)


class VoiceInputButton(Widget):
    """
    A button widget that provides voice input functionality.
    Can be placed next to any text input field for quick voice entry.
    """
    
    DEFAULT_CSS = """
    VoiceInputButton {
        width: auto;
        height: 3;
        layout: horizontal;
        align: center middle;
    }
    
    .voice-button {
        width: auto;
        margin: 0 1;
    }
    
    .voice-button.recording {
        background: $error;
    }
    
    .voice-button.processing {
        background: $warning;
    }
    
    .recording-indicator {
        width: 3;
        height: 3;
        background: $error;
        border: round $error;
        margin: 0 1;
    }
    
    .recording-indicator.pulse {
        /* Textual doesn't support CSS animations yet */
        opacity: 0.8;
    }
    
    .status-text {
        margin: 0 1;
        width: auto;
    }
    
    .error-text {
        color: $error;
    }
    """
    
    is_recording = reactive(False)
    is_processing = reactive(False)
    status_text = reactive("")
    
    def __init__(
        self,
        target_widget_id: Optional[str] = None,
        target_type: Literal["chat", "notes", "search", "general"] = "general",
        on_result: Optional[Callable[[str], None]] = None,
        show_status: bool = True,
        auto_insert: bool = True,
        **kwargs
    ):
        """
        Initialize voice input button.
        
        Args:
            target_widget_id: ID of the widget to insert text into
            target_type: Type of input for context-aware processing
            on_result: Callback for when transcription is complete
            show_status: Whether to show status text
            auto_insert: Whether to automatically insert text
        """
        super().__init__(**kwargs)
        self.target_widget_id = target_widget_id
        self.target_type = target_type
        self.on_result = on_result
        self.show_status = show_status
        self.auto_insert = auto_insert
        
        # Lazy-loaded service
        self._dictation_service = None
        self._current_session_id = None
    
    def compose(self) -> ComposeResult:
        """Compose the voice input UI."""
        with Container():
            yield Button(
                "🎤 Voice",
                id="voice-input-btn",
                classes="voice-button"
            )
            
            if self.show_status:
                yield Static(
                    "",
                    id="voice-status",
                    classes="status-text"
                )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press to toggle recording."""
        if event.button.id == "voice-input-btn":
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording()
    
    @work(exclusive=True)
    async def _start_recording(self):
        """Start voice recording."""
        try:
            # Update UI
            self.is_recording = True
            self._update_button_state()
            self._set_status("Initializing...")
            
            # Initialize service if needed
            if self._dictation_service is None:
                self._dictation_service = LazyLiveDictationService(
                    language=self.app.app_config.get('dictation', {}).get('language', 'en'),
                    enable_punctuation=True,
                    enable_commands=False  # Disable commands for inline input
                )
            
            # Start dictation
            success = await self.run_worker(
                self._dictation_service.start_dictation,
                on_partial_transcript=self._on_partial,
                on_final_transcript=self._on_final,
                on_error=self._on_error,
                on_state_change=self._on_state_change
            ).wait()
            
            if success:
                self._set_status("Listening...")
            else:
                self._set_status("Failed to start", is_error=True)
                self.is_recording = False
                self._update_button_state()
                
        except Exception as e:
            logger.error(f"Voice input error: {e}")
            self._set_status(f"Error: {str(e)}", is_error=True)
            self.is_recording = False
            self._update_button_state()
    
    def _stop_recording(self):
        """Stop voice recording."""
        if self._dictation_service:
            self.is_recording = False
            self.is_processing = True
            self._update_button_state()
            self._set_status("Processing...")
            
            result = self._dictation_service.stop_dictation()
            
            self.is_processing = False
            self._update_button_state()
            
            if result.transcript:
                self._handle_result(result.transcript)
                self._set_status(f"Added {len(result.transcript.split())} words")
            else:
                self._set_status("No speech detected", is_error=True)
    
    def _on_partial(self, text: str):
        """Handle partial transcript."""
        if text:
            self._set_status(f"Hearing: {text[:50]}...")
    
    def _on_final(self, text: str):
        """Handle final transcript segment."""
        # For inline voice input, we typically wait until stop
        pass
    
    def _on_error(self, error: Exception):
        """Handle dictation error."""
        self._set_status(f"Error: {str(error)}", is_error=True)
        self.is_recording = False
        self._update_button_state()
    
    def _on_state_change(self, state: str):
        """Handle state change."""
        if state == DictationState.LISTENING:
            self._set_status("Speak now...")
        elif state == DictationState.PROCESSING:
            self._set_status("Processing...")
    
    def _handle_result(self, text: str):
        """Handle the final transcription result."""
        # Fire event for other widgets to handle
        self.post_message(
            VoiceInputResponseEvent(
                source_widget_id=self.id,
                text=text,
                success=True
            )
        )
        
        # Auto-insert if configured
        if self.auto_insert:
            if self.target_widget_id:
                # Insert into specific widget
                self.post_message(
                    InsertDictationTextEvent(
                        text=text,
                        append_space=True
                    )
                )
            else:
                # Insert at active input
                self.app.post_message(
                    InsertDictationTextEvent(
                        text=text,
                        append_space=True
                    )
                )
        
        # Call custom callback if provided
        if self.on_result:
            self.on_result(text)
    
    def _update_button_state(self):
        """Update button appearance based on state."""
        button = self.query_one("#voice-input-btn", Button)
        
        if self.is_recording:
            button.label = "🛑 Stop"
            button.add_class("recording")
            button.variant = "error"
        elif self.is_processing:
            button.label = "⏳ Processing"
            button.add_class("processing")
            button.variant = "warning"
        else:
            button.label = "🎤 Voice"
            button.remove_class("recording", "processing")
            button.variant = "default"
    
    def _set_status(self, text: str, is_error: bool = False):
        """Update status text."""
        self.status_text = text
        if self.show_status:
            status_widget = self.query_one("#voice-status", Static)
            status_widget.update(text)
            
            if is_error:
                status_widget.add_class("error-text")
            else:
                status_widget.remove_class("error-text")
    
    def cleanup(self):
        """Clean up resources."""
        if self._dictation_service and self.is_recording:
            self._dictation_service.stop_dictation()
        self._dictation_service = None


class FloatingVoiceInput(Widget):
    """
    A floating voice input widget that can be positioned anywhere.
    Useful for adding voice input to existing screens without modification.
    """
    
    DEFAULT_CSS = """
    FloatingVoiceInput {
        layer: overlay;
        width: auto;
        height: auto;
        background: $panel;
        border: round $primary;
        padding: 1;
        dock: bottom right;
        margin: 2;
        offset: -2 -2;
    }
    
    .floating-container {
        layout: vertical;
        width: auto;
        height: auto;
    }
    
    .floating-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .transcript-preview {
        width: 40;
        height: 3;
        border: solid $surface;
        padding: 0 1;
        margin: 1 0;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize floating voice input."""
        super().__init__(**kwargs)
        self.transcript_preview = ""
    
    def compose(self) -> ComposeResult:
        """Compose floating UI."""
        with Container(classes="floating-container"):
            yield Label("Voice Input", classes="floating-title")
            yield VoiceInputButton(
                show_status=True,
                on_result=self._on_voice_result
            )
            yield Static(
                "Preview will appear here...",
                id="transcript-preview",
                classes="transcript-preview"
            )
            with Container():
                yield Button("Insert", id="insert-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn")
    
    def _on_voice_result(self, text: str):
        """Handle voice input result."""
        self.transcript_preview = text
        preview = self.query_one("#transcript-preview", Static)
        preview.update(text[:100] + "..." if len(text) > 100 else text)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "insert-btn":
            if self.transcript_preview:
                self.app.post_message(
                    InsertDictationTextEvent(
                        text=self.transcript_preview,
                        append_space=True
                    )
                )
            self.remove()
        elif event.button.id == "cancel-btn":
            self.remove()