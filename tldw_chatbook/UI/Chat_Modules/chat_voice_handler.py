"""
Chat Voice Handler Module

Handles all voice input functionality including:
- Voice recording initialization
- Microphone button management
- Speech-to-text processing
- Voice input widget integration
- Audio error handling
"""

from typing import TYPE_CHECKING, Optional
from loguru import logger
from textual import work
from textual.widgets import Button
from textual.worker import WorkerCancelled
from textual.css.query import NoMatches

if TYPE_CHECKING:
    from ..Chat_Window_Enhanced import ChatWindowEnhanced

logger = logger.bind(module="ChatVoiceHandler")


class ChatVoiceHandler:
    """Handles voice input and recording functionality."""
    
    def __init__(self, chat_window: 'ChatWindowEnhanced'):
        """Initialize the voice handler.
        
        Args:
            chat_window: Parent ChatWindowEnhanced instance
        """
        self.chat_window = chat_window
        self.app_instance = chat_window.app_instance
        self.voice_dictation_service = None
        self.is_voice_recording = False
    
    async def handle_mic_button(self, event):
        """Handle microphone button press for voice input.
        
        Args:
            event: Button.Pressed event
        """
        # Call the toggle action
        self.toggle_voice_input()
    
    def toggle_voice_input(self) -> None:
        """Toggle voice input recording."""
        if not hasattr(self, 'voice_dictation_service') or not self.voice_dictation_service:
            # Create voice dictation service if not exists
            self._create_voice_input_service()
            
        if not self.voice_dictation_service:
            self.app_instance.notify("Voice input not available", severity="error")
            return
        
        if self.is_voice_recording:
            self._stop_voice_recording()
        else:
            self._start_voice_recording()
    
    def _create_voice_input_service(self):
        """Create voice dictation service."""
        try:
            from ...config import get_cli_setting
            from ...Audio.dictation_service_lazy import LazyLiveDictationService, AudioInitializationError
            
            self.voice_dictation_service = LazyLiveDictationService(
                transcription_provider=get_cli_setting('transcription', 'default_provider', 'faster-whisper'),
                transcription_model=get_cli_setting('transcription', 'default_model', 'base'),
                language=get_cli_setting('transcription', 'default_language', 'en'),
                enable_punctuation=True,
                enable_commands=False
            )
            logger.info("Voice dictation service created")
        except ImportError as e:
            logger.error(f"Voice dictation dependencies not available: {e}")
            self.voice_dictation_service = None
        except AttributeError as e:
            logger.error(f"Failed to initialize voice dictation service: {e}")
            self.voice_dictation_service = None
    
    def _start_voice_recording(self):
        """Start voice recording with proper worker management."""
        try:
            # Update UI immediately with batch update
            try:
                mic_button = self.chat_window.query_one("#mic-button", Button)
                with self.chat_window.app.batch_update():
                    mic_button.label = "🛑"  # Stop icon
                    mic_button.variant = "error"
            except NoMatches:
                pass  # Mic button not found
            
            # Run recording in worker
            self.chat_window.run_worker(
                self._start_voice_recording_worker,
                exclusive=True,
                name="voice_recorder"
            )
        except (WorkerCancelled, RuntimeError) as e:
            logger.error(f"Failed to start voice recording worker: {e}")
            self._reset_mic_button()
    
    @work(thread=True)
    def _start_voice_recording_worker(self):
        """Start voice recording in a worker thread."""
        try:
            from ...Audio.dictation_service_lazy import AudioInitializationError
            
            # Start dictation (should be synchronous for thread workers)
            success = self.voice_dictation_service.start_dictation(
                on_partial_transcript=self._on_voice_partial,
                on_final_transcript=self._on_voice_final,
                on_error=self._on_voice_error
            )
            
            if success:
                self.chat_window.call_from_thread(self._on_voice_recording_started)
            else:
                self.chat_window.call_from_thread(
                    self.app_instance.notify,
                    "Failed to start recording",
                    severity="error"
                )
                self.chat_window.call_from_thread(self._reset_mic_button)
                
        except AudioInitializationError as e:
            logger.error(f"Audio initialization error: {e}", extra={"error_type": "audio_init"})
            self.chat_window.call_from_thread(
                self.app_instance.notify,
                str(e),
                severity="error",
                timeout=10
            )
            self.chat_window.call_from_thread(self._reset_mic_button)
        except (RuntimeError, AttributeError) as e:
            logger.error(f"Error starting voice recording: {e}", extra={"error_type": "voice_recording"})
            error_msg = self._get_voice_error_message(e)
            self.chat_window.call_from_thread(
                self.app_instance.notify,
                error_msg,
                severity="error",
                timeout=10 if "permission" in error_msg.lower() else 5
            )
            self.chat_window.call_from_thread(self._reset_mic_button)
    
    def _stop_voice_recording(self):
        """Stop voice recording."""
        if self.voice_dictation_service:
            try:
                self.voice_dictation_service.stop_dictation()
                self.is_voice_recording = False
                self._reset_mic_button()
                self.app_instance.notify("Recording stopped", timeout=2)
            except (RuntimeError, AttributeError) as e:
                logger.error(f"Error stopping voice recording: {e}")
                self.app_instance.notify("Failed to stop recording", severity="error")
    
    def _on_voice_recording_started(self):
        """Handle successful voice recording start."""
        self.is_voice_recording = True
        self.app_instance.notify("🎤 Listening...", timeout=2)
    
    def _on_voice_partial(self, text: str):
        """Handle partial voice transcript.
        
        Args:
            text: Partial transcript text
        """
        # Could update UI with partial text if desired
        logger.debug(f"Partial transcript: {text}")
    
    def _on_voice_final(self, text: str):
        """Handle final voice transcript.
        
        Args:
            text: Final transcript text
        """
        if text and self.chat_window._chat_input:
            # Insert text into chat input
            current_text = self.chat_window._chat_input.value
            if current_text and not current_text.endswith(' '):
                text = ' ' + text
            self.chat_window._chat_input.value = current_text + text
            
            # Stop recording after successful transcription
            self._stop_voice_recording()
    
    def _on_voice_error(self, error: str):
        """Handle voice recording error.
        
        Args:
            error: Error message
        """
        logger.error(f"Voice recording error: {error}")
        self.app_instance.notify(f"Voice error: {error}", severity="error")
        self._reset_mic_button()
        self.is_voice_recording = False
    
    def _reset_mic_button(self):
        """Reset microphone button to default state."""
        try:
            from textual.widgets import Button
            mic_button = self.chat_window.query_one("#mic-button", Button)
            with self.chat_window.app.batch_update():
                mic_button.label = "🎤"
                mic_button.variant = "default"
        except (AttributeError, NoMatches):
            # Widget might not exist yet
            pass
    
    def _get_voice_error_message(self, error: Exception) -> str:
        """Get user-friendly error message for voice recording errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            User-friendly error message
        """
        error_str = str(error).lower()
        
        if "permission" in error_str or "access" in error_str:
            return "🎤 Microphone permission denied. Please allow microphone access in System Settings."
        elif "no audio" in error_str or "no input" in error_str:
            return "🎤 No microphone detected. Please connect a microphone."
        elif "initialize" in error_str:
            return "🎤 Failed to initialize audio. Please check your audio settings."
        elif "busy" in error_str or "in use" in error_str:
            return "🎤 Microphone is being used by another application."
        else:
            return f"🎤 Voice input error: {error}"
    
    def cleanup(self):
        """Clean up voice resources."""
        if self.is_voice_recording:
            self._stop_voice_recording()
        
        if self.voice_dictation_service:
            try:
                # Clean up any resources
                self.voice_dictation_service.stop_dictation()
            except:
                pass
            self.voice_dictation_service = None