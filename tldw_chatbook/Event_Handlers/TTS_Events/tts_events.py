# tts_events.py
# Description: Event handlers for TTS functionality
#
# Imports
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
from loguru import logger

# Third-party imports
from textual.app import App
from textual.message import Message
from textual.worker import Worker, WorkerState

# Local imports
from tldw_chatbook.TTS import get_tts_service, OpenAISpeechRequest
from tldw_chatbook.config import get_cli_setting

#######################################################################################################################
#
# TTS Event Messages

class TTSRequestEvent(Message):
    """Event to request TTS generation"""
    
    def __init__(self, text: str, message_id: Optional[str] = None, voice: Optional[str] = None):
        super().__init__()
        self.text = text
        self.message_id = message_id  # ID of the chat message
        self.voice = voice  # Optional voice override

class TTSStreamingEvent(Message):
    """Event for streaming TTS audio chunks"""
    
    def __init__(self, chunk: bytes, message_id: str, is_final: bool = False):
        super().__init__()
        self.chunk = chunk
        self.message_id = message_id
        self.is_final = is_final

class TTSCompleteEvent(Message):
    """Event when TTS generation is complete"""
    
    def __init__(self, message_id: str, audio_file: Optional[Path] = None, error: Optional[str] = None):
        super().__init__()
        self.message_id = message_id
        self.audio_file = audio_file
        self.error = error

class TTSPlaybackEvent(Message):
    """Event to control TTS playback"""
    
    def __init__(self, action: str, message_id: Optional[str] = None):
        super().__init__()
        self.action = action  # "play", "pause", "stop"
        self.message_id = message_id

#######################################################################################################################
#
# TTS Event Handler Mixin

class TTSEventHandler:
    """Mixin class for handling TTS events"""
    
    def __init__(self):
        self._tts_workers: Dict[str, Worker] = {}
        self._tts_config = None
        self._tts_service = None
    
    async def initialize_tts(self) -> None:
        """Initialize TTS service"""
        try:
            # Load TTS configuration
            self._tts_config = {
                "default_provider": get_cli_setting("app_tts", "default_provider", "openai"),
                "default_voice": get_cli_setting("app_tts", "default_voice", "alloy"),
                "default_model": get_cli_setting("app_tts", "default_model", "tts-1"),
                "default_format": get_cli_setting("app_tts", "default_format", "mp3"),
                "default_speed": get_cli_setting("app_tts", "default_speed", 1.0),
            }
            
            # Initialize TTS service
            app_config = {
                "app_tts": self._tts_config,
                # Add other config sections as needed
            }
            self._tts_service = await get_tts_service(app_config)
            
            logger.info("TTS service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            self._tts_service = None
    
    async def handle_tts_request(self, event: TTSRequestEvent) -> None:
        """Handle TTS generation request"""
        if not self._tts_service:
            logger.error("TTS service not initialized")
            await self.post_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="TTS service not available"
                )
            )
            return
        
        # Create worker for this TTS request
        worker_name = f"tts_{event.message_id or 'adhoc'}"
        
        # Cancel existing worker if any
        if worker_name in self._tts_workers:
            existing_worker = self._tts_workers[worker_name]
            if existing_worker.state == WorkerState.RUNNING:
                existing_worker.cancel()
        
        # Start new worker
        worker = self.run_worker(
            self._generate_tts,
            event.text,
            event.message_id,
            event.voice,
            name=worker_name,
            thread=True
        )
        
        self._tts_workers[worker_name] = worker
    
    async def _generate_tts(
        self,
        text: str,
        message_id: Optional[str],
        voice: Optional[str]
    ) -> None:
        """Generate TTS audio (runs in worker thread)"""
        try:
            # Prepare request
            request = OpenAISpeechRequest(
                model=self._tts_config["default_model"],
                input=text,
                voice=voice or self._tts_config["default_voice"],
                response_format=self._tts_config["default_format"],
                speed=self._tts_config["default_speed"]
            )
            
            # Determine backend based on model
            if request.model in ["tts-1", "tts-1-hd"]:
                internal_model_id = f"openai_official_{request.model}"
            elif request.model == "kokoro":
                internal_model_id = "local_kokoro_default_onnx"
            else:
                internal_model_id = request.model
            
            # Create temporary file for audio
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f".{request.response_format}",
                delete=False
            )
            temp_path = Path(temp_file.name)
            
            try:
                # Stream audio and write to file
                async for chunk in self._tts_service.generate_audio_stream(request, internal_model_id):
                    temp_file.write(chunk)
                    
                    # Post streaming event for real-time playback (optional)
                    if message_id:
                        await self.call_from_thread(
                            self.post_message,
                            TTSStreamingEvent(chunk, message_id, is_final=False)
                        )
                
                temp_file.close()
                
                # Post completion event
                await self.call_from_thread(
                    self.post_message,
                    TTSCompleteEvent(message_id=message_id or "adhoc", audio_file=temp_path)
                )
                
            except Exception as e:
                temp_file.close()
                temp_path.unlink(missing_ok=True)
                raise
                
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            await self.call_from_thread(
                self.post_message,
                TTSCompleteEvent(
                    message_id=message_id or "adhoc",
                    error=str(e)
                )
            )
    
    async def handle_tts_playback(self, event: TTSPlaybackEvent) -> None:
        """Handle TTS playback control"""
        # This would integrate with an audio player
        # For now, just log the action
        logger.info(f"TTS playback action: {event.action} for message {event.message_id}")
        
        # TODO: Implement actual audio playback control
        # This could use pygame, pyaudio, or system audio players
    
    def on_tts_request_event(self, event: TTSRequestEvent) -> None:
        """Handle TTS request event"""
        asyncio.create_task(self.handle_tts_request(event))
    
    def on_tts_playback_event(self, event: TTSPlaybackEvent) -> None:
        """Handle TTS playback event"""
        asyncio.create_task(self.handle_tts_playback(event))

#######################################################################################################################
#
# Helper Functions

def play_audio_file(file_path: Path) -> None:
    """Play an audio file using system default player"""
    import platform
    import subprocess
    
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", str(file_path)], check=True)
        elif system == "Linux":
            # Try different players
            for player in ["aplay", "paplay", "ffplay"]:
                try:
                    subprocess.run([player, str(file_path)], check=True)
                    break
                except FileNotFoundError:
                    continue
        elif system == "Windows":
            # Windows Media Player
            subprocess.run(["start", "", str(file_path)], shell=True, check=True)
        else:
            logger.warning(f"Unsupported platform for audio playback: {system}")
            
    except Exception as e:
        logger.error(f"Failed to play audio file: {e}")

#
# End of tts_events.py
#######################################################################################################################