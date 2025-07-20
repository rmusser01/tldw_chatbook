# tts_events.py
# Description: Event handlers for TTS functionality
#
# Imports
import asyncio
import os
import platform
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
from datetime import datetime
from loguru import logger

# Third-party imports
from textual.app import App
from textual.message import Message
from textual.worker import Worker, WorkerState

# Local imports
from tldw_chatbook.TTS import get_tts_service, OpenAISpeechRequest
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Utils.secure_temp_files import get_temp_manager, secure_delete_file

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


class TTSExportEvent(Message):
    """Event to export TTS audio with custom naming"""
    
    def __init__(self, message_id: str, output_path: Path, include_metadata: bool = True):
        super().__init__()
        self.message_id = message_id
        self.output_path = output_path
        self.include_metadata = include_metadata


class TTSProgressEvent(Message):
    """Event for TTS generation progress updates"""
    
    def __init__(self, message_id: str, progress: float, status: str, 
                 estimated_time_remaining: Optional[float] = None):
        super().__init__()
        self.message_id = message_id
        self.progress = progress  # 0.0 to 1.0
        self.status = status  # e.g., "Processing", "Generating", "Finalizing"
        self.estimated_time_remaining = estimated_time_remaining  # seconds

#######################################################################################################################
#
# TTS Event Handler Mixin

class TTSEventHandler:
    """
    Mixin class for handling TTS events.
    
    Note: Rate limiting is handled by the TTSService itself with a global
    semaphore of 4 concurrent requests. This handler only implements
    cooldown periods to prevent rapid repeated requests for the same message.
    """
    
    # Cooldown tracking to prevent rapid repeated requests
    _request_cooldown: Dict[str, float] = {}  # Track last request time per message
    COOLDOWN_SECONDS = 2.0  # Minimum time between requests for same message
    COOLDOWN_CLEANUP_INTERVAL = 300.0  # Clean up old entries every 5 minutes
    MAX_COOLDOWN_ENTRIES = 1000  # Maximum entries to keep in memory
    
    def __init__(self):
        self._tts_config = None
        self._tts_service = None
        self._temp_manager = get_temp_manager()
        self._audio_files: Dict[str, Path] = {}  # Track audio files by message_id
        self._audio_files_lock = asyncio.Lock()  # Lock for audio files dictionary
        self._active_tasks: set[asyncio.Task] = set()  # Track active async tasks
        self._active_tasks_lock = asyncio.Lock()  # Lock for active tasks set
        self._last_cooldown_cleanup = 0.0  # Track last cleanup time
    
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
    
    def _cleanup_cooldown_dict(self, current_time: float) -> None:
        """Clean up old entries from cooldown dictionary"""
        # Remove entries older than 5 minutes
        cutoff_time = current_time - 300.0
        keys_to_remove = [
            key for key, timestamp in self._request_cooldown.items()
            if timestamp < cutoff_time
        ]
        for key in keys_to_remove:
            del self._request_cooldown[key]
        
        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} old cooldown entries")
    
    async def handle_tts_request(self, event: TTSRequestEvent) -> None:
        """Handle TTS generation request"""
        if not self._tts_service:
            logger.error("TTS service not initialized")
            self.app.post_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="TTS service not available"
                )
            )
            return
        
        # Validate input text
        if not event.text:
            self.app.post_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="No text provided for TTS generation"
                )
            )
            return
        
        # Check text length limits
        MAX_TTS_LENGTH = 5000  # Maximum characters for TTS
        if len(event.text) > MAX_TTS_LENGTH:
            logger.warning(f"TTS text too long: {len(event.text)} characters")
            self.app.post_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error=f"Text is too long for TTS. Maximum {MAX_TTS_LENGTH} characters allowed."
                )
            )
            return
        
        # Basic sanitization - remove excessive whitespace
        text = ' '.join(event.text.split())
        if len(text) < 1:
            self.app.post_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="Text contains only whitespace"
                )
            )
            return
        
        # Check rate limiting for this message
        message_id = event.message_id or "adhoc"
        current_time = asyncio.get_event_loop().time()
        
        # Clean up old cooldown entries if needed
        if current_time - self._last_cooldown_cleanup > self.COOLDOWN_CLEANUP_INTERVAL:
            self._cleanup_cooldown_dict(current_time)
            self._last_cooldown_cleanup = current_time
        
        if message_id in self._request_cooldown:
            time_since_last = current_time - self._request_cooldown[message_id]
            if time_since_last < self.COOLDOWN_SECONDS:
                logger.warning(f"TTS request too soon for message {message_id}. Please wait {self.COOLDOWN_SECONDS - time_since_last:.1f}s")
                self.app.post_message(
                    TTSCompleteEvent(
                        message_id=message_id,
                        error=f"Please wait {self.COOLDOWN_SECONDS - time_since_last:.1f} seconds before requesting TTS again"
                    )
                )
                return
        
        # Update cooldown tracker
        self._request_cooldown[message_id] = current_time
        
        # Check if we need to evict old entries (LRU style)
        if len(self._request_cooldown) > self.MAX_COOLDOWN_ENTRIES:
            # Remove oldest entries
            sorted_entries = sorted(self._request_cooldown.items(), key=lambda x: x[1])
            for key, _ in sorted_entries[:len(sorted_entries) // 2]:
                del self._request_cooldown[key]
        
        # Start TTS generation task
        task = asyncio.create_task(
            self._generate_tts_with_rate_limit(
                text,  # Use sanitized text
                event.message_id,
                event.voice
            )
        )
        # Track the task
        asyncio.create_task(self._add_active_task(task))
    
    async def _generate_tts_with_rate_limit(
        self,
        text: str,
        message_id: Optional[str],
        voice: Optional[str]
    ) -> None:
        """Generate TTS audio (rate limiting handled by TTSService)"""
        try:
            await self._generate_tts(text, message_id, voice)
        except asyncio.CancelledError:
            logger.info(f"TTS generation cancelled for message {message_id}")
            raise
    
    async def _generate_tts(
        self,
        text: str,
        message_id: Optional[str],
        voice: Optional[str]
    ) -> None:
        """Generate TTS audio"""
        try:
            # Import cost tracker
            from tldw_chatbook.TTS.cost_tracker import get_cost_tracker
            cost_tracker = get_cost_tracker()
            
            # Send initial progress
            if message_id:
                self.app.post_message(
                    TTSProgressEvent(
                        message_id=message_id,
                        progress=0.0,
                        status="Initializing TTS generation"
                    )
                )
            # Determine provider and format based on default_provider
            provider = self._tts_config["default_provider"]
            logger.info(f"TTS config: provider={provider}, default_format={self._tts_config.get('default_format', 'N/A')}")
            
            # Set provider-specific defaults
            if provider == "kokoro":
                model = "kokoro"
                format = "wav"  # Kokoro supports wav, pcm
            elif provider == "openai":
                model = self._tts_config["default_model"]
                format = self._tts_config["default_format"]
            elif provider == "elevenlabs":
                model = "elevenlabs"
                format = "mp3"  # ElevenLabs default
            elif provider == "chatterbox":
                model = "chatterbox"
                format = "wav"
            elif provider == "alltalk":
                model = "alltalk"
                format = "wav"
            else:
                # Fallback to OpenAI defaults
                model = self._tts_config["default_model"]
                format = self._tts_config["default_format"]
            
            # Ensure format is a string and valid
            format = str(format).lower().strip()
            valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
            if format not in valid_formats:
                logger.warning(f"Invalid format '{format}', defaulting to 'wav'")
                format = "wav"
            
            logger.info(f"Creating TTS request: model={model}, format={format}, voice={voice or self._tts_config['default_voice']}")
            
            # Prepare request - ensure model and voice are lowercase for consistency
            request = OpenAISpeechRequest(
                model=model.lower() if isinstance(model, str) else model,
                input=text,
                voice=(voice or self._tts_config["default_voice"]).lower() if isinstance(voice or self._tts_config["default_voice"], str) else (voice or self._tts_config["default_voice"]),
                response_format=format,
                speed=self._tts_config["default_speed"]
            )
            
            # Determine backend based on model
            if request.model in ["tts-1", "tts-1-hd"]:
                internal_model_id = f"openai_official_{request.model}"
                provider = "openai"
            elif request.model == "kokoro":
                internal_model_id = "local_kokoro_default_onnx"
                provider = "local"
            elif request.model == "alltalk":
                internal_model_id = "alltalk_default"
                provider = "local"
            elif request.model.startswith("elevenlabs"):
                internal_model_id = f"elevenlabs_{request.model}"
                provider = "elevenlabs"
            elif request.model == "chatterbox":
                internal_model_id = "local_chatterbox_default"
                provider = "local"
            else:
                internal_model_id = request.model
                provider = "unknown"
            
            # Estimate cost before generation
            estimated_cost = cost_tracker.estimate_cost(provider, request.model, len(text))
            if message_id and estimated_cost > 0:
                self.app.post_message(
                    TTSProgressEvent(
                        message_id=message_id,
                        progress=0.1,
                        status=f"Estimated cost: ${estimated_cost:.4f}"
                    )
                )
            
            # Collect audio chunks first
            audio_chunks = []
            total_size = 0
            chunk_count = 0
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Stream audio and collect chunks
                async for chunk in self._tts_service.generate_audio_stream(request, internal_model_id):
                    audio_chunks.append(chunk)
                    total_size += len(chunk)
                    chunk_count += 1
                    
                    # Calculate progress based on typical chunk patterns
                    # Most providers send chunks progressively
                    progress = min(0.9, 0.1 + (chunk_count * 0.05))
                    
                    # Estimate time remaining based on current rate
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > 0 and progress > 0.1:
                        estimated_total = elapsed / progress
                        remaining = estimated_total - elapsed
                    else:
                        remaining = None
                    
                    # Post progress update every 5 chunks
                    if message_id and chunk_count % 5 == 0:
                        self.app.post_message(
                            TTSProgressEvent(
                                message_id=message_id,
                                progress=progress,
                                status=f"Generating audio... ({total_size / 1024:.1f} KB)",
                                estimated_time_remaining=remaining
                            )
                        )
                    
                    # Post streaming event for real-time playback (optional)
                    if message_id:
                        self.app.post_message(
                            TTSStreamingEvent(chunk, message_id, is_final=False)
                        )
                
                # Create secure temporary file with all audio data
                audio_data = b''.join(audio_chunks)
                temp_path = self._temp_manager.create_temp_file(
                    content=audio_data,
                    suffix=f".{request.response_format}",
                    prefix="tts_audio_"
                )
                
                # Track the audio file with lock
                if message_id:
                    async with self._audio_files_lock:
                        self._audio_files[message_id] = Path(temp_path)
                
                # Final progress update
                if message_id:
                    self.app.post_message(
                        TTSProgressEvent(
                            message_id=message_id,
                            progress=1.0,
                            status="Audio generation complete"
                        )
                    )
                
                # Track usage for cost tracking
                generation_time = asyncio.get_event_loop().time() - start_time
                cost_tracker.track_usage(
                    provider=provider,
                    model=request.model,
                    text=text,
                    voice=request.voice,
                    format=request.response_format,
                    duration_seconds=generation_time
                )
                
                # Post completion event
                self.app.post_message(
                    TTSCompleteEvent(message_id=message_id or "adhoc", audio_file=Path(temp_path))
                )
                
            except Exception as e:
                # Clean up any partial file if created (with lock)
                if message_id:
                    async with self._audio_files_lock:
                        if message_id in self._audio_files:
                            secure_delete_file(self._audio_files[message_id])
                            del self._audio_files[message_id]
                raise
                
        except Exception as e:
            from rich.markup import escape
            logger.error(f"TTS generation failed: {e}")
            if hasattr(self, 'app') and self.app:
                self.app.post_message(
                    TTSCompleteEvent(
                        message_id=message_id or "adhoc",
                        error=escape(str(e))
                    )
                )
    
    async def handle_tts_playback(self, event: TTSPlaybackEvent) -> None:
        """Handle TTS playback control"""
        logger.info(f"TTS playback action: {event.action} for message {event.message_id}")
        
        if event.action == "play" and event.message_id:
            # Get audio file with lock
            async with self._audio_files_lock:
                audio_file = self._audio_files.get(event.message_id)
            
            if audio_file and audio_file.exists():
                # Play the audio file
                play_audio_file(audio_file)
                # Schedule cleanup after playback
                asyncio.create_task(self._cleanup_audio_file(event.message_id, delay=5.0))
            else:
                logger.warning(f"Audio file not found for message {event.message_id}")
        
        elif event.action == "pause" and event.message_id:
            # The audio player doesn't support pause, so we stop playback
            # but keep the audio file for resuming
            logger.info(f"Pausing audio for message {event.message_id}")
            # This will stop any playing audio but won't delete the file
            
        elif event.action == "stop" and event.message_id:
            # Clean up immediately if stopped
            await self._cleanup_audio_file(event.message_id)
    
    async def handle_tts_export(self, event: TTSExportEvent) -> None:
        """Handle TTS audio export"""
        import shutil
        import json
        
        # Get audio file
        async with self._audio_files_lock:
            source_file = self._audio_files.get(event.message_id)
        
        if not source_file or not source_file.exists():
            logger.error(f"No audio file found for message {event.message_id}")
            self.notify("No audio file found to export", severity="error")
            return
        
        try:
            # Validate output path
            output_path = event.output_path
            if not output_path.suffix:
                # Add extension from source file
                output_path = output_path.with_suffix(source_file.suffix)
            
            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy audio file
            shutil.copy2(source_file, output_path)
            logger.info(f"Exported audio to {output_path}")
            
            # Add metadata if requested
            if event.include_metadata:
                metadata = {
                    "message_id": event.message_id,
                    "export_time": datetime.now().isoformat(),
                    "format": source_file.suffix[1:],  # Remove dot
                    "source": "tldw_chatbook_tts"
                }
                
                # Save metadata as JSON sidecar file
                metadata_path = output_path.with_suffix(output_path.suffix + ".json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Saved metadata to {metadata_path}")
            
            self.notify(f"Audio exported to {output_path.name}", severity="success")
            
        except Exception as e:
            logger.error(f"Failed to export audio: {e}")
            self.notify(f"Failed to export audio: {str(e)}", severity="error")
    
    async def _cleanup_audio_file(self, message_id: str, delay: float = 0) -> None:
        """Clean up audio file after playback"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        async with self._audio_files_lock:
            if message_id in self._audio_files:
                audio_file = self._audio_files[message_id]
                if secure_delete_file(audio_file):
                    logger.debug(f"Cleaned up audio file for message {message_id}")
                del self._audio_files[message_id]
    
    def on_tts_request_event(self, event: TTSRequestEvent) -> None:
        """Handle TTS request event"""
        task = asyncio.create_task(self.handle_tts_request(event))
        # Use create_task to add task safely
        asyncio.create_task(self._add_active_task(task))
    
    def on_tts_playback_event(self, event: TTSPlaybackEvent) -> None:
        """Handle TTS playback event"""
        task = asyncio.create_task(self.handle_tts_playback(event))
        # Use create_task to add task safely
        asyncio.create_task(self._add_active_task(task))
    
    def on_tts_export_event(self, event: TTSExportEvent) -> None:
        """Handle TTS export event"""
        task = asyncio.create_task(self.handle_tts_export(event))
        asyncio.create_task(self._add_active_task(task))
    
    async def _add_active_task(self, task: asyncio.Task) -> None:
        """Add task to active tasks set with lock"""
        async with self._active_tasks_lock:
            self._active_tasks.add(task)
            task.add_done_callback(lambda t: asyncio.create_task(self._remove_active_task(t)))
    
    async def _remove_active_task(self, task: asyncio.Task) -> None:
        """Remove task from active tasks set with lock"""
        async with self._active_tasks_lock:
            self._active_tasks.discard(task)
    
    async def cleanup_tts_resources(self) -> None:
        """Clean up all TTS resources"""
        # Cancel all active tasks with lock
        async with self._active_tasks_lock:
            tasks_to_cancel = list(self._active_tasks)
        
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # Clean up audio files with lock
        async with self._audio_files_lock:
            files_to_clean = list(self._audio_files.items())
            self._audio_files.clear()
        
        for message_id, audio_file in files_to_clean:
            secure_delete_file(audio_file)
            logger.debug(f"Cleaned up audio file for message {message_id}")
        
        # Clear active tasks with lock
        async with self._active_tasks_lock:
            self._active_tasks.clear()

#######################################################################################################################
#
# Helper Functions

def play_audio_file(file_path: Path) -> None:
    """Play an audio file using system default player"""
    # Use the centralized audio player from audio_player module
    from tldw_chatbook.TTS.audio_player import play_audio_file as play_audio
    
    # Delegate to the secure implementation
    success = play_audio(file_path)
    if not success:
        logger.error(f"Failed to play audio file: {file_path}")

#
# End of tts_events.py
#######################################################################################################################