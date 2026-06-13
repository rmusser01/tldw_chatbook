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
from dataclasses import dataclass
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


@dataclass
class TTSUsageRecord:
    """Single TTS usage record for local cost tracking."""
    provider: str
    model: str
    characters: int
    voice: str
    format: str
    estimated_cost: float
    created_at: datetime


class CostTracker:
    """Lightweight local TTS usage and cost tracker."""

    DEFAULT_COSTS = {
        ("openai", "tts-1"): {"cost_per_1k_chars": 0.015, "free_tier_chars": 0},
        ("openai", "tts-1-hd"): {"cost_per_1k_chars": 0.030, "free_tier_chars": 0},
        ("local", "*"): {"cost_per_1k_chars": 0.0, "free_tier_chars": 0},
    }

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else None
        self._costs: Dict[tuple[str, str], Dict[str, float]] = {
            key: value.copy() for key, value in self.DEFAULT_COSTS.items()
        }
        self._usage: list[TTSUsageRecord] = []

    def update_cost_info(
        self,
        provider: str,
        cost_per_1k_chars: float,
        free_tier_chars: int = 0,
        model: str = "*",
    ) -> None:
        self._costs[(provider.lower(), model.lower())] = {
            "cost_per_1k_chars": float(cost_per_1k_chars),
            "free_tier_chars": int(free_tier_chars),
        }

    def estimate_cost(self, provider: str, model: str, characters: int) -> float:
        provider_key = provider.lower()
        model_key = model.lower()
        cost_info = (
            self._costs.get((provider_key, model_key))
            or self._costs.get((provider_key, "*"))
            or {"cost_per_1k_chars": 0.0, "free_tier_chars": 0}
        )
        free_tier_chars = int(cost_info.get("free_tier_chars", 0))
        used_chars = self._monthly_usage_for_provider(provider_key)
        billable_chars = max(0, int(characters) - max(0, free_tier_chars - used_chars))
        return (billable_chars / 1000.0) * float(cost_info.get("cost_per_1k_chars", 0.0))

    def track_usage(
        self,
        provider: str,
        model: str,
        text: str,
        voice: str,
        format: str,
    ) -> TTSUsageRecord:
        characters = len(text)
        record = TTSUsageRecord(
            provider=provider,
            model=model,
            characters=characters,
            voice=voice,
            format=format,
            estimated_cost=self.estimate_cost(provider, model, characters),
            created_at=datetime.now(),
        )
        self._usage.append(record)
        return record

    def get_monthly_usage(self) -> int:
        return sum(record.characters for record in self._usage)

    def get_monthly_cost(self) -> float:
        return sum(record.estimated_cost for record in self._usage)

    def _monthly_usage_for_provider(self, provider: str) -> int:
        return sum(
            record.characters
            for record in self._usage
            if record.provider.lower() == provider
        )

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

    def _enforce_cooldown_limit(self) -> None:
        """Trim cooldown tracking to the maximum configured entries."""
        if len(self._request_cooldown) <= self.MAX_COOLDOWN_ENTRIES:
            return

        sorted_entries = sorted(self._request_cooldown.items(), key=lambda x: x[1])
        overflow = len(sorted_entries) - self.MAX_COOLDOWN_ENTRIES
        for key, _ in sorted_entries[:overflow]:
            del self._request_cooldown[key]

    async def _post_tts_message(self, message: Message) -> None:
        """Post a message through Textual app wiring or a direct test handler."""
        app = getattr(self, "app", None)
        if app is not None and hasattr(app, "post_message"):
            app.post_message(message)
            return

        post_message = getattr(self, "post_message", None)
        if callable(post_message):
            result = post_message(message)
            if asyncio.iscoroutine(result):
                await result
    
    async def handle_tts_request(self, event: TTSRequestEvent) -> None:
        """Handle TTS generation request"""
        current_time = asyncio.get_event_loop().time()
        if current_time - self._last_cooldown_cleanup > self.COOLDOWN_CLEANUP_INTERVAL:
            self._cleanup_cooldown_dict(current_time)
            self._last_cooldown_cleanup = current_time
        self._enforce_cooldown_limit()

        if not self._tts_service:
            logger.error("TTS service not initialized")
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="TTS service not available"
                )
            )
            return
        
        # Validate input text
        if not event.text:
            await self._post_tts_message(
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
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error=f"Text is too long for TTS. Maximum {MAX_TTS_LENGTH} characters allowed."
                )
            )
            return
        
        # Basic sanitization - remove excessive whitespace
        text = ' '.join(event.text.split())
        if len(text) < 1:
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="Text contains only whitespace"
                )
            )
            return
        
        # Check rate limiting for this message
        message_id = event.message_id or "adhoc"
        
        if message_id in self._request_cooldown:
            time_since_last = current_time - self._request_cooldown[message_id]
            if time_since_last < self.COOLDOWN_SECONDS:
                logger.warning(f"TTS request too soon for message {message_id}. Please wait {self.COOLDOWN_SECONDS - time_since_last:.1f}s")
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=message_id,
                        error=f"Please wait {self.COOLDOWN_SECONDS - time_since_last:.1f} seconds before requesting TTS again"
                    )
                )
                return
        
        # Update cooldown tracker
        self._request_cooldown[message_id] = current_time
        
        # Check if we need to evict old entries (LRU style)
        self._enforce_cooldown_limit()
        
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
            # Import metrics logger
            from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
            
            # Send initial progress
            if message_id:
                await self._post_tts_message(
                    TTSProgressEvent(
                        message_id=message_id,
                        progress=0.0,
                        status="Initializing TTS generation"
                    )
                )
            # Determine provider and format based on configured defaults.
            tts_config = self._tts_config or {}
            provider = tts_config.get("default_provider", "openai")
            default_model = tts_config.get("default_model", "tts-1")
            default_voice = tts_config.get("default_voice", "alloy")
            default_format = tts_config.get("default_format", "mp3")
            default_speed = tts_config.get("default_speed", 1.0)
            logger.info(f"TTS config: provider={provider}, default_format={default_format}")
            
            # Normalize provider name to lowercase for comparison
            provider_lower = provider.lower() if isinstance(provider, str) else provider
            
            # Set provider-specific defaults
            if provider_lower == "kokoro":
                model = "kokoro"
                format = "wav"  # Kokoro supports wav, pcm
            elif provider_lower == "openai":
                model = default_model
                format = default_format
            elif provider_lower == "elevenlabs":
                model = "elevenlabs"
                format = "mp3"  # ElevenLabs default
            elif provider_lower == "chatterbox":
                model = "chatterbox"
                format = "wav"
            elif provider_lower == "alltalk":
                model = "alltalk"
                format = "wav"
            else:
                # Fallback to OpenAI defaults
                model = default_model
                format = default_format
            
            # Ensure format is a string and valid
            format = str(format).lower().strip()
            valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
            if format not in valid_formats:
                logger.warning(f"Invalid format '{format}', defaulting to 'wav'")
                format = "wav"
            
            selected_voice = voice or default_voice
            logger.info(f"Creating TTS request: model={model}, format={format}, voice={selected_voice}")
            
            # Prepare request - ensure model and voice are lowercase for consistency
            request = OpenAISpeechRequest(
                model=model.lower() if isinstance(model, str) else model,
                input=text,
                voice=selected_voice.lower() if isinstance(selected_voice, str) else selected_voice,
                response_format=format,
                speed=default_speed
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
            
            # Estimate cost before generation when a tracker is attached.
            tracker = getattr(self, "cost_tracker", None)
            try:
                estimated_cost = tracker.estimate_cost(provider, request.model, len(text)) if tracker else 0
            except Exception as e:
                logger.debug(f"TTS cost estimate unavailable: {e}")
                estimated_cost = 0
            if message_id and estimated_cost > 0:
                await self._post_tts_message(
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
                        await self._post_tts_message(
                            TTSProgressEvent(
                                message_id=message_id,
                                progress=progress,
                                status=f"Generating audio... ({total_size / 1024:.1f} KB)",
                                estimated_time_remaining=remaining
                            )
                        )
                    
                    # Post streaming event for real-time playback (optional)
                    if message_id:
                        await self._post_tts_message(
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
                    await self._post_tts_message(
                        TTSProgressEvent(
                            message_id=message_id,
                            progress=1.0,
                            status="Audio generation complete"
                        )
                    )
                
                # Track TTS generation metrics
                generation_time = asyncio.get_event_loop().time() - start_time
                log_counter("tts_generation_total", labels={
                    "provider": provider,
                    "model": request.model,
                    "voice": request.voice,
                    "format": request.response_format
                })
                log_histogram("tts_generation_duration_seconds", generation_time, labels={
                    "provider": provider,
                    "model": request.model
                })
                log_histogram("tts_character_count", len(text), labels={
                    "provider": provider
                })
                
                # Post completion event
                await self._post_tts_message(
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
            await self._post_tts_message(
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
