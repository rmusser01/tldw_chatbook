# stts_events.py
# Description: Event handlers for S/TT/S (Speech/Text-to-Speech) functionality
#
# Imports
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger
import tempfile
#
# Third-party imports
from textual.message import Message
from textual.widgets import Button, TextArea, Select, Input, RichLog, Static, ProgressBar
#
# Local imports
from tldw_chatbook.TTS import get_tts_service, OpenAISpeechRequest
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSRequestEvent, TTSCompleteEvent, TTSPlaybackEvent
from tldw_chatbook.config import get_cli_setting
#
#######################################################################################################################
#
# Event Messages

class STTSPlaygroundGenerateEvent(Message):
    """Event when TTS generation is requested from playground"""
    
    def __init__(
        self,
        text: str,
        provider: str,
        voice: str,
        model: str,
        speed: float,
        format: str,
        extra_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.text = text
        self.provider = provider
        self.voice = voice
        self.model = model
        self.speed = speed
        self.format = format
        self.extra_params = extra_params or {}


class STTSSettingsSaveEvent(Message):
    """Event when TTS settings are saved"""
    
    def __init__(self, settings: Dict[str, Any]):
        super().__init__()
        self.settings = settings


class STTSAudioBookGenerateEvent(Message):
    """Event when audiobook generation is requested"""
    
    def __init__(
        self,
        content: str,
        chapters: list,
        narrator_voice: str,
        output_format: str,
        options: Dict[str, Any]
    ):
        super().__init__()
        self.content = content
        self.chapters = chapters
        self.narrator_voice = narrator_voice
        self.output_format = output_format
        self.options = options


#######################################################################################################################
#
# Event Handler Mixin

class STTSEventHandler:
    """Event handler for S/TT/S functionality"""
    
    def __init__(self, app=None):
        self.app = app  # Reference to the main app
        self._stts_service = None
        self._current_audio_file = None
        self._is_generating = False
    
    async def initialize_stts(self) -> None:
        """Initialize S/TT/S service"""
        try:
            from tldw_chatbook.config import load_cli_config_and_ensure_existence
            
            # Load the full config to get API keys
            full_config = load_cli_config_and_ensure_existence()
            
            # Get TTS service instance
            app_config = {
                "app_tts": {
                    "default_provider": get_cli_setting("app_tts", "default_provider", "openai"),
                    "default_voice": get_cli_setting("app_tts", "default_voice", "alloy"),
                    "default_model": get_cli_setting("app_tts", "default_model", "tts-1"),
                    "default_format": get_cli_setting("app_tts", "default_format", "mp3"),
                    "default_speed": get_cli_setting("app_tts", "default_speed", 1.0),
                }
            }
            
            # Add API settings sections if they exist
            if "api_settings.openai" in full_config:
                app_config["api_settings.openai"] = full_config["api_settings.openai"]
            if "openai_api" in full_config:
                app_config["openai_api"] = full_config["openai_api"]
            if "API" in full_config:
                app_config["API"] = full_config["API"]
                
            self._stts_service = await get_tts_service(app_config)
            logger.info("S/TT/S service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize S/TT/S service: {e}")
            self._stts_service = None
    
    async def handle_playground_generate(self, event: STTSPlaygroundGenerateEvent) -> None:
        """Handle TTS generation from playground"""
        if self._is_generating:
            self.app.notify("TTS generation already in progress", severity="warning")
            return
        
        if not self._stts_service:
            self.app.notify("TTS service not initialized", severity="error")
            return
        
        self._is_generating = True
        
        # Get playground widget first
        try:
            from tldw_chatbook.UI.STTS_Window import TTSPlaygroundWidget
            playground = self.app.query_one(TTSPlaygroundWidget)
            logger.debug(f"Found playground widget: {playground}")
        except Exception as e:
            logger.warning(f"Could not find TTSPlaygroundWidget: {e}")
            playground = None
        
        # Run the generation in a worker to avoid blocking UI
        if playground and hasattr(playground, 'run_worker'):
            # Use the widget's worker to keep UI responsive
            # Run async function in a worker (not in a thread)
            playground.run_worker(
                self._generate_tts_worker(event, playground),
                exclusive=True
            )
        else:
            # Fallback to direct async execution
            await self._generate_tts_worker(event, playground)
    
    async def _generate_tts_worker(self, event: STTSPlaygroundGenerateEvent, playground=None) -> None:
        """Worker function for TTS generation"""
        try:
            # Show progress container if available
            if playground:
                def show_progress():
                    try:
                        status_container = playground.query_one("#generation-status-container")
                        status_container.remove_class("hidden")
                        progress_bar = playground.query_one("#generation-progress")
                        progress_bar.update(total=100, progress=0)
                    except Exception as e:
                        logger.debug(f"Could not show progress container: {e}")
                
                if hasattr(playground, 'call_from_thread'):
                    playground.call_from_thread(show_progress)
                else:
                    show_progress()
            # Validate and clean format
            format_value = event.format
            if isinstance(format_value, tuple):
                format_value = format_value[0]
            elif not format_value or format_value == Select.BLANK or str(format_value) == "Select.BLANK":
                format_value = "mp3"
            
            # Additional validation for allowed formats
            valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
            if format_value not in valid_formats:
                logger.warning(f"Invalid format value '{format_value}', defaulting to mp3")
                format_value = "mp3"
            
            # Store the requested format for later conversion
            requested_format = format_value
            
            # For non-WAV formats, generate WAV first to avoid choppiness
            generation_format = "wav"
            logger.debug(f"TTS Request - requested format: {requested_format}, generation format: {generation_format}, provider: {event.provider}")
            
            # Create TTS request with generation format (WAV for quality)
            request = OpenAISpeechRequest(
                model=event.model,
                input=event.text,
                voice=event.voice,
                response_format=generation_format,
                speed=event.speed
            )
            
            # Map provider to internal model ID (handle case-insensitive)
            provider_lower = event.provider.lower() if event.provider else ""
            
            # Extract the actual provider key from display names
            if "kokoro" in provider_lower:
                provider_lower = "kokoro"
            elif "elevenlabs" in provider_lower:
                provider_lower = "elevenlabs"
            elif "openai" in provider_lower:
                provider_lower = "openai"
            elif "chatterbox" in provider_lower:
                provider_lower = "chatterbox"
            elif "alltalk" in provider_lower:
                provider_lower = "alltalk"
            elif "higgs" in provider_lower:
                provider_lower = "higgs"
            
            if provider_lower == "openai":
                # Also convert model to lowercase
                model_lower = event.model.lower().replace("-", "")  # Convert TTS-1 to tts1
                internal_model_id = f"openai_official_{model_lower}"
            elif provider_lower == "elevenlabs":
                internal_model_id = f"elevenlabs_{event.model}"
            elif provider_lower == "kokoro":
                # Check if ONNX or PyTorch should be used
                use_onnx = event.extra_params.get("use_onnx", True)
                if use_onnx:
                    internal_model_id = "local_kokoro_default_onnx"
                else:
                    internal_model_id = "local_kokoro_default_pytorch"
                # Handle Kokoro language code
                if event.extra_params.get("language"):
                    # Kokoro voices include language code prefix
                    lang_code = event.extra_params["language"]
                    # Update voice to include language code if not already present
                    if not event.voice.startswith(f"{lang_code}"):
                        # Voice might need language adjustment
                        pass  # The voice already includes the language prefix
            elif provider_lower == "chatterbox":
                internal_model_id = "local_chatterbox_default"
                # Pass voice and extra params to the request
                if event.voice.startswith("custom:"):
                    # Voice contains reference audio path
                    request.voice = event.voice
                elif event.voice.startswith("profile:"):
                    # Voice is a saved profile
                    request.voice = event.voice
                # Add extra params to request for Chatterbox
                if event.extra_params:
                    request.extra_params = event.extra_params
            elif provider_lower == "higgs":
                internal_model_id = "local_higgs_v2"
                # Handle voice cloning reference
                if event.voice.startswith("custom:"):
                    # Voice contains reference audio path
                    request.voice = event.voice
                elif event.voice.startswith("profile:"):
                    # Voice is a saved profile
                    request.voice = event.voice
                # Pass extra params for Higgs features
                if event.extra_params:
                    request.extra_params = event.extra_params
            elif provider_lower == "alltalk":
                internal_model_id = f"alltalk_{event.model}"
            else:
                # Default case - use the model as-is
                internal_model_id = event.model
            
            # Log to playground
            if playground:
                # Use call_from_thread if we're in a worker thread
                def update_log(msg):
                    log = playground.query_one("#tts-generation-log", RichLog)
                    log.write(msg)
                
                if hasattr(playground, 'call_from_thread'):
                    playground.call_from_thread(update_log, f"[bold yellow]Starting generation with {event.provider}...[/bold yellow]")
                    if provider_lower in ["higgs", "chatterbox", "kokoro"]:
                        playground.call_from_thread(update_log, "[dim]⚠️  First-time model loading may take 2-5 minutes...[/dim]")
                        playground.call_from_thread(update_log, "[dim]The model will be cached for faster subsequent generations.[/dim]")
                else:
                    update_log(f"[bold yellow]Starting generation with {event.provider}...[/bold yellow]")
                    if provider_lower in ["higgs", "chatterbox", "kokoro"]:
                        update_log("[dim]⚠️  First-time model loading may take 2-5 minutes...[/dim]")
                        update_log("[dim]The model will be cached for faster subsequent generations.[/dim]")
            
            # Define progress callback
            async def progress_callback(info: Dict[str, Any]) -> None:
                """Update UI with generation progress"""
                if playground:
                    def update_progress():
                        try:
                            # Update status text
                            status_text = playground.query_one("#generation-status-text", Static)
                            status_text.update(info.get("status", "Generating..."))
                            
                            # Update progress bar
                            progress_bar = playground.query_one("#generation-progress", ProgressBar)
                            progress = info.get("progress", 0) * 100
                            progress_bar.update(progress=progress)
                            
                            # Log additional details
                            log = playground.query_one("#tts-generation-log", RichLog)
                            if "metrics" in info:
                                metrics = info["metrics"]
                                if "audio_duration" in metrics:
                                    log.write(f"[dim]Generated {metrics['audio_duration']:.1f}s of audio[/dim]")
                                elif "current_chunk" in info:
                                    log.write(f"[dim]Processing chunk {info.get('current_chunk')}/{info.get('total_chunks', '?')}[/dim]")
                        except Exception as e:
                            logger.debug(f"Progress update error: {e}")
                    
                    if hasattr(playground, 'call_from_thread'):
                        playground.call_from_thread(update_progress)
                    else:
                        update_progress()
            
            # Set progress callback on the backend if supported
            try:
                backend = await self._stts_service.backend_manager.get_backend(internal_model_id)
                if backend and hasattr(backend, 'set_progress_callback'):
                    backend.set_progress_callback(progress_callback)
                    logger.debug(f"Set progress callback for {internal_model_id}")
            except Exception as e:
                logger.debug(f"Could not set progress callback: {e}")
            
            # Generate audio with provider-specific settings
            audio_data = b""
            chunk_count = 0
            total_size = 0
            start_time = asyncio.get_event_loop().time()
            last_status_time = start_time
            
            # Pass extra params to the service if needed
            if event.provider == "elevenlabs" and event.extra_params:
                # For ElevenLabs, we need to pass these settings to the backend
                # The TTS service should handle passing these to the appropriate backend
                request_dict = request.dict()
                request_dict.update(event.extra_params)
                # Note: This requires the backend to support these extra parameters
            
            # Create a task to show periodic status while waiting
            async def show_waiting_status():
                wait_count = 0
                while chunk_count == 0:  # While we haven't received any chunks yet
                    await asyncio.sleep(5)  # Check every 5 seconds
                    wait_count += 1
                    if playground and chunk_count == 0:  # Still waiting
                        elapsed = int(asyncio.get_event_loop().time() - start_time)
                        wait_msg = f"[yellow]⏳ Still loading model... ({elapsed}s elapsed)[/yellow]"
                        if wait_count > 12:  # After 1 minute
                            wait_msg += "\n[dim]This is taking longer than usual. Please be patient.[/dim]"
                        
                        def update_wait():
                            log = playground.query_one("#tts-generation-log", RichLog)
                            log.write(wait_msg)
                        
                        if hasattr(playground, 'call_from_thread'):
                            playground.call_from_thread(update_wait)
                        else:
                            update_wait()
            
            # Start the waiting status task
            status_task = asyncio.create_task(show_waiting_status())
            
            try:
                async for chunk in self._stts_service.generate_audio_stream(request, internal_model_id):
                    audio_data += chunk
                    chunk_count += 1
                    total_size = len(audio_data)
                    
                    # Cancel the waiting status task once we start receiving data
                    if chunk_count == 1:
                        status_task.cancel()
                
                # Update progress periodically (every 10 chunks)
                if chunk_count % 10 == 0 and playground:
                    progress_msg = f"[cyan]Streaming audio... {total_size / 1024:.1f} KB received[/cyan]"
                    if hasattr(playground, 'call_from_thread'):
                        playground.call_from_thread(update_log, progress_msg)
                    else:
                        update_log(progress_msg)
                
                # Also update for first chunk to show we're receiving data
                if chunk_count == 1 and playground:
                    first_msg = "[green]✓ Model loaded, receiving audio data...[/green]"
                    if hasattr(playground, 'call_from_thread'):
                        playground.call_from_thread(update_log, first_msg)
                    else:
                        update_log(first_msg)
            
            finally:
                # Make sure to cancel the status task
                status_task.cancel()
            
            # Save to temporary WAV file first
            import tempfile
            wav_file = None
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
                prefix="stts_playground_"
            ) as tmp_file:
                tmp_file.write(audio_data)
                wav_file = Path(tmp_file.name)
            
            # Convert to requested format if needed
            if requested_format != "wav":
                if playground:
                    def update_converting():
                        log = playground.query_one("#tts-generation-log", RichLog)
                        log.write(f"[yellow]Converting to {requested_format.upper()}...[/yellow]")
                    
                    if hasattr(playground, 'call_from_thread'):
                        playground.call_from_thread(update_converting)
                    else:
                        update_converting()
                
                # Convert audio format
                converted_file = await self._convert_audio_format(wav_file, requested_format)
                if converted_file:
                    # Delete the temporary WAV file
                    try:
                        wav_file.unlink()
                    except (FileNotFoundError, PermissionError) as e:
                        logger.warning(f"Failed to delete temporary WAV file: {e}")
                        pass
                    self._current_audio_file = converted_file
                else:
                    # Conversion failed, use WAV file
                    logger.warning(f"Failed to convert to {requested_format}, using WAV")
                    self._current_audio_file = wav_file
            else:
                self._current_audio_file = wav_file
            
            # Update UI
            if playground:
                def complete_generation():
                    log = playground.query_one("#tts-generation-log", RichLog)
                    log.write(f"[bold green]✓ Generation complete! File: {self._current_audio_file.name}[/bold green]")
                    log.write(f"Size: {len(audio_data) / 1024:.1f} KB")
                    
                    # Call the widget's completion method
                    if hasattr(playground, '_generation_complete'):
                        playground._generation_complete(True, self._current_audio_file)
                    else:
                        # Fallback to direct UI updates
                        playground.query_one("#audio-play-btn", Button).disabled = False
                        playground.query_one("#audio-export-btn", Button).disabled = False
                        playground.query_one("#audio-player-status").update(
                            f"Audio ready: {self._current_audio_file.name}"
                        )
                
                if hasattr(playground, 'call_from_thread'):
                    playground.call_from_thread(complete_generation)
                else:
                    complete_generation()
            
            self.app.notify("TTS generation complete!", severity="information")
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            if playground:
                def handle_error():
                    log = playground.query_one("#tts-generation-log", RichLog)
                    log.write(f"[bold red]✗ Generation failed: {e}[/bold red]")
                    
                    # Call the widget's completion method
                    if hasattr(playground, '_generation_complete'):
                        playground._generation_complete(False)
                
                if hasattr(playground, 'call_from_thread'):
                    playground.call_from_thread(handle_error)
                else:
                    handle_error()
                
            from rich.markup import escape
            self.app.notify(f"TTS generation failed: {escape(str(e))}", severity="error")
        finally:
            self._is_generating = False
            # Re-enable generate button is now handled in _generation_complete
            # But ensure it's re-enabled as a fallback
            if playground:
                try:
                    playground.query_one("#tts-generate-btn", Button).disabled = False
                    # Hide progress container
                    def hide_progress():
                        try:
                            status_container = playground.query_one("#generation-status-container")
                            status_container.add_class("hidden")
                        except Exception:
                            pass
                    
                    if hasattr(playground, 'call_from_thread'):
                        playground.call_from_thread(hide_progress)
                    else:
                        hide_progress()
                except (OSError, FileNotFoundError):
                    pass
    
    async def handle_settings_save(self, event: STTSSettingsSaveEvent) -> None:
        """Handle settings save"""
        try:
            from tldw_chatbook.config import save_setting_to_cli_config
            
            # Save each setting to the appropriate section
            settings_map = {
                # Default settings - save to both sections for compatibility
                "default_provider": [("app_tts", "default_provider"), ("tts_settings", "default_tts_provider")],
                "default_voice": [("app_tts", "default_voice"), ("tts_settings", "default_tts_voice")],
                "default_model": [("app_tts", "default_model"), ("tts_settings", "default_openai_tts_model")],
                "default_format": [("app_tts", "default_format"), ("tts_settings", "default_openai_tts_output_format")],
                "default_speed": [("app_tts", "default_speed"), ("tts_settings", "default_openai_tts_speed")],
                
                # OpenAI settings
                "openai_api_key": ("API", "openai_api_key"),
                
                # ElevenLabs settings
                "elevenlabs_api_key": ("API", "elevenlabs_api_key"),
                "elevenlabs_voice_stability": ("app_tts", "ELEVENLABS_VOICE_STABILITY"),
                "elevenlabs_similarity_boost": ("app_tts", "ELEVENLABS_SIMILARITY_BOOST"),
                "elevenlabs_style": ("app_tts", "ELEVENLABS_STYLE"),
                "elevenlabs_use_speaker_boost": ("app_tts", "ELEVENLABS_USE_SPEAKER_BOOST"),
                
                # Kokoro settings
                "kokoro_device": ("app_tts", "KOKORO_DEVICE"),
                "kokoro_use_onnx": ("app_tts", "KOKORO_USE_ONNX"),
                "kokoro_model_path": ("app_tts", "KOKORO_MODEL_PATH"),
                
                # Higgs settings
                "HIGGS_MODEL_PATH": ("HiggsSettings", "model_path"),
                "HIGGS_VOICE_SAMPLES_DIR": ("HiggsSettings", "voice_samples_dir"),
                "HIGGS_DEVICE": ("HiggsSettings", "device"),
                "HIGGS_ENABLE_FLASH_ATTN": ("HiggsSettings", "enable_flash_attn"),
                "HIGGS_DTYPE": ("HiggsSettings", "dtype"),
                "HIGGS_MAX_REFERENCE_DURATION": ("HiggsSettings", "max_reference_duration"),
                "HIGGS_DEFAULT_LANGUAGE": ("HiggsSettings", "default_language"),
                "HIGGS_ENABLE_VOICE_CLONING": ("HiggsSettings", "enable_voice_cloning"),
                "HIGGS_ENABLE_MULTI_SPEAKER": ("HiggsSettings", "enable_multi_speaker"),
                "HIGGS_SPEAKER_DELIMITER": ("HiggsSettings", "speaker_delimiter"),
                "HIGGS_TRACK_PERFORMANCE": ("HiggsSettings", "track_performance"),
                "HIGGS_MAX_NEW_TOKENS": ("HiggsSettings", "max_new_tokens"),
                "HIGGS_TEMPERATURE": ("HiggsSettings", "temperature"),
                "HIGGS_TOP_P": ("HiggsSettings", "top_p"),
                "HIGGS_TOP_K": ("HiggsSettings", "top_k"),
            }
            
            # Save each setting
            for key, value in event.settings.items():
                if key in settings_map:
                    mapping = settings_map[key]
                    # Handle both single tuple and list of tuples
                    if isinstance(mapping, list):
                        for section, setting_name in mapping:
                            save_setting_to_cli_config(section, setting_name, value)
                            logger.info(f"Saved {key} = {value} to [{section}].{setting_name}")
                    else:
                        section, setting_name = mapping
                        save_setting_to_cli_config(section, setting_name, value)
                        logger.info(f"Saved {key} = {value} to [{section}].{setting_name}")
            
            # Reinitialize TTS service with new settings
            await self.initialize_stts()
            
            self.app.notify("Settings saved successfully!", severity="information")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            self.app.notify(f"Failed to save settings: {e}", severity="error")
    
    async def _convert_audio_format(self, input_file: Path, output_format: str) -> Optional[Path]:
        """Convert audio file to a different format using ffmpeg"""
        try:
            # Create output file with requested format
            output_file = input_file.with_suffix(f".{output_format}")
            
            # Use ffmpeg for conversion
            cmd = [
                "ffmpeg",
                "-i", str(input_file),
                "-y",  # Overwrite output files
                "-loglevel", "error",  # Suppress verbose output
            ]
            
            # Add format-specific options
            if output_format == "mp3":
                # High quality MP3 encoding
                cmd.extend(["-codec:a", "libmp3lame", "-b:a", "192k"])
            elif output_format == "opus":
                cmd.extend(["-codec:a", "libopus", "-b:a", "128k"])
            elif output_format == "aac":
                cmd.extend(["-codec:a", "aac", "-b:a", "192k"])
            elif output_format == "flac":
                cmd.extend(["-codec:a", "flac"])
            
            cmd.append(str(output_file))
            
            # Run conversion asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for the process to complete
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Successfully converted audio to {output_format}")
                return output_file
            else:
                stderr_text = stderr.decode('utf-8') if stderr else "Unknown error"
                logger.error(f"ffmpeg conversion failed: {stderr_text}")
                return None
                
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg for audio format conversion.")
            return None
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None
    
    async def handle_audiobook_generate(self, event: STTSAudioBookGenerateEvent) -> None:
        """Handle audiobook generation"""
        try:
            from tldw_chatbook.TTS.audiobook_generator import (
                AudioBookGenerator, AudioBookRequest, AudioBookProgress
            )
            
            logger.info("AudioBook generation requested")
            
            # Initialize audiobook generator
            generator = AudioBookGenerator(self._stts_service)
            await generator.initialize()
            
            # Create audiobook request from event data
            audiobook_request = AudioBookRequest(
                content=event.content,
                title=event.options.get("title", "Untitled Book"),
                author=event.options.get("author", "Unknown"),
                narrator_voice=event.narrator_voice,
                provider=event.options.get("provider", "openai"),
                model=event.options.get("model", "tts-1"),
                output_format=event.output_format,
                chapter_detection=event.options.get("chapter_detection", True),
                multi_voice=event.options.get("multi_voice", False),
                character_voices=event.options.get("character_voices", {}),
                voice_settings=event.options.get("voice_settings", {}),
                background_music=event.options.get("background_music"),
                music_volume=event.options.get("music_volume", 0.1),
                chapter_pause_duration=event.options.get("chapter_pause_duration", 2.0),
                paragraph_pause_duration=event.options.get("paragraph_pause_duration", 0.5),
                sentence_pause_duration=event.options.get("sentence_pause_duration", 0.3),
                max_chunk_size=event.options.get("max_chunk_size", 4000),
                enable_ssml=event.options.get("enable_ssml", False),
                normalize_audio=event.options.get("normalize_audio", True),
                target_db=event.options.get("target_db", -20.0)
            )
            
            # Get cost estimate
            estimated_cost = generator.get_cost_estimate(audiobook_request)
            
            # Update UI with initial status
            if hasattr(self.app, "query_one"):
                try:
                    from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget
                    audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                    if audiobook_widget:
                        log = audiobook_widget.query_one("#audiobook-generation-log", RichLog)
                        log.write(f"[bold yellow]Starting audiobook generation...[/bold yellow]")
                        log.write(f"Estimated cost: ${estimated_cost:.2f}")
                except Exception:
                    pass  # UI element not found, continue without UI updates
            
            # Define progress callback
            async def progress_callback(progress: AudioBookProgress):
                """Update UI with generation progress"""
                if hasattr(self.app, "query_one"):
                    try:
                        from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget
                        audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                        if audiobook_widget:
                            log = audiobook_widget.query_one("#audiobook-generation-log", RichLog)
                            
                            # Update progress message
                            if progress.current_chapter:
                                log.write(f"[cyan]Processing: {progress.current_chapter}[/cyan]")
                            
                            # Update progress bar if available
                            if progress.total_chapters > 0:
                                percent_complete = (progress.completed_chapters / progress.total_chapters) * 100
                                log.write(f"Progress: {progress.completed_chapters}/{progress.total_chapters} chapters ({percent_complete:.1f}%)")
                            
                            # Show time estimates
                            if progress.estimated_completion:
                                remaining_time = progress.estimated_completion - datetime.now()
                                if remaining_time.total_seconds() > 0:
                                    minutes_remaining = int(remaining_time.total_seconds() / 60)
                                    log.write(f"Estimated time remaining: {minutes_remaining} minutes")
                            
                            # Show errors if any
                            for error in progress.errors:
                                log.write(f"[bold red]Error: {error}[/bold red]")
                    except Exception:
                        pass  # UI element not found, continue without UI updates
            
            # Generate the audiobook
            output_path = await generator.generate_audiobook(
                audiobook_request,
                progress_callback=progress_callback
            )
            
            # Update UI with completion
            if hasattr(self.app, "query_one"):
                try:
                    from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget
                    audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                    if audiobook_widget:
                        log = audiobook_widget.query_one("#audiobook-generation-log", RichLog)
                        log.write(f"[bold green]✓ AudioBook generation complete![/bold green]")
                        log.write(f"Output file: {output_path}")
                        log.write(f"Total duration: {generator.progress.actual_duration / 60:.1f} minutes")
                        
                        # Enable export button if available
                        export_btn = audiobook_widget.query_one("#audiobook-export-btn", Button)
                        if export_btn:
                            export_btn.disabled = False
                            # Store the output path for export
                            audiobook_widget.generated_audiobook_path = output_path
                except Exception:
                    pass  # UI element not found
            
            # Store the generated audiobook path for playback
            self._current_audio_file = output_path
            
            # Notify the UI widget
            if audiobook_widget and hasattr(audiobook_widget, 'audiobook_generation_complete'):
                audiobook_widget.audiobook_generation_complete(True, output_path)
            
            self.app.notify(f"AudioBook generated successfully: {output_path.name}", severity="information")
            
        except ImportError as e:
            logger.error(f"Failed to import audiobook generator: {e}")
            self.app.notify("AudioBook generation module not available", severity="error")
        except Exception as e:
            logger.error(f"AudioBook generation failed: {e}")
            self.app.notify(f"AudioBook generation failed: {e}", severity="error")
            
            # Notify the UI widget of failure
            try:
                from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget
                audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                if audiobook_widget and hasattr(audiobook_widget, 'audiobook_generation_complete'):
                    audiobook_widget.audiobook_generation_complete(False)
            except Exception:
                pass
    
    async def play_current_audio(self) -> None:
        """Play the current audio file"""
        if not self._current_audio_file or not self._current_audio_file.exists():
            self.app.notify("No audio file to play", severity="warning")
            return
        
        try:
            # Use the existing play_audio_file function
            from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import play_audio_file
            play_audio_file(self._current_audio_file)
            self.app.notify("Playing audio...", severity="information")
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            self.app.notify(f"Failed to play audio: {e}", severity="error")
    
    async def export_current_audio(self, target_path: Path) -> None:
        """Export the current audio file"""
        if not self._current_audio_file or not self._current_audio_file.exists():
            self.app.notify("No audio file to export", severity="warning")
            return
        
        try:
            import shutil
            shutil.copy2(self._current_audio_file, target_path)
            self.app.notify(f"Audio exported to {target_path}", severity="information")
        except Exception as e:
            logger.error(f"Failed to export audio: {e}")
            self.app.notify(f"Failed to export audio: {e}", severity="error")
    
    def on_stts_playground_generate_event(self, event: STTSPlaygroundGenerateEvent) -> None:
        """Handle playground generate event"""
        asyncio.create_task(self.handle_playground_generate(event))
    
    def on_stts_settings_save_event(self, event: STTSSettingsSaveEvent) -> None:
        """Handle settings save event"""
        asyncio.create_task(self.handle_settings_save(event))
    
    def on_stts_audiobook_generate_event(self, event: STTSAudioBookGenerateEvent) -> None:
        """Handle audiobook generate event"""
        asyncio.create_task(self.handle_audiobook_generate(event))

#
# End of stts_events.py
#######################################################################################################################