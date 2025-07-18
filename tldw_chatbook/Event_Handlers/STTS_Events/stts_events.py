# stts_events.py
# Description: Event handlers for S/TT/S (Speech/Text-to-Speech) functionality
#
# Imports
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger

# Third-party imports
from textual.message import Message
from textual.widgets import Button, TextArea, Select, Input, RichLog

# Local imports
from tldw_chatbook.TTS import get_tts_service, OpenAISpeechRequest
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSRequestEvent, TTSCompleteEvent, TTSPlaybackEvent
from tldw_chatbook.config import get_cli_setting

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
        
        try:
            # Create TTS request
            request = OpenAISpeechRequest(
                model=event.model,
                input=event.text,
                voice=event.voice,
                response_format=event.format,
                speed=event.speed
            )
            
            # Map provider to internal model ID
            if event.provider == "openai":
                internal_model_id = f"openai_official_{event.model}"
            elif event.provider == "elevenlabs":
                internal_model_id = f"elevenlabs_{event.model}"
            elif event.provider == "kokoro":
                internal_model_id = "local_kokoro_default_onnx"
                # Handle Kokoro language code
                if event.extra_params.get("language"):
                    # Kokoro voices include language code prefix
                    lang_code = event.extra_params["language"]
                    # Update voice to include language code if not already present
                    if not event.voice.startswith(f"{lang_code}"):
                        # Voice might need language adjustment
                        pass  # The voice already includes the language prefix
            elif event.provider == "chatterbox":
                internal_model_id = "local_chatterbox_default"
                # Pass voice and extra params to the request
                if event.voice.startswith("custom:"):
                    # Voice contains reference audio path
                    request.voice = event.voice
                # Add extra params to request for Chatterbox
                if event.extra_params:
                    request.extra_params = event.extra_params
            else:
                internal_model_id = event.model
            
            # Log to playground
            playground = self.app.query_one("TTSPlaygroundWidget")
            if playground:
                log = playground.query_one("#tts-generation-log", RichLog)
                log.write(f"[bold yellow]Starting generation with {event.provider}...[/bold yellow]")
            
            # Generate audio with provider-specific settings
            audio_data = b""
            
            # Pass extra params to the service if needed
            if event.provider == "elevenlabs" and event.extra_params:
                # For ElevenLabs, we need to pass these settings to the backend
                # The TTS service should handle passing these to the appropriate backend
                request_dict = request.dict()
                request_dict.update(event.extra_params)
                # Note: This requires the backend to support these extra parameters
                
            async for chunk in self._stts_service.generate_audio_stream(request, internal_model_id):
                audio_data += chunk
            
            # Save to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{event.format}",
                prefix="stts_playground_"
            ) as tmp_file:
                tmp_file.write(audio_data)
                self._current_audio_file = Path(tmp_file.name)
            
            # Update UI
            if playground:
                log.write(f"[bold green]✓ Generation complete! File: {self._current_audio_file.name}[/bold green]")
                log.write(f"Size: {len(audio_data) / 1024:.1f} KB")
                
                # Enable playback controls
                playground.query_one("#audio-play-btn", Button).disabled = False
                playground.query_one("#audio-export-btn", Button).disabled = False
                playground.query_one("#audio-player-status").update(
                    f"Audio ready: {self._current_audio_file.name}"
                )
            
            self.app.notify("TTS generation complete!", severity="information")
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            if playground:
                log = playground.query_one("#tts-generation-log", RichLog)
                log.write(f"[bold red]✗ Generation failed: {e}[/bold red]")
            self.app.notify(f"TTS generation failed: {e}", severity="error")
        finally:
            self._is_generating = False
            # Re-enable generate button
            if playground:
                playground.query_one("#tts-generate-btn", Button).disabled = False
    
    async def handle_settings_save(self, event: STTSSettingsSaveEvent) -> None:
        """Handle settings save"""
        try:
            from tldw_chatbook.config import save_setting_to_cli_config
            
            # Save each setting to the appropriate section
            settings_map = {
                # Default settings
                "default_provider": ("app_tts", "default_provider"),
                "default_voice": ("app_tts", "default_voice"),
                "default_model": ("app_tts", "default_model"),
                "default_format": ("app_tts", "default_format"),
                "default_speed": ("app_tts", "default_speed"),
                
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
            }
            
            # Save each setting
            for key, value in event.settings.items():
                if key in settings_map:
                    section, setting_name = settings_map[key]
                    save_setting_to_cli_config(section, setting_name, value)
                    logger.info(f"Saved {key} = {value} to [{section}].{setting_name}")
            
            # Reinitialize TTS service with new settings
            await self.initialize_stts()
            
            self.app.notify("Settings saved successfully!", severity="information")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            self.app.notify(f"Failed to save settings: {e}", severity="error")
    
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