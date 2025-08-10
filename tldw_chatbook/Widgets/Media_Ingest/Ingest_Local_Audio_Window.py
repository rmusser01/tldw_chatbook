# tldw_chatbook/Widgets/Media_Ingest/AudioIngestWindowRedesigned.py
"""
Redesigned audio ingestion window using the video template as a base.
Implements similar functionality to video but without video-specific options.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional, AsyncIterator, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
import asyncio
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Static, Button, Input, Label, Select, Checkbox, TextArea,
    RadioSet, RadioButton
)
from textual import on

from .base_media_ingest_window import BaseMediaIngestWindow, MediaFormData, ProcessingStatus
from tldw_chatbook.config import get_media_ingestion_defaults
from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class AudioFormData(MediaFormData):
    """Audio-specific form data model."""
    # Audio processing options
    start_time: Optional[str] = Field(None, description="Start time (HH:MM:SS or seconds)")
    end_time: Optional[str] = Field(None, description="End time (HH:MM:SS or seconds)")
    
    # Transcription options
    transcription_provider: str = Field("faster-whisper", description="Transcription provider")
    transcription_model: str = Field("base", description="Transcription model")
    language: Optional[str] = Field(None, description="Audio language (auto-detect if None)")
    
    # Analysis options
    enable_analysis: bool = Field(False, description="Enable LLM analysis")
    analysis_provider: Optional[str] = Field(None, description="Analysis provider")
    analysis_model: Optional[str] = Field(None, description="Analysis model")
    
    # Chunking options
    chunk_method: str = Field("sentences", description="Chunking method")
    chunk_size: int = Field(1000, ge=100, le=10000, description="Chunk size in characters")
    overlap_size: int = Field(200, ge=0, le=1000, description="Overlap size in characters")


class AudioIngestWindowRedesigned(BaseMediaIngestWindow):
    """
    Redesigned audio ingestion window with proper architecture.
    
    Features:
    - Clean separation of concerns
    - Proper input visibility
    - Progressive disclosure (simple/advanced)
    - Real-time validation
    - Status updates during processing
    - Responsive layout
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, **kwargs)
        self.transcription_service = TranscriptionService()
        self._available_providers = []
        self._available_models = {}
        self._current_provider = "faster-whisper"
        
        # Load defaults
        self.audio_defaults = get_media_ingestion_defaults("audio")
        
        logger.debug("[Audio] AudioIngestWindowRedesigned initialized")
    
    def get_media_type(self) -> str:
        return "audio"
    
    def get_form_data_model(self) -> type[BaseModel]:
        return AudioFormData
    
    def get_file_filters(self) -> List[Tuple[str, str]]:
        """Get audio file filters for the file browser."""
        return [
            ("Audio Files", "*.mp3 *.wav *.flac *.aac *.ogg *.m4a *.wma *.opus"),
            ("MP3 Audio", "*.mp3"),
            ("WAV Audio", "*.wav"),
            ("FLAC Audio", "*.flac"),
            ("AAC Audio", "*.aac *.m4a"),
            ("OGG Audio", "*.ogg"),
            ("All Files", "*")
        ]
    
    def create_media_specific_options(self) -> ComposeResult:
        """Create audio-specific processing options."""
        
        # Audio Processing Options
        with Container(id="audio-processing-options", classes="options-section"):
            yield Static("Audio Processing Options", classes="section-header")
            
            # Time range options
            yield Label("Time Range (Optional):", classes="form-label")
            with Horizontal(classes="time-range-row"):
                with Vertical(classes="time-col"):
                    yield Label("Start Time:")
                    yield Input(
                        placeholder="HH:MM:SS or seconds",
                        id="start-time",
                        classes="form-input"
                    )
                with Vertical(classes="time-col"):
                    yield Label("End Time:")
                    yield Input(
                        placeholder="HH:MM:SS or seconds",
                        id="end-time", 
                        classes="form-input"
                    )
        
        # Transcription Options
        with Container(id="transcription-options", classes="options-section"):
            yield Static("Transcription Options", classes="section-header")
            yield from self.create_transcription_options()
        
        # Analysis Options (Advanced mode only)
        with Container(id="analysis-options", classes="options-section advanced-only"):
            yield Static("Analysis Options", classes="section-header") 
            yield from self.create_analysis_options()
        
        # Chunking Options (Advanced mode only)
        with Container(id="chunking-options", classes="options-section advanced-only"):
            yield Static("Chunking Options", classes="section-header")
            yield from self.create_chunking_options()
    
    def create_transcription_options(self) -> ComposeResult:
        """Create transcription configuration options."""
        # Get available providers
        self._available_providers = self.transcription_service.get_available_providers()
        
        if not self._available_providers:
            yield Static(
                "⚠️ No transcription providers available. Please install dependencies.",
                classes="warning-message"
            )
            return
        
        # Provider selection
        yield Label("Transcription Provider:", classes="form-label")
        provider_options = [(name, name) for name in self._available_providers]
        
        # Only set default value if we have providers available
        if self._available_providers:
            default_provider = self.audio_defaults.get("transcription_provider", "faster-whisper")
            if default_provider not in self._available_providers:
                default_provider = self._available_providers[0]
            
            yield Select(
                provider_options,
                value=default_provider,
                id="transcription-provider",
                classes="form-select"
            )
        else:
            yield Select(
                provider_options,
                id="transcription-provider",
                classes="form-select",
                allow_blank=True
            )
        
        # Model selection (populated after provider is selected)
        yield Label("Transcription Model:", classes="form-label")
        yield Select([], id="transcription-model", classes="form-select", allow_blank=True)
        
        # Language selection
        yield Label("Audio Language (auto-detect if blank):", classes="form-label")
        language_options = [
            ("auto", "Auto-detect"),
            ("en", "English"),
            ("es", "Spanish"),
            ("fr", "French"),
            ("de", "German"),
            ("it", "Italian"),
            ("pt", "Portuguese"),
            ("ru", "Russian"),
            ("ja", "Japanese"),
            ("ko", "Korean"),
            ("zh", "Chinese")
        ]
        yield Select(
            language_options,
            id="language",
            classes="form-select"
        )
    
    def create_analysis_options(self) -> ComposeResult:
        """Create analysis configuration options."""
        yield Checkbox(
            "Enable LLM Analysis",
            value=False,
            id="enable-analysis",
            classes="form-checkbox"
        )
        
        # Analysis provider (populated from app config)
        analysis_providers = list(self.app_instance.app_config.get("api_settings", {}).keys()) if self.app_instance.app_config else []
        if analysis_providers:
            yield Label("Analysis Provider:", classes="form-label")
            provider_options = [(name, name) for name in analysis_providers]
            yield Select(
                provider_options,
                value=analysis_providers[0],
                id="analysis-provider",
                classes="form-select"
            )
            
            # Model will be populated when provider is selected
            yield Label("Analysis Model:", classes="form-label")
            yield Select([], id="analysis-model", classes="form-select", allow_blank=True)
        else:
            yield Static(
                "No analysis providers configured in settings.",
                classes="info-message"
            )
    
    def create_chunking_options(self) -> ComposeResult:
        """Create chunking configuration options."""
        yield Label("Chunking Method:", classes="form-label")
        chunk_methods = [
            ("sentences", "By Sentences (recommended)"),
            ("words", "By Word Count"),
            ("characters", "By Character Count"),
            ("time", "By Time Segments")
        ]
        yield Select(
            chunk_methods,
            id="chunk-method",
            classes="form-select"
        )
        
        # Chunk size and overlap
        with Horizontal(classes="chunk-settings-row"):
            with Vertical(classes="chunk-col"):
                yield Label("Chunk Size:")
                yield Input(
                    value="1000",
                    placeholder="1000",
                    id="chunk-size",
                    classes="form-input"
                )
            with Vertical(classes="chunk-col"):
                yield Label("Overlap Size:")
                yield Input(
                    value="200",
                    placeholder="200",
                    id="overlap-size",
                    classes="form-input"
                )
    
    async def process_media(self, validated_data: dict) -> AsyncIterator[ProcessingStatus]:
        """Process audio files with status updates."""
        try:
            files = validated_data.get("files", [])
            urls = validated_data.get("urls", [])
            total_items = len(files) + len(urls)
            
            if total_items == 0:
                yield ProcessingStatus(
                    state="error",
                    error="No files or URLs provided",
                    message="No media to process"
                )
                return
            
            processed = 0
            
            # Process files
            for file_path in files:
                yield ProcessingStatus(
                    state="processing",
                    progress=processed / total_items,
                    current_file=file_path.name,
                    files_processed=processed,
                    total_files=total_items,
                    message=f"Processing audio file: {file_path.name}"
                )
                
                # Simulate processing (replace with actual processing logic)
                await asyncio.sleep(2)  # Simulate transcription time
                
                processed += 1
                
                yield ProcessingStatus(
                    state="processing",
                    progress=processed / total_items,
                    files_processed=processed,
                    total_files=total_items,
                    message=f"Completed: {file_path.name}"
                )
            
            # Process URLs
            for url in urls:
                yield ProcessingStatus(
                    state="processing",
                    progress=processed / total_items,
                    current_file=url,
                    files_processed=processed,
                    total_files=total_items,
                    message=f"Processing audio URL: {url[:50]}..."
                )
                
                # Simulate processing
                await asyncio.sleep(3)  # Simulate download + transcription time
                
                processed += 1
                
                yield ProcessingStatus(
                    state="processing",
                    progress=processed / total_items,
                    files_processed=processed,
                    total_files=total_items,
                    message=f"Completed URL processing"
                )
            
            # Final success status
            yield ProcessingStatus(
                state="complete",
                progress=1.0,
                files_processed=processed,
                total_files=total_items,
                message=f"Successfully processed {processed} audio items!"
            )
            
        except Exception as e:
            yield ProcessingStatus(
                state="error",
                error=str(e),
                message="Audio processing failed"
            )
            logger.error(f"[Audio] Processing error: {e}")
    
    # Event handlers for audio-specific controls
    
    @on(Select.Changed, "#transcription-provider")
    def handle_provider_change(self, event):
        """Update available models when provider changes."""
        provider = event.value
        self._current_provider = provider
        
        # Get models for this provider
        try:
            models = self.transcription_service.get_available_models(provider)
            self._available_models[provider] = models
            
            # Update model select
            model_select = self.query_one("#transcription-model")
            model_options = [(model, model) for model in models]
            model_select.set_options(model_options)
            
            if models:
                model_select.value = models[0]  # Select first model
                
        except Exception as e:
            logger.error(f"[Audio] Error loading models for {provider}: {e}")
    
    @on(Checkbox.Changed, "#enable-analysis")
    def handle_analysis_toggle(self, event):
        """Show/hide analysis options when checkbox is toggled."""
        enabled = event.value
        
        try:
            provider_select = self.query_one("#analysis-provider")
            model_select = self.query_one("#analysis-model")
            
            if enabled:
                provider_select.disabled = False
                model_select.disabled = False
                # Load models for current provider if available
                if provider_select.value:
                    self.update_analysis_models(provider_select.value)
            else:
                provider_select.disabled = True
                model_select.disabled = True
        except:
            pass  # Controls might not exist
    
    @on(Select.Changed, "#analysis-provider")
    def handle_analysis_provider_change(self, event):
        """Update analysis models when provider changes."""
        provider = event.value
        self.update_analysis_models(provider)
    
    def update_analysis_models(self, provider: str):
        """Update available analysis models for a provider."""
        try:
            # Get models from app config
            provider_config = self.app_instance.app_config.get("api_settings", {}).get(provider, {})
            models = provider_config.get("models", [])
            
            model_select = self.query_one("#analysis-model")
            model_options = [(model, model) for model in models]
            model_select.set_options(model_options)
            
            if models:
                model_select.value = models[0]
                
        except Exception as e:
            logger.error(f"[Audio] Error loading analysis models for {provider}: {e}")
    
    def validate_field(self, field_id: str, value: str) -> Optional[str]:
        """Validate audio-specific fields."""
        # Call parent validation first
        error = super().validate_field(field_id, value)
        if error:
            return error
        
        # Audio-specific validation
        if field_id in ["start-time", "end-time"]:
            if value and not self.validate_time_format(value):
                return "Time must be in format HH:MM:SS or seconds (e.g., 90)"
        elif field_id == "chunk-size":
            if value:
                try:
                    size = int(value)
                    if size < 100 or size > 10000:
                        return "Chunk size must be between 100 and 10000"
                except ValueError:
                    return "Chunk size must be a number"
        elif field_id == "overlap-size":
            if value:
                try:
                    size = int(value)
                    if size < 0 or size > 1000:
                        return "Overlap size must be between 0 and 1000"
                except ValueError:
                    return "Overlap size must be a number"
        
        return None
    
    def validate_time_format(self, time_str: str) -> bool:
        """Validate time format (HH:MM:SS or seconds)."""
        if not time_str:
            return True
        
        # Try seconds format first
        try:
            seconds = float(time_str)
            return seconds >= 0
        except ValueError:
            pass
        
        # Try HH:MM:SS format
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = parts
                int(hours)
                int(minutes) 
                float(seconds)
                return True
        except ValueError:
            pass
        
        return False
    
    def watch_simple_mode(self, simple: bool):
        """Handle simple/advanced mode changes for audio."""
        super().watch_simple_mode(simple)
        
        try:
            # Advanced-only sections
            analysis_options = self.query_one("#analysis-options")
            chunking_options = self.query_one("#chunking-options")
            
            if simple:
                # Simple mode: hide advanced sections
                analysis_options.add_class("hidden")
                chunking_options.add_class("hidden")
            else:
                # Advanced mode: show all sections
                analysis_options.remove_class("hidden")
                chunking_options.remove_class("hidden")
                
        except Exception as e:
            logger.error(f"[Audio] Error updating mode display: {e}")
    
    def on_mount(self):
        """Initialize audio-specific settings after mount."""
        # Load models for default transcription provider
        if self._available_providers:
            self.handle_provider_change_on_mount()
    
    def handle_provider_change_on_mount(self):
        """Load models for the default provider after mount."""
        try:
            provider_select = self.query_one("#transcription-provider")
            if provider_select.value:
                self.update_transcription_models(provider_select.value)
        except:
            pass
    
    def update_transcription_models(self, provider: str):
        """Update transcription models for a provider."""
        try:
            models = self.transcription_service.get_available_models(provider)
            self._available_models[provider] = models
            
            model_select = self.query_one("#transcription-model")
            model_options = [(model, model) for model in models]
            model_select.set_options(model_options)
            
            # Set default model
            default_model = self.audio_defaults.get("transcription_model", "base")
            if default_model in models:
                model_select.value = default_model
            elif models:
                model_select.value = models[0]
                
        except Exception as e:
            logger.error(f"[Audio] Error updating transcription models: {e}")
    
    # Extended CSS for audio-specific styling
    DEFAULT_CSS = BaseMediaIngestWindow.DEFAULT_CSS + """
    /* Audio-specific styling */
    .options-section {
        margin-bottom: 2;
        padding: 1;
        border: round $surface;
        background: $surface;
    }
    
    .section-header {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        border-bottom: solid $primary;
        padding-bottom: 1;
    }
    
    .time-range-row, .chunk-settings-row {
        layout: horizontal;
        margin-top: 1;
    }
    
    .time-col, .chunk-col {
        width: 1fr;
    }
    
    .form-checkbox {
        margin: 1 0;
    }
    
    .form-select {
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }
    
    .warning-message {
        color: $warning;
        background: $warning 10%;
        padding: 1;
        border: round $warning;
        margin-bottom: 1;
        text-style: italic;
    }
    
    .info-message {
        color: $text-muted;
        background: $surface;
        padding: 1;
        border: round $surface;
        margin-bottom: 1;
        text-style: italic;
    }
    
    .advanced-only.hidden {
        display: none;
    }
    """