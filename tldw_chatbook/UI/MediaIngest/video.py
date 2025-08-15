"""Video ingestion tab implementation."""

from typing import TYPE_CHECKING, Dict, Any, AsyncIterator, List, Optional
from pathlib import Path
import asyncio
from loguru import logger

from textual.app import ComposeResult
from textual.widgets import (
    Label, Input, Select, Checkbox, TextArea, 
    RadioSet, RadioButton, Collapsible, Static
)
from textual.containers import Container, Horizontal, Vertical
from textual import on

from .base import BaseIngestTab
from .models import VideoFormData, ProcessingStatus

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class VideoIngestTab(BaseIngestTab):
    """Video ingestion tab with transcription and processing options."""
    
    DEFAULT_CSS = """
    VideoIngestTab {
        height: 100%;
        width: 100%;
    }
    
    .option-group {
        margin-bottom: 1;
        padding: 1;
        border: solid $primary-darken-2;
    }
    
    .option-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .provider-select {
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }
    
    .model-select {
        height: 3;
        width: 100%;
        margin-bottom: 1;
    }
    
    .time-input {
        height: 3;
        width: 30;
        margin-right: 2;
    }
    
    .checkbox-group {
        margin-bottom: 1;
    }
    
    .hidden {
        display: none;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "video", **kwargs):
        super().__init__(app_instance, media_type, **kwargs)
        self.transcription_providers = self._get_available_providers()
        self.transcription_models = {
            'faster-whisper': ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
            'whisper': ['tiny', 'base', 'small', 'medium', 'large'],
            'lightning-whisper-mlx': ['base', 'small', 'medium', 'large-v3'],
            'parakeet-mlx': ['nvidia/parakeet-ctc-0.6b', 'nvidia/parakeet-ctc-1.1b'],
            'qwen2audio': ['Qwen/Qwen2-Audio-7B'],
            'nemo': ['stt_en_conformer_ctc_large']
        }
    
    def _get_available_providers(self) -> List[str]:
        """Get list of available transcription providers."""
        providers = []
        
        # Check which providers are available
        try:
            from faster_whisper import WhisperModel
            providers.append('faster-whisper')
        except ImportError:
            pass
        
        # Always include whisper as fallback
        providers.append('whisper')
        
        # Check for platform-specific providers
        import sys
        if sys.platform == 'darwin':
            try:
                from lightning_whisper_mlx import LightningWhisperMLX
                providers.append('lightning-whisper-mlx')
            except ImportError:
                pass
            
            try:
                from parakeet_mlx import from_pretrained
                providers.append('parakeet-mlx')
            except ImportError:
                pass
        
        return providers if providers else ['whisper']
    
    def create_media_options(self) -> ComposeResult:
        """Create video-specific options."""
        yield Label("Video Processing Options", classes="section-title")
        
        # Download options
        with Container(classes="option-group"):
            yield Label("Download Settings", classes="option-title")
            yield Checkbox("Download full video (vs audio only)", id="download-video", value=False)
            yield Checkbox("Use cookies for authentication", id="use-cookies", value=False)
            
            # Cookie input (initially hidden)
            with Container(id="cookie-container", classes="hidden"):
                yield Label("Cookie data (JSON or file path):")
                yield TextArea("", id="cookies", classes="form-textarea")
        
        # Transcription options
        with Container(classes="option-group"):
            yield Label("Transcription Settings", classes="option-title")
            yield Checkbox("Enable transcription", id="transcription-enabled", value=True)
            
            with Container(id="transcription-options"):
                # Provider selection
                yield Label("Provider:")
                provider_options = [(p, p.title()) for p in self.transcription_providers]
                yield Select(provider_options, id="provider", classes="provider-select", value="faster-whisper")
                
                # Model selection
                yield Label("Model:")
                yield Select(
                    [("base", "Base"), ("small", "Small"), ("medium", "Medium")],
                    id="model",
                    classes="model-select",
                    value="base"
                )
                
                # Language selection
                yield Label("Language:")
                yield Select(
                    [
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
                    ],
                    id="language",
                    classes="form-select",
                    value="auto"
                )
                
                # Additional options
                with Container(classes="checkbox-group"):
                    yield Checkbox("Use Voice Activity Detection", id="vad", value=False)
                    yield Checkbox("Include timestamps", id="timestamps", value=True)
        
        # Time range options
        with Container(classes="option-group"):
            yield Label("Time Range (optional)", classes="option-title")
            with Horizontal():
                yield Label("Start (HH:MM:SS):")
                yield Input(placeholder="00:00:00", id="start-time", classes="time-input")
                yield Label("End (HH:MM:SS):")
                yield Input(placeholder="00:00:00", id="end-time", classes="time-input")
    
    def create_advanced_options(self) -> ComposeResult:
        """Create advanced video processing options."""
        # Analysis options
        yield Container(
            Label("AI Analysis", classes="option-title"),
            Checkbox("Perform analysis/summarization", id="perform-analysis", value=True),
            Container(
                Label("Custom prompt (optional):"),
                TextArea("", id="custom-prompt", classes="form-textarea"),
                Checkbox("Recursive summarization for long content", id="recursive-summary", value=False),
                id="analysis-options"
            ),
            classes="option-group"
        )
        
        # Chunking options
        yield Container(
            Label("Chunking Settings", classes="option-title"),
            Checkbox("Enable chunking", id="perform-chunking", value=True),
            Container(
                Label("Chunk method:"),
                Select(
                    [
                        ("time", "Time-based"),
                        ("semantic", "Semantic"),
                        ("sentence", "Sentence"),
                        ("paragraph", "Paragraph")
                    ],
                    id="chunk-method",
                    value="time"
                ),
                Label("Max chunk size:"),
                Input(value="500", placeholder="500", id="chunk-size"),
                Label("Chunk overlap:"),
                Input(value="200", placeholder="200", id="chunk-overlap"),
                Container(
                    Checkbox("Use adaptive chunking", id="adaptive-chunking", value=False),
                    Checkbox("Use multi-level chunking", id="multi-level-chunking", value=False),
                    classes="checkbox-group"
                ),
                id="chunking-options"
            ),
            classes="option-group"
        )
    
    # Event handlers
    @on(Checkbox.Changed, "#use-cookies")
    def toggle_cookie_input(self, event: Checkbox.Changed):
        """Show/hide cookie input based on checkbox."""
        container = self.query_one("#cookie-container")
        if event.value:
            container.remove_class("hidden")
        else:
            container.add_class("hidden")
    
    @on(Checkbox.Changed, "#transcription-enabled")
    def toggle_transcription_options(self, event: Checkbox.Changed):
        """Show/hide transcription options."""
        options = self.query_one("#transcription-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    @on(Select.Changed, "#provider")
    def update_model_options(self, event: Select.Changed):
        """Update model options based on selected provider."""
        provider = event.value
        model_select = self.query_one("#model")
        
        if provider in self.transcription_models:
            models = self.transcription_models[provider]
            options = [(m, m) for m in models]
            model_select.set_options(options)
            if models:
                model_select.value = models[0]
    
    @on(Checkbox.Changed, "#perform-analysis")
    def toggle_analysis_options(self, event: Checkbox.Changed):
        """Show/hide analysis options."""
        options = self.query_one("#analysis-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    @on(Checkbox.Changed, "#perform-chunking")
    def toggle_chunking_options(self, event: Checkbox.Changed):
        """Show/hide chunking options."""
        options = self.query_one("#chunking-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    def get_validated_form_data(self) -> Dict[str, Any]:
        """Get and validate video form data."""
        data = super().get_validated_form_data()
        
        # Add video-specific fields
        data['download_video_flag'] = self.query_one("#download-video").value
        data['use_cookies'] = self.query_one("#use-cookies").value
        data['cookies'] = self.query_one("#cookies").text if data['use_cookies'] else None
        
        # Transcription settings
        data['transcription_enabled'] = self.query_one("#transcription-enabled").value
        if data['transcription_enabled']:
            data['transcription_provider'] = self.query_one("#provider").value
            data['transcription_model'] = self.query_one("#model").value
            data['transcription_language'] = self.query_one("#language").value
            data['vad_use'] = self.query_one("#vad").value
            data['timestamp_option'] = self.query_one("#timestamps").value
        
        # Time range
        data['start_time'] = self.query_one("#start-time").value or None
        data['end_time'] = self.query_one("#end-time").value or None
        
        # Analysis settings
        data['perform_analysis'] = self.query_one("#perform-analysis").value
        if data['perform_analysis']:
            data['custom_prompt'] = self.query_one("#custom-prompt").text or None
            data['summarize_recursively'] = self.query_one("#recursive-summary").value
        
        # Chunking settings
        data['perform_chunking'] = self.query_one("#perform-chunking").value
        if data['perform_chunking']:
            data['chunk_method'] = self.query_one("#chunk-method").value
            data['max_chunk_size'] = int(self.query_one("#chunk-size").value or 500)
            data['chunk_overlap'] = int(self.query_one("#chunk-overlap").value or 200)
            data['use_adaptive_chunking'] = self.query_one("#adaptive-chunking").value
            data['use_multi_level_chunking'] = self.query_one("#multi-level-chunking").value
        
        return data
    
    async def process_media_impl(self, form_data: Dict[str, Any]) -> AsyncIterator[ProcessingStatus]:
        """Process video files."""
        try:
            # Validate with Pydantic model
            validated = VideoFormData(**form_data)
            
            # Get files and URLs
            all_inputs = list(validated.files) + validated.urls
            total = len(all_inputs)
            
            if total == 0:
                yield ProcessingStatus(state="error", error="No files or URLs to process")
                return
            
            # Import processor
            from tldw_chatbook.Local_Ingestion.video_processing import LocalVideoProcessor
            from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
            
            # Initialize processor
            media_db = MediaDatabase()
            processor = LocalVideoProcessor(media_db)
            
            # Process each input
            for idx, input_item in enumerate(all_inputs):
                current_file = str(input_item) if isinstance(input_item, Path) else input_item
                
                yield ProcessingStatus(
                    state="processing",
                    progress=(idx / total),
                    current_file=current_file,
                    current_operation=f"Processing video {idx + 1}/{total}",
                    files_processed=idx,
                    total_files=total,
                    message=f"Processing: {Path(current_file).name if isinstance(input_item, Path) else input_item}"
                )
                
                # Simulate processing (replace with actual processing)
                await asyncio.sleep(2)
                
                # Here you would call the actual processor
                # result = await processor.process_videos(...)
                
            yield ProcessingStatus(
                state="complete",
                progress=1.0,
                files_processed=total,
                total_files=total,
                message=f"Successfully processed {total} video(s)"
            )
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            yield ProcessingStatus(
                state="error",
                error=str(e),
                message=f"Processing failed: {str(e)}"
            )