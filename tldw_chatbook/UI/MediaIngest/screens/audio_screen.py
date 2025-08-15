"""Audio ingestion screen implementation."""

from typing import TYPE_CHECKING, Dict, Any, AsyncIterator
from pathlib import Path
import asyncio
from loguru import logger

from textual.app import ComposeResult
from textual.widgets import (
    Label, Input, Select, Checkbox, TextArea
)
from textual.containers import Container
from textual import on

from .base_screen import BaseMediaIngestScreen
from ..models import AudioFormData, ProcessingStatus

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class AudioIngestScreen(BaseMediaIngestScreen):
    """Screen for audio ingestion with transcription options."""
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "audio", **kwargs):
        super().__init__(app_instance, media_type, **kwargs)
        self.transcription_providers = ['faster-whisper', 'whisper', 'lightning-whisper-mlx']
    
    def create_media_options(self) -> ComposeResult:
        """Create audio-specific options."""
        yield Label("Audio Processing Options", classes="section-title")
        
        # Transcription options
        with Container(classes="option-group"):
            yield Label("Transcription Settings", classes="option-title")
            yield Checkbox("Enable transcription", id="transcription-enabled", value=True)
            
            with Container(id="transcription-options"):
                yield Label("Provider:")
                yield Select(
                    [(p, p.title()) for p in self.transcription_providers],
                    id="provider",
                    classes="form-select",
                    value="faster-whisper"
                )
                
                yield Label("Model:")
                yield Select(
                    [("base", "Base"), ("small", "Small"), ("medium", "Medium")],
                    id="model",
                    classes="form-select",
                    value="base"
                )
                
                yield Label("Language:")
                yield Select(
                    [
                        ("auto", "Auto-detect"),
                        ("en", "English"),
                        ("es", "Spanish"),
                        ("fr", "French"),
                        ("de", "German"),
                        ("zh", "Chinese"),
                        ("ja", "Japanese")
                    ],
                    id="language",
                    classes="form-select",
                    value="auto"
                )
                
                yield Checkbox("Include timestamps", id="timestamps", value=True)
                yield Checkbox("Diarization (speaker detection)", id="diarization", value=False)
        
        # Audio processing
        with Container(classes="option-group"):
            yield Label("Audio Processing", classes="option-title")
            yield Checkbox("Enhance audio quality", id="enhance-audio", value=False)
            yield Checkbox("Normalize volume", id="normalize-volume", value=True)
            yield Checkbox("Remove silence", id="remove-silence", value=False)
            yield Checkbox("Noise reduction", id="noise-reduction", value=False)
    
    def create_advanced_options(self) -> ComposeResult:
        """Create advanced audio processing options."""
        yield Container(
            Label("Advanced Settings", classes="option-title"),
            Checkbox("Perform analysis/summarization", id="perform-analysis", value=True),
            Checkbox("Enable chunking", id="perform-chunking", value=True),
            Container(
                Label("Chunk size (seconds):"),
                Input(value="300", id="chunk-size", classes="form-input"),
                id="chunking-options"
            ),
            classes="option-group"
        )
        
        yield Container(
            Label("Output Options", classes="option-title"),
            Checkbox("Save transcript as SRT", id="save-srt", value=False),
            Checkbox("Save transcript as VTT", id="save-vtt", value=False),
            Checkbox("Save transcript as TXT", id="save-txt", value=True),
            Checkbox("Generate audio summary", id="generate-summary", value=True),
            classes="option-group"
        )
    
    @on(Checkbox.Changed, "#transcription-enabled")
    def toggle_transcription_options(self, event: Checkbox.Changed) -> None:
        """Show/hide transcription options."""
        options = self.query_one("#transcription-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    @on(Checkbox.Changed, "#perform-chunking")
    def toggle_chunking_options(self, event: Checkbox.Changed) -> None:
        """Show/hide chunking options."""
        options = self.query_one("#chunking-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    def get_validated_form_data(self) -> Dict[str, Any]:
        """Get and validate audio form data."""
        data = super().get_validated_form_data()
        
        # Transcription settings
        data['transcription_enabled'] = self.query_one("#transcription-enabled").value
        if data['transcription_enabled']:
            data['transcription_provider'] = self.query_one("#provider").value
            data['transcription_model'] = self.query_one("#model").value
            data['transcription_language'] = self.query_one("#language").value
            data['timestamp_option'] = self.query_one("#timestamps").value
            data['diarization'] = self.query_one("#diarization").value
        
        # Audio processing
        data['enhance_audio'] = self.query_one("#enhance-audio").value
        data['normalize_volume'] = self.query_one("#normalize-volume").value
        data['remove_silence'] = self.query_one("#remove-silence").value
        data['noise_reduction'] = self.query_one("#noise-reduction").value
        
        # Advanced options
        data['perform_analysis'] = self.query_one("#perform-analysis").value
        data['perform_chunking'] = self.query_one("#perform-chunking").value
        if data['perform_chunking']:
            data['chunk_size'] = int(self.query_one("#chunk-size").value or 300)
        
        # Output options
        data['save_srt'] = self.query_one("#save-srt").value
        data['save_vtt'] = self.query_one("#save-vtt").value
        data['save_txt'] = self.query_one("#save-txt").value
        data['generate_summary'] = self.query_one("#generate-summary").value
        
        return data
    
    async def process_media_impl(self, form_data: Dict[str, Any]) -> AsyncIterator[ProcessingStatus]:
        """Process audio files."""
        try:
            validated = AudioFormData(**form_data)
            all_inputs = list(validated.files) + validated.urls
            total = len(all_inputs)
            
            if total == 0:
                yield ProcessingStatus(
                    state="error",
                    error="No files or URLs to process",
                    message="Please provide at least one audio file or URL"
                )
                return
            
            for idx, input_item in enumerate(all_inputs):
                file_name = Path(input_item).name if isinstance(input_item, Path) else input_item
                
                yield ProcessingStatus(
                    state="processing",
                    progress=(idx / total),
                    current_file=str(input_item),
                    current_operation="Processing audio",
                    files_processed=idx,
                    total_files=total,
                    message=f"Processing: {file_name}"
                )
                
                # Simulate audio processing
                if validated.enhance_audio or validated.noise_reduction:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.2) / total,
                        current_file=str(input_item),
                        current_operation="Enhancing audio",
                        files_processed=idx,
                        total_files=total,
                        message=f"Enhancing: {file_name}"
                    )
                    await asyncio.sleep(0.5)
                
                if validated.transcription_enabled:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.5) / total,
                        current_file=str(input_item),
                        current_operation="Transcribing",
                        files_processed=idx,
                        total_files=total,
                        message=f"Transcribing: {file_name}"
                    )
                    await asyncio.sleep(1)
                
                if validated.perform_analysis:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.8) / total,
                        current_file=str(input_item),
                        current_operation="Analyzing",
                        files_processed=idx,
                        total_files=total,
                        message=f"Analyzing: {file_name}"
                    )
                    await asyncio.sleep(0.5)
            
            yield ProcessingStatus(
                state="complete",
                progress=1.0,
                files_processed=total,
                total_files=total,
                message=f"Successfully processed {total} audio file(s)"
            )
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            yield ProcessingStatus(
                state="error",
                error=str(e),
                message=f"Processing failed: {str(e)}"
            )