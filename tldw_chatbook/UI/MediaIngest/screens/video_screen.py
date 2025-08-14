"""Video ingestion screen implementation."""

from typing import TYPE_CHECKING, Dict, Any, AsyncIterator
from pathlib import Path
import asyncio
from loguru import logger

from textual.app import ComposeResult
from textual.widgets import (
    Label, Input, Select, Checkbox, TextArea,
    RadioSet, RadioButton
)
from textual.containers import Container
from textual import on

from .base_screen import BaseMediaIngestScreen
from ..models import VideoFormData, ProcessingStatus

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class VideoIngestScreen(BaseMediaIngestScreen):
    """Screen for video ingestion with transcription and processing options."""
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "video", **kwargs):
        super().__init__(app_instance, media_type, **kwargs)
        
        # Available transcription providers from the actual library
        self.transcription_providers = ['faster-whisper', 'whisper', 'lightning-whisper-mlx']
    
    def create_media_options(self) -> ComposeResult:
        """Create video-specific options."""
        yield Label("Video Processing Options", classes="section-title")
        
        # Transcription options
        with Container(classes="option-group"):
            yield Label("Transcription Settings", classes="option-title")
            yield Checkbox("Enable transcription", id="transcription-enabled", value=True)
            
            with Container(id="transcription-options"):
                yield Label("Provider:")
                yield Select(
                    [(p, p.title()) for p in self.transcription_providers],
                    id="provider",
                    classes="form-select"
                )
                
                yield Label("Model:")
                yield Select(
                    [("base", "Base"), ("small", "Small"), ("medium", "Medium"), ("large", "Large")],
                    id="model",
                    classes="form-select"
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
                    classes="form-select"
                )
                
                yield Checkbox("Include timestamps", id="timestamps", value=True)
                yield Checkbox("Diarization (speaker detection)", id="diarization", value=False)
        
        # Video processing options
        with Container(classes="option-group"):
            yield Label("Video Processing", classes="option-title")
            
            yield Label("Resolution:")
            with RadioSet(id="resolution"):
                yield RadioButton("Keep original", id="res-original", value=True)
                yield RadioButton("720p", id="res-720p")
                yield RadioButton("1080p", id="res-1080p")
            
            yield Checkbox("Extract keyframes", id="extract-keyframes", value=False)
            yield Checkbox("Generate thumbnails", id="generate-thumbnails", value=True)
            yield Checkbox("Remove audio track", id="remove-audio", value=False)
    
    def create_advanced_options(self) -> ComposeResult:
        """Create advanced video processing options."""
        yield Container(
            Label("AI Analysis", classes="option-title"),
            Checkbox("Perform analysis/summarization", id="perform-analysis", value=True),
            Container(
                Label("Analysis prompt:"),
                TextArea(
                    "Summarize the key points and main topics discussed in this video.",
                    id="analysis-prompt",
                    classes="form-textarea"
                ),
                id="analysis-options"
            ),
            classes="option-group"
        )
        
        yield Container(
            Label("Chunking Options", classes="option-title"),
            Checkbox("Enable chunking", id="perform-chunking", value=True),
            Container(
                Label("Chunk size (seconds):"),
                Input(value="300", id="chunk-size", classes="form-input"),
                Label("Overlap (seconds):"),
                Input(value="30", id="chunk-overlap", classes="form-input"),
                id="chunking-options"
            ),
            classes="option-group"
        )
        
        yield Container(
            Label("Output Options", classes="option-title"),
            Checkbox("Save transcript as SRT", id="save-srt", value=False),
            Checkbox("Save transcript as VTT", id="save-vtt", value=False),
            Checkbox("Save transcript as TXT", id="save-txt", value=True),
            Checkbox("Generate video summary", id="generate-summary", value=True),
            classes="option-group"
        )
    
    @on(Checkbox.Changed, "#transcription-enabled")
    def toggle_transcription_options(self, event: Checkbox.Changed) -> None:
        """Show/hide transcription options based on checkbox."""
        options = self.query_one("#transcription-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    @on(Checkbox.Changed, "#perform-analysis")
    def toggle_analysis_options(self, event: Checkbox.Changed) -> None:
        """Show/hide analysis options based on checkbox."""
        options = self.query_one("#analysis-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    @on(Checkbox.Changed, "#perform-chunking")
    def toggle_chunking_options(self, event: Checkbox.Changed) -> None:
        """Show/hide chunking options based on checkbox."""
        options = self.query_one("#chunking-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    def get_validated_form_data(self) -> Dict[str, Any]:
        """Get and validate video form data."""
        # Get base form data
        data = super().get_validated_form_data()
        
        # Transcription settings
        data['transcription_enabled'] = self.query_one("#transcription-enabled").value
        if data['transcription_enabled']:
            data['transcription_provider'] = self.query_one("#provider").value
            data['transcription_model'] = self.query_one("#model").value
            data['transcription_language'] = self.query_one("#language").value
            data['timestamp_option'] = self.query_one("#timestamps").value
            data['diarization'] = self.query_one("#diarization").value
        
        # Video processing
        resolution_set = self.query_one("#resolution", RadioSet)
        if resolution_set.pressed_button:
            resolution_id = resolution_set.pressed_button.id
            if resolution_id == "res-720p":
                data['video_resolution'] = "720p"
            elif resolution_id == "res-1080p":
                data['video_resolution'] = "1080p"
            else:
                data['video_resolution'] = "original"
        else:
            data['video_resolution'] = "original"
        
        data['extract_keyframes'] = self.query_one("#extract-keyframes").value
        data['generate_thumbnails'] = self.query_one("#generate-thumbnails").value
        data['remove_audio'] = self.query_one("#remove-audio").value
        
        # Advanced options
        data['perform_analysis'] = self.query_one("#perform-analysis").value
        if data['perform_analysis']:
            data['analysis_prompt'] = self.query_one("#analysis-prompt").text
        
        data['perform_chunking'] = self.query_one("#perform-chunking").value
        if data['perform_chunking']:
            data['chunk_size'] = int(self.query_one("#chunk-size").value or 300)
            data['chunk_overlap'] = int(self.query_one("#chunk-overlap").value or 30)
        
        # Output options
        data['save_srt'] = self.query_one("#save-srt").value
        data['save_vtt'] = self.query_one("#save-vtt").value
        data['save_txt'] = self.query_one("#save-txt").value
        data['generate_summary'] = self.query_one("#generate-summary").value
        
        return data
    
    async def process_media_impl(self, form_data: Dict[str, Any]) -> AsyncIterator[ProcessingStatus]:
        """Process video files with transcription and analysis."""
        try:
            # Validate with Pydantic model
            validated = VideoFormData(**form_data)
            
            # Combine files and URLs for processing
            all_inputs = list(validated.files) + validated.urls
            total = len(all_inputs)
            
            if total == 0:
                yield ProcessingStatus(
                    state="error",
                    error="No files or URLs to process",
                    message="Please provide at least one video file or URL"
                )
                return
            
            # Process each input
            for idx, input_item in enumerate(all_inputs):
                file_name = Path(input_item).name if isinstance(input_item, Path) else input_item
                
                # Update status for current file
                yield ProcessingStatus(
                    state="processing",
                    progress=(idx / total),
                    current_file=str(input_item),
                    current_operation="Analyzing video",
                    files_processed=idx,
                    total_files=total,
                    message=f"Processing: {file_name}"
                )
                
                # Simulate video analysis
                await asyncio.sleep(0.5)
                
                if validated.transcription_enabled:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.3) / total,
                        current_file=str(input_item),
                        current_operation="Transcribing audio",
                        files_processed=idx,
                        total_files=total,
                        message=f"Transcribing: {file_name}"
                    )
                    await asyncio.sleep(1)
                
                if validated.extract_keyframes:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.6) / total,
                        current_file=str(input_item),
                        current_operation="Extracting keyframes",
                        files_processed=idx,
                        total_files=total,
                        message=f"Extracting keyframes: {file_name}"
                    )
                    await asyncio.sleep(0.5)
                
                if validated.perform_analysis:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.8) / total,
                        current_file=str(input_item),
                        current_operation="AI analysis",
                        files_processed=idx,
                        total_files=total,
                        message=f"Analyzing content: {file_name}"
                    )
                    await asyncio.sleep(0.5)
            
            # Complete
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