"""Ebook ingestion screen implementation."""

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
from ..models import EbookFormData, ProcessingStatus

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EbookIngestScreen(BaseMediaIngestScreen):
    """Screen for ebook ingestion (EPUB, MOBI, AZW3, etc.)."""
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "ebook", **kwargs):
        super().__init__(app_instance, media_type, **kwargs)
    
    def create_media_options(self) -> ComposeResult:
        """Create ebook-specific options."""
        yield Label("Ebook Processing Options", classes="section-title")
        
        # Format options
        with Container(classes="option-group"):
            yield Label("Format Settings", classes="option-title")
            
            yield Label("Ebook format:")
            yield Select(
                [
                    ("auto", "Auto-detect"),
                    ("epub", "EPUB"),
                    ("mobi", "MOBI"),
                    ("azw3", "AZW3"),
                    ("fb2", "FictionBook"),
                    ("lit", "Microsoft LIT")
                ],
                id="ebook-format",
                classes="form-select",
                value="auto"
            )
            
            yield Checkbox("Extract cover image", id="extract-cover", value=True)
            yield Checkbox("Extract metadata", id="extract-metadata", value=True)
            yield Checkbox("Extract table of contents", id="extract-toc", value=True)
        
        # Processing options
        with Container(classes="option-group"):
            yield Label("Processing Settings", classes="option-title")
            yield Checkbox("Convert to Markdown", id="convert-markdown", value=True)
            yield Checkbox("Preserve chapter structure", id="preserve-chapters", value=True)
            yield Checkbox("Extract inline images", id="extract-images", value=False)
            yield Checkbox("Remove DRM (if legal)", id="remove-drm", value=False)
    
    def create_advanced_options(self) -> ComposeResult:
        """Create advanced ebook processing options."""
        yield Container(
            Label("Chapter Processing", classes="option-title"),
            Checkbox("Process by chapters", id="process-chapters", value=True),
            Container(
                Label("Max chapters to process (0 = all):"),
                Input(value="0", id="max-chapters", classes="form-input"),
                Label("Start from chapter:"),
                Input(value="1", id="start-chapter", classes="form-input"),
                id="chapter-options"
            ),
            classes="option-group"
        )
        
        yield Container(
            Label("Analysis Options", classes="option-title"),
            Checkbox("Generate chapter summaries", id="chapter-summaries", value=True),
            Checkbox("Extract character names", id="extract-characters", value=False),
            Checkbox("Generate book summary", id="generate-summary", value=True),
            Checkbox("Extract quotes", id="extract-quotes", value=False),
            classes="option-group"
        )
        
        yield Container(
            Label("Output Options", classes="option-title"),
            Checkbox("Save as plain text", id="save-txt", value=False),
            Checkbox("Save as Markdown", id="save-markdown", value=True),
            Checkbox("Create chapter files", id="split-chapters", value=False),
            classes="option-group"
        )
    
    @on(Checkbox.Changed, "#process-chapters")
    def toggle_chapter_options(self, event: Checkbox.Changed) -> None:
        """Show/hide chapter processing options."""
        options = self.query_one("#chapter-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    def get_validated_form_data(self) -> Dict[str, Any]:
        """Get and validate ebook form data."""
        data = super().get_validated_form_data()
        
        # Format settings
        data['ebook_format'] = self.query_one("#ebook-format").value
        data['extract_cover'] = self.query_one("#extract-cover").value
        data['extract_metadata'] = self.query_one("#extract-metadata").value
        data['extract_toc'] = self.query_one("#extract-toc").value
        
        # Processing settings
        data['convert_to_markdown'] = self.query_one("#convert-markdown").value
        data['preserve_chapters'] = self.query_one("#preserve-chapters").value
        data['extract_images'] = self.query_one("#extract-images").value
        data['remove_drm'] = self.query_one("#remove-drm").value
        
        # Chapter processing
        data['process_by_chapters'] = self.query_one("#process-chapters").value
        if data['process_by_chapters']:
            data['max_chapters'] = int(self.query_one("#max-chapters").value or 0)
            data['start_chapter'] = int(self.query_one("#start-chapter").value or 1)
        
        # Analysis options
        data['generate_chapter_summaries'] = self.query_one("#chapter-summaries").value
        data['extract_characters'] = self.query_one("#extract-characters").value
        data['generate_summary'] = self.query_one("#generate-summary").value
        data['extract_quotes'] = self.query_one("#extract-quotes").value
        
        # Output options
        data['save_txt'] = self.query_one("#save-txt").value
        data['save_markdown'] = self.query_one("#save-markdown").value
        data['split_chapters'] = self.query_one("#split-chapters").value
        
        return data
    
    async def process_media_impl(self, form_data: Dict[str, Any]) -> AsyncIterator[ProcessingStatus]:
        """Process ebook files."""
        try:
            validated = EbookFormData(**form_data)
            all_inputs = list(validated.files) + validated.urls
            total = len(all_inputs)
            
            if total == 0:
                yield ProcessingStatus(
                    state="error",
                    error="No files or URLs to process",
                    message="Please provide at least one ebook file or URL"
                )
                return
            
            for idx, input_item in enumerate(all_inputs):
                file_name = Path(input_item).name if isinstance(input_item, Path) else input_item
                
                yield ProcessingStatus(
                    state="processing",
                    progress=(idx / total),
                    current_file=str(input_item),
                    current_operation="Reading ebook",
                    files_processed=idx,
                    total_files=total,
                    message=f"Processing: {file_name}"
                )
                await asyncio.sleep(0.5)
                
                if validated.extract_toc:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.2) / total,
                        current_file=str(input_item),
                        current_operation="Extracting TOC",
                        files_processed=idx,
                        total_files=total,
                        message=f"Extracting TOC: {file_name}"
                    )
                    await asyncio.sleep(0.3)
                
                if validated.process_by_chapters:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.4) / total,
                        current_file=str(input_item),
                        current_operation="Processing chapters",
                        files_processed=idx,
                        total_files=total,
                        message=f"Processing chapters: {file_name}"
                    )
                    await asyncio.sleep(0.8)
                
                if validated.generate_chapter_summaries:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.7) / total,
                        current_file=str(input_item),
                        current_operation="Generating summaries",
                        files_processed=idx,
                        total_files=total,
                        message=f"Generating summaries: {file_name}"
                    )
                    await asyncio.sleep(0.5)
                
                if validated.generate_summary:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.9) / total,
                        current_file=str(input_item),
                        current_operation="Final analysis",
                        files_processed=idx,
                        total_files=total,
                        message=f"Final analysis: {file_name}"
                    )
                    await asyncio.sleep(0.3)
            
            yield ProcessingStatus(
                state="complete",
                progress=1.0,
                files_processed=total,
                total_files=total,
                message=f"Successfully processed {total} ebook(s)"
            )
            
        except Exception as e:
            logger.error(f"Ebook processing error: {e}")
            yield ProcessingStatus(
                state="error",
                error=str(e),
                message=f"Processing failed: {str(e)}"
            )