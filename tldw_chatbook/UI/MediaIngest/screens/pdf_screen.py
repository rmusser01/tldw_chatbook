"""PDF ingestion screen implementation."""

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
from ..models import PDFFormData, ProcessingStatus

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class PDFIngestScreen(BaseMediaIngestScreen):
    """Screen for PDF document ingestion and processing."""
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "pdf", **kwargs):
        super().__init__(app_instance, media_type, **kwargs)
    
    def create_media_options(self) -> ComposeResult:
        """Create PDF-specific options."""
        yield Label("PDF Processing Options", classes="section-title")
        
        # OCR options
        with Container(classes="option-group"):
            yield Label("OCR Settings", classes="option-title")
            yield Checkbox("Enable OCR for scanned PDFs", id="ocr-enabled", value=True)
            
            with Container(id="ocr-options"):
                yield Label("OCR Backend:")
                yield Select(
                    [
                        ("tesseract", "Tesseract"),
                        ("easyocr", "EasyOCR"),
                        ("doctr", "DocTR")
                    ],
                    id="ocr-backend",
                    classes="form-select",
                    value="tesseract"
                )
                
                yield Label("OCR Language:")
                yield Select(
                    [
                        ("eng", "English"),
                        ("spa", "Spanish"),
                        ("fra", "French"),
                        ("deu", "German"),
                        ("chi_sim", "Chinese Simplified"),
                        ("jpn", "Japanese")
                    ],
                    id="ocr-language",
                    classes="form-select",
                    value="eng"
                )
        
        # Extraction options
        with Container(classes="option-group"):
            yield Label("Extraction Settings", classes="option-title")
            yield Checkbox("Extract images", id="extract-images", value=True)
            yield Checkbox("Extract tables", id="extract-tables", value=True)
            yield Checkbox("Extract metadata", id="extract-metadata", value=True)
            yield Checkbox("Extract annotations", id="extract-annotations", value=False)
    
    def create_advanced_options(self) -> ComposeResult:
        """Create advanced PDF processing options."""
        yield Container(
            Label("Processing Options", classes="option-title"),
            Checkbox("Perform text analysis", id="perform-analysis", value=True),
            Checkbox("Enable chunking", id="perform-chunking", value=True),
            Container(
                Label("Chunk method:"),
                Select(
                    [
                        ("pages", "By pages"),
                        ("tokens", "By tokens"),
                        ("sentences", "By sentences")
                    ],
                    id="chunk-method",
                    classes="form-select",
                    value="pages"
                ),
                Label("Chunk size:"),
                Input(value="10", id="chunk-size", classes="form-input"),
                id="chunking-options"
            ),
            classes="option-group"
        )
        
        yield Container(
            Label("Output Options", classes="option-title"),
            Checkbox("Save as Markdown", id="save-markdown", value=True),
            Checkbox("Save as plain text", id="save-txt", value=False),
            Checkbox("Generate summary", id="generate-summary", value=True),
            classes="option-group"
        )
    
    @on(Checkbox.Changed, "#ocr-enabled")
    def toggle_ocr_options(self, event: Checkbox.Changed) -> None:
        """Show/hide OCR options."""
        options = self.query_one("#ocr-options")
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
        """Get and validate PDF form data."""
        data = super().get_validated_form_data()
        
        # OCR settings
        data['ocr_enabled'] = self.query_one("#ocr-enabled").value
        if data['ocr_enabled']:
            data['ocr_backend'] = self.query_one("#ocr-backend").value
            data['ocr_language'] = self.query_one("#ocr-language").value
        
        # Extraction settings
        data['extract_images'] = self.query_one("#extract-images").value
        data['extract_tables'] = self.query_one("#extract-tables").value
        data['extract_metadata'] = self.query_one("#extract-metadata").value
        data['extract_annotations'] = self.query_one("#extract-annotations").value
        
        # Advanced options
        data['perform_analysis'] = self.query_one("#perform-analysis").value
        data['perform_chunking'] = self.query_one("#perform-chunking").value
        if data['perform_chunking']:
            data['chunk_method'] = self.query_one("#chunk-method").value
            data['chunk_size'] = int(self.query_one("#chunk-size").value or 10)
        
        # Output options
        data['save_markdown'] = self.query_one("#save-markdown").value
        data['save_txt'] = self.query_one("#save-txt").value
        data['generate_summary'] = self.query_one("#generate-summary").value
        
        return data
    
    async def process_media_impl(self, form_data: Dict[str, Any]) -> AsyncIterator[ProcessingStatus]:
        """Process PDF files."""
        try:
            validated = PDFFormData(**form_data)
            all_inputs = list(validated.files) + validated.urls
            total = len(all_inputs)
            
            if total == 0:
                yield ProcessingStatus(
                    state="error",
                    error="No files or URLs to process",
                    message="Please provide at least one PDF file or URL"
                )
                return
            
            for idx, input_item in enumerate(all_inputs):
                file_name = Path(input_item).name if isinstance(input_item, Path) else input_item
                
                yield ProcessingStatus(
                    state="processing",
                    progress=(idx / total),
                    current_file=str(input_item),
                    current_operation="Reading PDF",
                    files_processed=idx,
                    total_files=total,
                    message=f"Processing: {file_name}"
                )
                
                # Simulate PDF processing stages
                if validated.ocr_enabled:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.3) / total,
                        current_file=str(input_item),
                        current_operation="OCR processing",
                        files_processed=idx,
                        total_files=total,
                        message=f"OCR processing: {file_name}"
                    )
                    await asyncio.sleep(1)
                
                if validated.extract_tables:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.5) / total,
                        current_file=str(input_item),
                        current_operation="Extracting tables",
                        files_processed=idx,
                        total_files=total,
                        message=f"Extracting tables: {file_name}"
                    )
                    await asyncio.sleep(0.5)
                
                if validated.perform_analysis:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.8) / total,
                        current_file=str(input_item),
                        current_operation="Analyzing content",
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
                message=f"Successfully processed {total} PDF(s)"
            )
            
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            yield ProcessingStatus(
                state="error",
                error=str(e),
                message=f"Processing failed: {str(e)}"
            )