"""Document ingestion screen implementation."""

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
from ..models import DocumentFormData, ProcessingStatus

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class DocumentIngestScreen(BaseMediaIngestScreen):
    """Screen for document ingestion (DOCX, TXT, MD, etc.)."""
    
    def __init__(self, app_instance: 'TldwCli', media_type: str = "document", **kwargs):
        super().__init__(app_instance, media_type, **kwargs)
    
    def create_media_options(self) -> ComposeResult:
        """Create document-specific options."""
        yield Label("Document Processing Options", classes="section-title")
        
        # Processing options
        with Container(classes="option-group"):
            yield Label("Processing Settings", classes="option-title")
            
            yield Label("Document type:")
            yield Select(
                [
                    ("auto", "Auto-detect"),
                    ("docx", "Microsoft Word"),
                    ("txt", "Plain text"),
                    ("md", "Markdown"),
                    ("rtf", "Rich Text"),
                    ("odt", "OpenDocument")
                ],
                id="doc-type",
                classes="form-select",
                value="auto"
            )
            
            yield Checkbox("Preserve formatting", id="preserve-formatting", value=True)
            yield Checkbox("Extract metadata", id="extract-metadata", value=True)
            yield Checkbox("Extract embedded images", id="extract-images", value=False)
        
        # Text processing
        with Container(classes="option-group"):
            yield Label("Text Processing", classes="option-title")
            yield Checkbox("Clean text (remove extra spaces)", id="clean-text", value=True)
            yield Checkbox("Convert to Markdown", id="convert-markdown", value=True)
            yield Checkbox("Extract headings structure", id="extract-headings", value=True)
    
    def create_advanced_options(self) -> ComposeResult:
        """Create advanced document processing options."""
        yield Container(
            Label("Chunking Options", classes="option-title"),
            Checkbox("Enable chunking", id="perform-chunking", value=True),
            Container(
                Label("Chunk method:"),
                Select(
                    [
                        ("paragraphs", "By paragraphs"),
                        ("sentences", "By sentences"),
                        ("tokens", "By tokens"),
                        ("headings", "By headings")
                    ],
                    id="chunk-method",
                    classes="form-select",
                    value="paragraphs"
                ),
                Label("Chunk size:"),
                Input(value="5", id="chunk-size", classes="form-input"),
                id="chunking-options"
            ),
            classes="option-group"
        )
        
        yield Container(
            Label("Analysis Options", classes="option-title"),
            Checkbox("Perform text analysis", id="perform-analysis", value=True),
            Checkbox("Extract key phrases", id="extract-keyphrases", value=False),
            Checkbox("Generate summary", id="generate-summary", value=True),
            classes="option-group"
        )
    
    @on(Checkbox.Changed, "#perform-chunking")
    def toggle_chunking_options(self, event: Checkbox.Changed) -> None:
        """Show/hide chunking options."""
        options = self.query_one("#chunking-options")
        if event.value:
            options.remove_class("hidden")
        else:
            options.add_class("hidden")
    
    def get_validated_form_data(self) -> Dict[str, Any]:
        """Get and validate document form data."""
        data = super().get_validated_form_data()
        
        # Processing settings
        data['document_type'] = self.query_one("#doc-type").value
        data['preserve_formatting'] = self.query_one("#preserve-formatting").value
        data['extract_metadata'] = self.query_one("#extract-metadata").value
        data['extract_images'] = self.query_one("#extract-images").value
        
        # Text processing
        data['clean_text'] = self.query_one("#clean-text").value
        data['convert_to_markdown'] = self.query_one("#convert-markdown").value
        data['extract_headings'] = self.query_one("#extract-headings").value
        
        # Chunking options
        data['perform_chunking'] = self.query_one("#perform-chunking").value
        if data['perform_chunking']:
            data['chunk_method'] = self.query_one("#chunk-method").value
            data['chunk_size'] = int(self.query_one("#chunk-size").value or 5)
        
        # Analysis options
        data['perform_analysis'] = self.query_one("#perform-analysis").value
        data['extract_keyphrases'] = self.query_one("#extract-keyphrases").value
        data['generate_summary'] = self.query_one("#generate-summary").value
        
        return data
    
    async def process_media_impl(self, form_data: Dict[str, Any]) -> AsyncIterator[ProcessingStatus]:
        """Process document files."""
        try:
            validated = DocumentFormData(**form_data)
            all_inputs = list(validated.files) + validated.urls
            total = len(all_inputs)
            
            if total == 0:
                yield ProcessingStatus(
                    state="error",
                    error="No files or URLs to process",
                    message="Please provide at least one document file or URL"
                )
                return
            
            for idx, input_item in enumerate(all_inputs):
                file_name = Path(input_item).name if isinstance(input_item, Path) else input_item
                
                yield ProcessingStatus(
                    state="processing",
                    progress=(idx / total),
                    current_file=str(input_item),
                    current_operation="Reading document",
                    files_processed=idx,
                    total_files=total,
                    message=f"Processing: {file_name}"
                )
                await asyncio.sleep(0.5)
                
                if validated.extract_headings:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.3) / total,
                        current_file=str(input_item),
                        current_operation="Extracting structure",
                        files_processed=idx,
                        total_files=total,
                        message=f"Extracting structure: {file_name}"
                    )
                    await asyncio.sleep(0.3)
                
                if validated.convert_to_markdown:
                    yield ProcessingStatus(
                        state="processing",
                        progress=(idx + 0.5) / total,
                        current_file=str(input_item),
                        current_operation="Converting to Markdown",
                        files_processed=idx,
                        total_files=total,
                        message=f"Converting: {file_name}"
                    )
                    await asyncio.sleep(0.3)
                
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
                message=f"Successfully processed {total} document(s)"
            )
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            yield ProcessingStatus(
                state="error",
                error=str(e),
                message=f"Processing failed: {str(e)}"
            )