# Example of refactored ingestion view using standardized components
"""
This is an example showing how to refactor existing ingestion views
to use the new standardized form components.
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from ..Widgets.form_components import (
    create_form_field, 
    create_form_row, 
    create_form_section,
    create_button_group,
    create_status_area
)
from ..Widgets.status_widget import EnhancedStatusWidget


def compose_prompts_tab_refactored() -> ComposeResult:
    """Example of refactored prompts tab using standardized components."""
    
    with Vertical(classes="ingest-form-container"):
        # File selection section
        yield from create_form_section(
            title="File Selection",
            fields=[],  # No fields, just buttons
            collapsible=False
        )
        
        yield from create_button_group([
            ("Select Prompt File(s)", "ingest-prompts-select-file-button", "default"),
            ("Clear Selection", "ingest-prompts-clear-files-button", "default")
        ])
        
        # Selected files display
        yield from create_form_field(
            label="Selected Files for Import",
            field_id="ingest-prompts-selected-files-list",
            field_type="select",  # Could be custom list widget
            options=[],
            classes="ingest-selected-files-list"
        )
        
        # Preview section
        yield from create_form_section(
            title="Preview",
            fields=[
                ("Preview of Parsed Prompts (Max 10 shown)", 
                 "ingest-prompts-preview-area", 
                 "textarea",
                 "Select files to see a preview...",
                 None,
                 None,
                 False,
                 {"read_only": True, "classes": "ingest-preview-area"})
            ],
            collapsible=True,
            collapsed=False,
            section_id="prompts-preview-section"
        )
        
        # Import action
        yield from create_button_group([
            ("Import Selected Prompts Now", "ingest-prompts-import-now-button", "primary")
        ], alignment="center")
        
        # Status area using enhanced widget
        yield EnhancedStatusWidget(
            title="Import Status",
            id="prompt-import-status-widget",
            max_messages=50
        )


def compose_video_tab_refactored() -> ComposeResult:
    """Example of refactored video tab with complex form."""
    
    with Vertical(classes="ingest-form-container"):
        # Basic info section
        yield from create_form_section(
            title="Media Information",
            fields=[
                ("Video URLs (one per line)", "video-urls", "textarea", 
                 "https://youtube.com/watch?v=...\nhttps://vimeo.com/..."),
                ("Title Override", "video-title", "input", "Optional custom title"),
                ("Keywords", "video-keywords", "input", "comma, separated, keywords")
            ],
            collapsible=False
        )
        
        # Processing options in a row
        yield from create_form_row(
            ("Chunk Size", "chunk-size", "input", "500", "500"),
            ("Chunk Overlap", "chunk-overlap", "input", "200", "200")
        )
        
        # Advanced options section
        yield from create_form_section(
            title="Advanced Options",
            fields=[
                ("Enable Transcription", "enable-transcription", "checkbox", "", True),
                ("Transcription Language", "trans-lang", "select", "", "en",
                 [("English", "en"), ("Spanish", "es"), ("French", "fr")]),
                ("Include Timestamps", "include-timestamps", "checkbox", "", True)
            ],
            collapsible=True,
            collapsed=True,
            section_id="video-advanced-options"
        )
        
        # Action buttons
        yield from create_button_group([
            ("Cancel", "video-cancel-button", "default"),
            ("Process Video", "video-process-button", "success")
        ], alignment="right")
        
        # Enhanced status widget
        yield EnhancedStatusWidget(
            title="Processing Status",
            id="video-processing-status",
            show_timestamp=True
        )


# Example of using the enhanced status widget in event handlers
async def handle_video_processing(self, status_widget: EnhancedStatusWidget):
    """Example of using the enhanced status widget."""
    
    status_widget.add_info("Starting video processing...")
    
    try:
        # Simulate processing steps
        status_widget.add_info("Validating URLs...")
        await self.validate_urls()
        status_widget.add_success("URLs validated successfully")
        
        status_widget.add_info("Downloading video...")
        await self.download_video()
        status_widget.add_success("Video downloaded")
        
        status_widget.add_info("Transcribing audio...")
        await self.transcribe_audio()
        status_widget.add_success("Transcription complete")
        
        status_widget.add_info("Processing chunks...")
        await self.process_chunks()
        status_widget.add_success("Chunking complete")
        
        status_widget.add_success("Video processing completed successfully!")
        
    except ValidationError as e:
        status_widget.add_error(f"Validation failed: {e}")
    except DownloadError as e:
        status_widget.add_error(f"Download failed: {e}")
        status_widget.add_warning("You can retry with a different URL")
    except Exception as e:
        status_widget.add_error(f"Unexpected error: {e}")
        status_widget.add_debug(f"Stack trace: {traceback.format_exc()}")