# tldw_chatbook/UI/Ingest_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
from pathlib import Path
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.widgets import Static, Button, Input, Select, Checkbox, TextArea, Label, RadioSet, RadioButton, Collapsible, ListView, ListItem, Markdown, LoadingIndicator, TabbedContent, TabPane # Button, ListView, ListItem, Label are already here

# Configure logger with context
logger = logger.bind(module="Ingest_Window")

from ..Constants import TLDW_API_AUDIO_OPTIONS_ID, TLDW_API_VIDEO_OPTIONS_ID, TLDW_API_PDF_OPTIONS_ID, \
    TLDW_API_EBOOK_OPTIONS_ID, TLDW_API_DOCUMENT_OPTIONS_ID, TLDW_API_XML_OPTIONS_ID, TLDW_API_MEDIAWIKI_OPTIONS_ID
#
# Local Imports
from ..Third_Party.textual_fspicker.file_open import FileOpen
from ..tldw_api.schemas import MediaType, ChunkMethod, PdfEngine  # Import Enums
from ..Widgets.IngestTldwApiTabbedWindow import IngestTldwApiTabbedWindow
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

MEDIA_TYPES = ['video', 'audio', 'document', 'pdf', 'ebook', 'xml', 'mediawiki_dump']

INGEST_VIEW_IDS = [
    "ingest-view-prompts", "ingest-view-characters",
    "ingest-view-media", "ingest-view-notes",
    "ingest-view-tldw-api"  # Single view for all tldw API options
]
INGEST_NAV_BUTTON_IDS = [
    "ingest-nav-prompts", "ingest-nav-characters",
    "ingest-nav-media", "ingest-nav-notes",
    "ingest-nav-tldw-api"  # Single button for all tldw API options
]

class IngestWindow(Container):
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = {}  # Stores {media_type: [Path, ...]}
        self._current_media_type_for_file_dialog = None # Stores the media_type for the active file dialog
        logger.debug("IngestWindow initialized.")
    
    def _get_file_filters_for_media_type(self, media_type: str):
        """Returns appropriate file filters for the given media type."""
        from ..Third_Party.textual_fspicker import Filters
        
        if media_type == "video":
            return Filters(
                ("Video Files", lambda p: p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg")),
                ("All Files", lambda _: True)
            )
        elif media_type == "audio":
            return Filters(
                ("Audio Files", lambda p: p.suffix.lower() in (".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".opus", ".aiff")),
                ("All Files", lambda _: True)
            )
        elif media_type == "document":
            return Filters(
                ("Document Files", lambda p: p.suffix.lower() in (".docx", ".doc", ".odt", ".rtf", ".txt")),
                ("All Files", lambda _: True)
            )
        elif media_type == "pdf":
            return Filters(
                ("PDF Files", lambda p: p.suffix.lower() == ".pdf"),
                ("All Files", lambda _: True)
            )
        elif media_type == "ebook":
            return Filters(
                ("Ebook Files", lambda p: p.suffix.lower() in (".epub", ".mobi", ".azw", ".azw3", ".fb2")),
                ("All Files", lambda _: True)
            )
        elif media_type == "xml":
            return Filters(
                ("XML Files", lambda p: p.suffix.lower() in (".xml", ".xsd", ".xsl")),
                ("All Files", lambda _: True)
            )
        elif media_type == "plaintext":
            return Filters(
                ("Text Files", lambda p: p.suffix.lower() in (".txt", ".md", ".text", ".log", ".csv")),
                ("All Files", lambda _: True)
            )
        else:
            # Default filters
            return Filters(
                ("All Files", lambda _: True)
            )


    def compose(self) -> ComposeResult:
        logger.debug("Composing IngestWindow UI")
        with VerticalScroll(id="ingest-nav-pane", classes="ingest-nav-pane"):
            yield Static("Ingestion Methods", classes="sidebar-title")
            yield Button("Ingest Prompts", id="ingest-nav-prompts", classes="ingest-nav-button")
            yield Button("Ingest Characters", id="ingest-nav-characters", classes="ingest-nav-button")
            yield Button("Ingest Media (Local)", id="ingest-nav-media", classes="ingest-nav-button")
            yield Button("Ingest Notes", id="ingest-nav-notes", classes="ingest-nav-button")
            yield Button("Ingest Content via tldw API", id="ingest-nav-tldw-api", classes="ingest-nav-button")


        with Container(id="ingest-content-pane", classes="ingest-content-pane"):
            # --- Prompts Ingest View ---
            with Vertical(id="ingest-view-prompts", classes="ingest-view-area"):
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Select Prompt File(s)", id="ingest-prompts-select-file-button")
                    yield Button("Clear Selection", id="ingest-prompts-clear-files-button")
                yield Label("Selected Files for Import:", classes="ingest-label")
                yield ListView(id="ingest-prompts-selected-files-list", classes="ingest-selected-files-list")

                yield Label("Preview of Parsed Prompts (Max 10 shown):", classes="ingest-label")
                with VerticalScroll(id="ingest-prompts-preview-area", classes="ingest-preview-area"):
                    yield Static("Select files to see a preview.", id="ingest-prompts-preview-placeholder")
                yield Button("Import Selected Prompts Now", id="ingest-prompts-import-now-button", variant="primary")
                yield Label("Import Status:", classes="ingest-label")
                yield TextArea(id="prompt-import-status-area", read_only=True, classes="ingest-status-area")

            # --- Characters Ingest View ---
            with Vertical(id="ingest-view-characters", classes="ingest-view-area"):
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Select Character File(s)", id="ingest-characters-select-file-button")
                    yield Button("Clear Selection", id="ingest-characters-clear-files-button")
                yield Label("Selected Files for Import:", classes="ingest-label")
                yield ListView(id="ingest-characters-selected-files-list", classes="ingest-selected-files-list")

                yield Label("Preview of Parsed Characters (Max 5 shown):", classes="ingest-label")
                with VerticalScroll(id="ingest-characters-preview-area", classes="ingest-preview-area"):
                    yield Static("Select files to see a preview.", id="ingest-characters-preview-placeholder")

                yield Button("Import Selected Characters Now", id="ingest-characters-import-now-button",
                             variant="primary")
                yield Label("Import Status:", classes="ingest-label")
                yield TextArea(id="ingest-character-import-status-area", read_only=True, classes="ingest-status-area")

            # --- Notes Ingest View ---
            with Vertical(id="ingest-view-notes", classes="ingest-view-area"):
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Select Notes File(s)", id="ingest-notes-select-file-button")
                    yield Button("Clear Selection", id="ingest-notes-clear-files-button")
                yield Label("Selected Files for Import:", classes="ingest-label")
                yield ListView(id="ingest-notes-selected-files-list", classes="ingest-selected-files-list")

                yield Label("Preview of Parsed Notes (Max 10 shown):", classes="ingest-label")
                with VerticalScroll(id="ingest-notes-preview-area", classes="ingest-preview-area"):
                    yield Static("Select files to see a preview.", id="ingest-notes-preview-placeholder")

                # ID used in ingest_events.py will be:
                # ingest-notes-import-now-button
                yield Button("Import Selected Notes Now", id="ingest-notes-import-now-button", variant="primary")
                yield Label("Import Status:", classes="ingest-label")
                yield TextArea(id="ingest-notes-import-status-area", read_only=True, classes="ingest-status-area")

            # --- Other Ingest Views ---
            with Container(id="ingest-view-media", classes="ingest-view-area"):
                yield Static("Local Media Ingestion", classes="sidebar-title")
                with TabbedContent(id="ingest-local-tabs"):
                    with TabPane("Video", id="ingest-local-tab-video"):
                        yield from self.compose_local_video_tab()
                    with TabPane("Audio", id="ingest-local-tab-audio"):
                        yield from self.compose_local_audio_tab()
                    with TabPane("Document", id="ingest-local-tab-document"):
                        yield from self.compose_local_document_tab()
                    with TabPane("PDF", id="ingest-local-tab-pdf"):
                        yield from self.compose_local_pdf_tab()
                    with TabPane("Ebook", id="ingest-local-tab-ebook"):
                        yield from self.compose_local_ebook_tab()
                    with TabPane("Web Article", id="ingest-local-tab-web"):
                        yield from self.compose_local_web_article_tab()
                    with TabPane("XML", id="ingest-local-tab-xml"):
                        yield from self.compose_local_xml_tab()
                    with TabPane("Plaintext", id="ingest-local-tab-plaintext"):
                        yield from self.compose_local_plaintext_tab()

            # Container for the new tabbed tldw API interface
            with Container(id="ingest-view-tldw-api", classes="ingest-view-area"):
                # Create an instance of the tabbed window within the container
                yield IngestTldwApiTabbedWindow(self.app_instance, id="tldw-api-tabbed-window")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if not button_id: # Should always have an ID
            return

        # Log all button presses for debugging
        logger.info(f"IngestWindow.on_button_pressed: Received button press for ID: '{button_id}'")
        
        # Check if this is a navigation button - if so, don't handle it here
        if button_id in INGEST_NAV_BUTTON_IDS:
            logger.info(f"IngestWindow.on_button_pressed: Navigation button '{button_id}' pressed, not handling here")
            # Don't call event.stop() so it bubbles up to app level
            return

        if button_id.startswith("tldw-api-browse-local-files-button-"):
            event.stop()
            media_type = button_id.replace("tldw-api-browse-local-files-button-", "")
            self._current_media_type_for_file_dialog = media_type

            raw_initial_path = self.app_instance.app_config.get("user_data_path", Path.home())
            dialog_initial_path = str(raw_initial_path)

            logger.debug(f"Opening file dialog for media type '{media_type}' with initial path '{dialog_initial_path}'.")

            await self.app.push_screen(
                FileOpen(
                    title=f"Select Local File for {media_type.title()}"
                ),
                callback=self.handle_file_picker_dismissed
            )
        
        # Handle local media file selection buttons
        elif button_id.startswith("ingest-local-") and button_id.endswith("-select-files"):
            event.stop()
            # Extract media type from button ID: ingest-local-[media_type]-select-files
            parts = button_id.split("-")
            if len(parts) >= 4:
                media_type = parts[2]  # Get the media type part
                self._current_media_type_for_file_dialog = f"local_{media_type}"
                
                raw_initial_path = self.app_instance.app_config.get("user_data_path", Path.home())
                dialog_initial_path = str(raw_initial_path)
                
                # Set appropriate file filters based on media type
                filters = self._get_file_filters_for_media_type(media_type)
                
                logger.debug(f"Opening file dialog for local {media_type} with initial path '{dialog_initial_path}'.")
                
                await self.app.push_screen(
                    FileOpen(
                        title=f"Select {media_type.title()} Files",
                        filters=filters
                    ),
                    callback=self.handle_file_picker_dismissed
                )
        
        # Handle local media clear selection buttons
        elif button_id.startswith("ingest-local-") and button_id.endswith("-clear-files"):
            event.stop()
            # Extract media type from button ID
            parts = button_id.split("-")
            if len(parts) >= 4:
                media_type = parts[2]
                local_key = f"local_{media_type}"
                
                # Clear the selected files for this media type
                if local_key in self.selected_local_files:
                    self.selected_local_files[local_key] = []
                    
                    # Update the ListView
                    list_view_id = f"#ingest-local-{media_type}-files-list"
                    try:
                        list_view = self.query_one(list_view_id, ListView)
                        await list_view.clear()
                        logger.info(f"Cleared selected files for local {media_type}")
                    except Exception as e:
                        logger.error(f"Error clearing ListView for local {media_type}: {e}")
        
        # Handle web article clear URLs button
        elif button_id == "ingest-local-web-clear-urls":
            event.stop()
            try:
                urls_textarea = self.query_one("#ingest-local-web-urls", TextArea)
                urls_textarea.clear()
                logger.info("Cleared web article URLs")
            except Exception as e:
                logger.error(f"Error clearing web URLs: {e}")
        
        # If IngestWindow has a superclass that also defines on_button_pressed, consider calling it:
        # else:
        #     await super().on_button_pressed(event) # Example if there's a relevant superclass method

    async def handle_file_picker_dismissed(self, selected_file_path: Path | None) -> None:
        logger.debug(f"File picker dismissed, selected path: {selected_file_path}")
        if self._current_media_type_for_file_dialog is None:
            logger.warning("File picker dismissed but no media type context was set. Ignoring.")
            return

        media_type = self._current_media_type_for_file_dialog

        if not selected_file_path: # Handles None if dialog was cancelled or no path returned
            logger.info(f"No file selected or dialog cancelled for media type '{media_type}'.")
            return

        # Ensure the list for this media type exists in our tracking dictionary
        if media_type not in self.selected_local_files:
            self.selected_local_files[media_type] = []

        is_duplicate = False
        for existing_path in self.selected_local_files[media_type]:
            if str(existing_path) == str(selected_file_path):
                is_duplicate = True
                break

        if not is_duplicate:
            self.selected_local_files[media_type].append(selected_file_path)
            logger.info(f"Added '{selected_file_path}' to selected files for media type '{media_type}'.")
        else:
            logger.info(f"File '{selected_file_path}' already selected for media type '{media_type}'. Not adding again.")

        # Determine the correct ListView ID based on whether it's a local media type
        if media_type.startswith("local_"):
            # For local media ingestion, extract the actual media type
            actual_media_type = media_type.replace("local_", "")
            list_view_id = f"#ingest-local-{actual_media_type}-files-list"
        else:
            # For tldw API ingestion
            list_view_id = f"#tldw-api-selected-local-files-list-{media_type}"
        
        try:
            list_view = self.query_one(list_view_id, ListView)
            await list_view.clear()

            for path_item in self.selected_local_files[media_type]:
                list_item = ListItem(Label(str(path_item))) # Ensure Label is imported
                await list_view.append(list_item)
            logger.debug(f"Updated ListView '{list_view_id}' for media type '{media_type}'.")
        except Exception as e:
            logger.error(f"Error updating ListView {list_view_id} for {media_type}: {e}", exc_info=True)
    
    async def on_ingest_tldw_api_tabbed_window_back_button_pressed(self, message: IngestTldwApiTabbedWindow.BackButtonPressed) -> None:
        """Handle the back button press from the tldw API tabbed window."""
        logger.debug("Back button pressed in tldw API tabbed window, returning to main ingest view")
        # Go back to the main ingest navigation view
        # This will trigger the reactive watcher which will hide the tldw API view
        self.app_instance.ingest_active_view = "ingest-view-prompts"  # Or any other default view

    # --- Local Media Tab Composition Methods ---
    
    def compose_local_video_tab(self) -> ComposeResult:
        """Composes the Video tab content for local media ingestion."""
        with VerticalScroll(classes="ingest-media-tab-content"):
            # File Selection Section
            with Container(classes="ingest-file-section"):
                yield Static("Video File Selection", classes="sidebar-title")
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Select Video Files", id="ingest-local-video-select-files")
                    yield Button("Clear Selection", id="ingest-local-video-clear-files")
                yield Label("Selected Files:", classes="ingest-label")
                yield ListView(id="ingest-local-video-files-list", classes="ingest-selected-files-list")
            
            # Processing Options Section
            with Container(classes="ingest-options-section"):
                yield Static("Video Processing Options", classes="sidebar-title")
                yield Label("Transcription Model:")
                yield Input("deepdml/faster-whisper-large-v3-turbo-ct2", id="ingest-local-video-model")
                yield Label("Language (e.g., 'en'):")
                yield Input("en", id="ingest-local-video-language")
                yield Checkbox("Enable Speaker Diarization", False, id="ingest-local-video-diarize")
                yield Checkbox("Include Timestamps", True, id="ingest-local-video-timestamps")
                
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Start Time (HH:MM:SS):")
                        yield Input(id="ingest-local-video-start-time", placeholder="Optional")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("End Time (HH:MM:SS):")
                        yield Input(id="ingest-local-video-end-time", placeholder="Optional")
                
                with Collapsible(title="Advanced Options", collapsed=True):
                    yield Checkbox("Enable VAD (Voice Activity Detection)", False, id="ingest-local-video-vad")
                    yield Checkbox("Extract Key Frames", False, id="ingest-local-video-keyframes")
            
            # Metadata Section
            with Container(classes="ingest-metadata-section"):
                yield Static("Metadata", classes="sidebar-title")
                with Horizontal(classes="title-author-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Title Override:")
                        yield Input(id="ingest-local-video-title", placeholder="Optional")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Author:")
                        yield Input(id="ingest-local-video-author", placeholder="Optional")
                yield Label("Keywords (comma-separated):")
                yield TextArea(id="ingest-local-video-keywords", classes="ingest-textarea-small")
            
            # Action Section
            with Container(classes="ingest-action-section"):
                yield Button("Process Videos", id="ingest-local-video-process", variant="primary")
                yield LoadingIndicator(id="ingest-local-video-loading", classes="hidden")
                yield TextArea("", id="ingest-local-video-status", read_only=True, classes="ingest-status-area")

    def compose_local_audio_tab(self) -> ComposeResult:
        """Composes the Audio tab content for local media ingestion."""
        with VerticalScroll(classes="ingest-media-tab-content"):
            # File Selection Section
            with Container(classes="ingest-file-section"):
                yield Static("Audio File Selection", classes="sidebar-title")
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Select Audio Files", id="ingest-local-audio-select-files")
                    yield Button("Clear Selection", id="ingest-local-audio-clear-files")
                yield Label("Selected Files:", classes="ingest-label")
                yield ListView(id="ingest-local-audio-files-list", classes="ingest-selected-files-list")
            
            # Processing Options Section
            with Container(classes="ingest-options-section"):
                yield Static("Audio Processing Options", classes="sidebar-title")
                yield Label("Transcription Model:")
                yield Input("deepdml/faster-distil-whisper-large-v3.5", id="ingest-local-audio-model")
                yield Label("Language (e.g., 'en'):")
                yield Input("en", id="ingest-local-audio-language")
                yield Checkbox("Enable Speaker Diarization", False, id="ingest-local-audio-diarize")
                yield Checkbox("Include Timestamps", True, id="ingest-local-audio-timestamps")
                
                with Collapsible(title="Advanced Options", collapsed=True):
                    yield Checkbox("Enable VAD (Voice Activity Detection)", False, id="ingest-local-audio-vad")
                    yield Checkbox("Noise Reduction", False, id="ingest-local-audio-noise-reduction")
            
            # Metadata Section
            with Container(classes="ingest-metadata-section"):
                yield Static("Metadata", classes="sidebar-title")
                with Horizontal(classes="title-author-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Title Override:")
                        yield Input(id="ingest-local-audio-title", placeholder="Optional")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Author/Artist:")
                        yield Input(id="ingest-local-audio-author", placeholder="Optional")
                yield Label("Keywords (comma-separated):")
                yield TextArea(id="ingest-local-audio-keywords", classes="ingest-textarea-small")
            
            # Action Section
            with Container(classes="ingest-action-section"):
                yield Button("Process Audio", id="ingest-local-audio-process", variant="primary")
                yield LoadingIndicator(id="ingest-local-audio-loading", classes="hidden")
                yield TextArea("", id="ingest-local-audio-status", read_only=True, classes="ingest-status-area")

    def compose_local_document_tab(self) -> ComposeResult:
        """Composes the Document tab content for local media ingestion."""
        with VerticalScroll(classes="ingest-media-tab-content"):
            # File Selection Section
            with Container(classes="ingest-file-section"):
                yield Static("Document File Selection", classes="sidebar-title")
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Select Document Files", id="ingest-local-document-select-files")
                    yield Button("Clear Selection", id="ingest-local-document-clear-files")
                yield Label("Selected Files:", classes="ingest-label")
                yield ListView(id="ingest-local-document-files-list", classes="ingest-selected-files-list")
            
            # Processing Options Section
            with Container(classes="ingest-options-section"):
                yield Static("Document Processing Options", classes="sidebar-title")
                yield Label("Document Parser:")
                yield Select(
                    [("Auto-detect", "auto"), ("Python-docx", "docx"), ("LibreOffice", "libreoffice")],
                    id="ingest-local-document-parser",
                    value="auto"
                )
                yield Checkbox("Extract Metadata", True, id="ingest-local-document-extract-metadata")
                yield Checkbox("Preserve Formatting", True, id="ingest-local-document-preserve-formatting")
                
                with Collapsible(title="Advanced Options", collapsed=True):
                    yield Checkbox("Extract Images", False, id="ingest-local-document-extract-images")
                    yield Checkbox("Extract Tables as Structured Data", False, id="ingest-local-document-extract-tables")
            
            # Metadata Section
            with Container(classes="ingest-metadata-section"):
                yield Static("Metadata", classes="sidebar-title")
                with Horizontal(classes="title-author-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Title Override:")
                        yield Input(id="ingest-local-document-title", placeholder="Optional")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Author:")
                        yield Input(id="ingest-local-document-author", placeholder="Optional")
                yield Label("Keywords (comma-separated):")
                yield TextArea(id="ingest-local-document-keywords", classes="ingest-textarea-small")
            
            # Action Section
            with Container(classes="ingest-action-section"):
                yield Button("Process Documents", id="ingest-local-document-process", variant="primary")
                yield LoadingIndicator(id="ingest-local-document-loading", classes="hidden")
                yield TextArea("", id="ingest-local-document-status", read_only=True, classes="ingest-status-area")

    def compose_local_pdf_tab(self) -> ComposeResult:
        """Composes the PDF tab content for local media ingestion."""
        with VerticalScroll(classes="ingest-media-tab-content"):
            # File Selection Section
            with Container(classes="ingest-file-section"):
                yield Static("PDF File Selection", classes="sidebar-title")
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Select PDF Files", id="ingest-local-pdf-select-files")
                    yield Button("Clear Selection", id="ingest-local-pdf-clear-files")
                yield Label("Selected Files:", classes="ingest-label")
                yield ListView(id="ingest-local-pdf-files-list", classes="ingest-selected-files-list")
            
            # Processing Options Section
            with Container(classes="ingest-options-section"):
                yield Static("PDF Processing Options", classes="sidebar-title")
                yield Label("PDF Engine:")
                yield Select(
                    [("PyMuPDF4LLM", "pymupdf4llm"), ("PDFPlumber", "pdfplumber"), ("PyPDF2", "pypdf2")],
                    id="ingest-local-pdf-engine",
                    value="pymupdf4llm"
                )
                yield Checkbox("Extract Images", False, id="ingest-local-pdf-extract-images")
                yield Checkbox("OCR for Scanned PDFs", True, id="ingest-local-pdf-ocr")
                
                with Collapsible(title="Advanced Options", collapsed=True):
                    yield Checkbox("Extract Tables", True, id="ingest-local-pdf-extract-tables")
                    yield Checkbox("Extract Annotations", False, id="ingest-local-pdf-extract-annotations")
                    yield Label("Page Range (e.g., 1-10,15,20-25):")
                    yield Input(id="ingest-local-pdf-page-range", placeholder="All pages")
            
            # Metadata Section
            with Container(classes="ingest-metadata-section"):
                yield Static("Metadata", classes="sidebar-title")
                with Horizontal(classes="title-author-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Title Override:")
                        yield Input(id="ingest-local-pdf-title", placeholder="Optional")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Author:")
                        yield Input(id="ingest-local-pdf-author", placeholder="Optional")
                yield Label("Keywords (comma-separated):")
                yield TextArea(id="ingest-local-pdf-keywords", classes="ingest-textarea-small")
            
            # Action Section
            with Container(classes="ingest-action-section"):
                yield Button("Process PDFs", id="ingest-local-pdf-process", variant="primary")
                yield LoadingIndicator(id="ingest-local-pdf-loading", classes="hidden")
                yield TextArea("", id="ingest-local-pdf-status", read_only=True, classes="ingest-status-area")

    def compose_local_ebook_tab(self) -> ComposeResult:
        """Composes the Ebook tab content for local media ingestion."""
        with VerticalScroll(classes="ingest-media-tab-content"):
            # File Selection Section
            with Container(classes="ingest-file-section"):
                yield Static("Ebook File Selection", classes="sidebar-title")
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Select Ebook Files", id="ingest-local-ebook-select-files")
                    yield Button("Clear Selection", id="ingest-local-ebook-clear-files")
                yield Label("Selected Files:", classes="ingest-label")
                yield ListView(id="ingest-local-ebook-files-list", classes="ingest-selected-files-list")
            
            # Processing Options Section
            with Container(classes="ingest-options-section"):
                yield Static("Ebook Processing Options", classes="sidebar-title")
                yield Label("Extraction Method:")
                yield Select(
                    [("Filtered", "filtered"), ("Markdown", "markdown"), ("Basic", "basic")],
                    id="ingest-local-ebook-extraction",
                    value="filtered"
                )
                yield Checkbox("Extract Cover Image", True, id="ingest-local-ebook-extract-cover")
                yield Checkbox("Extract Table of Contents", True, id="ingest-local-ebook-extract-toc")
                
                with Collapsible(title="Advanced Options", collapsed=True):
                    yield Checkbox("Chapter Detection", True, id="ingest-local-ebook-chapter-detection")
                    yield Checkbox("Extract Metadata", True, id="ingest-local-ebook-extract-metadata")
                    yield Label("Chapter Split Pattern:")
                    yield Input(id="ingest-local-ebook-chapter-pattern", placeholder="Auto-detect")
            
            # Metadata Section
            with Container(classes="ingest-metadata-section"):
                yield Static("Metadata", classes="sidebar-title")
                with Horizontal(classes="title-author-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Title Override:")
                        yield Input(id="ingest-local-ebook-title", placeholder="Optional")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Author:")
                        yield Input(id="ingest-local-ebook-author", placeholder="Optional")
                yield Label("Keywords (comma-separated):")
                yield TextArea(id="ingest-local-ebook-keywords", classes="ingest-textarea-small")
            
            # Action Section
            with Container(classes="ingest-action-section"):
                yield Button("Process Ebooks", id="ingest-local-ebook-process", variant="primary")
                yield LoadingIndicator(id="ingest-local-ebook-loading", classes="hidden")
                yield TextArea("", id="ingest-local-ebook-status", read_only=True, classes="ingest-status-area")

    def compose_local_web_article_tab(self) -> ComposeResult:
        """Composes the Web Article tab content for local media ingestion."""
        with VerticalScroll(classes="ingest-media-tab-content"):
            # URL Input Section (instead of file selection)
            with Container(classes="ingest-file-section"):
                yield Static("Web Article URLs", classes="sidebar-title")
                yield Label("Enter URLs (one per line):")
                yield TextArea(id="ingest-local-web-urls", classes="ingest-textarea-medium")
                yield Button("Clear URLs", id="ingest-local-web-clear-urls")
            
            # Processing Options Section
            with Container(classes="ingest-options-section"):
                yield Static("Web Scraping Options", classes="sidebar-title")
                yield Checkbox("Extract Main Content Only", True, id="ingest-local-web-main-content")
                yield Checkbox("Include Images", False, id="ingest-local-web-include-images")
                yield Checkbox("Follow Redirects", True, id="ingest-local-web-follow-redirects")
                
                with Collapsible(title="Authentication Options", collapsed=True):
                    yield Label("Cookie String (optional):")
                    yield Input(id="ingest-local-web-cookies", placeholder="name=value; name2=value2")
                    yield Label("User Agent:")
                    yield Input(id="ingest-local-web-user-agent", placeholder="Default browser agent")
                
                with Collapsible(title="Advanced Options", collapsed=True):
                    yield Label("CSS Selector for Content:")
                    yield Input(id="ingest-local-web-css-selector", placeholder="Auto-detect")
                    yield Checkbox("JavaScript Rendering", False, id="ingest-local-web-js-render")
                    yield Label("Wait Time (seconds):")
                    yield Input("3", id="ingest-local-web-wait-time", type="integer")
            
            # Metadata Section
            with Container(classes="ingest-metadata-section"):
                yield Static("Metadata", classes="sidebar-title")
                with Horizontal(classes="title-author-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Title Override:")
                        yield Input(id="ingest-local-web-title", placeholder="Use page title")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Author Override:")
                        yield Input(id="ingest-local-web-author", placeholder="Extract from page")
                yield Label("Keywords (comma-separated):")
                yield TextArea(id="ingest-local-web-keywords", classes="ingest-textarea-small")
            
            # Action Section
            with Container(classes="ingest-action-section"):
                yield Button("Scrape Articles", id="ingest-local-web-process", variant="primary")
                yield LoadingIndicator(id="ingest-local-web-loading", classes="hidden")
                yield TextArea("", id="ingest-local-web-status", read_only=True, classes="ingest-status-area")

    def compose_local_xml_tab(self) -> ComposeResult:
        """Composes the XML tab content for local media ingestion."""
        with VerticalScroll(classes="ingest-media-tab-content"):
            # File Selection Section
            with Container(classes="ingest-file-section"):
                yield Static("XML File Selection", classes="sidebar-title")
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Select XML Files", id="ingest-local-xml-select-files")
                    yield Button("Clear Selection", id="ingest-local-xml-clear-files")
                yield Label("Selected Files:", classes="ingest-label")
                yield ListView(id="ingest-local-xml-files-list", classes="ingest-selected-files-list")
            
            # Processing Options Section
            with Container(classes="ingest-options-section"):
                yield Static("XML Processing Options", classes="sidebar-title")
                yield Checkbox("Validate Against Schema", False, id="ingest-local-xml-validate")
                yield Checkbox("Pretty Print Output", True, id="ingest-local-xml-pretty-print")
                
                with Collapsible(title="XPath Extraction", collapsed=True):
                    yield Label("Content XPath (e.g., //article/content):")
                    yield Input(id="ingest-local-xml-content-xpath", placeholder="Extract all text")
                    yield Label("Title XPath:")
                    yield Input(id="ingest-local-xml-title-xpath", placeholder="//title")
                    yield Label("Author XPath:")
                    yield Input(id="ingest-local-xml-author-xpath", placeholder="//author")
                
                with Collapsible(title="Namespace Handling", collapsed=True):
                    yield Label("Namespace Prefixes (prefix=uri, comma-separated):")
                    yield TextArea(id="ingest-local-xml-namespaces", classes="ingest-textarea-small")
            
            # Metadata Section
            with Container(classes="ingest-metadata-section"):
                yield Static("Metadata", classes="sidebar-title")
                with Horizontal(classes="title-author-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Title Override:")
                        yield Input(id="ingest-local-xml-title", placeholder="Optional")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Author:")
                        yield Input(id="ingest-local-xml-author", placeholder="Optional")
                yield Label("Keywords (comma-separated):")
                yield TextArea(id="ingest-local-xml-keywords", classes="ingest-textarea-small")
            
            # Action Section
            with Container(classes="ingest-action-section"):
                yield Button("Process XML", id="ingest-local-xml-process", variant="primary")
                yield LoadingIndicator(id="ingest-local-xml-loading", classes="hidden")
                yield TextArea("", id="ingest-local-xml-status", read_only=True, classes="ingest-status-area")

    def compose_local_plaintext_tab(self) -> ComposeResult:
        """Composes the Plaintext tab content for local media ingestion."""
        with VerticalScroll(classes="ingest-media-tab-content"):
            # File Selection Section
            with Container(classes="ingest-file-section"):
                yield Static("Text File Selection", classes="sidebar-title")
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Select Text Files", id="ingest-local-plaintext-select-files")
                    yield Button("Clear Selection", id="ingest-local-plaintext-clear-files")
                yield Label("Selected Files:", classes="ingest-label")
                yield ListView(id="ingest-local-plaintext-files-list", classes="ingest-selected-files-list")
            
            # Processing Options Section
            with Container(classes="ingest-options-section"):
                yield Static("Text Processing Options", classes="sidebar-title")
                yield Label("Encoding:")
                yield Select(
                    [("UTF-8", "utf-8"), ("ASCII", "ascii"), ("Latin-1", "latin-1"), ("Auto-detect", "auto")],
                    id="ingest-local-plaintext-encoding",
                    value="utf-8"
                )
                yield Label("Line Ending:")
                yield Select(
                    [("Auto", "auto"), ("Unix (LF)", "lf"), ("Windows (CRLF)", "crlf")],
                    id="ingest-local-plaintext-line-ending",
                    value="auto"
                )
                
                with Collapsible(title="Text Processing", collapsed=True):
                    yield Checkbox("Remove Extra Whitespace", True, id="ingest-local-plaintext-remove-whitespace")
                    yield Checkbox("Convert to Paragraphs", False, id="ingest-local-plaintext-paragraphs")
                    yield Label("Split Pattern (regex):")
                    yield Input(id="ingest-local-plaintext-split-pattern", placeholder="Empty for no splitting")
            
            # Metadata Section
            with Container(classes="ingest-metadata-section"):
                yield Static("Metadata", classes="sidebar-title")
                with Horizontal(classes="title-author-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Title Override:")
                        yield Input(id="ingest-local-plaintext-title", placeholder="Use filename")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Author:")
                        yield Input(id="ingest-local-plaintext-author", placeholder="Optional")
                yield Label("Keywords (comma-separated):")
                yield TextArea(id="ingest-local-plaintext-keywords", classes="ingest-textarea-small")
            
            # Action Section
            with Container(classes="ingest-action-section"):
                yield Button("Process Text Files", id="ingest-local-plaintext-process", variant="primary")
                yield LoadingIndicator(id="ingest-local-plaintext-loading", classes="hidden")
                yield TextArea("", id="ingest-local-plaintext-status", read_only=True, classes="ingest-status-area")

#
# End of Logs_Window.py
#######################################################################################################################
