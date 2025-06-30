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
    "ingest-view-prompts", "ingest-view-characters", "ingest-view-notes",
    # Local media types
    "ingest-view-local-video", "ingest-view-local-audio", "ingest-view-local-document",
    "ingest-view-local-pdf", "ingest-view-local-ebook", "ingest-view-local-web",
    "ingest-view-local-xml", "ingest-view-local-plaintext",
    # tldw API media types
    "ingest-view-api-video", "ingest-view-api-audio", "ingest-view-api-document",
    "ingest-view-api-pdf", "ingest-view-api-ebook", "ingest-view-api-xml",
    "ingest-view-api-mediawiki"
]
INGEST_NAV_BUTTON_IDS = [
    "ingest-nav-prompts", "ingest-nav-characters", "ingest-nav-notes",
    # Local media types
    "ingest-nav-local-video", "ingest-nav-local-audio", "ingest-nav-local-document",
    "ingest-nav-local-pdf", "ingest-nav-local-ebook", "ingest-nav-local-web",
    "ingest-nav-local-xml", "ingest-nav-local-plaintext",
    # tldw API media types
    "ingest-nav-api-video", "ingest-nav-api-audio", "ingest-nav-api-document",
    "ingest-nav-api-pdf", "ingest-nav-api-ebook", "ingest-nav-api-xml",
    "ingest-nav-api-mediawiki"
]

class IngestWindow(Container):
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = {}  # Stores {media_type: [Path, ...]}
        self._current_media_type_for_file_dialog = None # Stores the media_type for the active file dialog
        logger.debug("IngestWindow initialized.")
    
    def on_mount(self) -> None:
        """Handle initial mount to ensure views are properly hidden."""
        logger.debug("IngestWindow mounted, initializing view states")
        
        # Ensure all views start hidden
        try:
            content_pane = self.query_one("#ingest-content-pane")
            for child in content_pane.children:
                if child.id and child.id.startswith("ingest-view-"):
                    child.styles.display = "none"
                    logger.debug(f"Initially hiding view: {child.id}")
            
            # The default view will be set by the reactive watcher
            logger.debug("All ingest views hidden, waiting for reactive watcher to set default")
        except QueryError as e:
            logger.error(f"Error during IngestWindow mount: {e}")
    
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
            yield Static("Basic Ingestion", classes="sidebar-title")
            yield Button("Ingest Prompts", id="ingest-nav-prompts", classes="ingest-nav-button")
            yield Button("Ingest Characters", id="ingest-nav-characters", classes="ingest-nav-button")
            yield Button("Ingest Notes", id="ingest-nav-notes", classes="ingest-nav-button")
            
            yield Static("Local Media Ingestion", classes="sidebar-title")
            yield Button("Video (Local)", id="ingest-nav-local-video", classes="ingest-nav-button")
            yield Button("Audio (Local)", id="ingest-nav-local-audio", classes="ingest-nav-button")
            yield Button("Document (Local)", id="ingest-nav-local-document", classes="ingest-nav-button")
            yield Button("PDF (Local)", id="ingest-nav-local-pdf", classes="ingest-nav-button")
            yield Button("Ebook (Local)", id="ingest-nav-local-ebook", classes="ingest-nav-button")
            yield Button("Web Article (Local)", id="ingest-nav-local-web", classes="ingest-nav-button")
            yield Button("XML (Local)", id="ingest-nav-local-xml", classes="ingest-nav-button")
            yield Button("Plaintext (Local)", id="ingest-nav-local-plaintext", classes="ingest-nav-button")
            
            yield Static("TLDW API Ingestion", classes="sidebar-title")
            yield Button("Video (API)", id="ingest-nav-api-video", classes="ingest-nav-button")
            yield Button("Audio (API)", id="ingest-nav-api-audio", classes="ingest-nav-button")
            yield Button("Document (API)", id="ingest-nav-api-document", classes="ingest-nav-button")
            yield Button("PDF (API)", id="ingest-nav-api-pdf", classes="ingest-nav-button")
            yield Button("Ebook (API)", id="ingest-nav-api-ebook", classes="ingest-nav-button")
            yield Button("XML (API)", id="ingest-nav-api-xml", classes="ingest-nav-button")
            yield Button("MediaWiki Dump (API)", id="ingest-nav-api-mediawiki", classes="ingest-nav-button")


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

            # --- Local Media Views ---
            with Vertical(id="ingest-view-local-video", classes="ingest-view-area"):
                yield from self.compose_local_video_tab()
                
            with Vertical(id="ingest-view-local-audio", classes="ingest-view-area"):
                yield from self.compose_local_audio_tab()
                
            with Vertical(id="ingest-view-local-document", classes="ingest-view-area"):
                yield from self.compose_local_document_tab()
                
            with Vertical(id="ingest-view-local-pdf", classes="ingest-view-area"):
                yield from self.compose_local_pdf_tab()
                
            with Vertical(id="ingest-view-local-ebook", classes="ingest-view-area"):
                yield from self.compose_local_ebook_tab()
                
            with Vertical(id="ingest-view-local-web", classes="ingest-view-area"):
                yield from self.compose_local_web_article_tab()
                
            with Vertical(id="ingest-view-local-xml", classes="ingest-view-area"):
                yield from self.compose_local_xml_tab()
                
            with Vertical(id="ingest-view-local-plaintext", classes="ingest-view-area"):
                yield from self.compose_local_plaintext_tab()
            
            # --- TLDW API Views ---
            with Vertical(id="ingest-view-api-video", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("video")
                
            with Vertical(id="ingest-view-api-audio", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("audio")
                
            with Vertical(id="ingest-view-api-document", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("document")
                
            with Vertical(id="ingest-view-api-pdf", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("pdf")
                
            with Vertical(id="ingest-view-api-ebook", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("ebook")
                
            with Vertical(id="ingest-view-api-xml", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("xml")
                
            with Vertical(id="ingest-view-api-mediawiki", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("mediawiki_dump")

    def compose_tldw_api_view(self, media_type: str) -> ComposeResult:
        """Compose a TLDW API view for a specific media type."""
        # Create a temporary instance of IngestTldwApiTabbedWindow to reuse its form composition
        temp_window = IngestTldwApiTabbedWindow(self.app_instance)
        yield from temp_window.compose_tldw_api_form(media_type)

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
                yield Static("Transcription Options", classes="sidebar-title")
                yield Label("Transcription Model:")
                yield Input("deepdml/faster-whisper-large-v3-turbo-ct2", id="ingest-local-video-transcription-model")
                yield Label("Transcription Language (e.g., 'en'):")
                yield Input("en", id="ingest-local-video-transcription-language")
                yield Checkbox("Enable Speaker Diarization", False, id="ingest-local-video-diarize")
                yield Checkbox("Include Timestamps", True, id="ingest-local-video-timestamp-option")
                yield Checkbox("Enable VAD (Voice Activity Detection)", False, id="ingest-local-video-vad-use")
                
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Start Time (HH:MM:SS or seconds):")
                        yield Input(id="ingest-local-video-start-time", placeholder="Optional")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("End Time (HH:MM:SS or seconds):")
                        yield Input(id="ingest-local-video-end-time", placeholder="Optional")
            
            # Analysis Options
            with Container(classes="ingest-options-section"):
                yield Static("Analysis Options", classes="sidebar-title")
                yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="ingest-local-video-perform-analysis")
                yield Checkbox("Perform Confabulation Check", False, id="ingest-local-video-perform-confabulation-check")
                yield Label("Custom Prompt (for analysis):")
                yield TextArea(id="ingest-local-video-custom-prompt", classes="ingest-textarea-medium")
                yield Label("System Prompt (for analysis):")
                yield TextArea(id="ingest-local-video-system-prompt", classes="ingest-textarea-medium")
                
                with Collapsible(title="Advanced Analysis", collapsed=True):
                    yield Checkbox("Summarize Recursively", False, id="ingest-local-video-summarize-recursively")
                    yield Checkbox("Perform Rolling Summarization", False, id="ingest-local-video-perform-rolling-summarization")
            
            # Chunking Options
            with Collapsible(title="Chunking Options", collapsed=True):
                yield Checkbox("Perform Chunking", True, id="ingest-local-video-perform-chunking")
                yield Label("Chunk Method:")
                yield Select(
                    [("Semantic", "semantic"), ("Tokens", "tokens"), ("Sentences", "sentences"), 
                     ("Words", "words"), ("Paragraphs", "paragraphs")],
                    id="ingest-local-video-chunk-method",
                    prompt="Select chunking method..."
                )
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("500", id="ingest-local-video-chunk-size", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id="ingest-local-video-chunk-overlap", type="integer")
                yield Checkbox("Use Adaptive Chunking", False, id="ingest-local-video-use-adaptive-chunking")
                yield Checkbox("Use Multi-level Chunking", False, id="ingest-local-video-use-multi-level-chunking")
            
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
            
            # Database Options
            yield Checkbox("Overwrite if exists in database", False, id="ingest-local-video-overwrite-existing")
            
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
                yield Static("Transcription Options", classes="sidebar-title")
                yield Label("Transcription Model:")
                yield Input("deepdml/faster-distil-whisper-large-v3.5", id="ingest-local-audio-transcription-model")
                yield Label("Transcription Language (e.g., 'en'):")
                yield Input("en", id="ingest-local-audio-transcription-language")
                yield Checkbox("Enable Speaker Diarization", False, id="ingest-local-audio-diarize")
                yield Checkbox("Include Timestamps", True, id="ingest-local-audio-timestamp-option")
                yield Checkbox("Enable VAD (Voice Activity Detection)", False, id="ingest-local-audio-vad-use")
            
            # Analysis Options
            with Container(classes="ingest-options-section"):
                yield Static("Analysis Options", classes="sidebar-title")
                yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="ingest-local-audio-perform-analysis")
                yield Label("Custom Prompt (for analysis):")
                yield TextArea(id="ingest-local-audio-custom-prompt", classes="ingest-textarea-medium")
                yield Label("System Prompt (for analysis):")
                yield TextArea(id="ingest-local-audio-system-prompt", classes="ingest-textarea-medium")
                
                with Collapsible(title="Advanced Analysis", collapsed=True):
                    yield Checkbox("Summarize Recursively", False, id="ingest-local-audio-summarize-recursively")
                    yield Checkbox("Perform Rolling Summarization", False, id="ingest-local-audio-perform-rolling-summarization")
            
            # Chunking Options
            with Collapsible(title="Chunking Options", collapsed=True):
                yield Checkbox("Perform Chunking", True, id="ingest-local-audio-perform-chunking")
                yield Label("Chunk Method:")
                yield Select(
                    [("Semantic", "semantic"), ("Tokens", "tokens"), ("Sentences", "sentences"), 
                     ("Words", "words"), ("Paragraphs", "paragraphs")],
                    id="ingest-local-audio-chunk-method",
                    prompt="Select chunking method..."
                )
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("500", id="ingest-local-audio-chunk-size", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id="ingest-local-audio-chunk-overlap", type="integer")
                yield Checkbox("Use Adaptive Chunking", False, id="ingest-local-audio-use-adaptive-chunking")
                yield Checkbox("Use Multi-level Chunking", False, id="ingest-local-audio-use-multi-level-chunking")
            
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
            
            # Database Options
            yield Checkbox("Overwrite if exists in database", False, id="ingest-local-audio-overwrite-existing")
            
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
            
            # Analysis Options
            with Container(classes="ingest-options-section"):
                yield Static("Analysis Options", classes="sidebar-title")
                yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="ingest-local-document-perform-analysis")
                yield Label("Custom Prompt (for analysis):")
                yield TextArea(id="ingest-local-document-custom-prompt", classes="ingest-textarea-medium")
                yield Label("System Prompt (for analysis):")
                yield TextArea(id="ingest-local-document-system-prompt", classes="ingest-textarea-medium")
                
                with Collapsible(title="Advanced Analysis", collapsed=True):
                    yield Checkbox("Summarize Recursively", False, id="ingest-local-document-summarize-recursively")
                    yield Checkbox("Perform Rolling Summarization", False, id="ingest-local-document-perform-rolling-summarization")
            
            # Chunking Options
            with Collapsible(title="Chunking Options", collapsed=True):
                yield Checkbox("Perform Chunking", True, id="ingest-local-document-perform-chunking")
                yield Label("Chunk Method:")
                yield Select(
                    [("Sentences", "sentences"), ("Semantic", "semantic"), ("Tokens", "tokens"), 
                     ("Words", "words"), ("Paragraphs", "paragraphs")],
                    id="ingest-local-document-chunk-method",
                    value="sentences"
                )
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("1000", id="ingest-local-document-chunk-size", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id="ingest-local-document-chunk-overlap", type="integer")
                yield Checkbox("Use Adaptive Chunking", False, id="ingest-local-document-use-adaptive-chunking")
                yield Checkbox("Use Multi-level Chunking", False, id="ingest-local-document-use-multi-level-chunking")
            
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
            
            # Database Options
            yield Checkbox("Overwrite if exists in database", False, id="ingest-local-document-overwrite-existing")
            
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
                yield Label("PDF Parsing Engine:")
                yield Select(
                    [("PyMuPDF4LLM", "pymupdf4llm"), ("PyMuPDF", "pymupdf"), ("Docling", "docling")],
                    id="ingest-local-pdf-pdf-parsing-engine",
                    value="pymupdf4llm"
                )
            
            # Analysis Options
            with Container(classes="ingest-options-section"):
                yield Static("Analysis Options", classes="sidebar-title")
                yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="ingest-local-pdf-perform-analysis")
                yield Label("Custom Prompt (for analysis):")
                yield TextArea(id="ingest-local-pdf-custom-prompt", classes="ingest-textarea-medium")
                yield Label("System Prompt (for analysis):")
                yield TextArea(id="ingest-local-pdf-system-prompt", classes="ingest-textarea-medium")
                
                with Collapsible(title="Advanced Analysis", collapsed=True):
                    yield Checkbox("Summarize Recursively", False, id="ingest-local-pdf-summarize-recursively")
                    yield Checkbox("Perform Rolling Summarization", False, id="ingest-local-pdf-perform-rolling-summarization")
            
            # Chunking Options
            with Collapsible(title="Chunking Options", collapsed=True):
                yield Checkbox("Perform Chunking", True, id="ingest-local-pdf-perform-chunking")
                yield Label("Chunk Method:")
                yield Select(
                    [("Semantic", "semantic"), ("Tokens", "tokens"), ("Sentences", "sentences"), 
                     ("Words", "words"), ("Paragraphs", "paragraphs")],
                    id="ingest-local-pdf-chunk-method",
                    prompt="Select chunking method..."
                )
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("500", id="ingest-local-pdf-chunk-size", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id="ingest-local-pdf-chunk-overlap", type="integer")
                yield Checkbox("Use Adaptive Chunking", False, id="ingest-local-pdf-use-adaptive-chunking")
                yield Checkbox("Use Multi-level Chunking", False, id="ingest-local-pdf-use-multi-level-chunking")
            
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
            
            # Database Options
            yield Checkbox("Overwrite if exists in database", False, id="ingest-local-pdf-overwrite-existing")
            
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
                    id="ingest-local-ebook-extraction-method",
                    value="filtered"
                )
            
            # Analysis Options
            with Container(classes="ingest-options-section"):
                yield Static("Analysis Options", classes="sidebar-title")
                yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="ingest-local-ebook-perform-analysis")
                yield Label("Custom Prompt (for analysis):")
                yield TextArea(id="ingest-local-ebook-custom-prompt", classes="ingest-textarea-medium")
                yield Label("System Prompt (for analysis):")
                yield TextArea(id="ingest-local-ebook-system-prompt", classes="ingest-textarea-medium")
                
                with Collapsible(title="Advanced Analysis", collapsed=True):
                    yield Checkbox("Summarize Recursively", False, id="ingest-local-ebook-summarize-recursively")
                    yield Checkbox("Perform Rolling Summarization", False, id="ingest-local-ebook-perform-rolling-summarization")
            
            # Chunking Options
            with Collapsible(title="Chunking Options", collapsed=True):
                yield Checkbox("Perform Chunking", True, id="ingest-local-ebook-perform-chunking")
                yield Label("Chunk Method:")
                yield Select(
                    [("Ebook Chapters", "ebook_chapters"), ("Semantic", "semantic"), ("Tokens", "tokens"), 
                     ("Sentences", "sentences"), ("Words", "words"), ("Paragraphs", "paragraphs")],
                    id="ingest-local-ebook-chunk-method",
                    value="ebook_chapters"
                )
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("500", id="ingest-local-ebook-chunk-size", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id="ingest-local-ebook-chunk-overlap", type="integer")
                yield Checkbox("Use Adaptive Chunking", False, id="ingest-local-ebook-use-adaptive-chunking")
                yield Checkbox("Use Multi-level Chunking", False, id="ingest-local-ebook-use-multi-level-chunking")
                yield Label("Custom Chapter Pattern (Regex):")
                yield Input(id="ingest-local-ebook-custom-chapter-pattern", placeholder="e.g., ^Chapter\\s+\\d+")
            
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
            
            # Database Options
            yield Checkbox("Overwrite if exists in database", False, id="ingest-local-ebook-overwrite-existing")
            
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
                yield Checkbox("Auto Summarize", False, id="ingest-local-xml-auto-summarize")
            
            # Analysis Options (if auto_summarize is true)
            with Container(classes="ingest-options-section"):
                yield Static("Analysis Options", classes="sidebar-title")
                yield Label("Custom Prompt (for analysis):")
                yield TextArea(id="ingest-local-xml-custom-prompt", classes="ingest-textarea-medium")
                yield Label("System Prompt (for analysis):")
                yield TextArea(id="ingest-local-xml-system-prompt", classes="ingest-textarea-medium")
                
                yield Label("API Provider (for summarization):")
                analysis_api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
                analysis_provider_options = [(name, name) for name in analysis_api_providers if name]
                if not analysis_provider_options:
                    analysis_provider_options = [("No Providers Configured", "")]
                yield Select(
                    analysis_provider_options,
                    id="ingest-local-xml-api-name",
                    prompt="Select API for Analysis..."
                )
            
            # Metadata Section
            with Container(classes="ingest-metadata-section"):
                yield Static("Metadata", classes="sidebar-title")
                with Horizontal(classes="title-author-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Title:")
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
