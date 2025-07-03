# tldw_chatbook/UI/Ingest_Window.py
#
#
# Imports
from typing import TYPE_CHECKING, List, Dict, Any
from pathlib import Path
import asyncio
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.css.query import QueryError
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.widgets import Static, Button, Input, Select, Checkbox, TextArea, Label, RadioSet, RadioButton, Collapsible, ListView, ListItem, Markdown, LoadingIndicator, TabbedContent, TabPane # Button, ListView, ListItem, Label are already here
from textual.worker import Worker
from textual import work
from textual.reactive import reactive

# Configure logger with context
logger = logger.bind(module="Ingest_Window")

from ..Constants import TLDW_API_AUDIO_OPTIONS_ID, TLDW_API_VIDEO_OPTIONS_ID, TLDW_API_PDF_OPTIONS_ID, \
    TLDW_API_EBOOK_OPTIONS_ID, TLDW_API_DOCUMENT_OPTIONS_ID, TLDW_API_XML_OPTIONS_ID, TLDW_API_MEDIAWIKI_OPTIONS_ID
#
# Local Imports
from ..Third_Party.textual_fspicker.file_open import FileOpen
from ..tldw_api.schemas import MediaType, ChunkMethod, PdfEngine  # Import Enums
from ..Widgets.IngestTldwApiVideoWindow import IngestTldwApiVideoWindow
from ..Widgets.IngestTldwApiAudioWindow import IngestTldwApiAudioWindow
from ..Widgets.IngestTldwApiPdfWindow import IngestTldwApiPdfWindow
from ..Widgets.IngestTldwApiEbookWindow import IngestTldwApiEbookWindow
from ..Widgets.IngestTldwApiDocumentWindow import IngestTldwApiDocumentWindow
from ..Widgets.IngestTldwApiXmlWindow import IngestTldwApiXmlWindow
from ..Widgets.IngestTldwApiMediaWikiWindow import IngestTldwApiMediaWikiWindow
from ..Widgets.IngestTldwApiPlaintextWindow import IngestTldwApiPlaintextWindow
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

MEDIA_TYPES = ['video', 'audio', 'document', 'pdf', 'ebook', 'xml', 'mediawiki_dump', 'plaintext']

INGEST_VIEW_IDS = [
    "ingest-view-prompts", "ingest-view-characters", "ingest-view-notes",
    # Local media types
    "ingest-view-local-video", "ingest-view-local-audio", "ingest-view-local-document",
    "ingest-view-local-pdf", "ingest-view-local-ebook", "ingest-view-local-web",
    "ingest-view-local-xml", "ingest-view-local-plaintext", "ingest-view-subscriptions",
    # tldw API media types
    "ingest-view-api-video", "ingest-view-api-audio", "ingest-view-api-document",
    "ingest-view-api-pdf", "ingest-view-api-ebook", "ingest-view-api-xml",
    "ingest-view-api-mediawiki", "ingest-view-api-plaintext"
]
INGEST_NAV_BUTTON_IDS = [
    "ingest-nav-prompts", "ingest-nav-characters", "ingest-nav-notes",
    # Local media types
    "ingest-nav-local-video", "ingest-nav-local-audio", "ingest-nav-local-document",
    "ingest-nav-local-pdf", "ingest-nav-local-ebook", "ingest-nav-local-web",
    "ingest-nav-local-xml", "ingest-nav-local-plaintext", "ingest-nav-subscriptions",
    # tldw API media types
    "ingest-nav-api-video", "ingest-nav-api-audio", "ingest-nav-api-document",
    "ingest-nav-api-pdf", "ingest-nav-api-ebook", "ingest-nav-api-xml",
    "ingest-nav-api-mediawiki", "ingest-nav-api-plaintext"
]

class IngestWindow(Container):
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = {}  # Stores {media_type: [Path, ...]}
        self._current_media_type_for_file_dialog = None # Stores the media_type for the active file dialog
        self._failed_urls_for_retry = []  # Store failed URLs for retry
        self._retry_attempts = {}  # Track retry attempts per URL
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
    
    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle TextArea changes."""
        if event.text_area.id == "ingest-local-web-urls":
            # Update URL count when user types/pastes
            self._update_url_count()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        logger.debug(f"IngestWindow handling button press: {button_id}")
        
        # Local web article buttons
        if button_id == "ingest-local-web-clear-urls":
            await self._handle_clear_urls()
            event.stop()  # Prevent further propagation
        elif button_id == "ingest-local-web-import-urls":
            await self._handle_import_urls_from_file()
            event.stop()
        elif button_id == "ingest-local-web-remove-duplicates":
            await self._handle_remove_duplicate_urls()
            event.stop()
        elif button_id == "ingest-local-web-process":
            await self.handle_local_web_article_process()
            event.stop()
        elif button_id == "ingest-local-web-stop":
            await self._handle_stop_web_scraping()
            event.stop()
        elif button_id == "ingest-local-web-retry":
            await self._handle_retry_failed_urls()
            event.stop()
    
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
            yield Button("Subscriptions", id="ingest-nav-subscriptions", classes="ingest-nav-button")
            
            yield Static("TLDW API Ingestion", classes="sidebar-title")
            yield Button("Video (API)", id="ingest-nav-api-video", classes="ingest-nav-button")
            yield Button("Audio (API)", id="ingest-nav-api-audio", classes="ingest-nav-button")
            yield Button("Document (API)", id="ingest-nav-api-document", classes="ingest-nav-button")
            yield Button("PDF (API)", id="ingest-nav-api-pdf", classes="ingest-nav-button")
            yield Button("Ebook (API)", id="ingest-nav-api-ebook", classes="ingest-nav-button")
            yield Button("XML (API)", id="ingest-nav-api-xml", classes="ingest-nav-button")
            yield Button("MediaWiki Dump (API)", id="ingest-nav-api-mediawiki", classes="ingest-nav-button")
            yield Button("Plaintext (API)", id="ingest-nav-api-plaintext", classes="ingest-nav-button")


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
            
            with Vertical(id="ingest-view-subscriptions", classes="ingest-view-area"):
                yield from self.compose_subscriptions_tab()
            
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
                
            with Vertical(id="ingest-view-api-plaintext", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("plaintext")

    def compose_tldw_api_view(self, media_type: str) -> ComposeResult:
        """Compose a TLDW API view for a specific media type."""
        # Use individual window classes for each media type
        if media_type == "video":
            window = IngestTldwApiVideoWindow(self.app_instance)
        elif media_type == "audio":
            window = IngestTldwApiAudioWindow(self.app_instance)
        elif media_type == "pdf":
            window = IngestTldwApiPdfWindow(self.app_instance)
        elif media_type == "ebook":
            window = IngestTldwApiEbookWindow(self.app_instance)
        elif media_type == "document":
            window = IngestTldwApiDocumentWindow(self.app_instance)
        elif media_type == "xml":
            window = IngestTldwApiXmlWindow(self.app_instance)
        elif media_type == "mediawiki_dump":
            window = IngestTldwApiMediaWikiWindow(self.app_instance)
        elif media_type == "plaintext":
            window = IngestTldwApiPlaintextWindow(self.app_instance)
        else:
            logger.error(f"Unknown media type: {media_type}")
            yield Static(f"Error: Unknown media type '{media_type}'")
            return
        
        yield from window.compose()

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

            await self.app_instance.push_screen(
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
                
                await self.app_instance.push_screen(
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
                self._update_url_count()
                logger.info("Cleared web article URLs")
            except Exception as e:
                logger.error(f"Error clearing web URLs: {e}")
        
        # Handle web article import URLs button
        elif button_id == "ingest-local-web-import-urls":
            event.stop()
            await self._handle_import_urls_from_file()
        
        # Handle web article remove duplicates button
        elif button_id == "ingest-local-web-remove-duplicates":
            event.stop()
            await self._handle_remove_duplicate_urls()
        
        # Handle local plaintext process button
        elif button_id == "ingest-local-plaintext-process":
            event.stop()
            await self.handle_local_plaintext_process()
        
        # Handle web article process button
        elif button_id == "ingest-local-web-process":
            event.stop()
            await self.handle_local_web_article_process()
        
        # Handle web article stop button
        elif button_id == "ingest-local-web-stop":
            event.stop()
            await self._handle_stop_web_scraping()
        
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
                yield Checkbox("Perform Confabulation Check", False, id="ingest-local-video-perform-confabulation-check-of-analysis")
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
                with Horizontal(classes="ingest-controls-row"):
                    yield Button("Clear URLs", id="ingest-local-web-clear-urls")
                    yield Button("Import from File", id="ingest-local-web-import-urls")
                    yield Button("Remove Duplicates", id="ingest-local-web-remove-duplicates")
                yield Label("URL Count: 0 valid, 0 invalid", id="ingest-local-web-url-count", classes="ingest-label")
            
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
                
                # Progress section
                with Container(id="ingest-local-web-progress", classes="hidden"):
                    yield Static("Progress: 0/0", id="ingest-local-web-progress-text")
                    yield Static("✅ 0  ❌ 0  ⏳ 0", id="ingest-local-web-counters")
                
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

    async def handle_local_plaintext_process(self) -> None:
        """Handle processing of local plaintext files."""
        logger.info("Processing local plaintext files")
        
        # Get UI elements
        try:
            loading_indicator = self.query_one("#ingest-local-plaintext-loading", LoadingIndicator)
            status_area = self.query_one("#ingest-local-plaintext-status", TextArea)
            process_button = self.query_one("#ingest-local-plaintext-process", Button)
        except Exception as e:
            logger.error(f"Error finding UI elements: {e}")
            self.app_instance.notify("Error: UI elements not found", severity="error")
            return
        
        # Show loading state
        loading_indicator.display = True
        status_area.clear()
        status_area.load_text("Processing plaintext files...")
        status_area.display = True
        process_button.disabled = True
        
        try:
            # Get selected files
            local_key = "local_plaintext"
            selected_files = self.selected_local_files.get(local_key, [])
            
            if not selected_files:
                self.app_instance.notify("Please select at least one text file", severity="warning")
                return
            
            # Get processing options
            encoding_select = self.query_one("#ingest-local-plaintext-encoding", Select)
            encoding = str(encoding_select.value)
            
            line_ending_select = self.query_one("#ingest-local-plaintext-line-ending", Select)
            line_ending = str(line_ending_select.value)
            
            remove_whitespace = self.query_one("#ingest-local-plaintext-remove-whitespace", Checkbox).value
            convert_paragraphs = self.query_one("#ingest-local-plaintext-paragraphs", Checkbox).value
            split_pattern = self.query_one("#ingest-local-plaintext-split-pattern", Input).value.strip()
            
            # Get metadata
            title_override = self.query_one("#ingest-local-plaintext-title", Input).value.strip()
            author = self.query_one("#ingest-local-plaintext-author", Input).value.strip()
            keywords_text = self.query_one("#ingest-local-plaintext-keywords", TextArea).text.strip()
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()] if keywords_text else []
            
            # Check if media DB is available
            if not self.app_instance.media_db:
                logger.error("Media database not initialized")
                self.app_instance.notify("Error: Media database not available", severity="error")
                status_area.load_text("Error: Media database not available")
                return
            
            # Process each file
            processed_count = 0
            error_count = 0
            status_messages = []
            
            for file_path in selected_files:
                try:
                    # Read file content
                    content = await self._read_text_file(file_path, encoding)
                    
                    if content is None:
                        error_count += 1
                        status_messages.append(f"❌ Failed to read: {file_path.name}")
                        continue
                    
                    # Process content based on options
                    if line_ending != "auto":
                        content = self._normalize_line_endings(content, line_ending)
                    
                    if remove_whitespace:
                        content = self._remove_extra_whitespace(content)
                    
                    if convert_paragraphs:
                        content = self._convert_to_paragraphs(content)
                    
                    if split_pattern:
                        # For now, we'll just note this option exists
                        # Actual splitting would be handled by chunking
                        pass
                    
                    # Use filename as title if no override
                    title = title_override or file_path.stem
                    
                    # Add to media database
                    media_id, media_uuid, msg = self.app_instance.media_db.add_media_with_keywords(
                        url=str(file_path),
                        title=title,
                        media_type="plaintext",
                        content=content,
                        keywords=keywords,
                        author=author,
                        ingestion_date=None,  # Will use current time
                        overwrite=False
                    )
                    
                    if media_id:
                        processed_count += 1
                        status_messages.append(f"✅ Processed: {file_path.name} (ID: {media_id})")
                        logger.info(f"Successfully ingested plaintext file: {file_path}")
                    else:
                        error_count += 1
                        status_messages.append(f"❌ Failed to ingest: {file_path.name} - {msg}")
                        logger.error(f"Failed to ingest plaintext file: {file_path} - {msg}")
                        
                except Exception as e:
                    error_count += 1
                    status_messages.append(f"❌ Error processing {file_path.name}: {str(e)}")
                    logger.error(f"Error processing plaintext file {file_path}: {e}", exc_info=True)
            
            # Update status
            summary = f"## Processing Complete\n\n"
            summary += f"✅ Successfully processed: {processed_count} files\n"
            if error_count > 0:
                summary += f"❌ Errors: {error_count} files\n"
            summary += "\n### Details:\n"
            summary += "\n".join(status_messages)
            
            status_area.load_text(summary)
            
            if processed_count > 0:
                self.app_instance.notify(f"Successfully processed {processed_count} text files", severity="information")
            if error_count > 0:
                self.app_instance.notify(f"Failed to process {error_count} text files", severity="warning")
                
        except Exception as e:
            logger.error(f"Error in plaintext processing: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
            status_area.load_text(f"Error: {str(e)}")
        finally:
            # Reset UI state
            loading_indicator.display = False
            process_button.disabled = False
    
    async def handle_local_web_article_process(self) -> None:
        """Handle processing of web articles from URLs."""
        logger.info("Processing web articles")
        
        # Get UI elements
        try:
            loading_indicator = self.query_one("#ingest-local-web-loading", LoadingIndicator)
            status_area = self.query_one("#ingest-local-web-status", TextArea)
            process_button = self.query_one("#ingest-local-web-process", Button)
        except Exception as e:
            logger.error(f"Error finding UI elements: {e}")
            self.app_instance.notify("Error: UI elements not found", severity="error")
            return
        
        # Check if already processing
        if hasattr(self, '_web_scraping_worker') and self._web_scraping_worker and not self._web_scraping_worker.is_finished:
            self.app_instance.notify("Already processing URLs. Please wait or stop the current process.", severity="warning")
            return
        
        # Show loading state
        loading_indicator.display = True
        loading_indicator.classes = loading_indicator.classes - {"hidden"}
        status_area.clear()
        status_area.load_text("Starting web article scraping...")
        status_area.display = True
        process_button.disabled = True
        
        # Show progress container
        try:
            progress_container = self.query_one("#ingest-local-web-progress", Container)
            progress_container.classes = progress_container.classes - {"hidden"}
            
            # Initialize progress tracking
            self._current_progress = {
                'total': len(urls),
                'done': 0,
                'success': 0,
                'failed': 0,
                'pending': len(urls)
            }
            
            # Update initial progress display
            progress_text = self.query_one("#ingest-local-web-progress-text", Static)
            counters = self.query_one("#ingest-local-web-counters", Static)
            progress_text.update(f"Progress: 0/{len(urls)}")
            counters.update(f"✅ 0  ❌ 0  ⏳ {len(urls)}")
        except Exception as e:
            logger.error(f"Error showing progress container: {e}")
        
        try:
            # Get URLs from the textarea
            urls_textarea = self.query_one("#ingest-local-web-urls", TextArea)
            urls_text = urls_textarea.text.strip()
            
            if not urls_text:
                self.app_instance.notify("Please enter at least one URL", severity="warning")
                return
            
            # Split URLs by newline and filter empty lines
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            
            # Get scraping options
            main_content_only = self.query_one("#ingest-local-web-main-content", Checkbox).value
            include_images = self.query_one("#ingest-local-web-include-images", Checkbox).value
            follow_redirects = self.query_one("#ingest-local-web-follow-redirects", Checkbox).value
            
            # Get authentication options
            cookies_str = self.query_one("#ingest-local-web-cookies", Input).value.strip()
            user_agent = self.query_one("#ingest-local-web-user-agent", Input).value.strip()
            
            # Get advanced options
            css_selector = self.query_one("#ingest-local-web-css-selector", Input).value.strip()
            js_render = self.query_one("#ingest-local-web-js-render", Checkbox).value
            wait_time_str = self.query_one("#ingest-local-web-wait-time", Input).value.strip()
            wait_time = int(wait_time_str) if wait_time_str else 3
            
            # Get metadata
            title_override = self.query_one("#ingest-local-web-title", Input).value.strip()
            author_override = self.query_one("#ingest-local-web-author", Input).value.strip()
            keywords_text = self.query_one("#ingest-local-web-keywords", TextArea).text.strip()
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()] if keywords_text else []
            
            # Parse cookies if provided
            custom_cookies = None
            if cookies_str:
                try:
                    custom_cookies = self._parse_cookie_string(cookies_str)
                except Exception as e:
                    logger.warning(f"Failed to parse cookies: {e}")
                    self.app_instance.notify("Warning: Failed to parse cookies, continuing without them", severity="warning")
            
            # Check if media DB is available
            if not self.app_instance.media_db:
                logger.error("Media database not initialized")
                self.app_instance.notify("Error: Media database not available", severity="error")
                status_area.load_text("Error: Media database not available")
                return
            
            # Prepare worker data
            worker_data = {
                'urls': urls,
                'custom_cookies': custom_cookies,
                'title_override': title_override,
                'author_override': author_override,
                'keywords': keywords,
                'js_render': js_render,
                'css_selector': css_selector
            }
            
            # Add stop button
            stop_button = Button("Stop Processing", id="ingest-local-web-stop", variant="error")
            action_section = process_button.parent
            if action_section and not self.query("#ingest-local-web-stop"):
                action_section.mount(stop_button, after=process_button)
            
            # Start worker
            self._web_scraping_worker = self._process_urls_worker(worker_data)
            
            # Handle worker completion
            def on_worker_done(worker: Worker) -> None:
                """Handle worker completion."""
                if worker.cancelled:
                    self.app_instance.notify("Processing cancelled", severity="warning")
                    return
                    
                result = worker.result
                if not result:
                    self.app_instance.notify("No results from processing", severity="error")
                    self._cleanup_after_processing()
                    return
                
                processed_count = result['processed_count']
                error_count = result['error_count']
                failed_urls = result['failed_urls']
                
                # Update final status
                summary = f"\n## Processing Complete\n\n"
                summary += f"✅ Successfully processed: {processed_count} articles\n"
                if error_count > 0:
                    summary += f"❌ Errors: {error_count} articles\n"
                summary += "\n### Details:\n"
                
                # Show results
                for res in result['results'][-10:]:  # Last 10 results
                    if isinstance(res, dict):
                        if res['status'] == 'success':
                            summary += f"✅ {res['title']} - ID: {res['media_id']}\n"
                        else:
                            summary += f"❌ {res['url']} - {res['error']}\n"
                
                if len(result['results']) > 10:
                    summary += f"\n... and {len(result['results']) - 10} more"
                
                status_area.append(summary)
                
                # Show failed URLs section if any
                if failed_urls:
                    status_area.append("\n\n### Failed URLs for retry:\n")
                    for fail in failed_urls:
                        status_area.append(f"- {fail['url']} ({fail.get('error', 'Unknown error')})\n")
                    
                    # Store failed URLs for retry
                    self._failed_urls_for_retry = failed_urls
                    
                    # Add retry button
                    retry_button = Button(f"Retry {len(failed_urls)} Failed URLs", id="ingest-local-web-retry", variant="warning")
                    action_section = process_button.parent
                    if action_section and not self.query("#ingest-local-web-retry"):
                        action_section.mount(retry_button, after=process_button)
                
                # Notifications
                if processed_count > 0:
                    self.app_instance.notify(f"Successfully processed {processed_count} web articles", severity="information")
                if error_count > 0:
                    self.app_instance.notify(f"Failed to process {error_count} web articles", severity="warning")
                
                # Clean up UI
                self._cleanup_after_processing()
            
            # Add callback
            self._web_scraping_worker.add_done_callback(on_worker_done)
                
        except Exception as e:
            logger.error(f"Error in web article processing: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
            status_area.load_text(f"Error: {str(e)}")
        finally:
            # Reset UI state
            loading_indicator.display = False
            loading_indicator.classes = loading_indicator.classes | {"hidden"}
            process_button.disabled = False
    
    def _parse_cookie_string(self, cookie_str: str) -> List[Dict[str, Any]]:
        """Parse cookie string into format expected by playwright."""
        cookies = []
        # Simple cookie parsing - format: "name=value; name2=value2"
        for cookie_part in cookie_str.split(';'):
            cookie_part = cookie_part.strip()
            if '=' in cookie_part:
                name, value = cookie_part.split('=', 1)
                cookies.append({
                    'name': name.strip(),
                    'value': value.strip(),
                    'domain': '',  # Will be set by playwright based on URL
                    'path': '/'
                })
        return cookies
    
    def _validate_url(self, url: str) -> bool:
        """Basic URL validation."""
        import re
        # Basic URL pattern
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(url))
    
    def _update_url_count(self) -> None:
        """Update the URL count label based on current TextArea content."""
        try:
            urls_textarea = self.query_one("#ingest-local-web-urls", TextArea)
            url_count_label = self.query_one("#ingest-local-web-url-count", Label)
            
            urls_text = urls_textarea.text.strip()
            if not urls_text:
                url_count_label.update("URL Count: 0 valid, 0 invalid")
                return
            
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            valid_count = sum(1 for url in urls if self._validate_url(url))
            invalid_count = len(urls) - valid_count
            
            url_count_label.update(f"URL Count: {valid_count} valid, {invalid_count} invalid")
        except Exception as e:
            logger.error(f"Error updating URL count: {e}")
    
    async def _handle_clear_urls(self) -> None:
        """Handle clearing URLs."""
        try:
            urls_textarea = self.query_one("#ingest-local-web-urls", TextArea)
            urls_textarea.clear()
            self._update_url_count()
            self.app_instance.notify("URLs cleared", severity="information")
        except Exception as e:
            logger.error(f"Error clearing URLs: {e}")
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
    
    async def _handle_remove_duplicate_urls(self) -> None:
        """Handle removing duplicate URLs."""
        try:
            urls_textarea = self.query_one("#ingest-local-web-urls", TextArea)
            urls_text = urls_textarea.text.strip()
            
            if not urls_text:
                self.app_instance.notify("No URLs to process", severity="warning")
                return
            
            # Split URLs and remove duplicates while preserving order
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            seen = set()
            unique_urls = []
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            removed_count = len(urls) - len(unique_urls)
            
            if removed_count > 0:
                urls_textarea.text = '\n'.join(unique_urls)
                self._update_url_count()
                self.app_instance.notify(f"Removed {removed_count} duplicate URLs", severity="information")
            else:
                self.app_instance.notify("No duplicate URLs found", severity="information")
                
        except Exception as e:
            logger.error(f"Error removing duplicate URLs: {e}")
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
    
    async def _handle_import_urls_from_file(self) -> None:
        """Handle importing URLs from a file."""
        from ..Third_Party.textual_fspicker import FileOpen, Filters
        
        def handle_file_selected(file_path: Path | None) -> None:
            if file_path and file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    urls_textarea = self.query_one("#ingest-local-web-urls", TextArea)
                    
                    # Append to existing URLs
                    existing_text = urls_textarea.text.strip()
                    if existing_text:
                        urls_textarea.text = existing_text + '\n' + content
                    else:
                        urls_textarea.text = content
                    
                    self._update_url_count()
                    self.app_instance.notify(f"Imported URLs from {file_path.name}", severity="information")
                except Exception as e:
                    logger.error(f"Error importing URLs from file: {e}")
                    self.app_instance.notify(f"Error importing URLs: {str(e)}", severity="error")
        
        await self.app_instance.push_screen(
            FileOpen(
                title="Select URL List File",
                filters=Filters(
                    ("Text Files", lambda p: p.suffix.lower() in (".txt", ".csv")),
                    ("All Files", lambda _: True)
                )
            ),
            handle_file_selected
        )
    
    async def _handle_remove_duplicate_urls(self) -> None:
        """Remove duplicate URLs from the TextArea."""
        try:
            urls_textarea = self.query_one("#ingest-local-web-urls", TextArea)
            urls_text = urls_textarea.text.strip()
            
            if not urls_text:
                self.app_instance.notify("No URLs to process", severity="warning")
                return
            
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            removed_count = len(urls) - len(unique_urls)
            
            if removed_count > 0:
                urls_textarea.text = '\n'.join(unique_urls)
                self._update_url_count()
                self.app_instance.notify(f"Removed {removed_count} duplicate URLs", severity="information")
            else:
                self.app_instance.notify("No duplicate URLs found", severity="information")
                
        except Exception as e:
            logger.error(f"Error removing duplicate URLs: {e}")
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
    
    @work(thread=True)
    async def _process_urls_worker(self, data: dict) -> dict:
        """Worker to process URLs concurrently."""
        urls = data['urls']
        custom_cookies = data['custom_cookies']
        title_override = data['title_override']
        author_override = data['author_override']
        keywords = data['keywords']
        js_render = data['js_render']
        css_selector = data['css_selector']
        is_retry = data.get('is_retry', False)
        max_retries = data.get('max_retries', 2)
        
        # Import scraping function
        from tldw_chatbook.Web_Scraping.Article_Extractor_Lib import scrape_article
        
        processed_count = 0
        error_count = 0
        failed_urls = []
        results = []
        
        # Process URLs with limited concurrency
        max_concurrent = 3
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_url(idx: int, url: str, retry_count: int = 0) -> dict:
            async with semaphore:
                try:
                    # Track retry attempts
                    if url not in self._retry_attempts:
                        self._retry_attempts[url] = 0
                    
                    attempt_str = f" (Retry {retry_count}/{max_retries})" if retry_count > 0 else ""
                    
                    # Update progress
                    self.call_from_thread(
                        self._update_scraping_progress,
                        f"[{idx}/{len(urls)}] Scraping{attempt_str}: {url}"
                    )
                    
                    # Add exponential backoff for retries
                    if retry_count > 0:
                        wait_time = min(2 ** (retry_count - 1), 10)  # Max 10 seconds
                        await asyncio.sleep(wait_time)
                    
                    # Scrape the article
                    article_data = await scrape_article(url, custom_cookies=custom_cookies)
                    
                    if not article_data.get('extraction_successful', False):
                        return {
                            'url': url,
                            'status': 'failed',
                            'error': 'Extraction failed'
                        }
                    
                    # Override metadata if provided
                    title = title_override or article_data.get('title', url)
                    author = author_override or article_data.get('author', '')
                    content = article_data.get('content', '')
                    
                    if not content:
                        return {
                            'url': url,
                            'status': 'failed',
                            'error': 'No content found'
                        }
                    
                    # Add to media database
                    media_id, media_uuid, msg = self.app_instance.media_db.add_media_with_keywords(
                        url=url,
                        title=title,
                        media_type="web_article",
                        content=content,
                        keywords=keywords,
                        author=author,
                        metadata={
                            'publication_date': article_data.get('date'),
                            'extraction_method': 'trafilatura',
                            'js_rendered': js_render,
                            'custom_selector': css_selector
                        }
                    )
                    
                    if media_id:
                        return {
                            'url': url,
                            'status': 'success',
                            'title': title,
                            'media_id': media_id
                        }
                    else:
                        return {
                            'url': url,
                            'status': 'failed',
                            'error': f'Database error: {msg}'
                        }
                    
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}", exc_info=True)
                    error_msg = str(e)
                    
                    # Check if we should retry
                    self._retry_attempts[url] = retry_count + 1
                    if retry_count < max_retries and not is_retry:
                        # Automatic retry with backoff
                        self.call_from_thread(
                            self._update_scraping_progress,
                            f"[{idx}/{len(urls)}] Retrying {url} after error: {error_msg}"
                        )
                        return await process_single_url(idx, url, retry_count + 1)
                    
                    return {
                        'url': url,
                        'status': 'failed',
                        'error': error_msg,
                        'retry_count': self._retry_attempts.get(url, 0)
                    }
        
        # Create tasks for all URLs
        # If this is a retry, preserve retry counts
        tasks = []
        for idx, url in enumerate(urls):
            if is_retry and isinstance(url, dict):
                # URL from failed_urls list with retry info
                retry_count = url.get('retry_count', 0)
                tasks.append(process_single_url(idx + 1, url['url'], retry_count))
            else:
                # Normal URL string
                tasks.append(process_single_url(idx + 1, url))
        
        # Process all URLs concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count results
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
            elif isinstance(result, dict):
                if result['status'] == 'success':
                    processed_count += 1
                else:
                    error_count += 1
                    failed_urls.append(result)
        
        return {
            'processed_count': processed_count,
            'error_count': error_count,
            'failed_urls': failed_urls,
            'results': results
        }
    
    def _update_scraping_progress(self, message: str) -> None:
        """Update the status area with progress message."""
        try:
            status_area = self.query_one("#ingest-local-web-status", TextArea)
            status_area.append(f"\n{message}")
            
            # Also update counters if we have results info
            if hasattr(self, '_current_progress'):
                progress_text = self.query_one("#ingest-local-web-progress-text", Static)
                counters = self.query_one("#ingest-local-web-counters", Static)
                
                progress_text.update(f"Progress: {self._current_progress['done']}/{self._current_progress['total']}")
                counters.update(f"✅ {self._current_progress['success']}  ❌ {self._current_progress['failed']}  ⏳ {self._current_progress['pending']}")
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    async def _handle_stop_web_scraping(self) -> None:
        """Handle stopping the web scraping process."""
        if hasattr(self, '_web_scraping_worker') and self._web_scraping_worker:
            try:
                self._web_scraping_worker.cancel()
                self.app_instance.notify("Stopping web scraping...", severity="warning")
                
                # Update status
                status_area = self.query_one("#ingest-local-web-status", TextArea)
                status_area.append("\n\n⚠️ Processing stopped by user")
                
                # Clean up UI
                self._cleanup_after_processing()
            except Exception as e:
                logger.error(f"Error stopping web scraping: {e}")
    
    def _cleanup_after_processing(self) -> None:
        """Clean up UI after processing completes or is stopped."""
        try:
            # Re-enable process button
            process_button = self.query_one("#ingest-local-web-process", Button)
            process_button.disabled = False
            
            # Hide loading indicator
            loading_indicator = self.query_one("#ingest-local-web-loading", LoadingIndicator)
            loading_indicator.display = False
            loading_indicator.classes = loading_indicator.classes | {"hidden"}
            
            # Hide progress container
            try:
                progress_container = self.query_one("#ingest-local-web-progress", Container)
                progress_container.classes = progress_container.classes | {"hidden"}
            except QueryError:
                pass
            
            # Remove stop button
            try:
                stop_button = self.query_one("#ingest-local-web-stop", Button)
                stop_button.remove()
            except QueryError:
                pass  # Button might not exist
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _handle_retry_failed_urls(self) -> None:
        """Handle retrying failed URLs."""
        if not hasattr(self, '_failed_urls_for_retry') or not self._failed_urls_for_retry:
            self.app_instance.notify("No failed URLs to retry", severity="warning")
            return
        
        logger.info(f"Retrying {len(self._failed_urls_for_retry)} failed URLs")
        
        # Get UI elements
        try:
            loading_indicator = self.query_one("#ingest-local-web-loading", LoadingIndicator)
            status_area = self.query_one("#ingest-local-web-status", TextArea)
            process_button = self.query_one("#ingest-local-web-process", Button)
            retry_button = self.query_one("#ingest-local-web-retry", Button)
        except Exception as e:
            logger.error(f"Error finding UI elements: {e}")
            self.app_instance.notify("Error: UI elements not found", severity="error")
            return
        
        # Show loading state
        loading_indicator.display = True
        loading_indicator.classes = loading_indicator.classes - {"hidden"}
        status_area.append("\n\n## Retrying Failed URLs...\n")
        process_button.disabled = True
        retry_button.disabled = True
        
        # Show progress container
        try:
            progress_container = self.query_one("#ingest-local-web-progress", Container)
            progress_container.classes = progress_container.classes - {"hidden"}
            
            # Initialize progress tracking
            self._current_progress = {
                'total': len(self._failed_urls_for_retry),
                'done': 0,
                'success': 0,
                'failed': 0,
                'pending': len(self._failed_urls_for_retry)
            }
            
            # Update initial progress display
            progress_text = self.query_one("#ingest-local-web-progress-text", Static)
            counters = self.query_one("#ingest-local-web-counters", Static)
            progress_text.update(f"Progress: 0/{len(self._failed_urls_for_retry)}")
            counters.update(f"✅ 0  ❌ 0  ⏳ {len(self._failed_urls_for_retry)}")
        except Exception as e:
            logger.error(f"Error showing progress container: {e}")
        
        # Get scraping options from UI (reuse existing settings)
        try:
            custom_cookies = None
            cookies_str = self.query_one("#ingest-local-web-cookies", Input).value.strip()
            if cookies_str:
                try:
                    custom_cookies = self._parse_cookie_string(cookies_str)
                except Exception as e:
                    logger.warning(f"Failed to parse cookies: {e}")
            
            title_override = self.query_one("#ingest-local-web-title", Input).value.strip()
            author_override = self.query_one("#ingest-local-web-author", Input).value.strip()
            keywords_text = self.query_one("#ingest-local-web-keywords", TextArea).text.strip()
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()] if keywords_text else []
            js_render = self.query_one("#ingest-local-web-js-render", Checkbox).value
            css_selector = self.query_one("#ingest-local-web-css-selector", Input).value.strip()
            
            # Prepare worker data
            worker_data = {
                'urls': self._failed_urls_for_retry,  # Pass the failed URL objects
                'custom_cookies': custom_cookies,
                'title_override': title_override,
                'author_override': author_override,
                'keywords': keywords,
                'js_render': js_render,
                'css_selector': css_selector,
                'is_retry': True,  # Flag this as a retry
                'max_retries': 1   # Allow 1 more retry attempt
            }
            
            # Clear the failed URLs list
            self._failed_urls_for_retry = []
            
            # Add stop button
            stop_button = Button("Stop Processing", id="ingest-local-web-stop", variant="error")
            action_section = process_button.parent
            if action_section and not self.query("#ingest-local-web-stop"):
                await action_section.mount(stop_button, after=retry_button)
            
            # Start worker
            self._web_scraping_worker = self._process_urls_worker(worker_data)
            
            # Handle worker completion
            def on_worker_done(worker: Worker) -> None:
                """Handle worker completion."""
                if worker.cancelled:
                    self.app_instance.notify("Retry processing cancelled", severity="warning")
                    return
                    
                result = worker.result
                if not result:
                    self.app_instance.notify("No results from retry processing", severity="error")
                    self._cleanup_after_processing()
                    return
                
                processed_count = result['processed_count']
                error_count = result['error_count']
                failed_urls = result['failed_urls']
                
                # Update final status
                summary = f"\n## Retry Complete\n\n"
                summary += f"✅ Successfully processed: {processed_count} articles\n"
                if error_count > 0:
                    summary += f"❌ Still failed: {error_count} articles\n"
                summary += "\n### Details:\n"
                
                # Show results
                for res in result['results'][-10:]:  # Last 10 results
                    if isinstance(res, dict):
                        if res['status'] == 'success':
                            summary += f"✅ {res['title']} - ID: {res['media_id']}\n"
                        else:
                            summary += f"❌ {res['url']} - {res['error']} (Retry attempts: {res.get('retry_count', 0)})\n"
                
                if len(result['results']) > 10:
                    summary += f"\n... and {len(result['results']) - 10} more"
                
                status_area.append(summary)
                
                # Show failed URLs section if any still remain
                if failed_urls:
                    status_area.append("\n\n### Still Failed URLs:\n")
                    for fail in failed_urls:
                        status_area.append(f"- {fail['url']} ({fail.get('error', 'Unknown error')}) - Retry attempts: {fail.get('retry_count', 0)}\n")
                    
                    # Store failed URLs for potential future retry
                    self._failed_urls_for_retry = failed_urls
                    
                    # Update retry button
                    if retry_button:
                        retry_button.label = f"Retry {len(failed_urls)} Failed URLs"
                        retry_button.disabled = False
                
                # Notifications
                if processed_count > 0:
                    self.app_instance.notify(f"Successfully processed {processed_count} articles on retry", severity="information")
                if error_count > 0:
                    self.app_instance.notify(f"Still failed to process {error_count} articles", severity="warning")
                
                # Clean up UI
                self._cleanup_after_processing()
                
                # Re-enable retry button if there are still failures
                if failed_urls and retry_button:
                    retry_button.disabled = False
            
            # Add callback
            self._web_scraping_worker.add_done_callback(on_worker_done)
            
        except Exception as e:
            logger.error(f"Error in retry processing: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
            status_area.append(f"\nError during retry: {str(e)}")
            # Reset UI state
            loading_indicator.display = False
            loading_indicator.classes = loading_indicator.classes | {"hidden"}
            process_button.disabled = False
            if retry_button:
                retry_button.disabled = False
    
    async def _read_text_file(self, file_path: Path, encoding: str) -> str | None:
        """Read a text file with specified encoding."""
        try:
            if encoding == "auto":
                # Try common encodings
                for enc in ["utf-8", "latin-1", "ascii"]:
                    try:
                        return file_path.read_text(encoding=enc)
                    except UnicodeDecodeError:
                        continue
                # If all fail, use utf-8 with errors='replace'
                return file_path.read_text(encoding="utf-8", errors="replace")
            else:
                return file_path.read_text(encoding=encoding)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def _normalize_line_endings(self, content: str, line_ending: str) -> str:
        """Normalize line endings in content."""
        if line_ending == "lf":
            return content.replace("\r\n", "\n").replace("\r", "\n")
        elif line_ending == "crlf":
            return content.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
        return content
    
    def _remove_extra_whitespace(self, content: str) -> str:
        """Remove extra whitespace from content."""
        import re
        # Replace multiple spaces with single space
        content = re.sub(r' +', ' ', content)
        # Replace multiple newlines with double newline
        content = re.sub(r'\n\n+', '\n\n', content)
        # Strip whitespace from each line
        lines = [line.strip() for line in content.split('\n')]
        return '\n'.join(lines)
    
    def _convert_to_paragraphs(self, content: str) -> str:
        """Convert content to paragraph format."""
        import re
        # Split on double newlines or more
        paragraphs = re.split(r'\n\n+', content)
        # Clean up each paragraph
        cleaned_paragraphs = []
        for para in paragraphs:
            # Replace single newlines with spaces
            para = para.replace('\n', ' ')
            # Clean up multiple spaces
            para = re.sub(r' +', ' ', para)
            para = para.strip()
            if para:
                cleaned_paragraphs.append(para)
        return '\n\n'.join(cleaned_paragraphs)
    
    def compose_subscriptions_tab(self) -> ComposeResult:
        """Composes the Subscriptions tab content for RSS/Atom feed and URL monitoring."""
        with VerticalScroll(classes="ingest-media-tab-content"):
            # Introduction Section
            with Container(classes="ingest-intro-section"):
                yield Static("📰 Website Subscriptions & URL Monitoring", classes="sidebar-title")
                yield Markdown(
                    "Monitor RSS/Atom feeds, podcasts, and track changes to specific web pages. "
                    "Get notified when new content is available and automatically ingest it into your media library.",
                    classes="subscription-intro"
                )
            
            # Add Subscription Section
            with Container(classes="ingest-subscription-section"):
                yield Static("Add New Subscription", classes="sidebar-title")
                with Horizontal(classes="subscription-type-row"):
                    yield Label("Subscription Type:")
                    yield Select(
                        [("RSS/Atom Feed", "rss"), ("JSON Feed", "json_feed"), 
                         ("Podcast RSS", "podcast"), ("Single URL", "url"), 
                         ("URL List", "url_list"), ("Sitemap", "sitemap"), 
                         ("API Endpoint", "api")],
                        id="subscription-type-select",
                        value="rss"
                    )
                
                yield Label("URL/Feed Address:")
                yield Input(id="subscription-url-input", placeholder="https://example.com/feed.xml")
                
                yield Label("Name:")
                yield Input(id="subscription-name-input", placeholder="Tech News Feed")
                
                yield Label("Description (optional):")
                yield Input(id="subscription-description-input", placeholder="Latest technology news and updates")
                
                # Organization fields
                with Horizontal(classes="subscription-org-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Tags (comma-separated):")
                        yield Input(id="subscription-tags-input", placeholder="tech, news, ai")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Folder:")
                        yield Input(id="subscription-folder-input", placeholder="Technology")
                
                # Priority and Frequency
                with Horizontal(classes="subscription-priority-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Priority:")
                        yield Select(
                            [("1 - Lowest", "1"), ("2 - Low", "2"), ("3 - Normal", "3"), 
                             ("4 - High", "4"), ("5 - Highest", "5")],
                            id="subscription-priority-select",
                            value="3"
                        )
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Check Frequency:")
                        yield Select(
                            [("Every 15 minutes", "900"), ("Every 30 minutes", "1800"), 
                             ("Every hour", "3600"), ("Every 6 hours", "21600"), 
                             ("Daily", "86400"), ("Weekly", "604800")],
                            id="subscription-frequency-select",
                            value="3600"
                        )
                
                with Collapsible(title="Authentication Options", collapsed=True):
                    yield Label("Authentication Type:")
                    yield Select(
                        [("None", "none"), ("Basic Auth", "basic"), 
                         ("Bearer Token", "bearer"), ("API Key", "api_key")],
                        id="subscription-auth-type",
                        value="none"
                    )
                    yield Label("Username/API Key:")
                    yield Input(id="subscription-auth-username", placeholder="username or API key")
                    yield Label("Password/Token (will be encrypted):")
                    yield Input(id="subscription-auth-password", placeholder="password or token", password=True)
                    yield Label("Custom Headers (JSON):")
                    yield TextArea('{"User-Agent": "CustomBot/1.0"}', 
                                 id="subscription-custom-headers", classes="ingest-textarea-small")
                
                with Collapsible(title="Advanced Options", collapsed=True):
                    yield Checkbox("Auto-ingest new items", False, id="subscription-auto-ingest")
                    yield Checkbox("Extract full content (for RSS)", True, id="subscription-extract-full")
                    yield Label("Change Threshold (% for URLs):")
                    yield Input("10", id="subscription-change-threshold", type="integer")
                    yield Label("CSS Selectors to Ignore (for URLs):")
                    yield TextArea(".ads, .timestamp, .cookie-banner",
                                 id="subscription-ignore-selectors", classes="ingest-textarea-small")
                    yield Label("Rate Limit (requests per minute):")
                    yield Input("60", id="subscription-rate-limit", type="integer")
                    yield Label("Auto-pause after failures:")
                    yield Input("10", id="subscription-auto-pause-threshold", type="integer")
                
                yield Button("Add Subscription", id="subscription-add-button", variant="primary")
            
            # Active Subscriptions Section
            with Container(classes="ingest-subscriptions-list-section"):
                yield Static("Active Subscriptions", classes="sidebar-title")
                
                # Filter controls
                with Horizontal(classes="subscription-filter-controls"):
                    yield Label("Filter by:")
                    yield Select(
                        [("All Types", "all"), ("RSS/Atom", "rss"), ("URLs", "url"), 
                         ("Podcasts", "podcast"), ("APIs", "api")],
                        id="subscription-type-filter",
                        value="all"
                    )
                    yield Input(id="subscription-tag-filter", placeholder="Filter by tag...")
                    yield Select(
                        [("All", "all"), ("Active", "active"), ("Paused", "paused"), 
                         ("Error", "error")],
                        id="subscription-status-filter",
                        value="all"
                    )
                
                yield ListView(id="subscription-active-list", classes="subscription-list")
                with Horizontal(classes="subscription-actions-row"):
                    yield Button("Check All Now", id="subscription-check-all-button")
                    yield Button("Import OPML", id="subscription-import-opml-button")
                    yield Button("Export", id="subscription-export-button")
                    yield Button("Manage Templates", id="subscription-templates-button")
            
            # Health Dashboard Section
            with Container(classes="ingest-health-dashboard-section"):
                yield Static("📊 Subscription Health Dashboard", classes="sidebar-title")
                
                # Summary stats
                with Horizontal(classes="health-stats-row"):
                    with Vertical(classes="health-stat-card"):
                        yield Static("Active", classes="stat-label")
                        yield Static("0", id="stat-active-count", classes="stat-value")
                    with Vertical(classes="health-stat-card"):
                        yield Static("Paused", classes="stat-label")
                        yield Static("0", id="stat-paused-count", classes="stat-value")
                    with Vertical(classes="health-stat-card"):
                        yield Static("Errors", classes="stat-label")
                        yield Static("0", id="stat-error-count", classes="stat-value")
                    with Vertical(classes="health-stat-card"):
                        yield Static("Today's Items", classes="stat-label")
                        yield Static("0", id="stat-today-items", classes="stat-value")
                
                # Failing subscriptions alert
                with Container(id="failing-subscriptions-alert", classes="alert-container hidden"):
                    yield Markdown(
                        "⚠️ **Attention Required**: Some subscriptions are experiencing repeated failures.",
                        classes="alert-message"
                    )
                    yield ListView(id="failing-subscriptions-list", classes="failing-list")
                
                # Recent activity log
                yield Static("Recent Activity", classes="subsection-title")
                activity_log = TextArea("", id="subscription-activity-log", read_only=True, 
                                      classes="activity-log")
                activity_log.styles.max_height = 10
                yield activity_log
            
            # New Items Section
            with Container(classes="ingest-new-items-section"):
                yield Static("New Items to Review", classes="sidebar-title")
                
                with Horizontal(classes="items-filter-row"):
                    yield Label("Filter by Source:")
                    yield Select(
                        [("All Sources", "all")],
                        id="subscription-filter-source",
                        value="all"
                    )
                    yield Label("Status:")
                    yield Select(
                        [("New", "new"), ("Reviewed", "reviewed"), ("All", "all")],
                        id="subscription-item-status-filter",
                        value="new"
                    )
                
                yield ListView(id="subscription-new-items-list", classes="subscription-items-list")
                with Horizontal(classes="subscription-review-actions"):
                    yield Button("Accept Selected", id="subscription-accept-button", variant="success")
                    yield Button("Ignore Selected", id="subscription-ignore-button", variant="warning")
                    yield Button("Mark as Reviewed", id="subscription-mark-reviewed-button")
                    yield Button("Apply Filters", id="subscription-apply-filters-button")
            
            # Smart Filters Section
            with Container(classes="ingest-filters-section"):
                yield Static("🔧 Smart Filters", classes="sidebar-title")
                yield Markdown(
                    "Create rules to automatically process items based on conditions.",
                    classes="filters-intro"
                )
                yield ListView(id="subscription-filters-list", classes="filters-list")
                yield Button("Add Filter Rule", id="subscription-add-filter-button")
            
            # Status Section
            with Container(classes="ingest-status-section"):
                yield Static("Monitoring Status", classes="sidebar-title")
                yield TextArea("", id="subscription-status-area", read_only=True, classes="ingest-status-area")
            
            # Placeholder Notice
            with Container(classes="placeholder-notice"):
                yield Markdown(
                    "**Note:** This is a placeholder interface showing enhanced features. The subscription "
                    "monitoring functionality is not yet fully implemented. See `SUBSCRIPTION_IMPLEMENTATION_PLAN.md` "
                    "for implementation details.",
                    classes="warning-notice"
                )

#
# End of Ingest_Window.py
#######################################################################################################################
