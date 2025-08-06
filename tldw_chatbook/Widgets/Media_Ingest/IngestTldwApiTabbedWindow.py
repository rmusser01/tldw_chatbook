# tldw_chatbook/Widgets/IngestTldwApiTabbedWindow.py
#
#
# Imports
from typing import TYPE_CHECKING, List, Dict
from pathlib import Path
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical, Container
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    TabbedContent, TabPane, ListView, ListItem, LoadingIndicator,
    Collapsible
)
from textual.message import Message
#
# Local Imports
from tldw_chatbook.Constants import (
    TLDW_API_AUDIO_OPTIONS_ID, TLDW_API_VIDEO_OPTIONS_ID, 
    TLDW_API_PDF_OPTIONS_ID, TLDW_API_EBOOK_OPTIONS_ID, 
    TLDW_API_DOCUMENT_OPTIONS_ID, TLDW_API_XML_OPTIONS_ID, 
    TLDW_API_MEDIAWIKI_OPTIONS_ID, TLDW_API_PLAINTEXT_OPTIONS_ID
)
from tldw_chatbook.tldw_api.schemas import MediaType, ChunkMethod, PdfEngine
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
#
#######################################################################################################################
#
# Classes:

MEDIA_TYPES = ['video', 'audio', 'document', 'pdf', 'ebook', 'xml', 'mediawiki_dump', 'plaintext']

class IngestTldwApiTabbedWindow(Vertical):
    """A tabbed window containing forms for ingesting different media types via tldw API."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = {}  # Stores {media_type: [Path, ...]}
        self._current_media_type_for_file_dialog = None
        logger.debug("IngestTldwApiTabbedWindow initialized.")
    
    def compose_tldw_api_form(self, media_type: str) -> ComposeResult:
        """Composes the common part of the form for 'Ingest Media via tldw API'."""
        # Get default API URL from app config
        default_api_url = self.app_instance.app_config.get("tldw_api", {}).get("base_url", "http://127.0.0.1:8000")
        
        # Get available API providers for analysis from app config
        analysis_api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
        analysis_provider_options = [(name, name) for name in analysis_api_providers if name]
        if not analysis_provider_options:
            analysis_provider_options = [("No Providers Configured", Select.BLANK)]
        
        with VerticalScroll(classes="ingest-form-scrollable"):
            yield Static("TLDW API Configuration", classes="sidebar-title")
            yield Label("API Endpoint URL:")
            yield Input(default_api_url, id=f"tldw-api-endpoint-url-{media_type}", placeholder="http://localhost:8000")
            
            yield Label("Authentication Method:")
            yield Select(
                [
                    ("Token from Config", "config_token"),
                    ("Custom Token", "custom_token"),
                ],
                prompt="Select Auth Method...",
                id=f"tldw-api-auth-method-{media_type}",
                value="config_token"
            )
            yield Label("Custom Auth Token:", id=f"tldw-api-custom-token-label-{media_type}", classes="hidden")
            yield Input(
                "",
                id=f"tldw-api-custom-token-{media_type}",
                placeholder="Enter custom Bearer token",
                password=True,
                classes="hidden",
                tooltip="Enter your Bearer token for the TLDW API. This is used if 'Custom Token' is selected as the authentication method."
            )
            
            yield Static("Media Details & Processing Options", classes="sidebar-title")
            
            # --- Common Input Fields ---
            yield Label("Media URLs (one per line):")
            yield TextArea(id=f"tldw-api-urls-{media_type}", classes="ingest-textarea-small")
            yield Button("Browse Local Files...", id=f"tldw-api-browse-local-files-button-{media_type}")
            yield Label("Selected Local Files:", classes="ingest-label")
            yield ListView(id=f"tldw-api-selected-local-files-list-{media_type}", classes="ingest-selected-files-list")
            
            with Horizontal(classes="title-author-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id=f"tldw-api-title-{media_type}", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id=f"tldw-api-author-{media_type}", placeholder="Optional author override")
            
            yield Label("Keywords (comma-separated):")
            yield TextArea(id=f"tldw-api-keywords-{media_type}", classes="ingest-textarea-small")
            
            # --- Web Scraping Options (for URLs) ---
            with Collapsible(title="Web Scraping Options", collapsed=True, id=f"tldw-api-webscraping-collapsible-{media_type}"):
                yield Checkbox("Use Cookies for Web Scraping", False, id=f"tldw-api-use-cookies-{media_type}")
                yield Label("Cookies (JSON format):")
                yield TextArea(
                    id=f"tldw-api-cookies-{media_type}", 
                    classes="ingest-textarea-small",
                    tooltip="Paste cookies in JSON format for authenticated web scraping"
                )
            
            # --- Common Processing Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id=f"tldw-api-custom-prompt-{media_type}", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id=f"tldw-api-system-prompt-{media_type}", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id=f"tldw-api-perform-analysis-{media_type}")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id=f"tldw-api-analysis-api-name-{media_type}",
                         prompt="Select API for Analysis...")
            yield Label("Analysis API Key (if needed):")
            yield Input(
                "",
                id=f"tldw-api-analysis-api-key-{media_type}",
                placeholder="API key for analysis provider",
                password=True,
                tooltip="API key for the selected analysis provider. Leave empty to use default from config."
            )
            
            # --- Common Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id=f"tldw-api-chunking-collapsible-{media_type}"):
                yield Checkbox("Perform Chunking", True, id=f"tldw-api-perform-chunking-{media_type}")
                yield Label("Chunking Method:")
                chunk_method_options = [
                    ("semantic", "semantic"),
                    ("tokens", "tokens"),
                    ("paragraphs", "paragraphs"),
                    ("sentences", "sentences"),
                    ("words", "words"),
                    ("ebook_chapters", "ebook_chapters"),
                    ("json", "json")
                ]
                yield Select(chunk_method_options, id=f"tldw-api-chunk-method-{media_type}", prompt="Default (per type)")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("500", id=f"tldw-api-chunk-size-{media_type}", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id=f"tldw-api-chunk-overlap-{media_type}", type="integer")
                yield Label("Chunk Language (e.g., 'en', optional):")
                yield Input(id=f"tldw-api-chunk-lang-{media_type}", placeholder="Defaults to media language")
                yield Checkbox("Use Adaptive Chunking", False, id=f"tldw-api-adaptive-chunking-{media_type}")
                yield Checkbox("Use Multi-level Chunking", False, id=f"tldw-api-multi-level-chunking-{media_type}")
                yield Label("Custom Chapter Pattern (Regex, optional):")
                yield Input(id=f"tldw-api-custom-chapter-pattern-{media_type}", placeholder="e.g., ^Chapter\\s+\\d+")
            
            # --- Common Analysis Options ---
            with Collapsible(title="Advanced Analysis Options", collapsed=True,
                             id=f"tldw-api-analysis-opts-collapsible-{media_type}"):
                yield Checkbox("Summarize Recursively (if chunked)", False, id=f"tldw-api-summarize-recursively-{media_type}")
                yield Checkbox("Perform Rolling Summarization", False, id=f"tldw-api-perform-rolling-summarization-{media_type}")
            
            # --- Media-Type Specific Options ---
            if media_type == "video":
                with Container(id=TLDW_API_VIDEO_OPTIONS_ID, classes="tldw-api-media-specific-options"):
                    yield Static("Video Specific Options", classes="sidebar-title")
                    yield Label("Transcription Model:")
                    yield Input("deepdml/faster-whisper-large-v3-turbo-ct2", id=f"tldw-api-video-transcription-model-{media_type}")
                    yield Label("Transcription Language (e.g., 'en'):")
                    yield Input("en", id=f"tldw-api-video-transcription-language-{media_type}")
                    yield Checkbox("Enable Speaker Diarization", False, id=f"tldw-api-video-diarize-{media_type}")
                    yield Checkbox("Include Timestamps in Transcription", True, id=f"tldw-api-video-timestamp-{media_type}")
                    yield Checkbox("Enable VAD (Voice Activity Detection)", False, id=f"tldw-api-video-vad-{media_type}")
                    yield Checkbox("Perform Confabulation Check of Analysis", False, id=f"tldw-api-video-confab-check-{media_type}")
                    with Horizontal(classes="ingest-form-row"):
                        with Vertical(classes="ingest-form-col"):
                            yield Label("Start Time (HH:MM:SS or secs):")
                            yield Input(id=f"tldw-api-video-start-time-{media_type}", placeholder="Optional")
                        with Vertical(classes="ingest-form-col"):
                            yield Label("End Time (HH:MM:SS or secs):")
                            yield Input(id=f"tldw-api-video-end-time-{media_type}", placeholder="Optional")
            elif media_type == "audio":
                with Container(id=TLDW_API_AUDIO_OPTIONS_ID, classes="tldw-api-media-specific-options"):
                    yield Static("Audio Specific Options", classes="sidebar-title")
                    yield Label("Transcription Model:")
                    yield Input("deepdml/faster-distil-whisper-large-v3.5", id=f"tldw-api-audio-transcription-model-{media_type}")
                    yield Label("Transcription Language (e.g., 'en'):")
                    yield Input("en", id=f"tldw-api-audio-transcription-language-{media_type}")
                    yield Checkbox("Enable Speaker Diarization", False, id=f"tldw-api-audio-diarize-{media_type}")
                    yield Checkbox("Include Timestamps in Transcription", True, id=f"tldw-api-audio-timestamp-{media_type}")
                    yield Checkbox("Enable VAD (Voice Activity Detection)", False, id=f"tldw-api-audio-vad-{media_type}")
            elif media_type == "pdf":
                pdf_engine_options = [
                    ("pymupdf4llm", "pymupdf4llm"),
                    ("pymupdf", "pymupdf"),
                    ("docling", "docling")
                ]
                with Container(id=TLDW_API_PDF_OPTIONS_ID, classes="tldw-api-media-specific-options"):
                    yield Static("PDF Specific Options", classes="sidebar-title")
                    yield Label("PDF Parsing Engine:")
                    yield Select(pdf_engine_options, id=f"tldw-api-pdf-engine-{media_type}", value="pymupdf4llm")
            elif media_type == "ebook":
                ebook_extraction_options = [("filtered", "filtered"), ("markdown", "markdown"), ("basic", "basic")]
                with Container(id=TLDW_API_EBOOK_OPTIONS_ID, classes="tldw-api-media-specific-options"):
                    yield Static("Ebook Specific Options", classes="sidebar-title")
                    yield Label("Ebook Extraction Method:")
                    yield Select(ebook_extraction_options, id=f"tldw-api-ebook-extraction-method-{media_type}", value="filtered")
            elif media_type == "document":
                with Container(id=TLDW_API_DOCUMENT_OPTIONS_ID, classes="tldw-api-media-specific-options"):
                    yield Static("Document Specific Options", classes="sidebar-title")
            elif media_type == "xml":
                with Container(id=TLDW_API_XML_OPTIONS_ID, classes="tldw-api-media-specific-options"):
                    yield Static("XML Specific Options (Note: Only one local file at a time)", classes="sidebar-title")
                    yield Checkbox("Auto Summarize XML Content", False, id=f"tldw-api-xml-auto-summarize-{media_type}")
            elif media_type == "mediawiki_dump":
                with Container(id=TLDW_API_MEDIAWIKI_OPTIONS_ID, classes="tldw-api-media-specific-options"):
                    yield Static("MediaWiki Dump Specific Options (Note: Only one local file at a time)", classes="sidebar-title")
                    yield Label("Wiki Name (for identification):")
                    yield Input(id=f"tldw-api-mediawiki-wiki-name-{media_type}", placeholder="e.g., my_wiki_backup")
                    yield Label("Namespaces (comma-sep IDs, optional):")
                    yield Input(id=f"tldw-api-mediawiki-namespaces-{media_type}", placeholder="e.g., 0,14")
                    yield Checkbox("Skip Redirect Pages (recommended)", True, id=f"tldw-api-mediawiki-skip-redirects-{media_type}")
                    yield Label("Chunk Max Size:")
                    yield Input("1000", id=f"tldw-api-mediawiki-chunk-max-size-{media_type}", type="integer")
                    yield Label("Vector DB API (optional):")
                    yield Input(id=f"tldw-api-mediawiki-api-name-vector-db-{media_type}", placeholder="For embeddings")
                    yield Label("Vector DB API Key (optional):")
                    yield Input(id=f"tldw-api-mediawiki-api-key-vector-db-{media_type}", password=True, placeholder="API key for vector DB")
            elif media_type == "plaintext":
                with Container(id=TLDW_API_PLAINTEXT_OPTIONS_ID, classes="tldw-api-media-specific-options"):
                    yield Static("Plaintext Specific Options", classes="sidebar-title")
                    yield Label("Text Encoding:")
                    yield Select(
                        [("UTF-8", "utf-8"), ("ASCII", "ascii"), ("Latin-1", "latin-1"), ("Auto-detect", "auto")],
                        id=f"tldw-api-encoding-{media_type}",
                        value="utf-8"
                    )
                    yield Label("Line Ending:")
                    yield Select(
                        [("Auto", "auto"), ("Unix (LF)", "lf"), ("Windows (CRLF)", "crlf")],
                        id=f"tldw-api-line-ending-{media_type}",
                        value="auto"
                    )
                    yield Checkbox("Remove Extra Whitespace", True, id=f"tldw-api-remove-whitespace-{media_type}")
                    yield Checkbox("Convert to Paragraphs", False, id=f"tldw-api-convert-paragraphs-{media_type}")
                    yield Label("Split Pattern (Regex, optional):")
                    yield Input(id=f"tldw-api-split-pattern-{media_type}", placeholder="e.g., \\n\\n+ for double newlines")
            
            yield Static("Local Database Options", classes="sidebar-title")
            yield Checkbox("Overwrite if media exists in local DB", False, id=f"tldw-api-overwrite-db-{media_type}")
            
            yield Button("Submit to TLDW API", id=f"tldw-api-submit-{media_type}", variant="primary", classes="ingest-submit-button")
            # LoadingIndicator and TextArea for API status/error messages
            yield LoadingIndicator(id=f"tldw-api-loading-indicator-{media_type}", classes="hidden")
            yield TextArea(
                "",
                id=f"tldw-api-status-area-{media_type}",
                read_only=True,
                classes="ingest-status-area hidden"
            )
    
    def compose(self) -> ComposeResult:
        """Compose the tabbed interface for all media types."""
        logger.debug("Composing IngestTldwApiTabbedWindow UI")
        
        yield Static("Ingest Content via tldw API", classes="window-title")
        
        with TabbedContent(id="tldw-api-tabs"):
            for media_type in MEDIA_TYPES:
                # Create user-friendly tab titles
                tab_title = media_type.replace('_', ' ').title()
                if media_type == 'mediawiki_dump':
                    tab_title = "MediaWiki Dump"
                elif media_type == 'pdf':
                    tab_title = "PDF"
                elif media_type == 'xml':
                    tab_title = "XML"
                
                with TabPane(tab_title, id=f"tab-tldw-api-{media_type}"):
                    yield from self.compose_tldw_api_form(media_type=media_type)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses within the tabbed window."""
        button_id = event.button.id
        if not button_id:
            return
        
        if button_id.startswith("tldw-api-browse-local-files-button-"):
            event.stop()
            media_type = button_id.replace("tldw-api-browse-local-files-button-", "")
            self._current_media_type_for_file_dialog = media_type
            
            raw_initial_path = self.app_instance.app_config.get("user_data_path", Path.home())
            dialog_initial_path = str(raw_initial_path)
            
            logger.debug(f"Opening file dialog for media type '{media_type}' with initial path '{dialog_initial_path}'.")
            
            from ..Third_Party.textual_fspicker.file_open import FileOpen
            await self.app.push_screen(
                FileOpen(
                    title=f"Select Local File for {media_type.title()}"
                ),
                callback=self.handle_file_picker_dismissed
            )
    
    async def handle_file_picker_dismissed(self, selected_file_path: Path | None) -> None:
        """Handle file picker results."""
        logger.debug(f"File picker dismissed, selected path: {selected_file_path}")
        if self._current_media_type_for_file_dialog is None:
            logger.warning("File picker dismissed but no media type context was set. Ignoring.")
            return
        
        media_type = self._current_media_type_for_file_dialog
        
        if not selected_file_path:
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
        
        list_view_id = f"#tldw-api-selected-local-files-list-{media_type}"
        try:
            list_view = self.query_one(list_view_id, ListView)
            await list_view.clear()
            
            for path_item in self.selected_local_files[media_type]:
                list_item = ListItem(Label(str(path_item)))
                await list_view.append(list_item)
            logger.debug(f"Updated ListView '{list_view_id}' for media type '{media_type}'.")
        except Exception as e:
            logger.error(f"Error updating ListView {list_view_id} for {media_type}: {e}", exc_info=True)
    
    class BackButtonPressed(Message):
        """Message sent when the back button is pressed."""
        pass

#
# End of IngestTldwApiTabbedWindow.py
#######################################################################################################################