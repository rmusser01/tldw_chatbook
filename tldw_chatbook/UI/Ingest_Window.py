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
from textual.widgets import Static, Button, Input, Select, Checkbox, TextArea, Label, RadioSet, RadioButton, Collapsible, ListView, ListItem, Markdown, LoadingIndicator # Button, ListView, ListItem, Label are already here

# Configure logger with context
logger = logger.bind(module="Ingest_Window")

from ..Constants import TLDW_API_AUDIO_OPTIONS_ID, TLDW_API_VIDEO_OPTIONS_ID, TLDW_API_PDF_OPTIONS_ID, \
    TLDW_API_EBOOK_OPTIONS_ID, TLDW_API_DOCUMENT_OPTIONS_ID, TLDW_API_XML_OPTIONS_ID, TLDW_API_MEDIAWIKI_OPTIONS_ID
#
# Local Imports
from ..Third_Party.textual_fspicker.file_open import FileOpen
from ..tldw_api.schemas import MediaType, ChunkMethod, PdfEngine  # Import Enums
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
    *[f"ingest-view-tldw-api-{mt}" for mt in MEDIA_TYPES]
]
INGEST_NAV_BUTTON_IDS = [
    "ingest-nav-prompts", "ingest-nav-characters",
    "ingest-nav-media", "ingest-nav-notes",
    *[f"ingest-nav-tldw-api-{mt}" for mt in MEDIA_TYPES]
]

class IngestWindow(Container):
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = {}  # Stores {media_type: [Path, ...]}
        self._current_media_type_for_file_dialog = None # Stores the media_type for the active file dialog
        logger.debug("IngestWindow initialized.")

    def compose_tldw_api_form(self, media_type: str) -> ComposeResult:
        """Composes the common part of the form for 'Ingest Media via tldw API'."""
        # Get default API URL from app config
        default_api_url = self.app_instance.app_config.get("tldw_api", {}).get("base_url", "http://127.0.0.1:8000")

        # Get available API providers for analysis from app config
        # This assumes your main app_config has a structure like:
        # "api_settings": { "openai": {"model": "gpt-4"}, "anthropic": { ... } }
        # Or use self.app_instance.providers_models if that's more appropriate
        analysis_api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
        analysis_provider_options = [(name, name) for name in analysis_api_providers if name]
        if not analysis_provider_options:
            analysis_provider_options = [("No Providers Configured", Select.BLANK)]

        with VerticalScroll(classes="ingest-form-scrollable"): # FIXME/TODO: Needs unique Header ID since this is temlplate for whatever media type is selected
            yield Static("TLDW API Configuration", classes="sidebar-title")
            yield Label("API Endpoint URL:")
            yield Input(default_api_url, id=f"tldw-api-endpoint-url-{media_type}", placeholder="http://localhost:8000")

            yield Label("Authentication Method:")
            yield Select(
                [
                    ("Token from Config", "config_token"),
                    ("Custom Token", "custom_token"),
                    # ("Environment Variable", "env_var_token") # Future: add if needed
                ],
                prompt="Select Auth Method...",
                id=f"tldw-api-auth-method-{media_type}",
                value="config_token"  # Default
            )
            yield Label("Custom Auth Token:", id=f"tldw-api-custom-token-label-{media_type}", classes="hidden")  # Hidden by default
            yield Input(
                "",
                id=f"tldw-api-custom-token-{media_type}",
                placeholder="Enter custom Bearer token",
                password=True,
                classes="hidden",  # Hidden by default
                tooltip="Enter your Bearer token for the TLDW API. This is used if 'Custom Token' is selected as the authentication method."
            )

            yield Static("Media Details & Processing Options", classes="sidebar-title")
            # Media Type selection is now handled by which form is shown

            # --- Common Input Fields ---
            # FIXME/TODO: Consider if URL/Local File input is applicable for all media_type or if this also needs to be specific
            # For example, mediawiki_dump typically uses a local file path.
            yield Label("Media URLs (one per line):")
            yield TextArea(id=f"tldw-api-urls-{media_type}", classes="ingest-textarea-small")
            # yield Label(
            #     "Local File Paths (one per line, if API supports local path references or for client-side upload):")
            # yield TextArea(id=f"tldw-api-local-files-{media_type}", classes="ingest-textarea-small")
            yield Button("Browse Local Files...", id=f"tldw-api-browse-local-files-button-{media_type}")
            yield Label("Selected Local Files:", classes="ingest-label") # Added class for consistency
            yield ListView(id=f"tldw-api-selected-local-files-list-{media_type}", classes="ingest-selected-files-list")


            with Horizontal(classes="title-author-row"): # Changed class here
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id=f"tldw-api-title-{media_type}", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id=f"tldw-api-author-{media_type}", placeholder="Optional author override")

            yield Label("Keywords (comma-separated):")
            yield TextArea(id=f"tldw-api-keywords-{media_type}", classes="ingest-textarea-small")

            # --- Common Processing Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id=f"tldw-api-custom-prompt-{media_type}", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id=f"tldw-api-system-prompt-{media_type}", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id=f"tldw-api-perform-analysis-{media_type}")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id=f"tldw-api-analysis-api-name-{media_type}",
                         prompt="Select API for Analysis...")

            # --- Common Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id=f"tldw-api-chunking-collapsible-{media_type}"):
                yield Checkbox("Perform Chunking", True, id=f"tldw-api-perform-chunking-{media_type}")
                yield Label("Chunking Method:")
                chunk_method_options = [(cm, cm) for cm in ChunkMethod.__args__]
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
                # Add more analysis options here as needed

            # --- Media-Type Specific Option Containers have been removed from this common method ---
            # They will be added to the media-type specific views/containers directly.

            # --- Inserted Media-Type Specific Options ---
            if media_type == "video":
                with Container(id=TLDW_API_VIDEO_OPTIONS_ID, classes="tldw-api-media-specific-options"): # ID of container itself is fine
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
                    # TODO: Add other audio specific fields from ProcessAudioRequest: confab (e.g. id=f"tldw-api-audio-confab-check-{media_type}")
            elif media_type == "pdf":
                pdf_engine_options = [(engine, engine) for engine in PdfEngine.__args__]
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
                    # yield Label("Note: Document specific options are minimal beyond common settings.") # Placeholder - can be removed or made more specific if needed
                    # If adding specific fields, ensure their IDs are dynamic e.g. id=f"tldw-api-doc-some-option-{media_type}"
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
            # --- End of Inserted Media-Type Specific Options ---

            yield Static("Local Database Options", classes="sidebar-title")
            yield Checkbox("Overwrite if media exists in local DB", False, id=f"tldw-api-overwrite-db-{media_type}")

            yield Button("Submit to TLDW API", id=f"tldw-api-submit-{media_type}", variant="primary", classes="ingest-submit-button")
            # LoadingIndicator and TextArea for API status/error messages
            yield LoadingIndicator(id=f"tldw-api-loading-indicator-{media_type}", classes="hidden") # Initially hidden
            yield TextArea(
                "",
                id=f"tldw-api-status-area-{media_type}",
                read_only=True,
                classes="ingest-status-area hidden"  # Initially hidden, common styling
            )

    def compose(self) -> ComposeResult:
        logger.debug("Composing IngestWindow UI")
        with VerticalScroll(id="ingest-nav-pane", classes="ingest-nav-pane"):
            yield Static("Ingestion Methods", classes="sidebar-title")
            # Add new buttons for each media type
            for media_type in MEDIA_TYPES:
                label = f"Ingest {media_type.replace('_', ' ').title()} via tldw API"
                if media_type == 'mediawiki_dump':
                    label = "Ingest MediaWiki Dump via tldw API"
                button_id = f"ingest-nav-tldw-api-{media_type}"
                yield Button(label, id=button_id, classes="ingest-nav-button")
            yield Button("Ingest Prompts", id="ingest-nav-prompts", classes="ingest-nav-button")
            yield Button("Ingest Characters", id="ingest-nav-characters", classes="ingest-nav-button")
            yield Button("Ingest Media (Local)", id="ingest-nav-media", classes="ingest-nav-button")
            yield Button("Ingest Notes", id="ingest-nav-notes", classes="ingest-nav-button")


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
            yield Container(
                Static("Local Media Ingestion Area - Content Coming Soon!"),  # For direct local processing
                id="ingest-view-media",
                classes="ingest-view-area",
            )

            # New containers for tldw API forms for each media type
            for media_type in MEDIA_TYPES:
                with Container(id=f"ingest-view-tldw-api-{media_type}", classes="ingest-view-area hidden"): # Start hidden
                    # TODO: Later, tailor the form composition if needed per media type, (This is the next step)
                    # or decide if one generic form is enough and it's just the nav that changes.
                    # For now, we assume compose_tldw_api_form is generic enough or will be adapted.
                    yield from self.compose_tldw_api_form(media_type=media_type)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if not button_id: # Should always have an ID
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

#
# End of Logs_Window.py
#######################################################################################################################
