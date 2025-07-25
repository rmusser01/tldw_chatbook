# tldw_chatbook/UI/Ingest_Window.py
#
#
# Imports
from typing import TYPE_CHECKING, List, Dict, Any, Optional
from pathlib import Path
import asyncio
import time
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.css.query import QueryError
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.widgets import Static, Button, Input, Select, Checkbox, TextArea, Label, RadioSet, RadioButton, Collapsible, ListView, ListItem, Markdown, LoadingIndicator, TabbedContent, TabPane # Button, ListView, ListItem, Label are already here
from textual import on
from textual.worker import Worker
from textual import work
from textual.reactive import reactive
from ..Widgets.form_components import (
    create_form_field, create_button_group, create_status_area
)
from ..Widgets.status_widget import EnhancedStatusWidget

# Configure logger with context
logger = logger.bind(module="Ingest_Window")

from ..Constants import TLDW_API_AUDIO_OPTIONS_ID, TLDW_API_VIDEO_OPTIONS_ID, TLDW_API_PDF_OPTIONS_ID, \
    TLDW_API_EBOOK_OPTIONS_ID, TLDW_API_DOCUMENT_OPTIONS_ID, TLDW_API_XML_OPTIONS_ID, TLDW_API_MEDIAWIKI_OPTIONS_ID
#
# Local Imports
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from ..tldw_api.schemas import MediaType, ChunkMethod, PdfEngine  # Import Enums
from ..Widgets.IngestTldwApiVideoWindow import IngestTldwApiVideoWindow
from ..Widgets.IngestTldwApiAudioWindow import IngestTldwApiAudioWindow
from ..Widgets.IngestTldwApiPdfWindow import IngestTldwApiPdfWindow
from ..Widgets.IngestTldwApiEbookWindow import IngestTldwApiEbookWindow
from ..Widgets.IngestTldwApiDocumentWindow import IngestTldwApiDocumentWindow
from ..Widgets.IngestTldwApiXmlWindow import IngestTldwApiXmlWindow
from ..Widgets.IngestTldwApiMediaWikiWindow import IngestTldwApiMediaWikiWindow
from ..Widgets.IngestTldwApiPlaintextWindow import IngestTldwApiPlaintextWindow
from ..Widgets.IngestLocalPlaintextWindow import IngestLocalPlaintextWindow
from ..Widgets.IngestLocalWebArticleWindow import IngestLocalWebArticleWindow
from ..Widgets.IngestLocalDocumentWindow import IngestLocalDocumentWindow
from ..Widgets.IngestLocalEbookWindow import IngestLocalEbookWindow
from ..Widgets.IngestLocalPdfWindow import IngestLocalPdfWindow
from ..Widgets.IngestLocalAudioWindow import IngestLocalAudioWindow
from ..Widgets.IngestLocalVideoWindow import IngestLocalVideoWindow
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

def append_to_text_area(text_area: TextArea, new_text: str) -> None:
    """Helper function to append text to a TextArea widget.
    
    Args:
        text_area: The TextArea widget to update
        new_text: The text to append
    """
    current_text = text_area.text
    text_area.text = current_text + new_text

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
    # Reactive property for sidebar collapse state
    sidebar_collapsed = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = {}  # Stores {media_type: [Path, ...]}
        self._current_media_type_for_file_dialog = None # Stores the media_type for the active file dialog
        self._failed_urls_for_retry = []  # Store failed URLs for retry
        self._retry_attempts = {}  # Track retry attempts per URL
        self._local_video_window = None
        self._local_audio_window = None
        logger.debug("IngestWindow initialized.")
    
    def get_default_model_for_provider(self, provider: str) -> str:
        """Get default model for a transcription provider."""
        provider_default_models = {
            'parakeet-mlx': 'mlx-community/parakeet-tdt-0.6b-v2',
            'lightning-whisper-mlx': 'base',
            'faster-whisper': 'base',
            'qwen2audio': 'Qwen2-Audio-7B-Instruct',
            'parakeet': 'nvidia/parakeet-tdt-1.1b',
            'canary': 'nvidia/canary-1b-flash'
        }
        return provider_default_models.get(provider, 'base')
    
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
        
        # Handle collapse/expand button
        if button_id == "ingest-nav-collapse":
            self.sidebar_collapsed = not self.sidebar_collapsed
            event.stop()
            return
        
        # Handle navigation buttons
        if button_id.startswith("ingest-nav-"):
            view_id = button_id.replace("ingest-nav-", "ingest-view-")
            logger.debug(f"Switching to view: {view_id}")
            # Call the app's show_ingest_view method
            self.app_instance.show_ingest_view(view_id)
            # Update active button styling
            await self._update_active_nav_button(button_id)
            event.stop()
            return
        
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
    
    @on(RadioSet.Changed, "#ingest-notes-import-type")
    async def on_notes_import_type_changed(self, event: RadioSet.Changed) -> None:
        """Handle import type change for notes."""
        logger.debug(f"Notes import type changed to index: {event.radio_set.pressed_index}")
        
        # Update the preview if files are already selected
        if hasattr(self.app_instance, 'parsed_notes_for_preview') and self.app_instance.parsed_notes_for_preview:
            # Clear existing preview
            self.app_instance.parsed_notes_for_preview.clear()
            
            # Re-parse selected files with new import type
            try:
                list_view = self.query_one("#ingest-notes-selected-files-list", ListView)
                import_as_template = event.radio_set.pressed_index == 1
                
                for item in list_view.children:
                    if isinstance(item, ListItem):
                        label = item.children[0] if item.children else None
                        if isinstance(label, Label):
                            file_path = Path(str(label.renderable).strip())
                            if file_path.exists():
                                from ..Event_Handlers.ingest_events import _parse_single_note_file_for_preview
                                parsed_notes = _parse_single_note_file_for_preview(
                                    file_path, 
                                    self.app_instance,
                                    import_as_template=import_as_template
                                )
                                self.app_instance.parsed_notes_for_preview.extend(parsed_notes)
                
                # Update the preview display
                from ..Event_Handlers.ingest_events import _update_note_preview_display
                await _update_note_preview_display(self.app_instance)
                
            except Exception as e:
                logger.error(f"Error updating notes preview after import type change: {e}")
    
    async def _update_active_nav_button(self, active_button_id: str) -> None:
        """Update the active state of navigation buttons."""
        try:
            # Remove active class from all nav buttons
            for button in self.query(".ingest-nav-button"):
                button.remove_class("active")
            
            # Add active class to the clicked button
            active_button = self.query_one(f"#{active_button_id}")
            active_button.add_class("active")
            logger.debug(f"Updated active nav button: {active_button_id}")
        except QueryError as e:
            logger.error(f"Error updating active nav button: {e}")
    
    def watch_sidebar_collapsed(self, collapsed: bool) -> None:
        """React to sidebar collapse state changes."""
        try:
            nav_pane = self.query_one("#ingest-nav-pane")
            toggle_button = self.query_one("#ingest-nav-collapse")
            
            if collapsed:
                nav_pane.add_class("collapsed")
                toggle_button.label = "▶"
                toggle_button.tooltip = "Expand sidebar"
                # Hide all text elements when collapsed
                for element in nav_pane.query(".sidebar-title, .ingest-nav-button"):
                    element.add_class("collapsed-hidden")
            else:
                nav_pane.remove_class("collapsed")
                toggle_button.label = "◀"
                toggle_button.tooltip = "Collapse sidebar"
                # Show all text elements when expanded
                for element in nav_pane.query(".sidebar-title, .ingest-nav-button"):
                    element.remove_class("collapsed-hidden")
                    
            logger.debug(f"Sidebar collapsed state changed to: {collapsed}")
        except QueryError as e:
            logger.error(f"Error updating sidebar collapse state: {e}")
    
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
                ("Document Files", lambda p: p.suffix.lower() in (".docx", ".doc", ".odt", ".rtf", ".pptx", ".ppt", ".xlsx", ".xls", ".ods", ".odp")),
                ("Microsoft Word", lambda p: p.suffix.lower() in (".docx", ".doc")),
                ("OpenDocument", lambda p: p.suffix.lower() in (".odt", ".ods", ".odp")),
                ("Microsoft Office", lambda p: p.suffix.lower() in (".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls")),
                ("Rich Text", lambda p: p.suffix.lower() == ".rtf"),
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
            # Add collapse/expand button at the top
            with Horizontal(classes="nav-header"):
                yield Static("Navigation", classes="sidebar-title flex-grow")
                yield Button("◀", id="ingest-nav-collapse", classes="nav-toggle-button", tooltip="Collapse sidebar")
            
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
            with VerticalScroll(id="ingest-view-prompts", classes="ingest-view-area"):
                # File selection buttons
                yield from create_button_group([
                    ("Select Prompt File(s)", "ingest-prompts-select-file-button", "default"),
                    ("Clear Selection", "ingest-prompts-clear-files-button", "default")
                ])
                
                yield Label("Selected Files for Import:", classes="form-label")
                yield ListView(id="ingest-prompts-selected-files-list", classes="ingest-selected-files-list")

                yield Label("Preview of Parsed Prompts (Max 10 shown):", classes="form-label")
                # Remove nested VerticalScroll - just use a container
                with Container(id="ingest-prompts-preview-area", classes="ingest-preview-area"):
                    yield Static("Select files to see a preview.", id="ingest-prompts-preview-placeholder")
                
                # Import button centered
                yield from create_button_group([
                    ("Import Selected Prompts Now", "ingest-prompts-import-now-button", "primary")
                ], alignment="center")
                
                # Enhanced status widget instead of TextArea
                yield EnhancedStatusWidget(
                    title="Import Status",
                    id="prompt-import-status-widget",
                    max_messages=50
                )

            # --- Characters Ingest View ---
            with VerticalScroll(id="ingest-view-characters", classes="ingest-view-area"):
                # File selection buttons
                yield from create_button_group([
                    ("Select Character File(s)", "ingest-characters-select-file-button", "default"),
                    ("Clear Selection", "ingest-characters-clear-files-button", "default")
                ])
                
                yield Label("Selected Files for Import:", classes="form-label")
                yield ListView(id="ingest-characters-selected-files-list", classes="ingest-selected-files-list")

                yield Label("Preview of Parsed Characters (Max 5 shown):", classes="form-label")
                # Remove nested VerticalScroll
                with Container(id="ingest-characters-preview-area", classes="ingest-preview-area"):
                    yield Static("Select files to see a preview.", id="ingest-characters-preview-placeholder")

                # Import button centered
                yield from create_button_group([
                    ("Import Selected Characters Now", "ingest-characters-import-now-button", "primary")
                ], alignment="center")
                
                # Enhanced status widget
                yield EnhancedStatusWidget(
                    title="Import Status",
                    id="ingest-character-import-status-widget",
                    max_messages=50
                )

            # --- Notes Ingest View ---
            with VerticalScroll(id="ingest-view-notes", classes="ingest-view-area"):
                # File selection buttons
                yield from create_button_group([
                    ("Select Notes File(s)", "ingest-notes-select-file-button", "default"),
                    ("Clear Selection", "ingest-notes-clear-files-button", "default")
                ])
                
                # Import type selection
                yield Label("Import Type:", classes="form-label")
                with RadioSet(id="ingest-notes-import-type"):
                    yield RadioButton("Import as Notes", value=True, id="import-as-notes-radio")
                    yield RadioButton("Import as Templates", id="import-as-templates-radio")
                
                yield Label("Selected Files for Import:", classes="form-label")
                yield ListView(id="ingest-notes-selected-files-list", classes="ingest-selected-files-list")

                yield Label("Preview of Parsed Notes (Max 10 shown):", classes="form-label")
                # Remove nested VerticalScroll
                with Container(id="ingest-notes-preview-area", classes="ingest-preview-area"):
                    yield Static("Select files to see a preview.", id="ingest-notes-preview-placeholder")

                # Import button centered
                yield from create_button_group([
                    ("Import Selected Notes Now", "ingest-notes-import-now-button", "primary")
                ], alignment="center")
                
                # Enhanced status widget
                yield EnhancedStatusWidget(
                    title="Import Status",
                    id="ingest-notes-import-status-widget",
                    max_messages=50
                )

            # --- Local Media Views ---
            with VerticalScroll(id="ingest-view-local-video", classes="ingest-view-area"):
                yield from self.compose_local_video_tab()
                
            with VerticalScroll(id="ingest-view-local-audio", classes="ingest-view-area"):
                yield from self.compose_local_audio_tab()
                
            with VerticalScroll(id="ingest-view-local-document", classes="ingest-view-area"):
                window = IngestLocalDocumentWindow(self.app_instance)
                yield from window.compose()
                
            with VerticalScroll(id="ingest-view-local-pdf", classes="ingest-view-area"):
                from ..Widgets.IngestLocalPdfWindow import IngestLocalPdfWindow
                window = IngestLocalPdfWindow(self.app_instance)
                yield from window.compose()
                
            with VerticalScroll(id="ingest-view-local-ebook", classes="ingest-view-area"):
                from ..Widgets.IngestLocalEbookWindow import IngestLocalEbookWindow
                window = IngestLocalEbookWindow(self.app_instance)
                yield from window.compose()
                
            with VerticalScroll(id="ingest-view-local-web", classes="ingest-view-area"):
                window = IngestLocalWebArticleWindow(self.app_instance)
                yield from window.compose()
                
            with VerticalScroll(id="ingest-view-local-xml", classes="ingest-view-area"):
                yield from self.compose_local_xml_tab()
                
            with VerticalScroll(id="ingest-view-local-plaintext", classes="ingest-view-area"):
                window = IngestLocalPlaintextWindow(self.app_instance)
                yield from window.compose()
            
            with VerticalScroll(id="ingest-view-subscriptions", classes="ingest-view-area"):
                yield from self.compose_subscriptions_tab()
            
            # --- TLDW API Views ---
            with VerticalScroll(id="ingest-view-api-video", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("video")
                
            with VerticalScroll(id="ingest-view-api-audio", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("audio")
                
            with VerticalScroll(id="ingest-view-api-document", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("document")
                
            with VerticalScroll(id="ingest-view-api-pdf", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("pdf")
                
            with VerticalScroll(id="ingest-view-api-ebook", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("ebook")
                
            with VerticalScroll(id="ingest-view-api-xml", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("xml")
                
            with VerticalScroll(id="ingest-view-api-mediawiki", classes="ingest-view-area"):
                yield from self.compose_tldw_api_view("mediawiki_dump")
                
            with VerticalScroll(id="ingest-view-api-plaintext", classes="ingest-view-area"):
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
        
        # Handle local clear files buttons
        elif button_id.startswith("local-clear-files-"):
            event.stop()
            media_type = button_id.replace("local-clear-files-", "")
            await self._handle_clear_local_files(f"local_{media_type}")
        
        # Handle local PDF/Ebook browse buttons
        elif button_id.startswith("local-browse-local-files-button-"):
            event.stop()
            media_type = button_id.replace("local-browse-local-files-button-", "")
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
        
        # If IngestWindow has a superclass that also defines on_button_pressed, consider calling it:
        # else:
        #     await super().on_button_pressed(event) # Example if there's a relevant superclass method

    async def _handle_clear_local_files(self, media_type: str) -> None:
        """Clear selected files for a specific media type."""
        try:
            # Clear the stored file list
            if media_type in self.selected_local_files:
                self.selected_local_files[media_type].clear()
                logger.info(f"Cleared selected files for {media_type}")
            
            # Update the ListView
            actual_media_type = media_type.replace("local_", "")
            list_view_id = f"#local-selected-local-files-list-{actual_media_type}"
            
            try:
                list_view = self.query_one(list_view_id, ListView)
                await list_view.clear()
                logger.debug(f"Cleared ListView {list_view_id}")
                self.app_instance.notify(f"Cleared selected {actual_media_type} files")
            except Exception as e:
                logger.error(f"Error clearing ListView {list_view_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error in _handle_clear_local_files: {e}", exc_info=True)
            self.app_instance.notify("Error clearing files", severity="error")

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
            list_view_id = f"#local-selected-local-files-list-{actual_media_type}"
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
    
    @on(Select.Changed, "#local-transcription-provider-video")
    def on_video_provider_changed(self, event: Select.Changed) -> None:
        """Update available models when video provider changes."""
        if event.value and event.value != Select.BLANK and self._local_video_window:
            provider = str(event.value)
            logger.info(f"[IngestWindow] Video transcription provider changed to: {provider}")
            model_select = self.query_one("#local-transcription-model-video", Select)
            self._local_video_window._update_models_for_provider(provider, model_select)
    
    @on(Select.Changed, "#local-transcription-provider-audio")
    def on_audio_provider_changed(self, event: Select.Changed) -> None:
        """Update available models when audio provider changes."""
        if event.value and event.value != Select.BLANK and self._local_audio_window:
            provider = str(event.value)
            logger.info(f"[IngestWindow] Audio transcription provider changed to: {provider}")
            model_select = self.query_one("#local-transcription-model-audio", Select)
            self._local_audio_window._update_models_for_provider(provider, model_select)
    
    def compose_local_video_tab(self) -> ComposeResult:
        """Composes the Video tab content for local media ingestion."""
        window = IngestLocalVideoWindow(self.app_instance)
        # Store reference to initialize models later
        self._local_video_window = window
        yield from window.compose()

    def compose_local_audio_tab(self) -> ComposeResult:
        """Composes the Audio tab content for local media ingestion."""
        window = IngestLocalAudioWindow(self.app_instance)
        # Store reference to initialize models later
        self._local_audio_window = window
        yield from window.compose()

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
            
            # Get chunking options
            perform_chunking = self.query_one("#ingest-local-plaintext-perform-chunking", Checkbox).value
            chunk_method = self.query_one("#ingest-local-plaintext-chunk-method", Select).value
            chunk_size = int(self.query_one("#ingest-local-plaintext-chunk-size", Input).value or "500")
            chunk_overlap = int(self.query_one("#ingest-local-plaintext-chunk-overlap", Input).value or "200")
            
            # If chunk method is Select.BLANK (Default per type), get media-specific defaults
            if chunk_method == Select.BLANK:
                from ..config import get_media_ingestion_defaults
                plaintext_defaults = get_media_ingestion_defaults("plaintext")
                chunk_method = plaintext_defaults.get("chunk_method", "paragraphs")
            
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
                    
                    # Build chunk options dict
                    chunk_options = {
                        'method': chunk_method,
                        'max_size': chunk_size,
                        'overlap': chunk_overlap
                    } if perform_chunking else None
                    
                    # Add to media database
                    media_id, media_uuid, msg = self.app_instance.media_db.add_media_with_keywords(
                        url=str(file_path),
                        title=title,
                        media_type="plaintext",
                        content=content,
                        keywords=keywords,
                        author=author,
                        chunk_options=chunk_options,
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
            
            # Get chunking options
            perform_chunking = self.query_one("#ingest-local-web-perform-chunking", Checkbox).value
            chunk_method = self.query_one("#ingest-local-web-chunk-method", Select).value
            chunk_size = int(self.query_one("#ingest-local-web-chunk-size", Input).value or "500")
            chunk_overlap = int(self.query_one("#ingest-local-web-chunk-overlap", Input).value or "200")
            
            # If chunk method is Select.BLANK (Default per type), get media-specific defaults
            if chunk_method == Select.BLANK:
                from ..config import get_media_ingestion_defaults
                web_article_defaults = get_media_ingestion_defaults("web_article")
                chunk_method = web_article_defaults.get("chunk_method", "paragraphs")
            
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
                'css_selector': css_selector,
                'perform_chunking': perform_chunking,
                'chunk_method': chunk_method,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
            
            # Add stop button
            stop_button = Button("Stop Processing", id="ingest-local-web-stop", variant="error")
            action_section = process_button.parent
            if action_section and not self.query("#ingest-local-web-stop"):
                action_section.mount(stop_button, after=process_button)
            
            # Start worker
            self._web_scraping_worker = self.app_instance.run_worker(
                self._process_urls_worker,
                worker_data,
                thread=True,
                name="web_scraping_worker",
                description="Processing web articles"
            )
            
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
                
                status_area.load_text(status_area.text + summary)
                
                # Show failed URLs section if any
                if failed_urls:
                    status_area.load_text(status_area.text + "\n\n### Failed URLs for retry:\n")
                    for fail in failed_urls:
                        status_area.load_text(status_area.text + f"- {fail['url']} ({fail.get('error', 'Unknown error')})\n")
                    
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
                ),
                context="ingest_urls"
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
    def _process_urls_worker(self, data: dict) -> dict:
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
        
        # Create event loop for async operations
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async scraping logic
            result = loop.run_until_complete(self._process_urls_async(data))
            return result
        finally:
            loop.close()
    
    async def _process_urls_async(self, data: dict) -> dict:
        """Async implementation of URL processing."""
        urls = data['urls']
        custom_cookies = data['custom_cookies']
        title_override = data['title_override']
        author_override = data['author_override']
        keywords = data['keywords']
        js_render = data['js_render']
        css_selector = data['css_selector']
        is_retry = data.get('is_retry', False)
        max_retries = data.get('max_retries', 2)
        
        # Extract chunking options
        perform_chunking = data.get('perform_chunking', True)
        chunk_method = data.get('chunk_method', 'paragraphs')
        chunk_size = data.get('chunk_size', 500)
        chunk_overlap = data.get('chunk_overlap', 200)
        
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
                    
                    # Build chunk options dict
                    chunk_options = {
                        'method': chunk_method,
                        'max_size': chunk_size,
                        'overlap': chunk_overlap
                    } if perform_chunking else None
                    
                    # Add to media database
                    media_id, media_uuid, msg = self.app_instance.media_db.add_media_with_keywords(
                        url=url,
                        title=title,
                        media_type="web_article",
                        content=content,
                        keywords=keywords,
                        author=author,
                        chunk_options=chunk_options,
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
            status_area.load_text(status_area.text + f"\n{message}")
            
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
                status_area.load_text(status_area.text + "\n\n⚠️ Processing stopped by user")
                
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
        status_area.load_text(status_area.text + "\n\n## Retrying Failed URLs...\n")
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
            self._web_scraping_worker = self.app_instance.run_worker(
                self._process_urls_worker,
                worker_data,
                thread=True,
                name="web_scraping_retry_worker",
                description="Retrying failed web articles"
            )
            
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
                
                status_area.load_text(status_area.text + summary)
                
                # Show failed URLs section if any still remain
                if failed_urls:
                    status_area.load_text(status_area.text + "\n\n### Still Failed URLs:\n")
                    for fail in failed_urls:
                        status_area.load_text(status_area.text + f"- {fail['url']} ({fail.get('error', 'Unknown error')}) - Retry attempts: {fail.get('retry_count', 0)}\n")
                    
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
            status_area.load_text(status_area.text + f"\nError during retry: {str(e)}")
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
    
    async def handle_local_pdf_process(self) -> None:
        """Handle processing of local PDF files."""
        logger.info("Processing local PDF files")
        
        # Get UI elements
        try:
            loading_indicator = self.query_one("#local-loading-indicator-pdf", LoadingIndicator)
            status_area = self.query_one("#local-status-area-pdf", TextArea)
            process_button = self.query_one("#local-submit-pdf", Button)
        except Exception as e:
            logger.error(f"Error finding UI elements: {e}")
            self.app_instance.notify("Error: UI elements not found", severity="error")
            return
        
        # Show loading state
        loading_indicator.display = True
        loading_indicator.classes = loading_indicator.classes - {"hidden"}
        status_area.clear()
        status_area.load_text("Processing PDF files locally...")
        status_area.display = True
        status_area.classes = status_area.classes - {"hidden"}
        process_button.disabled = True
        
        try:
            # Get selected files
            local_key = "local_pdf"
            selected_files = self.selected_local_files.get(local_key, [])
            
            # Also check URLs
            urls_textarea = self.query_one("#local-urls-pdf", TextArea)
            urls_text = urls_textarea.text.strip()
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            
            if not selected_files and not urls:
                self.app_instance.notify("Please select at least one PDF file or provide URLs", severity="warning")
                return
            
            # Get processing options
            pdf_engine_select = self.query_one("#local-pdf-engine-pdf", Select)
            pdf_engine = str(pdf_engine_select.value)
            
            # Get metadata
            title_override = self.query_one("#local-title-pdf", Input).value.strip()
            author = self.query_one("#local-author-pdf", Input).value.strip()
            keywords_text = self.query_one("#local-keywords-pdf", TextArea).text.strip()
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()] if keywords_text else []
            
            # Get processing options
            perform_analysis = self.query_one("#local-perform-analysis-pdf", Checkbox).value
            custom_prompt = self.query_one("#local-custom-prompt-pdf", TextArea).text.strip()
            system_prompt = self.query_one("#local-system-prompt-pdf", TextArea).text.strip()
            
            # Get API options for analysis
            api_name = None
            api_key = None
            if perform_analysis:
                api_name_select = self.query_one("#local-analysis-api-name-pdf", Select)
                if api_name_select.value != Select.BLANK:
                    api_name = str(api_name_select.value)
                    api_key_input = self.query_one("#local-analysis-api-key-pdf", Input)
                    api_key = api_key_input.value.strip() if api_key_input.value else None
                    
                    # If no API key provided in UI, try to get from config
                    if not api_key and api_name:
                        from ..config import get_api_key
                        api_key = get_api_key(api_name)
            
            # Get chunking options
            perform_chunking = self.query_one("#local-perform-chunking-pdf", Checkbox).value
            chunk_method = self.query_one("#local-chunk-method-pdf", Select).value
            chunk_size = int(self.query_one("#local-chunk-size-pdf", Input).value or "500")
            chunk_overlap = int(self.query_one("#local-chunk-overlap-pdf", Input).value or "200")
            
            # If chunk method is Select.BLANK (Default per type), get media-specific defaults
            if chunk_method == Select.BLANK:
                from ..config import get_media_ingestion_defaults
                pdf_defaults = get_media_ingestion_defaults("pdf")
                chunk_method = pdf_defaults.get("chunk_method", "semantic")
            
            # Check if media DB is available
            if not self.app_instance.media_db:
                logger.error("Media database not initialized")
                self.app_instance.notify("Error: Media database not available", severity="error")
                status_area.load_text("Error: Media database not available")
                return
            
            # Import the local PDF processing function
            try:
                from ..Local_Ingestion.PDF_Processing_Lib import process_pdf
            except ImportError as e:
                logger.error(f"Failed to import PDF processing library: {e}")
                self.app_instance.notify("Error: PDF processing library not available. Please install with: pip install tldw-chatbook[pdf]", severity="error")
                status_area.load_text("Error: PDF processing library not available.\nPlease install with: pip install tldw-chatbook[pdf]")
                return
            
            # Process files
            processed_count = 0
            error_count = 0
            status_messages = []
            
            # Process local files
            for file_path in selected_files:
                try:
                    status_area.load_text(status_area.text + f"\nProcessing: {file_path.name}...")
                    
                    # Build chunk options dict
                    chunk_options = {
                        'method': chunk_method,  # chunk_method already has the proper default
                        'max_size': chunk_size,
                        'overlap': chunk_overlap
                    } if perform_chunking else None
                    
                    # Process PDF using local library
                    def process_single_pdf():
                        return process_pdf(
                            file_input=str(file_path),
                            filename=file_path.name,
                            parser=pdf_engine,
                            title_override=title_override,
                            author_override=author,
                            keywords=keywords,
                            perform_chunking=perform_chunking,
                            chunk_options=chunk_options,
                            perform_analysis=perform_analysis,
                            api_name=api_name,
                            api_key=api_key,
                            custom_prompt=custom_prompt if custom_prompt else None,
                            system_prompt=system_prompt if system_prompt else None,
                            summarize_recursively=False  # TODO: Add to UI
                        )
                    
                    # Run in worker thread
                    worker = self.app_instance.run_worker(
                        process_single_pdf,
                        thread=True,
                        name=f"pdf_process_{file_path.name}",
                        description=f"Processing {file_path.name}"
                    )
                    
                    # Wait for the worker to complete
                    result = await worker.wait()
                    
                    if result and result.get('status') in ['Success', 'Warning']:
                        # Extract content and metadata
                        content = result.get('content', '')
                        title = title_override or result.get('metadata', {}).get('title', file_path.stem)
                        
                        # Add to media database
                        media_id, media_uuid, msg = self.app_instance.media_db.add_media_with_keywords(
                            url=str(file_path),
                            title=title,
                            media_type="pdf",
                            content=content,
                            keywords=keywords,
                            author=author,
                            analysis_content=result.get('analysis', ''),
                            chunks=result.get('chunks', []),
                            chunk_options=chunk_options,
                            prompt=custom_prompt if custom_prompt else None
                        )
                        
                        if media_id:
                            processed_count += 1
                            status_messages.append(f"✅ {title} - ID: {media_id}")
                            status_area.load_text(status_area.text + f"\n✅ Successfully processed: {title}")
                        else:
                            error_count += 1
                            status_messages.append(f"❌ {file_path.name} - Database error: {msg}")
                            status_area.load_text(status_area.text + f"\n❌ Database error for {file_path.name}: {msg}")
                    else:
                        error_count += 1
                        error_msg = result.get('error', 'Unknown error') if result else 'Processing failed'
                        status_messages.append(f"❌ {file_path.name} - {error_msg}")
                        status_area.load_text(status_area.text + f"\n❌ Failed to process {file_path.name}: {error_msg}")
                        
                except Exception as e:
                    error_count += 1
                    error_msg = str(e)
                    status_messages.append(f"❌ {file_path.name} - {error_msg}")
                    status_area.load_text(status_area.text + f"\n❌ Error processing {file_path.name}: {error_msg}")
                    logger.error(f"Error processing PDF {file_path}: {e}", exc_info=True)
            
            # Process URLs if any
            if urls:
                status_area.load_text(status_area.text + f"\n\nProcessing {len(urls)} URLs...")
                # URLs would need web scraping support - for now just notify
                status_area.load_text(status_area.text + "\n⚠️ URL processing for PDFs requires web scraping support")
            
            # Final summary
            status_area.load_text(status_area.text + f"\n\n## Processing Complete\n")
            status_area.load_text(status_area.text + f"✅ Successfully processed: {processed_count} files\n")
            if error_count > 0:
                status_area.load_text(status_area.text + f"❌ Errors: {error_count} files\n")
            
            # Notifications
            if processed_count > 0:
                self.app_instance.notify(f"Successfully processed {processed_count} PDF files", severity="information")
            if error_count > 0:
                self.app_instance.notify(f"Failed to process {error_count} PDF files", severity="warning")
                
        except Exception as e:
            logger.error(f"Error in PDF processing: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
            status_area.load_text(status_area.text + f"\n\nError: {str(e)}")
        finally:
            # Reset UI state
            loading_indicator.display = False
            loading_indicator.classes = loading_indicator.classes | {"hidden"}
            process_button.disabled = False
    
    async def handle_local_ebook_process(self) -> None:
        """Handle processing of local ebook files."""
        logger.info("Processing local ebook files")
        
        # Get UI elements
        try:
            loading_indicator = self.query_one("#local-loading-indicator-ebook", LoadingIndicator)
            status_area = self.query_one("#local-status-area-ebook", TextArea)
            process_button = self.query_one("#local-submit-ebook", Button)
        except Exception as e:
            logger.error(f"Error finding UI elements: {e}")
            self.app_instance.notify("Error: UI elements not found", severity="error")
            return
        
        # Show loading state
        loading_indicator.display = True
        loading_indicator.classes = loading_indicator.classes - {"hidden"}
        status_area.clear()
        status_area.load_text("Processing ebook files locally...")
        status_area.display = True
        status_area.classes = status_area.classes - {"hidden"}
        process_button.disabled = True
        
        try:
            # Get selected files
            local_key = "local_ebook"
            selected_files = self.selected_local_files.get(local_key, [])
            
            # Also check URLs
            urls_textarea = self.query_one("#local-urls-ebook", TextArea)
            urls_text = urls_textarea.text.strip()
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            
            if not selected_files and not urls:
                self.app_instance.notify("Please select at least one ebook file or provide URLs", severity="warning")
                return
            
            # Get processing options
            extraction_method_select = self.query_one("#local-ebook-extraction-method-ebook", Select)
            extraction_method = str(extraction_method_select.value)
            
            # Get metadata
            title_override = self.query_one("#local-title-ebook", Input).value.strip()
            author = self.query_one("#local-author-ebook", Input).value.strip()
            keywords_text = self.query_one("#local-keywords-ebook", TextArea).text.strip()
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()] if keywords_text else []
            
            # Get processing options
            perform_analysis = self.query_one("#local-perform-analysis-ebook", Checkbox).value
            custom_prompt = self.query_one("#local-custom-prompt-ebook", TextArea).text.strip()
            system_prompt = self.query_one("#local-system-prompt-ebook", TextArea).text.strip()
            
            # Get API options for analysis
            api_name = None
            api_key = None
            if perform_analysis:
                api_name_select = self.query_one("#local-analysis-api-name-ebook", Select)
                if api_name_select.value != Select.BLANK:
                    api_name = str(api_name_select.value)
                    api_key_input = self.query_one("#local-analysis-api-key-ebook", Input)
                    api_key = api_key_input.value.strip() if api_key_input.value else None
                    
                    # If no API key provided in UI, try to get from config
                    if not api_key and api_name:
                        from ..config import get_api_key
                        api_key = get_api_key(api_name)
            
            # Get chunking options
            perform_chunking = self.query_one("#local-perform-chunking-ebook", Checkbox).value
            chunk_method = self.query_one("#local-chunk-method-ebook", Select).value
            chunk_size = int(self.query_one("#local-chunk-size-ebook", Input).value or "500")
            chunk_overlap = int(self.query_one("#local-chunk-overlap-ebook", Input).value or "200")
            
            # If chunk method is Select.BLANK (Default per type), get media-specific defaults
            if chunk_method == Select.BLANK:
                from ..config import get_media_ingestion_defaults
                ebook_defaults = get_media_ingestion_defaults("ebook")
                chunk_method = ebook_defaults.get("chunk_method", "ebook_chapters")
            
            # Check if media DB is available
            if not self.app_instance.media_db:
                logger.error("Media database not initialized")
                self.app_instance.notify("Error: Media database not available", severity="error")
                status_area.load_text("Error: Media database not available")
                return
            
            # Import the local ebook processing function
            try:
                from ..Local_Ingestion.Book_Ingestion_Lib import process_ebook
            except ImportError as e:
                logger.error(f"Failed to import ebook processing library: {e}")
                self.app_instance.notify("Error: Ebook processing library not available. Please install with: pip install tldw-chatbook[ebook]", severity="error")
                status_area.load_text("Error: Ebook processing library not available.\nPlease install with: pip install tldw-chatbook[ebook]")
                return
            
            # Process files
            processed_count = 0
            error_count = 0
            status_messages = []
            
            # Process local files
            for file_path in selected_files:
                try:
                    status_area.load_text(status_area.text + f"\nProcessing: {file_path.name}...")
                    
                    # Process ebook using local library
                    # Build chunk options dict
                    chunk_options = {
                        'method': chunk_method,  # chunk_method already has the proper default
                        'max_size': chunk_size,
                        'overlap': chunk_overlap
                    } if perform_chunking else None
                    
                    # Define the processing function
                    def process_single_ebook():
                        return process_ebook(
                            file_path=str(file_path),
                            title_override=title_override,
                            author_override=author,
                            keywords=keywords,
                            custom_prompt=custom_prompt if custom_prompt else None,
                            system_prompt=system_prompt if system_prompt else None,
                            perform_chunking=perform_chunking,
                            chunk_options=chunk_options,
                            perform_analysis=perform_analysis,
                            api_name=api_name,
                            api_key=api_key,
                            summarize_recursively=False,  # TODO: Add to UI
                            extraction_method=extraction_method
                        )
                    
                    # Run in worker thread
                    worker = self.app_instance.run_worker(
                        process_single_ebook,
                        thread=True,
                        name=f"ebook_process_{file_path.name}",
                        description=f"Processing {file_path.name}"
                    )
                    
                    # Wait for the worker to complete
                    result = await worker.wait()
                    
                    if result and result.get('status') in ['Success', 'Warning']:
                        # Extract content and metadata
                        content = result.get('content', '')
                        title = title_override or result.get('metadata', {}).get('title', file_path.stem)
                        book_author = result.get('metadata', {}).get('author', '')
                        
                        # Add to media database
                        media_id, media_uuid, msg = self.app_instance.media_db.add_media_with_keywords(
                            url=str(file_path),
                            title=title,
                            media_type="ebook",
                            content=content,
                            keywords=keywords,
                            author=author or book_author,
                            analysis_content=result.get('analysis', ''),
                            chunks=result.get('chunks', []),
                            chunk_options=chunk_options,
                            prompt=custom_prompt if custom_prompt else None
                        )
                        
                        if media_id:
                            processed_count += 1
                            status_messages.append(f"✅ {title} - ID: {media_id}")
                            status_area.load_text(status_area.text + f"\n✅ Successfully processed: {title}")
                        else:
                            error_count += 1
                            status_messages.append(f"❌ {file_path.name} - Database error: {msg}")
                            status_area.load_text(status_area.text + f"\n❌ Database error for {file_path.name}: {msg}")
                    else:
                        error_count += 1
                        error_msg = result.get('error', 'Unknown error') if result else 'Processing failed'
                        status_messages.append(f"❌ {file_path.name} - {error_msg}")
                        status_area.load_text(status_area.text + f"\n❌ Failed to process {file_path.name}: {error_msg}")
                        
                except Exception as e:
                    error_count += 1
                    error_msg = str(e)
                    status_messages.append(f"❌ {file_path.name} - {error_msg}")
                    status_area.load_text(status_area.text + f"\n❌ Error processing {file_path.name}: {error_msg}")
                    logger.error(f"Error processing ebook {file_path}: {e}", exc_info=True)
            
            # Process URLs if any
            if urls:
                status_area.load_text(status_area.text + f"\n\nProcessing {len(urls)} URLs...")
                # URLs would need web scraping support - for now just notify
                status_area.load_text(status_area.text + "\n⚠️ URL processing for ebooks requires web scraping support")
            
            # Final summary
            status_area.load_text(status_area.text + f"\n\n## Processing Complete\n")
            status_area.load_text(status_area.text + f"✅ Successfully processed: {processed_count} files\n")
            if error_count > 0:
                status_area.load_text(status_area.text + f"❌ Errors: {error_count} files\n")
            
            # Notifications
            if processed_count > 0:
                self.app_instance.notify(f"Successfully processed {processed_count} ebook files", severity="information")
            if error_count > 0:
                self.app_instance.notify(f"Failed to process {error_count} ebook files", severity="warning")
                
        except Exception as e:
            logger.error(f"Error in ebook processing: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
            status_area.load_text(status_area.text + f"\n\nError: {str(e)}")
        finally:
            # Reset UI state
            loading_indicator.display = False
            loading_indicator.classes = loading_indicator.classes | {"hidden"}
            process_button.disabled = False
    
    async def handle_local_document_process(self) -> None:
        """Handle processing of local document files."""
        logger.info("Processing local document files")
        
        # Get UI elements
        try:
            loading_indicator = self.query_one("#local-loading-indicator-document", LoadingIndicator)
            status_area = self.query_one("#local-status-area-document", TextArea)
            process_button = self.query_one("#local-process-button-document", Button)
        except Exception as e:
            logger.error(f"Error finding UI elements: {e}")
            self.app_instance.notify("Error: UI elements not found", severity="error")
            return
        
        # Show loading state
        loading_indicator.display = True
        loading_indicator.classes = loading_indicator.classes - {"hidden"}
        status_area.clear()
        status_area.load_text("Processing document files locally...")
        status_area.display = True
        status_area.classes = status_area.classes - {"hidden"}
        process_button.disabled = True
        
        try:
            # Get selected files
            local_key = "local_document"
            selected_files = self.selected_local_files.get(local_key, [])
            
            if not selected_files:
                self.app_instance.notify("Please select at least one document file", severity="warning")
                return
            
            # Get processing method
            processing_method_select = self.query_one("#local-processing-method-document", Select)
            processing_method = str(processing_method_select.value)
            
            # Get metadata
            title_override = self.query_one("#local-title-document", Input).value.strip()
            author = self.query_one("#local-author-document", Input).value.strip()
            keywords_text = self.query_one("#local-keywords-document", TextArea).text.strip()
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()] if keywords_text else []
            
            # Get processing options
            perform_analysis = self.query_one("#local-perform-analysis-document", Checkbox).value
            custom_prompt = self.query_one("#local-custom-prompt-document", TextArea).text.strip()
            system_prompt = self.query_one("#local-system-prompt-document", TextArea).text.strip()
            
            # Get API options for analysis
            api_name = None
            api_key = None
            if perform_analysis:
                api_name_select = self.query_one("#local-analysis-api-name-document", Select)
                if api_name_select.value != Select.BLANK:
                    api_name = str(api_name_select.value)
                    api_key_input = self.query_one("#local-analysis-api-key-document", Input)
                    api_key = api_key_input.value.strip() if api_key_input.value else None
                    
                    # If no API key provided in UI, try to get from config
                    if not api_key and api_name:
                        from ..config import get_api_key
                        api_key = get_api_key(api_name)
            
            # Get chunking options
            perform_chunking = self.query_one("#local-perform-chunking-document", Checkbox).value
            chunk_method = self.query_one("#local-chunk-method-document", Select).value
            chunk_size = int(self.query_one("#local-chunk-size-document", Input).value or "1500")
            chunk_overlap = int(self.query_one("#local-chunk-overlap-document", Input).value or "100")
            
            # If chunk method is Select.BLANK (Default per type), get media-specific defaults
            if chunk_method == Select.BLANK:
                from ..config import get_media_ingestion_defaults
                document_defaults = get_media_ingestion_defaults("document")
                chunk_method = document_defaults.get("chunk_method", "sentences")
            
            # Get document-specific options
            extract_tables = self.query_one("#local-extract-tables-document", Checkbox).value
            preserve_formatting = self.query_one("#local-preserve-formatting-document", Checkbox).value
            include_metadata = self.query_one("#local-include-metadata-document", Checkbox).value
            
            # Check if media DB is available
            if not self.app_instance.media_db:
                logger.error("Media database not initialized")
                self.app_instance.notify("Error: Media database not available", severity="error")
                status_area.load_text("Error: Media database not available")
                return
            
            # Import the local document processing function
            try:
                from ..Local_Ingestion.Document_Processing_Lib import process_document
            except ImportError as e:
                logger.error(f"Failed to import document processing library: {e}")
                self.app_instance.notify("Error: Document processing library not available. Please install with: pip install tldw-chatbook[documents]", severity="error")
                status_area.load_text("Error: Document processing library not available.\nPlease install with: pip install tldw-chatbook[documents]")
                return
            
            # Process files
            processed_count = 0
            error_count = 0
            status_messages = []
            
            # Build chunk options dict
            chunk_options = {
                'method': chunk_method,
                'max_size': chunk_size,
                'overlap': chunk_overlap
            } if perform_chunking else None
            
            # Process local files
            for file_path in selected_files:
                try:
                    status_area.load_text(status_area.text + f"\nProcessing: {file_path.name}...")
                    
                    # Process document using local library
                    def process_single_document():
                        return process_document(
                            file_path=str(file_path),
                            title_override=title_override,
                            author_override=author,
                            keywords=keywords,
                            custom_prompt=custom_prompt if custom_prompt else None,
                            system_prompt=system_prompt if system_prompt else None,
                            auto_summarize=perform_analysis,
                            api_name=api_name,
                            api_key=api_key,
                            chunk_options=chunk_options,
                            processing_method=processing_method
                        )
                    
                    # Run in worker thread
                    worker = self.app_instance.run_worker(
                        process_single_document,
                        thread=True,
                        name=f"document_process_{file_path.name}",
                        description=f"Processing {file_path.name}"
                    )
                    
                    # Wait for the worker to complete
                    result = await worker.wait()
                    
                    if result and result.get('extraction_successful'):
                        # Extract content and metadata
                        content = result.get('content', '')
                        title = title_override or result.get('title', file_path.stem)
                        summary = result.get('summary', '')
                        metadata = result.get('metadata', {})
                        
                        # Add to media database
                        media_id, media_uuid, msg = self.app_instance.media_db.add_media_with_keywords(
                            url=str(file_path),
                            title=title,
                            media_type="document",
                            content=content,
                            keywords=keywords,
                            author=author,
                            analysis_content=summary,
                            chunks=None,  # Chunking will be handled by the database
                            chunk_options=chunk_options,
                            prompt=custom_prompt if custom_prompt else None,
                            metadata=metadata
                        )
                        
                        if media_id:
                            processed_count += 1
                            status_messages.append(f"✅ {title} - ID: {media_id}")
                            status_area.load_text(status_area.text + f"\n✅ Successfully processed: {title}")
                        else:
                            error_count += 1
                            status_messages.append(f"❌ {file_path.name} - Database error: {msg}")
                            status_area.load_text(status_area.text + f"\n❌ Database error for {file_path.name}: {msg}")
                    else:
                        error_count += 1
                        error_msg = result.get('metadata', {}).get('error', 'Unknown error') if result else 'Processing failed'
                        status_messages.append(f"❌ {file_path.name} - {error_msg}")
                        status_area.load_text(status_area.text + f"\n❌ Failed to process {file_path.name}: {error_msg}")
                        
                except Exception as e:
                    error_count += 1
                    error_msg = str(e)
                    status_messages.append(f"❌ {file_path.name} - {error_msg}")
                    status_area.load_text(status_area.text + f"\n❌ Error processing {file_path.name}: {error_msg}")
                    logger.error(f"Error processing document {file_path}: {e}", exc_info=True)
            
            # Final summary
            status_area.load_text(status_area.text + f"\n\n## Processing Complete\n")
            status_area.load_text(status_area.text + f"✅ Successfully processed: {processed_count} files\n")
            if error_count > 0:
                status_area.load_text(status_area.text + f"❌ Errors: {error_count} files\n")
            
            # Notifications
            if processed_count > 0:
                self.app_instance.notify(f"Successfully processed {processed_count} document files", severity="information")
            if error_count > 0:
                self.app_instance.notify(f"Failed to process {error_count} document files", severity="warning")
                
        except Exception as e:
            logger.error(f"Error in document processing: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
            status_area.load_text(status_area.text + f"\n\nError: {str(e)}")
        finally:
            # Reset UI state
            loading_indicator.display = False
            loading_indicator.classes = loading_indicator.classes | {"hidden"}
            process_button.disabled = False
    
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

    async def handle_local_audio_process(self) -> None:
        """Handle processing of local audio files."""
        logger.info("Processing local audio files")
        
        # Get UI elements
        try:
            loading_indicator = self.query_one("#local-loading-indicator-audio", LoadingIndicator)
            status_area = self.query_one("#local-status-audio", TextArea)
            process_button = self.query_one("#local-submit-audio", Button)
        except Exception as e:
            logger.error(f"Error finding UI elements: {e}")
            self.app_instance.notify("Error: UI elements not found", severity="error")
            return
        
        # Show loading state
        loading_indicator.display = True
        loading_indicator.classes = loading_indicator.classes - {"hidden"}
        status_area.clear()
        status_area.load_text("Processing audio files locally...")
        status_area.display = True
        status_area.classes = status_area.classes - {"hidden"}
        process_button.disabled = True
        
        try:
            # Get selected files
            local_key = "local_audio"
            selected_files = self.selected_local_files.get(local_key, [])
            
            # Also check URLs
            urls_textarea = self.query_one("#local-urls-audio", TextArea)
            urls_text = urls_textarea.text.strip()
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            
            # Combine all inputs
            all_inputs = []
            if selected_files:
                all_inputs.extend([str(f) for f in selected_files])
            if urls:
                all_inputs.extend(urls)
            
            if not all_inputs:
                self.app_instance.notify("Please select at least one audio file or provide URLs", severity="warning")
                return
            
            # Get metadata
            title_override = self.query_one("#local-title-audio", Input).value.strip()
            author = self.query_one("#local-author-audio", Input).value.strip()
            keywords_text = self.query_one("#local-keywords-audio", TextArea).text.strip()
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()] if keywords_text else []
            
            # Get transcription options
            transcription_provider = self.query_one("#local-transcription-provider-audio", Select).value
            # Get the model ID directly from the Select widget
            transcription_model = self.query_one("#local-transcription-model-audio", Select).value
            transcription_language = self.query_one("#local-transcription-language-audio", Input).value.strip() or "en"
            # Get translation target if available
            translation_target = None
            try:
                translation_container = self.query_one("#local-translation-container-audio", Container)
                if not translation_container.has_class("hidden"):
                    translation_target_input = self.query_one("#local-translation-target-audio", Input)
                    translation_target = translation_target_input.value.strip() if translation_target_input.value else None
            except Exception:
                pass
            vad_filter = self.query_one("#local-vad-filter-audio", Checkbox).value
            diarize = self.query_one("#local-diarize-audio", Checkbox).value
            timestamps = self.query_one("#local-timestamps-audio", Checkbox).value
            
            # Get time range options
            start_time = self.query_one("#local-start-time-audio", Input).value.strip()
            end_time = self.query_one("#local-end-time-audio", Input).value.strip()
            
            # Get processing options
            perform_analysis = self.query_one("#local-perform-analysis-audio", Checkbox).value
            custom_prompt = self.query_one("#local-custom-prompt-audio", TextArea).text.strip()
            system_prompt = self.query_one("#local-system-prompt-audio", TextArea).text.strip()
            
            # Get API options for analysis
            api_name = None
            api_key = None
            if perform_analysis:
                api_name_select = self.query_one("#local-analysis-api-name-audio", Select)
                if api_name_select.value != Select.BLANK:
                    api_name = str(api_name_select.value)
                    api_key_input = self.query_one("#local-analysis-api-key-audio", Input)
                    api_key = api_key_input.value.strip() if api_key_input.value else None
                    
                    # If no API key provided in UI, try to get from config
                    if not api_key and api_name:
                        from ..config import get_api_key
                        api_key = get_api_key(api_name)
            
            # Get chunking options
            perform_chunking = self.query_one("#local-perform-chunking-audio", Checkbox).value
            chunk_method = self.query_one("#local-chunk-method-audio", Select).value
            chunk_size = int(self.query_one("#local-chunk-size-audio", Input).value or "500")
            chunk_overlap = int(self.query_one("#local-chunk-overlap-audio", Input).value or "200")
            use_adaptive_chunking = self.query_one("#local-use-adaptive-chunking-audio", Checkbox).value
            use_multi_level_chunking = self.query_one("#local-use-multi-level-chunking-audio", Checkbox).value
            chunk_language = self.query_one("#local-chunk-language-audio", Input).value.strip()
            summarize_recursively = self.query_one("#local-summarize-recursively-audio", Checkbox).value
            
            # Get cookie options
            use_cookies = self.query_one("#local-use-cookies-audio", Checkbox).value
            cookies = self.query_one("#local-cookies-audio", TextArea).text.strip()
            
            # Other options
            keep_original = self.query_one("#local-keep-original-audio", Checkbox).value
            
            # Validate transcription model
            if transcription_model == Select.BLANK or not transcription_model:
                if transcription_provider and transcription_provider != Select.BLANK:
                    transcription_model = self.get_default_model_for_provider(str(transcription_provider))
                    logger.warning(f"Transcription model was blank, using default for {transcription_provider}: {transcription_model}")
                else:
                    # If no provider selected either, use a sensible default
                    transcription_model = "base"
                    logger.warning("Both transcription model and provider were blank, using default model: base")
            
            # Convert Select values to strings after validation
            if transcription_provider != Select.BLANK:
                transcription_provider = str(transcription_provider)
            else:
                transcription_provider = "faster-whisper"  # Default provider
            
            # Ensure transcription_model is a string
            transcription_model = str(transcription_model)
            
            # Check if media DB is available
            if not self.app_instance.media_db:
                logger.error("Media database not initialized")
                self.app_instance.notify("Error: Media database not available", severity="error")
                status_area.load_text("Error: Media database not available")
                return
            
            # Import the audio processing function
            try:
                from ..Local_Ingestion import LocalAudioProcessor
            except ImportError as e:
                logger.error(f"Failed to import audio processing library: {e}")
                self.app_instance.notify("Error: Audio processing library not available. Please install with: pip install tldw-chatbook[audio]", severity="error")
                status_area.load_text("Error: Audio processing library not available.\nPlease install with: pip install tldw-chatbook[audio]")
                return
            
            # Create processor instance
            processor = LocalAudioProcessor(self.app_instance.media_db)
            
            # Process audio files
            status_area.load_text("Processing audio files...\n")
            
            results = processor.process_audio_files(
                inputs=all_inputs,
                transcription_provider=transcription_provider,
                transcription_model=transcription_model,
                transcription_language=transcription_language,
                translation_target_language=translation_target,
                perform_chunking=perform_chunking,
                chunk_method=chunk_method,
                max_chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_adaptive_chunking=use_adaptive_chunking,
                use_multi_level_chunking=use_multi_level_chunking,
                chunk_language=chunk_language or transcription_language,
                diarize=diarize,
                vad_use=vad_filter,
                timestamp_option=timestamps,
                start_time=start_time if start_time else None,
                end_time=end_time if end_time else None,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                summarize_recursively=summarize_recursively,
                use_cookies=use_cookies,
                cookies=cookies,
                keep_original=keep_original,
                custom_title=title_override,
                author=author
            )
            
            # Display results
            processed_count = results.get("processed_count", 0)
            errors_count = results.get("errors_count", 0)
            
            status_messages = [f"Processing complete: {processed_count} succeeded, {errors_count} failed\n"]
            
            for result in results.get("results", []):
                input_ref = result.get("input_ref", "Unknown")
                status = result.get("status", "Unknown")
                title = result.get("metadata", {}).get("title", "Untitled")
                
                if status == "Success":
                    status_messages.append(f"✓ {title} ({input_ref})")
                else:
                    error = result.get("error", "Unknown error")
                    status_messages.append(f"✗ {input_ref}: {error}")
            
            status_area.load_text("\n".join(status_messages))
            
            if processed_count > 0:
                self.app_instance.notify(f"Successfully processed {processed_count} audio file(s)", severity="information")
            if errors_count > 0:
                self.app_instance.notify(f"{errors_count} file(s) failed to process", severity="warning")
            
        except Exception as e:
            logger.error(f"Error processing audio files: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
            status_area.load_text(f"Error: {str(e)}")
        finally:
            # Hide loading state
            loading_indicator.display = False
            loading_indicator.classes = loading_indicator.classes | {"hidden"}
            process_button.disabled = False

    async def handle_local_video_process(self) -> None:
        """Handle processing of local video files."""
        logger.info("Processing local video files")
        
        # Get UI elements
        try:
            loading_indicator = self.query_one("#local-loading-indicator-video", LoadingIndicator)
            status_area = self.query_one("#local-status-video", TextArea)
            process_button = self.query_one("#local-submit-video", Button)
        except Exception as e:
            logger.error(f"Error finding UI elements: {e}")
            self.app_instance.notify("Error: UI elements not found", severity="error")
            return
        
        # Collect all UI values before starting the worker
        try:
            # Get selected files
            local_key = "local_video"
            selected_files = self.selected_local_files.get(local_key, [])
            
            # Get URL input (TextArea for multiple URLs)
            url_input = self.query_one("#local-urls-video", TextArea).text.strip()
            
            # Prepare inputs list
            all_inputs = []
            if url_input:
                # Split by newlines for multiple URLs
                urls = [url.strip() for url in url_input.split('\n') if url.strip()]
                all_inputs.extend(urls)
            
            all_inputs.extend(selected_files)
            
            if not all_inputs:
                self.app_instance.notify("No video files or URLs selected for processing", severity="warning")
                loading_indicator.display = False
                loading_indicator.classes = loading_indicator.classes | {"hidden"}
                process_button.disabled = False
                return
            
            # Collect all processing options
            options = {
                "inputs": all_inputs,
                "download_video": self.query_one("#local-download-video-video", Checkbox).value,
                "extract_audio_only": self.query_one("#local-extract-audio-only-video", Checkbox).value,
                "start_time": self.query_one("#local-start-time-video", Input).value.strip(),
                "end_time": self.query_one("#local-end-time-video", Input).value.strip(),
                "title_override": self.query_one("#local-title-video", Input).value.strip(),
                "author": self.query_one("#local-author-video", Input).value.strip(),
                
                # Keywords
                "keywords": [k.strip() for k in self.query_one("#local-keywords-video", TextArea).text.strip().split(',') if k.strip()],
                
                # Transcription options
                "transcription_provider": self.query_one("#local-transcription-provider-video", Select).value,
                # Get the model ID directly from the Select widget
                "transcription_model": self.query_one("#local-transcription-model-video", Select).value,
                "transcription_language": self.query_one("#local-transcription-language-video", Input).value.strip(),
                "translation_target": self.query_one("#local-translation-target-video", Input).value.strip(),
                
                "vad_filter": self.query_one("#local-vad-filter-video", Checkbox).value,
                "diarize": self.query_one("#local-diarize-video", Checkbox).value,
                "timestamps": self.query_one("#local-timestamps-video", Checkbox).value,
                
                # Processing options
                "perform_analysis": self.query_one("#local-perform-analysis-video", Checkbox).value,
                "custom_prompt": self.query_one("#local-custom-prompt-video", TextArea).text.strip(),
                "system_prompt": self.query_one("#local-system-prompt-video", TextArea).text.strip(),
                
                # Chunking options
                "perform_chunking": self.query_one("#local-perform-chunking-video", Checkbox).value,
                "chunk_method": self.query_one("#local-chunk-method-video", Select).value,
                "chunk_size": int(self.query_one("#local-chunk-size-video", Input).value or "500"),
                "chunk_overlap": int(self.query_one("#local-chunk-overlap-video", Input).value or "200"),
                "use_adaptive_chunking": self.query_one("#local-use-adaptive-chunking-video", Checkbox).value,
                "use_multi_level_chunking": self.query_one("#local-use-multi-level-chunking-video", Checkbox).value,
                "chunk_language": self.query_one("#local-chunk-language-video", Input).value.strip(),
                "summarize_recursively": self.query_one("#local-summarize-recursively-video", Checkbox).value,
                
                # Cookie options
                "use_cookies": self.query_one("#local-use-cookies-video", Checkbox).value,
                "cookies": self.query_one("#local-cookies-video", TextArea).text.strip(),
                
                # Other options
                "keep_original": self.query_one("#local-keep-original-video", Checkbox).value,
            }
            
            # Get API options for analysis if needed
            if options["perform_analysis"]:
                api_name_select = self.query_one("#local-analysis-api-name-video", Select)
                if api_name_select.value != Select.BLANK:
                    options["api_name"] = str(api_name_select.value)
                    api_key_input = self.query_one("#local-analysis-api-key-video", Input)
                    options["api_key"] = api_key_input.value.strip() if api_key_input.value else None
                    
                    # Try to get analysis model if available
                    try:
                        analysis_model_select = self.query_one("#local-analysis-model-video", Select)
                        if analysis_model_select.value != Select.BLANK:
                            options["analysis_model"] = str(analysis_model_select.value)
                    except Exception:
                        # Field might not exist in older UI versions
                        pass
                    
                    # If no API key provided in UI, try to get from config
                    if not options.get("api_key") and options.get("api_name"):
                        from ..config import get_api_key
                        options["api_key"] = get_api_key(options["api_name"])
            
            # Validate transcription model
            if options["transcription_model"] == Select.BLANK or not options["transcription_model"]:
                provider = options.get("transcription_provider")
                if provider and provider != Select.BLANK:
                    options["transcription_model"] = self.get_default_model_for_provider(str(provider))
                    logger.warning(f"Transcription model was blank, using default for {provider}: {options['transcription_model']}")
                else:
                    # If no provider selected either, use a sensible default
                    options["transcription_model"] = "base"
                    logger.warning("Both transcription model and provider were blank, using default model: base")
            
            # Convert Select values to strings after validation
            if options["transcription_provider"] != Select.BLANK:
                options["transcription_provider"] = str(options["transcription_provider"])
            else:
                options["transcription_provider"] = "faster-whisper"  # Default provider
            
            # Ensure transcription_model is a string
            options["transcription_model"] = str(options["transcription_model"])
            
        except Exception as e:
            logger.error(f"Error collecting UI values: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)}", severity="error")
            loading_indicator.display = False
            loading_indicator.classes = loading_indicator.classes | {"hidden"}
            process_button.disabled = False
            return
        
        # Show loading state
        loading_indicator.display = True
        loading_indicator.classes = loading_indicator.classes - {"hidden"}
        status_area.clear()
        status_area.load_text("Processing video files locally...")
        status_area.display = True
        status_area.classes = status_area.classes - {"hidden"}
        process_button.disabled = True
        
        # Run the actual processing in a worker thread with the collected options
        # Create a closure to pass the options parameter
        def video_worker():
            return self._process_video_files_worker(options)
        
        worker = self.run_worker(
            video_worker,
            exclusive=True, 
            thread=True,
            name="video_processing_worker",
            description="Processing video files"
        )

    def _process_video_files_worker(self, options: dict) -> dict:
        """Process video files in a background worker thread.
        
        Args:
            options: Dictionary containing all processing options collected from UI
        
        Returns:
            Dictionary with success status and results or error message
        """
        result = {"success": False, "error": "Unknown error"}
        
        try:
            # Extract all options from the passed dictionary
            all_inputs = options.get("inputs", [])
            
            if not all_inputs:
                result = {
                    "success": False,
                    "error": "No video files or URLs selected for processing"
                }
            elif not self.app_instance.media_db:
                logger.error("Media database not initialized")
                result = {
                    "success": False,
                    "error": "Media database not available"
                }
            else:
                # Import the video processing function
                try:
                    from ..Local_Ingestion import LocalVideoProcessor
                except ImportError as e:
                    logger.error(f"Failed to import video processing library: {e}")
                    result = {
                        "success": False,
                        "error": "Video processing library not available. Please install with: pip install tldw-chatbook[video]"
                    }
                else:
                    # Create processor instance
                    processor = LocalVideoProcessor(self.app_instance.media_db)
                    
                    # Process each video individually to provide progress updates
                    results_list = []
                    errors_list = []
                    
                    logger.info(f"Starting video processing batch with {len(all_inputs)} inputs")
                    logger.debug(f"Video processing options: {options}")
                    
                    # Initial status
                    self.app_instance.call_from_thread(
                        self._update_status_area, 
                        f"Starting video processing for {len(all_inputs)} file(s)...\n\n"
                    )
                    
                    for idx, input_item in enumerate(all_inputs, 1):
                        # Update progress
                        logger.info(f"Processing video {idx}/{len(all_inputs)}: {input_item}")
                        progress_msg = f"\n[{idx}/{len(all_inputs)}] Processing: {input_item}\n"
                        self.app_instance.call_from_thread(self._append_to_status_area, progress_msg)
                        
                        # Show processing stages
                        logger.debug(f"Stage: Downloading/Loading video for {input_item}")
                        self.app_instance.call_from_thread(
                            self._append_to_status_area, 
                            "  → Downloading/Loading video...\n"
                        )
                        
                        try:
                            # Process single video with stage updates
                            # We'll wrap the processor call to add stage updates
                            start_time = time.time()
                            logger.debug(f"Started processing {input_item} at {start_time}")
                            
                            # Show initial transcription stage message
                            logger.info(f"Stage: Preparing to transcribe audio for {input_item} using {options['transcription_provider']}")
                            self.app_instance.call_from_thread(
                                self._append_to_status_area, 
                                f"  → Transcribing audio (using {options['transcription_provider']})...\n"
                            )
                            
                            # Create transcription progress callback
                            def transcription_progress(progress: float, status: str, data: Optional[Dict] = None):
                                """Handle transcription progress updates."""
                                # Update the last line with progress info
                                progress_msg = f"  → Transcription: {status} [{progress:.0f}%]"
                                
                                # Add segment preview if available
                                if data and data.get("segment_text"):
                                    preview = data["segment_text"][:40].strip()
                                    if len(data["segment_text"]) > 40:
                                        preview += "..."
                                    if preview:  # Only add if there's actual text
                                        progress_msg += f' - "{preview}"'
                                
                                # Log that we're updating progress
                                logger.debug(f"Updating UI with: {progress_msg}")
                                
                                # Use update instead of append to overwrite the line
                                try:
                                    self.app_instance.call_from_thread(
                                        self._update_transcription_progress,
                                        progress_msg
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to update progress via call_from_thread: {e}")
                                    # Try direct append as fallback
                                    try:
                                        self.app_instance.call_from_thread(
                                            self._append_to_status_area,
                                            f"\n{progress_msg}"
                                        )
                                    except Exception as e2:
                                        logger.error(f"Fallback also failed: {e2}")
                                
                                # Log detailed progress
                                if data:
                                    logger.debug(f"Transcription progress: {progress:.0f}% - {status} - segment {data.get('segment_num', 0)}")
                            
                            # Process single video with progress callback
                            single_result = processor.process_videos(
                                inputs=[input_item],
                                download_video_flag=options["download_video"] and not options["extract_audio_only"],
                                transcription_progress_callback=transcription_progress,
                                start_time=options["start_time"],
                                end_time=options["end_time"],
                                transcription_provider=options["transcription_provider"],
                                transcription_model=options["transcription_model"],
                                transcription_language=options["transcription_language"],
                                translation_target_language=options["translation_target"],
                                perform_chunking=options["perform_chunking"],
                                chunk_method=options["chunk_method"],
                                max_chunk_size=options["chunk_size"],
                                chunk_overlap=options["chunk_overlap"],
                                use_adaptive_chunking=options["use_adaptive_chunking"],
                                use_multi_level_chunking=options["use_multi_level_chunking"],
                                chunk_language=options["chunk_language"] or options["transcription_language"],
                                diarize=options["diarize"],
                                vad_use=options["vad_filter"],
                                timestamp_option=options["timestamps"],
                                perform_analysis=options["perform_analysis"],
                                api_name=options.get("api_name"),
                                api_key=options.get("api_key"),
                                analysis_model=options.get("analysis_model"),
                                custom_prompt=options["custom_prompt"],
                                system_prompt=options["system_prompt"],
                                summarize_recursively=options["summarize_recursively"],
                                use_cookies=options["use_cookies"],
                                cookies=options["cookies"],
                                keep_original=options["keep_original"],
                                custom_title=options["title_override"],
                                author=options["author"],
                                keywords=options["keywords"]
                            )
                            
                            # Log result details
                            logger.debug(f"Process result for {input_item}: {single_result}")
                            
                            # Add result to list
                            if single_result.get("results"):
                                results_list.extend(single_result["results"])
                                logger.debug(f"Added {len(single_result['results'])} results from {input_item}")
                            
                            # Calculate processing time
                            elapsed_time = time.time() - start_time
                            time_str = f"{elapsed_time:.1f}s"
                            logger.info(f"Completed processing {input_item} in {elapsed_time:.2f} seconds")
                            
                            # Update status with result
                            if single_result.get("processed_count", 0) > 0:
                                # Extract some useful info from the result
                                result_info = ""
                                if single_result.get("results") and len(single_result["results"]) > 0:
                                    first_result = single_result["results"][0]
                                    if first_result.get("metadata", {}).get("duration"):
                                        duration = first_result["metadata"]["duration"]
                                        result_info = f" (duration: {duration:.1f}s)"
                                        logger.debug(f"Video duration for {input_item}: {duration:.1f}s")
                                    
                                    # Log additional metadata
                                    metadata = first_result.get("metadata", {})
                                    logger.debug(f"Video metadata for {input_item}: {metadata}")
                                
                                logger.info(f"Successfully processed {input_item} - Success")
                                status_update = f"  ✓ Completed in {time_str}{result_info}\n"
                            else:
                                logger.warning(f"Failed to process {input_item}")
                                status_update = f"  ✗ Failed after {time_str}\n"
                                if single_result.get("errors"):
                                    errors_list.extend(single_result["errors"])
                                    error_msg = single_result["errors"][0] if single_result["errors"] else "Unknown error"
                                    logger.error(f"Processing error for {input_item}: {error_msg}")
                                    status_update += f"     Error: {error_msg}\n"
                            
                            self.app_instance.call_from_thread(
                                self._append_to_status_area, status_update
                            )
                            
                        except Exception as e:
                            elapsed_time = time.time() - start_time
                            logger.error(f"Exception processing {input_item} after {elapsed_time:.2f}s: {e}", exc_info=True)
                            error_msg = f"  ✗ Error after {elapsed_time:.1f}s: {str(e)}\n"
                            self.app_instance.call_from_thread(
                                self._append_to_status_area, error_msg
                            )
                            errors_list.append(str(e))
                            results_list.append({
                                "status": "Error",
                                "input_ref": input_item,
                                "error": str(e)
                            })
                            logger.debug(f"Added error result for {input_item}: {str(e)}")
                    
                    # Aggregate results
                    processed_count = sum(1 for r in results_list if r.get("status") == "Success")
                    errors_count = len(results_list) - processed_count
                    
                    logger.info(f"Video batch processing complete: {processed_count} succeeded, {errors_count} failed out of {len(all_inputs)} total")
                    
                    # Add final summary
                    summary_msg = f"\n{'='*50}\n"
                    summary_msg += f"Processing Complete: {processed_count} succeeded, {errors_count} failed\n"
                    summary_msg += f"{'='*50}\n"
                    self.app_instance.call_from_thread(
                        self._append_to_status_area, summary_msg
                    )
                    
                    # Log summary of results
                    if errors_count > 0:
                        logger.warning(f"Failed inputs: {[r['input_ref'] for r in results_list if r.get('status') != 'Success']}")
                    
                    result = {
                        "success": True,
                        "results": {
                            "processed_count": processed_count,
                            "errors_count": errors_count,
                            "errors": errors_list,
                            "results": results_list
                        }
                    }
                    logger.debug(f"Final batch result: {processed_count} processed, {errors_count} errors")
            
        except Exception as e:
            logger.error(f"Error processing video files: {e}", exc_info=True)
            result = {
                "success": False,
                "error": str(e)
            }
        
        # Handle completion - update UI with results
        try:
            if result["success"]:
                # Process the results
                results = result["results"]
                processed_count = results.get("processed_count", 0)
                errors_count = results.get("errors_count", 0)
                
                status_messages = [f"Processing complete: {processed_count} succeeded, {errors_count} failed\n"]
                
                for video_result in results.get("results", []):
                    input_ref = video_result.get("input_ref", "Unknown")
                    status = video_result.get("status", "Unknown")
                    title = video_result.get("metadata", {}).get("title", "Untitled")
                    
                    if status == "Success":
                        status_messages.append(f"✓ {title} ({input_ref})")
                    else:
                        error = video_result.get("error", "Unknown error")
                        status_messages.append(f"✗ {input_ref}: {error}")
                
                self.app_instance.call_from_thread(self._update_status_area, "\n".join(status_messages))
                
                if processed_count > 0:
                    self.app_instance.call_from_thread(
                        self.app_instance.notify,
                        f"Successfully processed {processed_count} video file(s)",
                        severity="information"
                    )
                if errors_count > 0:
                    self.app_instance.call_from_thread(
                        self.app_instance.notify,
                        f"{errors_count} file(s) failed to process",
                        severity="warning"
                    )
            else:
                # Show error
                error_msg = result.get("error", "Unknown error occurred")
                logger.error(f"Video processing failed: {error_msg}")
                self.app_instance.call_from_thread(
                    self.app_instance.notify,
                    f"Error: {error_msg}",
                    severity="error"
                )
                self.app_instance.call_from_thread(self._update_status_area, f"Error: {error_msg}")
                
        except Exception as e:
            logger.error(f"Error in video processing completion: {e}", exc_info=True)
            self.app_instance.call_from_thread(
                self.app_instance.notify,
                f"Error: {str(e)}",
                severity="error"
            )
            
        finally:
            # Reset UI state
            self.app_instance.call_from_thread(self._reset_video_ui_state)
            
        return result
    
    def _update_status_area(self, text: str) -> None:
        """Update the status area from the worker thread."""
        try:
            status_area = self.query_one("#local-status-video", TextArea)
            status_area.load_text(text)
        except Exception:
            pass
    
    def _reset_video_ui_state(self) -> None:
        """Reset the video UI state after processing completes."""
        try:
            loading_indicator = self.query_one("#local-loading-indicator-video", LoadingIndicator)
            process_button = self.query_one("#local-submit-video", Button)
            
            loading_indicator.display = False
            loading_indicator.classes = loading_indicator.classes | {"hidden"}
            process_button.disabled = False
        except Exception:
            pass
    
    def _append_to_status_area(self, text: str) -> None:
        """Append text to the status area."""
        try:
            status_area = self.query_one("#local-status-video", TextArea)
            current_text = status_area.text
            status_area.load_text(current_text + text)
        except Exception:
            pass
    
    def _update_transcription_progress(self, text: str) -> None:
        """Update the last line in status area for transcription progress."""
        try:
            status_area = self.query_one("#local-status-video", TextArea)
            current_text = status_area.text
            lines = current_text.split('\n')
            
            # Find the last transcription line and update it
            for i in range(len(lines) - 1, -1, -1):
                if "→ Transcription:" in lines[i] or "→ Transcribing audio" in lines[i]:
                    lines[i] = text.rstrip()
                    break
            else:
                # If no transcription line found, append
                lines.append(text.rstrip())
            
            # Ensure we don't have extra blank lines at the end
            while len(lines) > 1 and lines[-1] == '':
                lines.pop()
            
            status_area.load_text('\n'.join(lines))
            # Scroll to bottom to show latest progress
            status_area.scroll_end(animate=False)
        except Exception as e:
            logger.error(f"Error updating transcription progress: {e}")
            # Fallback to append if update fails
            try:
                self._append_to_status_area(f"\n{text}")
            except Exception as e2:
                logger.error(f"Fallback append also failed: {e2}")

#
# End of Ingest_Window.py
#######################################################################################################################
