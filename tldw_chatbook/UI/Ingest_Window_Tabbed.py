# tldw_chatbook/UI/Ingest_Window_Tabbed.py
#
# Refactored Ingest Window with tabbed navigation for better UX
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
from textual.binding import Binding
from textual.css.query import QueryError
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.widgets import (
    Static, Button, Input, Select, Checkbox, TextArea, Label, 
    RadioSet, RadioButton, Collapsible, ListView, ListItem, 
    Markdown, LoadingIndicator, TabbedContent, TabPane
)
from textual import on
from textual.worker import Worker
from textual import work
from textual.reactive import reactive
from ..Widgets.form_components import (
    create_form_field, create_button_group, create_status_area
)
from ..Widgets.status_widget import EnhancedStatusWidget

# Configure logger with context
logger = logger.bind(module="Ingest_Window_Tabbed")

from ..Constants import (
    TLDW_API_AUDIO_OPTIONS_ID, TLDW_API_VIDEO_OPTIONS_ID, 
    TLDW_API_PDF_OPTIONS_ID, TLDW_API_EBOOK_OPTIONS_ID, 
    TLDW_API_DOCUMENT_OPTIONS_ID, TLDW_API_XML_OPTIONS_ID, 
    TLDW_API_MEDIAWIKI_OPTIONS_ID
)
#
# Local Imports
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, Filters
from ..tldw_api.schemas import MediaType, ChunkMethod, PdfEngine
from ..Widgets.IngestTldwApiVideoWindow import IngestTldwApiVideoWindow
from ..Widgets.IngestTldwApiAudioWindow import IngestTldwApiAudioWindow
from ..Widgets.IngestTldwApiPdfWindow import IngestTldwApiPdfWindow
from ..Widgets.IngestTldwApiEbookWindow import IngestTldwApiEbookWindow
from ..Widgets.IngestTldwApiDocumentWindow import IngestTldwApiDocumentWindow
from ..Widgets.IngestTldwApiXmlWindow import IngestTldwApiXmlWindow
from ..Widgets.IngestTldwApiMediaWikiWindow import IngestTldwApiMediaWikiWindow
from ..Widgets.IngestTldwApiPlaintextWindow import IngestTldwApiPlaintextWindow
from ..Widgets.IngestLocalPlaintextWindowSimplified import IngestLocalPlaintextWindowSimplified
from ..Widgets.IngestLocalWebArticleWindow import IngestLocalWebArticleWindow
from ..Widgets.IngestLocalDocumentWindowSimplified import IngestLocalDocumentWindowSimplified
from ..Widgets.IngestLocalEbookWindowSimplified import IngestLocalEbookWindowSimplified
from ..Widgets.IngestLocalPdfWindowSimplified import IngestLocalPdfWindowSimplified
from ..Widgets.IngestLocalAudioWindowSimplified import IngestLocalAudioWindowSimplified
from ..Widgets.IngestLocalVideoWindowSimplified import IngestLocalVideoWindowSimplified
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

def append_to_text_area(text_area: TextArea, new_text: str) -> None:
    """Helper function to append text to a TextArea widget."""
    current_text = text_area.text
    text_area.text = current_text + new_text

MEDIA_TYPES = ['video', 'audio', 'document', 'pdf', 'ebook', 'xml', 'mediawiki_dump', 'plaintext']

class IngestWindowTabbed(Container):
    """Refactored IngestWindow using tabbed navigation instead of sidebar."""
    
    BINDINGS = [
        Binding("alt+1", "switch_tab(0)", "Prompts", show=True),
        Binding("alt+2", "switch_tab(1)", "Characters", show=True),
        Binding("alt+3", "switch_tab(2)", "Notes", show=True),
        Binding("alt+4", "switch_tab(3)", "Video", show=True),
        Binding("alt+5", "switch_tab(4)", "Audio", show=True),
        Binding("alt+6", "switch_tab(5)", "Document", show=True),
        Binding("alt+7", "switch_tab(6)", "PDF", show=True),
        Binding("alt+8", "switch_tab(7)", "Ebook", show=True),
        Binding("alt+9", "switch_tab(8)", "Web", show=True),
        Binding("alt+0", "switch_tab(9)", "Plaintext", show=True),
        Binding("ctrl+s", "switch_tab(10)", "Subscriptions", show=False),
    ]
    
    # Reactive properties
    current_source_type = reactive("local")  # "local" or "api"
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.selected_local_files = {}  # Stores {media_type: [Path, ...]}
        self._current_media_type_for_file_dialog = None
        self._failed_urls_for_retry = []
        self._retry_attempts = {}
        self._local_video_window = None
        self._local_audio_window = None
        logger.debug("IngestWindowTabbed initialized.")
    
    def compose(self) -> ComposeResult:
        """Compose the tabbed interface."""
        logger.debug("Composing IngestWindowTabbed UI")
        
        with TabbedContent(id="ingest-media-tabs"):
            # Basic ingestion tabs
            with TabPane("📝 Prompts", id="tab-prompts"):
                yield from self._compose_prompts_content()
            
            with TabPane("👤 Characters", id="tab-characters"):
                yield from self._compose_characters_content()
            
            with TabPane("📓 Notes", id="tab-notes"):
                yield from self._compose_notes_content()
            
            # Media type tabs with Local/API toggle inside each
            with TabPane("🎬 Video", id="tab-video"):
                yield from self._compose_media_tab("video")
            
            with TabPane("🎵 Audio", id="tab-audio"):
                yield from self._compose_media_tab("audio")
            
            with TabPane("📄 Document", id="tab-document"):
                yield from self._compose_media_tab("document")
            
            with TabPane("📕 PDF", id="tab-pdf"):
                yield from self._compose_media_tab("pdf")
            
            with TabPane("📚 Ebook", id="tab-ebook"):
                yield from self._compose_media_tab("ebook")
            
            with TabPane("🌐 Web", id="tab-web"):
                # Web articles are local only
                with Container(classes="media-content-container"):
                    window = IngestLocalWebArticleWindow(self.app_instance)
                    yield from window.compose()
            
            with TabPane("📝 Plaintext", id="tab-plaintext"):
                yield from self._compose_media_tab("plaintext")
            
            with TabPane("📡 Subscriptions", id="tab-subscriptions"):
                yield from self._compose_subscriptions_content()
    
    def _compose_media_tab(self, media_type: str) -> ComposeResult:
        """Compose a media tab with Local/API toggle."""
        with Container(classes="media-tab-container"):
            # Add source toggle at the top
            with Container(classes="source-toggle-container"):
                yield Static(f"{media_type.title()} Ingestion", classes="tab-title")
                with RadioSet(id=f"{media_type}-source-toggle", classes="source-toggle"):
                    yield RadioButton("Local Processing", value=True, id=f"{media_type}-local-radio")
                    yield RadioButton("API Processing", id=f"{media_type}-api-radio")
            
            # Content containers that will be shown/hidden based on toggle
            with Container(id=f"{media_type}-local-content", classes="source-content"):
                yield from self._compose_local_content(media_type)
            
            with Container(id=f"{media_type}-api-content", classes="source-content hidden"):
                yield from self._compose_api_content(media_type)
    
    def _compose_local_content(self, media_type: str) -> ComposeResult:
        """Compose local processing content for a media type."""
        if media_type == "video":
            window = IngestLocalVideoWindowSimplified(self.app_instance)
            self._local_video_window = window
        elif media_type == "audio":
            window = IngestLocalAudioWindowSimplified(self.app_instance)
            self._local_audio_window = window
        elif media_type == "document":
            window = IngestLocalDocumentWindowSimplified(self.app_instance)
        elif media_type == "pdf":
            window = IngestLocalPdfWindowSimplified(self.app_instance)
        elif media_type == "ebook":
            window = IngestLocalEbookWindowSimplified(self.app_instance)
        elif media_type == "plaintext":
            window = IngestLocalPlaintextWindowSimplified(self.app_instance)
        else:
            yield Static(f"Local {media_type} processing not yet implemented")
            return
        
        yield from window.compose()
    
    def _compose_api_content(self, media_type: str) -> ComposeResult:
        """Compose API processing content for a media type."""
        if media_type == "video":
            window = IngestTldwApiVideoWindow(self.app_instance)
        elif media_type == "audio":
            window = IngestTldwApiAudioWindow(self.app_instance)
        elif media_type == "document":
            window = IngestTldwApiDocumentWindow(self.app_instance)
        elif media_type == "pdf":
            window = IngestTldwApiPdfWindow(self.app_instance)
        elif media_type == "ebook":
            window = IngestTldwApiEbookWindow(self.app_instance)
        elif media_type == "plaintext":
            window = IngestTldwApiPlaintextWindow(self.app_instance)
        else:
            yield Static(f"API {media_type} processing not yet implemented")
            return
        
        yield from window.compose()
    
    def _compose_prompts_content(self) -> ComposeResult:
        """Compose prompts tab content."""
        with VerticalScroll(classes="ingest-view-area"):
            # File selection buttons
            yield from create_button_group([
                ("Select Prompt File(s)", "ingest-prompts-select-file-button", "default"),
                ("Clear Selection", "ingest-prompts-clear-files-button", "default")
            ])
            
            yield Label("Selected Files for Import:", classes="form-label")
            yield ListView(id="ingest-prompts-selected-files-list", classes="ingest-selected-files-list")
            
            yield Label("Preview of Parsed Prompts (Max 10 shown):", classes="form-label")
            with Container(id="ingest-prompts-preview-area", classes="ingest-preview-area"):
                yield Static("Select files to see a preview.", id="ingest-prompts-preview-placeholder")
            
            # Import button centered
            yield from create_button_group([
                ("Import Selected Prompts Now", "ingest-prompts-import-now-button", "primary")
            ], alignment="center")
            
            # Enhanced status widget
            yield EnhancedStatusWidget(
                title="Import Status",
                id="prompt-import-status-widget",
                max_messages=50
            )
    
    def _compose_characters_content(self) -> ComposeResult:
        """Compose characters tab content."""
        with VerticalScroll(classes="ingest-view-area"):
            # File selection buttons
            yield from create_button_group([
                ("Select Character File(s)", "ingest-characters-select-file-button", "default"),
                ("Clear Selection", "ingest-characters-clear-files-button", "default")
            ])
            
            yield Label("Selected Files for Import:", classes="form-label")
            yield ListView(id="ingest-characters-selected-files-list", classes="ingest-selected-files-list")
            
            yield Label("Preview of Parsed Characters (Max 5 shown):", classes="form-label")
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
    
    def _compose_notes_content(self) -> ComposeResult:
        """Compose notes tab content."""
        with VerticalScroll(classes="ingest-view-area"):
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
    
    def _compose_subscriptions_content(self) -> ComposeResult:
        """Compose subscriptions tab content."""
        with VerticalScroll(classes="ingest-view-area"):
            yield Static("Subscription Management", classes="sidebar-title")
            yield Static("Subscriptions feature coming soon...", classes="placeholder-text")
    
    @on(RadioSet.Changed)
    async def handle_source_toggle(self, event: RadioSet.Changed) -> None:
        """Handle Local/API toggle changes."""
        radio_set_id = event.radio_set.id
        if not radio_set_id or not radio_set_id.endswith("-source-toggle"):
            return
        
        media_type = radio_set_id.replace("-source-toggle", "")
        is_local = event.radio_set.pressed_index == 0
        
        # Toggle visibility of content containers
        local_content = self.query_one(f"#{media_type}-local-content")
        api_content = self.query_one(f"#{media_type}-api-content")
        
        if is_local:
            local_content.remove_class("hidden")
            api_content.add_class("hidden")
        else:
            local_content.add_class("hidden")
            api_content.remove_class("hidden")
        
        logger.debug(f"Toggled {media_type} to {'local' if is_local else 'API'} processing")
    
    def on_mount(self) -> None:
        """Initialize transcription models when mounted."""
        # Initialize models for local audio/video windows if they exist
        if self._local_video_window:
            self._local_video_window.run_worker(
                self._local_video_window._initialize_models,
                exclusive=True,
                thread=True
            )
        if self._local_audio_window:
            self._local_audio_window.run_worker(
                self._local_audio_window._initialize_models,
                exclusive=True,
                thread=True
            )
    
    def action_switch_tab(self, tab_index: int) -> None:
        """Switch to a specific tab by index."""
        try:
            tabs = self.query_one("#ingest-media-tabs", TabbedContent)
            if 0 <= tab_index < len(tabs.children):
                tabs.active = list(tabs.children)[tab_index].id
                logger.debug(f"Switched to tab index {tab_index}")
        except Exception as e:
            logger.error(f"Error switching tab: {e}")
    
    # Event handlers for file selection
    @on(Button.Pressed, "#ingest-prompts-select-file-button")
    async def handle_prompts_file_select(self, event: Button.Pressed) -> None:
        """Handle prompts file selection."""
        filters = Filters(
            ("JSON Files", lambda p: p.suffix.lower() == ".json"),
            ("All Files", lambda _: True)
        )
        
        await self.app.push_screen(
            FileOpen(
                title="Select Prompt Files",
                filters=filters
            ),
            callback=lambda path: self._handle_file_selection(path, "prompts")
        )
    
    @on(Button.Pressed, "#ingest-characters-select-file-button")
    async def handle_characters_file_select(self, event: Button.Pressed) -> None:
        """Handle characters file selection."""
        filters = Filters(
            ("Character Files", lambda p: p.suffix.lower() in (".json", ".yaml", ".yml", ".png", ".jpg", ".jpeg")),
            ("All Files", lambda _: True)
        )
        
        await self.app.push_screen(
            FileOpen(
                title="Select Character Files",
                filters=filters
            ),
            callback=lambda path: self._handle_file_selection(path, "characters")
        )
    
    @on(Button.Pressed, "#ingest-notes-select-file-button")
    async def handle_notes_file_select(self, event: Button.Pressed) -> None:
        """Handle notes file selection."""
        filters = Filters(
            ("Text Files", lambda p: p.suffix.lower() in (".txt", ".md", ".markdown")),
            ("All Files", lambda _: True)
        )
        
        await self.app.push_screen(
            FileOpen(
                title="Select Note Files",
                filters=filters
            ),
            callback=lambda path: self._handle_file_selection(path, "notes")
        )
    
    async def _handle_file_selection(self, path: Path | None, file_type: str) -> None:
        """Handle file selection callback."""
        if not path:
            return
        
        # Update the appropriate file list
        list_id = f"ingest-{file_type}-selected-files-list"
        try:
            file_list = self.query_one(f"#{list_id}", ListView)
            
            # Add file to list if not already present
            file_items = [item.data for item in file_list.children if hasattr(item, 'data')]
            if str(path) not in file_items:
                file_list.append(ListItem(Label(path.name), data=str(path)))
                
                # Store in selected files
                if file_type not in self.selected_local_files:
                    self.selected_local_files[file_type] = []
                self.selected_local_files[file_type].append(path)
                
                logger.debug(f"Added {path} to {file_type} selection")
        except QueryError:
            logger.error(f"Could not find file list {list_id}")
    
    @on(Button.Pressed)
    async def handle_clear_files(self, event: Button.Pressed) -> None:
        """Handle clear files buttons."""
        button_id = event.button.id
        if not button_id or not button_id.endswith("-clear-files-button"):
            return
        
        # Extract the file type from button ID
        if "prompts" in button_id:
            file_type = "prompts"
        elif "characters" in button_id:
            file_type = "characters"
        elif "notes" in button_id:
            file_type = "notes"
        else:
            return
        
        # Clear the file list
        list_id = f"ingest-{file_type}-selected-files-list"
        try:
            file_list = self.query_one(f"#{list_id}", ListView)
            file_list.clear()
            
            # Clear stored files
            if file_type in self.selected_local_files:
                self.selected_local_files[file_type].clear()
                
            logger.debug(f"Cleared {file_type} file selection")
        except QueryError:
            logger.error(f"Could not find file list {list_id}")
    
    @on(Button.Pressed, "#ingest-prompts-import-now-button")
    async def handle_prompts_import(self, event: Button.Pressed) -> None:
        """Handle prompts import."""
        from ..Event_Handlers.ingest_events import handle_import_prompts
        await handle_import_prompts(self.app_instance)
    
    @on(Button.Pressed, "#ingest-characters-import-now-button")
    async def handle_characters_import(self, event: Button.Pressed) -> None:
        """Handle characters import."""
        from ..Event_Handlers.ingest_events import handle_import_characters
        await handle_import_characters(self.app_instance)
    
    @on(Button.Pressed, "#ingest-notes-import-now-button")
    async def handle_notes_import(self, event: Button.Pressed) -> None:
        """Handle notes import."""
        from ..Event_Handlers.ingest_events import handle_import_notes
        await handle_import_notes(self.app_instance)

#
# End of Ingest_Window_Tabbed.py
#######################################################################################################################