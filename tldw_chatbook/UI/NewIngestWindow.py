# NewIngestWindow.py
"""
Modern Ingest Content UI - Built from scratch using Textual best practices.
Completely new design with no legacy code dependencies.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional
from pathlib import Path
import asyncio
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid, VerticalScroll
from textual.widgets import Static, Button, Label, ProgressBar, Input, TextArea, Checkbox, Select
from textual.widget import Widget
from textual.reactive import reactive
from textual import on, work
from textual.message import Message

if TYPE_CHECKING:
    from ..app import TldwCli

# Configure logger
logger = logger.bind(module="NewIngestWindow")


class FileDropped(Message):
    """Custom message for file drop events."""
    
    def __init__(self, files: List[Path]):
        super().__init__()
        self.files = files


class MediaTypeSelected(Message):
    """Custom message for media type selection."""
    
    def __init__(self, media_type: str):
        super().__init__()
        self.media_type = media_type


class MediaTypeCard(Widget):
    """Interactive card for selecting media types."""
    
    # Enable focus so the widget can be interacted with
    can_focus = True
    
    def __init__(self, media_type: str, title: str, description: str, icon: str, **kwargs):
        super().__init__(**kwargs)
        self.media_type = media_type
        self.title = title  
        self.description = description
        self.icon = icon
        self.add_class("media-type-card")
        
    def compose(self) -> ComposeResult:
        """Compose the media card."""
        with Container(classes="media-card"):
            # Icon and title
            with Horizontal(classes="card-header"):
                yield Static(self.icon, classes="card-icon")
                yield Static(self.title, classes="card-title")
            
            # Description
            yield Static(self.description, classes="card-description")
            
            # No separate button needed since whole card is clickable
    
    def on_click(self, event) -> None:
        """Make entire card clickable."""
        event.stop()
        logger.info(f"MediaTypeCard clicked: {self.media_type}")
        # Post message to parent
        self.post_message(MediaTypeSelected(self.media_type))
    
    @on(Button.Pressed)
    def handle_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        event.stop()
        # Check if this is our button
        if event.button.id == f"select-{self.media_type}":
            logger.info(f"MediaTypeCard button pressed: {self.media_type}")
            # Post message to parent
            self.post_message(MediaTypeSelected(self.media_type))


class GlobalDropZone(Widget):
    """Global drag-and-drop zone for files."""
    
    # Reactive state for drop zone
    is_active = reactive(False)
    has_files = reactive(False)
    file_count = reactive(0)
    
    def compose(self) -> ComposeResult:
        """Compose the drop zone."""
        with Container(classes="drop-zone"):
            yield Static("📁", classes="drop-icon")
            yield Static("Drag files here to start processing", 
                        id="drop-message", classes="drop-message")
            yield Static("", id="file-count", classes="file-count hidden")
            
    def watch_is_active(self, active: bool):
        """Update visual state when drag is active."""
        if active:
            self.add_class("active")
        else:
            self.remove_class("active")
            
    def watch_has_files(self, has_files: bool):
        """Update display when files are present."""
        file_count_widget = self.query_one("#file-count")
        drop_message = self.query_one("#drop-message")
        
        if has_files:
            file_count_widget.remove_class("hidden")
            drop_message.update("Files ready for processing")
            file_count_widget.update(f"{self.file_count} files selected")
        else:
            file_count_widget.add_class("hidden")
            drop_message.update("Drag files here to start processing")
    
    def add_files(self, files: List[Path]):
        """Add files to the drop zone."""
        self.file_count = len(files)
        self.has_files = len(files) > 0
        self.post_message(FileDropped(files))


class ActivityFeed(Widget):
    """Live activity feed showing current processing status."""
    
    # Reactive state
    activities = reactive([])
    
    def compose(self) -> ComposeResult:
        """Compose the activity feed."""
        with Container(classes="activity-feed"):
            yield Label("Recent Activity", classes="feed-title")
            yield Container(id="activity-list", classes="activity-list")
            
    def watch_activities(self, activities: List[Dict[str, Any]]):
        """Update activity display when activities change."""
        try:
            activity_list = self.query_one("#activity-list", Container)
            activity_list.remove_children()
            
            if not activities:
                activity_list.mount(Static("No recent activity", classes="empty-message"))
                return
                
            for activity in activities[-5:]:  # Show last 5 activities
                # Create activity item container
                item_container = Horizontal(classes="activity-item")
                
                # Status icon
                icon = self._get_status_icon(activity.get("status", "unknown"))
                item_container.mount(Static(icon, classes="activity-icon"))
                
                # Activity details
                details_container = Vertical(classes="activity-details")
                details_container.mount(Static(activity.get("title", "Unknown"), classes="activity-title"))
                details_container.mount(Static(activity.get("time", ""), classes="activity-time"))
                item_container.mount(details_container)
                
                # Progress bar if processing
                if activity.get("status") == "processing":
                    progress = activity.get("progress", 0.0)
                    item_container.mount(ProgressBar(progress=progress, classes="activity-progress"))
                
                # Mount the complete item
                activity_list.mount(item_container)
        except Exception as e:
            logger.error(f"Error updating activity feed: {e}")
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for activity status."""
        icons = {
            "completed": "✅",
            "processing": "⚙️",
            "failed": "❌",
            "queued": "⏳"
        }
        return icons.get(status, "📄")
    
    def add_activity(self, title: str, status: str, progress: float = 0.0):
        """Add new activity to the feed."""
        from datetime import datetime
        
        new_activity = {
            "title": title,
            "status": status,
            "progress": progress,
            "time": datetime.now().strftime("%H:%M:%S")
        }
        
        current_activities = list(self.activities)
        current_activities.append(new_activity)
        self.activities = current_activities


class NewIngestWindow(Container):
    """
    Modern Ingest Content UI - Completely new design.
    
    Features:
    - Card-based media type selection
    - Global drag-and-drop zone
    - Live activity feed
    - Clean, modern interface
    """
    
    DEFAULT_CSS = """
    NewIngestWindow {
        height: 100%;
        width: 100%;
    }
    
    .main-title {
        dock: top;
        height: 3;
        text-align: center;
        text-style: bold;
        color: $primary;
        background: $surface;
        border-bottom: thick $primary;
        padding: 1;
    }
    
    .main-subtitle {
        dock: top;
        height: 2;
        text-align: center;
        color: $text-muted;
        background: $surface;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    .main-content {
        height: 1fr;
        width: 100%;
    }
    
    .media-selection-panel {
        width: 30%;
        height: 100%;
        padding: 1;
        background: $surface;
        border-right: thick $primary;
    }
    
    .panel-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        padding-bottom: 1;
    }
    
    .media-cards-list {
        height: 1fr;
        overflow-y: auto;
        padding: 0;
    }
    
    .media-card {
        height: auto; /* CHANGED: Was 6, now resizes to content */
        background: $panel;
        border: round $primary;
        padding: 1; /* MODIFIED: Adjusted padding slightly for better look */
        margin-bottom: 1; /* MODIFIED: Added margin for spacing between cards */
        width: 100%;
    }
    
    .media-card:hover {
        background: $primary 10%;
        border: round $accent;
    }
    
    .media-card.selected {
        background: $primary 20%;
        border: thick $accent;
    }
    
    .media-type-card {
        width: 100%;
        height: auto; /* CHANGED: Was 100%, now allows shrinking */
    }
    
    .media-type-card:focus {
        border: thick $accent;
    }
    
    .card-header {
        height: auto; /* CHANGED: Was 2, now flexible */
        align: left middle;
        margin-bottom: 1; /* MODIFIED: Added a small margin */
    }
    
    .card-icon {
        width: 4;
        text-align: center;
        text-style: bold;
    }
    
    .card-title {
        width: 1fr;
        text-style: bold;
        color: $text;
    }
    
    .card-description {
        color: $text-muted;
        height: auto; /* CHANGED: Was 1, now allows multiple lines */
        margin-bottom: 0;
    }
    
    .card-button {
        height: 2;
        width: 100%;
        margin-top: 0;
    }
    
    Button {
        min-width: 10;
    }
    
    Button:focus {
        background: $accent;
    }
    
    Button:hover {
        background: $primary-lighten-1;
    }
    
    .ingestion-panel {
        width: 70%;
        height: 100%;
        padding: 2;
        background: $surface;
    }
    
    .form-container {
        height: 100%;
        overflow-y: auto;
        padding: 1;
    }
    
    .form-section {
        margin-bottom: 2;
    }
    
    .form-label {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .form-input {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }
    
    .form-textarea {
        width: 100%;
        min-height: 5;
        margin-bottom: 1;
    }
    
    .form-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 2;
        padding: 1;
        border-bottom: solid $primary;
    }
    
    .drop-zone {
        height: 12;
        background: $panel;
        border: dashed $primary;
        text-align: center;
        padding: 2;
        margin-bottom: 2;
    }
    
    .drop-zone.active {
        background: $primary 20%;
        border: dashed $accent;
    }
    
    .drop-icon {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .drop-message {
        color: $text;
        margin-bottom: 1;
    }
    
    .file-count {
        color: $accent;
        text-style: bold;
    }
    
    .activity-feed {
        height: 1fr;
        background: $panel;
        border: round $primary;
        padding: 1;
    }
    
    .feed-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        padding-bottom: 1;
    }
    
    .activity-list {
        height: 1fr;
        overflow-y: auto;
    }
    
    .activity-item {
        height: 4;
        margin-bottom: 1;
        padding: 1;
    }
    
    .activity-icon {
        width: 3;
        text-align: center;
    }
    
    .activity-details {
        width: 1fr;
    }
    
    .activity-title {
        text-style: bold;
        color: $text;
    }
    
    .activity-time {
        color: $text-muted;
    }
    
    .activity-progress {
        width: 20;
        height: 1;
    }
    
    .empty-message {
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    
    .quick-actions {
        dock: bottom;
        height: 5;
        background: $surface;
        border-top: thick $primary;
        padding: 1;
        align: center middle;
    }
    
    .quick-actions Button {
        margin-right: 2;
        height: 3;
    }
    
    .hidden {
        display: none;
    }
    """
    
    # Reactive state
    selected_files = reactive([])
    current_media_type = reactive("video")  # Default to video
    processing_active = reactive(False)
    selected_card = reactive(None)  # Track which card is selected
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        logger.info("NewIngestWindow initialized - fresh modern interface")
    
    def on_mount(self) -> None:
        """Handle mount event."""
        logger.info("NewIngestWindow mounted successfully")
        # Initialize with video form by default
        self.call_after_refresh(self._initialize_default_view)
    
    def _initialize_default_view(self) -> None:
        """Initialize the default view."""
        try:
            # Select video card by default
            video_card = self.query_one("#media-card-video", MediaTypeCard)
            video_card.add_class("selected")
            self.selected_card = "video"
            # Load video form
            self._update_ingestion_form("video")
        except Exception as e:
            logger.error(f"Could not initialize default view: {e}")
    
    def compose(self) -> ComposeResult:
        """Compose the modern ingest interface."""
        # Main header
        yield Static("Content Ingestion Hub", classes="main-title")
        yield Static("Select media type or drag files to begin", classes="main-subtitle")
        
        # Main content area
        with Horizontal(classes="main-content"):
            # Left side - Media type selection (single column)
            with Vertical(classes="media-selection-panel"):
                yield Label("Select Import Type", classes="panel-title")
                
                # Media type cards in vertical list
                with VerticalScroll(classes="media-cards-list"):
                    # Media content cards
                    yield MediaTypeCard(
                        "video",
                        "Video Content", 
                        "YouTube, MP4, AVI files",
                        "🎬",
                        id="media-card-video"
                    )
                    
                    yield MediaTypeCard(
                        "audio",
                        "Audio Content",
                        "Podcasts, music, recordings", 
                        "🎵",
                        id="media-card-audio"
                    )
                    
                    yield MediaTypeCard(
                        "document", 
                        "Documents",
                        "Word, text files, articles",
                        "📄",
                        id="media-card-document"
                    )
                    
                    yield MediaTypeCard(
                        "pdf",
                        "PDF Files", 
                        "Papers, books, reports",
                        "📕",
                        id="media-card-pdf"
                    )
                    
                    yield MediaTypeCard(
                        "web",
                        "Web Content",
                        "Articles, blogs, web pages",
                        "🌐",
                        id="media-card-web"
                    )
                    
                    yield MediaTypeCard(
                        "ebook",
                        "E-Books",
                        "EPUB, MOBI, digital books", 
                        "📚",
                        id="media-card-ebook"
                    )
                    
                    # New cards for Notes, Character Cards, Conversations
                    yield MediaTypeCard(
                        "notes",
                        "Notes Import",
                        "Import notes and templates",
                        "📝",
                        id="media-card-notes"
                    )
                    
                    yield MediaTypeCard(
                        "character",
                        "Character Cards",
                        "Import character definitions",
                        "👤",
                        id="media-card-character"
                    )
                    
                    yield MediaTypeCard(
                        "conversation",
                        "Conversations",
                        "Import chat histories",
                        "💬",
                        id="media-card-conversation"
                    )
            
            # Right side - Ingestion settings form
            with Vertical(classes="ingestion-panel"):
                yield Container(id="ingestion-form-container", classes="form-container")
    
    @on(MediaTypeSelected)
    def handle_media_type_selected(self, event: MediaTypeSelected) -> None:
        """Handle media type selection."""
        event.stop()  # Stop propagation
        
        # Don't update if same card is clicked
        if self.selected_card == event.media_type:
            logger.debug(f"Same card clicked: {event.media_type}")
            return
        
        # Update selected card highlighting
        if self.selected_card:
            try:
                old_card = self.query_one(f"#media-card-{self.selected_card}", MediaTypeCard)
                old_card.remove_class("selected")
            except:
                pass
        
        # Highlight new card
        try:
            new_card = self.query_one(f"#media-card-{event.media_type}", MediaTypeCard)
            new_card.add_class("selected")
        except:
            pass
        
        self.selected_card = event.media_type
        self.current_media_type = event.media_type
        logger.info(f"Media type selected: {event.media_type}")
        
        # Update the right panel with appropriate form
        self._update_ingestion_form(event.media_type)
    
    @on(FileDropped)
    def handle_files_dropped(self, event: FileDropped) -> None:
        """Handle files being dropped."""
        event.stop()
        self.selected_files = event.files
        logger.info(f"Files dropped: {len(event.files)} files")
        
        # Add activity
        activity_feed = self.query_one(ActivityFeed)
        activity_feed.add_activity(
            f"{len(event.files)} files selected",
            "queued"
        )
        
        # Auto-detect media type if only one type
        detected_type = self._detect_media_type(event.files)
        if detected_type:
            self.current_media_type = detected_type
            self._open_media_processor(detected_type)
        else:
            # Mixed types - show selection
            self.app.notify("Multiple file types detected. Please select processing mode.", 
                          severity="information")
    
    @on(Button.Pressed)
    async def handle_button_press(self, event: Button.Pressed) -> None:
        """Handle all button presses in the ingestion forms."""
        event.stop()
        button_id = event.button.id
        
        if not button_id:
            return
            
        logger.debug(f"Button pressed: {button_id}")
        
        # Handle browse buttons
        if button_id.endswith("-browse"):
            await self._handle_browse_button(button_id)
        # Handle submit buttons
        elif button_id.startswith("submit-"):
            await self._handle_submit_button(button_id)
    
    async def _handle_browse_button(self, button_id: str) -> None:
        """Handle file browse button clicks."""
        try:
            # Use the existing file picker from the codebase
            from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
            
            # Push the file picker screen and wait for result
            result = await self.app.push_screen_wait(FileOpen())
            
            if result:
                # Handle single file or list of files
                if isinstance(result, Path):
                    file_path = result
                elif isinstance(result, list) and result:
                    file_path = result[0]  # Take first file
                else:
                    file_path = Path(result) if result else None
                
                if file_path:
                    # Update the corresponding input field
                    media_type = button_id.replace("-browse", "")
                    try:
                        input_field = self.query_one(f"#{media_type}-source", Input)
                        input_field.value = str(file_path)
                    except:
                        logger.debug(f"Could not find input field for {media_type}")
                    
        except ImportError:
            # Fallback to basic file picker if enhanced not available
            try:
                from ..Third_Party.textual_fspicker.file_open import FileOpen
                result = await self.app.push_screen_wait(FileOpen())
                if result:
                    file_path = result if isinstance(result, Path) else Path(result)
                    media_type = button_id.replace("-browse", "")
                    try:
                        input_field = self.query_one(f"#{media_type}-source", Input)
                        input_field.value = str(file_path)
                    except:
                        pass
            except Exception as e:
                logger.error(f"Error with file picker: {e}")
                self.app.notify("File picker unavailable", severity="error")
        except Exception as e:
            logger.error(f"Error browsing files: {e}")
            self.app.notify(f"Error selecting files: {e}", severity="error")
    
    async def _handle_submit_button(self, button_id: str) -> None:
        """Handle submit button clicks."""
        media_type = button_id.replace("submit-", "")
        logger.info(f"Processing {media_type} ingestion")
        
        # Gather form data based on media type
        form_data = self._gather_form_data(media_type)
        
        if not form_data.get("source"):
            self.app.notify(f"Please provide a source file or URL", severity="warning")
            return
        
        # Show processing notification
        self.app.notify(f"Processing {media_type}...", severity="information")
        
        # TODO: Actually process the media using the appropriate backend
        # For now, just log the form data
        logger.info(f"Form data for {media_type}: {form_data}")
        
    def _gather_form_data(self, media_type: str) -> dict:
        """Gather form data from the current form."""
        form_data = {}
        
        try:
            if media_type == "video":
                form_data["source"] = self.query_one("#video-source", Input).value
                form_data["title"] = self.query_one("#video-title", Input).value
                form_data["author"] = self.query_one("#video-author", Input).value
                form_data["transcribe"] = self.query_one("#video-transcribe", Checkbox).value
                form_data["timestamps"] = self.query_one("#video-timestamps", Checkbox).value
                form_data["diarize"] = self.query_one("#video-diarize", Checkbox).value
                form_data["prompt"] = self.query_one("#video-prompt", TextArea).text
            elif media_type == "audio":
                form_data["source"] = self.query_one("#audio-source", Input).value
                form_data["transcribe"] = self.query_one("#audio-transcribe", Checkbox).value
                form_data["diarize"] = self.query_one("#audio-diarize", Checkbox).value
            elif media_type == "pdf":
                form_data["source"] = self.query_one("#pdf-source", Input).value
                form_data["engine"] = self.query_one("#pdf-engine", Select).value
            elif media_type == "web":
                form_data["source"] = self.query_one("#web-url", Input).value
                form_data["use_cookies"] = self.query_one("#web-cookies", Checkbox).value
            # Add other media types as needed
            else:
                # Generic source field
                source_field = self.query_one(f"#{media_type}-source", Input)
                if source_field:
                    form_data["source"] = source_field.value
        except Exception as e:
            logger.error(f"Error gathering form data: {e}")
        
        return form_data
    
    @on(Button.Pressed, "#old-add-urls") 
    def handle_add_urls(self, event: Button.Pressed) -> None:
        """Show URL input dialog."""
        event.stop()
        logger.debug("Add URLs button pressed")
        # TODO: Implement URL input dialog
        self.app.notify("URL input coming soon!", severity="information")
    
    @on(Button.Pressed, "#old-view-queue")
    def handle_view_queue(self, event: Button.Pressed) -> None:
        """Show processing queue."""
        event.stop()
        logger.debug("View queue button pressed")
        # TODO: Implement queue viewer
        self.app.notify("Queue viewer coming soon!", severity="information")
    
    @on(Button.Pressed, "#old-settings")
    def handle_settings(self, event: Button.Pressed) -> None:
        """Show ingest settings."""
        event.stop()
        logger.debug("Settings button pressed")
        # TODO: Implement settings dialog
        self.app.notify("Settings coming soon!", severity="information")
    
    def _detect_media_type(self, files: List[Path]) -> Optional[str]:
        """Auto-detect media type from file extensions."""
        if not files:
            return None
            
        # Get all extensions
        extensions = {f.suffix.lower() for f in files}
        
        # Video extensions
        video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
        if extensions.issubset(video_exts):
            return "video"
        
        # Audio extensions  
        audio_exts = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        if extensions.issubset(audio_exts):
            return "audio"
            
        # PDF
        if extensions == {'.pdf'}:
            return "pdf"
            
        # Document extensions
        doc_exts = {'.txt', '.doc', '.docx', '.rtf', '.md'}
        if extensions.issubset(doc_exts):
            return "document"
            
        # Ebook extensions
        ebook_exts = {'.epub', '.mobi', '.azw', '.azw3'}
        if extensions.issubset(ebook_exts):
            return "ebook"
        
        # Mixed or unknown types
        return None
    
    def _update_ingestion_form(self, media_type: str):
        """Update the right panel with the appropriate ingestion form."""
        logger.info(f"Updating ingestion form for: {media_type}")
        
        try:
            # Get the form container
            form_container = self.query_one("#ingestion-form-container", Container)
            form_container.remove_children()
            
            # Create form based on media type
            if media_type == "video":
                form_widgets = self._create_video_form()
            elif media_type == "audio":
                form_widgets = self._create_audio_form()
            elif media_type == "pdf":
                form_widgets = self._create_pdf_form()
            elif media_type == "document":
                form_widgets = self._create_document_form()
            elif media_type == "web":
                form_widgets = self._create_web_form()
            elif media_type == "ebook":
                form_widgets = self._create_ebook_form()
            elif media_type == "notes":
                form_widgets = self._create_notes_form()
            elif media_type == "character":
                form_widgets = self._create_character_form()
            elif media_type == "conversation":
                form_widgets = self._create_conversation_form()
            else:
                form_widgets = [Static(f"Form for {media_type} not yet implemented")]
            
            # Mount the widgets
            for widget in form_widgets:
                form_container.mount(widget)
                
        except Exception as e:
            logger.error(f"Error updating ingestion form: {e}")
            self.app.notify(f"Error loading form: {e}", severity="error")
    
    def _create_video_form(self) -> list:
        """Create form widgets for video ingestion."""
        from textual.containers import Container
        from textual.widgets import Input, TextArea, Checkbox, Select, Button
        
        widgets = []
        
        # Title section
        widgets.append(Static("Video Ingestion Settings", classes="form-title"))
        
        # File/URL input
        widgets.append(Label("Video Source:", classes="form-label"))
        widgets.append(Input(placeholder="Enter YouTube URL or file path", id="video-source", classes="form-input"))
        widgets.append(Button("Browse Files", id="video-browse", variant="default"))
        
        # Metadata
        widgets.append(Label("Title (optional):", classes="form-label"))
        widgets.append(Input(placeholder="Video title", id="video-title", classes="form-input"))
        
        widgets.append(Label("Author (optional):", classes="form-label"))
        widgets.append(Input(placeholder="Video author", id="video-author", classes="form-input"))
        
        # Processing options
        widgets.append(Label("Processing Options:", classes="form-label"))
        widgets.append(Checkbox("Enable transcription", True, id="video-transcribe"))
        widgets.append(Checkbox("Include timestamps", True, id="video-timestamps"))
        widgets.append(Checkbox("Speaker diarization", False, id="video-diarize"))
        
        # Custom prompt
        widgets.append(Label("Analysis Prompt (optional):", classes="form-label"))
        widgets.append(TextArea(id="video-prompt", classes="form-textarea"))
        
        # Submit button
        widgets.append(Button("Process Video", id="submit-video", variant="primary"))
        
        return widgets
    
    def _create_audio_form(self) -> list:
        """Create form widgets for audio ingestion."""
        widgets = []
        widgets.append(Static("Audio Ingestion Settings", classes="form-title"))
        widgets.append(Label("Audio Source:", classes="form-label"))
        widgets.append(Input(placeholder="Enter audio file path", id="audio-source", classes="form-input"))
        widgets.append(Button("Browse Files", id="audio-browse", variant="default"))
        widgets.append(Checkbox("Enable transcription", True, id="audio-transcribe"))
        widgets.append(Checkbox("Speaker diarization", False, id="audio-diarize"))
        widgets.append(Button("Process Audio", id="submit-audio", variant="primary"))
        return widgets
    
    def _create_pdf_form(self) -> list:
        """Create form widgets for PDF ingestion."""
        widgets = []
        widgets.append(Static("PDF Ingestion Settings", classes="form-title"))
        widgets.append(Label("PDF File:", classes="form-label"))
        widgets.append(Input(placeholder="Enter PDF file path", id="pdf-source", classes="form-input"))
        widgets.append(Button("Browse Files", id="pdf-browse", variant="default"))
        widgets.append(Label("PDF Engine:", classes="form-label"))
        pdf_engines = [("PyMuPDF4LLM", "pymupdf4llm"), ("PyMuPDF", "pymupdf"), ("Docling", "docling")]
        widgets.append(Select(pdf_engines, id="pdf-engine", value="pymupdf4llm"))
        widgets.append(Button("Process PDF", id="submit-pdf", variant="primary"))
        return widgets
    
    def _create_document_form(self) -> list:
        """Create form widgets for document ingestion."""
        widgets = []
        widgets.append(Static("Document Ingestion Settings", classes="form-title"))
        widgets.append(Label("Document File:", classes="form-label"))
        widgets.append(Input(placeholder="Enter document file path", id="doc-source", classes="form-input"))
        widgets.append(Button("Browse Files", id="doc-browse", variant="default"))
        widgets.append(Button("Process Document", id="submit-doc", variant="primary"))
        return widgets
    
    def _create_web_form(self) -> list:
        """Create form widgets for web content ingestion."""
        widgets = []
        widgets.append(Static("Web Content Ingestion", classes="form-title"))
        widgets.append(Label("Web URL:", classes="form-label"))
        widgets.append(Input(placeholder="Enter web page URL", id="web-url", classes="form-input"))
        widgets.append(Checkbox("Use cookies for scraping", False, id="web-cookies"))
        widgets.append(Button("Process Web Page", id="submit-web", variant="primary"))
        return widgets
    
    def _create_ebook_form(self) -> list:
        """Create form widgets for ebook ingestion."""
        widgets = []
        widgets.append(Static("E-Book Ingestion Settings", classes="form-title"))
        widgets.append(Label("E-Book File:", classes="form-label"))
        widgets.append(Input(placeholder="Enter ebook file path", id="ebook-source", classes="form-input"))
        widgets.append(Button("Browse Files", id="ebook-browse", variant="default"))
        widgets.append(Button("Process E-Book", id="submit-ebook", variant="primary"))
        return widgets
    
    def _create_notes_form(self) -> list:
        """Create form widgets for notes import."""
        widgets = []
        widgets.append(Static("Notes Import Settings", classes="form-title"))
        widgets.append(Label("Notes File/Directory:", classes="form-label"))
        widgets.append(Input(placeholder="Enter notes file or directory path", id="notes-source", classes="form-input"))
        widgets.append(Button("Browse", id="notes-browse", variant="default"))
        widgets.append(Checkbox("Import as templates", False, id="notes-templates"))
        widgets.append(Checkbox("Enable sync", True, id="notes-sync"))
        widgets.append(Button("Import Notes", id="submit-notes", variant="primary"))
        return widgets
    
    def _create_character_form(self) -> list:
        """Create form widgets for character card import."""
        widgets = []
        widgets.append(Static("Character Card Import", classes="form-title"))
        widgets.append(Label("Character Card File:", classes="form-label"))
        widgets.append(Input(placeholder="Enter character card file path (.json, .png)", id="char-source", classes="form-input"))
        widgets.append(Button("Browse Files", id="char-browse", variant="default"))
        widgets.append(Label("Card Format:", classes="form-label"))
        formats = [("Tavern/SillyTavern", "tavern"), ("CharacterAI", "cai"), ("Auto-detect", "auto")]
        widgets.append(Select(formats, id="char-format", value="auto"))
        widgets.append(Button("Import Character", id="submit-char", variant="primary"))
        return widgets
    
    def _create_conversation_form(self) -> list:
        """Create form widgets for conversation import."""
        widgets = []
        widgets.append(Static("Conversation Import", classes="form-title"))
        widgets.append(Label("Conversation File:", classes="form-label"))
        widgets.append(Input(placeholder="Enter conversation export file", id="conv-source", classes="form-input"))
        widgets.append(Button("Browse Files", id="conv-browse", variant="default"))
        widgets.append(Label("Import Format:", classes="form-label"))
        formats = [("JSON", "json"), ("ChatGPT Export", "chatgpt"), ("Discord", "discord")]
        widgets.append(Select(formats, id="conv-format", value="json"))
        widgets.append(Button("Import Conversation", id="submit-conv", variant="primary"))
        return widgets
    
    def _open_media_processor(self, media_type: str):
        """Open the appropriate processor for the selected media type."""
        logger.info(f"Opening processor for media type: {media_type}")
        
        try:
            # Create a modal screen with the processor
            from textual.screen import ModalScreen
            from textual.widgets import Header, Footer
            from textual.containers import VerticalScroll
            
            class MediaProcessorModal(ModalScreen):
                """Modal screen for media processing."""
                
                DEFAULT_CSS = """
                MediaProcessorModal {
                    align: center middle;
                }
                
                MediaProcessorModal > Container {
                    width: 90%;
                    height: 90%;
                    background: $panel;
                    border: thick $primary;
                    padding: 1;
                }
                
                .modal-header {
                    dock: top;
                    height: 3;
                    background: $primary;
                    color: $text;
                    text-align: center;
                    text-style: bold;
                    padding: 1;
                }
                
                .close-button {
                    dock: top;
                    height: 3;
                    margin: 1;
                }
                """
                
                def __init__(self, app_instance, media_type: str, selected_files: list = None):
                    super().__init__()
                    self.app_instance = app_instance
                    self.media_type = media_type
                    self.selected_files = selected_files or []
                
                def compose(self) -> ComposeResult:
                    with Container():
                        yield Static(f"Process {self.media_type.title()} Content", classes="modal-header")
                        yield Button("← Back to Selection", id="close-modal", classes="close-button")
                        
                        # Use the tabbed window for now
                        from ..Widgets.Media_Ingest.IngestTldwApiTabbedWindow import IngestTldwApiTabbedWindow
                        processor = IngestTldwApiTabbedWindow(self.app_instance)
                        
                        # Add files if available
                        if self.selected_files:
                            processor.selected_local_files[self.media_type] = self.selected_files
                        
                        yield processor
                
                @on(Button.Pressed, "#close-modal")
                def close_modal(self, event):
                    """Close the modal and return to selection."""
                    event.stop()
                    self.dismiss()
            
            # Push the modal screen
            modal = MediaProcessorModal(
                self.app_instance,
                media_type,
                self.selected_files
            )
            self.app.push_screen(modal)
            
        except ImportError as e:
            logger.error(f"Error importing processor: {e}")
            self.app.notify(f"Processor not available: {e}", severity="error")
        except Exception as e:
            logger.error(f"Error opening processor: {e}")
            self.app.notify(f"Error opening processor: {e}", severity="error")


# Default CSS for the new ingest window (kept for reference but now in class)
DEFAULT_CSS_BACKUP = """
/* Modern Ingest Window Styles */

.main-title {
    dock: top;
    height: 3;
    text-align: center;
    text-style: bold;
    color: $primary;
    background: $surface;
    border-bottom: thick $primary;
    padding: 1;
}

.main-subtitle {
    dock: top;
    height: 2;
    text-align: center;
    color: $text-muted;
    background: $surface;
    padding: 0 1;
    margin-bottom: 1;
}

.main-content {
    height: 1fr;
    width: 100%;
}

/* Left Panel - Media Type Selection */
.media-selection-panel {
    width: 60%;
    height: 100%;
    padding: 1;
    background: $surface;
    border-right: thick $primary;
}

.panel-title {
    text-style: bold;
    color: $primary;
    margin-bottom: 1;
    border-bottom: solid $primary;
    padding-bottom: 1;
}

.media-cards-grid {
    grid-size: 2;
    grid-columns: 1fr 1fr;
    grid-rows: auto;
    grid-gutter: 1;
    height: auto;
}

/* Media Type Cards */
.media-card {
    height: 8;
    background: $surface;
    border: round $primary;
    padding: 1;
    cursor: pointer;
}

.media-card:hover {
    background: $primary 10%;
    border: round $accent;
}

.card-header {
    height: 3;
    align: left middle;
    margin-bottom: 1;
}

.card-icon {
    width: 4;
    text-align: center;
    text-style: bold;
}

.card-title {
    width: 1fr;
    text-style: bold;
    color: $text;
}

.card-description {
    color: $text-muted;
    height: 2;
    margin-bottom: 1;
}

.card-button {
    height: 3;
    width: 100%;
}

/* Right Panel - Activity */
.activity-panel {
    width: 40%;
    height: 100%;
    padding: 1;
}

/* Global Drop Zone */
.drop-zone {
    height: 12;
    background: $surface;
    border: dashed $primary;
    text-align: center;
    padding: 2;
    margin-bottom: 2;
}

.drop-zone.active {
    background: $primary 20%;
    border: dashed $accent;
}

.drop-icon {
    text-style: bold;
    color: $primary;
    margin-bottom: 1;
}

.drop-message {
    color: $text;
    text-style: bold;
    margin-bottom: 1;
}

.file-count {
    color: $accent;
    text-style: bold;
}

/* Activity Feed */
.activity-feed {
    height: 1fr;
    background: $surface;
    border: round $primary;
    padding: 1;
}

.feed-title {
    text-style: bold;
    color: $primary;
    margin-bottom: 1;
    border-bottom: solid $primary;
    padding-bottom: 1;
}

.activity-list {
    height: 1fr;
    overflow-y: auto;
}

.activity-item {
    height: 4;
    margin-bottom: 1;
    background: $surface-lighten-1;
    border: round $surface-lighten-2;
    padding: 1;
}

.activity-icon {
    width: 3;
    text-align: center;
}

.activity-details {
    width: 1fr;
}

.activity-title {
    text-style: bold;
    color: $text;
}

.activity-time {
    color: $text-muted;
}

.activity-progress {
    width: 20;
    height: 1;
}

.empty-message {
    text-align: center;
    color: $text-muted;
    padding: 2;
}

/* Quick Actions Bar */
.quick-actions {
    dock: bottom;
    height: 5;
    background: $surface;
    border-top: thick $primary;
    padding: 1;
    align: center middle;
}

.quick-actions Button {
    margin-right: 2;
    height: 3;
}

/* Utility classes */
.hidden {
    display: none;
}

.processing {
    opacity: 0.7;
}

.success {
    color: $success;
}

.error {
    color: $error;
}

.warning {
    color: $warning;
}
"""