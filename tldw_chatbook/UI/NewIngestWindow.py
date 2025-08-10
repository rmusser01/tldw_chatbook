# NewIngestWindow.py
"""
Modern Ingest Content UI - Built from scratch using Textual best practices.
Completely new design with no legacy code dependencies.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional
from pathlib import Path
import asyncio
from dataclasses import dataclass
from datetime import datetime
import os
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid, VerticalScroll
from textual.widgets import Static, Button, Label, ProgressBar, Input, TextArea, Checkbox, Select, Switch, Collapsible
from textual.widget import Widget
from textual.reactive import reactive
from textual import on, work
from textual.message import Message
from textual.screen import ModalScreen

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


@dataclass
class QueueItem:
    """Data class for queue items."""
    media_type: str
    sources: List[str]
    metadata: Dict[str, Any]
    processing_options: Dict[str, Any]
    status: str = "queued"
    id: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = f"{self.media_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class PromptSelectorModal(ModalScreen):
    """Modal for selecting prompts from database."""
    
    DEFAULT_CSS = """
    PromptSelectorModal {
        align: center middle;
    }
    
    PromptSelectorModal > Container {
        width: 80%;
        height: 80%;
        background: $panel;
        border: thick $primary;
        padding: 2;
    }
    
    .prompt-modal-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .prompt-search {
        width: 100%;
        margin-bottom: 1;
    }
    
    .prompt-list {
        height: 1fr;
        border: solid $primary;
        margin-bottom: 1;
    }
    """
    
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
    
    def compose(self) -> ComposeResult:
        with Container():
            yield Static("Select Prompt from Library", classes="prompt-modal-title")
            yield Input(placeholder="Search prompts...", id="prompt-search", classes="prompt-search")
            
            # TODO: Connect to PromptsDatabase and load actual prompts
            with VerticalScroll(classes="prompt-list"):
                yield Static("[Placeholder: Prompt library will be loaded here]")
                yield Static("Example: Summarization Prompt")
                yield Static("Example: Analysis Prompt")
                yield Static("Example: Q&A Generation Prompt")
            
            with Horizontal():
                yield Button("Select", id="select-prompt", variant="primary")
                yield Button("Cancel", id="cancel-prompt")
    
    @on(Button.Pressed, "#select-prompt")
    def select_prompt(self):
        # TODO: Get selected prompt and pass to callback
        if self.callback:
            self.callback("[Selected prompt text will go here]")
        self.dismiss()
    
    @on(Button.Pressed, "#cancel-prompt")
    def cancel(self):
        self.dismiss()


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
    
    /* New styles for enhanced features */
    .form-textarea-source {
        width: 100%;
        min-height: 6;
        max-height: 12;
        margin-bottom: 1;
    }
    
    .form-textarea-metadata {
        width: 100%;
        min-height: 4;
        max-height: 8; 
        margin-bottom: 1;
    }
    
    .small-label {
        text-style: italic;
        color: $text-muted;
        height: 1;
        margin-top: -1;
        margin-bottom: 1;
    }
    
    .time-input-container {
        height: auto;
        margin-bottom: 1;
    }
    
    .time-input-container > Vertical {
        width: 1fr;
        padding-right: 1;
    }
    
    .processing-mode-container {
        margin-bottom: 2;
        padding-bottom: 1;
        border-bottom: solid $primary;
    }
    
    .queue-buttons {
        margin-top: 2;
    }
    
    .queue-buttons > Button {
        margin-right: 1;
    }
    
    .prompt-row {
        margin-bottom: 1;
    }
    
    .load-prompt-btn {
        height: 3;
        width: auto;
        margin-left: 1;
    }
    
    .analysis-section {
        margin-top: 2;
        padding-top: 1;
        border-top: solid $primary;
    }
    """
    
    # Reactive state
    selected_files = reactive([])
    current_media_type = reactive("video")  # Default to video
    processing_active = reactive(False)
    selected_card = reactive(None)  # Track which card is selected
    ingestion_queue = reactive([])  # Queue for batch processing
    processing_mode = reactive("local")  # local or remote
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.processing_worker = None
        logger.info("NewIngestWindow initialized - fresh modern interface with batch processing")
    
    def on_mount(self) -> None:
        """Handle mount event."""
        logger.info("NewIngestWindow mounted successfully")
        # Initialize with video form by default
        self.call_after_refresh(self._initialize_default_view)
        # Start the queue processor
        self.processing_worker = self.process_ingestion_queue()
    
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
                with VerticalScroll(classes="form-scroll-container"):
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
        # Handle add to queue buttons
        elif button_id.endswith("-add-queue"):
            await self._handle_add_to_queue(button_id)
        # Handle load prompt buttons
        elif button_id.endswith("-load-prompt"):
            await self._handle_load_prompt(button_id)
    
    async def _handle_browse_button(self, button_id: str) -> None:
        """Handle file browse button clicks."""
        try:
            # Use the existing file picker from the codebase
            from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
            
            # Push the file picker screen and wait for result
            result = await self.app.push_screen_wait(FileOpen())
            
            if result:
                # Handle single file or list of files
                files_to_add = []
                if isinstance(result, Path):
                    files_to_add = [str(result)]
                elif isinstance(result, list):
                    files_to_add = [str(f) for f in result if f]
                else:
                    if result:
                        files_to_add = [str(result)]
                
                if files_to_add:
                    # Update the corresponding TextArea by appending
                    media_type = button_id.replace("-browse", "")
                    try:
                        source_widget = self.query_one(f"#{media_type}-source", TextArea)
                        current_text = source_widget.text.strip()
                        
                        # Append new files to existing content
                        if current_text:
                            new_text = current_text + "\n" + "\n".join(files_to_add)
                        else:
                            new_text = "\n".join(files_to_add)
                        
                        source_widget.text = new_text
                    except Exception as e:
                        logger.debug(f"Could not find source TextArea for {media_type}: {e}")
                    
        except ImportError:
            # Fallback to basic file picker if enhanced not available
            try:
                from ..Third_Party.textual_fspicker.file_open import FileOpen
                result = await self.app.push_screen_wait(FileOpen())
                if result:
                    file_path = str(result if isinstance(result, Path) else Path(result))
                    media_type = button_id.replace("-browse", "")
                    try:
                        source_widget = self.query_one(f"#{media_type}-source", TextArea)
                        current_text = source_widget.text.strip()
                        
                        if current_text:
                            source_widget.text = current_text + "\n" + file_path
                        else:
                            source_widget.text = file_path
                    except Exception as e:
                        logger.debug(f"Error updating source widget: {e}")
            except Exception as e:
                logger.error(f"Error with file picker: {e}")
                self.app.notify("File picker unavailable", severity="error")
        except Exception as e:
            logger.error(f"Error browsing files: {e}")
            self.app.notify(f"Error selecting files: {e}", severity="error")
    
    async def _handle_submit_button(self, button_id: str) -> None:
        """Handle submit button clicks - process immediately."""
        media_type = button_id.replace("submit-", "")
        logger.info(f"Processing {media_type} ingestion")
        
        # Gather form data based on media type
        form_data = self._gather_form_data(media_type)
        
        if not form_data.get("sources"):
            self.app.notify(f"Please provide at least one source file or URL", severity="warning")
            return
        
        # Create queue item and add directly to queue for immediate processing
        queue_item = QueueItem(
            media_type=media_type,
            sources=form_data.get("sources", []),
            metadata=form_data.get("items", []),
            processing_options=form_data
        )
        
        # Add to front of queue for immediate processing
        current_queue = [queue_item] + list(self.ingestion_queue)
        self.ingestion_queue = current_queue
        
        # Show processing notification
        self.app.notify(f"Processing {len(queue_item.sources)} {media_type} file(s)...", severity="information")
    
    async def _handle_add_to_queue(self, button_id: str) -> None:
        """Handle add to queue button clicks."""
        media_type = button_id.replace("-add-queue", "")
        logger.info(f"Adding {media_type} to queue")
        
        # Gather form data
        form_data = self._gather_form_data(media_type)
        
        if not form_data.get("sources"):
            self.app.notify(f"Please provide at least one source file or URL", severity="warning")
            return
        
        # Create queue item
        queue_item = QueueItem(
            media_type=media_type,
            sources=form_data.get("sources", []),
            metadata=form_data.get("items", []),
            processing_options=form_data
        )
        
        # Add to queue
        self._add_to_queue(queue_item)
        
        # Show notification
        self.app.notify(f"Added {len(queue_item.sources)} {media_type} file(s) to queue", severity="success")
    
    async def _handle_load_prompt(self, button_id: str) -> None:
        """Handle load prompt button clicks."""
        media_type = button_id.replace("-load-prompt", "")
        logger.info(f"Loading prompt for {media_type}")
        
        # Create callback to set the prompt text
        def set_prompt_text(prompt_text: str):
            try:
                prompt_widget = self.query_one(f"#{media_type}-prompt", TextArea)
                prompt_widget.text = prompt_text
            except Exception as e:
                logger.error(f"Error setting prompt text: {e}")
        
        # Open the prompt selector modal
        modal = PromptSelectorModal(callback=set_prompt_text)
        await self.app.push_screen(modal)
        
    def _gather_form_data(self, media_type: str) -> dict:
        """Gather form data from the current form."""
        form_data = {}
        
        try:
            # Parse multi-line sources
            source_widget = self.query_one(f"#{media_type}-source", TextArea)
            sources = self._parse_multiline_input(source_widget.text)
            form_data["sources"] = sources
            
            # Parse titles and authors if available
            try:
                title_widget = self.query_one(f"#{media_type}-title", TextArea)
                titles = self._parse_multiline_input(title_widget.text)
            except:
                titles = []
            
            try:
                author_widget = self.query_one(f"#{media_type}-author", TextArea)
                authors = self._parse_multiline_input(author_widget.text)
            except:
                authors = []
            
            # Match metadata to sources
            form_data["items"] = self._match_metadata_to_sources(sources, titles, authors)
            
            # Common processing options
            form_data["processing_mode"] = self.processing_mode
            
            # Media-specific options
            if media_type == "video":
                form_data["transcribe"] = self.query_one("#video-transcribe", Checkbox).value
                form_data["vad"] = self.query_one("#video-vad", Checkbox).value
                form_data["timestamps"] = self.query_one("#video-timestamps", Checkbox).value
                form_data["diarize"] = self.query_one("#video-diarize", Checkbox).value
                form_data["save_original"] = self.query_one("#video-save-original", Checkbox).value
                form_data["start_time"] = self.query_one("#video-start-time", Input).value
                form_data["end_time"] = self.query_one("#video-end-time", Input).value
                form_data["enable_analysis"] = self.query_one("#video-enable-analysis", Checkbox).value
                form_data["analysis_provider"] = self.query_one("#video-analysis-provider", Select).value
                form_data["analysis_model"] = self.query_one("#video-analysis-model", Select).value
                form_data["prompt"] = self.query_one("#video-prompt", TextArea).text
                
            elif media_type == "audio":
                form_data["transcribe"] = self.query_one("#audio-transcribe", Checkbox).value
                form_data["vad"] = self.query_one("#audio-vad", Checkbox).value
                form_data["diarize"] = self.query_one("#audio-diarize", Checkbox).value
                form_data["save_original"] = self.query_one("#audio-save-original", Checkbox).value
                form_data["start_time"] = self.query_one("#audio-start-time", Input).value
                form_data["end_time"] = self.query_one("#audio-end-time", Input).value
                form_data["enable_analysis"] = self.query_one("#audio-enable-analysis", Checkbox).value
                form_data["analysis_provider"] = self.query_one("#audio-analysis-provider", Select).value
                form_data["analysis_model"] = self.query_one("#audio-analysis-model", Select).value
                form_data["prompt"] = self.query_one("#audio-prompt", TextArea).text
                
            elif media_type == "pdf":
                form_data["engine"] = self.query_one("#pdf-engine", Select).value
                form_data["enable_analysis"] = self.query_one("#pdf-enable-analysis", Checkbox).value
                form_data["analysis_provider"] = self.query_one("#pdf-analysis-provider", Select).value
                form_data["analysis_model"] = self.query_one("#pdf-analysis-model", Select).value
                form_data["prompt"] = self.query_one("#pdf-prompt", TextArea).text
                
            elif media_type == "web":
                form_data["use_cookies"] = self.query_one("#web-cookies", Checkbox).value
                form_data["save_original"] = self.query_one("#web-save-original", Checkbox).value
                form_data["enable_analysis"] = self.query_one("#web-enable-analysis", Checkbox).value
                form_data["analysis_provider"] = self.query_one("#web-analysis-provider", Select).value
                form_data["analysis_model"] = self.query_one("#web-analysis-model", Select).value
                form_data["prompt"] = self.query_one("#web-prompt", TextArea).text
                
            # Add gathering for other media types as needed
                
        except Exception as e:
            logger.error(f"Error gathering form data for {media_type}: {e}")
        
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
    
    def _parse_multiline_input(self, text: str) -> List[str]:
        """Parse TextArea content into list of non-empty lines."""
        return [line.strip() for line in text.splitlines() if line.strip()]
    
    def _match_metadata_to_sources(self, sources: List[str], titles: List[str], authors: List[str] = None) -> List[dict]:
        """Match titles and authors to sources by line position."""
        result = []
        for i, source in enumerate(sources):
            metadata = {"source": source}
            if i < len(titles) and titles[i]:
                metadata["title"] = titles[i]
            if authors and i < len(authors) and authors[i]:
                metadata["author"] = authors[i]
            result.append(metadata)
        return result
    
    def _save_original_file(self, file_path: str, media_type: str, content: bytes = None):
        """Save original downloaded file to user's Downloads folder."""
        try:
            # Create directory structure
            download_dir = Path.home() / "Downloads" / "tldw_Chatbook_Processed_Files" / media_type
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Get filename from path
            filename = Path(file_path).name
            target_path = download_dir / filename
            
            if content:
                # Save provided content
                target_path.write_bytes(content)
            else:
                # Copy existing file
                import shutil
                shutil.copy2(file_path, target_path)
            
            logger.info(f"Saved original file to: {target_path}")
            return target_path
        except Exception as e:
            logger.error(f"Error saving original file: {e}")
            return None
    
    def _add_to_queue(self, item: QueueItem):
        """Add item to processing queue."""
        current_queue = list(self.ingestion_queue)
        current_queue.append(item)
        self.ingestion_queue = current_queue
        
        # Update activity feed
        try:
            activity_feed = self.query_one(ActivityFeed)
            activity_feed.add_activity(
                f"Added {item.media_type} to queue ({len(item.sources)} files)",
                "queued"
            )
        except:
            pass
    
    @work(exclusive=True, group="ingestion_worker")
    async def process_ingestion_queue(self):
        """Worker to process files from the ingestion queue one by one."""
        logger.info("Ingestion queue processor started.")
        
        while self.is_mounted:
            if self.ingestion_queue:
                item = self.ingestion_queue[0]
                logger.info(f"Processing queue item: {item.id}")
                
                try:
                    # Update activity feed
                    activity_feed = self.query_one(ActivityFeed)
                    activity_feed.add_activity(
                        f"Processing {item.media_type} ({len(item.sources)} files)",
                        "processing",
                        0.0
                    )
                    
                    # TODO: Actual processing logic here
                    # Simulate processing with progress updates
                    total_steps = 10
                    for i in range(total_steps):
                        activity_feed.add_activity(
                            f"Processing {item.media_type}",
                            "processing",
                            (i + 1) / total_steps * 100
                        )
                        await asyncio.sleep(0.5)
                    
                    activity_feed.add_activity(
                        f"Completed {item.media_type}",
                        "completed"
                    )
                    logger.info(f"Queue item completed: {item.id}")
                    
                except Exception as e:
                    logger.error(f"Queue item failed: {item.id}. Error: {e}")
                    try:
                        activity_feed = self.query_one(ActivityFeed)
                        activity_feed.add_activity(
                            f"Failed {item.media_type}: {str(e)}",
                            "failed"
                        )
                    except:
                        pass
                
                # Remove completed/failed item from queue
                self.ingestion_queue = self.ingestion_queue[1:]
            
            await asyncio.sleep(1)
        
        logger.info("Ingestion queue processor stopped.")
    
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
            
            # Check if container is mounted
            if not form_container.is_mounted:
                logger.warning("Form container not mounted yet, skipping update")
                return
            
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
        widgets = []
        
        # Title section
        widgets.append(Static("Video Ingestion Settings", classes="form-title"))
        
        # Processing Mode Toggle
        mode_container = Container(
            Label("Processing Mode:", classes="form-label"),
            Switch(value=self.processing_mode == "local", id="video-mode-switch"),
            Static("⚫ Local  ⚪ Remote", id="video-mode-label"),
            classes="processing-mode-container"
        )
        widgets.append(mode_container)
        
        # Multi-line source input
        widgets.append(Label("Video Sources (one per line):", classes="form-label"))
        widgets.append(Static("Enter YouTube URLs or file paths, one per line", classes="small-label"))
        widgets.append(TextArea("", 
                               id="video-source", classes="form-textarea-source"))
        widgets.append(Button("Browse Files", id="video-browse", variant="default"))
        
        # Multi-line metadata
        widgets.append(Label("Titles (optional, one per source line):", classes="form-label"))
        widgets.append(Static("Line 1 title corresponds to line 1 source, etc.", classes="small-label"))
        widgets.append(TextArea("", 
                               id="video-title", classes="form-textarea-metadata"))
        
        widgets.append(Label("Authors (optional, one per source line):", classes="form-label"))
        widgets.append(TextArea("", 
                               id="video-author", classes="form-textarea-metadata"))
        
        # Time range options
        widgets.append(Label("Time Range (optional, applies to all):", classes="form-label"))
        time_container = Horizontal(
            Vertical(
                Label("Start Time:"),
                Input(placeholder="HH:MM:SS or seconds", id="video-start-time", classes="form-input")
            ),
            Vertical(
                Label("End Time:"),
                Input(placeholder="HH:MM:SS or seconds", id="video-end-time", classes="form-input")
            ),
            classes="time-input-container"
        )
        widgets.append(time_container)
        
        # Processing options
        widgets.append(Label("Processing Options:", classes="form-label"))
        widgets.append(Checkbox("Enable transcription", True, id="video-transcribe"))
        widgets.append(Checkbox("Enable Voice Activity Detection (VAD)", False, id="video-vad"))
        widgets.append(Checkbox("Include timestamps", True, id="video-timestamps"))
        widgets.append(Checkbox("Speaker diarization", False, id="video-diarize"))
        widgets.append(Checkbox("Save original file (if downloaded)", False, id="video-save-original"))
        
        # Analysis options
        # Get available providers from app config
        providers = list(self.app_instance.app_config.get("api_settings", {}).keys()) if self.app_instance.app_config else []
        provider_options = [(name, name) for name in providers] if providers else [("No providers configured", "none")]
        default_provider = provider_options[0][1] if provider_options and provider_options[0][1] != "none" else "none"
        
        analysis_container = Container(
            Label("Analysis Options:", classes="form-label"),
            Checkbox("Enable LLM Analysis", False, id="video-enable-analysis"),
            Label("Analysis Provider:", classes="form-label"),
            Select(provider_options, id="video-analysis-provider", value=default_provider),
            Label("Analysis Model:", classes="form-label"),
            Select([("Select provider first", "none")], id="video-analysis-model", value="none"),
            Label("Analysis Prompt:", classes="form-label"),
            Horizontal(
                TextArea("", 
                         id="video-prompt", classes="form-textarea"),
                Button("Load Prompt", id="video-load-prompt", classes="load-prompt-btn"),
                classes="prompt-row"
            ),
            classes="analysis-section"
        )
        widgets.append(analysis_container)
        
        # Queue/Submit buttons
        queue_buttons = Horizontal(
            Button("Add to Queue", id="video-add-queue", variant="default"),
            Button("Process Now", id="submit-video", variant="primary"),
            classes="queue-buttons"
        )
        widgets.append(queue_buttons)
        
        return widgets
    
    def _create_audio_form(self) -> list:
        """Create form widgets for audio ingestion."""
        widgets = []
        
        # Title section
        widgets.append(Static("Audio Ingestion Settings", classes="form-title"))
        
        # Processing Mode Toggle
        mode_container = Container(
            Label("Processing Mode:", classes="form-label"),
            Switch(value=self.processing_mode == "local", id="audio-mode-switch"),
            Static("⚫ Local  ⚪ Remote", id="audio-mode-label"),
            classes="processing-mode-container"
        )
        widgets.append(mode_container)
        
        # Multi-line source input
        widgets.append(Label("Audio Sources (one per line):", classes="form-label"))
        widgets.append(Static("Enter audio file paths or URLs, one per line", classes="small-label"))
        widgets.append(TextArea("/path/to/audio1.mp3\n/path/to/podcast.wav\n...", 
                               id="audio-source", classes="form-textarea-source"))
        widgets.append(Button("Browse Files", id="audio-browse", variant="default"))
        
        # Multi-line metadata
        widgets.append(Label("Titles (optional, one per source line):", classes="form-label"))
        widgets.append(TextArea("Audio Title 1\nAudio Title 2\n...", 
                               id="audio-title", classes="form-textarea-metadata"))
        
        widgets.append(Label("Authors (optional, one per source line):", classes="form-label"))
        widgets.append(TextArea("Speaker/Author 1\nSpeaker/Author 2\n...", 
                               id="audio-author", classes="form-textarea-metadata"))
        
        # Time range options
        widgets.append(Label("Time Range (optional, applies to all):", classes="form-label"))
        time_container = Horizontal(
            Vertical(
                Label("Start Time:"),
                Input(placeholder="HH:MM:SS or seconds", id="audio-start-time", classes="form-input")
            ),
            Vertical(
                Label("End Time:"),
                Input(placeholder="HH:MM:SS or seconds", id="audio-end-time", classes="form-input")
            ),
            classes="time-input-container"
        )
        widgets.append(time_container)
        
        # Processing options
        widgets.append(Label("Processing Options:", classes="form-label"))
        widgets.append(Checkbox("Enable transcription", True, id="audio-transcribe"))
        widgets.append(Checkbox("Enable Voice Activity Detection (VAD)", False, id="audio-vad"))
        widgets.append(Checkbox("Speaker diarization", False, id="audio-diarize"))
        widgets.append(Checkbox("Save original file (if downloaded)", False, id="audio-save-original"))
        
        # Analysis options
        # Get available providers from app config
        providers = list(self.app_instance.app_config.get("api_settings", {}).keys()) if self.app_instance.app_config else []
        provider_options = [(name, name) for name in providers] if providers else [("No providers configured", "none")]
        default_provider = provider_options[0][1] if provider_options and provider_options[0][1] != "none" else "none"
        
        analysis_container = Container(
            Label("Analysis Options:", classes="form-label"),
            Checkbox("Enable LLM Analysis", False, id="audio-enable-analysis"),
            Label("Analysis Provider:", classes="form-label"),
            Select(provider_options, id="audio-analysis-provider", value=default_provider),
            Label("Analysis Model:", classes="form-label"),
            Select([("Select provider first", "none")], id="audio-analysis-model", value="none"),
            Label("Analysis Prompt:", classes="form-label"),
            Horizontal(
                TextArea("Enter custom analysis prompt or load from library...", 
                         id="audio-prompt", classes="form-textarea"),
                Button("Load Prompt", id="audio-load-prompt", classes="load-prompt-btn"),
                classes="prompt-row"
            ),
            classes="analysis-section"
        )
        widgets.append(analysis_container)
        
        # Queue/Submit buttons
        queue_buttons = Horizontal(
            Button("Add to Queue", id="audio-add-queue", variant="default"),
            Button("Process Now", id="submit-audio", variant="primary"),
            classes="queue-buttons"
        )
        widgets.append(queue_buttons)
        
        return widgets
    
    def _create_pdf_form(self) -> list:
        """Create form widgets for PDF ingestion."""
        widgets = []
        
        widgets.append(Static("PDF Ingestion Settings", classes="form-title"))
        
        # Multi-line source input
        widgets.append(Label("PDF Files (one per line):", classes="form-label"))
        widgets.append(TextArea("/path/to/document.pdf\n/path/to/paper.pdf\n...", 
                               id="pdf-source", classes="form-textarea-source"))
        widgets.append(Button("Browse Files", id="pdf-browse", variant="default"))
        
        # Multi-line metadata
        widgets.append(Label("Titles (optional, one per source line):", classes="form-label"))
        widgets.append(TextArea("Document Title 1\nDocument Title 2\n...", 
                               id="pdf-title", classes="form-textarea-metadata"))
        
        widgets.append(Label("Authors (optional, one per source line):", classes="form-label"))
        widgets.append(TextArea("Author 1\nAuthor 2\n...", 
                               id="pdf-author", classes="form-textarea-metadata"))
        
        # Processing options
        widgets.append(Label("PDF Engine:", classes="form-label"))
        pdf_engines = [("PyMuPDF4LLM", "pymupdf4llm"), ("PyMuPDF", "pymupdf"), ("Docling", "docling")]
        widgets.append(Select(pdf_engines, id="pdf-engine", value="pymupdf4llm"))
        
        # Analysis options
        providers = list(self.app_instance.app_config.get("api_settings", {}).keys()) if self.app_instance.app_config else []
        provider_options = [(name, name) for name in providers] if providers else [("No providers configured", "none")]
        default_provider = provider_options[0][1] if provider_options and provider_options[0][1] != "none" else "none"
        
        analysis_container = Container(
            Label("Analysis Options:", classes="form-label"),
            Checkbox("Enable LLM Analysis", False, id="pdf-enable-analysis"),
            Label("Analysis Provider:", classes="form-label"),
            Select(provider_options, id="pdf-analysis-provider", value=default_provider),
            Label("Analysis Model:", classes="form-label"),
            Select([("Select provider first", "none")], id="pdf-analysis-model", value="none"),
            Label("Analysis Prompt:", classes="form-label"),
            Horizontal(
                TextArea("Enter custom analysis prompt...", 
                         id="pdf-prompt", classes="form-textarea"),
                Button("Load Prompt", id="pdf-load-prompt", classes="load-prompt-btn"),
                classes="prompt-row"
            ),
            classes="analysis-section"
        )
        widgets.append(analysis_container)
        
        # Queue/Submit buttons
        queue_buttons = Horizontal(
            Button("Add to Queue", id="pdf-add-queue", variant="default"),
            Button("Process Now", id="submit-pdf", variant="primary"),
            classes="queue-buttons"
        )
        widgets.append(queue_buttons)
        
        return widgets
    
    def _create_document_form(self) -> list:
        """Create form widgets for document ingestion."""
        widgets = []
        
        widgets.append(Static("Document Ingestion Settings", classes="form-title"))
        
        # Multi-line source input
        widgets.append(Label("Document Files (one per line):", classes="form-label"))
        widgets.append(TextArea("/path/to/document.txt\n/path/to/article.md\n...", 
                               id="doc-source", classes="form-textarea-source"))
        widgets.append(Button("Browse Files", id="doc-browse", variant="default"))
        
        # Multi-line metadata
        widgets.append(Label("Titles (optional, one per source line):", classes="form-label"))
        widgets.append(TextArea("Document Title 1\nDocument Title 2\n...", 
                               id="doc-title", classes="form-textarea-metadata"))
        
        # Analysis options
        analysis_container = Container(classes="analysis-section")
        widgets.append(analysis_container)
        analysis_container.mount(Label("Analysis Options:", classes="form-label"))
        analysis_container.mount(Checkbox("Enable LLM Analysis", False, id="doc-enable-analysis"))
        
        providers = list(self.app_instance.app_config.get("api_settings", {}).keys()) if self.app_instance.app_config else []
        provider_options = [(name, name) for name in providers] if providers else [("No providers configured", "none")]
        
        analysis_container.mount(Label("Analysis Provider:", classes="form-label"))
        default_provider = provider_options[0][1] if provider_options and provider_options[0][1] != "none" else "none"
        analysis_container.mount(Select(provider_options, id="doc-analysis-provider", value=default_provider))
        
        analysis_container.mount(Label("Analysis Model:", classes="form-label"))
        analysis_container.mount(Select([("Select provider first", "none")], id="doc-analysis-model", value="none"))
        
        analysis_container.mount(Label("Analysis Prompt:", classes="form-label"))
        prompt_row = Horizontal(classes="prompt-row")
        analysis_container.mount(prompt_row)
        prompt_row.mount(TextArea("Enter custom analysis prompt...", 
                                 id="doc-prompt", classes="form-textarea"))
        prompt_row.mount(Button("Load Prompt", id="doc-load-prompt", classes="load-prompt-btn"))
        
        # Queue/Submit buttons
        queue_buttons = Horizontal(classes="queue-buttons")
        widgets.append(queue_buttons)
        queue_buttons.mount(Button("Add to Queue", id="doc-add-queue", variant="default"))
        queue_buttons.mount(Button("Process Now", id="submit-doc", variant="primary"))
        
        return widgets
    
    def _create_web_form(self) -> list:
        """Create form widgets for web content ingestion."""
        widgets = []
        
        widgets.append(Static("Web Content Ingestion", classes="form-title"))
        
        # Multi-line URL input
        widgets.append(Label("Web URLs (one per line):", classes="form-label"))
        widgets.append(TextArea("https://example.com/article1\nhttps://example.com/article2\n...", 
                               id="web-url", classes="form-textarea-source"))
        
        # Multi-line metadata
        widgets.append(Label("Titles (optional, one per URL):", classes="form-label"))
        widgets.append(TextArea("Article Title 1\nArticle Title 2\n...", 
                               id="web-title", classes="form-textarea-metadata"))
        
        # Processing options
        widgets.append(Checkbox("Use cookies for scraping", False, id="web-cookies"))
        widgets.append(Checkbox("Save scraped content", False, id="web-save-original"))
        
        # Analysis options
        analysis_container = Container(classes="analysis-section")
        widgets.append(analysis_container)
        analysis_container.mount(Label("Analysis Options:", classes="form-label"))
        analysis_container.mount(Checkbox("Enable LLM Analysis", False, id="web-enable-analysis"))
        
        providers = list(self.app_instance.app_config.get("api_settings", {}).keys()) if self.app_instance.app_config else []
        provider_options = [(name, name) for name in providers] if providers else [("No providers configured", "none")]
        
        analysis_container.mount(Label("Analysis Provider:", classes="form-label"))
        default_provider = provider_options[0][1] if provider_options and provider_options[0][1] != "none" else "none"
        analysis_container.mount(Select(provider_options, id="web-analysis-provider", value=default_provider))
        
        analysis_container.mount(Label("Analysis Model:", classes="form-label"))
        analysis_container.mount(Select([("Select provider first", "none")], id="web-analysis-model", value="none"))
        
        analysis_container.mount(Label("Analysis Prompt:", classes="form-label"))
        prompt_row = Horizontal(classes="prompt-row")
        analysis_container.mount(prompt_row)
        prompt_row.mount(TextArea("Enter custom analysis prompt...", 
                                 id="web-prompt", classes="form-textarea"))
        prompt_row.mount(Button("Load Prompt", id="web-load-prompt", classes="load-prompt-btn"))
        
        # Queue/Submit buttons
        queue_buttons = Horizontal(classes="queue-buttons")
        widgets.append(queue_buttons)
        queue_buttons.mount(Button("Add to Queue", id="web-add-queue", variant="default"))
        queue_buttons.mount(Button("Process Now", id="submit-web", variant="primary"))
        
        return widgets
    
    def _create_ebook_form(self) -> list:
        """Create form widgets for ebook ingestion."""
        widgets = []
        
        widgets.append(Static("E-Book Ingestion Settings", classes="form-title"))
        
        # Multi-line source input
        widgets.append(Label("E-Book Files (one per line):", classes="form-label"))
        widgets.append(TextArea("/path/to/book.epub\n/path/to/novel.mobi\n...", 
                               id="ebook-source", classes="form-textarea-source"))
        widgets.append(Button("Browse Files", id="ebook-browse", variant="default"))
        
        # Multi-line metadata
        widgets.append(Label("Titles (optional, one per source line):", classes="form-label"))
        widgets.append(TextArea("Book Title 1\nBook Title 2\n...", 
                               id="ebook-title", classes="form-textarea-metadata"))
        
        widgets.append(Label("Authors (optional, one per source line):", classes="form-label"))
        widgets.append(TextArea("Author 1\nAuthor 2\n...", 
                               id="ebook-author", classes="form-textarea-metadata"))
        
        # Analysis options
        analysis_container = Container(classes="analysis-section")
        widgets.append(analysis_container)
        analysis_container.mount(Label("Analysis Options:", classes="form-label"))
        analysis_container.mount(Checkbox("Enable LLM Analysis", False, id="ebook-enable-analysis"))
        
        providers = list(self.app_instance.app_config.get("api_settings", {}).keys()) if self.app_instance.app_config else []
        provider_options = [(name, name) for name in providers] if providers else [("No providers configured", "none")]
        
        analysis_container.mount(Label("Analysis Provider:", classes="form-label"))
        default_provider = provider_options[0][1] if provider_options and provider_options[0][1] != "none" else "none"
        analysis_container.mount(Select(provider_options, id="ebook-analysis-provider", value=default_provider))
        
        analysis_container.mount(Label("Analysis Model:", classes="form-label"))
        analysis_container.mount(Select([("Select provider first", "none")], id="ebook-analysis-model", value="none"))
        
        analysis_container.mount(Label("Analysis Prompt:", classes="form-label"))
        prompt_row = Horizontal(classes="prompt-row")
        analysis_container.mount(prompt_row)
        prompt_row.mount(TextArea("Enter custom analysis prompt...", 
                                 id="ebook-prompt", classes="form-textarea"))
        prompt_row.mount(Button("Load Prompt", id="ebook-load-prompt", classes="load-prompt-btn"))
        
        # Queue/Submit buttons
        queue_buttons = Horizontal(classes="queue-buttons")
        widgets.append(queue_buttons)
        queue_buttons.mount(Button("Add to Queue", id="ebook-add-queue", variant="default"))
        queue_buttons.mount(Button("Process Now", id="submit-ebook", variant="primary"))
        
        return widgets
    
    def _create_notes_form(self) -> list:
        """Create form widgets for notes import."""
        widgets = []
        
        widgets.append(Static("Notes Import Settings", classes="form-title"))
        
        # Multi-line source input
        widgets.append(Label("Notes Files/Directories (one per line):", classes="form-label"))
        widgets.append(TextArea("/path/to/notes.md\n/path/to/notes_folder/\n...", 
                               id="notes-source", classes="form-textarea-source"))
        widgets.append(Button("Browse", id="notes-browse", variant="default"))
        
        # Processing options
        widgets.append(Label("Import Options:", classes="form-label"))
        widgets.append(Checkbox("Import as templates", False, id="notes-templates"))
        widgets.append(Checkbox("Enable sync", True, id="notes-sync"))
        widgets.append(Checkbox("Process YAML frontmatter", True, id="notes-frontmatter"))
        widgets.append(Checkbox("Import subdirectories recursively", True, id="notes-recursive"))
        
        # Queue/Submit buttons
        queue_buttons = Horizontal(classes="queue-buttons")
        widgets.append(queue_buttons)
        queue_buttons.mount(Button("Add to Queue", id="notes-add-queue", variant="default"))
        queue_buttons.mount(Button("Import Now", id="submit-notes", variant="primary"))
        
        return widgets
    
    def _create_character_form(self) -> list:
        """Create form widgets for character card import."""
        widgets = []
        
        widgets.append(Static("Character Card Import", classes="form-title"))
        
        # Multi-line source input
        widgets.append(Label("Character Card Files (one per line):", classes="form-label"))
        widgets.append(Static("Supports .json, .png with embedded data", classes="small-label"))
        widgets.append(TextArea("/path/to/character1.json\n/path/to/character2.png\n...", 
                               id="char-source", classes="form-textarea-source"))
        widgets.append(Button("Browse Files", id="char-browse", variant="default"))
        
        # Processing options
        widgets.append(Label("Card Format:", classes="form-label"))
        formats = [("Auto-detect", "auto"), ("Tavern/SillyTavern", "tavern"), ("CharacterAI", "cai")]
        widgets.append(Select(formats, id="char-format", value="auto"))
        
        widgets.append(Label("Import Options:", classes="form-label"))
        widgets.append(Checkbox("Extract and save character images", True, id="char-extract-image"))
        widgets.append(Checkbox("Import example messages", True, id="char-import-examples"))
        
        # Queue/Submit buttons
        queue_buttons = Horizontal(
            Button("Add to Queue", id="char-add-queue", variant="default"),
            Button("Import Now", id="submit-char", variant="primary"),
            classes="queue-buttons"
        )
        widgets.append(queue_buttons)
        
        return widgets
    
    def _create_conversation_form(self) -> list:
        """Create form widgets for conversation import."""
        widgets = []
        
        widgets.append(Static("Conversation Import", classes="form-title"))
        
        # Multi-line source input
        widgets.append(Label("Conversation Files (one per line):", classes="form-label"))
        widgets.append(TextArea("/path/to/conversation1.json\n/path/to/chat_export.json\n...", 
                               id="conv-source", classes="form-textarea-source"))
        widgets.append(Button("Browse Files", id="conv-browse", variant="default"))
        
        # Processing options
        widgets.append(Label("Import Format:", classes="form-label"))
        formats = [("Auto-detect", "auto"), ("JSON", "json"), ("ChatGPT Export", "chatgpt"), ("Discord", "discord")]
        widgets.append(Select(formats, id="conv-format", value="auto"))
        
        widgets.append(Label("Import Options:", classes="form-label"))
        widgets.append(Checkbox("Parse timestamps", True, id="conv-timestamps"))
        widgets.append(Checkbox("Import attachments/images", True, id="conv-attachments"))
        widgets.append(Checkbox("Preserve message IDs", False, id="conv-preserve-ids"))
        
        # Queue/Submit buttons
        queue_buttons = Horizontal(
            Button("Add to Queue", id="conv-add-queue", variant="default"),
            Button("Import Now", id="submit-conv", variant="primary"),
            classes="queue-buttons"
        )
        widgets.append(queue_buttons)
        
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