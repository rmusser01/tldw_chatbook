"""
MediaViewerPanel - Viewer for media content and metadata.

This component provides:
- Metadata display and editing
- Content viewing with formatting options
- Content search functionality
- Analysis display
"""

from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Static, Button, Label, Input, TextArea, Markdown,
    Checkbox, Collapsible, TabbedContent, TabPane
)
from textual.message import Message
from loguru import logger
import re

if TYPE_CHECKING:
    from ...app import TldwCli


class ContentSearchEvent(Message):
    """Event fired when searching within content."""
    
    def __init__(self, search_term: str) -> None:
        super().__init__()
        self.search_term = search_term


class MediaViewerPanel(Container):
    """
    Viewer panel for media content and metadata.
    
    Provides comprehensive viewing and editing capabilities for media items.
    """
    
    DEFAULT_CSS = """
    MediaViewerPanel {
        width: 65%;
        height: 100%;
        layout: vertical;
    }
    
    MediaViewerPanel .viewer-header {
        display: none;
    }
    
    MediaViewerPanel .metadata-section {
        width: 100%;
        padding: 1;
        background: $boost;
        margin-bottom: 1;
    }
    
    MediaViewerPanel .metadata-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    MediaViewerPanel .metadata-field {
        margin-bottom: 0;
    }
    
    MediaViewerPanel .metadata-label {
        text-style: bold;
        color: $text-muted;
    }
    
    MediaViewerPanel .metadata-value {
        color: $text;
    }
    
    MediaViewerPanel Collapsible {
        margin-top: 1;
    }
    
    MediaViewerPanel .metadata-buttons {
        layout: horizontal;
        height: 3;
        width: 100%;
        align: center middle;
    }
    
    MediaViewerPanel .metadata-buttons Button {
        min-width: 10;
        margin: 0 1;
    }
    
    MediaViewerPanel .metadata-buttons #edit-button {
        dock: left;
    }
    
    MediaViewerPanel .metadata-buttons #delete-button {
        dock: right;
    }
    
    MediaViewerPanel .edit-section {
        width: 100%;
        padding: 1;
        background: $surface;
    }
    
    MediaViewerPanel .edit-field {
        margin-bottom: 1;
    }
    
    MediaViewerPanel .edit-label {
        text-style: bold;
        margin-bottom: 0;
    }
    
    MediaViewerPanel .edit-input {
        width: 100%;
        margin-top: 0;
    }
    
    MediaViewerPanel .edit-actions {
        layout: horizontal;
        height: 3;
        margin-top: 1;
    }
    
    MediaViewerPanel .edit-actions Button {
        margin-right: 1;
        min-width: 10;
    }
    
    MediaViewerPanel .content-search {
        dock: top;
        height: auto;
        padding: 1;
        background: $boost;
        border-bottom: solid $background-darken-1;
    }
    
    MediaViewerPanel .search-controls {
        layout: horizontal;
        height: 3;
    }
    
    MediaViewerPanel .content-search-input {
        width: 1fr;
        margin-right: 1;
    }
    
    MediaViewerPanel .search-nav-button {
        width: auto;
        min-width: 5;
        margin-right: 1;
    }
    
    MediaViewerPanel .search-status {
        width: auto;
        color: $text-muted;
        padding: 0 1;
    }
    
    MediaViewerPanel .content-viewer {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    MediaViewerPanel .no-selection {
        text-align: center;
        color: $text-muted;
        padding: 4;
    }
    
    MediaViewerPanel TabbedContent {
        height: 1fr;
    }
    
    MediaViewerPanel TabPane {
        padding: 1;
    }
    """
    
    # Reactive properties
    media_data: reactive[Optional[Dict[str, Any]]] = reactive(None)
    edit_mode: reactive[bool] = reactive(False)
    format_for_reading: reactive[bool] = reactive(False)
    search_matches: reactive[List[Tuple[int, int]]] = reactive([])
    current_match: reactive[int] = reactive(-1)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the viewer panel."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._original_data = None
        
    def compose(self) -> ComposeResult:
        """Compose the viewer panel UI."""
        # Header
        with Container(classes="viewer-header"):
            yield Label("Media Viewer", classes="viewer-title")
        
        # Main content area with tabs
        with TabbedContent():
            # Metadata tab
            with TabPane("Metadata", id="metadata-tab"):
                # View mode
                with Container(id="metadata-view", classes="metadata-section"):
                    yield Static("", id="metadata-display", classes="metadata-display")
                    with Collapsible(title="Actions", collapsed=True):
                        with Horizontal(classes="metadata-buttons"):
                            yield Button("Edit", id="edit-button", variant="primary")
                            yield Button("Delete", id="delete-button", variant="error")
                
                # Edit mode (hidden by default)
                with Container(id="metadata-edit", classes="edit-section hidden"):
                    with Vertical(classes="edit-fields"):
                        yield Label("Title:", classes="edit-label")
                        yield Input(id="edit-title", classes="edit-input")
                        
                        yield Label("Author:", classes="edit-label")
                        yield Input(id="edit-author", classes="edit-input")
                        
                        yield Label("URL:", classes="edit-label")
                        yield Input(id="edit-url", classes="edit-input")
                        
                        yield Label("Keywords (comma-separated):", classes="edit-label")
                        yield Input(id="edit-keywords", classes="edit-input")
                    
                    with Horizontal(classes="edit-actions"):
                        yield Button("Save", id="save-button", variant="success")
                        yield Button("Cancel", id="cancel-button", variant="default")
            
            # Content tab
            with TabPane("Content", id="content-tab"):
                # Content search bar
                with Container(classes="content-search"):
                    with Horizontal(classes="search-controls"):
                        yield Input(
                            placeholder="Search within content...",
                            id="content-search-input",
                            classes="content-search-input"
                        )
                        yield Button("◀", id="prev-match", classes="search-nav-button")
                        yield Button("▶", id="next-match", classes="search-nav-button")
                        yield Static("", id="search-status", classes="search-status")
                    
                    yield Checkbox(
                        "Format for reading",
                        id="format-reading-checkbox",
                        value=False
                    )
                
                # Content display
                with VerticalScroll(classes="content-viewer"):
                    yield Markdown("", id="content-display")
            
            # Analysis tab
            with TabPane("Analysis", id="analysis-tab"):
                with VerticalScroll(classes="content-viewer"):
                    yield Markdown("", id="analysis-display")
    
    def watch_media_data(self, media_data: Optional[Dict[str, Any]]) -> None:
        """Update display when media data changes."""
        if media_data:
            self.update_metadata_display()
            self.update_content_display()
            self.update_analysis_display()
        else:
            self.clear_display()
    
    def watch_edit_mode(self, edit_mode: bool) -> None:
        """Toggle between view and edit modes."""
        try:
            metadata_view = self.query_one("#metadata-view")
            metadata_edit = self.query_one("#metadata-edit")
            
            if edit_mode:
                metadata_view.add_class("hidden")
                metadata_edit.remove_class("hidden")
                self.populate_edit_fields()
            else:
                metadata_view.remove_class("hidden")
                metadata_edit.add_class("hidden")
        except:
            pass
    
    def watch_format_for_reading(self, format: bool) -> None:
        """Update content formatting."""
        self.update_content_display()
    
    def update_metadata_display(self) -> None:
        """Update the metadata display."""
        if not self.media_data:
            return
            
        try:
            display = self.query_one("#metadata-display", Static)
            
            lines = []
            
            # Title
            lines.append(f"[bold]Title:[/bold] {self.media_data.get('title', 'Untitled')}")
            
            # Type
            lines.append(f"[bold]Type:[/bold] {self.media_data.get('media_type', 'Unknown')}")
            
            # Author
            if self.media_data.get('author'):
                lines.append(f"[bold]Author:[/bold] {self.media_data['author']}")
            
            # URL
            if self.media_data.get('url'):
                lines.append(f"[bold]URL:[/bold] {self.media_data['url']}")
            
            # Keywords
            if self.media_data.get('keywords'):
                keywords = self.media_data['keywords']
                if isinstance(keywords, list):
                    keywords = ", ".join(keywords)
                lines.append(f"[bold]Keywords:[/bold] {keywords}")
            
            # Dates
            if self.media_data.get('created_at'):
                lines.append(f"[bold]Created:[/bold] {self.media_data['created_at']}")
            
            if self.media_data.get('updated_at'):
                lines.append(f"[bold]Updated:[/bold] {self.media_data['updated_at']}")
            
            # Status
            if self.media_data.get('is_deleted'):
                lines.append("[red][bold]Status:[/bold] DELETED[/red]")
            
            display.update("\n".join(lines))
        except Exception as e:
            logger.error(f"Error updating metadata display: {e}")
    
    def update_content_display(self) -> None:
        """Update the content display."""
        if not self.media_data:
            return
            
        try:
            content_display = self.query_one("#content-display", Markdown)
            
            content = self.media_data.get('content', '')
            if not content:
                content_display.update("*No content available*")
                return
            
            if self.format_for_reading:
                # Format for better readability
                content = self._format_content_for_reading(content)
            
            # Apply search highlighting if active
            if self.search_matches:
                content = self._highlight_matches(content)
                # Update search status
                self._update_search_status()
            
            # Clear and update to ensure proper refresh
            content_display.update("")
            content_display.update(content)
        except Exception as e:
            logger.error(f"Error updating content display: {e}")
    
    def update_analysis_display(self) -> None:
        """Update the analysis display."""
        if not self.media_data:
            return
            
        try:
            analysis_display = self.query_one("#analysis-display", Markdown)
            
            # Check for analysis data
            analysis = self.media_data.get('analysis', '')
            if not analysis:
                # Check for summary as fallback
                analysis = self.media_data.get('summary', '')
            
            if not analysis:
                analysis_display.update("*No analysis available*")
                return
            
            analysis_display.update(analysis)
        except Exception as e:
            logger.error(f"Error updating analysis display: {e}")
    
    def populate_edit_fields(self) -> None:
        """Populate edit fields with current data."""
        if not self.media_data:
            return
            
        try:
            self.query_one("#edit-title", Input).value = self.media_data.get('title', '')
            self.query_one("#edit-author", Input).value = self.media_data.get('author', '')
            self.query_one("#edit-url", Input).value = self.media_data.get('url', '')
            
            keywords = self.media_data.get('keywords', [])
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)
            self.query_one("#edit-keywords", Input).value = keywords
            
            # Store original data for cancel
            self._original_data = self.media_data.copy()
        except Exception as e:
            logger.error(f"Error populating edit fields: {e}")
    
    def clear_display(self) -> None:
        """Clear all displays when no item is selected."""
        try:
            self.query_one("#metadata-display", Static).update("*No item selected*")
            self.query_one("#content-display", Markdown).update("*No item selected*")
            self.query_one("#analysis-display", Markdown).update("*No item selected*")
            # Clear search when no item is selected
            self.clear_search()
            # Clear search input
            search_input = self.query_one("#content-search-input", Input)
            search_input.value = ""
        except:
            pass
    
    def _format_content_for_reading(self, content: str) -> str:
        """Format content for better readability."""
        # Add line breaks after sentences
        content = re.sub(r'([.!?])\s+', r'\1\n\n', content)
        
        # Add headers for sections if detected
        content = re.sub(r'^(\d+\.)\s+', r'## \1 ', content, flags=re.MULTILINE)
        
        return content
    
    def _highlight_matches(self, content: str) -> str:
        """Highlight search matches in content."""
        if not self.search_matches:
            return content
            
        # Sort matches by position
        sorted_matches = sorted(self.search_matches)
        
        # Build highlighted content
        result = []
        last_end = 0
        
        for i, (start, end) in enumerate(sorted_matches):
            # Add text before match
            result.append(content[last_end:start])
            
            # Add highlighted match
            match_text = content[start:end]
            if i == self.current_match:
                # Current match gets special highlighting with inline code style
                result.append(f" **`▶ {match_text} ◀`** ")
            else:
                # Other matches get regular highlighting
                result.append(f" `{match_text}` ")
            
            last_end = end
            
        # Add remaining text
        result.append(content[last_end:])
        
        return ''.join(result)
    
    @on(Button.Pressed, "#edit-button")
    def handle_edit_button(self) -> None:
        """Handle edit button press."""
        self.edit_mode = True
    
    @on(Button.Pressed, "#save-button")
    def handle_save_button(self) -> None:
        """Handle save button press."""
        if not self.media_data:
            return
            
        try:
            # Gather edited data
            title = self.query_one("#edit-title", Input).value
            author = self.query_one("#edit-author", Input).value
            url = self.query_one("#edit-url", Input).value
            keywords = self.query_one("#edit-keywords", Input).value
            
            # Parse keywords
            keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
            
            # Post update event
            from ...Event_Handlers.media_events import MediaMetadataUpdateEvent
            self.post_message(MediaMetadataUpdateEvent(
                media_id=self.media_data['id'],
                title=title,
                media_type=self.media_data.get('media_type', ''),
                author=author,
                url=url,
                keywords=keyword_list,
                type_slug=""  # Will be set by MediaWindow
            ))
            
            # Exit edit mode
            self.edit_mode = False
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel_button(self) -> None:
        """Handle cancel button press."""
        self.edit_mode = False
    
    @on(Button.Pressed, "#delete-button")
    def handle_delete_button(self) -> None:
        """Handle delete button press."""
        if self.media_data:
            from ...Event_Handlers.media_events import MediaDeleteConfirmationEvent
            self.post_message(MediaDeleteConfirmationEvent(
                media_id=self.media_data['id'],
                media_title=self.media_data.get('title', 'Untitled'),
                type_slug=""  # Will be set by MediaWindow
            ))
    
    @on(Input.Changed, "#content-search-input")
    def handle_content_search(self, event: Input.Changed) -> None:
        """Handle content search input."""
        if event.value:
            self.search_content(event.value)
        else:
            self.clear_search()
    
    @on(Button.Pressed, "#prev-match")
    def handle_prev_match(self) -> None:
        """Navigate to previous search match."""
        if self.search_matches:
            if self.current_match > 0:
                self.current_match -= 1
            else:
                # Wrap around to last match
                self.current_match = len(self.search_matches) - 1
            self.highlight_current_match()
    
    @on(Button.Pressed, "#next-match")
    def handle_next_match(self) -> None:
        """Navigate to next search match."""
        if self.search_matches:
            if self.current_match < len(self.search_matches) - 1:
                self.current_match += 1
            else:
                # Wrap around to first match
                self.current_match = 0
            self.highlight_current_match()
    
    @on(Checkbox.Changed, "#format-reading-checkbox")
    def handle_format_change(self, event: Checkbox.Changed) -> None:
        """Handle reading format checkbox change."""
        self.format_for_reading = event.value
    
    def search_content(self, search_term: str) -> None:
        """Search for term in content."""
        if not self.media_data or not search_term:
            self.clear_search()
            return
            
        content = self.media_data.get('content', '')
        if not content:
            self.search_matches = []
            self.current_match = -1
            self._update_search_status()
            return
            
        # Find all matches (case-insensitive)
        search_lower = search_term.lower()
        content_lower = content.lower()
        
        matches = []
        start = 0
        while True:
            pos = content_lower.find(search_lower, start)
            if pos == -1:
                break
            matches.append((pos, pos + len(search_term)))
            start = pos + 1
            
        self.search_matches = matches
        self.current_match = 0 if matches else -1
        self._update_search_status()
        self.update_content_display()
    
    def clear_search(self) -> None:
        """Clear search results."""
        self.search_matches = []
        self.current_match = -1
        self.update_content_display()
    
    def highlight_current_match(self) -> None:
        """Highlight the current search match and scroll to it."""
        self.update_content_display()
        # TODO: Implement scrolling to match position
    
    def _update_search_status(self) -> None:
        """Update the search status display."""
        try:
            status_widget = self.query_one("#search-status", Static)
            if not self.search_matches:
                status_widget.update("")
            else:
                current = self.current_match + 1 if self.current_match >= 0 else 0
                total = len(self.search_matches)
                status_widget.update(f"{current}/{total}")
        except Exception:
            pass
    
    def load_media(self, media_data: Dict[str, Any]) -> None:
        """Load new media data into the viewer."""
        self.media_data = media_data
        self.edit_mode = False
        self.clear_search()
        # Clear search input when loading new media
        try:
            search_input = self.query_one("#content-search-input", Input)
            search_input.value = ""
        except:
            pass