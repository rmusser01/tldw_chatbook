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
    Checkbox, Collapsible, TabbedContent, TabPane, Select
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
        width: 2fr;
        height: 100%;
        layout: vertical;
    }
    
    MediaViewerPanel .viewer-header {
        display: none;
    }
    
    MediaViewerPanel .tab-header {
        height: 3;
        layout: horizontal;
        align-vertical: middle;
        padding: 0;
        margin-bottom: 0;
    }
    
    MediaViewerPanel .collapse-media-list {
        height: 3;
        width: auto;
        min-width: 5;
        margin: 0 1 0 0;
        background: $boost;
        border: solid $primary;
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
        height: auto;
    }
    
    MediaViewerPanel Collapsible > Container {
        height: auto;
        padding: 0;
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
        width: 100%;
        min-height: 0;
    }
    
    MediaViewerPanel TabbedContent > ContentSwitcher {
        height: 1fr;
        width: 100%;
        min-height: 0;
    }
    
    MediaViewerPanel TabPane {
        padding: 0;
        height: 1fr;
        min-height: 0;
    }
    
    /* Specific rule for analysis tab to ensure it fills container */
    MediaViewerPanel #analysis-tab {
        height: 1fr;
        padding: 0;
        min-height: 0;
    }
    
    /* Force the analysis scroll container to work */
    MediaViewerPanel #analysis-scroll-fix {
        height: 1fr;
        width: 100%;
        overflow-y: scroll;
        padding: 1;
    }
    
    /* Ensure collapsible doesn't constrain height */
    MediaViewerPanel #analysis-api-settings {
        height: auto;
        margin-bottom: 1;
    }
    
    /* Ensure all containers inside use auto height */
    MediaViewerPanel #analysis-scroll-fix > * {
        height: auto;
    }
    
    /* VerticalScroll wrapper for analysis content */
    MediaViewerPanel .analysis-content-scroll {
        height: 1fr;
        width: 100%;
        padding: 1;
    }
    
    MediaViewerPanel .analysis-controls {
        padding: 1;
        background: $boost;
        margin-bottom: 1;
        height: auto;
    }
    
    MediaViewerPanel .analysis-display-scroll {
        height: 1fr;
        padding: 1;
    }
    
    MediaViewerPanel #analysis-display {
        min-height: 5;
        max-height: 30;
        margin: 1;
        height: auto;
        padding: 1;
        border: solid $primary;
        background: $surface;
        overflow-y: auto;
    }
    
    MediaViewerPanel .provider-row {
        layout: horizontal;
        height: 3;
        margin-top: 1;
        margin-bottom: 1;
    }
    
    MediaViewerPanel .provider-row Select {
        width: 1fr;
        margin-right: 1;
    }
    
    MediaViewerPanel .compact-collapsible {
        height: auto;
        padding: 0;
        margin-bottom: 1;
    }
    
    MediaViewerPanel .compact-collapsible > Container {
        height: auto;
        padding: 1 0;
    }
    
    MediaViewerPanel .compact-collapsible CollapsibleTitle {
        padding: 0 1;
    }
    
    MediaViewerPanel .api-params-row {
        layout: horizontal;
        height: auto;
        margin-top: 1;
        margin-bottom: 0;
    }
    
    MediaViewerPanel .param-group {
        layout: vertical;
        width: 1fr;
        height: auto;
        margin: 0 1;
    }
    
    MediaViewerPanel .param-group Label {
        margin-bottom: 0;
        text-style: bold;
        color: $text-muted;
        height: 1;
    }
    
    MediaViewerPanel .param-group Input {
        width: 100%;
        height: 3;
        margin-bottom: 0;
    }
    
    MediaViewerPanel .prompt-label {
        margin-top: 1;
        margin-bottom: 0;
        text-style: bold;
    }
    
    MediaViewerPanel .prompt-textarea {
        height: 4;
        margin-bottom: 1;
        width: 100%;
    }
    
    MediaViewerPanel #generate-analysis-btn {
        width: auto;
        margin: 1;
        height: 3;
    }
    
    MediaViewerPanel .analysis-actions {
        layout: horizontal;
        height: 3;
        margin-top: 1;
        margin-bottom: 2;
        padding: 1;
    }
    
    MediaViewerPanel .analysis-actions Button {
        margin-right: 1;
        min-width: 10;
    }
    
    MediaViewerPanel .bottom-spacer {
        height: 10;
        width: 100%;
        min-height: 10;
    }
    """
    
    # Reactive properties
    media_data: reactive[Optional[Dict[str, Any]]] = reactive(None)
    edit_mode: reactive[bool] = reactive(False)
    format_for_reading: reactive[bool] = reactive(False)
    search_matches: reactive[List[Tuple[int, int]]] = reactive([])
    current_match: reactive[int] = reactive(-1)
    current_analysis: reactive[Optional[str]] = reactive(None)
    has_existing_analysis: reactive[bool] = reactive(False)
    analysis_edit_mode: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the viewer panel."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._original_data = None
        
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        # Populate providers on mount
        self.populate_providers()
        
    def compose(self) -> ComposeResult:
        """Compose the viewer panel UI."""
        # Header
        with Container(classes="viewer-header"):
            yield Label("Media Viewer", classes="viewer-title")
        
        # Tab header with collapse button
        with Horizontal(classes="tab-header"):
            yield Button(
                "◀",
                id="collapse-media-list",
                classes="collapse-media-list"
            )
        
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
                # Wrap everything in a scrollable container
                with VerticalScroll(id="analysis-scroll-fix"):
                    # API Settings in a Collapsible
                    with Collapsible(title="API Settings", collapsed=False, id="analysis-api-settings", classes="compact-collapsible"):
                        # Provider and Model selection row
                        with Horizontal(classes="provider-row"):
                                yield Select(
                                    [],  # Will be populated on mount
                                    prompt="Select Provider",
                                    id="analysis-provider-select"
                                )
                                yield Select(
                                    [],  # Will be populated based on provider
                                    prompt="Select Model",
                                    id="analysis-model-select"
                                )
                            
                        # Temperature, Top-P, Min-P, Max Tokens settings row
                        with Horizontal(classes="api-params-row"):
                                with Vertical(classes="param-group"):
                                    yield Label("Temperature")
                                    yield Input(
                                        placeholder="0.7",
                                        id="analysis-temperature",
                                        value="0.7"
                                    )
                                with Vertical(classes="param-group"):
                                    yield Label("Top P")
                                    yield Input(
                                        placeholder="0.95",
                                        id="analysis-top-p",
                                        value="0.95"
                                    )
                                with Vertical(classes="param-group"):
                                    yield Label("Min P")
                                    yield Input(
                                        placeholder="0.05",
                                        id="analysis-min-p",
                                        value="0.05"
                                    )
                                with Vertical(classes="param-group"):
                                    yield Label("Max Tokens")
                                    yield Input(
                                        placeholder="4096",
                                        id="analysis-max-tokens",
                                        value="4096"
                                    )
                    
                    # Prompt search and filtering
                    yield Label("Search Prompts:", classes="prompt-label")
                    yield Input(
                        placeholder="Search for prompts...",
                        id="prompt-search-input"
                    )
                        
                    yield Label("Filter by Keywords:", classes="prompt-label")
                    yield Input(
                        placeholder="Enter keywords separated by commas...",
                        id="prompt-keyword-input"
                    )
                    
                    # Prompt selection dropdown
                    yield Select(
                        [],  # Will be populated by search results
                        prompt="Select a prompt",
                        id="prompt-select"
                    )
                    
                    # System prompt
                    yield Label("System Prompt:", classes="prompt-label")
                    yield TextArea(
                        "",
                        id="system-prompt-area",
                        classes="prompt-textarea"
                    )
                    
                    # User prompt
                    yield Label("User Prompt:", classes="prompt-label")
                    yield TextArea(
                        "",
                        id="user-prompt-area",
                        classes="prompt-textarea"
                    )
                    
                    # Generate button
                    yield Button(
                        "Generate Analysis",
                        id="generate-analysis-btn",
                        variant="primary"
                    )
                    
                    # Analysis display area
                    yield Markdown("", id="analysis-display")
                    
                    # Analysis action buttons
                    with Horizontal(classes="analysis-actions"):
                        yield Button("Save", id="save-analysis-btn", variant="success", disabled=True)
                        yield Button("Edit", id="edit-analysis-btn", variant="primary", disabled=True)
                        yield Button("Overwrite", id="overwrite-analysis-btn", variant="warning", disabled=True)
                    
                    # Add some padding at the bottom to ensure scrolling works
                    yield Static("", classes="bottom-spacer")
    
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
                self.has_existing_analysis = False
                self.current_analysis = None
            else:
                analysis_display.update(analysis)
                self.has_existing_analysis = True
                self.current_analysis = analysis
                
            # Update button states
            self._update_analysis_button_states()
            
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
    
    @on(Button.Pressed, "#collapse-media-list")
    def handle_collapse_media_list(self) -> None:
        """Handle collapse media list button press."""
        # Post a custom event that MediaWindow can listen for
        from ...Event_Handlers.media_events import MediaListCollapseEvent
        self.post_message(MediaListCollapseEvent())
    
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
        # Populate providers
        try:
            self.populate_providers()
        except Exception as e:
            logger.debug(f"Could not populate providers: {e}")
    
    # Analysis Methods
    def populate_providers(self) -> None:
        """Populate the provider dropdown with available LLM providers."""
        try:
            from ...config import get_cli_providers_and_models, load_settings
            
            providers_models = get_cli_providers_and_models()
            config = load_settings()
            analysis_defaults = config.get('analysis_defaults', {})
            
            if providers_models:
                provider_options = [(provider, provider) for provider in providers_models.keys()]
                provider_select = self.query_one("#analysis-provider-select", Select)
                provider_select.set_options(provider_options)
                
                # Set default provider from config or use first available
                default_provider = analysis_defaults.get('provider', provider_options[0][0] if provider_options else None)
                if default_provider and any(p[0] == default_provider for p in provider_options):
                    provider_select.value = default_provider
                    self.update_models_for_provider(default_provider)
                elif provider_options:
                    provider_select.value = provider_options[0][0]
                    self.update_models_for_provider(provider_options[0][0])
                    
            # Set default temperature, top_p, min_p, max_tokens
            temp_input = self.query_one("#analysis-temperature", Input)
            temp_input.value = str(analysis_defaults.get('temperature', '0.7'))
            
            top_p_input = self.query_one("#analysis-top-p", Input)
            top_p_input.value = str(analysis_defaults.get('top_p', '0.95'))
            
            min_p_input = self.query_one("#analysis-min-p", Input)
            min_p_input.value = str(analysis_defaults.get('min_p', '0.05'))
            
            max_tokens_input = self.query_one("#analysis-max-tokens", Input)
            max_tokens_input.value = str(analysis_defaults.get('max_tokens', '4096'))
            
            # Set default system prompt
            system_prompt_area = self.query_one("#system-prompt-area", TextArea)
            system_prompt_area.text = analysis_defaults.get('system_prompt', 'You are an AI assistant specialized in analyzing media content.')
            
        except Exception as e:
            logger.error(f"Error populating providers: {e}")
    
    def update_models_for_provider(self, provider: str) -> None:
        """Update model dropdown based on selected provider."""
        try:
            from ...config import get_cli_providers_and_models, load_settings
            
            providers_models = get_cli_providers_and_models()
            config = load_settings()
            analysis_defaults = config.get('analysis_defaults', {})
            
            models_list = providers_models.get(provider, [])
            
            if models_list:
                model_options = [(model, model) for model in models_list]
                model_select = self.query_one("#analysis-model-select", Select)
                model_select.set_options(model_options)
                
                # Set default model from config or use first available
                default_model = analysis_defaults.get('model')
                if default_model and any(m[0] == default_model for m in model_options):
                    model_select.value = default_model
                else:
                    model_select.value = model_options[0][0]
                
                # Select first model if available
                if model_options:
                    model_select.value = model_options[0][0]
            else:
                # No models available for this provider
                model_select = self.query_one("#analysis-model-select", Select)
                model_select.set_options([])
            
        except Exception as e:
            logger.error(f"Error updating models for provider {provider}: {e}")
    
    @work(thread=True)
    def search_prompts(self, search_term: str, keywords: str = "") -> None:
        """Search for prompts in the database."""
        try:
            from ...DB.Prompts_DB import get_prompts_db
            
            prompts_db = get_prompts_db()
            if not prompts_db:
                return
            
            # Parse keywords
            keyword_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
            
            # Search prompts
            if keyword_list:
                results = prompts_db.search_prompts_by_keyword(keyword_list, search_term)
            else:
                results = prompts_db.search_prompts(search_term) if search_term else prompts_db.get_all_prompts()
            
            # Format results for Select widget
            options = [(str(p['id']), f"{p['name']} - {p['description'][:50]}...") for p in results]
            
            # Update select widget from thread
            self.call_from_thread(self._update_prompt_select, options)
            
        except Exception as e:
            logger.error(f"Error searching prompts: {e}")
    
    def _update_prompt_select(self, options: List[Tuple[str, str]]) -> None:
        """Update prompt select options from thread."""
        try:
            prompt_select = self.query_one("#prompt-select", Select)
            prompt_select.set_options(options)
        except:
            pass
    
    def load_prompt_details(self, prompt_id: str) -> None:
        """Load selected prompt into text areas."""
        try:
            from ...DB.Prompts_DB import get_prompts_db
            
            prompts_db = get_prompts_db()
            if not prompts_db:
                return
            
            # Get prompt details
            prompt = prompts_db.get_prompt_details(int(prompt_id))
            if not prompt:
                return
            
            # Update text areas
            system_area = self.query_one("#system-prompt-area", TextArea)
            user_area = self.query_one("#user-prompt-area", TextArea)
            
            system_area.text = prompt.get('system_prompt', '')
            user_area.text = prompt.get('user_prompt', '')
            
        except Exception as e:
            logger.error(f"Error loading prompt details: {e}")
    
    def prepare_analysis_messages(self) -> Tuple[str, str]:
        """Prepare system and user prompts with media content."""
        try:
            system_area = self.query_one("#system-prompt-area", TextArea)
            user_area = self.query_one("#user-prompt-area", TextArea)
            
            system_prompt = system_area.text
            user_prompt = user_area.text
            
            # Replace placeholders with actual media content
            if self.media_data:
                # Truncate content if too long
                content = self.media_data.get('content', '')
                if len(content) > 10000:
                    content = content[:10000] + "\n\n[Content truncated...]"
                
                replacements = {
                    "{title}": self.media_data.get('title', 'Untitled'),
                    "{content}": content,
                    "{type}": self.media_data.get('type', ''),
                    "{author}": self.media_data.get('author', ''),
                    "{url}": self.media_data.get('url', ''),
                }
                
                for placeholder, value in replacements.items():
                    system_prompt = system_prompt.replace(placeholder, str(value))
                    user_prompt = user_prompt.replace(placeholder, str(value))
            
            return system_prompt, user_prompt
            
        except Exception as e:
            logger.error(f"Error preparing analysis messages: {e}")
            return "", ""
    
    def _update_analysis_button_states(self) -> None:
        """Update analysis button states based on current state."""
        try:
            save_btn = self.query_one("#save-analysis-btn", Button)
            edit_btn = self.query_one("#edit-analysis-btn", Button)
            overwrite_btn = self.query_one("#overwrite-analysis-btn", Button)
            
            if self.analysis_edit_mode:
                save_btn.disabled = True
                edit_btn.label = "Cancel Edit"
                overwrite_btn.disabled = False
            else:
                save_btn.disabled = not self.current_analysis or self.has_existing_analysis
                edit_btn.label = "Edit"
                edit_btn.disabled = not self.current_analysis
                overwrite_btn.disabled = not self.has_existing_analysis
                
        except Exception:
            pass
    
    # Analysis Event Handlers
    @on(Select.Changed, "#analysis-provider-select")
    def handle_provider_change(self, event: Select.Changed) -> None:
        """Handle provider selection change."""
        if event.value and event.value != Select.BLANK:
            self.update_models_for_provider(event.value)
    
    @on(Select.Changed, "#analysis-model-select")
    def handle_model_change(self, event: Select.Changed) -> None:
        """Handle model selection change."""
        # Just store the selection, no additional action needed
        pass
    
    @on(Input.Changed, "#prompt-search-input")
    def handle_prompt_search(self, event: Input.Changed) -> None:
        """Handle prompt search input with debouncing."""
        # Get keyword filter value
        try:
            keyword_input = self.query_one("#prompt-keyword-input", Input)
            keywords = keyword_input.value
        except:
            keywords = ""
        
        # Trigger search
        self.search_prompts(event.value, keywords)
    
    @on(Input.Changed, "#prompt-keyword-input")
    def handle_prompt_keyword_change(self, event: Input.Changed) -> None:
        """Handle prompt keyword filter change."""
        # Get search term
        try:
            search_input = self.query_one("#prompt-search-input", Input)
            search_term = search_input.value
        except:
            search_term = ""
        
        # Trigger search
        self.search_prompts(search_term, event.value)
    
    @on(Select.Changed, "#prompt-select")
    def handle_prompt_selection(self, event: Select.Changed) -> None:
        """Handle prompt selection."""
        if event.value and event.value != Select.BLANK:
            self.load_prompt_details(event.value)
    
    @on(Button.Pressed, "#generate-analysis-btn")
    def handle_generate_analysis(self) -> None:
        """Handle generate analysis button press."""
        if not self.media_data:
            self.app_instance.notify("No media item selected", severity="warning")
            return
        
        try:
            # Get selected provider and model
            provider_select = self.query_one("#analysis-provider-select", Select)
            model_select = self.query_one("#analysis-model-select", Select)
            
            provider = provider_select.value if provider_select.value != Select.BLANK else None
            model = model_select.value if model_select.value != Select.BLANK else None
            
            if not provider or not model:
                self.app_instance.notify("Please select a provider and model", severity="warning")
                return
            
            # Get prompts from text areas
            system_prompt, user_prompt = self.prepare_analysis_messages()
            
            if not system_prompt and not user_prompt:
                self.app_instance.notify("Please provide at least one prompt", severity="warning")
                return
            
            # Get temperature, top_p, min_p, max_tokens values
            try:
                temperature = float(self.query_one("#analysis-temperature", Input).value or "0.7")
            except ValueError:
                temperature = 0.7
            
            try:
                top_p = float(self.query_one("#analysis-top-p", Input).value or "0.95")
            except ValueError:
                top_p = 0.95
            
            try:
                min_p = float(self.query_one("#analysis-min-p", Input).value or "0.05")
            except ValueError:
                min_p = 0.05
            
            try:
                max_tokens = int(self.query_one("#analysis-max-tokens", Input).value or "4096")
            except ValueError:
                max_tokens = 4096
            
            # Post analysis request event
            from ...Event_Handlers.media_events import MediaAnalysisRequestEvent
            self.post_message(MediaAnalysisRequestEvent(
                media_id=self.media_data['id'],
                provider=provider.lower(),  # Normalize provider name
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                max_tokens=max_tokens,
                type_slug=""  # Will be set by MediaWindow
            ))
            
            # Show loading state
            analysis_display = self.query_one("#analysis-display", Markdown)
            analysis_display.update("*Generating analysis...*")
            
            # Store current analysis as None while generating
            self.current_analysis = None
            self._update_analysis_button_states()
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            self.app_instance.notify(f"Error: {str(e)[:100]}", severity="error")
    
    @on(Button.Pressed, "#save-analysis-btn")
    def handle_save_analysis(self) -> None:
        """Handle save analysis button press."""
        if not self.current_analysis or not self.media_data:
            return
        
        from ...Event_Handlers.media_events import MediaAnalysisSaveEvent
        self.post_message(MediaAnalysisSaveEvent(
            media_id=self.media_data['id'],
            analysis_content=self.current_analysis,
            type_slug=""  # Will be set by MediaWindow
        ))
    
    @on(Button.Pressed, "#edit-analysis-btn")
    def handle_edit_analysis(self) -> None:
        """Handle edit analysis button press."""
        if not self.current_analysis:
            return
        
        self.analysis_edit_mode = not self.analysis_edit_mode
        
        if self.analysis_edit_mode:
            # Switch to edit mode - convert markdown to textarea
            try:
                analysis_display = self.query_one("#analysis-display", Markdown)
                # Store current markdown content
                current_content = self.current_analysis
                
                # Replace markdown with textarea
                parent = analysis_display.parent
                analysis_display.remove()
                
                edit_area = TextArea(current_content, id="analysis-edit-area")
                parent.mount(edit_area, before=0)
                
            except Exception as e:
                logger.error(f"Error entering edit mode: {e}")
        else:
            # Exit edit mode - convert textarea back to markdown
            try:
                edit_area = self.query_one("#analysis-edit-area", TextArea)
                # Get edited content
                self.current_analysis = edit_area.text
                
                # Replace textarea with markdown
                parent = edit_area.parent
                edit_area.remove()
                
                analysis_display = Markdown(self.current_analysis, id="analysis-display")
                parent.mount(analysis_display, before=0)
                
            except Exception as e:
                logger.error(f"Error exiting edit mode: {e}")
        
        self._update_analysis_button_states()
    
    @on(Button.Pressed, "#overwrite-analysis-btn")
    def handle_overwrite_analysis(self) -> None:
        """Handle overwrite analysis button press."""
        if not self.media_data:
            return
        
        # Get current content from edit area if in edit mode
        if self.analysis_edit_mode:
            try:
                edit_area = self.query_one("#analysis-edit-area", TextArea)
                analysis_content = edit_area.text
            except:
                analysis_content = self.current_analysis
        else:
            analysis_content = self.current_analysis
        
        if not analysis_content:
            return
        
        from ...Event_Handlers.media_events import MediaAnalysisOverwriteEvent
        self.post_message(MediaAnalysisOverwriteEvent(
            media_id=self.media_data['id'],
            analysis_content=analysis_content,
            type_slug=""  # Will be set by MediaWindow
        ))
        
        # Exit edit mode if active
        if self.analysis_edit_mode:
            self.handle_edit_analysis()