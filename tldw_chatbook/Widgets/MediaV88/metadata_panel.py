"""
Metadata Panel for Media UI V88.

Displays media metadata in a 4-row layout with edit and delete functionality.
"""

from typing import TYPE_CHECKING, Dict, Any, Optional
from datetime import datetime
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Grid
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, TextArea, Static
from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from ...app import TldwCli


class MediaMetadataUpdateEvent(Message):
    """Event fired when metadata is updated."""
    
    def __init__(self, media_id: int, changes: Dict[str, Any]) -> None:
        super().__init__()
        self.media_id = media_id
        self.changes = changes


class MediaDeleteRequestEvent(Message):
    """Event fired when deletion is requested."""
    
    def __init__(self, media_id: int, media_title: str) -> None:
        super().__init__()
        self.media_id = media_id
        self.media_title = media_title


class MetadataPanel(Container):
    """
    Metadata display panel with 4-row layout.
    
    Layout:
    Row 1: Title, Type, Date Created
    Row 2: Author, URL/Source, Date Modified
    Row 3: Keywords/Tags (scrollable horizontal)
    Row 4: Description/Summary
    Bottom: Edit and Delete buttons
    """
    
    DEFAULT_CSS = """
    MetadataPanel {
        height: auto;
        max-height: 40%;
        layout: vertical;
        background: $boost;
        border: solid $primary-lighten-2;
        padding: 1;
        margin-bottom: 1;
    }
    
    .metadata-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .metadata-grid {
        layout: grid;
        grid-size: 3;
        grid-rows: auto;
        grid-gutter: 1;
        margin-bottom: 1;
    }
    
    .metadata-field {
        layout: vertical;
        height: auto;
    }
    
    .field-label {
        text-style: bold;
        color: $text-muted;
        margin-bottom: 0;
    }
    
    .field-value {
        color: $text;
        min-height: 2;
        padding: 0 1;
        background: $surface;
        border: solid $primary-lighten-3;
    }
    
    .field-value.editable {
        background: $primary-background;
    }
    
    .keywords-container {
        column-span: 3;
        layout: vertical;
    }
    
    .keywords-list {
        layout: horizontal;
        height: 3;
        overflow-x: auto;
        overflow-y: hidden;
        background: $surface;
        border: solid $primary-lighten-3;
        padding: 0 1;
    }
    
    .keyword-tag {
        margin: 0 1 0 0;
        padding: 0 1;
        background: $accent;
        border: round $accent-lighten-1;
        height: 2;
        content-align: center middle;
    }
    
    .description-container {
        column-span: 3;
        layout: vertical;
    }
    
    .description-value {
        min-height: 4;
        max-height: 8;
        overflow-y: auto;
        padding: 1;
        background: $surface;
        border: solid $primary-lighten-3;
    }
    
    .action-buttons {
        layout: horizontal;
        height: 3;
        margin-top: 1;
        align-horizontal: center;
    }
    
    .action-buttons Button {
        width: auto;
        min-width: 12;
        margin: 0 2;
    }
    
    #edit-button {
        background: $primary;
    }
    
    #save-button {
        background: $success;
    }
    
    #cancel-button {
        background: $warning;
    }
    
    #delete-button {
        background: $error;
    }
    
    .empty-state {
        text-align: center;
        color: $text-muted;
        padding: 4;
    }
    
    /* Edit mode styles */
    .edit-input {
        width: 100%;
        height: 3;
    }
    
    .edit-textarea {
        width: 100%;
        min-height: 4;
        max-height: 8;
    }
    """
    
    # Reactive properties
    edit_mode: reactive[bool] = reactive(False)
    current_media: reactive[Optional[Dict[str, Any]]] = reactive(None)
    has_unsaved_changes: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the metadata panel."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.original_values: Dict[str, Any] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the metadata panel UI."""
        yield Label("Media Metadata", classes="metadata-title")
        
        # Main metadata grid
        with Grid(classes="metadata-grid", id="metadata-grid"):
            # Row 1
            with Container(classes="metadata-field"):
                yield Label("Title", classes="field-label")
                yield Static("", id="title-value", classes="field-value")
            
            with Container(classes="metadata-field"):
                yield Label("Type", classes="field-label")
                yield Static("", id="type-value", classes="field-value")
            
            with Container(classes="metadata-field"):
                yield Label("Created", classes="field-label")
                yield Static("", id="created-value", classes="field-value")
            
            # Row 2
            with Container(classes="metadata-field"):
                yield Label("Author", classes="field-label")
                yield Static("", id="author-value", classes="field-value")
            
            with Container(classes="metadata-field"):
                yield Label("URL/Source", classes="field-label")
                yield Static("", id="url-value", classes="field-value")
            
            with Container(classes="metadata-field"):
                yield Label("Modified", classes="field-label")
                yield Static("", id="modified-value", classes="field-value")
            
            # Row 3 - Keywords (spans all columns)
            with Container(classes="keywords-container"):
                yield Label("Keywords/Tags", classes="field-label")
                yield Horizontal(id="keywords-list", classes="keywords-list")
            
            # Row 4 - Description (spans all columns)
            with Container(classes="description-container"):
                yield Label("Description/Summary", classes="field-label")
                yield Static("", id="description-value", classes="description-value")
        
        # Action buttons
        with Horizontal(classes="action-buttons", id="action-buttons"):
            yield Button("Edit", id="edit-button", variant="primary")
            yield Button("Delete", id="delete-button", variant="error")
        
        # Edit mode buttons (initially hidden)
        with Horizontal(classes="action-buttons", id="edit-buttons"):
            yield Button("Save", id="save-button", variant="success")
            yield Button("Cancel", id="cancel-button", variant="warning")
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        logger.info("MetadataPanel mounted")
        
        # Hide edit buttons initially
        self._toggle_edit_buttons(False)
        
        # Show empty state
        self.clear_display()
    
    def watch_edit_mode(self, edit_mode: bool) -> None:
        """React to edit mode changes."""
        if edit_mode:
            self._enter_edit_mode()
        else:
            self._exit_edit_mode()
    
    @on(Button.Pressed, "#edit-button")
    def handle_edit(self, event: Button.Pressed) -> None:
        """Enter edit mode."""
        if self.current_media:
            self.edit_mode = True
    
    @on(Button.Pressed, "#save-button")
    def handle_save(self, event: Button.Pressed) -> None:
        """Save metadata changes."""
        if self.current_media:
            self._save_changes()
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self, event: Button.Pressed) -> None:
        """Cancel edit mode without saving."""
        self.edit_mode = False
    
    @on(Button.Pressed, "#delete-button")
    def handle_delete(self, event: Button.Pressed) -> None:
        """Request media deletion."""
        if self.current_media:
            media_id = self.current_media.get('id')
            media_title = self.current_media.get('title', 'Untitled')
            
            if media_id:
                # Post delete request event
                self.post_message(MediaDeleteRequestEvent(media_id, media_title))
    
    def load_media(self, media_data: Dict[str, Any]) -> None:
        """Load media data into the panel."""
        logger.info(f"Loading media: {media_data.get('id')} - {media_data.get('title', 'Untitled')}")
        
        self.current_media = media_data
        self.has_unsaved_changes = False
        
        # Exit edit mode if active
        if self.edit_mode:
            self.edit_mode = False
        
        # Update display fields
        self._update_display()
    
    def _update_display(self) -> None:
        """Update the display with current media data."""
        if not self.current_media:
            return
        
        try:
            # Update text fields
            fields = {
                'title-value': self.current_media.get('title', 'Untitled'),
                'type-value': self.current_media.get('type', 'Unknown'),
                'author-value': self.current_media.get('author', 'Unknown'),
                'url-value': self.current_media.get('url', 'N/A'),
                'description-value': self.current_media.get('description', 
                                     self.current_media.get('summary', 'No description available')),
            }
            
            for field_id, value in fields.items():
                try:
                    field = self.query_one(f"#{field_id}", Static)
                    field.update(str(value) if value else "N/A")
                except Exception as e:
                    logger.debug(f"Could not update field {field_id}: {e}")
            
            # Format dates
            created_date = self._format_date(self.current_media.get('created_at'))
            modified_date = self._format_date(self.current_media.get('last_modified'))
            
            try:
                created_field = self.query_one("#created-value", Static)
                created_field.update(created_date)
                
                modified_field = self.query_one("#modified-value", Static)
                modified_field.update(modified_date)
            except Exception as e:
                logger.debug(f"Could not update date fields: {e}")
            
            # Update keywords
            self._update_keywords_display()
            
        except Exception as e:
            logger.error(f"Error updating display: {e}", exc_info=True)
    
    def _update_keywords_display(self) -> None:
        """Update the keywords/tags display."""
        try:
            keywords_list = self.query_one("#keywords-list", Horizontal)
            keywords_list.remove_children()
            
            # Get keywords from media data
            keywords = self.current_media.get('keywords', [])
            if isinstance(keywords, str):
                # Split comma-separated string
                keywords = [k.strip() for k in keywords.split(',') if k.strip()]
            
            if keywords:
                for keyword in keywords:
                    tag = Static(keyword, classes="keyword-tag")
                    keywords_list.mount(tag)
            else:
                # Show placeholder
                placeholder = Static("No keywords", classes="keyword-tag")
                placeholder.styles.opacity = "0.5"
                keywords_list.mount(placeholder)
                
        except Exception as e:
            logger.debug(f"Could not update keywords: {e}")
    
    def _format_date(self, date_value: Any) -> str:
        """Format a date value for display."""
        if not date_value:
            return "N/A"
        
        try:
            if isinstance(date_value, str):
                # Parse ISO format
                dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            elif isinstance(date_value, (int, float)):
                # Unix timestamp
                dt = datetime.fromtimestamp(date_value)
            elif isinstance(date_value, datetime):
                dt = date_value
            else:
                return str(date_value)
            
            # Format as readable date
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(date_value)
    
    def _enter_edit_mode(self) -> None:
        """Enter edit mode for metadata."""
        if not self.current_media:
            return
        
        logger.info("Entering edit mode")
        
        # Store original values
        self.original_values = {
            'title': self.current_media.get('title', ''),
            'author': self.current_media.get('author', ''),
            'url': self.current_media.get('url', ''),
            'description': self.current_media.get('description', 
                          self.current_media.get('summary', '')),
            'keywords': self.current_media.get('keywords', ''),
        }
        
        # Replace static fields with inputs
        try:
            # Title
            title_container = self.query_one("#title-value").parent
            title_static = self.query_one("#title-value")
            title_static.remove()
            title_input = Input(
                value=self.original_values['title'],
                id="title-input",
                classes="edit-input"
            )
            title_container.mount(title_input)
            
            # Author
            author_container = self.query_one("#author-value").parent
            author_static = self.query_one("#author-value")
            author_static.remove()
            author_input = Input(
                value=self.original_values['author'],
                id="author-input",
                classes="edit-input"
            )
            author_container.mount(author_input)
            
            # URL
            url_container = self.query_one("#url-value").parent
            url_static = self.query_one("#url-value")
            url_static.remove()
            url_input = Input(
                value=self.original_values['url'],
                id="url-input",
                classes="edit-input"
            )
            url_container.mount(url_input)
            
            # Description
            desc_container = self.query_one("#description-value").parent
            desc_static = self.query_one("#description-value")
            desc_static.remove()
            desc_textarea = TextArea(
                self.original_values['description'],
                id="description-input",
                classes="edit-textarea"
            )
            desc_container.mount(desc_textarea)
            
            # Keywords
            keywords_container = self.query_one("#keywords-list").parent
            keywords_list = self.query_one("#keywords-list")
            keywords_list.remove()
            
            # Convert keywords list to comma-separated string
            keywords_str = self.original_values['keywords']
            if isinstance(keywords_str, list):
                keywords_str = ', '.join(keywords_str)
            
            keywords_input = Input(
                value=keywords_str,
                placeholder="Enter keywords separated by commas",
                id="keywords-input",
                classes="edit-input"
            )
            keywords_container.mount(keywords_input)
            
        except Exception as e:
            logger.error(f"Error entering edit mode: {e}", exc_info=True)
        
        # Toggle buttons
        self._toggle_edit_buttons(True)
    
    def _exit_edit_mode(self) -> None:
        """Exit edit mode and restore display."""
        logger.info("Exiting edit mode")
        
        # Rebuild the display
        try:
            # We need to restore the original structure
            # For simplicity, we'll remount the entire grid
            grid = self.query_one("#metadata-grid")
            grid.remove_children()
            
            # Recreate the original structure
            self._recreate_metadata_grid(grid)
            
            # Update with current data
            self._update_display()
            
        except Exception as e:
            logger.error(f"Error exiting edit mode: {e}", exc_info=True)
        
        # Toggle buttons
        self._toggle_edit_buttons(False)
        self.has_unsaved_changes = False
    
    def _recreate_metadata_grid(self, grid: Grid) -> None:
        """Recreate the metadata grid structure."""
        # Row 1
        field1 = Container(classes="metadata-field")
        field1.mount(Label("Title", classes="field-label"))
        field1.mount(Static("", id="title-value", classes="field-value"))
        grid.mount(field1)
        
        field2 = Container(classes="metadata-field")
        field2.mount(Label("Type", classes="field-label"))
        field2.mount(Static("", id="type-value", classes="field-value"))
        grid.mount(field2)
        
        field3 = Container(classes="metadata-field")
        field3.mount(Label("Created", classes="field-label"))
        field3.mount(Static("", id="created-value", classes="field-value"))
        grid.mount(field3)
        
        # Row 2
        field4 = Container(classes="metadata-field")
        field4.mount(Label("Author", classes="field-label"))
        field4.mount(Static("", id="author-value", classes="field-value"))
        grid.mount(field4)
        
        field5 = Container(classes="metadata-field")
        field5.mount(Label("URL/Source", classes="field-label"))
        field5.mount(Static("", id="url-value", classes="field-value"))
        grid.mount(field5)
        
        field6 = Container(classes="metadata-field")
        field6.mount(Label("Modified", classes="field-label"))
        field6.mount(Static("", id="modified-value", classes="field-value"))
        grid.mount(field6)
        
        # Row 3 - Keywords
        keywords_container = Container(classes="keywords-container")
        keywords_container.mount(Label("Keywords/Tags", classes="field-label"))
        keywords_container.mount(Horizontal(id="keywords-list", classes="keywords-list"))
        grid.mount(keywords_container)
        
        # Row 4 - Description
        desc_container = Container(classes="description-container")
        desc_container.mount(Label("Description/Summary", classes="field-label"))
        desc_container.mount(Static("", id="description-value", classes="description-value"))
        grid.mount(desc_container)
    
    def _save_changes(self) -> None:
        """Save the edited metadata."""
        if not self.current_media:
            return
        
        try:
            # Collect changed values
            changes = {}
            
            # Get input values
            title = self.query_one("#title-input", Input).value
            author = self.query_one("#author-input", Input).value
            url = self.query_one("#url-input", Input).value
            description = self.query_one("#description-input", TextArea).text
            keywords = self.query_one("#keywords-input", Input).value
            
            # Check for changes
            if title != self.original_values['title']:
                changes['title'] = title
            if author != self.original_values['author']:
                changes['author'] = author
            if url != self.original_values['url']:
                changes['url'] = url
            if description != self.original_values['description']:
                changes['description'] = description
            if keywords != self.original_values['keywords']:
                # Convert to list
                keywords_list = [k.strip() for k in keywords.split(',') if k.strip()]
                changes['keywords'] = keywords_list
            
            if changes:
                # Update current media
                self.current_media.update(changes)
                
                # Post update event
                media_id = self.current_media.get('id')
                if media_id:
                    self.post_message(MediaMetadataUpdateEvent(media_id, changes))
                    self.app_instance.notify("Metadata updated", severity="information")
            else:
                self.app_instance.notify("No changes to save", severity="information")
            
            # Exit edit mode
            self.edit_mode = False
            
        except Exception as e:
            logger.error(f"Error saving changes: {e}", exc_info=True)
            self.app_instance.notify(f"Save failed: {str(e)[:100]}", severity="error")
    
    def _toggle_edit_buttons(self, show_edit: bool) -> None:
        """Toggle between normal and edit mode buttons."""
        try:
            action_buttons = self.query_one("#action-buttons", Horizontal)
            edit_buttons = self.query_one("#edit-buttons", Horizontal)
            
            if show_edit:
                action_buttons.styles.display = "none"
                edit_buttons.styles.display = "block"
            else:
                action_buttons.styles.display = "block"
                edit_buttons.styles.display = "none"
                
        except Exception as e:
            logger.debug(f"Could not toggle buttons: {e}")
    
    def clear_display(self) -> None:
        """Clear the metadata display."""
        self.current_media = None
        self.has_unsaved_changes = False
        
        if self.edit_mode:
            self.edit_mode = False
        
        # Show empty state
        try:
            grid = self.query_one("#metadata-grid", Grid)
            grid.styles.display = "none"
            
            # Could add an empty state message here
            
        except Exception as e:
            logger.debug(f"Could not clear display: {e}")