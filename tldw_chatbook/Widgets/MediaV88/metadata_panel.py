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
from textual.widgets import Button, Input, Label, TextArea, Static, Checkbox
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


class FormatForReadingChangeEvent(Message):
    """Event fired when format for reading checkbox is toggled."""
    
    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled


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
    
    MetadataPanel.collapsed #metadata-content {
        display: none;
    }
    
    .metadata-header {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    
    .metadata-title {
        text-style: bold;
        color: $primary;
        width: 1fr;
    }
    
    #collapse-button {
        width: auto;
        min-width: 3;
        height: 1;
        background: transparent;
        border: none;
        padding: 0 1;
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
    
    .keywords-header {
        layout: horizontal;
        height: auto;
        margin-bottom: 0;
    }
    
    .keywords-label {
        width: auto;
    }
    
    #format-checkbox {
        margin-left: 2;
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
    panel_collapsed: reactive[bool] = reactive(False)
    format_for_reading: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the metadata panel."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.original_values: Dict[str, Any] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the metadata panel UI."""
        # Header with title and collapse button
        with Horizontal(classes="metadata-header"):
            yield Label("Media Metadata", classes="metadata-title")
            yield Button("▼", id="collapse-button", variant="default")
        
        # Content container that can be collapsed
        with Container(id="metadata-content"):
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
                    with Horizontal(classes="keywords-header"):
                        yield Label("Keywords/Tags", classes="field-label keywords-label")
                        yield Checkbox("Format for Reading", id="format-checkbox", value=False)
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
        # Don't try to manipulate UI during mount
        if not self.is_mounted:
            return
            
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
    
    @on(Button.Pressed, "#collapse-button")
    def handle_collapse_toggle(self, event: Button.Pressed) -> None:
        """Toggle the panel collapse state."""
        self.panel_collapsed = not self.panel_collapsed
        
        # Update button text
        button = self.query_one("#collapse-button", Button)
        button.label = "▶" if self.panel_collapsed else "▼"
        
        # Toggle the collapsed class
        if self.panel_collapsed:
            self.add_class("collapsed")
        else:
            self.remove_class("collapsed")
    
    @on(Checkbox.Changed, "#format-checkbox")
    def handle_format_checkbox(self, event: Checkbox.Changed) -> None:
        """Handle format for reading checkbox change."""
        self.format_for_reading = event.value
        # Post event to notify content viewer
        self.post_message(FormatForReadingChangeEvent(event.value))
    
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
        
        # Replace static fields with inputs following Textual best practices
        try:
            # We'll hide statics and show inputs that are pre-mounted but hidden
            # First, ensure edit inputs exist (mount them if needed)
            grid = self.query_one("#metadata-grid", Grid)
            
            # Title field
            self._replace_with_input("#title-value", "title-input", self.original_values['title'])
            
            # Author field  
            self._replace_with_input("#author-value", "author-input", self.original_values['author'])
            
            # URL field
            self._replace_with_input("#url-value", "url-input", self.original_values['url'])
            
            # Description field - use TextArea
            self._replace_with_textarea("#description-value", "description-input", self.original_values['description'])
            
            # Keywords field
            keywords_str = self.original_values['keywords']
            if isinstance(keywords_str, list):
                keywords_str = ', '.join(keywords_str)
            self._replace_keywords_with_input(keywords_str)
            
        except Exception as e:
            logger.error(f"Error entering edit mode: {e}", exc_info=True)
            self.app_instance.notify("Failed to enter edit mode", severity="error")
            return
        
        # Toggle buttons
        self._toggle_edit_buttons(True)
    
    def _replace_with_input(self, static_id: str, input_id: str, value: str) -> None:
        """Replace a static widget with an input widget."""
        try:
            static_widget = self.query_one(static_id, Static)
            parent = static_widget.parent
            
            # Remove the static widget
            static_widget.remove()
            
            # Create and mount the input
            input_widget = Input(value=value, id=input_id, classes="edit-input")
            parent.mount(input_widget)
        except Exception as e:
            logger.error(f"Error replacing {static_id}: {e}")
    
    def _replace_with_textarea(self, static_id: str, textarea_id: str, value: str) -> None:
        """Replace a static widget with a textarea widget."""
        try:
            static_widget = self.query_one(static_id, Static)
            parent = static_widget.parent
            
            # Remove the static widget
            static_widget.remove()
            
            # Create and mount the textarea
            textarea_widget = TextArea(value, id=textarea_id, classes="edit-textarea")
            parent.mount(textarea_widget)
        except Exception as e:
            logger.error(f"Error replacing {static_id}: {e}")
    
    def _replace_keywords_with_input(self, keywords_str: str) -> None:
        """Replace keywords list with input field."""
        try:
            keywords_list = self.query_one("#keywords-list", Horizontal)
            parent = keywords_list.parent
            
            # Remove the keywords list
            keywords_list.remove()
            
            # Create and mount the input
            keywords_input = Input(
                value=keywords_str,
                placeholder="Enter keywords separated by commas",
                id="keywords-input",
                classes="edit-input"
            )
            parent.mount(keywords_input)
        except Exception as e:
            logger.error(f"Error replacing keywords: {e}")
    
    def _exit_edit_mode(self) -> None:
        """Exit edit mode and restore display."""
        logger.info("Exiting edit mode")
        
        try:
            # Replace inputs back with static widgets
            self._restore_static("#title-input", "title-value", self.current_media.get('title', ''))
            self._restore_static("#author-input", "author-value", self.current_media.get('author', ''))
            self._restore_static("#url-input", "url-value", self.current_media.get('url', ''))
            self._restore_static_textarea("#description-input", "description-value", 
                                        self.current_media.get('description', self.current_media.get('summary', '')))
            self._restore_keywords()
            
        except Exception as e:
            logger.error(f"Error exiting edit mode: {e}", exc_info=True)
            # If there's an error, try to reload the whole panel
            if self.current_media:
                self.load_media(self.current_media)
        
        # Toggle buttons
        self._toggle_edit_buttons(False)
        self.has_unsaved_changes = False
    
    def _restore_static(self, input_id: str, static_id: str, value: str) -> None:
        """Restore a static widget from an input widget."""
        try:
            input_widget = self.query_one(input_id, Input)
            parent = input_widget.parent
            
            # Remove the input widget
            input_widget.remove()
            
            # Create and mount the static
            static_widget = Static(value or "N/A", id=static_id, classes="field-value")
            parent.mount(static_widget)
        except Exception as e:
            logger.error(f"Error restoring {static_id}: {e}")
    
    def _restore_static_textarea(self, textarea_id: str, static_id: str, value: str) -> None:
        """Restore a static widget from a textarea widget."""
        try:
            textarea_widget = self.query_one(textarea_id, TextArea)
            parent = textarea_widget.parent
            
            # Remove the textarea widget
            textarea_widget.remove()
            
            # Create and mount the static
            static_widget = Static(value or "No description available", id=static_id, classes="description-value")
            parent.mount(static_widget)
        except Exception as e:
            logger.error(f"Error restoring {static_id}: {e}")
    
    def _restore_keywords(self) -> None:
        """Restore keywords list from input field."""
        try:
            keywords_input = self.query_one("#keywords-input", Input)
            parent = keywords_input.parent
            
            # Remove the input
            keywords_input.remove()
            
            # Create and mount the keywords list
            keywords_list = Horizontal(id="keywords-list", classes="keywords-list")
            parent.mount(keywords_list)
            
            # Populate with keywords
            keywords = self.current_media.get('keywords', [])
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(',') if k.strip()]
            
            if keywords:
                for keyword in keywords:
                    tag = Static(keyword, classes="keyword-tag")
                    keywords_list.mount(tag)
            else:
                placeholder = Static("No keywords", classes="keyword-tag")
                placeholder.styles.opacity = "0.5"
                keywords_list.mount(placeholder)
                
        except Exception as e:
            logger.error(f"Error restoring keywords: {e}")
    
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
            
            # Get input values safely
            try:
                title = self.query_one("#title-input", Input).value
            except:
                logger.error("Could not find title input")
                return
                
            try:
                author = self.query_one("#author-input", Input).value
            except:
                logger.error("Could not find author input")
                return
                
            try:
                url = self.query_one("#url-input", Input).value
            except:
                logger.error("Could not find url input")
                return
                
            try:
                description = self.query_one("#description-input", TextArea).text
            except:
                logger.error("Could not find description textarea")
                return
                
            try:
                keywords = self.query_one("#keywords-input", Input).value
            except:
                logger.error("Could not find keywords input")
                return
            
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
        
        # Don't trigger edit mode change during mount
        if self.edit_mode and self.is_mounted:
            self.edit_mode = False
        
        # Show empty state - but keep grid visible with empty values
        try:
            # Update fields to show "N/A" or empty state
            fields = {
                'title-value': 'Select a media item',
                'type-value': 'N/A',
                'author-value': 'N/A',
                'url-value': 'N/A',
                'created-value': 'N/A',
                'modified-value': 'N/A',
                'description-value': 'No media selected',
            }
            
            for field_id, value in fields.items():
                try:
                    field = self.query_one(f"#{field_id}", Static)
                    field.update(value)
                except Exception:
                    pass
            
            # Clear keywords
            try:
                keywords_list = self.query_one("#keywords-list", Horizontal)
                keywords_list.remove_children()
            except Exception:
                pass
            
        except Exception as e:
            logger.debug(f"Could not clear display: {e}")