# tldw_chatbook/Widgets/media_details_widget.py
"""
Widget for displaying and editing media item metadata with content display.
"""

from typing import TYPE_CHECKING, Dict, Any, Optional, List
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, Label, Input, TextArea
from textual.reactive import reactive
from textual.screen import ModalScreen
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli


class MediaDetailsWidget(Container):
    """
    A widget that displays media metadata and content with inline editing capabilities.
    Toggles between view mode (showing metadata as static text) and edit mode 
    (showing input fields for metadata editing).
    """
    
    # Reactive states
    edit_mode = reactive(False)
    media_data: reactive[Optional[Dict[str, Any]]] = reactive(None)
    
    def __init__(self, app_instance: 'TldwCli', type_slug: str, **kwargs):
        """
        Initialize the MediaDetailsWidget.
        
        Args:
            app_instance: Reference to the main app instance
            type_slug: The type slug for this media view (e.g., "all-media", "video")
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.type_slug = type_slug
        self._original_data = None  # Store original data for cancel functionality
        
    def compose(self) -> ComposeResult:
        """Compose the widget's UI structure."""
        # Don't wrap in VerticalScroll - the container itself should handle layout
        # Metadata section
        with Container(id=f"metadata-section-{self.type_slug}", classes="metadata-section"):
            # View mode elements
            with Container(id=f"metadata-view-{self.type_slug}", classes="metadata-view"):
                yield Static("", id=f"metadata-display-{self.type_slug}", classes="metadata-display")
                with Horizontal(classes="metadata-buttons"):
                    yield Button("Edit", id=f"edit-button-{self.type_slug}", classes="metadata-edit-button")
                    yield Button("Delete", id=f"delete-button-{self.type_slug}", classes="metadata-delete-button", variant="error")
            
            # Edit mode elements (initially hidden)
            with Container(id=f"metadata-edit-{self.type_slug}", classes="metadata-edit hidden"):
                # Input fields for editing
                with Vertical(classes="edit-fields"):
                    yield Label("Title:")
                    yield Input(id=f"edit-title-{self.type_slug}", placeholder="Enter title")
                    
                    yield Label("Type:")
                    yield Input(id=f"edit-type-{self.type_slug}", placeholder="Enter type", disabled=True)
                    
                    yield Label("Author:")
                    yield Input(id=f"edit-author-{self.type_slug}", placeholder="Enter author")
                    
                    yield Label("URL:")
                    yield Input(id=f"edit-url-{self.type_slug}", placeholder="Enter URL")
                    
                    yield Label("Keywords (comma-separated):")
                    yield Input(id=f"edit-keywords-{self.type_slug}", placeholder="keyword1, keyword2, keyword3")
                
                # Action buttons
                with Horizontal(classes="edit-actions"):
                    yield Button("Save", id=f"save-button-{self.type_slug}", variant="primary")
                    yield Button("Cancel", id=f"cancel-button-{self.type_slug}", variant="default")
        
        # Content section (read-only)
        yield TextArea(
            "Select an item from the list to see its details.",
            id=f"content-display-{self.type_slug}",
            classes="media-content-display",
            read_only=True
        )
    
    def watch_edit_mode(self, old_value: bool, new_value: bool) -> None:
        """React to edit mode changes."""
        try:
            view_container = self.query_one(f"#metadata-view-{self.type_slug}")
            edit_container = self.query_one(f"#metadata-edit-{self.type_slug}")
            
            if new_value:  # Entering edit mode
                view_container.add_class("hidden")
                edit_container.remove_class("hidden")
                self._populate_edit_fields()
            else:  # Exiting edit mode
                view_container.remove_class("hidden")
                edit_container.add_class("hidden")
                
        except Exception as e:
            logger.error(f"Error toggling edit mode: {e}")
    
    def watch_media_data(self, old_data: Optional[Dict], new_data: Optional[Dict]) -> None:
        """React to media data changes by updating the display."""
        if new_data:
            self._update_metadata_display()
            self._update_content_display()
            self._update_delete_button()
        else:
            self._clear_displays()
    
    def _update_metadata_display(self) -> None:
        """Update the metadata display with current media data."""
        if not self.media_data:
            return
            
        try:
            display = self.query_one(f"#metadata-display-{self.type_slug}", Static)
            
            # Get keywords
            keywords_str = "N/A"
            media_id = self.media_data.get('id')
            if self.app_instance.media_db and media_id:
                try:
                    from ..DB.Client_Media_DB_v2 import fetch_keywords_for_media
                    keywords = fetch_keywords_for_media(self.app_instance.media_db, media_id)
                    keywords_str = ", ".join(keywords) if keywords else "N/A"
                except Exception as e:
                    keywords_str = f"Error: {e}"
            
            # Check if item is deleted
            is_deleted = self.media_data.get('deleted', 0) == 1
            status_text = "  [red][DELETED][/red]" if is_deleted else ""
            
            # Format metadata for display
            metadata_text = (
                f"[bold]ID:[/bold] {self.media_data.get('id', 'N/A')}  "
                f"[bold]UUID:[/bold] {self.media_data.get('uuid', 'N/A')}{status_text}\n"
                f"[bold]Type:[/bold] {self.media_data.get('type', 'N/A')}  "
                f"[bold]Author:[/bold] {self.media_data.get('author', 'N/A')}\n"
                f"[bold]URL:[/bold] {self.media_data.get('url', 'N/A')}\n"
                f"[bold]Keywords:[/bold] {keywords_str}"
            )
            
            display.update(metadata_text)
            
        except Exception as e:
            logger.error(f"Error updating metadata display: {e}")
    
    def _update_content_display(self) -> None:
        """Update the content display with media content."""
        if not self.media_data:
            return
            
        try:
            content_area = self.query_one(f"#content-display-{self.type_slug}", TextArea)
            
            title = self.media_data.get('title', 'Untitled')
            content = self.media_data.get('content', 'No content available')
            
            # Format content with title
            formatted_content = f"# {title}\n\n{content}"
            
            # Add analysis content if viewing analysis-review
            if self.type_slug == "analysis-review":
                analysis = self.media_data.get('analysis_content', '')
                if analysis:
                    formatted_content += f"\n\n## Analysis\n\n{analysis}"
            
            content_area.text = formatted_content
            content_area.scroll_home(animate=False)
            
        except Exception as e:
            logger.error(f"Error updating content display: {e}")
    
    def _clear_displays(self) -> None:
        """Clear all displays when no media is selected."""
        try:
            metadata_display = self.query_one(f"#metadata-display-{self.type_slug}", Static)
            metadata_display.update("No media item selected")
            
            content_area = self.query_one(f"#content-display-{self.type_slug}", TextArea)
            content_area.text = "Select an item from the list to see its details."
            
        except Exception as e:
            logger.error(f"Error clearing displays: {e}")
    
    def _populate_edit_fields(self) -> None:
        """Populate edit fields with current media data."""
        if not self.media_data:
            return
            
        # Store original data for cancel functionality
        self._original_data = self.media_data.copy()
        
        try:
            # Populate each field
            title_input = self.query_one(f"#edit-title-{self.type_slug}", Input)
            title_input.value = self.media_data.get('title', '')
            
            type_input = self.query_one(f"#edit-type-{self.type_slug}", Input)
            type_input.value = self.media_data.get('type', '')
            
            author_input = self.query_one(f"#edit-author-{self.type_slug}", Input)
            author_input.value = self.media_data.get('author', '')
            
            url_input = self.query_one(f"#edit-url-{self.type_slug}", Input)
            url_input.value = self.media_data.get('url', '')
            
            # Get keywords for this media item
            keywords_input = self.query_one(f"#edit-keywords-{self.type_slug}", Input)
            media_id = self.media_data.get('id')
            if self.app_instance.media_db and media_id:
                try:
                    from ..DB.Client_Media_DB_v2 import fetch_keywords_for_media
                    keywords = fetch_keywords_for_media(self.app_instance.media_db, media_id)
                    keywords_input.value = ", ".join(keywords) if keywords else ""
                except Exception:
                    keywords_input.value = ""
            else:
                keywords_input.value = ""
                
        except Exception as e:
            logger.error(f"Error populating edit fields: {e}")
    
    @on(Button.Pressed, ".metadata-edit-button")
    def handle_edit_button(self, event: Button.Pressed) -> None:
        """Handle edit button press."""
        if event.button.id == f"edit-button-{self.type_slug}":
            self.edit_mode = True
    
    @on(Button.Pressed)
    def handle_save_button(self, event: Button.Pressed) -> None:
        """Handle save button press."""
        if event.button.id == f"save-button-{self.type_slug}":
            self._save_metadata()
    
    @on(Button.Pressed)
    def handle_cancel_button(self, event: Button.Pressed) -> None:
        """Handle cancel button press."""
        if event.button.id == f"cancel-button-{self.type_slug}":
            self.edit_mode = False
            # Restore original data
            if self._original_data:
                self.media_data = self._original_data
    
    def _save_metadata(self) -> None:
        """Save the edited metadata."""
        if not self.media_data:
            return
            
        try:
            # Collect values from input fields
            title = self.query_one(f"#edit-title-{self.type_slug}", Input).value.strip()
            # Type is disabled, so we use the original value
            media_type = self.media_data.get('type', '')
            author = self.query_one(f"#edit-author-{self.type_slug}", Input).value.strip()
            url = self.query_one(f"#edit-url-{self.type_slug}", Input).value.strip()
            keywords_str = self.query_one(f"#edit-keywords-{self.type_slug}", Input).value.strip()
            
            # Parse keywords
            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()] if keywords_str else []
            
            # Validate required fields
            if not title:
                self.app_instance.notify("Title cannot be empty", severity="warning")
                return
            
            # Post message to trigger the update event handler
            from ..Event_Handlers.media_events import MediaMetadataUpdateEvent
            self.post_message(MediaMetadataUpdateEvent(
                media_id=self.media_data['id'],
                title=title,
                media_type=media_type,
                author=author,
                url=url,
                keywords=keywords,
                type_slug=self.type_slug
            ))
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            self.app_instance.notify(f"Error saving metadata: {str(e)}", severity="error")
    
    def update_media_data(self, media_data: Dict[str, Any]) -> None:
        """Update the displayed media data."""
        self.media_data = media_data
        self.edit_mode = False  # Exit edit mode when new data is loaded
    
    def _update_delete_button(self) -> None:
        """Update the delete button text based on whether the item is deleted."""
        if not self.media_data:
            return
            
        try:
            delete_button = self.query_one(f"#delete-button-{self.type_slug}", Button)
            is_deleted = self.media_data.get('deleted', 0) == 1
            
            if is_deleted:
                delete_button.label = "Undelete"
                delete_button.variant = "success"
            else:
                delete_button.label = "Delete"
                delete_button.variant = "error"
                
        except Exception as e:
            logger.error(f"Error updating delete button: {e}")
    
    @on(Button.Pressed)
    def handle_delete_button(self, event: Button.Pressed) -> None:
        """Handle delete/undelete button press."""
        if event.button.id == f"delete-button-{self.type_slug}":
            if not self.media_data:
                return
                
            is_deleted = self.media_data.get('deleted', 0) == 1
            
            if is_deleted:
                # Direct undelete without confirmation
                self._perform_undelete()
            else:
                # Show confirmation dialog for deletion
                self._show_delete_confirmation()
    
    def _show_delete_confirmation(self) -> None:
        """Show confirmation dialog before deletion."""
        from ..Event_Handlers.media_events import MediaDeleteConfirmationEvent
        self.post_message(MediaDeleteConfirmationEvent(
            media_id=self.media_data['id'],
            media_title=self.media_data.get('title', 'Untitled'),
            type_slug=self.type_slug
        ))
    
    def _perform_undelete(self) -> None:
        """Perform undelete operation."""
        from ..Event_Handlers.media_events import MediaUndeleteEvent
        self.post_message(MediaUndeleteEvent(
            media_id=self.media_data['id'],
            type_slug=self.type_slug
        ))
    
