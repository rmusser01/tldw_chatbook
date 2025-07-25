# tldw_chatbook/Widgets/media_details_widget.py
"""
Widget for displaying and editing media item metadata with content display.
"""

from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, Label, Input, TextArea, Checkbox, Select, Collapsible
from textual.reactive import reactive
from textual.screen import ModalScreen
from loguru import logger
import json

from ..Widgets.form_components import create_form_field
from ..Chunking.chunking_interop_library import (
    ChunkingInteropService, 
    get_chunking_service,
    ChunkingTemplateError
)

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
    format_for_reading = reactive(False)
    
    # Search-related state
    search_matches: reactive[List[Tuple[int, int]]] = reactive([])
    current_match_index: reactive[int] = reactive(-1)
    
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
        self.chunking_service = None
        
    def on_mount(self) -> None:
        """Set default values after the widget is mounted."""
        try:
            # Set default chunking method value
            chunk_method_select = self.query_one(f"#chunk-method-{self.type_slug}", Select)
            if chunk_method_select and not chunk_method_select.value:
                chunk_method_select.value = "words"
        except Exception as e:
            logger.debug(f"Could not set default chunk method: {e}")
        
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
        
        # Chunking configuration section
        with Collapsible(title="Chunking Configuration", id=f"chunking-config-section-{self.type_slug}", classes="chunking-config-section", collapsed=True):
            # Current config display
            yield Static("Current: Default configuration", id=f"chunking-config-display-{self.type_slug}", classes="chunking-config-display")
            
            # Template selector
            yield Label("Template:", classes="form-label")
            yield Select(
                [("Default", "default"), ("Custom Configuration", "custom")],
                id=f"chunking-template-select-{self.type_slug}",
                classes="chunking-template-selector"
            )
            
            # Advanced settings
            with Collapsible(title="Advanced Settings", collapsed=True, id=f"chunking-advanced-{self.type_slug}"):
                with Vertical(classes="chunking-advanced-settings"):
                    yield from create_form_field(
                        "Chunk Size (words)", 
                        f"chunk-size-{self.type_slug}", 
                        "input", 
                        placeholder="400",
                        default_value="400"
                    )
                    
                    yield from create_form_field(
                        "Overlap (words)", 
                        f"chunk-overlap-{self.type_slug}", 
                        "input", 
                        placeholder="100",
                        default_value="100"
                    )
                    
                    yield from create_form_field(
                        "Chunking Method", 
                        f"chunk-method-{self.type_slug}", 
                        "select",
                        options=[
                            ("words", "Words"),
                            ("sentences", "Sentences"),
                            ("paragraphs", "Paragraphs"),
                            ("hierarchical", "Hierarchical"),
                            ("structural", "Structural"),
                            ("contextual", "Contextual")
                        ],
                        default_value="words"
                    )
                    
                    yield from create_form_field(
                        "Enable Late Chunking",
                        f"enable-late-chunking-{self.type_slug}",
                        "checkbox",
                        default_value=False
                    )
            
            # Action buttons
            with Horizontal(classes="chunking-actions"):
                yield Button("Save Config", id=f"save-chunking-{self.type_slug}", variant="primary")
                yield Button("Preview Chunks", id=f"preview-chunks-{self.type_slug}", variant="default")
                yield Button("Reset to Default", id=f"reset-chunking-{self.type_slug}", variant="warning")
        
        # Formatting options
        with Container(classes="formatting-options"):
            yield Checkbox(
                "Format for easier reading",
                id=f"format-reading-checkbox-{self.type_slug}",
                classes="format-reading-checkbox",
                value=False
            )
        
        # Content search section
        with Container(id=f"content-search-section-{self.type_slug}", classes="content-search-section"):
            with Horizontal(classes="search-controls"):
                yield Input(
                    id=f"content-search-input-{self.type_slug}",
                    placeholder="Search within content...",
                    classes="content-search-input"
                )
                yield Button("🔍", id=f"content-search-button-{self.type_slug}", classes="content-search-button")
                yield Button("⬆", id=f"content-search-prev-{self.type_slug}", classes="content-search-nav", disabled=True)
                yield Button("⬇", id=f"content-search-next-{self.type_slug}", classes="content-search-nav", disabled=True)
                yield Static("", id=f"content-search-status-{self.type_slug}", classes="content-search-status")
        
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
        # Initialize chunking service if needed
        if not self.chunking_service and hasattr(self.app_instance, 'media_db') and self.app_instance.media_db:
            self.chunking_service = get_chunking_service(self.app_instance.media_db)
        
        if new_data:
            self._update_metadata_display()
            self._update_content_display()
            self._update_delete_button()
            self._load_chunking_config()
        else:
            self._clear_displays()
    
    def watch_format_for_reading(self, old_value: bool, new_value: bool) -> None:
        """React to format checkbox changes by re-rendering content."""
        if self.media_data:
            # Store current scroll position
            try:
                content_area = self.query_one(f"#content-display-{self.type_slug}", TextArea)
                scroll_y = content_area.scroll_y
                # Update the display with new formatting
                self._update_content_display()
                # Try to restore scroll position
                content_area.scroll_to(0, scroll_y, animate=False)
            except Exception as e:
                logger.error(f"Error updating format: {e}")
                # Fall back to just updating display
                self._update_content_display()
    
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
    
    def _format_text_for_reading(self, text: str) -> str:
        """Format text for easier reading by adding newlines after periods."""
        if not text:
            return text
            
        # Preserve existing double newlines (paragraph breaks)
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            # Skip formatting for certain patterns
            # Check if it looks like a URL (contains :// or starts with www.)
            if '://' in paragraph or paragraph.strip().startswith('www.'):
                formatted_paragraphs.append(paragraph)
                continue
                
            # Check if it looks like code (starts with spaces/tabs or contains common code patterns)
            if paragraph.strip().startswith(('    ', '\t', '```', '~~~')):
                formatted_paragraphs.append(paragraph)
                continue
                
            # Format regular text: add newline after ". " (period followed by space)
            # But not for common abbreviations
            formatted = paragraph
            
            # Common abbreviations to skip
            abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.', 'Ph.D.', 'M.D.', 
                           'B.A.', 'M.A.', 'B.S.', 'M.S.', 'LL.B.', 'LL.M.', 'U.S.', 'U.K.', 
                           'E.g.', 'e.g.', 'I.e.', 'i.e.', 'etc.', 'vs.', 'Inc.', 'Ltd.', 'Co.']
            
            # Create a pattern that matches ". " but not when preceded by abbreviations
            import re
            
            # First, protect abbreviations by temporarily replacing them
            for abbr in abbreviations:
                formatted = formatted.replace(abbr, abbr.replace('.', '<!DOT!>'))
            
            # Now add newlines after periods followed by space
            formatted = re.sub(r'\. (?=[A-Z])', '.\n', formatted)
            
            # Restore the abbreviations
            formatted = formatted.replace('<!DOT!>', '.')
            
            formatted_paragraphs.append(formatted)
        
        return '\n\n'.join(formatted_paragraphs)
    
    def _update_content_display(self) -> None:
        """Update the content display with media content."""
        if not self.media_data:
            return
            
        try:
            content_area = self.query_one(f"#content-display-{self.type_slug}", TextArea)
            
            title = self.media_data.get('title', 'Untitled')
            content = self.media_data.get('content', 'No content available')
            
            # Apply formatting if checkbox is checked
            if self.format_for_reading:
                content = self._format_text_for_reading(content)
            
            # Format content with title
            formatted_content = f"# {title}\n\n{content}"
            
            # Add analysis content if viewing analysis-review
            if self.type_slug == "analysis-review":
                analysis = self.media_data.get('analysis_content', '')
                if analysis:
                    # Apply formatting to analysis too if enabled
                    if self.format_for_reading:
                        analysis = self._format_text_for_reading(analysis)
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
    
    @on(Button.Pressed)
    def handle_chunking_buttons(self, event: Button.Pressed) -> None:
        """Handle chunking configuration button presses."""
        button_id = event.button.id
        
        if button_id == f"save-chunking-{self.type_slug}":
            self._save_chunking_config()
        elif button_id == f"preview-chunks-{self.type_slug}":
            self._preview_chunks()
        elif button_id == f"reset-chunking-{self.type_slug}":
            self._reset_chunking_config()
    
    @on(Select.Changed)
    def handle_template_change(self, event: Select.Changed) -> None:
        """Handle chunking template selection changes."""
        if event.select.id == f"chunking-template-select-{self.type_slug}":
            self._load_template_config(event.value)
    
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
    
    @on(Checkbox.Changed)
    def handle_format_checkbox(self, event: Checkbox.Changed) -> None:
        """Handle format reading checkbox changes."""
        if event.checkbox.id == f"format-reading-checkbox-{self.type_slug}":
            self.format_for_reading = event.value
    
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
    
    def _save_chunking_config(self) -> None:
        """Save the chunking configuration for this media item."""
        if not self.media_data or not self.chunking_service:
            return
        
        try:
            # Get values from form
            template = self.query_one(f"#chunking-template-select-{self.type_slug}", Select).value
            chunk_size = self.query_one(f"#chunk-size-{self.type_slug}", Input).value
            chunk_overlap = self.query_one(f"#chunk-overlap-{self.type_slug}", Input).value
            chunk_method = self.query_one(f"#chunk-method-{self.type_slug}", Select).value
            enable_late = self.query_one(f"#enable-late-chunking-{self.type_slug}", Checkbox).value
            
            # Build configuration
            config = {
                "template": template if template != "default" else None,
                "chunk_size": int(chunk_size) if chunk_size else 400,
                "chunk_overlap": int(chunk_overlap) if chunk_overlap else 100,
                "method": chunk_method,
                "enable_late_chunking": enable_late
            }
            
            # Save using the service
            self.chunking_service.set_document_config(self.media_data['id'], config)
            
            # Update display
            self._update_chunking_display(config)
            
            # Show success notification
            self.app_instance.notify("Chunking configuration saved", severity="information")
            
        except ChunkingTemplateError as e:
            logger.error(f"Error saving chunking config: {e}")
            self.app_instance.notify(f"Error saving configuration: {str(e)}", severity="error")
        except Exception as e:
            logger.error(f"Unexpected error saving chunking config: {e}")
            self.app_instance.notify(f"Unexpected error: {str(e)}", severity="error")
    
    def _preview_chunks(self) -> None:
        """Preview chunks with current configuration."""
        if not self.media_data or not self.media_data.get('content'):
            self.app_instance.notify("No content available to preview", severity="warning")
            return
        
        # Import and show preview modal
        from ..Widgets.chunk_preview_modal import ChunkPreviewModal
        
        # Get current config from form
        config = {
            "chunk_size": int(self.query_one(f"#chunk-size-{self.type_slug}", Input).value or 400),
            "chunk_overlap": int(self.query_one(f"#chunk-overlap-{self.type_slug}", Input).value or 100),
            "method": self.query_one(f"#chunk-method-{self.type_slug}", Select).value
        }
        
        # Show preview modal
        self.app_instance.push_screen(
            ChunkPreviewModal(
                content=self.media_data['content'],
                config=config,
                media_title=self.media_data.get('title', 'Untitled')
            )
        )
    
    def _reset_chunking_config(self) -> None:
        """Reset chunking configuration to defaults."""
        try:
            # Reset form values
            self.query_one(f"#chunking-template-select-{self.type_slug}", Select).value = "default"
            self.query_one(f"#chunk-size-{self.type_slug}", Input).value = "400"
            self.query_one(f"#chunk-overlap-{self.type_slug}", Input).value = "100"
            self.query_one(f"#chunk-method-{self.type_slug}", Select).value = "words"
            self.query_one(f"#enable-late-chunking-{self.type_slug}", Checkbox).value = False
            
            # Clear database config using the service
            if self.media_data and self.chunking_service:
                self.chunking_service.clear_document_config(self.media_data['id'])
                
                # Update display
                self._update_chunking_display(None)
                
                self.app_instance.notify("Chunking configuration reset to defaults", severity="information")
                
        except ChunkingTemplateError as e:
            logger.error(f"Error resetting chunking config: {e}")
            self.app_instance.notify(f"Error resetting configuration: {str(e)}", severity="error")
        except Exception as e:
            logger.error(f"Unexpected error resetting chunking config: {e}")
            self.app_instance.notify(f"Unexpected error: {str(e)}", severity="error")
    
    def _load_template_config(self, template_name: str) -> None:
        """Load configuration from a template."""
        if template_name == "default" or not self.chunking_service:
            return
        
        try:
            # Load template using the service
            template = self.chunking_service.get_template_by_name(template_name)
            
            if template:
                template_data = json.loads(template['template_json'])
                
                # Apply template settings to form
                if 'pipeline' in template_data:
                    for stage in template_data['pipeline']:
                        if stage.get('stage') == 'chunk':
                            options = stage.get('options', {})
                            
                            if 'max_size' in options:
                                self.query_one(f"#chunk-size-{self.type_slug}", Input).value = str(options['max_size'])
                            if 'overlap' in options:
                                self.query_one(f"#chunk-overlap-{self.type_slug}", Input).value = str(options['overlap'])
                            if 'method' in stage:
                                self.query_one(f"#chunk-method-{self.type_slug}", Select).value = stage['method']
                            
                            break
        
        except ChunkingTemplateError as e:
            logger.error(f"Error loading template: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading template: {e}")
    
    def _update_chunking_display(self, config: Optional[Dict[str, Any]]) -> None:
        """Update the chunking configuration display."""
        try:
            display = self.query_one(f"#chunking-config-display-{self.type_slug}", Static)
            
            if config:
                template = config.get('template', 'Custom')
                method = config.get('method', 'words')
                size = config.get('chunk_size', 400)
                overlap = config.get('chunk_overlap', 100)
                
                display.update(f"Current: {template} - {method} ({size} words, {overlap} overlap)")
            else:
                display.update("Current: Default configuration")
                
        except Exception as e:
            logger.error(f"Error updating chunking display: {e}")
    
    def _load_chunking_config(self) -> None:
        """Load and display the current chunking configuration."""
        if not self.media_data or not self.chunking_service:
            return
        
        try:
            # Get config using the service
            config = self.chunking_service.get_document_config(self.media_data['id'])
            
            if config:
                # Update form fields
                if config.get('template'):
                    self.query_one(f"#chunking-template-select-{self.type_slug}", Select).value = config['template']
                
                self.query_one(f"#chunk-size-{self.type_slug}", Input).value = str(config.get('chunk_size', 400))
                self.query_one(f"#chunk-overlap-{self.type_slug}", Input).value = str(config.get('chunk_overlap', 100))
                self.query_one(f"#chunk-method-{self.type_slug}", Select).value = config.get('method', 'words')
                self.query_one(f"#enable-late-chunking-{self.type_slug}", Checkbox).value = config.get('enable_late_chunking', False)
                
                # Update display
                self._update_chunking_display(config)
            else:
                self._update_chunking_display(None)
                
        except ChunkingTemplateError as e:
            logger.error(f"Error loading chunking config: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading chunking config: {e}")
    
    # Content search methods
    @on(Button.Pressed, ".content-search-button")
    @on(Input.Submitted, ".content-search-input")
    async def _perform_content_search(self, event) -> None:
        """Perform search within the content."""
        try:
            search_input = self.query_one(f"#content-search-input-{self.type_slug}", Input)
            search_term = search_input.value.strip()
            
            if not search_term:
                self.search_matches = []
                self.current_match_index = -1
                self._update_search_status()
                return
            
            content_area = self.query_one(f"#content-display-{self.type_slug}", TextArea)
            content = content_area.text
            
            # Find all matches (case-insensitive)
            matches = []
            search_lower = search_term.lower()
            content_lower = content.lower()
            start = 0
            
            while True:
                pos = content_lower.find(search_lower, start)
                if pos == -1:
                    break
                matches.append((pos, pos + len(search_term)))
                start = pos + 1
            
            self.search_matches = matches
            
            if matches:
                self.current_match_index = 0
                self._highlight_current_match()
                self._update_search_navigation()
            else:
                self.current_match_index = -1
                self.app_instance.notify(f"No matches found for '{search_term}'", severity="information")
            
            self._update_search_status()
            
        except Exception as e:
            logger.error(f"Error performing content search: {e}")
            self.app_instance.notify("Search error occurred", severity="error")
    
    @on(Button.Pressed, ".content-search-nav")
    async def _navigate_search_results(self, event: Button.Pressed) -> None:
        """Navigate through search results."""
        if not self.search_matches:
            return
        
        button_id = event.button.id
        
        if f"content-search-prev-{self.type_slug}" in button_id:
            self.current_match_index = (self.current_match_index - 1) % len(self.search_matches)
        elif f"content-search-next-{self.type_slug}" in button_id:
            self.current_match_index = (self.current_match_index + 1) % len(self.search_matches)
        
        self._highlight_current_match()
        self._update_search_status()
    
    def _highlight_current_match(self) -> None:
        """Highlight the current search match in the content."""
        if self.current_match_index < 0 or not self.search_matches:
            return
        
        try:
            content_area = self.query_one(f"#content-display-{self.type_slug}", TextArea)
            start, end = self.search_matches[self.current_match_index]
            
            # Calculate line and column for the match
            text_before = content_area.text[:start]
            line = text_before.count('\n')
            last_newline = text_before.rfind('\n')
            column = start - last_newline - 1 if last_newline != -1 else start
            
            # Move cursor to the match
            content_area.cursor_location = (line, column)
            content_area.selection = (line, column, line, column + (end - start))
            
            # Scroll to make the match visible
            content_area.scroll_to_center(line)
            
        except Exception as e:
            logger.error(f"Error highlighting match: {e}")
    
    def _update_search_status(self) -> None:
        """Update the search status display."""
        try:
            status = self.query_one(f"#content-search-status-{self.type_slug}", Static)
            
            if not self.search_matches:
                status.update("")
            else:
                status.update(f"{self.current_match_index + 1} of {len(self.search_matches)}")
                
        except Exception as e:
            logger.error(f"Error updating search status: {e}")
    
    def _update_search_navigation(self) -> None:
        """Update search navigation button states."""
        try:
            prev_button = self.query_one(f"#content-search-prev-{self.type_slug}", Button)
            next_button = self.query_one(f"#content-search-next-{self.type_slug}", Button)
            
            has_matches = len(self.search_matches) > 0
            prev_button.disabled = not has_matches
            next_button.disabled = not has_matches
            
        except Exception as e:
            logger.error(f"Error updating search navigation: {e}")
    
    def watch_search_matches(self, old_matches: List[Tuple[int, int]], new_matches: List[Tuple[int, int]]) -> None:
        """React to changes in search matches."""
        self._update_search_navigation()
        if not new_matches:
            # Clear any existing selection
            try:
                content_area = self.query_one(f"#content-display-{self.type_slug}", TextArea)
                content_area.selection = None
            except Exception:
                pass
    
