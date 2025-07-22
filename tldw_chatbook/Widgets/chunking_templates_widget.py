"""
Widget for managing chunking templates with CRUD operations.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple
from datetime import datetime
import json
from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import (
    Static, Button, Label, DataTable, Input, 
    ListView, ListItem, Markdown, Header, Footer
)
from textual.reactive import reactive
from textual.message import Message
from loguru import logger

from ..Chunking.chunking_interop_library import (
    ChunkingInteropService, 
    get_chunking_service,
    ChunkingTemplateError,
    TemplateNotFoundError,
    SystemTemplateError
)
from ..DB.Client_Media_DB_v2 import InputError

if TYPE_CHECKING:
    from ..app import TldwCli


class ChunkingTemplatesWidget(Container):
    """
    A widget for managing chunking templates with full CRUD operations.
    """
    
    DEFAULT_CSS = """
    ChunkingTemplatesWidget {
        height: 100%;
        width: 100%;
    }
    
    .templates-header {
        height: 3;
        padding: 1;
        background: $boost;
        margin-bottom: 1;
    }
    
    .templates-actions {
        height: 3;
        margin: 1 0;
    }
    
    .templates-table-container {
        height: 70%;
        border: solid $primary;
        margin: 1 0;
    }
    
    .template-details {
        height: 25%;
        border: solid $secondary;
        padding: 1;
        margin: 1 0;
    }
    
    .filter-container {
        height: 3;
        margin-bottom: 1;
    }
    
    .action-button {
        margin: 0 1;
    }
    """
    
    # Reactive attributes
    selected_template_id = reactive(None)
    filter_text = reactive("")
    templates_data = reactive([])
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """
        Initialize the ChunkingTemplatesWidget.
        
        Args:
            app_instance: Reference to the main app instance
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.templates_cache = {}
        self.chunking_service = None
        
    def compose(self) -> ComposeResult:
        """Compose the widget's UI."""
        # Header
        with Container(classes="templates-header"):
            yield Label("📋 Chunking Templates Manager", classes="section-title")
        
        # Filter section
        with Horizontal(classes="filter-container"):
            yield Label("Filter:", classes="filter-label")
            yield Input(
                placeholder="Search templates...",
                id="template-filter-input",
                classes="filter-input"
            )
            yield Button("Refresh", id="refresh-templates-btn", variant="default")
        
        # Action buttons
        with Horizontal(classes="templates-actions"):
            yield Button("➕ New Template", id="new-template-btn", variant="primary", classes="action-button")
            yield Button("📝 Edit", id="edit-template-btn", variant="default", classes="action-button", disabled=True)
            yield Button("📋 Duplicate", id="duplicate-template-btn", variant="default", classes="action-button", disabled=True)
            yield Button("🗑️ Delete", id="delete-template-btn", variant="error", classes="action-button", disabled=True)
            yield Button("📥 Import", id="import-template-btn", variant="default", classes="action-button")
            yield Button("📤 Export", id="export-template-btn", variant="default", classes="action-button", disabled=True)
        
        # Templates table
        with Container(classes="templates-table-container"):
            table = DataTable(id="templates-table", zebra_stripes=True)
            table.add_columns(
                ("Name", 25),
                ("Description", 40),
                ("Type", 10),
                ("Method", 15),
                ("Created", 20),
                ("ID", 8)
            )
            yield table
        
        # Template details preview
        with VerticalScroll(classes="template-details"):
            yield Markdown("", id="template-details-display")
    
    def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        # Initialize the chunking service
        if hasattr(self.app_instance, 'media_db') and self.app_instance.media_db:
            self.chunking_service = get_chunking_service(self.app_instance.media_db)
            self.refresh_templates()
        else:
            logger.warning("Media database not available for chunking service")
    
    def watch_selected_template_id(self, old_id: Optional[int], new_id: Optional[int]) -> None:
        """React to template selection changes."""
        # Update button states
        has_selection = new_id is not None
        is_system = False
        
        if has_selection and new_id in self.templates_cache:
            template = self.templates_cache[new_id]
            is_system = template.get('is_system', False)
        
        # Enable/disable buttons based on selection and type
        self.query_one("#edit-template-btn", Button).disabled = not has_selection or is_system
        self.query_one("#duplicate-template-btn", Button).disabled = not has_selection
        self.query_one("#delete-template-btn", Button).disabled = not has_selection or is_system
        self.query_one("#export-template-btn", Button).disabled = not has_selection
        
        # Update details display
        self._update_template_details()
    
    def watch_filter_text(self, old_text: str, new_text: str) -> None:
        """React to filter text changes."""
        self._apply_filter()
    
    @work(thread=True)
    def refresh_templates(self) -> None:
        """Refresh the templates list from the database."""
        try:
            if not self.chunking_service:
                logger.warning("Chunking service not available")
                return
            
            # Get all templates using the service
            templates = self.chunking_service.get_all_templates()
            
            # Clear and rebuild cache
            self.templates_cache.clear()
            
            # Process templates for display
            display_templates = []
            for template in templates:
                # Parse template JSON to get method
                try:
                    template_obj = json.loads(template['template_json'])
                    template['base_method'] = template_obj.get('base_method', 'unknown')
                except:
                    template['base_method'] = 'unknown'
                
                self.templates_cache[template['id']] = template
                display_templates.append(template)
            
            self.app_instance.call_from_thread(self._update_table, display_templates)
            
        except ChunkingTemplateError as e:
            logger.error(f"Error refreshing templates: {e}")
            self.app_instance.call_from_thread(
                self.app_instance.notify,
                f"Error loading templates: {str(e)}",
                severity="error"
            )
        except Exception as e:
            logger.error(f"Unexpected error refreshing templates: {e}")
            self.app_instance.call_from_thread(
                self.app_instance.notify,
                f"Unexpected error: {str(e)}",
                severity="error"
            )
    
    def _update_table(self, templates: List[Dict[str, Any]]) -> None:
        """Update the templates table with new data."""
        self.templates_data = templates
        self._apply_filter()
    
    def _apply_filter(self) -> None:
        """Apply the current filter to the templates list."""
        table = self.query_one("#templates-table", DataTable)
        table.clear()
        
        filter_text = self.filter_text.lower()
        
        for template in self.templates_data:
            # Apply filter
            if filter_text:
                searchable = f"{template['name']} {template['description']} {template['base_method']}".lower()
                if filter_text not in searchable:
                    continue
            
            # Format data for display
            template_type = "System" if template['is_system'] else "Custom"
            created_date = datetime.fromisoformat(template['created_at']).strftime("%Y-%m-%d %H:%M")
            
            # Add row with styled type
            if template['is_system']:
                template_type = f"[bold cyan]{template_type}[/bold cyan]"
            
            table.add_row(
                template['name'],
                template['description'][:40] + "..." if len(template['description']) > 40 else template['description'],
                template_type,
                template['base_method'],
                created_date,
                str(template['id'])
            )
    
    def _update_template_details(self) -> None:
        """Update the template details display."""
        details_display = self.query_one("#template-details-display", Markdown)
        
        if not self.selected_template_id or self.selected_template_id not in self.templates_cache:
            details_display.update("*Select a template to view details*")
            return
        
        template = self.templates_cache[self.selected_template_id]
        
        try:
            template_obj = json.loads(template['template_json'])
            pipeline_stages = template_obj.get('pipeline', [])
            
            # Format pipeline stages for display
            pipeline_text = ""
            for i, stage in enumerate(pipeline_stages, 1):
                stage_type = stage.get('stage', 'unknown')
                method = stage.get('method', '')
                options = stage.get('options', {})
                
                pipeline_text += f"\n{i}. **{stage_type.title()} Stage**"
                if method:
                    pipeline_text += f" - Method: `{method}`"
                if options:
                    pipeline_text += f"\n   - Options: `{json.dumps(options, indent=2)}`"
            
            details_md = f"""## {template['name']}

**Type:** {"System Template" if template['is_system'] else "Custom Template"}  
**Base Method:** `{template_obj.get('base_method', 'unknown')}`  
**Description:** {template['description']}

### Pipeline Configuration
{pipeline_text if pipeline_text else "*No pipeline stages defined*"}

### Metadata
- **Version:** {template_obj.get('metadata', {}).get('version', '1.0')}
- **Created:** {template['created_at']}
- **Updated:** {template['updated_at']}
- **ID:** {template['id']}
"""
            
            details_display.update(details_md)
            
        except Exception as e:
            logger.error(f"Error parsing template details: {e}")
            details_display.update(f"*Error displaying template details: {str(e)}*")
    
    @on(DataTable.RowSelected)
    def handle_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle template selection from the table."""
        if event.row_index is None:
            return
        
        table = self.query_one("#templates-table", DataTable)
        row_data = table.get_row_at(event.row_index)
        
        if row_data and len(row_data) > 5:
            template_id = int(row_data[5])  # ID is in the last column
            self.selected_template_id = template_id
    
    @on(Input.Changed, "#template-filter-input")
    def handle_filter_change(self, event: Input.Changed) -> None:
        """Handle filter input changes."""
        self.filter_text = event.value
    
    @on(Button.Pressed, "#refresh-templates-btn")
    def handle_refresh(self) -> None:
        """Handle refresh button press."""
        self.refresh_templates()
    
    @on(Button.Pressed, "#new-template-btn")
    def handle_new_template(self) -> None:
        """Handle new template button press."""
        from .chunking_template_editor import ChunkingTemplateEditor
        
        self.app_instance.push_screen(
            ChunkingTemplateEditor(
                self.app_instance,
                mode="create",
                on_save=self._on_template_saved
            )
        )
    
    @on(Button.Pressed, "#edit-template-btn")
    def handle_edit_template(self) -> None:
        """Handle edit template button press."""
        if not self.selected_template_id:
            return
        
        template = self.templates_cache.get(self.selected_template_id)
        if not template or template.get('is_system'):
            self.app_instance.notify("Cannot edit system templates", severity="warning")
            return
        
        from .chunking_template_editor import ChunkingTemplateEditor
        
        self.app_instance.push_screen(
            ChunkingTemplateEditor(
                self.app_instance,
                mode="edit",
                template_data=template,
                on_save=self._on_template_saved
            )
        )
    
    @on(Button.Pressed, "#duplicate-template-btn")
    def handle_duplicate_template(self) -> None:
        """Handle duplicate template button press."""
        if not self.selected_template_id:
            return
        
        template = self.templates_cache.get(self.selected_template_id)
        if not template:
            return
        
        # Create a copy with modified name
        duplicate = template.copy()
        duplicate['name'] = f"{template['name']} (Copy)"
        duplicate['is_system'] = False  # Duplicates are always custom
        duplicate.pop('id', None)  # Remove ID so a new one is created
        
        from .chunking_template_editor import ChunkingTemplateEditor
        
        self.app_instance.push_screen(
            ChunkingTemplateEditor(
                self.app_instance,
                mode="create",
                template_data=duplicate,
                on_save=self._on_template_saved
            )
        )
    
    @on(Button.Pressed, "#delete-template-btn")
    def handle_delete_template(self) -> None:
        """Handle delete template button press."""
        if not self.selected_template_id:
            return
        
        template = self.templates_cache.get(self.selected_template_id)
        if not template:
            return
        
        if template.get('is_system'):
            self.app_instance.notify("Cannot delete system templates", severity="warning")
            return
        
        # Show confirmation dialog
        from ..Event_Handlers.template_events import TemplateDeleteConfirmationEvent
        self.post_message(TemplateDeleteConfirmationEvent(
            template_id=self.selected_template_id,
            template_name=template['name']
        ))
    
    @on(Button.Pressed, "#import-template-btn")
    def handle_import_template(self) -> None:
        """Handle import template button press."""
        # Show file picker for JSON import
        from ..Third_Party.textual_fspicker import FileOpen
        
        self.app_instance.push_screen(
            FileOpen(
                ".",
                filters=[("*.json", "JSON files"), ("*", "All files")],
                select_button_text="Import"
            ),
            self._import_template_file
        )
    
    @on(Button.Pressed, "#export-template-btn")
    def handle_export_template(self) -> None:
        """Handle export template button press."""
        if not self.selected_template_id or not self.chunking_service:
            return
        
        try:
            # Export using the service
            export_data = self.chunking_service.export_template(self.selected_template_id)
            
            # Save to file
            template_name = export_data['name'].replace(' ', '_')
            export_path = Path.home() / f"chunking_template_{template_name}.json"
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            self.app_instance.notify(
                f"Template exported to: {export_path}",
                severity="information"
            )
            
        except TemplateNotFoundError:
            self.app_instance.notify("Template not found", severity="warning")
        except Exception as e:
            logger.error(f"Error exporting template: {e}")
            self.app_instance.notify(
                f"Error exporting template: {str(e)}",
                severity="error"
            )
    
    def _import_template_file(self, path: Optional[Path]) -> None:
        """Import a template from a JSON file."""
        if not path or not path.exists() or not self.chunking_service:
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Import using the service
            template_id = self.chunking_service.import_template(import_data)
            
            # Get the imported template to show its name
            imported_template = self.chunking_service.get_template_by_id(template_id)
            
            self.app_instance.notify(
                f"Template '{imported_template['name']}' imported successfully",
                severity="information"
            )
            
            # Refresh the templates list
            self.refresh_templates()
            
        except InputError as e:
            logger.error(f"Invalid template data: {e}")
            self.app_instance.notify(
                f"Invalid template: {str(e)}",
                severity="error"
            )
        except Exception as e:
            logger.error(f"Error importing template: {e}")
            self.app_instance.notify(
                f"Error importing template: {str(e)}",
                severity="error"
            )
    
    def _on_template_saved(self) -> None:
        """Callback when a template is saved from the editor."""
        self.refresh_templates()
    
    def delete_template(self, template_id: int) -> None:
        """Delete a template from the database."""
        try:
            if not self.chunking_service:
                raise ChunkingTemplateError("Chunking service not available")
            
            # Delete using the service
            self.chunking_service.delete_template(template_id)
            
            self.app_instance.notify("Template deleted successfully", severity="information")
            self.selected_template_id = None
            self.refresh_templates()
            
        except SystemTemplateError:
            self.app_instance.notify("Cannot delete system templates", severity="warning")
        except TemplateNotFoundError:
            self.app_instance.notify("Template not found", severity="warning")
        except ChunkingTemplateError as e:
            logger.error(f"Error deleting template: {e}")
            self.app_instance.notify(
                f"Error deleting template: {str(e)}",
                severity="error"
            )
        except Exception as e:
            logger.error(f"Unexpected error deleting template: {e}")
            self.app_instance.notify(
                f"Unexpected error: {str(e)}",
                severity="error"
            )