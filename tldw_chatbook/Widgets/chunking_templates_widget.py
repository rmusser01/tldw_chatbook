"""
Widget for managing chunking templates with CRUD operations.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any
from datetime import datetime
import json
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import (
    Static, Button, Label, DataTable, Input, 
    Markdown
)
from textual.reactive import reactive
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


CHUNKING_EDIT_DISABLED_TOOLTIP = "Select a custom chunking template before editing."
CHUNKING_DUPLICATE_DISABLED_TOOLTIP = "Select a chunking template before duplicating it."
CHUNKING_DELETE_DISABLED_TOOLTIP = "Select a custom chunking template before deleting."
CHUNKING_EXPORT_DISABLED_TOOLTIP = "Select a chunking template before exporting it."
CHUNKING_EDIT_ENABLED_TOOLTIP = "Edit the selected custom chunking template."
CHUNKING_DUPLICATE_ENABLED_TOOLTIP = "Duplicate the selected chunking template."
CHUNKING_DELETE_ENABLED_TOOLTIP = "Delete the selected custom chunking template."
CHUNKING_EXPORT_ENABLED_TOOLTIP = "Export the selected chunking template."
CHUNKING_EDIT_BUILTIN_TOOLTIP = "Built-in chunking templates cannot be edited; duplicate it first."
CHUNKING_DELETE_BUILTIN_TOOLTIP = "Built-in chunking templates cannot be deleted."


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
    selected_template_record_id = reactive(None)
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
        self.templates_cache_by_record_id: Dict[str, Dict[str, Any]] = {}
        self.templates_cache_by_name: Dict[str, Dict[str, Any]] = {}
        self.filtered_templates: List[Dict[str, Any]] = []
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
            yield Button(
                "📝 Edit",
                id="edit-template-btn",
                variant="default",
                classes="action-button",
                disabled=True,
                tooltip=CHUNKING_EDIT_DISABLED_TOOLTIP,
            )
            yield Button(
                "📋 Duplicate",
                id="duplicate-template-btn",
                variant="default",
                classes="action-button",
                disabled=True,
                tooltip=CHUNKING_DUPLICATE_DISABLED_TOOLTIP,
            )
            yield Button(
                "🗑️ Delete",
                id="delete-template-btn",
                variant="error",
                classes="action-button",
                disabled=True,
                tooltip=CHUNKING_DELETE_DISABLED_TOOLTIP,
            )
            yield Button("📥 Import", id="import-template-btn", variant="default", classes="action-button")
            yield Button(
                "📤 Export",
                id="export-template-btn",
                variant="default",
                classes="action-button",
                disabled=True,
                tooltip=CHUNKING_EXPORT_DISABLED_TOOLTIP,
            )
        
        # Templates table
        with Container(classes="templates-table-container"):
            yield DataTable(id="templates-table", zebra_stripes=True)
        
        # Template details preview
        with VerticalScroll(classes="template-details"):
            yield Markdown("", id="template-details-display")
    
    async def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        # Setup the table columns
        table = self.query_one("#templates-table", DataTable)
        table.add_columns(
            "Name",
            "Description",
            "Type",
            "Method",
            "Created",
            "ID"
        )
        
        # Initialize the local chunking service for import/export helpers.
        if hasattr(self.app_instance, 'media_db') and self.app_instance.media_db:
            self.chunking_service = get_chunking_service(self.app_instance.media_db)
        else:
            logger.warning("Media database not available for chunking service")
        await self.refresh_templates()
    
    def _runtime_backend(self) -> str:
        """Resolve the active runtime backend for this admin surface."""
        candidates = (
            getattr(getattr(self.app_instance, "media_runtime_state", None), "runtime_backend", None),
            getattr(self.app_instance, "current_runtime_backend", None),
            getattr(self.app_instance, "runtime_backend", None),
        )
        for candidate in candidates:
            normalized = str(candidate or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        return "local"

    def _scope_service(self) -> Any:
        return getattr(self.app_instance, "rag_admin_scope_service", None)

    def _record_for_selection(self) -> Optional[Dict[str, Any]]:
        if not self.selected_template_record_id:
            return None
        return self.templates_cache_by_record_id.get(str(self.selected_template_record_id))

    def _extract_template_object(self, template: Dict[str, Any]) -> Dict[str, Any]:
        payload = template.get("template")
        if isinstance(payload, dict):
            return dict(payload)
        raw_json = template.get("template_json")
        if isinstance(raw_json, str) and raw_json.strip():
            try:
                parsed = json.loads(raw_json)
            except (TypeError, ValueError):
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _extract_base_method(self, template: Dict[str, Any]) -> str:
        template_obj = self._extract_template_object(template)
        chunking = template_obj.get("chunking")
        if isinstance(chunking, dict):
            method = chunking.get("method")
            if method:
                return str(method)

        base_method = template_obj.get("base_method")
        if base_method:
            return str(base_method)

        for stage in template_obj.get("pipeline", []) if isinstance(template_obj.get("pipeline"), list) else []:
            if stage.get("stage") == "chunk" and stage.get("method"):
                return str(stage["method"])
        return "unknown"

    def _format_created_date(self, created_at: Any) -> str:
        if isinstance(created_at, datetime):
            return created_at.strftime("%Y-%m-%d %H:%M")
        if isinstance(created_at, str) and created_at.strip():
            try:
                parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                return created_at
            return parsed.strftime("%Y-%m-%d %H:%M")
        return "Unknown"

    def watch_selected_template_record_id(self, old_id: Optional[str], new_id: Optional[str]) -> None:
        """React to template selection changes."""
        # Update button states
        has_selection = new_id is not None
        is_system = False
        
        if has_selection and new_id in self.templates_cache_by_record_id:
            template = self.templates_cache_by_record_id[new_id]
            is_system = bool(template.get('is_builtin', False))
        
        # Enable/disable buttons based on selection and type
        self.query_one("#edit-template-btn", Button).disabled = not has_selection or is_system
        self.query_one("#duplicate-template-btn", Button).disabled = not has_selection
        self.query_one("#delete-template-btn", Button).disabled = not has_selection or is_system
        self.query_one("#export-template-btn", Button).disabled = not has_selection
        self._update_action_tooltips(has_selection=has_selection, is_system=is_system)
        
        # Update details display
        self._update_template_details()

    def _update_action_tooltips(self, *, has_selection: bool, is_system: bool) -> None:
        """Explain why template actions are unavailable and what enabled actions do."""
        edit_button = self.query_one("#edit-template-btn", Button)
        duplicate_button = self.query_one("#duplicate-template-btn", Button)
        delete_button = self.query_one("#delete-template-btn", Button)
        export_button = self.query_one("#export-template-btn", Button)

        if not has_selection:
            edit_button.tooltip = CHUNKING_EDIT_DISABLED_TOOLTIP
            duplicate_button.tooltip = CHUNKING_DUPLICATE_DISABLED_TOOLTIP
            delete_button.tooltip = CHUNKING_DELETE_DISABLED_TOOLTIP
            export_button.tooltip = CHUNKING_EXPORT_DISABLED_TOOLTIP
            return

        edit_button.tooltip = (
            CHUNKING_EDIT_BUILTIN_TOOLTIP if is_system else CHUNKING_EDIT_ENABLED_TOOLTIP
        )
        duplicate_button.tooltip = CHUNKING_DUPLICATE_ENABLED_TOOLTIP
        delete_button.tooltip = (
            CHUNKING_DELETE_BUILTIN_TOOLTIP if is_system else CHUNKING_DELETE_ENABLED_TOOLTIP
        )
        export_button.tooltip = CHUNKING_EXPORT_ENABLED_TOOLTIP
    
    def watch_filter_text(self, old_text: str, new_text: str) -> None:
        """React to filter text changes."""
        self._apply_filter()
    
    async def refresh_templates(self) -> None:
        """Refresh the templates list from the active backend."""
        try:
            scope = self._scope_service()
            if scope is None:
                logger.warning("RAG admin scope service not available")
                return
            templates = await scope.list_templates(
                mode=self._runtime_backend(),
                include_builtin=True,
                include_custom=True,
            )
            for template in templates:
                template["base_method"] = self._extract_base_method(template)

            self._update_table(templates)

        except Exception as e:
            logger.error(f"Error refreshing templates: {e}")
            self.app_instance.notify(
                f"Error loading templates: {str(e)}",
                severity="error"
            )
    
    def _update_table(self, templates: List[Dict[str, Any]]) -> None:
        """Update the templates table with new data."""
        self.templates_data = templates
        self.templates_cache_by_record_id = {
            str(template["record_id"]): dict(template)
            for template in templates
        }
        self.templates_cache_by_name = {
            str(template["name"]): dict(template)
            for template in templates
        }
        self._apply_filter()
    
    def _apply_filter(self) -> None:
        """Apply the current filter to the templates list."""
        table = self.query_one("#templates-table", DataTable)
        table.clear()
        
        filter_text = self.filter_text.lower()
        
        self.filtered_templates = []
        for template in self.templates_data:
            # Apply filter
            if filter_text:
                searchable = f"{template['name']} {template['description']} {template['base_method']}".lower()
                if filter_text not in searchable:
                    continue
            self.filtered_templates.append(template)
            
            # Format data for display
            template_type = "Built-in" if template.get('is_builtin') else "Custom"
            created_date = self._format_created_date(template.get('created_at'))
            
            # Add row with styled type
            if template.get('is_builtin'):
                template_type = f"[bold cyan]{template_type}[/bold cyan]"
            
            table.add_row(
                template['name'],
                template['description'][:40] + "..." if len(template['description']) > 40 else template['description'],
                template_type,
                template['base_method'],
                created_date,
                str(template.get('backing_id') or template.get('record_id'))
            )
    
    def _update_template_details(self) -> None:
        """Update the template details display."""
        details_display = self.query_one("#template-details-display", Markdown)
        
        template = self._record_for_selection()
        if template is None:
            details_display.update("*Select a template to view details*")
            return
        
        try:
            template_obj = self._extract_template_object(template)
            details_md = f"""## {template['name']}

**Type:** {"Built-in Template" if template.get('is_builtin') else "Custom Template"}  
**Backend:** `{template.get('backend', self._runtime_backend())}`  
**Method:** `{self._extract_base_method(template)}`  
**Description:** {template.get('description') or ""}

### Template JSON
```json
{json.dumps(template_obj, indent=2)}
```

### Metadata
- **Created:** {template.get('created_at')}
- **Updated:** {template.get('updated_at')}
- **Version:** {template.get('version', 1)}
- **Record ID:** {template.get('record_id')}
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
        if event.row_index < len(self.filtered_templates):
            self.selected_template_record_id = self.filtered_templates[event.row_index]["record_id"]
    
    @on(Input.Changed, "#template-filter-input")
    def handle_filter_change(self, event: Input.Changed) -> None:
        """Handle filter input changes."""
        self.filter_text = event.value
    
    @on(Button.Pressed, "#refresh-templates-btn")
    async def handle_refresh(self) -> None:
        """Handle refresh button press."""
        await self.refresh_templates()
    
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
        if not self.selected_template_record_id:
            return
        
        template = self._record_for_selection()
        if not template or template.get('is_builtin'):
            self.app_instance.notify("Cannot edit built-in templates", severity="warning")
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
        if not self.selected_template_record_id:
            return
        
        template = self._record_for_selection()
        if not template:
            return
        
        # Create a copy with modified name
        duplicate = template.copy()
        duplicate['name'] = f"{template['name']} (Copy)"
        duplicate['is_builtin'] = False  # Duplicates are always custom
        duplicate.pop('record_id', None)
        duplicate.pop('backing_id', None)
        
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
    async def handle_delete_template(self) -> None:
        """Handle delete template button press."""
        if not self.selected_template_record_id:
            return
        
        template = self._record_for_selection()
        if not template:
            return
        
        if template.get('is_builtin'):
            self.app_instance.notify("Cannot delete built-in templates", severity="warning")
            return
        
        # Show confirmation dialog
        from ..Widgets.delete_confirmation_dialog import create_delete_confirmation
        dialog = create_delete_confirmation(
            item_type="Template",
            item_name=template['name'],
            additional_warning="This template will no longer be available for creating new items."
        )
        
        confirmed = await self.app_instance.push_screen_wait(dialog)
        if confirmed:
            await self.delete_template(template["backing_template_name"])
    
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
        if self._runtime_backend() != "local":
            self.app_instance.notify("Server template export is not available yet", severity="warning")
            return
        template = self._record_for_selection()
        if template is None or not self.chunking_service:
            return
        
        try:
            # Export using the service
            export_data = self.chunking_service.export_template(int(template["backing_id"]))
            
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
        if self._runtime_backend() != "local":
            self.app_instance.notify("Server template import is not available yet", severity="warning")
            return
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
            self.app_instance.run_worker(self.refresh_templates(), exclusive=True)
            
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
    
    async def _on_template_saved(self) -> None:
        """Callback when a template is saved from the editor."""
        await self.refresh_templates()
    
    async def delete_template(self, template_name: str) -> None:
        """Delete a template through the active backend seam."""
        try:
            scope = self._scope_service()
            if scope is None:
                raise ChunkingTemplateError("RAG admin scope service not available")
            await scope.delete_template(
                template_name,
                mode=self._runtime_backend(),
            )
            
            self.app_instance.notify("Template deleted successfully", severity="information")
            self.selected_template_record_id = None
            await self.refresh_templates()
            
        except SystemTemplateError:
            self.app_instance.notify("Cannot delete built-in templates", severity="warning")
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
