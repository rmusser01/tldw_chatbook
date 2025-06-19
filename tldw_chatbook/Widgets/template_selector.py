# template_selector.py
# Description: Template selection widget for evaluation tasks
#
"""
Template Selector Widget
------------------------

Provides an organized interface for browsing and selecting evaluation templates:
- Categorized template display
- Template preview and description
- Quick template creation
- Search and filtering capabilities
"""

from typing import Dict, List, Any, Optional, Callable
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Label, Input, ListView, ListItem, Static, 
    Collapsible, Tree, Tabs, TabPane
)
from loguru import logger

class TemplatePreviewWidget(Container):
    """Widget for displaying template preview information."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_template = None
    
    def compose(self) -> ComposeResult:
        yield Label("Template Preview", classes="preview-title")
        yield Static("Select a template to see details", id="template-description", classes="template-description")
        
        with Collapsible(title="Configuration Details", collapsed=True, id="config-details"):
            yield Static("", id="config-display", classes="config-display")
        
        with Horizontal(classes="preview-actions"):
            yield Button("Create Task", id="create-task-btn", variant="primary", disabled=True)
            yield Button("Export Template", id="export-template-btn", disabled=True)
    
    def update_preview(self, template: Dict[str, Any]):
        """Update the preview with template information."""
        self.current_template = template
        
        try:
            # Update description
            description = f"**{template.get('name', 'Unknown')}**\n\n"
            description += template.get('description', 'No description available.')
            description += f"\n\n**Category:** {template.get('category', 'General')}"
            description += f"\n**Difficulty:** {template.get('difficulty', 'Unknown')}"
            description += f"\n**Task Type:** {template.get('task_type', 'Unknown')}"
            
            desc_widget = self.query_one("#template-description")
            desc_widget.update(description)
            
            # Update configuration details
            config_text = "**Configuration:**\n"
            for key, value in template.items():
                if key not in ['name', 'description', 'category', 'difficulty']:
                    config_text += f"- {key}: {value}\n"
            
            config_widget = self.query_one("#config-display")
            config_widget.update(config_text)
            
            # Enable buttons
            self.query_one("#create-task-btn").disabled = False
            self.query_one("#export-template-btn").disabled = False
            
        except Exception as e:
            logger.error(f"Error updating template preview: {e}")
    
    def clear_preview(self):
        """Clear the preview display."""
        self.current_template = None
        
        try:
            self.query_one("#template-description").update("Select a template to see details")
            self.query_one("#config-display").update("")
            self.query_one("#create-task-btn").disabled = True
            self.query_one("#export-template-btn").disabled = True
        except:
            pass

class TemplateListWidget(Container):
    """Widget for displaying template lists organized by category."""
    
    def __init__(self, 
                 templates: List[Dict[str, Any]], 
                 on_template_selected: Optional[Callable[[Dict[str, Any]], None]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.templates = templates
        self.on_template_selected = on_template_selected
        self.templates_by_category = self._organize_by_category()
    
    def _organize_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Organize templates by category."""
        categories = {}
        for template in self.templates:
            category = template.get('category', 'General')
            if category not in categories:
                categories[category] = []
            categories[category].append(template)
        return categories
    
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search templates...", id="template-search")
        
        with Tabs(id="category-tabs"):
            # Create tab for each category
            for category, templates in self.templates_by_category.items():
                with TabPane(category.title(), id=f"tab-{category}"):
                    with ListView(id=f"list-{category}"):
                        for template in templates:
                            yield ListItem(
                                Label(template.get('display_name', template.get('name', 'Unknown'))),
                                name=template.get('name', ''),
                                id=f"template-{template.get('name', '')}"
                            )
    
    @on(Input.Changed, "#template-search")
    def handle_search(self, event: Input.Changed):
        """Handle template search."""
        search_term = event.value.lower()
        
        # Filter and update template lists
        for category, templates in self.templates_by_category.items():
            try:
                list_widget = self.query_one(f"#list-{category}")
                list_widget.clear()
                
                filtered_templates = [
                    t for t in templates 
                    if search_term in t.get('name', '').lower() or 
                       search_term in t.get('description', '').lower()
                ]
                
                for template in filtered_templates:
                    list_widget.append(
                        ListItem(
                            Label(template.get('display_name', template.get('name', 'Unknown'))),
                            name=template.get('name', ''),
                            id=f"template-{template.get('name', '')}"
                        )
                    )
            except Exception as e:
                logger.error(f"Error filtering templates for category {category}: {e}")
    
    @on(ListView.Selected)
    def handle_template_selected(self, event: ListView.Selected):
        """Handle template selection."""
        if event.item and event.item.name:
            template_name = event.item.name
            
            # Find the template
            template = None
            for templates in self.templates_by_category.values():
                for t in templates:
                    if t.get('name') == template_name:
                        template = t
                        break
                if template:
                    break
            
            if template and self.on_template_selected:
                self.on_template_selected(template)

class TemplateSelectorDialog(ModalScreen):
    """Modal dialog for selecting evaluation templates."""
    
    def __init__(self, 
                 callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.selected_template = None
        self.templates = []
    
    def on_mount(self):
        """Load templates when dialog is mounted."""
        self._load_templates()
    
    def _load_templates(self):
        """Load available templates."""
        try:
            from ..App_Functions.Evals.eval_templates import get_eval_templates
            template_manager = get_eval_templates()
            self.templates = template_manager.list_templates()
            
            # Update the template list widget
            try:
                template_list = self.query_one("#template-list")
                template_list.templates = self.templates
                template_list.templates_by_category = template_list._organize_by_category()
                # Refresh the display
                self.refresh()
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self.templates = []
    
    def compose(self) -> ComposeResult:
        with Container(classes="template-selector-dialog"):
            yield Label("Select Evaluation Template", classes="dialog-title")
            
            with Horizontal(classes="template-content"):
                # Left side - template list
                with Vertical(classes="template-list-container"):
                    yield TemplateListWidget(
                        templates=self.templates,
                        on_template_selected=self._on_template_selected,
                        id="template-list"
                    )
                
                # Right side - preview
                with Vertical(classes="template-preview-container"):
                    yield TemplatePreviewWidget(id="template-preview")
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Select Template", id="select-button", variant="primary", disabled=True)
    
    def _on_template_selected(self, template: Dict[str, Any]):
        """Handle template selection."""
        self.selected_template = template
        
        # Update preview
        try:
            preview = self.query_one("#template-preview")
            preview.update_preview(template)
        except:
            pass
        
        # Enable select button
        try:
            self.query_one("#select-button").disabled = False
        except:
            pass
    
    @on(Button.Pressed, "#create-task-btn")
    def handle_create_task(self):
        """Handle create task from template."""
        if self.selected_template:
            # Close dialog and return template for task creation
            if self.callback:
                self.callback(self.selected_template)
            self.dismiss(self.selected_template)
    
    @on(Button.Pressed, "#export-template-btn")
    def handle_export_template(self):
        """Handle template export."""
        if self.selected_template:
            # This would open an export dialog
            self.app.notify("Template export not yet implemented", severity="information")
    
    @on(Button.Pressed, "#select-button")
    def handle_select(self):
        """Handle select button press."""
        if self.selected_template:
            if self.callback:
                self.callback(self.selected_template)
            self.dismiss(self.selected_template)
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Handle cancel button press."""
        if self.callback:
            self.callback(None)
        self.dismiss(None)

class QuickTemplateSelector(Container):
    """Quick template selector for inline use."""
    
    def __init__(self, 
                 category_filter: Optional[str] = None,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.category_filter = category_filter
        self.callback = callback
        self.templates = []
    
    def on_mount(self):
        """Load templates when widget is mounted."""
        self._load_templates()
    
    def _load_templates(self):
        """Load and filter templates."""
        try:
            from ..App_Functions.Evals.eval_templates import get_eval_templates
            template_manager = get_eval_templates()
            all_templates = template_manager.list_templates()
            
            if self.category_filter:
                self.templates = [
                    t for t in all_templates 
                    if t.get('category', '').lower() == self.category_filter.lower()
                ]
            else:
                self.templates = all_templates
                
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self.templates = []
    
    def compose(self) -> ComposeResult:
        yield Label("Quick Templates", classes="section-title")
        
        with Horizontal(classes="quick-template-buttons"):
            # Show first few templates as quick buttons
            for template in self.templates[:6]:  # Limit to 6 quick buttons
                yield Button(
                    template.get('display_name', template.get('name', 'Unknown')),
                    name=template.get('name', ''),
                    classes="template-quick-button"
                )
            
            yield Button("More Templates...", id="more-templates-btn", classes="more-templates-button")
    
    @on(Button.Pressed, ".template-quick-button")
    def handle_quick_template(self, event: Button.Pressed):
        """Handle quick template button press."""
        template_name = event.button.name
        
        # Find the template
        template = None
        for t in self.templates:
            if t.get('name') == template_name:
                template = t
                break
        
        if template and self.callback:
            self.callback(template)
    
    @on(Button.Pressed, "#more-templates-btn")
    def handle_more_templates(self):
        """Open the full template selector dialog."""
        dialog = TemplateSelectorDialog(callback=self.callback)
        self.app.push_screen(dialog)