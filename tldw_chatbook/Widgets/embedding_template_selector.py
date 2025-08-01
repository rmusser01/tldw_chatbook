# tldw_chatbook/Widgets/embedding_template_selector.py
# Template selection widget for embedding configuration
#
# Imports
from __future__ import annotations
from typing import Optional, List, Dict, Any

# Third-party imports
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.widgets import Static, Button, RadioSet, RadioButton, Label
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual.screen import ModalScreen
from loguru import logger

# Local imports
from ..Utils.embedding_templates import EmbeddingTemplateManager, EmbeddingTemplate, TemplateCategory

# Configure logger
logger = logger.bind(module="embedding_template_selector")


class EmbeddingTemplateSelected(Message):
    """Message sent when a template is selected."""
    
    def __init__(self, template: EmbeddingTemplate) -> None:
        self.template = template
        super().__init__()


class EmbeddingTemplateCard(Widget):
    """Card widget displaying template information."""
    
    DEFAULT_CLASSES = "embedding-template-card"
    
    def __init__(self, template: EmbeddingTemplate, **kwargs):
        super().__init__(**kwargs)
        self.template = template
        
    def compose(self) -> ComposeResult:
        """Compose the template card."""
        with Container(classes="embedding-template-card-container"):
            # Header
            with Horizontal(classes="embedding-template-card-header"):
                yield Static(self.template.name, classes="embedding-template-card-title")
                yield Static(self.template.category.value.replace("_", " ").title(), classes="embedding-template-card-category")
            
            # Description
            yield Static(self.template.description, classes="embedding-template-card-description")
            
            # Recommended for
            if self.template.recommended_for:
                yield Label("Recommended for:", classes="embedding-template-card-label")
                with Container(classes="embedding-template-card-list"):
                    for item in self.template.recommended_for[:3]:
                        yield Static(f"• {item}", classes="embedding-template-card-item")
            
            # Action button
            yield Button(
                "Use This Template",
                id=f"select-{self.template.id}",
                classes="embedding-template-card-button"
            )


class EmbeddingTemplateSelectorDialog(ModalScreen):
    """Modal dialog for selecting embedding configuration templates."""
    
    BINDINGS = [("escape", "cancel", "Cancel")]
    
    def __init__(self, template_manager: Optional[EmbeddingTemplateManager] = None):
        super().__init__()
        self.template_manager = template_manager or EmbeddingTemplateManager()
        self.selected_category: reactive[TemplateCategory] = reactive(TemplateCategory.QUICK_START)
        
    def compose(self) -> ComposeResult:
        """Compose the template selector dialog."""
        with Container(id="embedding-template-selector-container"):
            yield Label("Select Embedding Configuration Template", id="embedding-template-selector-title")
            
            with Horizontal(id="embedding-template-selector-content"):
                # Category selector
                with Vertical(id="embedding-template-category-pane"):
                    yield Label("Categories", classes="sidebar-title")
                    
                    category_options = [
                        (cat.value.replace("_", " ").title(), cat.value)
                        for cat in TemplateCategory
                    ]
                    
                    yield RadioSet(
                        *category_options,
                        id="embedding-template-category-set",
                        value=TemplateCategory.QUICK_START.value
                    )
                
                # Template grid
                with Container(id="embedding-template-grid-container"):
                    yield Grid(id="embedding-template-grid", classes="embedding-template-grid")
            
            # Action buttons
            with Horizontal(id="embedding-template-selector-actions"):
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Browse All", id="browse-all", variant="primary")
    
    def on_mount(self) -> None:
        """Initialize with templates when mounted."""
        self._update_template_grid()
    
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle category selection change."""
        if event.radio_set.id == "embedding-template-category-set":
            try:
                self.selected_category = TemplateCategory(event.value)
                self._update_template_grid()
            except ValueError:
                logger.error(f"Invalid category value: {event.value}")
    
    def _update_template_grid(self) -> None:
        """Update the template grid based on selected category."""
        grid = self.query_one("#embedding-template-grid", Grid)
        grid.clear()
        
        # Get templates for category
        templates = self.template_manager.get_templates_by_category(self.selected_category)
        
        # Create cards
        for template in templates:
            card = EmbeddingTemplateCard(template, id=f"card-{template.id}")
            grid.mount(card)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "browse-all":
            # Show all templates
            grid = self.query_one("#embedding-template-grid", Grid)
            grid.clear()
            
            for template in self.template_manager.get_all_templates():
                card = EmbeddingTemplateCard(template, id=f"card-{template.id}")
                grid.mount(card)
        elif event.button.id and event.button.id.startswith("select-"):
            # Template selected
            template_id = event.button.id.replace("select-", "")
            template = self.template_manager.get_template(template_id)
            if template:
                self.dismiss(template)
    
    def action_cancel(self) -> None:
        """Cancel the dialog."""
        self.dismiss(None)


class EmbeddingTemplateQuickSelect(Widget):
    """Quick template selector widget for embedding forms."""
    
    DEFAULT_CLASSES = "embedding-template-quick-select"
    
    def __init__(self, template_manager: Optional[EmbeddingTemplateManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.template_manager = template_manager or EmbeddingTemplateManager()
        self.current_template: reactive[Optional[str]] = reactive(None)
        
    def compose(self) -> ComposeResult:
        """Compose the quick select widget."""
        with Horizontal(classes="embedding-template-quick-select-container"):
            yield Label("Template:", classes="embedding-template-select-label")
            
            # Quick access buttons for common templates
            yield Button("Quick Start", id="template-quick_local", classes="embedding-template-quick-button")
            yield Button("High Quality", id="template-high_quality_local", classes="embedding-template-quick-button")
            yield Button("Performance", id="template-balanced_performance", classes="embedding-template-quick-button")
            
            # Browse all button
            yield Button("Browse All...", id="template-browse", classes="embedding-template-browse-button", variant="primary")
            
            # Current template indicator
            yield Static("", id="current-template-name", classes="current-template-name")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id and event.button.id.startswith("template-"):
            if event.button.id == "template-browse":
                # Open full browser
                self.app.push_screen(
                    EmbeddingTemplateSelectorDialog(self.template_manager),
                    self._handle_template_selection
                )
            else:
                # Quick select
                template_id = event.button.id.replace("template-", "")
                template = self.template_manager.get_template(template_id)
                if template:
                    self._apply_template(template)
    
    def _handle_template_selection(self, template: Optional[EmbeddingTemplate]) -> None:
        """Handle template selection from dialog."""
        if template:
            self._apply_template(template)
    
    def _apply_template(self, template: EmbeddingTemplate) -> None:
        """Apply the selected template."""
        self.current_template = template.id
        
        # Update display
        name_display = self.query_one("#current-template-name", Static)
        name_display.update(f"Using: {template.name}")
        
        # Send message
        self.post_message(EmbeddingTemplateSelected(template))
        
        logger.info(f"Applied template: {template.name}")
    
    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """Get the configuration from the current template."""
        if self.current_template:
            template = self.template_manager.get_template(self.current_template)
            if template:
                return template.config.copy()
        return None