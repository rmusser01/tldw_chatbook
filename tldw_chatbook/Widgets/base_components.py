# base_components.py
# Description: Reusable base components for consistent UI patterns
#
"""
Base Component Library
---------------------

Provides reusable UI components for building consistent interfaces:
- SectionContainer: Styled containers with optional collapsibility
- ActionButtonRow: Horizontal button groups with consistent spacing
- StatusDisplay: Status messages with contextual styling
- ConfigurationForm: Form layouts for configuration inputs
- NavigationButton: Styled navigation buttons with active states
"""

from typing import List, Optional, Tuple, Callable, Any, Dict
from dataclasses import dataclass
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static, Input, Label, Select
from textual.reactive import reactive
from textual import on
from loguru import logger

# Configure logger
logger = logger.bind(module="base_components")


@dataclass
class ButtonConfig:
    """Configuration for a button in ActionButtonRow."""
    label: str
    id: str
    variant: str = "default"
    disabled: bool = False
    tooltip: Optional[str] = None


@dataclass
class FormField:
    """Configuration for a form field."""
    name: str
    label: str
    field_type: str = "text"  # text, number, select, password
    placeholder: str = ""
    options: Optional[List[Tuple[str, Any]]] = None
    default_value: Any = ""
    required: bool = False
    validator: Optional[Callable[[Any], bool]] = None


class SectionContainer(Container):
    """
    A styled container for UI sections with optional collapsibility.
    
    Features:
    - Consistent styling with title
    - Optional collapsible behavior
    - Customizable through CSS classes
    """
    
    DEFAULT_CSS = """
    SectionContainer {
        background: $panel;
        margin: 0 0 1 0;
        padding: 1;
        border: round $secondary;
    }
    
    .section-header {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
        width: 100%;
    }
    
    .section-title {
        text-style: bold;
        width: 1fr;
        color: $primary;
    }
    
    .collapse-button {
        width: 3;
        min-width: 3;
        height: 3;
        background: transparent;
        border: none;
    }
    
    .collapse-button:hover {
        background: $primary 20%;
    }
    
    .section-content {
        width: 100%;
    }
    
    .section-content.collapsed {
        display: none;
    }
    """
    
    _collapsed = reactive(False)
    
    def __init__(
        self,
        title: str,
        *children,
        collapsible: bool = False,
        initially_collapsed: bool = False,
        classes: str = "",
        id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a SectionContainer.
        
        Args:
            title: The section title
            *children: Child widgets to include in the section
            collapsible: Whether the section can be collapsed
            initially_collapsed: Initial collapsed state
            classes: Additional CSS classes
            id: Widget ID
            **kwargs: Additional arguments passed to Container
        """
        super().__init__(classes=f"section-container {classes}", id=id, **kwargs)
        self.title = title
        self._children = children
        self.collapsible = collapsible
        self._collapsed = initially_collapsed
    
    def compose(self) -> ComposeResult:
        """Build the section structure."""
        with Container(classes="section-header"):
            yield Static(self.title, classes="section-title")
            if self.collapsible:
                yield Button(
                    "▼" if not self._collapsed else "▶",
                    classes="collapse-button",
                    id=f"{self.id}-collapse-btn" if self.id else None
                )
        
        content_classes = "section-content"
        if self._collapsed:
            content_classes += " collapsed"
        
        with Container(classes=content_classes, id=f"{self.id}-content" if self.id else None):
            yield from self._children
    
    def watch__collapsed(self, collapsed: bool) -> None:
        """Update UI when collapsed state changes."""
        try:
            # Update button text
            if self.collapsible:
                btn = self.query_one(".collapse-button", Button)
                btn.label = "▶" if collapsed else "▼"
            
            # Toggle content visibility
            content = self.query_one(".section-content")
            if collapsed:
                content.add_class("collapsed")
            else:
                content.remove_class("collapsed")
        except Exception as e:
            logger.warning(f"Error updating collapsed state: {e}")
    
    @on(Button.Pressed, ".collapse-button")
    def toggle_collapse(self) -> None:
        """Toggle the collapsed state."""
        self._collapsed = not self._collapsed


class ActionButtonRow(Horizontal):
    """
    A horizontal row of action buttons with consistent styling.
    
    Features:
    - Automatic spacing between buttons
    - Support for different button variants
    - Disabled state handling
    """
    
    DEFAULT_CSS = """
    ActionButtonRow {
        height: auto;
        margin: 0 0 1 0;
    }
    
    ActionButtonRow Button {
        margin: 0 1 0 0;
        min-width: 15;
        height: 3;
    }
    
    ActionButtonRow Button:last-child {
        margin-right: 0;
    }
    """
    
    def __init__(
        self,
        buttons: List[ButtonConfig],
        classes: str = "",
        id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize an ActionButtonRow.
        
        Args:
            buttons: List of button configurations
            classes: Additional CSS classes
            id: Widget ID
            **kwargs: Additional arguments passed to Horizontal
        """
        super().__init__(classes=f"action-button-row {classes}", id=id, **kwargs)
        self.buttons = buttons
    
    def compose(self) -> ComposeResult:
        """Build the button row."""
        for config in self.buttons:
            yield Button(
                config.label,
                id=config.id,
                variant=config.variant,
                disabled=config.disabled,
                tooltip=config.tooltip,
                classes="action-button"
            )


class StatusDisplay(Static):
    """
    Displays status messages with contextual styling.
    
    Features:
    - Different styles for info, success, warning, error
    - Automatic styling based on status type
    - Optional auto-hide after timeout
    """
    
    DEFAULT_CSS = """
    StatusDisplay {
        padding: 0 1;
        height: 3;
        margin: 1 0;
        width: 100%;
        text-align: center;
        opacity: 0;
        transition: opacity 200ms;
    }
    
    StatusDisplay.visible {
        opacity: 1;
    }
    
    StatusDisplay.info {
        background: $primary 20%;
        color: $text;
    }
    
    StatusDisplay.success {
        background: $success 20%;
        color: $text;
    }
    
    StatusDisplay.warning {
        background: $warning 20%;
        color: $text;
    }
    
    StatusDisplay.error {
        background: $error 20%;
        color: $text;
    }
    """
    
    status_text = reactive("")
    status_type = reactive("info")
    
    def __init__(
        self,
        initial_text: str = "",
        initial_type: str = "info",
        auto_hide_seconds: Optional[float] = None,
        id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a StatusDisplay.
        
        Args:
            initial_text: Initial status text
            initial_type: Initial status type (info, success, warning, error)
            auto_hide_seconds: Auto-hide after this many seconds
            id: Widget ID
            **kwargs: Additional arguments passed to Static
        """
        super().__init__(initial_text, id=id, **kwargs)
        self.status_text = initial_text
        self.status_type = initial_type
        self.auto_hide_seconds = auto_hide_seconds
        self._hide_timer = None
        
        if initial_text:
            self.add_class("visible")
    
    def watch_status_text(self, text: str) -> None:
        """Update display when text changes."""
        self.update(text)
        if text:
            self.add_class("visible")
            if self.auto_hide_seconds:
                self._start_hide_timer()
        else:
            self.remove_class("visible")
    
    def watch_status_type(self, status_type: str) -> None:
        """Update styling when type changes."""
        self.remove_class("info", "success", "warning", "error")
        self.add_class(status_type)
    
    def set_status(self, text: str, status_type: str = "info") -> None:
        """
        Set status message and type.
        
        Args:
            text: Status message text
            status_type: Type of status (info, success, warning, error)
        """
        self.status_type = status_type
        self.status_text = text
    
    def clear(self) -> None:
        """Clear the status message."""
        self.status_text = ""
    
    def _start_hide_timer(self) -> None:
        """Start auto-hide timer."""
        if self._hide_timer:
            self._hide_timer.cancel()
        
        self._hide_timer = self.set_timer(
            self.auto_hide_seconds,
            lambda: self.clear()
        )


class ConfigurationForm(Container):
    """
    A form container for configuration inputs.
    
    Features:
    - Automatic field layout
    - Built-in validation
    - Consistent styling
    """
    
    DEFAULT_CSS = """
    ConfigurationForm {
        layout: vertical;
        padding: 1;
        background: $surface;
        border: round $secondary;
    }
    
    .form-field {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
        margin: 0 0 1 0;
        height: 3;
    }
    
    .form-field:last-child {
        margin-bottom: 0;
    }
    
    .form-label {
        text-align: right;
        padding: 0 1;
        color: $text-muted;
    }
    
    .form-label.required {
        color: $text-muted;
    }
    
    .form-input {
        width: 100%;
    }
    
    .form-error {
        color: $error;
        text-style: italic;
        height: 2;
        padding: 0 1;
    }
    """
    
    def __init__(
        self,
        fields: List[FormField],
        submit_button: Optional[ButtonConfig] = None,
        classes: str = "",
        id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a ConfigurationForm.
        
        Args:
            fields: List of form field configurations
            submit_button: Optional submit button configuration
            classes: Additional CSS classes
            id: Widget ID
            **kwargs: Additional arguments passed to Container
        """
        super().__init__(classes=f"configuration-form {classes}", id=id, **kwargs)
        self.fields = fields
        self.submit_button = submit_button
        self._field_widgets: Dict[str, Any] = {}
        self._error_widgets: Dict[str, Static] = {}
    
    def compose(self) -> ComposeResult:
        """Build the form structure."""
        for field in self.fields:
            with Container(classes="form-field"):
                # Label
                label_classes = "form-label"
                if field.required:
                    label_classes += " required"
                    label_text = f"{field.label} *"
                else:
                    label_text = field.label
                yield Label(label_text, classes=label_classes)
                
                # Input widget
                if field.field_type == "select" and field.options:
                    widget = Select(
                        field.options,
                        value=field.default_value,
                        id=f"{self.id}-{field.name}" if self.id else field.name,
                        classes="form-input"
                    )
                else:
                    widget = Input(
                        value=str(field.default_value),
                        placeholder=field.placeholder,
                        id=f"{self.id}-{field.name}" if self.id else field.name,
                        classes="form-input",
                        type=field.field_type
                    )
                yield widget
                
                # Error display
                error_widget = Static("", classes="form-error", id=f"{field.name}-error")
                yield error_widget
        
        # Submit button
        if self.submit_button:
            yield Button(
                self.submit_button.label,
                id=self.submit_button.id,
                variant=self.submit_button.variant,
                disabled=self.submit_button.disabled,
                classes="form-submit"
            )
    
    def on_mount(self) -> None:
        """Cache references to form widgets."""
        for field in self.fields:
            field_id = f"{self.id}-{field.name}" if self.id else field.name
            try:
                self._field_widgets[field.name] = self.query_one(f"#{field_id}")
                self._error_widgets[field.name] = self.query_one(f"#{field.name}-error")
            except Exception as e:
                logger.warning(f"Could not find widget for field {field.name}: {e}")
    
    def get_values(self) -> Dict[str, Any]:
        """
        Get all form values.
        
        Returns:
            Dictionary of field names to values
        """
        values = {}
        for field in self.fields:
            widget = self._field_widgets.get(field.name)
            if widget:
                if isinstance(widget, Select):
                    values[field.name] = widget.value
                else:
                    values[field.name] = widget.value
        return values
    
    def set_values(self, values: Dict[str, Any]) -> None:
        """
        Set form values.
        
        Args:
            values: Dictionary of field names to values
        """
        for field_name, value in values.items():
            widget = self._field_widgets.get(field_name)
            if widget:
                if isinstance(widget, Select):
                    widget.value = value
                else:
                    widget.value = str(value)
    
    def validate(self) -> bool:
        """
        Validate all form fields.
        
        Returns:
            True if all fields are valid
        """
        all_valid = True
        
        for field in self.fields:
            widget = self._field_widgets.get(field.name)
            error_widget = self._error_widgets.get(field.name)
            
            if not widget or not error_widget:
                continue
            
            value = widget.value if hasattr(widget, 'value') else ""
            
            # Required field check
            if field.required and not value:
                error_widget.update("This field is required")
                all_valid = False
                continue
            
            # Custom validator
            if field.validator and value:
                try:
                    if not field.validator(value):
                        error_widget.update("Invalid value")
                        all_valid = False
                        continue
                except Exception as e:
                    error_widget.update(f"Validation error: {e}")
                    all_valid = False
                    continue
            
            # Clear error if valid
            error_widget.update("")
        
        return all_valid
    
    def clear_errors(self) -> None:
        """Clear all error messages."""
        for error_widget in self._error_widgets.values():
            error_widget.update("")


class NavigationButton(Button):
    """
    A styled navigation button with active state support.
    
    Features:
    - Visual indication of active state
    - Consistent navigation styling
    - Optional icon support
    """
    
    DEFAULT_CSS = """
    NavigationButton {
        width: 100%;
        height: 3;
        margin: 0 0 1 0;
        text-align: left;
        background: transparent;
        color: $text-muted;
        border: none;
        padding: 0 1;
    }
    
    NavigationButton:hover {
        background: $accent 30%;
        color: $text;
    }
    
    NavigationButton.active {
        background: $accent;
        color: $text;
        text-style: bold;
        border-left: thick $primary;
    }
    
    NavigationButton:focus {
        text-style: bold;
    }
    """
    
    is_active = reactive(False)
    
    def __init__(
        self,
        label: str,
        icon: Optional[str] = None,
        active: bool = False,
        **kwargs
    ):
        """
        Initialize a NavigationButton.
        
        Args:
            label: Button label
            icon: Optional icon character
            active: Initial active state
            **kwargs: Additional arguments passed to Button
        """
        full_label = f"{icon} {label}" if icon else label
        super().__init__(full_label, **kwargs)
        self.is_active = active
    
    def watch_is_active(self, active: bool) -> None:
        """Update styling when active state changes."""
        if active:
            self.add_class("active")
        else:
            self.remove_class("active")
    
    def activate(self) -> None:
        """Set this button as active."""
        self.is_active = True
    
    def deactivate(self) -> None:
        """Set this button as inactive."""
        self.is_active = False


# Utility functions for common patterns

def create_button_row(
    *buttons: Tuple[str, str, str],
    row_id: Optional[str] = None
) -> ActionButtonRow:
    """
    Convenience function to create a button row.
    
    Args:
        *buttons: Tuples of (label, id, variant)
        row_id: Optional ID for the row
    
    Returns:
        Configured ActionButtonRow
    """
    configs = [
        ButtonConfig(label, id, variant)
        for label, id, variant in buttons
    ]
    return ActionButtonRow(configs, id=row_id)


def create_form_field(
    name: str,
    label: str,
    field_type: str = "text",
    **kwargs
) -> FormField:
    """
    Convenience function to create a form field.
    
    Args:
        name: Field name
        label: Field label
        field_type: Type of field
        **kwargs: Additional field configuration
    
    Returns:
        Configured FormField
    """
    return FormField(name, label, field_type, **kwargs)


# Export all components
__all__ = [
    'SectionContainer',
    'ActionButtonRow',
    'StatusDisplay',
    'ConfigurationForm',
    'NavigationButton',
    'ButtonConfig',
    'FormField',
    'create_button_row',
    'create_form_field',
]