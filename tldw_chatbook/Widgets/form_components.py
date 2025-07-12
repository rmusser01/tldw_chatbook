# tldw_chatbook/Widgets/form_components.py
"""
Reusable form components for standardized UI layouts.
"""

from typing import Optional, List, Tuple, Any
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, Input, TextArea, Select, Checkbox, Button, Collapsible


def create_form_field(
    label: str,
    field_id: str,
    field_type: str = "input",
    placeholder: str = "",
    default_value: Any = None,
    options: Optional[List[Tuple[str, Any]]] = None,
    required: bool = False,
    **kwargs
) -> ComposeResult:
    """
    Create a standardized form field with label.
    
    Args:
        label: The label text for the field
        field_id: The ID for the field widget
        field_type: Type of field ("input", "textarea", "select", "checkbox")
        placeholder: Placeholder text for input fields
        default_value: Default value for the field
        options: Options for select fields
        required: Whether the field is required
        **kwargs: Additional arguments passed to the widget
    """
    # Add asterisk for required fields
    label_text = f"{label}*" if required else label
    
    if field_type == "checkbox":
        # Checkbox has different layout - label comes after
        yield Checkbox(
            label_text,
            value=bool(default_value) if default_value is not None else False,
            id=field_id,
            classes="form-checkbox",
            **kwargs
        )
    else:
        # Standard fields have label above
        yield Label(f"{label_text}:", classes="form-label")
        
        if field_type == "input":
            yield Input(
                value=str(default_value) if default_value is not None else "",
                placeholder=placeholder,
                id=field_id,
                classes="form-input",
                **kwargs
            )
        elif field_type == "textarea":
            yield TextArea(
                str(default_value) if default_value is not None else "",
                id=field_id,
                classes="form-textarea",
                **kwargs
            )
        elif field_type == "select":
            if options is None:
                options = []
            yield Select(
                options,
                id=field_id,
                classes="form-select",
                value=default_value,
                **kwargs
            )


def create_form_row(*fields) -> ComposeResult:
    """
    Create a horizontal row of form fields.
    
    Args:
        *fields: Tuples of field arguments to pass to create_form_field
    """
    with Horizontal(classes="form-row"):
        for field_args in fields:
            with Vertical(classes="form-col"):
                yield from create_form_field(*field_args)


def create_form_section(
    title: str,
    fields: List[tuple],
    collapsible: bool = False,
    collapsed: bool = True,
    section_id: Optional[str] = None
) -> ComposeResult:
    """
    Create a form section with optional collapsibility.
    
    Args:
        title: Section title
        fields: List of field configurations
        collapsible: Whether section should be collapsible
        collapsed: Initial collapsed state
        section_id: Optional ID for the section
    """
    if collapsible:
        with Collapsible(
            title=title,
            collapsed=collapsed,
            id=section_id,
            classes="form-section-collapsible"
        ):
            for field_args in fields:
                yield from create_form_field(*field_args)
    else:
        yield Label(title, classes="form-section-title")
        for field_args in fields:
            yield from create_form_field(*field_args)


def create_button_group(
    buttons: List[Tuple[str, str, str]],
    alignment: str = "left"
) -> ComposeResult:
    """
    Create a group of buttons.
    
    Args:
        buttons: List of (label, id, variant) tuples
        alignment: Button alignment ("left", "center", "right")
    """
    with Horizontal(classes=f"button-group button-group-{alignment}"):
        for label, button_id, variant in buttons:
            yield Button(
                label,
                id=button_id,
                variant=variant,
                classes="form-button"
            )


def create_status_area(
    area_id: str,
    label: str = "Status:",
    initial_content: str = "",
    min_height: int = 5,
    max_height: int = 15
) -> ComposeResult:
    """
    Create a standardized status area.
    
    Args:
        area_id: ID for the status area
        label: Label for the status area
        initial_content: Initial content to display
        min_height: Minimum height
        max_height: Maximum height
    """
    yield Label(label, classes="status-label")
    area = TextArea(
        initial_content,
        id=area_id,
        read_only=True,
        classes="status-area"
    )
    # Apply dynamic styling
    area.styles.min_height = min_height
    area.styles.max_height = max_height
    area.styles.height = "auto"
    yield area