# Textual Media Ingestion UI Patterns

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Principles](#architecture-principles)
3. [Form Layout Best Practices](#form-layout-best-practices)
4. [Input Visibility Solutions](#input-visibility-solutions)
5. [Responsive Design Patterns](#responsive-design-patterns)
6. [Progressive Disclosure](#progressive-disclosure)
7. [Validation and Error Handling](#validation-and-error-handling)
8. [File Selection Patterns](#file-selection-patterns)
9. [Status Dashboard Design](#status-dashboard-design)
10. [Accessibility Considerations](#accessibility-considerations)

## Introduction

This guide provides proven patterns and best practices for creating media ingestion interfaces using the Textual framework. Based on lessons learned from the existing tldw_chatbook ingestion UI, these patterns solve common problems like invisible inputs, broken scrolling, and poor user experience.

### Key Problems Solved
- **Invisible input widgets** - Proper CSS height and width specifications
- **Double scrolling issues** - Correct container nesting patterns
- **Poor progressive disclosure** - Simple/advanced mode patterns
- **Inconsistent layouts** - Standardized form components
- **Bad responsive behavior** - Terminal size adaptation patterns

## Architecture Principles

### Single Source of Truth
All form state should be managed in one place using reactive attributes:

```python
class MediaIngestWindow(Container):
    # Single source of truth for form data
    form_data = reactive({})
    validation_errors = reactive({})
    processing_state = reactive("idle")  # idle, processing, complete, error
    
    def get_field_value(self, field_id: str, default=""):
        return self.form_data.get(field_id, default)
    
    def set_field_value(self, field_id: str, value: str):
        self.form_data = {**self.form_data, field_id: value}
        self.validate_field(field_id, value)
```

### Clean Separation of Concerns

```python
# UI Components (presentation)
class FileSelector(Container): pass
class MetadataForm(Container): pass
class AdvancedOptions(Container): pass

# Data Layer (business logic)
class IngestFormValidator:
    def validate_field(self, field_id: str, value: str) -> Optional[str]: pass
    def validate_form(self, form_data: dict) -> dict: pass

class IngestProcessor:
    async def process_media(self, form_data: dict) -> AsyncIterator[StatusUpdate]: pass

# Main Window (orchestration)
class VideoIngestWindow(Container):
    def __init__(self):
        self.validator = IngestFormValidator()
        self.processor = IngestProcessor()
```

### Component-Based Design

```python
# Reusable components across media types
from .components import (
    FileSelector,
    BasicMetadataForm,
    ProcessingOptionsForm,
    StatusDashboard,
    ProcessButton
)

class MediaIngestWindow(Container):
    """Base class for all media ingestion windows."""
    
    def compose(self) -> ComposeResult:
        yield StatusDashboard(id="status")
        yield FileSelector(id="files")
        yield BasicMetadataForm(id="metadata")
        yield self.create_media_specific_options()
        yield ProcessButton(id="process")
```

## Form Layout Best Practices

### Input Widget Visibility

**Problem**: Input widgets not rendering despite being in DOM

**Solution**: Always specify explicit height and proper container structure

```python
def create_text_input(label: str, field_id: str, placeholder: str = "") -> ComposeResult:
    """Create a properly sized text input with label."""
    with Container(classes="form-field-container"):
        yield Label(f"{label}:", classes="form-label")
        yield Input(
            placeholder=placeholder,
            id=field_id,
            classes="form-input"
        )
```

**Required CSS**:
```tcss
/* Critical: Inputs MUST have explicit height */
.form-input {
    height: 3;           /* Required for visibility */
    width: 100%;
    margin-bottom: 1;
    border: solid $primary;
    padding: 0 1;
}

.form-field-container {
    height: auto;        /* Container sizes to content */
    width: 100%;
    margin-bottom: 1;
}

.form-label {
    height: 1;           /* Label height */
    margin-bottom: 1;
    text-style: bold;
}
```

### Two-Column Layout Pattern

For title/author and similar paired fields:

```python
def create_metadata_row() -> ComposeResult:
    """Create a responsive two-column metadata row."""
    with Horizontal(classes="metadata-row"):
        # Left column
        with Vertical(classes="metadata-col"):
            yield Label("Title (Optional):", classes="form-label")
            yield Input(
                placeholder="Auto-detected from file",
                id="title",
                classes="form-input"
            )
        
        # Right column  
        with Vertical(classes="metadata-col"):
            yield Label("Author (Optional):", classes="form-label")
            yield Input(
                placeholder="Leave blank if unknown",
                id="author",
                classes="form-input"
            )
```

**CSS for responsive columns**:
```tcss
.metadata-row {
    layout: horizontal;
    width: 100%;
    height: auto;
    gap: 2;  /* Space between columns */
}

.metadata-col {
    width: 1fr;  /* Equal column widths */
    height: auto;
}

/* Responsive: stack on narrow terminals */
@media (max-width: 80) {
    .metadata-row {
        layout: vertical;
        gap: 1;
    }
    
    .metadata-col {
        width: 100%;
    }
}
```

### Scrolling Container Pattern

**Problem**: Double scrolling breaks UI

**Solution**: Single VerticalScroll at the right level

```python
class MediaIngestWindow(Container):
    def compose(self) -> ComposeResult:
        # Single scrolling container at top level
        with VerticalScroll(classes="main-scroll"):
            # All content goes inside - no nested scrolling
            yield StatusDashboard()
            yield FileSelector()
            yield BasicMetadataForm()
            yield AdvancedOptions()
            yield ProcessButton()
```

**CSS**:
```tcss
.main-scroll {
    height: 100%;
    width: 100%;
    padding: 1;
}

/* Child containers should NOT scroll */
.form-section {
    height: auto;  /* NOT 100% or 1fr */
    width: 100%;
    margin-bottom: 2;
}
```

## Input Visibility Solutions

### The Input Height Problem

Textual Input widgets require explicit height to be visible in the terminal. This is the #1 cause of "missing" inputs.

```python
# WRONG - invisible input
yield Input(id="title")

# RIGHT - visible input
yield Input(id="title", classes="visible-input")
```

```tcss
.visible-input {
    height: 3;      /* Minimum 3 lines for single-line input */
    width: 100%;    /* Full width of parent */
    border: solid $primary;
    padding: 0 1;   /* Inner padding for text */
}
```

### TextArea Sizing

```python
yield TextArea(
    placeholder="Enter description...",
    id="description",
    classes="form-textarea"
)
```

```tcss
.form-textarea {
    min-height: 5;    /* Minimum visible area */
    max-height: 15;   /* Prevent taking over screen */
    height: auto;     /* Grow with content */
    width: 100%;
}
```

### Container Auto-Sizing

```tcss
/* Containers that hold inputs must auto-size */
.form-field-container {
    height: auto;     /* NOT fixed height */
    width: 100%;
}

.form-section {
    height: auto;     /* Let content determine height */
    width: 100%;
    margin-bottom: 2;
    padding: 1;
    border: round $surface;
}
```

## Responsive Design Patterns

### Terminal Size Adaptation

```python
class ResponsiveIngestWindow(Container):
    layout_mode = reactive("normal")  # "normal" or "compact"
    
    def compose(self) -> ComposeResult:
        with Container(classes="responsive-container"):
            yield StatusDashboard(id="status")
            
            # File selection always full width
            yield FileSelector(id="files")
            
            # Metadata fields - responsive layout
            with Container(classes="metadata-container"):
                yield self.create_metadata_fields()
            
            yield ProcessButton(id="process")
    
    def on_mount(self):
        self.update_layout()
    
    def on_resize(self, event):
        self.update_layout()
    
    def update_layout(self):
        """Update layout based on terminal size."""
        size = self.app.size
        
        if size.width < 90:
            self.layout_mode = "compact"
            self.add_class("compact-layout")
            self.remove_class("wide-layout")
        else:
            self.layout_mode = "normal"
            self.add_class("wide-layout")
            self.remove_class("compact-layout")
    
    def watch_layout_mode(self, mode: str):
        """Adjust form layout when mode changes."""
        metadata_container = self.query_one(".metadata-container")
        
        if mode == "compact":
            metadata_container.add_class("single-column")
        else:
            metadata_container.remove_class("single-column")
```

**Responsive CSS**:
```tcss
/* Default: side-by-side fields */
.metadata-container {
    layout: grid;
    grid-size: 2 1;  /* 2 columns */
    grid-columns: 1fr 1fr;
    gap: 2;
}

/* Compact: stacked fields */
.compact-layout .metadata-container,
.single-column {
    layout: vertical;
    gap: 1;
}

.compact-layout .form-field-container {
    width: 100%;
}
```

### Breakpoint-Based Design

```tcss
/* Wide terminals (>120 columns) */
@media (min-width: 120) {
    .form-container {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 2fr 1fr;
    }
    
    .main-form {
        grid-column-span: 2;
    }
}

/* Medium terminals (80-120 columns) */
@media (min-width: 80) and (max-width: 119) {
    .form-container {
        layout: vertical;
        padding: 1;
    }
}

/* Narrow terminals (<80 columns) */
@media (max-width: 79) {
    .form-container {
        layout: vertical;
        padding: 0;
    }
    
    .form-input {
        height: 2;  /* Smaller inputs for narrow screens */
    }
}
```

## Progressive Disclosure

### Simple/Advanced Mode Toggle

```python
class ProgressiveIngestWindow(Container):
    simple_mode = reactive(True)
    
    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="main-scroll"):
            # Mode selector
            with Container(classes="mode-selector"):
                with RadioSet(id="mode-toggle"):
                    yield RadioButton("Simple", value=True, id="simple")
                    yield RadioButton("Advanced", id="advanced")
            
            # Essential fields (always visible)
            with Container(classes="essential-section"):
                yield Label("Essential Information", classes="section-title")
                yield FileSelector()
                yield BasicMetadataForm()
                yield ProcessButton()
            
            # Advanced options (collapsible)
            with Collapsible(
                "Advanced Options",
                collapsed=True,
                id="advanced-section",
                classes="advanced-options"
            ):
                yield TranscriptionOptions()
                yield ChunkingSettings()
                yield AnalysisOptions()
    
    @on(RadioSet.Changed, "#mode-toggle")
    def handle_mode_change(self, event):
        self.simple_mode = event.pressed.id == "simple"
    
    def watch_simple_mode(self, simple: bool):
        """Show/hide advanced options based on mode."""
        advanced_section = self.query_one("#advanced-section")
        advanced_section.collapsed = simple
        
        # Update UI styling
        if simple:
            self.add_class("simple-mode")
            self.remove_class("advanced-mode")
        else:
            self.add_class("advanced-mode")
            self.remove_class("simple-mode")
```

### Collapsible Sections Pattern

```python
def create_collapsible_section(
    title: str,
    section_id: str,
    collapsed: bool = True
) -> ComposeResult:
    """Create a standardized collapsible section."""
    with Collapsible(
        title=title,
        collapsed=collapsed,
        id=section_id,
        classes="collapsible-section"
    ):
        yield Container(classes="section-content")
```

**CSS for collapsible sections**:
```tcss
.collapsible-section {
    margin-bottom: 2;
    border: round $primary;
}

.section-content {
    padding: 1;
    height: auto;
}

.collapsible-section > .collapsible--title {
    background: $primary;
    color: $text;
    padding: 1;
    text-style: bold;
}
```

## Validation and Error Handling

### Real-Time Validation Pattern

```python
class ValidatedIngestWindow(Container):
    form_data = reactive({})
    errors = reactive({})
    
    @on(Input.Changed)
    def handle_input_change(self, event):
        """Validate input in real-time."""
        field_id = event.input.id
        value = event.value
        
        # Update form data
        self.form_data = {**self.form_data, field_id: value}
        
        # Validate field
        error = self.validate_field(field_id, value)
        
        # Update errors
        errors = dict(self.errors)
        if error:
            errors[field_id] = error
            event.input.add_class("error")
        else:
            errors.pop(field_id, None)
            event.input.remove_class("error")
        
        self.errors = errors
        
        # Update error display
        self.update_error_display(field_id, error)
    
    def validate_field(self, field_id: str, value: str) -> Optional[str]:
        """Validate a single field."""
        if field_id == "title":
            if value and len(value.strip()) < 2:
                return "Title must be at least 2 characters"
        elif field_id == "email":
            if value and "@" not in value:
                return "Please enter a valid email address"
        # Add more field validations
        return None
    
    def update_error_display(self, field_id: str, error: Optional[str]):
        """Show/hide error message for a field."""
        try:
            error_widget = self.query_one(f"#{field_id}-error")
            if error:
                error_widget.update(f"❌ {error}")
                error_widget.remove_class("hidden")
            else:
                error_widget.add_class("hidden")
        except NoMatches:
            # Error widget doesn't exist, which is okay
            pass
```

### Error Display Pattern

```python
def create_validated_input(
    label: str,
    field_id: str,
    placeholder: str = "",
    required: bool = False
) -> ComposeResult:
    """Create an input with error display."""
    with Container(classes="validated-field"):
        # Label with required indicator
        label_text = f"{label}{'*' if required else ''}:"
        yield Label(label_text, classes="form-label")
        
        # Input field
        yield Input(
            placeholder=placeholder,
            id=field_id,
            classes="form-input"
        )
        
        # Error display (initially hidden)
        yield Static(
            "",
            id=f"{field_id}-error",
            classes="error-message hidden"
        )
```

**Error styling CSS**:
```tcss
.error-message {
    color: $error;
    margin-top: 1;
    margin-bottom: 1;
    text-style: italic;
}

.error-message.hidden {
    display: none;
}

.form-input.error {
    border: solid $error;
    background: $error 10%;
}

.validated-field {
    margin-bottom: 2;
}
```

## File Selection Patterns

### Modern File Selector

```python
class EnhancedFileSelector(Container):
    selected_files = reactive([])
    
    def compose(self) -> ComposeResult:
        with Container(classes="file-selector"):
            yield Label("Select Files", classes="section-title")
            
            # Action buttons
            with Horizontal(classes="file-actions"):
                yield Button("Browse Files", id="browse", variant="primary")
                yield Button("Clear All", id="clear", variant="default")
                yield Button("Add URLs", id="urls", variant="default")
            
            # File list display
            yield Container(id="file-list", classes="file-list-container")
            
            # URL input (initially hidden)
            with Container(id="url-input", classes="url-input hidden"):
                yield Label("Enter URLs (one per line):")
                yield TextArea(
                    placeholder="https://example.com/video.mp4",
                    id="urls-textarea",
                    classes="url-textarea"
                )
                with Horizontal(classes="url-actions"):
                    yield Button("Add URLs", id="add-urls", variant="primary")
                    yield Button("Cancel", id="cancel-urls", variant="default")
    
    @on(Button.Pressed, "#browse")
    async def handle_browse(self):
        """Open file browser."""
        try:
            files = await self.app.push_screen_wait(FileOpen())
            if files:
                self.add_files(files)
        except Exception as e:
            self.app.notify(f"Error selecting files: {e}", severity="error")
    
    @on(Button.Pressed, "#clear")
    def handle_clear(self):
        """Clear all selected files."""
        self.selected_files = []
        self.update_file_display()
    
    @on(Button.Pressed, "#urls")
    def handle_show_urls(self):
        """Show URL input area."""
        url_input = self.query_one("#url-input")
        url_input.remove_class("hidden")
    
    def add_files(self, files: List[Path]):
        """Add files to selection."""
        new_files = list(self.selected_files) + files
        self.selected_files = new_files
        self.update_file_display()
    
    def update_file_display(self):
        """Update the file list display."""
        file_list = self.query_one("#file-list")
        file_list.remove_children()
        
        if not self.selected_files:
            file_list.mount(Static("No files selected", classes="empty-message"))
        else:
            for i, file_path in enumerate(self.selected_files):
                file_list.mount(self.create_file_item(i, file_path))
    
    def create_file_item(self, index: int, file_path: Path) -> Container:
        """Create a file list item with remove button."""
        with Container(classes="file-item"):
            with Horizontal(classes="file-item-content"):
                yield Static(f"📁 {file_path.name}", classes="file-name")
                yield Static(f"{file_path.stat().st_size // 1024} KB", classes="file-size")
                yield Button("✕", id=f"remove-{index}", classes="remove-button")
        
        return container
```

## Status Dashboard Design

### Real-Time Processing Status

```python
class StatusDashboard(Container):
    status = reactive("idle")  # idle, processing, complete, error
    progress = reactive(0.0)
    current_file = reactive("")
    files_processed = reactive(0)
    total_files = reactive(0)
    error_message = reactive("")
    
    def compose(self) -> ComposeResult:
        with Container(classes="status-dashboard"):
            # Main status row
            with Horizontal(classes="status-main"):
                yield Static("Ready", id="status-text", classes="status-text")
                yield Static("", id="file-counter", classes="file-counter")
                yield Static("", id="time-display", classes="time-display")
            
            # Progress bar (hidden by default)
            yield ProgressBar(
                id="progress-bar",
                classes="progress-bar hidden"
            )
            
            # Current operation display (hidden by default)
            yield Static(
                "",
                id="current-operation",
                classes="current-operation hidden"
            )
            
            # Error display (hidden by default)
            yield Static(
                "",
                id="error-display",
                classes="error-display hidden"
            )
    
    def watch_status(self, status: str):
        """Update display when status changes."""
        status_text = self.query_one("#status-text")
        progress_bar = self.query_one("#progress-bar")
        current_op = self.query_one("#current-operation")
        error_display = self.query_one("#error-display")
        
        if status == "idle":
            status_text.update("Ready to process files")
            progress_bar.add_class("hidden")
            current_op.add_class("hidden")
            error_display.add_class("hidden")
            
        elif status == "processing":
            status_text.update("Processing...")
            progress_bar.remove_class("hidden")
            current_op.remove_class("hidden")
            error_display.add_class("hidden")
            
        elif status == "complete":
            status_text.update("✅ Processing complete")
            progress_bar.add_class("hidden")
            current_op.add_class("hidden")
            error_display.add_class("hidden")
            
        elif status == "error":
            status_text.update("❌ Processing failed")
            progress_bar.add_class("hidden")
            current_op.add_class("hidden")
            error_display.remove_class("hidden")
    
    def watch_progress(self, progress: float):
        """Update progress bar."""
        progress_bar = self.query_one("#progress-bar")
        progress_bar.progress = progress
    
    def watch_current_file(self, filename: str):
        """Update current operation display."""
        current_op = self.query_one("#current-operation")
        if filename:
            current_op.update(f"Processing: {filename}")
    
    def watch_files_processed(self, processed: int):
        """Update file counter."""
        counter = self.query_one("#file-counter")
        if self.total_files > 0:
            counter.update(f"{processed}/{self.total_files} files")
    
    def watch_error_message(self, error: str):
        """Update error display."""
        error_display = self.query_one("#error-display")
        if error:
            error_display.update(f"Error: {error}")
```

**Status dashboard CSS**:
```tcss
.status-dashboard {
    dock: top;
    height: auto;
    min-height: 3;
    background: $surface;
    border: round $primary;
    padding: 1;
    margin-bottom: 1;
}

.status-main {
    height: 3;
    align: left middle;
}

.status-text {
    width: 1fr;
    text-style: bold;
}

.file-counter, .time-display {
    width: auto;
    margin-left: 2;
    color: $text-muted;
}

.progress-bar {
    margin-top: 1;
    height: 1;
}

.current-operation {
    margin-top: 1;
    color: $text-muted;
    text-style: italic;
}

.error-display {
    margin-top: 1;
    padding: 1;
    background: $error 10%;
    border: solid $error;
    color: $error;
}
```

## Accessibility Considerations

### Keyboard Navigation

```python
class AccessibleIngestWindow(Container):
    BINDINGS = [
        ("tab", "focus_next", "Next field"),
        ("shift+tab", "focus_previous", "Previous field"),
        ("enter", "submit_form", "Submit"),
        ("escape", "cancel", "Cancel"),
        ("f1", "show_help", "Help"),
    ]
    
    def action_focus_next(self):
        """Move to next focusable widget."""
        self.screen.focus_next()
    
    def action_focus_previous(self):
        """Move to previous focusable widget."""
        self.screen.focus_previous()
    
    def action_submit_form(self):
        """Submit the form."""
        if self.can_submit():
            self.submit_form()
    
    def action_show_help(self):
        """Show help information."""
        self.app.push_screen(HelpScreen())
```

### Screen Reader Support

```python
# Add meaningful tooltips and labels
yield Input(
    placeholder="Enter video title",
    id="title",
    tooltip="Optional title for the video. Will be auto-detected if left blank."
)

# Use semantic HTML where possible
yield Button(
    "Process Video",
    id="submit",
    variant="primary",
    tooltip="Start processing the selected video files"
)

# Add ARIA-like labels for screen readers
yield Label(
    "Required fields are marked with *",
    classes="sr-only"  # Screen reader only
)
```

### High Contrast Mode

```tcss
/* High contrast theme support */
@media (prefers-high-contrast) {
    .form-input {
        border: thick $text;
        background: $background;
        color: $text;
    }
    
    .form-input:focus {
        border: thick $accent;
        background: $accent 20%;
    }
    
    .error-message {
        color: $error;
        background: $background;
        border: solid $error;
        padding: 1;
    }
}

/* Respect reduced motion preferences */
@media (prefers-reduced-motion) {
    .status-dashboard {
        transition: none;
    }
    
    .progress-bar {
        animation: none;
    }
}
```

### Focus Indicators

```tcss
/* Clear focus indicators */
Input:focus {
    border: thick $accent;
    background: $accent 10%;
    outline: none;  /* Textual handles this */
}

Button:focus {
    text-style: bold reverse;
    border: thick $accent;
}

.collapsible-section:focus {
    border: thick $accent;
}

/* Focus trap for modal dialogs */
.modal-dialog {
    /* Ensure focus stays within modal */
}
```

## Summary

These patterns provide a solid foundation for creating robust, accessible media ingestion interfaces in Textual. The key principles are:

1. **Always specify input heights** - Critical for visibility
2. **Use single-level scrolling** - Avoid nested scroll containers
3. **Implement progressive disclosure** - Simple/advanced modes
4. **Real-time validation** - Immediate feedback
5. **Responsive design** - Adapt to terminal size
6. **Proper error handling** - Clear, helpful messages
7. **Accessibility first** - Keyboard navigation and screen reader support

By following these patterns, you'll create media ingestion UIs that are both functional and user-friendly across different terminal environments.