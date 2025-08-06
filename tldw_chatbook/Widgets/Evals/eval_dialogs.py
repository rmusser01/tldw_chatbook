# eval_dialogs.py
# Description: Dialog widgets for the evaluation system
#
"""
Evaluation Dialog Widgets
------------------------

Custom dialog widgets for the evaluation system:
- DatasetFilePickerDialog: File picker for dataset uploads
- AdvancedConfigDialog: Advanced evaluation configuration
- TemplateSelectorDialog: Evaluation template selection
- ExportDialog: Export format and options selection
"""

from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from textual import on
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Static, Label, Select, Input,
    ListView, ListItem, TextArea,
    DataTable, Checkbox, RadioButton, RadioSet
)
from textual.containers import (
    Container, Horizontal, Vertical, VerticalScroll, Grid
)
from textual.reactive import reactive
from loguru import logger

# Import existing enhanced file picker
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen, Filters


class DatasetFilePickerDialog(ModalScreen):
    """Metadata form dialog for dataset upload after file selection."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(
        self,
        file_path: Path,
        callback: Optional[Callable[[Path, Dict[str, Any]], None]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.callback = callback
        self.file_path = file_path
        self.dataset_metadata = {}
    
    def compose(self) -> ComposeResult:
        with Container(classes="dataset-metadata-dialog"):
            yield Static("📁 Dataset Information", classes="dialog-title")
            
            # Show selected file
            yield Static(f"Selected file: {self.file_path.name}", classes="file-info")
            
            # Dataset metadata form
            with Container(classes="metadata-section"):
                with Grid(classes="metadata-grid"):
                    yield Label("Name:")
                    yield Input(
                        value=self.file_path.stem,
                        placeholder="Dataset name",
                        id="dataset-name",
                        classes="metadata-input"
                    )
                    
                    yield Label("Type:")
                    yield Select(
                        [
                            ("Multiple Choice", "multiple_choice"),
                            ("Question Answering", "qa"),
                            ("Code Generation", "coding"),
                            ("Summarization", "summarization"),
                            ("Translation", "translation"),
                            ("Custom", "custom")
                        ],
                        id="dataset-type",
                        classes="metadata-select"
                    )
                    
                    yield Label("Format:")
                    yield Select(
                        [
                            ("JSON", "json"),
                            ("JSONL", "jsonl"),
                            ("CSV", "csv"),
                            ("Parquet", "parquet"),
                            ("HuggingFace", "huggingface")
                        ],
                        value=self._detect_format(),
                        id="dataset-format",
                        classes="metadata-select"
                    )
                    
                    yield Label("Description:")
                    yield Input(
                        placeholder="Brief description",
                        id="dataset-description",
                        classes="metadata-input"
                    )
            
            # Action buttons
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="default")
    
    def _detect_format(self) -> str:
        """Auto-detect format from file extension."""
        format_map = {
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.csv': 'csv',
            '.parquet': 'parquet'
        }
        return format_map.get(self.file_path.suffix.lower(), 'json')
    
    @on(Button.Pressed, "#save-btn")
    def handle_save(self) -> None:
        """Handle save button press."""
        # Gather metadata
        self.dataset_metadata = {
            "name": self.query_one("#dataset-name", Input).value,
            "type": self.query_one("#dataset-type", Select).value,
            "format": self.query_one("#dataset-format", Select).value,
            "description": self.query_one("#dataset-description", Input).value,
            "file_path": str(self.file_path)
        }
        
        # Validate required fields
        if not self.dataset_metadata["name"]:
            self.app.notify("Please enter a dataset name", severity="error")
            return
        
        # Call callback and dismiss
        if self.callback:
            self.callback(self.file_path, self.dataset_metadata)
        
        self.dismiss(self.dataset_metadata)
    
    @on(Button.Pressed, "#cancel-btn")
    def handle_cancel(self) -> None:
        """Handle cancel button press."""
        self.dismiss(None)
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(None)


def open_dataset_file_picker(app, callback: Callable[[Path, Dict[str, Any]], None]):
    """Helper function to open file picker and then metadata dialog."""
    
    def handle_file_selected(path: Optional[Path]) -> None:
        """Handle file selection from enhanced file picker."""
        if path:
            # Open metadata dialog
            app.push_screen(
                DatasetFilePickerDialog(
                    file_path=path,
                    callback=callback
                )
            )
    
    # Define filters for dataset files
    filters = Filters(
        [
            ("Dataset Files", ["*.json", "*.jsonl", "*.csv", "*.parquet"]),
            ("JSON Files", ["*.json", "*.jsonl"]),
            ("CSV Files", ["*.csv"]),
            ("Parquet Files", ["*.parquet"]),
            ("All Files", ["*"])
        ]
    )
    
    # Open enhanced file picker
    picker = EnhancedFileOpen(
        title="Select Dataset File",
        filters=filters,
        must_exist=True,
        context="evaluation_datasets"  # For recent files tracking
    )
    
    app.push_screen(picker, handle_file_selected)


class AdvancedConfigDialog(ModalScreen):
    """Advanced configuration dialog for evaluation settings."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(
        self,
        current_config: Dict[str, Any] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.callback = callback
        self.current_config = current_config or {}
    
    def compose(self) -> ComposeResult:
        with Container(classes="config-dialog large"):
            yield Static("⚙️ Advanced Configuration", classes="dialog-title")
            
            with VerticalScroll(classes="config-scroll"):
                # Sampling configuration
                with Container(classes="config-section"):
                    yield Static("Sampling Settings", classes="section-title")
                    
                    with Grid(classes="advanced-grid"):
                        yield Label("Max Samples:")
                        yield Input(
                            value=str(self.current_config.get("max_samples", 100)),
                            id="max-samples",
                            type="integer"
                        )
                        
                        yield Label("Batch Size:")
                        yield Input(
                            value=str(self.current_config.get("batch_size", 10)),
                            id="batch-size",
                            type="integer"
                        )
                        
                        yield Label("Random Seed:")
                        yield Input(
                            value=str(self.current_config.get("seed", 42)),
                            id="random-seed",
                            type="integer"
                        )
                        
                        yield Label("Shuffle:")
                        yield Checkbox(
                            value=self.current_config.get("shuffle", True),
                            id="shuffle-checkbox"
                        )
                
                # Model parameters
                with Container(classes="config-section"):
                    yield Static("Model Parameters", classes="section-title")
                    
                    with Grid(classes="advanced-grid"):
                        yield Label("Temperature:")
                        yield Input(
                            value=str(self.current_config.get("temperature", 0.7)),
                            id="temperature",
                            type="number"
                        )
                        
                        yield Label("Max Tokens:")
                        yield Input(
                            value=str(self.current_config.get("max_tokens", 1000)),
                            id="max-tokens",
                            type="integer"
                        )
                        
                        yield Label("Top P:")
                        yield Input(
                            value=str(self.current_config.get("top_p", 1.0)),
                            id="top-p",
                            type="number"
                        )
                        
                        yield Label("Frequency Penalty:")
                        yield Input(
                            value=str(self.current_config.get("frequency_penalty", 0.0)),
                            id="freq-penalty",
                            type="number"
                        )
                
                # Evaluation settings
                with Container(classes="config-section"):
                    yield Static("Evaluation Settings", classes="section-title")
                    
                    with Grid(classes="advanced-grid"):
                        yield Label("Timeout (seconds):")
                        yield Input(
                            value=str(self.current_config.get("timeout", 30)),
                            id="timeout",
                            type="integer"
                        )
                        
                        yield Label("Retry Attempts:")
                        yield Input(
                            value=str(self.current_config.get("retry_attempts", 3)),
                            id="retry-attempts",
                            type="integer"
                        )
                        
                        yield Label("Save Outputs:")
                        yield Checkbox(
                            value=self.current_config.get("save_outputs", True),
                            id="save-outputs"
                        )
                        
                        yield Label("Verbose Logging:")
                        yield Checkbox(
                            value=self.current_config.get("verbose", False),
                            id="verbose-logging"
                        )
            
            # Action buttons
            with Horizontal(classes="dialog-buttons"):
                yield Button("Apply", id="apply-btn", variant="primary")
                yield Button("Reset", id="reset-btn", variant="warning")
                yield Button("Cancel", id="cancel-btn", variant="default")
    
    @on(Button.Pressed, "#apply-btn")
    def handle_apply(self) -> None:
        """Apply the configuration."""
        try:
            # Gather all settings
            config = {
                "max_samples": int(self.query_one("#max-samples", Input).value),
                "batch_size": int(self.query_one("#batch-size", Input).value),
                "seed": int(self.query_one("#random-seed", Input).value),
                "shuffle": self.query_one("#shuffle-checkbox", Checkbox).value,
                "temperature": float(self.query_one("#temperature", Input).value),
                "max_tokens": int(self.query_one("#max-tokens", Input).value),
                "top_p": float(self.query_one("#top-p", Input).value),
                "frequency_penalty": float(self.query_one("#freq-penalty", Input).value),
                "timeout": int(self.query_one("#timeout", Input).value),
                "retry_attempts": int(self.query_one("#retry-attempts", Input).value),
                "save_outputs": self.query_one("#save-outputs", Checkbox).value,
                "verbose": self.query_one("#verbose-logging", Checkbox).value
            }
            
            # Validate ranges
            if not 0 <= config["temperature"] <= 2:
                raise ValueError("Temperature must be between 0 and 2")
            if not 0 <= config["top_p"] <= 1:
                raise ValueError("Top P must be between 0 and 1")
            if config["max_samples"] < 1:
                raise ValueError("Max samples must be at least 1")
            
            # Call callback and dismiss
            if self.callback:
                self.callback(config)
            
            self.dismiss(config)
            
        except ValueError as e:
            self.app.notify(f"Invalid configuration: {e}", severity="error")
    
    @on(Button.Pressed, "#reset-btn")
    def handle_reset(self) -> None:
        """Reset to default values."""
        defaults = {
            "max_samples": 100,
            "batch_size": 10,
            "seed": 42,
            "shuffle": True,
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "timeout": 30,
            "retry_attempts": 3,
            "save_outputs": True,
            "verbose": False
        }
        
        # Update all inputs
        self.query_one("#max-samples", Input).value = str(defaults["max_samples"])
        self.query_one("#batch-size", Input).value = str(defaults["batch_size"])
        self.query_one("#random-seed", Input).value = str(defaults["seed"])
        self.query_one("#shuffle-checkbox", Checkbox).value = defaults["shuffle"]
        self.query_one("#temperature", Input).value = str(defaults["temperature"])
        self.query_one("#max-tokens", Input).value = str(defaults["max_tokens"])
        self.query_one("#top-p", Input).value = str(defaults["top_p"])
        self.query_one("#freq-penalty", Input).value = str(defaults["frequency_penalty"])
        self.query_one("#timeout", Input).value = str(defaults["timeout"])
        self.query_one("#retry-attempts", Input).value = str(defaults["retry_attempts"])
        self.query_one("#save-outputs", Checkbox).value = defaults["save_outputs"]
        self.query_one("#verbose-logging", Checkbox).value = defaults["verbose"]
        
        self.app.notify("Reset to default values", severity="information")
    
    @on(Button.Pressed, "#cancel-btn")
    def handle_cancel(self) -> None:
        """Handle cancel button press."""
        self.dismiss(None)
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(None)


class TemplateSelectorDialog(ModalScreen):
    """Template selection dialog for predefined evaluation configurations."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    # Predefined templates
    TEMPLATES = {
        "basic_qa": {
            "name": "Basic Q&A",
            "description": "Simple question-answering evaluation with standard settings",
            "config": {
                "task_type": "simple_qa",
                "max_samples": 100,
                "temperature": 0.7,
                "max_tokens": 500
            }
        },
        "code_generation": {
            "name": "Code Generation",
            "description": "Evaluate code generation capabilities with execution tests",
            "config": {
                "task_type": "coding",
                "max_samples": 50,
                "temperature": 0.2,
                "max_tokens": 2000,
                "save_outputs": True
            }
        },
        "summarization": {
            "name": "Summarization",
            "description": "Text summarization evaluation with ROUGE metrics",
            "config": {
                "task_type": "summarization",
                "max_samples": 200,
                "temperature": 0.5,
                "max_tokens": 300
            }
        },
        "translation": {
            "name": "Translation",
            "description": "Language translation evaluation with BLEU scores",
            "config": {
                "task_type": "translation",
                "max_samples": 100,
                "temperature": 0.3,
                "max_tokens": 1000
            }
        },
        "custom": {
            "name": "Custom Template",
            "description": "Start with a blank template and configure all settings",
            "config": {}
        }
    }
    
    def __init__(
        self,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.callback = callback
        self.selected_template = reactive(None)
    
    def compose(self) -> ComposeResult:
        with Container(classes="template-selector-dialog"):
            yield Static("📋 Select Evaluation Template", classes="dialog-title")
            
            with Container(classes="template-content"):
                # Template list
                with Container(classes="template-list-container"):
                    yield Static("Available Templates", classes="section-title")
                    yield ListView(
                        id="template-list",
                        classes="template-list"
                    )
                
                # Template preview
                with Container(classes="template-preview-container"):
                    yield Static("Template Details", classes="section-title")
                    
                    yield Static(
                        "Select a template to view details",
                        id="template-description",
                        classes="template-description"
                    )
                    
                    yield Static(
                        "",
                        id="template-config",
                        classes="config-display"
                    )
                    
                    with Horizontal(classes="preview-actions"):
                        yield Button(
                            "Customize",
                            id="customize-btn",
                            disabled=True
                        )
            
            # Action buttons
            with Horizontal(classes="dialog-buttons"):
                yield Button("Use Template", id="use-btn", variant="primary", disabled=True)
                yield Button("Cancel", id="cancel-btn", variant="default")
    
    def on_mount(self) -> None:
        """Populate template list on mount."""
        template_list = self.query_one("#template-list", ListView)
        
        for template_id, template_data in self.TEMPLATES.items():
            item = ListItem(
                Label(f"{template_data['name']}")
            )
            item.data = (template_id, template_data)
            template_list.append(item)
    
    @on(ListView.Selected, "#template-list")
    def handle_template_selected(self, event: ListView.Selected) -> None:
        """Handle template selection."""
        if event.item and hasattr(event.item, 'data'):
            template_id, template_data = event.item.data
            self.selected_template = template_id
            
            # Update preview
            desc = self.query_one("#template-description", Static)
            desc.update(template_data['description'])
            
            # Show config
            config_display = self.query_one("#template-config", Static)
            config_text = "Configuration:\n"
            for key, value in template_data['config'].items():
                config_text += f"  {key}: {value}\n"
            config_display.update(config_text)
            
            # Enable buttons
            self.query_one("#use-btn", Button).disabled = False
            self.query_one("#customize-btn", Button).disabled = False
    
    @on(Button.Pressed, "#use-btn")
    def handle_use_template(self) -> None:
        """Use the selected template."""
        if self.selected_template:
            template_data = self.TEMPLATES[self.selected_template]
            
            if self.callback:
                self.callback(self.selected_template, template_data['config'])
            
            self.dismiss((self.selected_template, template_data['config']))
    
    @on(Button.Pressed, "#customize-btn")
    def handle_customize(self) -> None:
        """Open advanced config with template as base."""
        if self.selected_template:
            template_config = self.TEMPLATES[self.selected_template]['config']
            self.dismiss(("customize", template_config))
    
    @on(Button.Pressed, "#cancel-btn")
    def handle_cancel(self) -> None:
        """Handle cancel button press."""
        self.dismiss(None)
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(None)


class ExportDialog(ModalScreen):
    """Export dialog for selecting format and options."""
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(
        self,
        run_data: Dict[str, Any] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.callback = callback
        self.run_data = run_data or {}
        self.selected_format = reactive("csv")
    
    def compose(self) -> ComposeResult:
        with Container(classes="export-dialog"):
            yield Static("📥 Export Evaluation Results", classes="dialog-title")
            
            # Format selection
            with Container(classes="format-section"):
                yield Static("Export Format", classes="section-title")
                
                with RadioSet(id="format-radio"):
                    yield RadioButton("CSV - Tabular data", value=True, id="format-csv")
                    yield RadioButton("JSON - Full data with metadata", id="format-json")
                    yield RadioButton("PDF - Formatted report", id="format-pdf")
            
            # Export options
            with Container(classes="options-section"):
                yield Static("Export Options", classes="section-title")
                
                with Grid(classes="export-options-grid"):
                    yield Checkbox("Include raw outputs", value=True, id="include-raw")
                    yield Checkbox("Include metrics", value=True, id="include-metrics")
                    yield Checkbox("Include configuration", value=True, id="include-config")
                    yield Checkbox("Include timestamps", value=False, id="include-timestamps")
                    yield Checkbox("Aggregate by metric", value=False, id="aggregate-metrics")
                    yield Checkbox("Add summary statistics", value=True, id="add-summary")
            
            # File name
            with Container(classes="filename-section"):
                yield Label("Filename:")
                yield Input(
                    value=f"eval_results_{self.run_data.get('run_id', 'export')}",
                    id="filename-input",
                    placeholder="Enter filename (without extension)"
                )
            
            # Action buttons
            with Horizontal(classes="dialog-buttons"):
                yield Button("Export", id="export-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="default")
    
    @on(RadioButton.Changed)
    def handle_format_change(self, event: RadioButton.Changed) -> None:
        """Handle format selection change."""
        if event.value:  # RadioButton was selected
            if event.radio_button.id == "format-csv":
                self.selected_format = "csv"
            elif event.radio_button.id == "format-json":
                self.selected_format = "json"
            elif event.radio_button.id == "format-pdf":
                self.selected_format = "pdf"
    
    @on(Button.Pressed, "#export-btn")
    def handle_export(self) -> None:
        """Handle export button press."""
        # Gather options
        options = {
            "format": self.selected_format,
            "filename": self.query_one("#filename-input", Input).value,
            "include_raw": self.query_one("#include-raw", Checkbox).value,
            "include_metrics": self.query_one("#include-metrics", Checkbox).value,
            "include_config": self.query_one("#include-config", Checkbox).value,
            "include_timestamps": self.query_one("#include-timestamps", Checkbox).value,
            "aggregate_metrics": self.query_one("#aggregate-metrics", Checkbox).value,
            "add_summary": self.query_one("#add-summary", Checkbox).value
        }
        
        # Validate filename
        if not options["filename"]:
            self.app.notify("Please enter a filename", severity="error")
            return
        
        # Add extension if not present
        if not options["filename"].endswith(f".{self.selected_format}"):
            options["filename"] += f".{self.selected_format}"
        
        # Call callback and dismiss
        if self.callback:
            self.callback(self.selected_format, options)
        
        self.dismiss(options)
    
    @on(Button.Pressed, "#cancel-btn")
    def handle_cancel(self) -> None:
        """Handle cancel button press."""
        self.dismiss(None)
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(None)