# eval_additional_dialogs.py
# Description: Additional dialogs for evaluation system
#
"""
Additional Evaluation Dialogs
-----------------------------

Additional dialogs to complete the evaluation system:
- FileUploadDialog: File upload and validation
- ExportDialog: Results export configuration
- FilterDialog: Results filtering
- RunSelectionDialog: Run selection for comparison
"""

from typing import Dict, List, Any, Optional, Callable
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Label, Input, Select, TextArea, Checkbox, 
    Static, ListView, ListItem, Collapsible
)
from textual.validation import Number, Length
from loguru import logger

class FileUploadDialog(ModalScreen):
    """Dialog for uploading task files."""
    
    def __init__(self, 
                 callback: Optional[Callable[[Optional[str]], None]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.selected_file: Optional[str] = None
    
    def compose(self) -> ComposeResult:
        with Container(classes="config-dialog"):
            yield Label("Upload Task File", classes="dialog-title")
            
            with Grid(classes="config-grid"):
                yield Label("File Path:")
                yield Input(
                    placeholder="/path/to/task/file.yaml",
                    id="file-path"
                )
                
                yield Label("File Type:")
                yield Select(
                    [
                        ("Auto-detect", "auto"),
                        ("Eleuther AI YAML", "eleuther"),
                        ("Custom JSON", "json"),
                        ("CSV Dataset", "csv")
                    ],
                    value="auto",
                    id="file-type"
                )
                
                yield Label("Description:")
                yield TextArea(
                    placeholder="Optional description for this task file",
                    id="file-description"
                )
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Browse", id="browse-button")
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Upload", id="upload-button", variant="primary")
    
    @on(Button.Pressed, "#browse-button")
    def handle_browse(self):
        """Open file browser (placeholder)."""
        # In a real implementation, this would open a file dialog
        self.app.notify("File browser not implemented - enter path manually", severity="information")
    
    @on(Button.Pressed, "#upload-button")
    def handle_upload(self):
        """Handle file upload."""
        file_path = self.query_one("#file-path").value.strip()
        if not file_path:
            self._show_error("Please specify a file path")
            return
        
        # Validate file exists
        from pathlib import Path
        if not Path(file_path).exists():
            self._show_error("File does not exist")
            return
        
        if self.callback:
            self.callback(file_path)
        self.dismiss(file_path)
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Cancel the dialog."""
        if self.callback:
            self.callback(None)
        self.dismiss(None)
    
    def _show_error(self, message: str):
        """Show error message to user."""
        logger.error(message)
        self.app.notify(message, severity="error")

class ExportDialog(ModalScreen):
    """Dialog for configuring result exports."""
    
    def __init__(self, 
                 callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None,
                 data_type: str = "results",
                 **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.data_type = data_type
    
    def compose(self) -> ComposeResult:
        with Container(classes="config-dialog"):
            yield Label(f"Export {self.data_type.title()}", classes="dialog-title")
            
            with Grid(classes="config-grid"):
                yield Label("Export Format:")
                yield Select(
                    [
                        ("CSV (Spreadsheet)", "csv"),
                        ("JSON (Structured)", "json"),
                        ("Excel Workbook", "xlsx"),
                        ("Text Report", "txt")
                    ],
                    value="csv",
                    id="export-format"
                )
                
                yield Label("Output Path:")
                yield Input(
                    placeholder="/path/to/export/file",
                    id="output-path"
                )
                
                yield Label("Include Columns:")
                yield Select(
                    [
                        ("All columns", "all"),
                        ("Essential only", "essential"),
                        ("Custom selection", "custom")
                    ],
                    value="all",
                    id="column-selection"
                )
            
            with Collapsible(title="Options", collapsed=True):
                yield Checkbox("Include metadata", value=True, id="include-metadata")
                yield Checkbox("Include error details", value=True, id="include-errors")
                yield Checkbox("Compress output", value=False, id="compress-output")
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Export", id="export-button", variant="primary")
    
    @on(Button.Pressed, "#export-button")
    def handle_export(self):
        """Handle export request."""
        config = self._collect_config()
        
        if not config.get('output_path'):
            self._show_error("Please specify an output path")
            return
        
        if self.callback:
            self.callback(config)
        self.dismiss(config)
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Cancel the dialog."""
        if self.callback:
            self.callback(None)
        self.dismiss(None)
    
    def _collect_config(self) -> Dict[str, Any]:
        """Collect export configuration."""
        try:
            return {
                'format': self.query_one("#export-format").value,
                'output_path': self.query_one("#output-path").value.strip(),
                'column_selection': self.query_one("#column-selection").value,
                'include_metadata': self.query_one("#include-metadata").value,
                'include_errors': self.query_one("#include-errors").value,
                'compress_output': self.query_one("#compress-output").value,
                'data_type': self.data_type
            }
        except Exception as e:
            logger.error(f"Error collecting export config: {e}")
            return {}
    
    def _show_error(self, message: str):
        """Show error message to user."""
        logger.error(message)
        self.app.notify(message, severity="error")

class FilterDialog(ModalScreen):
    """Dialog for filtering evaluation results."""
    
    def __init__(self, 
                 callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
    
    def compose(self) -> ComposeResult:
        with Container(classes="config-dialog"):
            yield Label("Filter Results", classes="dialog-title")
            
            with Grid(classes="config-grid"):
                yield Label("Score Range:")
                with Horizontal():
                    yield Input(placeholder="Min", id="min-score", validators=[Number(minimum=0.0, maximum=1.0)])
                    yield Label(" to ")
                    yield Input(placeholder="Max", id="max-score", validators=[Number(minimum=0.0, maximum=1.0)])
                
                yield Label("Status:")
                yield Select(
                    [
                        ("All", "all"),
                        ("Success only", "success"),
                        ("Errors only", "error")
                    ],
                    value="all",
                    id="status-filter"
                )
                
                yield Label("Text Contains:")
                yield Input(
                    placeholder="Search in input/output text",
                    id="text-search"
                )
                
                yield Label("Sample Count:")
                yield Input(
                    placeholder="Limit results (optional)",
                    validators=[Number(minimum=1)],
                    id="limit-count"
                )
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Clear", id="clear-button")
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Apply", id="apply-button", variant="primary")
    
    @on(Button.Pressed, "#clear-button")
    def handle_clear(self):
        """Clear all filter fields."""
        try:
            self.query_one("#min-score").value = ""
            self.query_one("#max-score").value = ""
            self.query_one("#status-filter").value = "all"
            self.query_one("#text-search").value = ""
            self.query_one("#limit-count").value = ""
        except Exception as e:
            logger.error(f"Error clearing filters: {e}")
    
    @on(Button.Pressed, "#apply-button")
    def handle_apply(self):
        """Apply the filters."""
        config = self._collect_config()
        
        if self.callback:
            self.callback(config)
        self.dismiss(config)
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Cancel the dialog."""
        if self.callback:
            self.callback(None)
        self.dismiss(None)
    
    def _collect_config(self) -> Dict[str, Any]:
        """Collect filter configuration."""
        try:
            config = {}
            
            min_score = self.query_one("#min-score").value.strip()
            if min_score:
                config['min_score'] = float(min_score)
            
            max_score = self.query_one("#max-score").value.strip()
            if max_score:
                config['max_score'] = float(max_score)
            
            status_filter = self.query_one("#status-filter").value
            if status_filter != "all":
                config['status'] = status_filter
            
            text_search = self.query_one("#text-search").value.strip()
            if text_search:
                config['text_contains'] = text_search
            
            limit_count = self.query_one("#limit-count").value.strip()
            if limit_count:
                config['limit'] = int(limit_count)
            
            return config
            
        except Exception as e:
            logger.error(f"Error collecting filter config: {e}")
            return {}

class RunSelectionDialog(ModalScreen):
    """Dialog for selecting runs for comparison."""
    
    def __init__(self, 
                 callback: Optional[Callable[[Optional[List[str]]], None]] = None,
                 available_runs: Optional[List[Dict[str, Any]]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.available_runs = available_runs or []
        self.selected_runs = set()
    
    def compose(self) -> ComposeResult:
        with Container(classes="config-dialog large"):
            yield Label("Select Runs for Comparison", classes="dialog-title")
            
            yield Label("Available Runs:")
            yield ListView(id="runs-list", classes="runs-list")
            
            yield Label("Selected: 0 runs", id="selection-count")
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Select All", id="select-all-button")
                yield Button("Clear All", id="clear-all-button")
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Compare", id="compare-button", variant="primary")
    
    def on_mount(self):
        """Populate the runs list."""
        try:
            runs_list = self.query_one("#runs-list")
            
            for run in self.available_runs:
                run_id = run.get('id', '')
                run_name = run.get('name', 'Unknown')
                task_name = run.get('task_name', 'Unknown')
                model_name = run.get('model_name', 'Unknown')
                status = run.get('status', 'Unknown')
                
                # Create list item with checkbox-like behavior
                item_text = f"[{status}] {run_name} | {task_name} | {model_name}"
                list_item = ListItem(Label(item_text), id=f"run-{run_id}")
                runs_list.append(list_item)
                
        except Exception as e:
            logger.error(f"Error populating runs list: {e}")
    
    @on(ListView.Selected)
    def handle_run_selected(self, event):
        """Handle run selection."""
        try:
            # Extract run ID from the item ID
            item_id = event.item.id
            if item_id and item_id.startswith("run-"):
                run_id = item_id[4:]  # Remove "run-" prefix
                
                if run_id in self.selected_runs:
                    self.selected_runs.remove(run_id)
                    # Update visual state (remove highlight)
                    event.item.remove_class("selected")
                else:
                    self.selected_runs.add(run_id)
                    # Update visual state (add highlight)
                    event.item.add_class("selected")
                
                # Update selection count
                count_label = self.query_one("#selection-count")
                count_label.update(f"Selected: {len(self.selected_runs)} runs")
                
        except Exception as e:
            logger.error(f"Error handling run selection: {e}")
    
    @on(Button.Pressed, "#select-all-button")
    def handle_select_all(self):
        """Select all runs."""
        try:
            runs_list = self.query_one("#runs-list")
            self.selected_runs = set()
            
            for item in runs_list.children:
                if hasattr(item, 'id') and item.id and item.id.startswith("run-"):
                    run_id = item.id[4:]
                    self.selected_runs.add(run_id)
                    item.add_class("selected")
            
            count_label = self.query_one("#selection-count")
            count_label.update(f"Selected: {len(self.selected_runs)} runs")
            
        except Exception as e:
            logger.error(f"Error selecting all runs: {e}")
    
    @on(Button.Pressed, "#clear-all-button")
    def handle_clear_all(self):
        """Clear all selections."""
        try:
            runs_list = self.query_one("#runs-list")
            self.selected_runs.clear()
            
            for item in runs_list.children:
                item.remove_class("selected")
            
            count_label = self.query_one("#selection-count")
            count_label.update("Selected: 0 runs")
            
        except Exception as e:
            logger.error(f"Error clearing selections: {e}")
    
    @on(Button.Pressed, "#compare-button")
    def handle_compare(self):
        """Start comparison with selected runs."""
        if len(self.selected_runs) < 2:
            self.app.notify("Please select at least 2 runs to compare", severity="error")
            return
        
        if self.callback:
            self.callback(list(self.selected_runs))
        self.dismiss(list(self.selected_runs))
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Cancel the dialog."""
        if self.callback:
            self.callback(None)
        self.dismiss(None)

# Template and utility functions
def get_model_templates() -> List[Dict[str, Any]]:
    """Get predefined model configuration templates."""
    return [
        {
            'name': 'GPT-4 Turbo',
            'provider': 'openai',
            'model_id': 'gpt-4-turbo-preview',
            'temperature': 0.0,
            'max_tokens': 1024
        },
        {
            'name': 'GPT-3.5 Turbo',
            'provider': 'openai',
            'model_id': 'gpt-3.5-turbo',
            'temperature': 0.0,
            'max_tokens': 1024
        },
        {
            'name': 'Claude 3 Opus',
            'provider': 'anthropic',
            'model_id': 'claude-3-opus-20240229',
            'temperature': 0.0,
            'max_tokens': 1024
        },
        {
            'name': 'Claude 3.5 Sonnet',
            'provider': 'anthropic',
            'model_id': 'claude-3-5-sonnet-20241022',
            'temperature': 0.0,
            'max_tokens': 1024
        },
        {
            'name': 'Llama 3.1 70B (Groq)',
            'provider': 'groq',
            'model_id': 'llama-3.1-70b-versatile',
            'temperature': 0.0,
            'max_tokens': 1024
        }
    ]

def get_task_templates() -> List[Dict[str, Any]]:
    """Get predefined task configuration templates."""
    return [
        {
            'name': 'Math Word Problems',
            'task_type': 'question_answer',
            'description': 'Grade school math word problems requiring multi-step reasoning',
            'max_samples': 100,
            'temperature': 0.0,
            'max_tokens': 256
        },
        {
            'name': 'Multiple Choice QA',
            'task_type': 'classification',
            'description': 'Multiple choice questions with single correct answer',
            'max_samples': 200,
            'temperature': 0.0,
            'max_tokens': 10
        },
        {
            'name': 'Code Generation',
            'task_type': 'generation',
            'description': 'Python function implementation from natural language',
            'max_samples': 50,
            'temperature': 0.1,
            'max_tokens': 512
        },
        {
            'name': 'Sentiment Analysis',
            'task_type': 'classification',
            'description': 'Classify text sentiment as positive, negative, or neutral',
            'max_samples': 300,
            'temperature': 0.0,
            'max_tokens': 5
        }
    ]