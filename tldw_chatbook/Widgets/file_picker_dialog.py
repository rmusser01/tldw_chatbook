# file_picker_dialog.py
# Description: File picker dialog for evaluation tasks and datasets
#
"""
File Picker Dialog for Evaluations
----------------------------------

Provides file selection functionality for:
- Task upload (YAML, JSON files)
- Dataset import (CSV, TSV, JSON files)
- Result export destination

Uses the existing fspicker component with evaluation-specific filters.
"""

from pathlib import Path
from typing import List, Optional, Callable, Any
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static, ListView, ListItem
from textual.binding import Binding
from loguru import logger

from ..Third_Party.textual_fspicker import FileOpen, FileSave, Filters
from .enhanced_file_picker import EnhancedFileOpen, EnhancedFileSave, RecentLocations

class EvalFilePickerDialog(ModalScreen):
    """Modal dialog for file selection in evaluation context."""
    
    BINDINGS = [
        Binding("ctrl+h", "toggle_hidden", "Toggle hidden files"),
        Binding("ctrl+r", "toggle_recent", "Show recent files"),
        Binding("ctrl+f", "focus_search", "Search files"),
        Binding("f5", "refresh", "Refresh directory"),
        Binding("escape", "dismiss(None)", "Cancel"),
    ]
    
    def __init__(self, 
                 title: str = "Select File",
                 filters: Optional[Filters] = None,
                 callback: Optional[Callable[[Optional[str]], None]] = None,
                 save_mode: bool = False,
                 use_enhanced: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.filters = filters or self._get_default_filters()
        self.callback = callback
        self.save_mode = save_mode
        self.use_enhanced = use_enhanced
        self.selected_file: Optional[str] = None
        self.recent_locations = RecentLocations() if use_enhanced else None
    
    def _get_default_filters(self) -> Filters:
        """Get default file filters for evaluation files."""
        return Filters(
            ("All Evaluation Files", "*.yaml;*.yml;*.json;*.csv;*.tsv"),
            ("YAML Files", "*.yaml;*.yml"),
            ("JSON Files", "*.json"),
            ("CSV Files", "*.csv;*.tsv"),
            ("All Files", "*.*")
        )
    
    def compose(self) -> ComposeResult:
        with Container(classes="file-picker-dialog"):
            yield Label(self.title, classes="dialog-title")
            
            if self.use_enhanced:
                # Use enhanced file picker with new features
                if self.save_mode:
                    yield EnhancedFileSave(
                        path=".",
                        filters=self.filters,
                        id="file-picker"
                    )
                else:
                    yield EnhancedFileOpen(
                        path=".",
                        filters=self.filters,
                        id="file-picker"
                    )
            else:
                # Fallback to original file picker
                if self.save_mode:
                    yield FileSave(
                        path=".",
                        filters=self.filters,
                        id="file-picker"
                    )
                else:
                    yield FileOpen(
                        path=".",
                        filters=self.filters,
                        id="file-picker"
                    )
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Select", id="select-button", variant="primary")
    
    def on_dismiss(self, result):
        """Handle dialog dismissal with selected file."""
        if result:
            self.selected_file = str(result)
            logger.info(f"File selected: {self.selected_file}")
            
            # Add to recent locations if using enhanced picker
            if self.use_enhanced and self.recent_locations:
                from pathlib import Path
                self.recent_locations.add(Path(result))
            
            # Call callback if provided
            if self.callback:
                self.callback(self.selected_file)
    
    @on(Button.Pressed, "#select-button")
    def handle_select(self):
        """Handle select button press."""
        if self.selected_file and self.callback:
            self.callback(self.selected_file)
        self.dismiss(self.selected_file)
    
    def action_toggle_hidden(self) -> None:
        """Forward toggle hidden action to file picker"""
        try:
            picker = self.query_one("#file-picker")
            if hasattr(picker, 'action_toggle_hidden'):
                picker.action_toggle_hidden()
        except Exception:
            pass
    
    def action_toggle_recent(self) -> None:
        """Forward toggle recent action to file picker"""
        try:
            picker = self.query_one("#file-picker")
            if hasattr(picker, 'action_toggle_recent'):
                picker.action_toggle_recent()
        except Exception:
            pass
    
    def action_focus_search(self) -> None:
        """Forward focus search action to file picker"""
        try:
            picker = self.query_one("#file-picker")
            if hasattr(picker, 'action_focus_search'):
                picker.action_focus_search()
        except Exception:
            pass
    
    def action_refresh(self) -> None:
        """Forward refresh action to file picker"""
        try:
            picker = self.query_one("#file-picker")
            if hasattr(picker, 'action_refresh'):
                picker.action_refresh()
        except Exception:
            pass
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Handle cancel button press."""
        self.dismiss(None)

class TaskFilePickerDialog(EvalFilePickerDialog):
    """Specialized file picker for evaluation task files."""
    
    def __init__(self, callback: Optional[Callable[[Optional[str]], None]] = None, **kwargs):
        filters = Filters(
            ("Task Files", "*.yaml;*.yml;*.json"),
            ("YAML Files", "*.yaml;*.yml"),
            ("JSON Files", "*.json"),
            ("All Files", "*.*")
        )
        super().__init__(
            title="Select Evaluation Task File",
            filters=filters,
            callback=callback,
            **kwargs
        )

class DatasetFilePickerDialog(EvalFilePickerDialog):
    """Specialized file picker for dataset files."""
    
    def __init__(self, callback: Optional[Callable[[Optional[str]], None]] = None, **kwargs):
        filters = Filters(
            ("Dataset Files", "*.json;*.csv;*.tsv"),
            ("JSON Files", "*.json"),
            ("CSV Files", "*.csv;*.tsv"),
            ("All Files", "*.*")
        )
        super().__init__(
            title="Select Dataset File",
            filters=filters,
            callback=callback,
            **kwargs
        )

class ExportFilePickerDialog(EvalFilePickerDialog):
    """Specialized file picker for result export."""
    
    def __init__(self, callback: Optional[Callable[[Optional[str]], None]] = None, **kwargs):
        filters = Filters(
            ("JSON Files", "*.json"),
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        )
        super().__init__(
            title="Export Results To",
            filters=filters,
            callback=callback,
            save_mode=True,
            **kwargs
        )

class QuickPickerWidget(Container):
    """Quick file picker widget for inline use."""
    
    def __init__(self, 
                 label: str = "Selected File",
                 file_types: str = "evaluation files",
                 callback: Optional[Callable[[str], None]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.file_types = file_types
        self.callback = callback
        self.selected_file: Optional[str] = None
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="quick-picker"):
            yield Label(self.label, classes="picker-label")
            yield Static("No file selected", id="selected-file-display", classes="file-display")
            yield Button("Browse...", id="browse-button", classes="browse-button")
    
    @on(Button.Pressed, "#browse-button")
    def handle_browse(self):
        """Open file picker dialog."""
        def on_file_selected(file_path: Optional[str]):
            if file_path:
                self.selected_file = file_path
                file_name = Path(file_path).name
                try:
                    display = self.query_one("#selected-file-display")
                    display.update(file_name)
                    display.set_class(True, "file-selected")
                except:
                    pass
                
                if self.callback:
                    self.callback(file_path)
        
        # Determine dialog type based on file types
        if "task" in self.file_types.lower():
            dialog = TaskFilePickerDialog(callback=on_file_selected)
        elif "dataset" in self.file_types.lower():
            dialog = DatasetFilePickerDialog(callback=on_file_selected)
        else:
            dialog = EvalFilePickerDialog(callback=on_file_selected)
        
        self.app.push_screen(dialog)
    
    def get_selected_file(self) -> Optional[str]:
        """Get the currently selected file path."""
        return self.selected_file
    
    def clear_selection(self):
        """Clear the current file selection."""
        self.selected_file = None
        try:
            display = self.query_one("#selected-file-display")
            display.update("No file selected")
            display.set_class(False, "file-selected")
        except:
            pass