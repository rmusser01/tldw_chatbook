# file_picker_dialog.py
# Description: File picker dialogs for evaluation tasks and datasets
#
"""
File Picker Dialogs for Evaluations
-----------------------------------

Provides thin, evaluation-specific wrappers around the enhanced file picker:

- ``EvalFilePickerDialog``: generic eval file open dialog
- ``TaskFilePickerDialog``: task file selection (YAML/JSON)
- ``DatasetFilePickerDialog``: dataset file selection (JSON/CSV/TSV)
- ``ExportFilePickerDialog``: result export destination
- ``QuickPickerWidget``: inline file selection widget

These are no longer nested ``ModalScreen`` wrappers; each dialog is a direct
subclass of ``EnhancedFileOpen`` or ``EnhancedFileSave`` so there is only one
screen on the stack and no duplicated Cancel/Select buttons.
"""

from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Optional

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Label, Static

from ..Third_Party.textual_fspicker import Filters
from .enhanced_file_picker import EnhancedFileOpen, EnhancedFileSave


def create_filter(patterns: str) -> Callable[[Path], bool]:
    """Create a case-insensitive filter function from semicolon-separated globs.

    Args:
        patterns: Semicolon-separated glob patterns (e.g. ``"*.yaml;*.json"``).

    Returns:
        A callable that returns ``True`` when a file name matches any pattern.
    """
    pattern_list = [p.strip().lower() for p in patterns.split(";") if p.strip()]

    def filter_func(path: Path) -> bool:
        name = path.name.lower()
        return any(fnmatch(name, pattern) for pattern in pattern_list)

    return filter_func


class EvalFilePickerDialog(EnhancedFileOpen):
    """Generic evaluation file open dialog.

    This is a direct ``EnhancedFileOpen`` subclass rather than a wrapper screen,
    so callers can push it with ``app.push_screen`` exactly as before.
    """

    def __init__(
        self,
        title: str = "Select File",
        filters: Optional[Filters] = None,
        callback: Optional[Callable[[Optional[str]], None]] = None,
        context: str = "eval_file",
        id: Optional[str] = None,
        classes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self._callback = callback
        super().__init__(
            location=".",
            title=title,
            filters=filters or self._default_filters(),
            must_exist=True,
            context=context,
            id=id,
            classes=classes,
            name=name,
        )

    @staticmethod
    def _default_filters() -> Filters:
        """Default filters for evaluation files."""
        return Filters(
            ("All Evaluation Files", create_filter("*.yaml;*.yml;*.json;*.csv;*.tsv")),
            ("YAML Files", create_filter("*.yaml;*.yml")),
            ("JSON Files", create_filter("*.json")),
            ("CSV Files", create_filter("*.csv;*.tsv")),
            ("All Files", lambda _path: True),
        )

    def dismiss(self, result: Optional[Path]) -> None:
        """Dismiss and notify the caller's callback."""
        if self._callback:
            try:
                self._callback(str(result) if result else None)
            except Exception:
                logger.exception("Eval file picker callback failed")
        super().dismiss(result)


class TaskFilePickerDialog(EvalFilePickerDialog):
    """File open dialog for evaluation task files."""

    def __init__(
        self,
        callback: Optional[Callable[[Optional[str]], None]] = None,
        context: str = "eval_task",
        id: Optional[str] = None,
        classes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        filters = Filters(
            ("Task Files", create_filter("*.yaml;*.yml;*.json")),
            ("YAML Files", create_filter("*.yaml;*.yml")),
            ("JSON Files", create_filter("*.json")),
            ("All Files", lambda _path: True),
        )
        super().__init__(
            title="Select Evaluation Task File",
            filters=filters,
            callback=callback,
            context=context,
            id=id,
            classes=classes,
            name=name,
        )


class DatasetFilePickerDialog(EvalFilePickerDialog):
    """File open dialog for dataset files."""

    def __init__(
        self,
        callback: Optional[Callable[[Optional[str]], None]] = None,
        context: str = "eval_dataset",
        id: Optional[str] = None,
        classes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        filters = Filters(
            ("Dataset Files", create_filter("*.json;*.csv;*.tsv")),
            ("JSON Files", create_filter("*.json")),
            ("CSV Files", create_filter("*.csv;*.tsv")),
            ("All Files", lambda _path: True),
        )
        super().__init__(
            title="Select Dataset File",
            filters=filters,
            callback=callback,
            context=context,
            id=id,
            classes=classes,
            name=name,
        )


class ExportFilePickerDialog(EnhancedFileSave):
    """File save dialog for exporting evaluation results."""

    def __init__(
        self,
        callback: Optional[Callable[[Optional[str]], None]] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self._callback = callback
        filters = Filters(
            ("JSON Files", create_filter("*.json")),
            ("CSV Files", create_filter("*.csv")),
            ("All Files", lambda _path: True),
        )
        super().__init__(
            location=".",
            title="Export Results To",
            filters=filters,
            context="eval_export",
            id=id,
            classes=classes,
            name=name,
        )

    def dismiss(self, result: Optional[Path]) -> None:
        """Dismiss and notify the caller's callback."""
        if self._callback:
            try:
                self._callback(str(result) if result else None)
            except Exception:
                logger.exception("Export file picker callback failed")
        super().dismiss(result)


class QuickPickerWidget(Container):
    """Quick file picker widget for inline use."""

    def __init__(
        self,
        label: str = "Selected File",
        file_types: str = "evaluation files",
        callback: Optional[Callable[[str], None]] = None,
        context: str = "eval_file",
        id: Optional[str] = None,
        classes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(id=id, classes=classes, name=name)
        self.label = label
        self.file_types = file_types
        self.callback = callback
        self.context = context
        self.selected_file: Optional[str] = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="quick-picker"):
            yield Label(self.label, classes="picker-label")
            yield Static(
                "No file selected",
                id="selected-file-display",
                classes="file-display",
            )
            yield Button(
                "Browse...",
                id="browse-button",
                classes="browse-button",
                tooltip=f"Choose {self.file_types} from disk.",
            )

    @on(Button.Pressed, "#browse-button")
    def handle_browse(self):
        """Open the appropriate file picker dialog."""

        def on_file_selected(file_path: Optional[str]):
            if file_path:
                self.selected_file = file_path
                file_name = Path(file_path).name
                try:
                    display = self.query_one("#selected-file-display")
                    display.update(file_name)
                    display.set_class(True, "file-selected")
                except Exception:
                    pass

                if self.callback:
                    self.callback(file_path)

        file_types = self.file_types.lower()
        if "task" in file_types:
            dialog: EnhancedFileOpen = TaskFilePickerDialog(
                callback=on_file_selected, context=self.context
            )
        elif "dataset" in file_types:
            dialog = DatasetFilePickerDialog(
                callback=on_file_selected, context=self.context
            )
        else:
            dialog = EvalFilePickerDialog(
                callback=on_file_selected, context=self.context
            )

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
        except Exception:
            pass
