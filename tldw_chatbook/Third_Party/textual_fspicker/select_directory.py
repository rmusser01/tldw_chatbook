"""Provides a directory selection dialog."""

##############################################################################
# Backward compatibility.
from __future__ import annotations

##############################################################################
# Python imports.
from pathlib import Path

##############################################################################
# Textual imports.
from textual import on
from textual.app import ComposeResult
from textual.widgets import Button, Input, Label

##############################################################################
# Local imports.
from .base_dialog import (
    ButtonLabel,
    FileSystemPickerScreen,
    resolve_default_location,
    resolve_typed_directory,
)
from .parts import DirectoryNavigation


##############################################################################
class SelectDirectory(FileSystemPickerScreen):
    """A directory selection dialog."""

    DEFAULT_CSS = (
        FileSystemPickerScreen.DEFAULT_CSS
        + """
    SelectDirectory InputBar {
        Input { /* Style for the new path input */
            width: 2fr; /* Give it more space than buttons */
            margin-right: 1;
        }
    }
    """
    )

    def __init__(
        self,
        location: str | Path = ".",
        title: str = "Select directory",
        *,
        select_button: ButtonLabel = "",
        cancel_button: ButtonLabel = "",
    ) -> None:
        """Initialise the dialog.

        Args:
            location: Optional starting location.
            title: Optional title.
            select_button: The label for the select button.
            cancel_button: The label for the cancel button.

        Notes:
            `select_button` and `cancel_button` can either be strings that
            set the button label, or they can be functions that take the
            default button label as a parameter and return the label to use.
        """
        super().__init__(
            resolve_default_location(location),
            title,
            select_button=select_button,
            cancel_button=cancel_button,
        )

    def on_mount(self) -> None:
        """Configure the dialog once the DOM is ready."""
        navigation = self.query_one(DirectoryNavigation)
        navigation.show_files = False

        path_input = self.query_one("#path_input", Input)
        path_input.value = str(navigation.location)
        # navigation.focus() # Focus is handled by super().on_mount or should be reconsidered

    def _input_bar(self) -> ComposeResult:
        """Provide the labelled path input for direct navigation.

        A persistent "Folder path" label (task-32122 AC#2), not just a
        placeholder that vanishes the moment the user types.
        """
        yield Label("Folder path:", id="path-input-label")
        yield Input(id="path_input", placeholder="Type path or select below")

    def _hint_text(self) -> str:
        """Directory-mode hint: Enter descends, Select confirms (task-32122)."""
        select_label = self._label(self._select_button, "Select")
        return f"Enter Open  ·  {select_label} use this folder"

    @on(DirectoryNavigation.Changed)
    def _update_path_input_on_nav_change(
        self, event: DirectoryNavigation.Changed
    ) -> None:
        """Update the display of the current location in the Input widget.

        Args:
            event: The event with the selection information in.
        """
        path_input = self.query_one("#path_input", Input)
        path_input.value = str(event.control.location)

    @on(Input.Submitted, "#path_input")
    def _handle_path_input_submission(self, event: Input.Submitted) -> None:
        """Handle submission of the path Input widget."""
        event.stop()
        result = resolve_typed_directory(
            event.value, self.query_one(DirectoryNavigation).location
        )
        if isinstance(result, Path):
            # This will trigger DirectoryNavigation.Changed.
            self.query_one(DirectoryNavigation).location = result
            return
        self._set_error(result)
        self.query_one("#path_input", Input).focus()

    @on(Button.Pressed, "#select")
    def _select_directory(self, event: Button.Pressed) -> None:
        """Resolve the typed field first, then the directory being viewed.

        Used to always return ``DirectoryNavigation.location`` -- the
        directory merely being browsed -- silently discarding a path the
        user typed but never pressed Enter on (task-32122 AC#1).

        Args:
            event: The button press event.
        """
        event.stop()
        result = resolve_typed_directory(
            self.query_one("#path_input", Input).value,
            self.query_one(DirectoryNavigation).location,
        )
        if isinstance(result, Path):
            self.dismiss(result=result)
            return
        self._set_error(result)
        self.query_one("#path_input", Input).focus()


### select_directory.py ends here
