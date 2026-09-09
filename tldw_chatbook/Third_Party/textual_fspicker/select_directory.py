"""Provides a directory selection dialog."""

##############################################################################
# Backward compatibility.
from __future__ import annotations

##############################################################################
# Python imports.
from pathlib import Path

##############################################################################
# Textual imports.
from typing import Union

from textual import on
from textual.app import ComposeResult
from textual.widgets import Button, Input, Label

##############################################################################
# Local imports.
from .base_dialog import ButtonLabel, FileSystemPickerScreen, resolve_default_location
from .parts import DirectoryNavigation
from .path_maker import MakePath


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

    def _resolve_typed_directory(self, value: str) -> Union[Path, str]:
        """Resolve a typed field value to an absolute directory, or an error.

        Shared by Enter-to-navigate (``_handle_path_input_submission``) and
        Select-to-confirm (``_select_directory``) so both actions agree on
        what the typed text means (task-32122 AC#1): Select used to ignore
        the field entirely and return whatever directory was merely being
        browsed.

        Returns:
            The resolved absolute ``Path`` when it names an existing
            directory, or an error string ready for ``_set_error`` when it
            does not.
        """
        value = value.strip()
        current = self.query_one(DirectoryNavigation).location
        if not value or value == str(current):
            # Unchanged from the directory being browsed (the field is kept
            # in step with it -- see ``_update_path_input_on_nav_change``):
            # return that exact object rather than re-resolving its own
            # string form. Symlinked locations (macOS "/tmp" ->
            # "/private/tmp") would otherwise change the plain no-typing
            # Select result.
            return current
        try:
            target_path = MakePath.of(value).expanduser().resolve()
        except RuntimeError as error:
            return str(error)
        if target_path.is_dir():
            return target_path
        if target_path.exists():
            return f"Not a directory: {target_path.name}"
        return f"Path not found: {value}"

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
        try:
            result = self._resolve_typed_directory(event.value)
        except PermissionError:
            self._set_error(self.ERROR_PERMISSION_ERROR)
            self.query_one("#path_input", Input).focus()
            return
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
        try:
            result = self._resolve_typed_directory(
                self.query_one("#path_input", Input).value
            )
        except PermissionError:
            self._set_error(self.ERROR_PERMISSION_ERROR)
            self.query_one("#path_input", Input).focus()
            return
        if isinstance(result, Path):
            self.dismiss(result=result)
            return
        self._set_error(result)
        self.query_one("#path_input", Input).focus()


### select_directory.py ends here
