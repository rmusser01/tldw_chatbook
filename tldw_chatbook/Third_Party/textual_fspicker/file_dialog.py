"""Base file-oriented dialog."""

##############################################################################
# Backward compatibility.
from __future__ import annotations

##############################################################################
# Python imports.
import sys
from pathlib import Path

##############################################################################
# Textual imports.
from textual import on
from textual.app import ComposeResult
from textual.events import Mount
from textual.widgets import Button, Input, Select

##############################################################################
# Local imports.
from .base_dialog import ButtonLabel, FileSystemPickerScreen, InputBar
from .parts import DirectoryNavigation, DriveNavigation
from .path_filters import Filters
from .path_maker import MakePath


##############################################################################
class FileFilter(Select[int]):
    """The file type filtering widget.

    This widget provides a file filter drop-down selection for all dialogs
    that inherit from
    [`BaseFileDialog`][textual_fspicker.file_dialog.BaseFileDialog].
    """


##############################################################################
class BaseFileDialog(FileSystemPickerScreen):
    """The base dialog for file-oriented picking dialogs."""

    DEFAULT_CSS = """
    BaseFileDialog InputBar {
        Input {
            /* The filename field owns all of the row's flexible space; the
            file-type filter Select next to it gets a fixed width instead
            (see below) so it can't eat the Input's share (task-1479). */
            width: 1fr;
        }
        Select {
            /* Fixed, not `1fr`: a flexible Select would still starve the
            Input for width even without the app-bundle CSS-origin issue
            documented in components/_dialogs.tcss (that file pins this
            same width for FileSave/FileOpen so it also wins there, since
            no amount of specificity in *this* DEFAULT_CSS can beat a
            CSS_PATH-sourced bundle rule -- see that file's comment). */
            width: 24;
        }
    }
    """

    ERROR_A_FILE_MUST_BE_CHOSEN = "A file must be chosen"
    """An error to show the user when a file should be chosen."""

    def __init__(
        self,
        location: str | Path = ".",
        title: str = "Open",
        select_button: ButtonLabel = "",
        cancel_button: ButtonLabel = "",
        *,
        filters: Filters | None = None,
        default_file: str | Path | None = None,
    ) -> None:
        """Initialise the base dialog.

        Args:
            location: Optional starting location.
            title: Optional title.
            select_button: The label for the select button.
            cancel_button: The label for the cancel button.
            filters: Optional filters to show in the dialog.
            default_file: The default filename to place in the input.
        """
        super().__init__(
            location, title, select_button=select_button, cancel_button=cancel_button
        )
        self._filters = filters
        """The filters for the dialog."""
        self._default_file = default_file
        """The default filename to put in the input field."""
        self._filter_select_changed_by_user = False
        """Whether ``_change_filter`` has seen a real, user-driven change yet.

        ``Select`` posts its own ``Changed`` message as a side effect of
        mounting with an explicit initial ``value=`` (not from anyone
        picking a filter) -- ``_change_filter`` must not treat that
        synthetic first event the same as a real one, or it steals focus
        away from wherever ``_focus_initial_widget()`` put it right after
        mount (task-1479).
        """

    def _input_bar(self) -> ComposeResult:
        """Provide any widgets for the input before, before the buttons."""
        yield Input(Path(self._default_file or "").name)
        if self._filters:
            yield FileFilter(
                self._filters.selections,
                prompt="File filter",
                value=0,
                allow_blank=False,
            )

    @on(Mount)
    def _initial_filter(self) -> None:
        """Set the initial filter once the DOM is ready."""
        if self._filters:
            self.query_one(DirectoryNavigation).file_filter = self._filters[0]

    @on(DirectoryNavigation.Selected)
    def _select_file(self, event: DirectoryNavigation.Selected) -> None:
        """Handle a file being selected in the picker.

        Args:
            event: The event to handle.
        """
        # Scoped through InputBar, not a bare `self.query_one(Input)`: the
        # screen also carries a hidden `#path-input` (Ctrl+L) and a hidden
        # `#search-input` (Ctrl+F), both mounted before InputBar's own
        # filename Input in the compose tree, so an unscoped query_one(Input)
        # silently grabs one of those instead (task-1479).
        file_name = self.query_one(InputBar).query_one(Input)
        file_name.value = str(event.path.name)
        file_name.focus()

    @on(Input.Changed)
    def _clear_error(self) -> None:
        """Clear any error that might be showing."""
        super()._clear_error()

    @on(Select.Changed)
    def _change_filter(self, event: Select.Changed) -> None:
        """Handle a change in the filter.

        Args:
            event: The event to handle.
        """
        if self._filters is not None and isinstance(event.value, int):
            self.query_one(DirectoryNavigation).file_filter = self._filters[event.value]
        else:
            self.query_one(DirectoryNavigation).file_filter = None
        if not self._filter_select_changed_by_user:
            # The first Changed event is Select's own mount-time side effect
            # (see `_filter_select_changed_by_user`'s docstring), not a user
            # picking a filter -- apply the filter above (matches
            # `_initial_filter`'s own mount-time assignment) but don't move
            # focus for it (task-1479).
            self._filter_select_changed_by_user = True
            return
        self.query_one(DirectoryNavigation).focus()

    def _should_return(self, candidate: Path) -> bool:
        """Final check on a picked file before returning it to the caller.

        Args:
            candidate: The file to check.

        Returns:
            `True` if the file should be returned, `False` if not.

        Note:
            This method is designed to be called as a final check; this is a
            good place to set up the display of an error before returning
            `False`, for example.
        """
        del candidate
        return True

    @on(Input.Submitted)
    @on(Button.Pressed, "#select")
    def _confirm_file(self, event: Input.Submitted | Button.Pressed) -> None:
        """Confirm the selection of the file in the input box.

        Args:
            event: The event to handle.
        """
        event.stop()
        # Scoped through InputBar -- see `_select_file` above for why a bare
        # `self.query_one(Input)` is ambiguous on this screen (task-1479).
        # Getting this wrong means the real filename Input is never read: a
        # keyboard user presses Enter on the (correctly focused, correctly
        # filled) filename field, but this handler reads back the empty
        # hidden `#path-input` instead and rejects with "A file must be
        # chosen" -- the dialog silently refuses to confirm at all.
        file_name = self.query_one(InputBar).query_one(Input)

        # Only even try and process this if there's some input.
        if not file_name.value:
            self._set_error(self.ERROR_A_FILE_MUST_BE_CHOSEN)
            return

        # If it looks like the user is typing in some sort of home
        # directory path... (does pathlib let me test for this, or at
        # least ask what the home character is? Docs don't mention this;
        # so for now I'm going to hard-code this).
        if file_name.value.startswith("~"):
            # ...let's simply expand and go with that.
            try:
                chosen = MakePath.of(file_name.value).expanduser()
            except RuntimeError as error:
                self._set_error(str(error))
                return
        else:
            # It's not a home directory path, so let's combine with the
            # location of the directory navigator widget.
            chosen = (
                self.query_one(DirectoryNavigation).location / file_name.value
            ).resolve()

        # If it's a directory, approach it like it's the user simply
        # doing a "cd".
        try:
            if chosen.is_dir():
                if sys.platform == "win32":
                    if drive_letter := MakePath.of(chosen).drive:
                        # Ensure DriveNavigation is present before querying
                        try:
                            drive_nav = self.query_one(DriveNavigation)
                            drive_nav.drive = drive_letter
                        except Exception:  # QueryError if not present
                            pass  # Silently ignore if DriveNavigation isn't there (e.g. non-Windows)
                self.query_one(DirectoryNavigation).location = chosen
                self.query_one(DirectoryNavigation).focus()
                self.query_one(InputBar).query_one(Input).value = ""
                return
        except PermissionError:
            self._set_error(self.ERROR_PERMISSION_ERROR)
            return

        # If the chosen file passes the final tests...
        if self._should_return(chosen):
            # ...return it.
            self.dismiss(result=chosen)


### file_dialog.py ends here
