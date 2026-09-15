"""Provides a file opening dialog."""

##############################################################################
# Backward compatibility.
from __future__ import annotations

##############################################################################
# Python imports.
from pathlib import Path

##############################################################################
# Textual imports.
from textual.widgets import Input

##############################################################################
# Local imports.
from .base_dialog import ButtonLabel, InputBar, resolve_default_location
from .file_dialog import BaseFileDialog
from .path_filters import Filters


##############################################################################
class FileOpen(BaseFileDialog):
    """A file opening dialog."""

    ERROR_A_FILE_MUST_EXIST = "The file must exist"
    """An error to show a user when a file must exist."""

    def __init__(
        self,
        location: str | Path = ".",
        title: str = "Open",
        *,
        open_button: ButtonLabel = "",
        cancel_button: ButtonLabel = "",
        filters: Filters | None = None,
        must_exist: bool = True,
        default_file: str | Path | None = None,
        offer_select_folder: bool = False,
    ) -> None:
        """Initialise the `FileOpen` dialog.

        Args:
            location: Optional starting location.
            title: Optional title.
            open_button: The label for the open button.
            cancel_button: The label for the cancel button.
            filters: Optional filters to show in the dialog.
            must_exist: Flag to say if the file must exist.
            default_file: The default filename to place in the input.
            offer_select_folder: When True, the dialog also offers a
                "Select folder" action returning the directory being
                viewed (task-2222) -- for callers that accept either a
                file or a folder.

        Notes:
            `open_button` and `cancel_button` can either be strings that
            set the button label, or they can be functions that take the
            default button label as a parameter and return the label to use.
        """
        super().__init__(
            resolve_default_location(location),
            title,
            select_button=self._label(open_button, "Open"),
            cancel_button=cancel_button,
            filters=filters,
            default_file=default_file,
        )
        self._must_exist = must_exist
        """Must the file exist?"""
        self._offer_select_folder = offer_select_folder
        """Offer the "select the folder being viewed" action?"""

    def _focus_initial_widget(self) -> None:
        """Focus the path field when this dialog also accepts a folder.

        task-32540 (critique #3, both assessors): Import once and "Keep a
        folder synced" both push this dialog, and the one thing a keyboard
        user does first is type a path. With the base class's default focus
        (the directory listing) every typed character went into the
        listing's type-ahead instead -- reproduced live at 235x52: typing
        "/Users" left the field on its placeholder and Enter activated the
        highlighted ".." row. ``FileSave`` already steers initial focus the
        same way for the same reason (task-1479).

        Only when ``offer_select_folder`` is on: a plain file-only
        ``FileOpen`` (character import, skill folders, TTS models, ...)
        keeps browsing-first focus, where Enter on a listing row is the
        natural first keystroke.
        """
        if not self._offer_select_folder:
            super()._focus_initial_widget()
            return
        field = self.query_one(InputBar).query_one(Input)
        field.focus()
        # Any seeded default is a starting point, not something to type
        # around -- select it so the first keystroke replaces it.
        if field.value:
            field.selection = (0, len(field.value))

    def _hint_text(self) -> str:
        """Name both actions once "Select folder" is offered (task-32122).

        Neither ``FileOpen(offer_select_folder=True)`` nor the vendored
        ``SelectDirectory`` had any on-screen hint distinguishing "Enter
        descends" from "the folder-confirming button uses this one" --
        AC#4 requires that hint actually render.
        """
        if not self._offer_select_folder:
            return ""
        open_label = self._label(self._select_button, "Open")
        return f"Enter {open_label}  ·  Select folder to use this folder"

    def _should_return(self, candidate: Path) -> bool:
        """Perform the final checks on the chosen file.

        Args:
            candidate: The file to check.
        """
        if self._must_exist and not candidate.exists():
            self._set_error(self.ERROR_A_FILE_MUST_EXIST)
            return False
        return True


### file_open.py ends here
