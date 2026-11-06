"""Lightweight messages shared by Library and the Folder Files workspace."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from textual.message import Message

if TYPE_CHECKING:
    from .library_file_notes_workspace import LibraryFileNotesWorkspace


class FileNotesWorkspaceMessage(Message):
    """Message whose control is the retained Folder Files workspace."""

    @property
    def control(self) -> "LibraryFileNotesWorkspace":
        return cast("LibraryFileNotesWorkspace", self._sender)


class FileNotesEditableOpened(FileNotesWorkspaceMessage):
    """Announce one admitted editable file identity."""

    def __init__(self, identity: str) -> None:
        super().__init__()
        self.identity = identity


class FileNotesIdentityCleared(FileNotesWorkspaceMessage):
    """Announce an explicit opened-file identity clear."""


class FileNotesRootChanged(FileNotesWorkspaceMessage):
    """Announce one admitted current Folder Files root."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root


class FileNotesReloadConfirmationChanged(FileNotesWorkspaceMessage):
    """Announce whether the destructive reload confirmation is active."""

    def __init__(self, active: bool) -> None:
        super().__init__()
        self.active = active


class FileNotesSaveStateChanged(FileNotesWorkspaceMessage):
    """Announce a new save state (idle/dirty/saving/saved/conflict/error).

    TASK-34000.40 ruling T9-i: the Library footer's Ctrl+S chip follows the
    save state both ways -- it leaves on a conflict or an error, from the
    key, the autosave or a poll alike, and returns with a dirty draft.
    """

    def __init__(self, state: str) -> None:
        super().__init__()
        self.state = state
