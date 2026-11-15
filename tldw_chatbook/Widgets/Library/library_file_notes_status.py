"""Independent content, folder authority, and safe-action status for File Notes."""

from __future__ import annotations

from pathlib import Path

from rich.cells import cell_len

from .library_file_notes_git_panel import _middle_elide_cells
from .library_notes_canvas import NotesStatusChannels


def resolve_file_note_status_channels(
    *,
    root: str | Path | None,
    file_open: bool = True,
    conflict: bool = False,
    unavailable: bool = False,
    read_only: bool = False,
    exact_export_available: bool = False,
    save_failed: bool = False,
    saving: bool = False,
    dirty: bool = False,
    git_failure: str = "",
    git_uncertain: str = "",
    git_running: str = "",
    git_changes: int = 0,
    authority_failure: str = "",
    authority_uncertain: str = "",
    authority_running: str = "",
    repository_confirmed: bool = False,
) -> NotesStatusChannels:
    """Resolve Folder content and Git status without cross-channel masking.

    Args:
        root: Active Folder Files root, or ``None`` when no root is linked.
        file_open: Whether a file is actually open in the editor. task-32621
            AC#1 (A cap 29): with nothing selected every save-state input is
            False, so the final ``else`` asserted "Saved" over an empty
            editor -- a save state claimed for a file that does not exist,
            on the one surface whose whole promise is that it edits the real
            file.
        conflict: Whether disk and editor content conflict.
        unavailable: Whether the linked root cannot currently be reached.
        read_only: Whether the opened document cannot be edited.
        exact_export_available: Whether the exact read-only content can be exported.
        save_failed: Whether the latest save attempt failed.
        saving: Whether a save is currently running.
        dirty: Whether the editor contains unsaved changes.
        git_failure: Bounded detail for a failed Git operation.
        git_uncertain: Bounded detail for a Git operation with an uncertain outcome.
        git_running: Bounded detail for the active Git operation.
        git_changes: Number of observed repository changes.
        authority_failure: Bounded detail for a failed authority operation.
        authority_uncertain: Bounded detail for an uncertain authority operation.
        authority_running: Bounded detail for the active authority operation.
        repository_confirmed: Whether a repository check (``git rev-parse``)
            has succeeded for this root. task-32543: without it the change
            count is this session's edit count, never "Git" -- a plain
            folder read "Git · 1 change" after one edit (critique #3, A 57;
            B 40, 50), which the solo operator took as "version-controlled".

    Returns:
        The independent content, authority, and safe-action status channels.
    """
    if conflict:
        content = "Conflict — the disk file changed; your draft is preserved."
        safe = "Save Copy"
    elif unavailable:
        content = "Unavailable — the folder cannot be reached; your draft is preserved."
        safe = "Choose folder"
    elif read_only:
        content = (
            "Read-only — the file cannot be edited; the current text is preserved."
        )
        safe = "Export exact copy" if exact_export_available else "Open Manage"
    elif save_failed:
        content = "Save failed — your draft remains in the editor."
        safe = "Save Copy"
    elif saving:
        content, safe = "Saving…", None
    elif dirty:
        content, safe = "Unsaved changes", None
    elif not file_open:
        # No "Next:" clause: this pane's safe actions name CONTROLS ("Save
        # Copy", "Choose folder"), the tree is already the only thing to act
        # on, and at 60x20 a sentence here is three rows of a two-row box.
        content, safe = "No file open.", None
    else:
        content, safe = "Saved", None
    if (
        unavailable
        and root is not None
        and (conflict or save_failed or saving or dirty)
    ):
        # TASK-38983: an unsaved draft refuses a folder change, and Save Copy
        # writes into the same unreachable folder, so neither step works.
        safe = "Reconnect the folder"

    if git_failure:
        failure_lower = git_failure.casefold()
        if "commit" in failure_lower:
            git_failure = "Commit failed"
        elif "push" in failure_lower:
            git_failure = "Push failed"
        elif "status" in failure_lower:
            git_failure = "Git status failed"
        elif "action" in failure_lower or "stage" in failure_lower:
            git_failure = "Git action failed"
        elif cell_len(git_failure) > 24:
            git_failure = "Git operation failed"
    if git_uncertain and cell_len(git_uncertain) > 24:
        git_uncertain = "Outcome uncertain"
    if git_running and cell_len(git_running) > 24:
        git_running = "Git operation running"

    status_copy = authority_failure or authority_uncertain or authority_running
    if not status_copy:
        status_copy = (
            (f"Git · {git_failure}" if git_failure else "")
            or (f"Git · {git_uncertain}" if git_uncertain else "")
            or (f"Git · {git_running}" if git_running else "")
        )
    if not status_copy and git_changes:
        change_word = "change" if git_changes == 1 else "changes"
        status_copy = (
            f"Git · {git_changes} {change_word}"
            if repository_confirmed
            else f"{git_changes} session {change_word}"
        )
    if cell_len(status_copy) > 34:
        status_copy = _middle_elide_cells(status_copy, 34)

    authority_suffix = f" · {status_copy}" if status_copy else ""
    if root is None:
        root_copy = "No folder selected"
    else:
        path = Path(root)
        folder_name = path.name or path.anchor or str(path)
        authority_prefix = "Folder files · Folder:"
        available = 60 - cell_len(authority_prefix + " " + authority_suffix)
        if available > 0:
            root_name = _middle_elide_cells(folder_name, min(14, available))
            root_copy = f"Folder: {root_name}"
        else:
            root_copy = "Folder:…"
    authority = f"Folder files · {root_copy}{authority_suffix}"
    if cell_len(authority) > 60:
        authority = _middle_elide_cells(authority, 60)
    return NotesStatusChannels(content, authority, safe)
