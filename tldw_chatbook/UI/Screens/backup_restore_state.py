"""Pure evidence labels and copy rules for the canonical backup and restore view."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath

from tldw_chatbook.Utils.launch_options import SECOND_MACHINE_SECTION

#: Titles per entry mode (TASK-34100.16): setup's "Restore a backup" opens the
#: view on Inspect, named for what the user came to do.
ENTRY_TITLES = {"home": "Backup & Restore", "inspect": "Restore from a backup"}

#: The archive format, named where an archive is chosen.
ARCHIVE_FORMAT_HINT = (
    "Choose a .tldw-backup.zip file (or .tldw-backup.zip.age when encrypted) "
    "made with Create backup."
)

FOLDER_NOT_ARCHIVE = "Choose the archive file, not a folder."

#: A config.toml chosen as an archive. No path in it: at the narrowest width
#: (54 columns) the message line shows four rows, and a path of any length
#: would push the command and the guide pointer out of view. The path is in
#: the field right above, and ``tldw-cli --help`` links the published guide.
SETTINGS_FILE_NOT_ARCHIVE = (
    "That's a settings file, not a backup archive. To use it, start chatbook "
    "with: tldw-cli --config <this file>. Guide: "
    f"“{SECOND_MACHINE_SECTION}” (link in tldw-cli --help)."
)

#: The line above a disabled Create backup until a review succeeds. One row at
#: the narrowest supported width: a wrap costs the scrolling form a row.
CREATE_NEEDS_REVIEW = "Create backup unlocks after a successful Review."

#: The line above Create once a backup is accepted: pressing Create voids the
#: review, and this stays true through the run and after it succeeds or fails.
CREATE_STARTED = (
    "Backup started; progress is shown at the top. Press Review to create another."
)


def typed_path(value: str) -> Path:
    """A path typed into the view, home-expanded without ever raising.

    ``Path.expanduser()`` raises RuntimeError for an unknown ``~user`` (a typo
    such as ``~mike/Downloads/x.tldw-backup.zip``) on Python 3.12, and an
    exception out of a button handler exits the whole app, with setup under
    it (TASK-34100.16 review). ``os.path.expanduser`` leaves such a path as
    typed, so the view then refuses it as not a full path.

    Args:
        value: The field's text.

    Returns:
        The path, with ``~`` and a known ``~user`` expanded.
    """
    return Path(os.path.expanduser(value))


def result_label(
    *,
    archive_verified: bool,
    restoration_validated: bool,
    opened: bool,
    needs_setup: bool,
) -> str:
    """Keep archive integrity distinct from restored data and successful opening."""
    if needs_setup:
        return "Needs setup"
    if opened:
        return "Opened successfully"
    if restoration_validated:
        return "Restoration validated"
    return "Archive verified" if archive_verified else "Not verified"


def archive_source_problem(source: Path) -> str | None:
    """Name an input that cannot be a backup archive, before inspecting it.

    A config.toml and a folder used to reach the archive reader and fail as
    the generic ``backup_operation_failed`` (TASK-34100.16).

    Never raises: ``os.path.isdir`` reports an unreadable path as "not a
    folder" (``Path.is_dir()`` raises PermissionError on Python 3.12), so
    such an archive goes on to the inspection, which reports its own failure.

    Args:
        source: The absolute path the user chose.

    Returns:
        The message to show instead of inspecting, or None to inspect.
    """
    if os.path.isdir(source):
        return FOLDER_NOT_ARCHIVE
    if source.suffix.lower() == ".toml":
        return SETTINGS_FILE_NOT_ARCHIVE
    return None


def create_unavailable_reason(
    *,
    complete: bool,
    allow_partial: bool,
    capacity: Sequence[Mapping[str, object]],
    available: bool,
    unavailable_message: str,
    destination: PurePath | None = None,
) -> str | None:
    """Why a reviewed backup cannot be created, or None when it can.

    The single rule for Create backup after a review: the view enables the
    button exactly when this returns None, and otherwise shows the reason
    beside it (TASK-34100.16 -- Create stayed disabled with no reason shown).

    Args:
        complete: Whether the reviewed archive would be complete.
        allow_partial: Whether the user acknowledged a Partial archive.
        capacity: The review's per-volume rows (``path``,
            ``required_bytes``, ``available_bytes``, ``sufficient``).
        available: The service's backup availability verdict.
        unavailable_message: The service's plain explanation when it is not
            available.
        destination: The reviewed backup file. A short volume that does not
            hold it is the staging volume (the system temporary folder),
            which the form cannot change, so its advice differs.

    Returns:
        The reason, leading with what to do, or None.
    """
    if not available:
        return f"Create backup is unavailable: {unavailable_message}"
    short = next((row for row in capacity if not row["sufficient"]), None)
    if short is not None:
        volume = PurePath(str(short["path"]))
        # The service names each volume by the file's nearest existing
        # ancestor, spelled as abspath spells it (no links resolved).
        target = None if destination is None else PurePath(os.path.abspath(destination))
        holds_destination = (
            target is None or target == volume or volume in target.parents
        )
        # Action first, figures last: at 54 columns the line shows four rows,
        # and the coverage list above repeats every volume's figures.
        needed = (
            f"{short['required_bytes']:,} bytes needed, "
            f"{short['available_bytes']:,} free"
        )
        if holds_destination:
            return (
                "Not enough free space for this backup: choose another location "
                f"and review again ({needed} at {volume})."
            )
        return (
            "Not enough free space in the system temporary folder, where the "
            f"backup is staged: free some space there and review again ({needed})."
        )
    if not (complete or allow_partial):
        # Action first: the line shows four rows at 54 columns.
        return (
            "To create a Partial backup, tick “Acknowledge Partial archive…” "
            "and press Review again, or change the selection."
        )
    return None
