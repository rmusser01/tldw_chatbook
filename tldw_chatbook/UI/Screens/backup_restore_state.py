"""Pure evidence labels and copy rules for the canonical backup and restore view."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from tldw_chatbook.Utils.launch_options import (
    SECOND_MACHINE_GUIDE,
    SECOND_MACHINE_SECTION,
)

#: Titles per entry mode (TASK-34100.16): setup's "Restore a backup" opens the
#: view on Inspect, named for what the user came to do.
ENTRY_TITLES = {"home": "Backup & Restore", "inspect": "Restore from a backup"}

#: The archive format, named where an archive is chosen.
ARCHIVE_FORMAT_HINT = (
    "Choose a .tldw-backup.zip file (or .tldw-backup.zip.age when encrypted) "
    "made with Create backup."
)

FOLDER_NOT_ARCHIVE = "Choose the archive file, not a folder."

#: The line above a disabled Create backup until a review succeeds. One row at
#: the narrowest supported width: a wrap costs the scrolling form a row.
CREATE_NEEDS_REVIEW = "Create backup unlocks after a successful Review."


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

    Args:
        source: The absolute path the user chose.

    Returns:
        The message to show instead of inspecting, or None to inspect.
    """
    if source.is_dir():
        return FOLDER_NOT_ARCHIVE
    if source.suffix.lower() == ".toml":
        return (
            "That's a settings file, not a backup archive. To set up this "
            f"computer from it, see “{SECOND_MACHINE_SECTION}” in "
            f"{SECOND_MACHINE_GUIDE}, or start chatbook with: "
            f"tldw-cli --config {shlex.quote(str(source))}"
        )
    return None


def create_unavailable_reason(
    *,
    complete: bool,
    allow_partial: bool,
    capacity: Sequence[Mapping[str, object]],
    available: bool,
    unavailable_message: str,
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

    Returns:
        A sentence starting "Create backup is unavailable", or None.
    """
    if not available:
        return f"Create backup is unavailable: {unavailable_message}"
    short = next((row for row in capacity if not row["sufficient"]), None)
    if short is not None:
        return (
            "Create backup is unavailable: not enough free space at "
            f"{short['path']} ({short['required_bytes']:,} bytes needed, "
            f"{short['available_bytes']:,} free). Choose another location "
            "and review again."
        )
    if not (complete or allow_partial):
        return (
            "Create backup is unavailable: this would be a Partial backup "
            "(the coverage above lists what can't be captured). Tick "
            "“Acknowledge Partial archive…” and press Review again, or "
            "change the selection."
        )
    return None
