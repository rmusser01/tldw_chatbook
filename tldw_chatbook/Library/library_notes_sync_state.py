"""Pure display-state for the Library notes sync panel."""
from __future__ import annotations

from dataclasses import dataclass

SYNC_DIRECTIONS = ("bidirectional", "disk_to_db", "db_to_disk")
SYNC_CONFLICTS = ("newer_wins", "disk_wins", "db_wins", "ask")
_DIRECTION_LABELS = {
    "bidirectional": "Bidirectional",
    "disk_to_db": "Disk → DB",
    "db_to_disk": "DB → Disk",
}
_CONFLICT_LABELS = {
    "newer_wins": "Newer wins",
    "disk_wins": "Disk wins",
    "db_wins": "DB wins",
    "ask": "Ask",
}


@dataclass(frozen=True)
class LibraryNotesSyncState:
    """Display state for the sync panel (folder, options, status, activity).

    Attributes:
        folder: The configured sync folder path (as text, not resolved).
        direction: One of ``SYNC_DIRECTIONS``.
        conflict: One of ``SYNC_CONFLICTS``.
        auto_sync: Whether auto-sync is enabled.
        status_line: The current status line, e.g. ``"idle"``,
            ``"syncing · 3/12"``, ``"done · 12 files · 2 conflicts"``, or
            ``"failed · <reason>"``.
        activity_lines: Most-recent-first activity log lines, capped at 20.
    """

    folder: str
    direction: str
    conflict: str
    auto_sync: bool
    status_line: str
    activity_lines: tuple[str, ...]


def next_sync_direction(value: str) -> str:
    """Cycle to the next sync direction in ``SYNC_DIRECTIONS`` order.

    An unknown ``value`` wraps around to the first direction rather than
    raising, so a stale/corrupt persisted preference degrades gracefully
    instead of crashing the cycling button.

    Args:
        value: The current sync direction.

    Returns:
        The next direction in ``SYNC_DIRECTIONS`` (wrapping past the end).
    """
    try:
        index = SYNC_DIRECTIONS.index(value)
    except ValueError:
        return SYNC_DIRECTIONS[0]
    return SYNC_DIRECTIONS[(index + 1) % len(SYNC_DIRECTIONS)]


def next_sync_conflict(value: str) -> str:
    """Cycle to the next conflict-resolution mode in ``SYNC_CONFLICTS`` order.

    An unknown ``value`` wraps around to the first mode rather than
    raising, mirroring ``next_sync_direction``.

    Args:
        value: The current conflict-resolution mode.

    Returns:
        The next mode in ``SYNC_CONFLICTS`` (wrapping past the end).
    """
    try:
        index = SYNC_CONFLICTS.index(value)
    except ValueError:
        return SYNC_CONFLICTS[0]
    return SYNC_CONFLICTS[(index + 1) % len(SYNC_CONFLICTS)]


def sync_direction_label(value: str) -> str:
    """Return the human label for a sync direction, raw value as fallback.

    Args:
        value: The sync direction value.

    Returns:
        The label from ``_DIRECTION_LABELS``, or ``value`` itself when
        unrecognized.
    """
    return _DIRECTION_LABELS.get(value, value)


def sync_conflict_label(value: str) -> str:
    """Return the human label for a conflict-resolution mode, raw value as fallback.

    Args:
        value: The conflict-resolution value.

    Returns:
        The label from ``_CONFLICT_LABELS``, or ``value`` itself when
        unrecognized.
    """
    return _CONFLICT_LABELS.get(value, value)


def sync_status_line(
    status: str,
    *,
    processed: int = 0,
    total: int = 0,
    conflicts: int = 0,
    error: str = "",
) -> str:
    """Render the sync panel's status line for a given status.

    Args:
        status: One of ``"idle"``/``"syncing"``/``"done"``/``"failed"``.
            Any other value renders as-is (no suffix).
        processed: Files processed so far (``"syncing"``/``"done"``).
        total: Total files to process (``"syncing"``/``"done"``).
        conflicts: Conflicts recorded (``"done"`` only; omitted from the
            line when zero).
        error: The failure reason (``"failed"`` only; omitted from the
            line when empty).

    Returns:
        The rendered status line, e.g. ``"syncing · 3/12"``,
        ``"done · 12 files · 2 conflicts"``, ``"done · 12 files"``,
        ``"failed · <reason>"``, ``"failed"``, or ``"idle"``.
    """
    if status == "syncing":
        return f"syncing · {processed}/{total}"
    if status == "done":
        line = f"done · {processed} files"
        if conflicts:
            line += f" · {conflicts} conflicts"
        return line
    if status == "failed":
        return f"failed · {error}" if error else "failed"
    return status


def append_activity(
    lines: tuple[str, ...], entry: str, *, cap: int = 20
) -> tuple[str, ...]:
    """Prepend a new activity entry, capping the log length.

    Args:
        lines: The current most-recent-first activity lines.
        entry: The new entry to record (becomes the newest, first line).
        cap: The maximum number of lines to retain.

    Returns:
        A new tuple with ``entry`` first, followed by the previous lines,
        truncated to ``cap`` entries.
    """
    return (entry, *lines)[:cap]
