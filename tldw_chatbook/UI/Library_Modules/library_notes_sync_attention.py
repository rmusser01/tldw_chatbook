"""Truthful lasting-sync health for the Notes tree, list and editor (TASK-34000.2).

Review finding N-02: an end-of-note edit wedged a synced folder -- nothing
reached the file and later disk edits never arrived -- while every surface
outside Manage sync folders went on saying all was well: the tree row read
"⇄ Sync managed", the Notes list read "Library notes · Ready" and the editor
read "Saved 21:40" over "In a synced folder". A user kept writing on both
sides and the copies diverged.

This module owns the UNHEALTHY half of that answer. It asks the sync runtime
which folders, and whether the open note's folder, are held for attention
(an open entry, a conflict or deletion to review, a failed pass), keeps the
answer on the screen's Notes state, and re-paints the Notes canvas when it
changes. Healthy-state wording belongs elsewhere (TASK-32633's slice); the
surfaces that render these facts give "needs attention" precedence over any
healthy copy.

Both helpers take the Notes "host" -- ``LibraryNotesController`` or the
``LibraryScreen`` it delegates to -- and read only attributes both expose.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import UTC, datetime
from functools import partial
from typing import Any

from loguru import logger

from ...Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
from .canvas_sync import _sync_library_canvas
from .screen_constants import LIBRARY_NOTES_SOURCE_DATABASE

#: How long the post-save re-read waits for the sync pass that save hinted.
#: The pass normally lands in well under a second; past this the row keeps
#: what it can say now and the next save or visit catches up.
SYNC_PASS_WAIT_SECONDS = 5.0

#: The Notes views whose canvas paints the tree, the list or the editor.
_ATTENTION_VIEWS = frozenset({"editor", "list"})


def library_notes_tree_folder_sets(notes_state: Any) -> dict[str, frozenset[str]]:
    """Return the tree builder's three folder-status sets from Notes state.

    One place that knows which state fields the tree's folder rows read, so
    the screen's tree call stays one line.

    Args:
        notes_state: The screen's ``LibraryNotesState``.

    Returns:
        Keyword arguments for ``build_paged_library_notes_tree``.
    """

    return {
        "protected_folder_ids": getattr(
            notes_state, "tree_protected_folder_ids", frozenset()
        ),
        "inactive_managed_folder_ids": getattr(
            notes_state, "tree_inactive_managed_folder_ids", frozenset()
        ),
        "attention_folder_ids": getattr(
            notes_state, "tree_attention_folder_ids", frozenset()
        ),
    }


def note_file_written_label(path: str) -> str:
    """Local "when the file was last written", or "" if it cannot be read.

    The file's own mtime, not a record of our writes: a vault edited in
    Obsidian and a note saved here are the same question to the reader, and
    only the filesystem answers both. Local time in the codebase's
    absolute-timestamp spelling, built tz-aware then localised (task-32640).
    """

    if not path:
        return ""
    try:
        modified = os.stat(path).st_mtime
    except OSError:
        return ""
    return (
        datetime.fromtimestamp(modified, tz=UTC).astimezone().strftime("%Y-%m-%d %H:%M")
    )


def _runtime(host: Any) -> Any:
    return getattr(
        getattr(host, "app_instance", None), "notes_sync_runtime_owner", None
    )


async def _ask(runtime: Any, name: str, *args: Any, default: Any) -> Any:
    """Ask the runtime one read-only question; any failure is ``default``."""

    method = getattr(runtime, name, None)
    if not callable(method):
        return default
    try:
        return await method(*args)
    except Exception as error:  # noqa: BLE001 - a status line, never the note
        # Metadata only: these answers are about folders and file paths.
        logger.debug(
            "library_notes_sync_attention_failed",
            question=name,
            error_type=type(error).__name__,
        )
        return default


async def _await_sync_pass(runtime: Any) -> None:
    """Let the pass a save just hinted land before the line is re-read.

    ``settle`` joins the runtime's own hint tasks; it is shielded so this
    worker being superseded (``exclusive=True``) can never cancel them.
    """

    settle = getattr(runtime, "settle", None)
    if not callable(settle):
        return
    with contextlib.suppress(Exception):
        await asyncio.wait_for(asyncio.shield(settle()), SYNC_PASS_WAIT_SECONDS)


async def load_library_note_location(
    host: Any, note_id: str, *, after_save: bool = False
) -> None:
    """Answer "where does this note live, and is that file in step?".

    task-32640: the path and the file's write time are read LIVE every time,
    never carried on the note record. TASK-34000.2 adds whether that file's
    folder is held for attention, and -- after a save -- waits (bounded) for
    the pass the save hinted, so the save that wedges a folder is the save
    whose row says so, instead of the one after it.

    Args:
        host: The Notes controller or screen.
        note_id: The note whose file, if any, to name.
        after_save: Whether a save just hinted the sync runtime.
    """

    if not note_id:
        return
    runtime = _runtime(host)
    if after_save and runtime is not None:
        await _await_sync_pass(runtime)
    path = str(await _ask(runtime, "note_file_location", note_id, default="") or "")
    attention = bool(path) and bool(
        await _ask(runtime, "note_sync_needs_attention", note_id, default=False)
    )
    written = await asyncio.to_thread(note_file_written_label, path)
    if note_id != host._selected_note_id or host._library_notes_view != "editor":
        return
    host._library_note_location = (path, written, attention)
    host._apply_library_note_presentation_state()
    await refresh_library_notes_sync_attention(host, runtime=runtime)


async def refresh_library_notes_sync_attention(
    host: Any, *, runtime: Any = None
) -> None:
    """Re-read which sync folders are held and re-paint Notes if that changed.

    Args:
        host: The Notes controller or screen.
        runtime: The sync runtime, when the caller already has it.
    """

    runtime = runtime if runtime is not None else _runtime(host)
    folder_ids = frozenset(
        await _ask(runtime, "attention_folder_ids", default=frozenset()) or ()
    )
    notes_state = host._notes_state
    if folder_ids == getattr(notes_state, "tree_attention_folder_ids", frozenset()):
        return
    notes_state.tree_attention_folder_ids = folder_ids
    note_id = getattr(host, "_selected_note_id", None)
    location = getattr(host, "_library_note_location", ("", "", False))
    if note_id and location[0]:
        attention = bool(
            await _ask(runtime, "note_sync_needs_attention", note_id, default=False)
        )
        if attention != location[2] and note_id == host._selected_note_id:
            host._library_note_location = (location[0], location[1], attention)
            host._apply_library_note_presentation_state()
    if (
        host.is_mounted
        and host._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        and host._library_notes_source == LIBRARY_NOTES_SOURCE_DATABASE
        and host._library_notes_view in _ATTENTION_VIEWS
    ):
        _sync_library_canvas(host, "notes")


def schedule_library_notes_sync_attention(host: Any) -> None:
    """Run one :func:`refresh_library_notes_sync_attention` as a Notes worker.

    Exclusive in its own group: a newer ask supersedes an older one, and
    the runtime calls it awaits are read-only. A host without workers (a
    test double, a screen not yet wired) asks nothing.
    """

    run_worker = getattr(host, "run_worker", None)
    if host is None or not callable(run_worker):
        return
    # A callable, not a coroutine object: an exclusive worker superseded
    # before it starts would otherwise leave a coroutine that is never
    # awaited (``RuntimeWarning`` in the Pilot run). Textual unwraps a
    # ``partial`` of a coroutine function itself.
    run_worker(
        partial(refresh_library_notes_sync_attention, host),
        exclusive=True,
        group="library_notes_sync_attention",
    )


__all__ = [
    "SYNC_PASS_WAIT_SECONDS",
    "library_notes_tree_folder_sets",
    "load_library_note_location",
    "note_file_written_label",
    "refresh_library_notes_sync_attention",
    "schedule_library_notes_sync_attention",
]
