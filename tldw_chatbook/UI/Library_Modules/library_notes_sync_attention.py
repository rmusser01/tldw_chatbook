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
import threading
import weakref
from datetime import UTC, datetime
from functools import partial
from typing import Any

from loguru import logger

from ...Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
from .canvas_sync import _sync_library_canvas
from .screen_constants import LIBRARY_NOTES_SOURCE_DATABASE

#: How long the post-save re-read waits for the sync pass that save hinted.
#: The pass normally lands in well under a second; past this the row keeps
#: what it can say now and the next save or visit catches up. A clean quit
#: flush waits this long too before the app exits (final review I4).
SYNC_PASS_WAIT_SECONDS = 5.0

#: How long a SAVE of a synced note waits for the pass its previous save
#: hinted before it commits (final review I1). The executor fences a folder
#: when a note moves on between a pass admitting its ``update_file`` and that
#: write completing, so a save must not land inside that window. Shorter than
#: the navigation and quit flush bound (5 s): a flush that has to wait must
#: still have time to commit. Past this the save proceeds, and a hold it
#: causes is visible and healable, as before.
RESAVE_SYNC_PASS_WAIT_SECONDS = 3.0

#: ``_ask``'s answer when the runtime could not say (final review I3). It is
#: not the question's default: "nothing is held" is an answer, and a refused
#: or failed read is not one.
_COULD_NOT_SAY: Any = object()

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
    """Ask the runtime one read-only question.

    Returns:
        The answer; ``default`` when there is nothing to ask (no runtime, or
        one without this question); ``_COULD_NOT_SAY`` when the question
        failed. Both runtime reads are producer calls, so they are refused
        while a backup holds the producer fence. The caller keeps its last
        known answer then: painting ``default`` would say "nothing is held"
        over a held folder (final review I3).
    """

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
        return _COULD_NOT_SAY


async def await_sync_pass(runtime: Any, *, timeout: float | None = None) -> None:
    """Let the sync pass a save just hinted land, bounded.

    ``settle`` joins the runtime's own hint tasks; it is shielded so a caller
    that is superseded (``exclusive=True``) or times out can never cancel
    them. Three callers: the post-save re-read of the editor's line, a save of
    a note whose previous save hinted a pass (final review I1), and a clean
    quit flush (I4).

    Args:
        runtime: The sync runtime, or None when the app has none.
        timeout: The bound; ``SYNC_PASS_WAIT_SECONDS`` when not given.
    """

    settle = getattr(runtime, "settle", None)
    if not callable(settle):
        return
    bound = SYNC_PASS_WAIT_SECONDS if timeout is None else timeout
    with contextlib.suppress(Exception):
        await asyncio.wait_for(asyncio.shield(settle()), bound)


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
        await await_sync_pass(runtime)
    located = await _ask(runtime, "note_file_location", note_id, default="")
    if located is _COULD_NOT_SAY:
        # Keep the line as it is: "" here would say "not in a synced folder".
        return
    path = str(located or "")
    asked = (
        await _ask(runtime, "note_sync_needs_attention", note_id, default=False)
        if path
        else False
    )
    written = await asyncio.to_thread(note_file_written_label, path)
    if note_id != host._selected_note_id or host._library_notes_view != "editor":
        return
    previous = host._library_note_location
    attention = (
        # Could not say: keep what was last known about this same file.
        (previous[2] if previous[0] == path else False)
        if asked is _COULD_NOT_SAY
        else bool(asked)
    )
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
    ensure_library_notes_sync_attention_listener(host, runtime=runtime)
    refresh_manage_sync_folders_rows(host)
    held = await _ask(runtime, "attention_folder_ids", default=frozenset())
    if held is _COULD_NOT_SAY:
        # Keep the last known answer and paint nothing (final review I3).
        return
    folder_ids = frozenset(held or ())
    notes_state = host._notes_state
    if folder_ids == getattr(notes_state, "tree_attention_folder_ids", frozenset()):
        return
    note_id = getattr(host, "_selected_note_id", None)
    asked: Any = _COULD_NOT_SAY
    if note_id and getattr(host, "_library_note_location", ("", "", False))[0]:
        asked = await _ask(runtime, "note_sync_needs_attention", note_id, default=False)
    # Both reads are done. Nothing below awaits: this worker is exclusive, and
    # a successor that cancelled it between the write and the paint would find
    # the stored answer "unchanged" and never paint it (deferred minor T2-a).
    if folder_ids == getattr(notes_state, "tree_attention_folder_ids", frozenset()):
        return
    notes_state.tree_attention_folder_ids = folder_ids
    location = getattr(host, "_library_note_location", ("", "", False))
    if (
        asked is not _COULD_NOT_SAY
        and location[0]
        and note_id == host._selected_note_id
        and bool(asked) != location[2]
    ):
        host._library_note_location = (location[0], location[1], bool(asked))
        host._apply_library_note_presentation_state()
    if (
        host.is_mounted
        and host._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        and host._library_notes_source == LIBRARY_NOTES_SOURCE_DATABASE
        and host._library_notes_view in _ATTENTION_VIEWS
    ):
        _sync_library_canvas(host, "notes")


def refresh_manage_sync_folders_rows(host: Any) -> bool:
    """Re-project the Manage sync folders rows if that view is showing.

    TASK-32633 slice (N-03): the root rows are drawn from the runtime's last
    publications, and before this they were re-read only on entering the view
    or acting on a row -- a pass that finished while the user sat on the list
    (a note deleted or restored, a disk edit) changed nothing on screen. The
    same listener seam that turns the tree, list and editor turns these rows.
    Publishing is conditional on a change: the publication itself schedules
    this refresh again, so an unconditional publish would loop.

    Args:
        host: The Notes controller or screen.

    Returns:
        Whether the rows changed and were published.
    """

    controller = getattr(host, "_library_notes_sync_controller", None)
    if (
        controller is None
        or getattr(host, "_library_notes_view", "") != "lasting_roots"
    ):
        return False
    before = controller.snapshot.roots
    controller.refresh_roots(publish=False)
    if controller.snapshot.roots == before:
        return False
    controller.refresh_roots()
    return True


async def signal_library_note_lasting_sync(host: Any, note_id: str) -> tuple[str, ...]:
    """Tell lasting sync that ``note_id`` was just written, deleted or restored.

    TASK-32633 slice (review finding N-03). The Library editor's save goes
    through its session port, which hints the runtime (task-32604); the
    Library's Delete and Undo/Restore did not, so a deleted synced note left
    its root reading "✓ Up to date" with the file still on disk. This is the
    one seam for the Notes side's writes that bypass that port. What the hint
    does is the runtime's: a note-side deletion is a deletion review (the file
    is never removed on its own), a restore re-plans the held root.

    The write that just succeeded is never failed by its signal: no runtime
    yet, a refusing runtime or any error is ``()``, logged as metadata.

    Args:
        host: The Notes controller or screen.
        note_id: The note just written.

    Returns:
        The root ids the runtime hinted, for callers that report or test it.
    """

    runtime = _runtime(host)
    note_changed = getattr(runtime, "note_changed", None)
    if not note_id or not callable(note_changed):
        return ()
    try:
        return tuple(await note_changed(note_id) or ())
    except Exception as error:  # noqa: BLE001 - bounded, metadata only
        logger.warning(
            "Lasting sync was not signalled for a Library note write; error_type={}",
            type(error).__name__,
        )
        return ()


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


#: How long a burst of runtime status publications is coalesced before one
#: refresh runs. A pass publishes several statuses in a row (checking, then
#: the outcome); the refresh reads the final one.
STATUS_LISTENER_DEBOUNCE_SECONDS = 0.2

#: One listener per Notes host, so registration is idempotent and release
#: finds it. Weak on the host: a screen that is gone holds nothing here.
_STATUS_LISTENERS: weakref.WeakKeyDictionary[Any, LibraryNotesSyncAttentionListener] = (
    weakref.WeakKeyDictionary()
)


class LibraryNotesSyncAttentionListener:
    """Bridge the runtime's status publications to one debounced Notes refresh.

    Fix round 1 (TASK-34000.2 review, Important #2): a hold produced by a
    WATCHER pass while the user was idle (a disk conflict, a pass fenced at
    attention) reached the runtime's ``_root_status`` but no Notes surface
    until the next save, sync action or Library visit -- the finding's own
    failure shape, one interaction wide. The runtime now calls its status
    listeners from ``_publish``; this one marshals onto the app thread,
    coalesces a burst into a single :func:`schedule_library_notes_sync_attention`
    (itself exclusive), and never raises back into the runtime.
    """

    def __init__(self, host: Any) -> None:
        self._host = host
        self._pending = False

    def __call__(self, *_snapshot: object) -> None:
        """Receive one publication, from whatever thread ``_publish`` runs on."""

        try:
            app = getattr(self._host, "app_instance", None)
            if app is None:
                return
            thread_id = getattr(app, "_thread_id", None)
            if thread_id is None or thread_id == threading.get_ident():
                self._arm(app)
            else:
                app.call_from_thread(self._arm, app)
        except Exception as error:  # noqa: BLE001 - a status refresh, never the runtime
            logger.debug(
                "library_notes_sync_attention_listener_failed",
                error_type=type(error).__name__,
            )

    def _arm(self, app: Any) -> None:
        """On the app thread: start the debounce once per burst."""

        if self._pending:
            return
        self._pending = True
        set_timer = getattr(app, "set_timer", None)
        if callable(set_timer):
            set_timer(STATUS_LISTENER_DEBOUNCE_SECONDS, self.fire)
        else:
            self.fire()

    def fire(self) -> None:
        """Run the one refresh a burst earned."""

        self._pending = False
        schedule_library_notes_sync_attention(self._host)


def ensure_library_notes_sync_attention_listener(
    host: Any, *, runtime: Any = None
) -> None:
    """Register this host's listener with the runtime once (idempotent).

    Called from every refresh, because the runtime can start after the
    screen mounts; the runtime keeps one entry per listener object.
    """

    runtime = runtime if runtime is not None else _runtime(host)
    add = getattr(runtime, "add_status_listener", None)
    if host is None or not callable(add):
        return
    try:
        listener = _STATUS_LISTENERS.get(host)
    except TypeError:
        return
    if listener is None:
        listener = LibraryNotesSyncAttentionListener(host)
        _STATUS_LISTENERS[host] = listener
    with contextlib.suppress(Exception):
        add(listener)


def release_library_notes_sync_attention_listener(host: Any) -> None:
    """Unregister this host's listener (the screen's unmount pairing)."""

    try:
        listener = _STATUS_LISTENERS.pop(host, None)
    except TypeError:
        return
    if listener is None:
        return
    remove = getattr(_runtime(host), "remove_status_listener", None)
    if callable(remove):
        with contextlib.suppress(Exception):
            remove(listener)


__all__ = [
    "RESAVE_SYNC_PASS_WAIT_SECONDS",
    "STATUS_LISTENER_DEBOUNCE_SECONDS",
    "SYNC_PASS_WAIT_SECONDS",
    "LibraryNotesSyncAttentionListener",
    "await_sync_pass",
    "ensure_library_notes_sync_attention_listener",
    "library_notes_tree_folder_sets",
    "load_library_note_location",
    "note_file_written_label",
    "refresh_library_notes_sync_attention",
    "refresh_manage_sync_folders_rows",
    "release_library_notes_sync_attention_listener",
    "schedule_library_notes_sync_attention",
    "signal_library_note_lasting_sync",
]
