"""Library's unsaved work when the app quits, and the blank-note GC.

TASK-34000.1 (review N-01). Ctrl+Q in Library ▸ Notes threw away whatever
was typed since the last autosave. The app's quit walk asks only the active
screen's ``confirm_quit`` / ``prepare_for_quit`` (``quit_confirmation_screens``
in ``Widgets/confirmation_dialog.py``); ``LibraryScreen`` had neither, and
its ``on_unmount`` cancels the pending debounced save. Only screen navigation
awaited ``flush_pending_work``.

* ``confirm_library_quit`` awaits that same flush -- Folder files, the
  Database note session, prompt and skill drafts -- so a quit persists what
  a tab switch would. After a clean flush it also waits, bounded, for the
  lasting-sync pass that flush hinted, so a synced note's file holds the last
  edit when the app exits. When the flush is vetoed (validation, conflict, a
  failed write), raises, or outlasts its wait, it asks
  'Quit and discard unsaved changes to "<title>"?' through the quit flow's
  one prompt choke point, ``await_quit_prompt`` (TASK-33622.10), with Keep
  editing as the default. It never exits silently past unsaved work.
* ``prepare_library_quit`` runs the untouched-blank-note GC once every quit
  prompt has been answered. Running it from ``confirm_quit`` would delete
  the row before a later prompt (Console runs, a workflow) could still answer
  Stay, leaving the editor open on a deleted note.
* ``gc_untouched_session_blank_note`` is that GC's one copy, shared with the
  navigation flush (``LibraryScreen._flush_library_note_save``).

The autosave policy (its maximum wait, and what a refused autosave may do)
lives in ``library_note_autosave.py``. This lives outside
``library_screen.py``, which is over its size ratchet, and the screen imports
it lazily so it stays out of the Library preimport census.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from ...Library.library_notes_session import NoteFlushOutcome, NoteFlushOutcomeKind
from ...Widgets.confirmation_dialog import confirm_quit_discarding_edits
from .screen_constants import LIBRARY_NOTE_BLANK_SEED_TITLE

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen

logger = logger.bind(module="LibraryScreen")

#: How long the quit waits for the flush when the app does not say; the same
#: bound navigation uses (``TldwCli.NAVIGATION_FLUSH_TIMEOUT_SECONDS``).
_DEFAULT_FLUSH_TIMEOUT_SECONDS = 5.0
#: Longest item name the quit prompt quotes before eliding it.
_NAME_MAX_CHARS = 48
#: Folder files save states that hold an edit not yet on disk.
_FILE_UNSAVED_STATES = frozenset({"dirty", "saving", "conflict", "error"})

#: Flushes the quit stopped waiting for. The wait is shielded, so the save
#: keeps running; asyncio holds only a weak reference to a task, so this one
#: keeps it alive until it finishes.
_RETAINED_FLUSHES: set[asyncio.Future[Any]] = set()


# --- Untouched blank note GC ---------------------------------------------------


async def gc_untouched_session_blank_note(
    screen: LibraryScreen, *, discard: bool = True
) -> bool:
    """Discard this session's new note if the user never really touched it.

    Args:
        screen: The Library screen whose open note is checked.
        discard: False only reports it. The quit flush (review #7) must not
            save a blank note's spaces -- the whitespace veto would then ask
            "discard?" about a note navigation discards without asking -- and
            ``prepare_for_quit`` discards it once the quit is approved.

    Returns:
        True when the open note is this session's effectively-empty new note
        (discarded when ``discard``); False otherwise.
    """
    state = screen._notes_state
    session = screen._library_note_session
    if not (
        state.session_blank_id
        and state.session_blank_id == state.selected_note_id
        # task-3315: never start a SECOND destructive op from the
        # untouched-blank GC while a discard/delete is already running
        # or admitted -- fall through to the session flush, whose own
        # destructive guard vetoes navigation until it settles.
        and not session.destructive_running
        and session.destructive_admission is None
    ):
        return False
    fields = screen._read_library_note_editor_fields()
    if fields is None:
        return False
    raw_title, raw_content, raw_keywords_text = fields
    # task-3315 (LIB-14 regression, pre-arc dev churn): the
    # session coordinator seeds a Blank note's title with the
    # literal seed and ``_read_library_note_editor_fields`` now
    # projects the SNAPSHOT rather than the widgets (13cf08f90,
    # notes-adaptive PR #1439) -- the editor presents that seed
    # as an empty placeholder-only Input, so it must count as
    # blank here or the untouched-blank GC never fires and every
    # abandoned Blank note leaves a permanent "Untitled" row
    # (exactly what task-2858 AC#5 forbids).
    # (P0, xhigh review + live-verify round) The seed only
    # counts as blank while it is still THE SEED. Keying on
    # string equality alone destroyed a note the user
    # deliberately titled "Untitled" (body empty) on
    # navigate-away, with no prompt and no undo -- a string
    # cannot tell the create seam's default from the same
    # letters typed by a human, so the provenance marker
    # decides. An emptied-out title is blank either way.
    # (rebase note: task-4021 independently re-derived this
    # same root cause -- the literal seed must count as blank
    # too, or this GC branch is unreachable -- but its version
    # lacked the ``_notes_state.title_user_edited`` provenance
    # guard below; dev's fuller check is kept as-is and covers
    # task-4021's reachability claim too.)
    title_blank = not raw_title.strip() or (
        raw_title == LIBRARY_NOTE_BLANK_SEED_TITLE and not state.title_user_edited
    )
    if not title_blank or any(
        value.strip() for value in (raw_content, raw_keywords_text)
    ):
        return False
    # task-32556 AC#1: a note the user never touched is
    # discarded silently on purpose (the guide documents
    # that). A title the user actually TYPED -- whitespace,
    # so still blank -- is a different event: keystrokes went
    # in, the row vanished, and nothing said so. Name it.
    if not discard:
        return True
    typed_a_blank_title = bool(raw_title) and not raw_title.strip()
    await screen._notes_controller._gc_pending_blank_note()
    if typed_a_blank_title:
        notify = getattr(screen.app_instance, "notify", None)
        if callable(notify):
            notify("Empty note discarded", severity="information")
    return True


# --- Quit ----------------------------------------------------------------------


async def confirm_library_quit(screen: LibraryScreen) -> bool:
    """Persist pending Library work before the app quits, or ask first.

    Must run inside a worker; the app's quit flow is one.

    Args:
        screen: The Library screen holding the work.

    Returns:
        True when the quit may proceed: everything flushed, or the user chose
        Discard and quit. False to stay, including when the prompt vanished.
    """
    reason = await _flush_for_quit(screen)
    if reason is None:
        await _settle_lasting_sync(screen)
        return True
    names = _unsaved_names(screen)
    if not names:
        return await confirm_quit_discarding_edits(
            screen,
            "A Library change has not finished saving. Quitting now may lose it.",
            title="Quit before Library finishes saving?",
            confirm_label="Quit anyway",
            cancel_label="Stay",
        )
    return await confirm_quit_discarding_edits(
        screen,
        f"{reason} Keep editing to fix or save them, or discard them and quit.",
        title=f"Quit and discard unsaved changes to {_quoted(names)}?",
    )


async def prepare_library_quit(screen: LibraryScreen) -> None:
    """Every quit prompt said yes: drop this session's untouched new note.

    Args:
        screen: The Library screen whose open note is checked.
    """
    await gc_untouched_session_blank_note(screen)


async def _settle_lasting_sync(screen: LibraryScreen) -> None:
    """Let the sync pass a clean quit flush hinted run before the app exits.

    Final review I4. The flush saves a synced note and hints its folder, and
    the pass that writes the file runs afterwards. The runtime's shutdown
    closes admission before a scheduled hint has run and starts no action
    once it is closing, so a quit that did not wait left the file without the
    last edit until the next launch. The wait is the editor's own post-save
    one: bounded (``SYNC_PASS_WAIT_SECONDS``) and shielded, so a slow folder
    delays the quit by that much at most and is never cancelled mid-write.
    With nothing in flight it returns at once.
    """
    runtime = getattr(
        getattr(screen, "app_instance", None), "notes_sync_runtime_owner", None
    )
    if runtime is None:
        return
    from .library_notes_sync_attention import await_sync_pass

    await await_sync_pass(runtime)


async def _flush_for_quit(screen: LibraryScreen) -> str | None:
    """Run the screen's pending-work flush, bounded like navigation's.

    Returns:
        None when everything was persisted; otherwise the sentence the quit
        prompt opens with.
    """
    task = asyncio.ensure_future(screen.flush_pending_work(quitting=True))
    _retain(task)
    timeout = getattr(
        screen.app, "NAVIGATION_FLUSH_TIMEOUT_SECONDS", _DEFAULT_FLUSH_TIMEOUT_SECONDS
    )
    try:
        allowed = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Library pending-work flush timed out before quitting")
        return "Saving is taking longer than expected."
    except Exception as error:  # noqa: BLE001 - a failed flush must ask, not exit
        logger.warning(
            "Library pending-work flush failed before quitting; error_type={}",
            type(error).__name__,
        )
        return "These changes could not be saved."
    if allowed is False:
        return _veto_reason(screen)
    return None


def _veto_reason(screen: LibraryScreen) -> str:
    """Why the flush refused, in the note's own words when it has them.

    Prompt and Skill drafts are explicit-Save only, so their veto means "not
    saved yet", never "the save failed".
    """
    snapshot = screen._library_note_session.snapshot
    status = (snapshot.status_message if snapshot is not None else "").strip()
    if snapshot is not None and snapshot.in_conflict:
        return "The note changed elsewhere since you opened it."
    if (
        snapshot is not None
        and snapshot.dirty
        and status
        and status != "Unsaved changes"
    ):
        return f"These changes could not be saved: {status}"
    return "These changes are not saved."


#: The head of every Esc-path veto sentence in
#: ``_library_note_editor_exit_veto_message`` (TASK-34000.29 owns that copy).
_ESC_VETO_HEAD = "Can't leave yet — "


def library_note_flush_veto_notice(
    outcome: NoteFlushOutcome, *, destination: str
) -> str:
    """The toast for a Notes flush veto behind a nav-bar click (TASK-34000.27).

    Review S-17: the veto used to be silent on this path, and N-25: the Esc
    path's validation sentence blamed the title for every veto.

    Args:
        outcome: The typed barrier result the flush refused with.
        destination: The clicked destination's nav-bar label (``"Console"``);
            empty when the caller does not know where the user was going.

    Returns:
        ``"Can't open <destination> yet: <the save's own message>"`` for a
        validation veto -- that message already names the field and the fix
        ("Title …", "Keywords …"), so a keyword veto never blames the title.
        Every other refusing kind reuses the Esc path's sentence
        (``_library_note_editor_exit_veto_message``) with the destination in
        its head, so the two paths never say different things. Empty for
        ``PERMITTED``.
    """
    if outcome.kind is NoteFlushOutcomeKind.PERMITTED:
        return ""
    head = _leave_head(destination)
    message = outcome.message.strip()
    if outcome.kind is NoteFlushOutcomeKind.VALIDATION_VETO and message:
        return f"{head}: {message}"
    # Lazy: the screen module imports this one lazily (its docstring says
    # why), and the sentence table lives beside the Esc path that owns it.
    from ..Screens.library_screen import _library_note_editor_exit_veto_message

    sentence = _library_note_editor_exit_veto_message(outcome.kind)
    if not sentence.startswith(_ESC_VETO_HEAD):
        return sentence
    return f"{head} — {sentence.removeprefix(_ESC_VETO_HEAD)}"


def library_file_notes_flush_veto_notice(*, destination: str) -> str:
    """The toast when a Folder-files draft refuses the flush (TASK-34000.27).

    The workspace shows its own save state in place; this names the refusal
    at the click, which used to get nothing.
    """
    return (
        f"{_leave_head(destination)} — the open file in Folder files isn't "
        "saved; check its status there first."
    )


def library_prompt_mutation_veto_notice(*, destination: str) -> str:
    """The toast when a prompt-collection mutation refuses the flush."""
    return (
        f"{_leave_head(destination)} — a prompt collection change is still in "
        "progress; wait for it to finish."
    )


def notify_library_flush_veto(screen: LibraryScreen, text: str) -> None:
    """Show one flush-veto toast through the app, when it can show one."""
    notify = getattr(screen.app_instance, "notify", None)
    if text and callable(notify):
        notify(text, severity="warning")


def _leave_head(destination: str) -> str:
    destination = destination.strip()
    return f"Can't open {destination} yet" if destination else "Can't leave yet"


def _unsaved_names(screen: LibraryScreen) -> list[str]:
    """Name every Library draft still holding unsaved work."""
    names: list[str] = []
    workspace = screen._notes_state.file_notes_workspace
    if workspace is not None and workspace.save_state in _FILE_UNSAVED_STATES:
        names.append(workspace.current_path or "the open file")
    snapshot = screen._library_note_session.snapshot
    if snapshot is not None and (
        snapshot.dirty or snapshot.saving or snapshot.in_conflict
    ):
        names.append(snapshot.title.strip() or LIBRARY_NOTE_BLANK_SEED_TITLE)
    if screen._prompts_state.dirty:
        names.append(screen._prompts_state.original_name.strip() or "new prompt")
    if screen._skills_state.dirty:
        names.append(screen._skills_state.original_name.strip() or "new skill")
    return names


def _quoted(names: list[str]) -> str:
    quoted = [
        f'"{name[: _NAME_MAX_CHARS - 1]}…"'
        if len(name) > _NAME_MAX_CHARS
        else f'"{name}"'
        for name in names
    ]
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _retain(task: asyncio.Future[Any]) -> None:
    _RETAINED_FLUSHES.add(task)
    task.add_done_callback(_release)


def _release(task: asyncio.Future[Any]) -> None:
    _RETAINED_FLUSHES.discard(task)
    if task.cancelled():
        return
    # Retrieved so asyncio never reports it as unhandled: an in-time failure
    # was logged by ``_flush_for_quit``, and a late one left its own error
    # state in the editor the user chose to keep.
    error = task.exception()
    if error is not None:
        logger.debug(
            "Retained Library quit flush failed; error_type={}", type(error).__name__
        )
