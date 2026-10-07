"""One seam for every single-file Library export (TASK-34000.3, review N-11).

The note (Markdown / text), prompt, Report-artifact and Collections
legacy-recovery exports each let the user pick any path in a ``FileSave``
dialog and then wrote it with a plain ``Path.write_text``: an existing file
outside Chatbook -- the user's own ``~/TODO.md`` -- was replaced without a
word, and the app reported "Export complete". Two notes titled "Reading
list" both default to ``~/Reading list.md``, so the collision is the default
path, not an edge case.

Every one of those exports now comes through here:

* ``request_export_write`` asks before an existing file is replaced. The
  prompt is the house ``ConfirmationDialog`` -- Cancel is the focused
  default, Escape and a backdrop click are Cancel, only the Replace button
  writes -- and it names the file and its folder. It is pushed with a result
  callback and never awaited from a handler or a ``call_after_refresh``
  callable (the W003 freeze shape, ``scripts/textual_wait_push_census.tsv``).
* ``write_export_text`` writes through a temporary file in the destination's
  own directory and ``os.replace``, so a write that fails partway leaves the
  previous file intact and no temporary file behind. A NEW file is published
  no-clobber, so one that appears after the seam looked is asked about, never
  replaced -- on a volume without hard links too (FAT32/exFAT sticks, many
  SMB/NFS mounts), where the shared writer reserves the name exclusively.
* ``remember_export_directory`` / ``library_export_picker_location`` make
  the next export picker in the same session open where the last export
  went, instead of always in the home folder.
* The success receipts name the full destination path, not just the file
  name.

The write bodies for the four exports live here rather than on
``LibraryScreen`` or their controllers, which are at or over their size
ratchets; the owners keep thin delegators. This module is imported lazily by
those delegators so it stays off the Library preimport closure.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from ...Utils.atomic_file_ops import atomic_write_text
from ...Utils.path_validation import validate_path_simple
from ...Widgets.confirmation_dialog import ConfirmationDialog

if TYPE_CHECKING:
    from ..Library_Modules.library_artifacts_controller import (
        LibraryArtifactsController,
    )
    from ..Library_Modules.library_collections_controller import (
        LibraryCollectionsController,
    )
    from ..Screens.library_screen import LibraryScreen

#: The replace prompt's title and button labels (ADR-031 modal grammar:
#: the safe choice is first, focused, and what Escape means).
EXPORT_REPLACE_TITLE = "Replace existing file?"
EXPORT_REPLACE_LABEL = "Replace"
EXPORT_KEEP_LABEL = "Cancel"

#: Where the session's last successful export went, kept on the running app
#: (one Library session, every export kind). Not persisted: a fresh launch
#: starts in the home folder again, as before.
_LAST_EXPORT_DIRECTORY_ATTR = "_library_last_export_directory"


# --- primitives -----------------------------------------------------------------


def describe_export_folder(folder: Path) -> str:
    """Name ``folder`` the way the prompt shows it: ``~/exp`` under home, else absolute.

    Args:
        folder: The folder to name. It is not read from disk.

    Returns:
        ``~`` for the home folder itself, ``~/<relative path>`` for a folder
        under it, and the absolute path for one anywhere else.
    """
    try:
        relative = folder.relative_to(Path.home())
    except ValueError:
        return str(folder)
    return "~" if str(relative) == "." else f"~/{relative.as_posix()}"


def describe_export_destination(destination: Path) -> str:
    """The destination as the receipts show it: its folder (``~``-shortened) and name.

    A one-line status region cannot show a 100-character absolute path (the
    prompt editor's status line clipped one to "Prompt exported successfully
    to"), so the home folder is shortened to ``~`` the way the picker shows
    it; anywhere else the path is absolute.

    Args:
        destination: The file that was written.

    Returns:
        ``<folder>/<file name>``, the folder named by ``describe_export_folder``.
    """
    return f"{describe_export_folder(destination.parent)}/{destination.name}"


def replace_prompt_message(destination: Path, picked: Path | None = None) -> str:
    """The replace prompt's body: the file, its folder, and what Replace does.

    Args:
        destination: The file that will actually be replaced.
        picked: The path the user chose when it was a symlink to
            ``destination``; the prompt then says so, so the user confirms
            the file that really changes.

    Returns:
        The prompt's message: the question, then one sentence on what
        Replace does.
    """
    message = (
        f'Replace "{destination.name}" in {describe_export_folder(destination.parent)}?'
    )
    if picked is not None and picked != destination:
        message += f' ("{picked.name}" in {describe_export_folder(picked.parent)} links to it.)'
    return message + "\n\nThe existing file will be overwritten by this export."


def resolve_export_destination(destination: Path) -> Path:
    """The file an export of ``destination`` really writes.

    A symlink to an existing file is written through -- the link stays and
    its target is replaced, which is what a plain ``write_text`` did before
    this seam, and what the user who created the link meant. ``os.replace``
    on the link itself would instead swap the link for a regular file and
    leave the target stale. A dangling link, or no link at all, is written
    at the chosen path.

    Args:
        destination: The validated path the user chose in the picker.

    Returns:
        The link's resolved target when ``destination`` is a symlink to an
        existing regular file; otherwise ``destination`` itself.

    Raises:
        OSError: ``destination`` could not be examined for a reason other
            than not being there -- its folder is not searchable, say.
    """
    if destination.is_symlink():
        try:
            target = destination.resolve(strict=True)
        except (OSError, RuntimeError):
            return destination
        if target.is_file():
            return target
    return destination


def export_destination_exists(destination: Path) -> bool:
    """Whether writing ``destination`` would replace something already there.

    A dangling symlink counts: ``os.replace`` would swap the link itself out.

    Args:
        destination: The path the export would write (see
            ``resolve_export_destination``).

    Returns:
        True when a file, a folder or a symlink -- dangling or not -- is at
        ``destination``; False when nothing is.

    Raises:
        OSError: ``destination`` could not be examined for a reason other
            than not being there -- its folder is not searchable, say.
    """
    return destination.is_symlink() or destination.exists()


def write_export_text(
    destination: Path, content: str, *, overwrite: bool = True
) -> None:
    """Write ``content`` to ``destination`` atomically (same-directory temp + replace).

    Keeps the existing file's permission bits when it is being replaced. Does
    not create a missing parent folder: the picker only hands back paths in
    folders it browsed, and a typed path with a typo should fail the way it
    always did rather than grow a new directory tree.

    Args:
        destination: The file to write (already resolved through
            ``resolve_export_destination`` by the seam's callers).
        content: The text to write.
        overwrite: ``False`` publishes no-clobber instead of replacing, so
            a file that appeared since the caller looked is never silently
            overwritten -- ``FileExistsError`` is raised and the caller
            asks. This closes the check-then-write window on the "nothing
            was there" path. The publish is a hard link where the volume has
            them and an exclusive create plus rename where it does not
            (``atomic_file_ops._publish_no_clobber``), so a new file also
            lands on a FAT32/exFAT stick or an SMB/NFS mount.

    Raises:
        FileNotFoundError: The destination's folder does not exist.
        FileExistsError: ``overwrite`` is False and something is there now.
            It is left untouched.
        OSError: The write or the publish failed; the previous file is
            intact, and no temporary file or placeholder is left behind.
    """
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"No such directory: {destination.parent}")
    atomic_write_text(
        destination,
        content,
        encoding="utf-8",
        preserve_existing_mode=True,
        privacy_safe_log=True,
        overwrite=overwrite,
    )


def library_export_picker_location(app: Any) -> str:
    """The folder the next export picker opens in: the last export's, else home.

    Args:
        app: The running app, which ``remember_export_directory`` records on.

    Returns:
        The folder this session last exported to, while it still exists;
        otherwise the home folder.
    """
    remembered = getattr(app, _LAST_EXPORT_DIRECTORY_ATTR, None)
    if remembered:
        folder = Path(remembered)
        try:
            if folder.is_dir():
                return str(folder)
        except OSError:
            pass
    return str(Path.home())


def remember_export_directory(app: Any, destination: Path) -> None:
    """Record ``destination``'s folder as where this session last exported to.

    Args:
        app: The running app. One that cannot take the attribute (a slotted
            or frozen stand-in) is left as it is.
        destination: The file just written, as the user chose it.
    """
    try:
        setattr(app, _LAST_EXPORT_DIRECTORY_ATTR, str(destination.parent))
    except (
        AttributeError,
        TypeError,
    ):  # a slotted or frozen stand-in: nothing to remember on
        logger.debug("library_export_directory_not_remembered")


def request_export_write(
    app: Any,
    destination: Path,
    *,
    on_replace: Callable[..., Any],
    on_cancel: Callable[[], Any],
    picked: Path | None = None,
) -> Any:
    """Run ``on_replace`` now, or after the user confirms replacing ``destination``.

    With nothing at ``destination`` the write proceeds at once as
    ``on_replace(overwrite=False)`` -- a no-clobber publish, so a file that
    appears between this check and the write raises ``FileExistsError`` and
    falls into the prompt instead of being overwritten. The result is
    returned (an async caller awaits it; an awaitable is wrapped so the same
    fall-through applies when it raises). Otherwise the replace prompt is
    pushed with a result callback and ``None`` is returned:
    ``on_replace(overwrite=True)`` runs only for an explicit Replace,
    ``on_cancel`` for the Cancel button, Escape, a backdrop click, or a
    prompt that closed without an answer. Either callback may be a coroutine
    function; Textual awaits what the result callback returns.

    Args:
        app: The running app (``screen.app``), which owns the screen stack.
        destination: The validated, resolved path the export would write
            (see ``resolve_export_destination``).
        on_replace: Performs the write and its receipt; takes ``overwrite``.
        on_cancel: Reports that nothing was written.
        picked: The path the user chose when ``destination`` is its symlink
            target; named in the prompt.

    Returns:
        ``on_replace``'s result when no prompt was needed; otherwise ``None``.

    Raises:
        OSError: ``destination`` could not be examined
            (``export_destination_exists``), or a synchronous ``on_replace``
            raised one. A ``FileExistsError`` is never raised: it opens the
            prompt.
    """

    def _answered(replace: bool | None) -> Any:
        return on_replace(overwrite=True) if replace is True else on_cancel()

    def _ask() -> None:
        app.push_screen(
            ConfirmationDialog(
                title=EXPORT_REPLACE_TITLE,
                message=replace_prompt_message(destination, picked),
                confirm_label=EXPORT_REPLACE_LABEL,
                cancel_label=EXPORT_KEEP_LABEL,
            ),
            callback=_answered,
        )

    if export_destination_exists(destination):
        _ask()
        return None
    try:
        outcome = on_replace(overwrite=False)
    except FileExistsError:
        _ask()
        return None
    if not inspect.isawaitable(outcome):
        return outcome

    async def _settle_or_ask() -> None:
        try:
            await outcome
        except FileExistsError:
            _ask()

    return _settle_or_ask()


async def _settle(outcome: Any) -> None:
    if inspect.isawaitable(outcome):
        await outcome


# --- the note export ---------------------------------------------------------------


def export_library_note_file(
    screen: LibraryScreen,
    validated_path: Path,
    export_format: str,
    title: str,
    content: str,
    keywords_text: str,
    note_id: str,
    operation: Any,
) -> None:
    """Write (after a replace check) the note export ``LibraryScreen`` validated.

    The body ``LibraryScreen._write_library_note_export_file`` carried before
    TASK-34000.3, with the replace prompt in front of the write, the write
    made atomic, and the receipt naming the full path. The screen keeps the
    validation step (``validate_path_simple``) and the operation token.

    Args:
        screen: The Library screen that owns the notes operation.
        validated_path: The destination, already through ``validate_path_simple``.
        export_format: ``"markdown"`` or ``"text"`` (``build_note_export_content``).
        title: The note title captured when Export was pressed.
        content: The note body captured when Export was pressed.
        keywords_text: The note's keywords as a comma-separated string.
        note_id: The note's id.
        operation: The claimed ``LibraryNotesOperationState`` token.
    """
    from ...Library.library_notes_state import build_note_export_content

    notify = getattr(screen.app_instance, "notify", None)
    target = resolve_export_destination(validated_path)

    def _write(*, overwrite: bool = True) -> None:
        try:
            write_export_text(
                target,
                build_note_export_content(
                    title, content, keywords_text, note_id, export_format
                ),
                overwrite=overwrite,
            )
        except FileExistsError:
            raise  # the seam asks instead of overwriting a file that appeared
        except Exception as exc:
            logger.warning(
                "Failed to export Library note {} (category={}).",
                note_id,
                type(exc).__name__,
            )
            if callable(notify):
                notify(f"Error exporting note: {type(exc).__name__}", severity="error")
            screen._finish_library_notes_operation(
                operation,
                success=False,
                failure_next_action="check the destination and try again",
            )
            return
        remember_export_directory(screen.app, validated_path)
        if callable(notify):
            notify(
                f"Note exported successfully to {describe_export_destination(target)}",
                severity="information",
            )
        screen._finish_library_notes_operation(operation, success=True)

    def _keep() -> None:
        if callable(notify):
            notify(
                f"Note export cancelled. {validated_path.name} was left unchanged.",
                severity="information",
            )
        # Cancel is the user's choice, not a failure: the status line says
        # what happened instead of the generic "Export failed" sentence.
        screen._finish_library_notes_operation(
            operation,
            success=False,
            failure_line=(
                f"Export cancelled — {validated_path.name} was left unchanged."
            ),
        )

    request_export_write(
        screen.app, target, on_replace=_write, on_cancel=_keep, picked=validated_path
    )


# --- the prompt export -------------------------------------------------------------


def export_library_prompt_file(
    screen: LibraryScreen,
    validated_path: Path,
    detail: Mapping[str, Any],
    prompt_id: int,
    notify: Callable[..., Any],
) -> None:
    """Write (after a replace check) the prompt export ``LibraryScreen`` validated.

    The body ``LibraryScreen._write_library_prompt_export_file`` carried
    before TASK-34000.3, with the replace prompt, the atomic write and the
    full-path receipt. The screen keeps validation and builds ``detail``.

    Args:
        screen: The Library screen.
        validated_path: The destination, already through ``validate_path_simple``.
        detail: The prompt record ``render_prompt_markdown`` renders.
        prompt_id: For the failure log only.
        notify: The screen's prompt-result reporter, already bound to ``prompt_id``.
    """
    from ...Prompt_Management.prompt_markdown_export import render_prompt_markdown

    target = resolve_export_destination(validated_path)

    def _write(*, overwrite: bool = True) -> None:
        try:
            write_export_text(
                target, render_prompt_markdown(detail), overwrite=overwrite
            )
        except FileExistsError:
            raise  # the seam asks instead of overwriting a file that appeared
        except Exception as exc:
            logger.warning(
                "Failed to export Library prompt {} (category={}).",
                prompt_id,
                type(exc).__name__,
            )
            notify(f"Error exporting prompt: {type(exc).__name__}", severity="error")
            return
        remember_export_directory(screen.app, validated_path)
        notify(
            f"Prompt exported successfully to {describe_export_destination(target)}",
            severity="information",
        )

    def _keep() -> None:
        notify(
            f"Prompt export cancelled. {validated_path.name} was left unchanged.",
            severity="information",
        )

    request_export_write(
        screen.app, target, on_replace=_write, on_cancel=_keep, picked=validated_path
    )


# --- the Report artifact export ----------------------------------------------------

REPORT_EXPORT_FAILED = "Report could not be exported. Check the destination and retry."


async def export_library_report_file(
    controller: LibraryArtifactsController,
    path: Any,
    report: Mapping[str, Any],
    profile: Any,
) -> None:
    """Write (after a replace check) a Report artifact as Markdown.

    The body ``LibraryArtifactsController._write_export`` carried before
    TASK-34000.3. The write stays off the UI loop (``asyncio.to_thread``):
    the picked destination can be a slow or network-mounted path.

    Args:
        controller: The artifacts controller that opened the picker.
        path: The picker's result, or a falsy value when it was cancelled.
        report: The briefing record to render.
        profile: The data profile the picker was opened under.
    """
    from ...Subscriptions.briefing_export import briefing_markdown_document

    if not path or controller.profile() != profile or controller.disposed:
        return
    try:
        destination = validate_path_simple(Path(path), require_exists=False)
    except Exception:  # noqa: BLE001 - preserve the mounted reader on owner failure
        controller.notify(REPORT_EXPORT_FAILED, "error")
        return

    target = resolve_export_destination(destination)

    async def _write(*, overwrite: bool = True) -> None:
        try:
            await asyncio.to_thread(
                write_export_text,
                target,
                briefing_markdown_document(report),
                overwrite=overwrite,
            )
        except FileExistsError:
            raise  # the seam asks instead of overwriting a file that appeared
        except Exception:  # noqa: BLE001 - preserve the mounted reader on owner failure
            controller.notify(REPORT_EXPORT_FAILED, "error")
            return
        remember_export_directory(controller.screen.app, destination)
        controller.notify(f"Report exported to {describe_export_destination(target)}")

    def _keep() -> None:
        controller.notify(f"Export cancelled. {target.name} was left unchanged.")

    await _settle(
        request_export_write(
            controller.screen.app,
            target,
            on_replace=_write,
            on_cancel=_keep,
            picked=destination,
        )
    )


# --- the Collections legacy-recovery export ----------------------------------------

#: How the recovery service says "something is at the destination" when it
#: was given no file to replace (``overwrite_identity=None``):
#: ``legacy_export_target_exists`` when the file is there as the export
#: starts, ``legacy_export_target_changed`` when it turns up between then
#: and the publish (``LegacyCollectionsRecovery._export_target`` and
#: ``_publish_export``). The second reason also covers a file that changed
#: under a CONFIRMED replace, so it is only read this way for a publish
#: that confirmed nothing.
_RECOVERY_DESTINATION_TAKEN_REASONS = frozenset(
    {"legacy_export_target_exists", "legacy_export_target_changed"}
)


def _recovery_found_destination_taken(exc: Exception, target: Path) -> bool:
    """Whether a refused no-clobber recovery publish is the Replace question.

    Args:
        exc: What ``export_json(..., overwrite_identity=None)`` raised.
        target: The destination that publish was for.

    Returns:
        True when the service refused because something is at ``target``
        and it is still there to ask about. False for every other failure,
        and for a name that was taken and released again.
    """
    if getattr(exc, "reason", None) not in _RECOVERY_DESTINATION_TAKEN_REASONS:
        return False
    try:
        return export_destination_exists(target)
    except OSError:
        return False


async def export_library_collections_recovery(
    controller: LibraryCollectionsController, selected_path: Path | None
) -> None:
    """Publish (after a replace check) a complete legacy-recovery JSON snapshot.

    The body ``LibraryCollectionsController._export_library_collection_legacy_
    recovery`` carried before TASK-34000.3. The recovery service already
    publishes atomically and guards the target's identity; what it lacked
    was the question. The replace check runs on the ``.json``-normalized
    destination, the path actually written. A file that appears after that
    check is refused by the service, and that refusal opens the same
    Replace prompt instead of being reported as a failed export.

    Args:
        controller: The collections controller that opened the picker.
        selected_path: The picker's result, or ``None`` when it was cancelled.
    """
    if selected_path is None:
        return
    recovery = getattr(
        controller.app_instance, "collections_legacy_recovery_service", None
    )
    if recovery is None:
        return
    try:
        destination = validate_path_simple(selected_path, require_exists=False)
        if destination.suffix.casefold() != ".json":
            destination = destination.with_suffix(".json")
    except Exception as exc:
        _report_recovery_export_failure(controller, exc)
        return

    target = resolve_export_destination(destination)

    async def _publish(*, overwrite: bool = True) -> None:
        # The target's identity is read only for an explicit Replace. Without
        # one, nothing was there when the seam checked and no one was asked:
        # the service gets ``overwrite_identity=None`` and is itself
        # no-clobber, refusing (``legacy_export_target_changed``) a file that
        # appeared since. Reading the identity here whatever ``overwrite``
        # said gave it permission to replace that file, so the no-clobber
        # claim held only by timing (final review M3). That refusal is the
        # late collision the other exports raise ``FileExistsError`` for, so
        # it is raised as one and the seam asks. After a Replace nothing is
        # re-raised: a refusal then is a failure, and this runs as the
        # prompt's own callback, where no one would catch it.
        try:
            overwrite_identity = None
            if overwrite and target.exists():
                metadata = target.lstat()
                overwrite_identity = (metadata.st_dev, metadata.st_ino)
            await asyncio.to_thread(
                recovery.export_json,
                target,
                overwrite_identity=overwrite_identity,
            )
        except Exception as exc:
            if not overwrite and _recovery_found_destination_taken(exc, target):
                raise FileExistsError(target.name) from None
            _report_recovery_export_failure(controller, exc)
            return
        remember_export_directory(controller.app, destination)
        controller._library_collections_action_status = (
            f"Legacy recovery export complete: {describe_export_destination(target)}"
        )
        controller._refresh_library_collections_capture_reader()

    def _keep() -> None:
        controller._library_collections_action_status = (
            f"Legacy export cancelled. {target.name} was left unchanged."
        )
        controller._refresh_library_collections_capture_reader()

    await _settle(
        request_export_write(
            controller.app,
            target,
            on_replace=_publish,
            on_cancel=_keep,
            picked=destination,
        )
    )


def _report_recovery_export_failure(
    controller: LibraryCollectionsController, exc: Exception
) -> None:
    reason = str(getattr(exc, "reason", "legacy_export_failed"))
    controller._library_collections_action_status = (
        f"Legacy export failed: {reason.replace('_', ' ')}."
    )
    controller._notify_library_collections_warning(reason)
    controller._refresh_library_collections_capture_reader()


__all__ = [
    "EXPORT_KEEP_LABEL",
    "EXPORT_REPLACE_LABEL",
    "EXPORT_REPLACE_TITLE",
    "describe_export_destination",
    "describe_export_folder",
    "export_destination_exists",
    "export_library_collections_recovery",
    "export_library_note_file",
    "export_library_prompt_file",
    "export_library_report_file",
    "library_export_picker_location",
    "remember_export_directory",
    "replace_prompt_message",
    "request_export_write",
    "resolve_export_destination",
    "write_export_text",
]
