"""Save .md… for one Console conversation row (TASK-25886, TASK-33621.12).

The conversation action menu's Copy as ▸ Save .md… renders the Clean markdown
off the UI loop, asks for a destination in ``ConsoleSaveMarkdownModal``, and
writes the file.

This flow used to live on ``ChatScreen`` and called ``self.push_screen``. A
``Screen`` has no ``push_screen`` (only the ``App`` does), so the worker raised
``AttributeError``, Textual treated it as fatal, and one menu click ended the
app with the user's tabs and drafts in it (2026-09-29 Console UX review,
G3-02). The modal is pushed through the app here, and every way a save can
fail -- the chat cannot be read, the path is refused, the file cannot be
written -- ends in a visible error instead of an exception. Nothing here
logs: the toast names the problem, and a traceback of this flow would carry
the user's transcript and paths into the persistent log.

Lazily imported from the screen's action handler (ADR-097), so none of this is
on the boot path.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

#: Render-and-prompt runs are exclusive in this group: choosing Save .md…
#: again supersedes a prompt still being prepared. Deliberately NOT the copy
#: group, so a Copy started meanwhile cannot silently cancel a save.
SAVE_MARKDOWN_WORKER_GROUP = "console-save-markdown"

#: Writes are never exclusive: a finished prompt's write must not be
#: cancelled by a later one, and a cancelled write would vanish without a
#: word (CancelledError is not an Exception). Two writes to the SAME file are
#: serialized by ``_one_writer_per_file`` instead.
WRITE_MARKDOWN_WORKER_GROUP = "console-save-markdown-write"

#: Destination -> (lock, number of writes holding or waiting for it). An
#: entry lives only while some write for that file is in flight.
_file_locks: dict[str, tuple[threading.Lock, int]] = {}
_file_locks_guard = threading.Lock()


class MarkdownSaveError(Exception):
    """A markdown save that could not complete; the message is user-facing."""


@contextmanager
def _one_writer_per_file(path: Path) -> Iterator[None]:
    """Hold the write lock for ``path`` until the block ends (Qodo #2932).

    Each write opens its file with truncation and writes through its own
    offset, so two overlapping writes to one file -- two confirmed prompts
    naming it while a slow disk still has the first -- leave the second
    export's head spliced onto the first one's tail. A write waits here for
    any earlier write to the same file instead; it is never cancelled, and
    writes to other files never wait. Keyed by the real path, so ``~/x.md``,
    the absolute spelling, and a path through a symlinked folder share one
    lock. Blocks, so it runs on the write's worker thread, never the UI loop.
    """
    key = os.path.realpath(path)
    with _file_locks_guard:
        lock, users = _file_locks.get(key, (threading.Lock(), 0))
        _file_locks[key] = (lock, users + 1)
    try:
        with lock:
            yield
    finally:
        with _file_locks_guard:
            lock, users = _file_locks[key]
            if users == 1:
                del _file_locks[key]
            else:
                _file_locks[key] = (lock, users - 1)


async def save_conversation_markdown(screen: Any, target: Any) -> None:
    """Render one conversation, prompt for a path, and write it there.

    Args:
        screen: The Console ``ChatScreen``. Supplies the renderer, the app to
            push the prompt on and notify through, and the worker host.
        target: The ``ConversationMenuTarget`` captured when the menu opened.
    """
    from tldw_chatbook.Widgets.Console.console_save_markdown_modal import (
        ConsoleSaveMarkdownModal,
        markdown_filename_slug,
    )

    try:
        # The paginated read + render are blocking work; coroutine workers
        # still run on the UI loop, so push them off it (PR #2262 review).
        markdown = await asyncio.to_thread(
            screen._render_console_conversation_markdown, target, "clean"
        )
    except Exception as exc:  # noqa: BLE001 - a save must never end the app
        screen.app.notify(
            f"Could not read this chat to save it ({type(exc).__name__}).",
            severity="error",
        )
        return
    if markdown is None:
        screen.app.notify("This chat has no messages to save.", severity="warning")
        return

    title = str(getattr(target, "title", "") or "")
    default_path = str(
        Path.home() / "Downloads" / f"{markdown_filename_slug(title)}.md"
    )

    def _write(chosen: "str | None") -> None:
        if not chosen:
            return
        screen.run_worker(
            _write_and_report(screen, chosen, markdown),
            exclusive=False,
            group=WRITE_MARKDOWN_WORKER_GROUP,
        )

    # The App owns the screen stack; a Screen has no push_screen (G3-02).
    screen.app.push_screen(
        ConsoleSaveMarkdownModal(default_path=default_path), callback=_write
    )


async def write_markdown_file(path_text: str, markdown: str) -> Path:
    """Validate ``path_text`` and write ``markdown`` there, off the UI loop.

    Args:
        path_text: The path as the user typed it; ``~`` is expanded, and a
            relative path is taken from the working directory.
        markdown: The rendered document.

    Returns:
        The absolute path that was written.

    Raises:
        MarkdownSaveError: The path was refused or the file could not be
            written. The message names the file and the problem, and always
            starts with "Could not save".
    """
    # Every step blocks on the filesystem (validation stats the path), and a
    # coroutine worker runs on the UI loop -- a sleeping external disk or a
    # network mount must not freeze the Console.
    return await asyncio.to_thread(_validate_and_write, path_text, markdown)


def _validate_and_write(path_text: str, markdown: str) -> Path:
    """The blocking half of :func:`write_markdown_file`; runs on a thread."""
    from tldw_chatbook.Utils.path_validation import validate_path_simple

    typed = Path(path_text)
    try:
        # expanduser FIRST: validate_path_simple rejects unresolved '~'
        # components, and the expansion is exactly what a user means by it.
        candidate = typed.expanduser()
    except RuntimeError as exc:
        # "~someone/..." for a user this machine does not have.
        raise MarkdownSaveError(
            f"Could not save {typed.name or path_text}: "
            "the home folder in this path does not exist."
        ) from exc
    try:
        # Only open()/stat() ever see this path, never a shell, so ';', '|'
        # and '$(' are ordinary characters in a folder name here.
        validated = validate_path_simple(
            candidate, require_exists=False, reject_shell_metacharacters=False
        )
    except ValueError as exc:
        raise MarkdownSaveError(
            f"Could not save {candidate.name or path_text}: {exc}"
        ) from exc

    # Absolute, so the success message can say where a bare name landed.
    target_path = validated.absolute()
    parent = target_path.parent
    with _one_writer_per_file(target_path):
        if parent.exists() and not parent.is_dir():
            raise MarkdownSaveError(
                f"Could not save {target_path.name}: {parent} is a file, not a folder."
            )
        try:
            parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(markdown, encoding="utf-8")
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            raise MarkdownSaveError(
                f"Could not save {target_path.name} to {parent}: {reason}."
            ) from exc
    return target_path


async def _write_and_report(screen: Any, path_text: str, markdown: str) -> None:
    """Write the export and tell the user how it went -- never raise."""
    # markup=False throughout: these messages carry the path exactly as the
    # user typed it, and "[...]" in a folder name must not be read as markup.
    try:
        saved = await write_markdown_file(path_text, markdown)
    except MarkdownSaveError as exc:
        screen.app.notify(str(exc), severity="error", markup=False)
        return
    except Exception as exc:  # noqa: BLE001 - a save must never end the app
        screen.app.notify(
            f"Could not save {Path(path_text).name}: {type(exc).__name__}.",
            severity="error",
            markup=False,
        )
        return
    screen.app.notify(f"Saved {saved.name} to {saved.parent}.", markup=False)
