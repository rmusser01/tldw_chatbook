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
from pathlib import Path
from typing import Any

#: Render-and-prompt runs are exclusive in this group: choosing Save .md…
#: again supersedes a prompt still being prepared. Deliberately NOT the copy
#: group, so a Copy started meanwhile cannot silently cancel a save.
SAVE_MARKDOWN_WORKER_GROUP = "console-save-markdown"

#: Writes are never exclusive: a finished prompt's write must not be
#: cancelled by a later one, and a cancelled write would vanish without a
#: word (CancelledError is not an Exception).
WRITE_MARKDOWN_WORKER_GROUP = "console-save-markdown-write"


class MarkdownSaveError(Exception):
    """A markdown save that could not complete; the message is user-facing."""


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
    """Validate ``path_text`` and write ``markdown`` there without blocking.

    Args:
        path_text: The path as the user typed it; ``~`` is expanded.
        markdown: The rendered document.

    Returns:
        The validated path that was written.

    Raises:
        MarkdownSaveError: The path was refused or the file could not be
            written. The message names the file and the problem, and always
            starts with "Could not save".
    """
    import aiofiles

    from tldw_chatbook.Utils.path_validation import validate_path_simple

    # expanduser FIRST: validate_path_simple rejects unresolved '~'
    # components, and the expansion is exactly what a user means by it.
    candidate = Path(path_text).expanduser()
    try:
        target_path = validate_path_simple(candidate, require_exists=False)
    except ValueError as exc:
        raise MarkdownSaveError(
            f"Could not save {candidate.name or path_text}: {exc}"
        ) from exc

    parent = target_path.parent
    if parent.exists() and not parent.is_dir():
        raise MarkdownSaveError(
            f"Could not save {target_path.name}: {parent} is a file, not a folder."
        )
    try:
        parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target_path, "w", encoding="utf-8") as handle:
            await handle.write(markdown)
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
    screen.app.notify(f"Saved {saved.name}.", markup=False)
