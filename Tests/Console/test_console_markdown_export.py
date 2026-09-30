"""The Console's Save .md… write step, against the real filesystem (TASK-33621.12).

``write_markdown_file`` is what runs after the user confirms a path in the
save prompt. The Console-level journeys (menu, prompt, toast) live in
``Tests/UI/test_console_conversation_action_menu.py``; these pin the write
step's own rules on real files under ``tmp_path``.
"""

from __future__ import annotations

import threading

import pytest

from tldw_chatbook.UI.Console_Modules import markdown_export
from tldw_chatbook.UI.Console_Modules.markdown_export import (
    MarkdownSaveError,
    write_markdown_file,
)


@pytest.mark.asyncio
async def test_a_folder_name_with_shell_characters_is_an_ordinary_folder(
    tmp_path,
) -> None:
    """';', '|', '$(' are legal in a folder name and never reach a shell here.

    The write used to refuse ``~/Notes; Q3/chat.md`` with "Path contains
    dangerous pattern: ;" -- a command-injection guard applied to a path that
    only ever reaches ``open()``.
    """
    target = tmp_path / "Notes; Q3 | $(draft)" / "chat.md"

    saved = await write_markdown_file(str(target), "# Chat\n")

    assert saved == target
    assert target.read_text(encoding="utf-8") == "# Chat\n"


@pytest.mark.asyncio
async def test_a_relative_path_comes_back_absolute(tmp_path, monkeypatch) -> None:
    """A bare file name lands in the working directory, and the caller is told
    exactly where, so the success message can name the folder."""
    monkeypatch.chdir(tmp_path)

    saved = await write_markdown_file("notes.md", "# Notes\n")

    assert saved.is_absolute()
    assert saved == tmp_path / "notes.md"
    assert saved.read_text(encoding="utf-8") == "# Notes\n"


@pytest.mark.asyncio
async def test_an_unknown_home_folder_is_a_plain_save_error(tmp_path) -> None:
    """``~someone/`` for a user who does not exist makes ``expanduser`` raise
    RuntimeError, which used to surface as "Could not save x.md: RuntimeError."
    """
    with pytest.raises(MarkdownSaveError) as caught:
        await write_markdown_file("~no-such-user-task-33621-12/chat.md", "# Chat\n")

    message = str(caught.value)
    assert message.startswith("Could not save chat.md:")
    assert "RuntimeError" not in message
    assert "home folder" in message


@pytest.mark.asyncio
async def test_a_file_in_the_folder_position_is_named(tmp_path) -> None:
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("occupied", encoding="utf-8")

    with pytest.raises(MarkdownSaveError) as caught:
        await write_markdown_file(str(blocker / "chat.md"), "# Chat\n")

    assert str(caught.value) == (
        f"Could not save chat.md: {blocker} is a file, not a folder."
    )
    assert blocker.read_text(encoding="utf-8") == "occupied"


@pytest.mark.asyncio
async def test_the_filesystem_work_runs_off_the_event_loop_thread(
    tmp_path, monkeypatch
) -> None:
    """Validation, the folder checks, mkdir and the write all block, so they
    run on a worker thread -- a sleeping external disk must not stall the UI."""
    seen: list[int] = []
    original = markdown_export._validate_and_write

    def _recording(path_text: str, markdown: str):
        seen.append(threading.get_ident())
        return original(path_text, markdown)

    monkeypatch.setattr(markdown_export, "_validate_and_write", _recording)

    await write_markdown_file(str(tmp_path / "chat.md"), "# Chat\n")

    assert seen and seen[0] != threading.get_ident()
    assert (tmp_path / "chat.md").exists()
