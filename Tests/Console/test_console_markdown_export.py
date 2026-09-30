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


@pytest.mark.asyncio
@pytest.mark.parametrize("second_spelling", ["same", "through-symlinked-folder"])
async def test_two_saves_to_one_file_never_interleave(
    tmp_path, monkeypatch, second_spelling
) -> None:
    """Qodo #2932: a second save to the same file waits for the first.

    Writes are deliberately not exclusive workers (a later save must not
    cancel an earlier one), so two confirmed prompts naming one file can
    overlap -- plausible on a slow external or network disk. Each ``write_text``
    opens with truncation and writes through its own offset, so an overlap
    leaves the second export's head spliced onto the first one's tail. The
    first write is held open halfway through, as a stalled disk would; the
    second must not write until it finishes, and the file must end up exactly
    one whole export -- here the second, which could only start once the
    first was done. (Only that much is promised: the lock is not FIFO, so of
    several overlapping saves the file keeps one, not necessarily the last
    confirmed.) ``through-symlinked-folder`` names the same file by another
    path, which must still count as the same file.
    """
    import asyncio
    from pathlib import Path

    folder = tmp_path / "exports"
    folder.mkdir()
    target = folder / "chat.md"
    second_target = target
    if second_spelling == "through-symlinked-folder":
        link = tmp_path / "exports-link"
        link.symlink_to(folder, target_is_directory=True)
        second_target = link / "chat.md"
    first = "# First export\n" + "a" * 4000
    second = "# Later export\n" + "b" * 4000
    assert len(first) == len(second)

    halfway = threading.Event()
    resume = threading.Event()
    original_write_text = Path.write_text

    def stalling_write_text(self, data, *args, **kwargs):
        if data != first:
            return original_write_text(self, data, *args, **kwargs)
        with open(self, "w", encoding="utf-8") as handle:
            handle.write(data[: len(data) // 2])
            handle.flush()
            halfway.set()
            assert resume.wait(10), "the first write was never resumed"
            handle.write(data[len(data) // 2 :])
        return len(data)

    monkeypatch.setattr(Path, "write_text", stalling_write_text)

    first_save = asyncio.create_task(write_markdown_file(str(target), first))
    try:
        assert await asyncio.to_thread(halfway.wait, 10), "first write never started"
        second_save = asyncio.create_task(
            write_markdown_file(str(second_target), second)
        )
        finished, _pending = await asyncio.wait({second_save}, timeout=0.5)
        second_finished_mid_first = bool(finished)
    finally:
        resume.set()
    await first_save
    await second_save

    assert not second_finished_mid_first, (
        "the second save wrote while the first was still writing"
    )
    assert target.read_text(encoding="utf-8") == second


@pytest.mark.asyncio
async def test_a_stalled_save_does_not_hold_up_a_save_to_another_file(
    tmp_path, monkeypatch
) -> None:
    """Only writes to the same file wait for each other.

    A save stuck on a sleeping network disk must not stop a later save to a
    local folder, and once both finish no per-file lock is left behind.
    """
    import asyncio
    from pathlib import Path

    stuck_target = tmp_path / "stuck" / "chat.md"
    free_target = tmp_path / "free" / "chat.md"
    stuck = "# Stuck export\n"

    started = threading.Event()
    resume = threading.Event()
    original_write_text = Path.write_text

    def stalling_write_text(self, data, *args, **kwargs):
        if data == stuck:
            started.set()
            assert resume.wait(10), "the stalled write was never resumed"
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", stalling_write_text)

    stuck_save = asyncio.create_task(write_markdown_file(str(stuck_target), stuck))
    try:
        assert await asyncio.to_thread(started.wait, 10), "stalled write never began"
        saved = await asyncio.wait_for(
            write_markdown_file(str(free_target), "# Free export\n"), timeout=5
        )
    finally:
        resume.set()
    await stuck_save

    assert saved == free_target
    assert free_target.read_text(encoding="utf-8") == "# Free export\n"
    assert stuck_target.read_text(encoding="utf-8") == stuck
    assert markdown_export._file_locks == {}
