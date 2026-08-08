"""task-3304 (MI-15): the file picker's listing must read as a table.

The vendored ``textual_fspicker`` dialog (this repo maintains its own copy
under ``Third_Party/`` and has patched it before -- TASK-378, task-1479,
task-2222) listed rows with no column headers, raw unitless byte counts,
and a size on directory rows -- so ``..`` showed "512" reading as a size --
above an unlabeled filename input. Pinned here:

- sizes humanize (``512 B`` / ``2.4 MB``), never a bare integer;
- directory rows (including ``..``) carry no size at all;
- the dialog shows a column header row (Name / Size / Modified);
- the filename input carries a visible label.
"""

from __future__ import annotations

import pytest
from rich.console import Console
from rich.style import Style
from textual.app import App
from textual.widgets import Input, Label

from tldw_chatbook.Third_Party.textual_fspicker import FileOpen
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import InputBar
from tldw_chatbook.Third_Party.textual_fspicker.parts.directory_navigation import (
    DirectoryEntry,
    DirectoryEntryStyling,
)


def _plain_styles() -> DirectoryEntryStyling:
    return DirectoryEntryStyling(Style(), Style(), Style(), Style())


def _rendered(entry: DirectoryEntry) -> str:
    console = Console(width=100)
    with console.capture() as capture:
        console.print(entry.prompt)
    return capture.get()


def _size_cell(rendered: str) -> str:
    """The 10-character size column of one rendered row.

    The entry grid's trailing columns are fixed-width (size 10, time 20,
    right pad 1), so slicing from the END is emoji-safe -- the folder/file
    icon earlier in the row is one character but two cells, which makes
    front-relative offsets lie.
    """
    line = rendered.rstrip("\n")
    return line[-(20 + 1 + 10) : -(20 + 1)].strip()


def test_file_sizes_are_humanized(tmp_path):
    small = tmp_path / "small.txt"
    small.write_bytes(b"x" * 512)
    big = tmp_path / "big.bin"
    big.write_bytes(b"\0" * 2_500_000)

    small_row = _rendered(DirectoryEntry(small, _plain_styles()))
    big_row = _rendered(DirectoryEntry(big, _plain_styles()))

    assert _size_cell(small_row) == "512 B", f"raw unitless size: {small_row!r}"
    assert _size_cell(big_row) == "2.4 MB", f"raw unitless size: {big_row!r}"
    assert "2500000" not in big_row


def test_directory_rows_carry_no_size(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "child.txt").write_text("x")

    dir_row = _rendered(DirectoryEntry(sub, _plain_styles()))
    updir_row = _rendered(DirectoryEntry(tmp_path / "..", _plain_styles()))

    # A directory's st_size (the 96/512-family on APFS) is meaningless to a
    # user and read exactly like a file size in the live capture -- the
    # ``..`` row "showing 512" was MI-15's headline.
    assert _size_cell(dir_row) == "", (
        f"directory row shows an apparent size: {dir_row!r}"
    )
    assert _size_cell(updir_row) == "", (
        f"'..' row shows an apparent size: {updir_row!r}"
    )


class _PickerHost(App):
    """Minimal host mirroring the shape callers push ``FileOpen`` with."""


@pytest.mark.asyncio
async def test_dialog_shows_column_headers(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    host = _PickerHost()
    async with host.run_test(size=(100, 30)) as pilot:
        await host.push_screen(FileOpen(location=str(tmp_path), title="Open"))
        await pilot.pause()
        header = host.screen.query_one("#file-dialog-column-headers")
        rendered = _render_static(header)
        for column in ("Name", "Size", "Modified"):
            assert column in rendered, (
                f"column header {column!r} missing: {rendered!r}"
            )


def _render_static(widget) -> str:
    console = Console(width=120)
    with console.capture() as capture:
        console.print(widget.renderable)
    return capture.get()


@pytest.mark.asyncio
async def test_filename_input_is_labeled(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    host = _PickerHost()
    async with host.run_test(size=(100, 30)) as pilot:
        await host.push_screen(FileOpen(location=str(tmp_path), title="Open"))
        await pilot.pause()
        bar = host.screen.query_one(InputBar)
        label = bar.query_one("#file-name-label", Label)
        assert "file name" in str(label.renderable).lower()
        # The label names the Input that follows it in the same bar; the
        # task-1479 scoped lookup (InputBar's first Input) must still
        # resolve to the filename field.
        assert bar.query_one(Input) is not None
