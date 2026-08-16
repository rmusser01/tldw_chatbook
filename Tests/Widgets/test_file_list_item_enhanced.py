# test_file_list_item_enhanced.py
# Description: Regression coverage for task-16844 (FileListItemEnhanced passed an
# unsupported ``tooltip=`` kwarg to ``Static``, crashing compose() for any
# non-empty ``FileListEnhanced.files``)
"""
task-16844: ``FileListItemEnhanced.compose()`` used to construct the file-name
row with ``Static(..., tooltip=str(self.file_path))``. Textual 8.2.8's
``Static.__init__`` takes only ``content, *, expand, shrink, markup, name, id,
classes, disabled`` -- no ``tooltip`` -- so mounting a ``FileListEnhanced``
with even one file raised, deterministically:

    TypeError: FileListItemEnhanced(id='file-item-...') compose() method
    returned an invalid result; Static.__init__() got an unexpected keyword
    argument 'tooltip'

``FileListEnhanced.files`` is a ``recompose=True`` reactive, so the crash
fires the moment a real row is composed -- the empty-list "No files selected"
placeholder path never exercises it, which is why no existing test caught it.

This test was born red against the pre-fix tree (the exact TypeError above)
and is green once the tooltip is set as a post-construction attribute
instead of a constructor kwarg.

Reachability note (task-16844 AC #1): a repo-wide grep found no production
caller of ``FileListEnhanced``/``FileListItemEnhanced`` outside this
definition file -- nothing composes it in the live app today. The fix is
applied anyway since it is a one-line, zero-risk correction that un-breaks
the widget for whichever future caller ends up mounting it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Widgets.file_list_item_enhanced import (
    FileListEnhanced,
    FileListItemEnhanced,
)


class _FileListEnhancedApp(App[None]):
    """Minimal host mounting a FileListEnhanced with one real file."""

    def __init__(self, files: list[Path]) -> None:
        super().__init__()
        self._files = files

    def compose(self) -> ComposeResult:
        yield FileListEnhanced(files=self._files, id="fle")


@pytest.mark.asyncio
async def test_file_list_enhanced_with_files_composes_without_error(tmp_path) -> None:
    """A non-empty ``files`` list must compose cleanly.

    Born red pre-fix: this raised ``TypeError: ... Static.__init__() got an
    unexpected keyword argument 'tooltip'`` from inside
    ``FileListItemEnhanced.compose()``.
    """
    target = tmp_path / "example.txt"
    target.write_text("hello")

    app = _FileListEnhancedApp(files=[target])
    async with app.run_test() as pilot:
        await pilot.pause()

        fle = app.query_one("#fle", FileListEnhanced)
        assert list(fle.files) == [target]

        item = app.query_one(FileListItemEnhanced)
        assert item.file_path == target


@pytest.mark.asyncio
async def test_file_list_item_name_static_carries_the_intended_tooltip(
    tmp_path,
) -> None:
    """The file-name ``Static`` must actually carry the path as its tooltip
    (the behavior the removed constructor kwarg was trying to achieve)."""
    target = tmp_path / "report.pdf"
    target.write_text("pdf-bytes")

    app = _FileListEnhancedApp(files=[target])
    async with app.run_test() as pilot:
        await pilot.pause()

        name_static = app.query_one(".file-name", Static)
        assert name_static.tooltip == str(target)


@pytest.mark.asyncio
async def test_file_list_enhanced_empty_files_still_shows_placeholder() -> None:
    """The empty-list path (the one every prior no-op test exercised) must
    keep working after the fix."""
    app = _FileListEnhancedApp(files=[])
    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.query(".no-files-message").nodes
        assert not app.query(FileListItemEnhanced).nodes
