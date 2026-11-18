"""Deferred Session Git row layout must stop during partial unmount (TASK-41248)."""

from __future__ import annotations

import asyncio
import inspect

import pytest
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widgets import ListView, Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Notes.file_notes_session_owner import (
    SessionChangeGroup,
    SessionGitRow,
    SessionGitRowState,
)
from tldw_chatbook.Widgets.Library.library_file_notes_git_panel import (
    _SessionGitListItem,
)


def _row(path: str, state: SessionGitRowState = "unstaged") -> SessionGitRow:
    return SessionGitRow(
        SessionChangeGroup(
            group_id=1,
            endpoints=(path,),
            source_path=path,
            destination_path=None,
            current_path=path,
            latest_action="modified",
            latest_sequence=1,
        ),
        state,
    )


class _RowHarness(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET

    def __init__(self) -> None:
        super().__init__()
        self.row = _SessionGitListItem(_row("before.md"))

    def compose(self) -> ComposeResult:
        yield ListView(self.row)


@pytest.mark.asyncio
async def test_live_row_fit_updates_copy_and_reports_missing_label() -> None:
    """Retirement must not disable live fitting or conceal a broken live row."""
    app = _RowHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        row = app.row
        assert row.is_running
        row.row = _row("after.md", "owned")
        row._fit_copy()
        primary = row.query_one(".file-notes-git-row-primary", Static)
        secondary = row.query_one(".file-notes-git-row-secondary", Static)
        assert "after.md" in str(primary.content)
        assert "before.md" not in str(primary.content)
        assert "STAGED" in str(secondary.content)

        secondary.remove_class("file-notes-git-row-secondary")
        try:
            with pytest.raises(NoMatches, match="file-notes-git-row-secondary"):
                row._fit_copy()
        finally:
            secondary.add_class("file-notes-git-row-secondary")


@pytest.mark.asyncio
async def test_queued_row_fit_during_partial_unmount_finishes_quietly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume the real queued fit after only its secondary label has detached."""
    app = _RowHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        row = app.row
        primary = row.query_one(".file-notes-git-row-primary", Static)
        secondary = row.query_one(".file-notes-git-row-secondary", Static)
        unmount_entered = asyncio.Event()
        unmount_release = asyncio.Event()
        fit_entered = asyncio.Event()
        fit_release = asyncio.Event()
        fit_attempted = asyncio.Event()
        original_unmount = getattr(Static, "on_unmount", None)

        async def held_unmount(widget: Static) -> None:
            if widget is primary:
                unmount_entered.set()
                await unmount_release.wait()
            if original_unmount is not None:
                result = original_unmount(widget)
                if inspect.isawaitable(result):
                    await result

        async def held_fit() -> None:
            fit_entered.set()
            await fit_release.wait()
            try:
                row._fit_copy()
            finally:
                fit_attempted.set()

        # Delay only the existing child lifecycle and the original callback;
        # do not fake attachment, pump state, queries, or the DOM contents.
        monkeypatch.setattr(Static, "on_unmount", held_unmount, raising=False)
        assert row.call_after_refresh(held_fit)
        removal = None
        try:
            await asyncio.wait_for(fit_entered.wait(), 10)
            removal = row.remove()
            await asyncio.wait_for(unmount_entered.wait(), 10)
            async with asyncio.timeout(10):
                while secondary.is_attached or row.is_running:
                    await asyncio.sleep(0)
            assert row.is_mounted and row.is_attached
            assert primary in row.children
            assert secondary not in row.children
            assert not row.is_running
            assert app.is_running
            fit_release.set()
            await asyncio.wait_for(fit_attempted.wait(), 10)
        finally:
            fit_release.set()
            unmount_release.set()
            if removal is not None:
                await asyncio.wait_for(removal, 10)
        assert fit_attempted.is_set()
        assert not row.is_attached
