"""Wave-4 pins for Manage sync folders (task-32534, task-32545).

Retarget/Disconnect state their reason at the control, focused root
buttons carry the shape cue and are named by the footer, and the sync copy
carries no engineering terms.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncRootRow,
    LastingSyncWriteReceipt,
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncRootRuntimeSnapshot,
    NotesSyncRuntimeSnapshot,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)
from tldw_chatbook.Widgets.Library.library_notes_sync_roots_canvas import (
    LibraryNotesSyncRootsCanvas,
)
from tldw_chatbook.app import TldwCli

pytestmark = pytest.mark.asyncio

_ENGINEERING_TERMS = (
    "durable receipt",
    "managed placements",
    "cutover",
    "Review root status",
    "content is scrollable",
    "effects are scrollable",
)


class _Host(App[None]):
    CSS_PATH: ClassVar[list[str]] = [*TldwCli.CSS_PATH, *LibraryScreen.CSS_PATH]

    def __init__(self, snapshot, *, canvas: str = "roots") -> None:
        super().__init__()
        self.snapshot = snapshot
        self.canvas = canvas

    def compose(self) -> ComposeResult:
        if self.canvas == "roots":
            yield LibraryNotesSyncRootsCanvas(self.snapshot)
        else:
            yield LibraryNotesAddFromFilesCanvas(self.snapshot)


class _JourneyHarness(LibraryHarness):
    CSS_PATH = TldwCli.CSS_PATH


def _frame(app: App[None]) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


def _roots_snapshot(*, failed: bool = False):
    row = LastingSyncRootRow(
        "root-1",
        "Vault",
        "needs_attention" if failed else "up_to_date",
        "resolve_cleanup" if failed else "sync_now",
        "⚠ Needs attention" if failed else "✓ Up to date",
        "Resolve recovery" if failed else "Check changes",
        failure="Check failed — recovery still open" if failed else "",
    )
    return replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="roots",
        status_line="Sync folders refreshed.",
        roots=(row,),
        write_receipts=(
            LastingSyncWriteReceipt(
                "2026-09-13 15:20", "Wrote note to file", "People/Sam.md", "Sam"
            ),
        ),
    )


async def test_retarget_and_disconnect_state_their_reason_at_the_control() -> None:
    """task-32545 AC#1: the disabled reason is the label, in the server-row grammar."""
    app = _Host(_roots_snapshot())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        retarget = app.query_one("#notes-sync-root-retarget-0", Button)
        disconnect = app.query_one("#notes-sync-root-disconnect-0", Button)
        assert retarget.disabled and disconnect.disabled
        assert retarget.label.plain == "○ Retarget unavailable — not in this release"
        assert disconnect.label.plain == "○ Disconnect unavailable — not in this release"
        painted = " ".join(_frame(app).split())
        assert (
            "Retarget/Disconnect unavailable — not in this release; nothing on disk "
            "or in Notes changes." in painted
        )


async def test_focused_roots_buttons_carry_the_shape_cue_and_the_footer_names_them() -> (
    None
):
    """task-32545 AC#2: focus is shape-cued and the footer names the focused button."""
    root = NotesSyncRootRuntimeSnapshot("root-1", "up_to_date", "sync_now")
    runtime = SimpleNamespace(
        snapshot=lambda: NotesSyncRuntimeSnapshot("active", "sync_now", (root,))
    )
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.notes_sync_runtime_owner = runtime
    host = _JourneyHarness(app)
    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        manage = await _wait_for_selector(
            screen, pilot, "#library-notes-manage-sync-folders"
        )
        manage.press()
        check = await _wait_for_selector(screen, pilot, "#notes-sync-root-check-0")
        check.focus()
        await pilot.pause()

        assert screen.focused is check
        assert check.has_class("library-canvas-action")
        assert ("enter", "check changes") in screen._library_notes_footer_shortcuts()

        pause = screen.query_one("#notes-sync-root-pause-0", Button)
        pause.focus()
        await pilot.pause()
        assert ("enter", "pause") in screen._library_notes_footer_shortcuts()
        assert screen._library_focus_enter_label(pause) == "pause"
        assert screen._library_focus_enter_label(
            screen.query_one("#notes-sync-root-retarget-0", Button)
        ) == ""


async def test_sync_copy_uses_no_engineering_terms() -> None:
    """task-32545 AC#3: roots, review, receipt and setup surfaces are user-facing."""
    frames: list[str] = []
    app = _Host(_roots_snapshot(failed=True))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        painted = " ".join(_frame(app).split())
        frames.append(painted)
        assert "Check failed — recovery still open" in painted
        assert "2026-09-13 15:20 · Wrote note to file · People/Sam.md · Sam" in painted

    base = initial_lasting_sync_snapshot(lasting_available=False)
    for phase, extra in (
        ("choose", {}),
        ("configure", {}),
        (
            "receipt",
            {"status_line": "Sync root activated.", "receipt_line": "60 applied · listed under Receipts"},
        ),
    ):
        snapshot = replace(base, phase=phase, **extra)
        app = _Host(snapshot, canvas="add")
        async with app.run_test(size=(60, 12)) as pilot:
            await pilot.pause()
            frames.append(" ".join(_frame(app).split()))

    joined = "\n".join(frames)
    for term in _ENGINEERING_TERMS:
        assert term not in joined, term
