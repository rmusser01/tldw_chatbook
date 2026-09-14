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
    """task-32545 AC#3: every surface these terms lived on, rendered from production.

    Fix round 1: this test used to build its own snapshots and supply its own
    `receipt_line`, so four of the six terms were unreachable from it --
    "durable receipt" and "Review root status" are controller-produced,
    "managed placements" only renders in the `review` phase it never
    rendered, and "cutover" needed a `validation_message` it never populated.
    It now drives the controller to each phase and paints what production
    produced.
    """
    from Tests.UI.Library_Modules.test_library_notes_sync_controller import (
        TOKEN,
        _ImportController,
        _Runtime,
    )
    from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
        InertLastingSyncRuntime,
        LibraryNotesSyncController,
    )

    frames: list[str] = []

    def paint(snapshot, canvas: str, size: tuple[int, int]) -> str:
        return snapshot, canvas, size

    async def render(snapshot, canvas: str, size: tuple[int, int]) -> None:
        app = _Host(snapshot, canvas=canvas)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            frames.append(" ".join(_frame(app).split()))

    # 1. The inert profile: "cutover" lived in both the status line and the
    #    setup validation message.
    inert = LibraryNotesSyncController(
        runtime=InertLastingSyncRuntime(), import_controller=_ImportController()
    )
    assert inert.choose_relationship("keep_synced") == "choose"
    await render(inert.snapshot, "add", (100, 30))
    inert.set_setup("display_name", "Vault")
    inert.set_setup("folder", "/tmp/vault")
    assert "cutover" not in inert.snapshot.setup.validation_message
    frames.append(inert.snapshot.setup.validation_message)
    frames.append(inert.snapshot.status_line)

    # 2. review -> receipt on a working runtime: "managed placements" and
    #    "durable receipt recorded".
    live = LibraryNotesSyncController(
        runtime=_Runtime(), import_controller=_ImportController()
    )
    await live.check_root("root-1")
    assert live.snapshot.phase == "review"
    await render(live.snapshot, "add", (120, 40))
    await live.apply_reviewed("root-1", TOKEN)
    assert live.snapshot.phase == "receipt"
    await render(live.snapshot, "add", (100, 30))
    frames.append(live.snapshot.receipt_line)

    # 3. A refused check on the roots surface: "Review root status".
    failed = _failed_roots_controller(
        RuntimeError("sync_recovery_unresolved"), "up_to_date", "sync_now"
    )
    await failed.sync_now("root-1")
    await render(failed.snapshot, "roots", (120, 40))
    frames.append(failed.snapshot.status_line)

    # 4. The scroll cues are a pure widget concern: they need overflow at a
    #    small size, which only a rendered canvas produces.
    await render(
        replace(initial_lasting_sync_snapshot(lasting_available=True), phase="configure"),
        "add",
        (60, 12),
    )

    # task-32451 owns the root row's placeholder name; it is the one
    # remaining "cutover" on screen and is explicitly out of this task.
    placeholder = "Sync folder (name unavailable before cutover)"
    joined = "\n".join(frames)
    assert placeholder in joined, "the placeholder moved -- re-check task-32451"
    joined = joined.replace(placeholder, "<task-32451 placeholder>")
    assert "Check failed — recovery still open" in joined
    assert "listed under Receipts" in joined
    for term in _ENGINEERING_TERMS:
        assert term not in joined, term


async def test_receipt_rows_render_what_the_projection_produced() -> None:
    """The roots canvas paints a receipt row; the projection is pinned in the
    controller suite, so this is the widget half only."""
    app = _Host(_roots_snapshot(failed=True))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        painted = " ".join(_frame(app).split())
        assert "2026-09-13 15:20 · Wrote note to file · People/Sam.md · Sam" in painted


def _failed_roots_controller(error: BaseException, status: str, next_action: str):
    """A controller whose only root is `status` and whose Check raises `error`."""
    from Tests.UI.Library_Modules.test_library_notes_sync_controller import (
        _ImportController,
        _Runtime,
    )
    from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
        LibraryNotesSyncController,
    )

    runtime = _Runtime()
    runtime.snapshot = lambda: NotesSyncRuntimeSnapshot(
        "active",
        "sync_now",
        (NotesSyncRootRuntimeSnapshot("root-1", status, next_action),),
    )

    async def fail(_root_id: str):
        raise error

    runtime.request_sync_now = fail
    return LibraryNotesSyncController(
        runtime=runtime, import_controller=_ImportController()
    )


async def test_a_failed_row_offers_the_control_it_names() -> None:
    """task-32534 AC#1: the named next action must be reachable at the row.

    Driven through the controller, not a constructed row: the row shape is
    exactly what production projects after a refused Check.
    """
    controller = _failed_roots_controller(
        RuntimeError("sync_root_not_active"), "paused", "resume_sync"
    )
    await controller.sync_now("root-1")

    app = _Host(replace(controller.snapshot, write_receipts=()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query("#notes-sync-root-resume-0")
        assert not app.query("#notes-sync-root-pause-0")
        assert not app.query("#notes-sync-root-recover-0")
        painted = " ".join(_frame(app).split())
        assert "⚠ Needs attention · Check failed — folder is paused · Next: Resume" in painted


async def test_a_failed_check_on_an_offline_root_keeps_the_blocked_controls_blocked() -> (
    None
):
    """Fix round 1: the overlay must not re-enable what the canvas blocks.

    An offline root's Check is refused by the lease gate. An earlier overlay
    rewrote `status` to "needs_attention", which is what the canvas reads for
    `check_blocked` and for Pause suppression -- so the row came back with an
    ENABLED Check and a Pause button, on a folder that is disconnected.
    """
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRootRefused

    controller = _failed_roots_controller(
        NotesSyncRootRefused("root_lease_unavailable", reason_code="root_offline"),
        "offline",
        "reconnect_folder",
    )
    await controller.sync_now("root-1")

    row = controller.snapshot.roots[0]
    assert row.status == "offline"
    assert row.status_label == "⚠ Needs attention"
    assert row.failure == "Check failed — folder isn't available"
    assert row.failed_action == "reconnect_folder"

    app = _Host(replace(controller.snapshot, write_receipts=()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        check = app.query_one("#notes-sync-root-check-0", Button)
        assert check.disabled is True
        assert check.label.plain == (
            "○ Check changes unavailable — the folder is disconnected"
        )
        assert not app.query("#notes-sync-root-pause-0")
        assert not app.query("#notes-sync-root-resume-0")
