"""A Trash landing settles only once its control has been laid out.

PR #3423 rebase 21 (2026-11-24). ``test_at_100x30_a_same_page_landing_
survives_a_sync_right_after_it`` went red in 2 of 48 instrumented runs under
8-way load: focus on the last row after a purge below the fold, the list at
offset 0, the row under it. In both reds every capture of the Trash list was
``off=None``, so no offset was replayed: the list was a fresh widget at 0
that nothing scrolled once it was laid out. In all 57 landings traced under
that load (the rebased head and the pre-rebase head alike) the landing's
keep-in-view (``LibraryNotesController._keep_landed_trash_control_in_view``)
settled -- scrolled, and dropped the marker that withholds the Trash list's
offset (ruling T8-j) -- before the landed control had any layout: the screen
ran its ``call_after_refresh`` callbacks before the recomposed list was laid
out, and the five refresh deferrals for a parked focus (ruling T8b-e) burned
out in about 6 ms. So the scroll did nothing, and the row was in view only
when a later one-shot scroll (a sync replay's deferred ``scroll_to_widget``)
happened to run after the layout.

Pinned without timing, at 100x30: Previous back to a full page 1 lands focus
on Next, the list's last row, below the fold. The app is held in
``batch_update`` from the press: the screen still runs its queued callbacks
but lays nothing out -- the order those landings saw under load (a
recomposing canvas holds the same batch). Right after the landing's first
keep-in-view check, ONE default Notes sync fires (a tree slice or a page read
landing there), and the batch is released only after that sync's replay, so
the replay's own one-shot scrolls run unlaid too. Base (2879640058): the check
scrolled the unlaid Next and dropped the marker, the sync replaced Next, and
nothing scrolled the replacement in once the list was laid out: Next at y=33
under a list ending at y=28, so the last wait times out. (That sync also
captured and replayed the unlaid list's offset, 0; that is incidental -- with
no Trash offset captured at all the base fails the same way, review M-3.)

First (review I-1), the same boot pins what the uncounted wait reads: the
canvas holding the Trash is the Notes work pane, a Notes sync queues its
replay there, and a check run while it is queued does not settle.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_library_notes_trash_pages_and_purges import (
    COMPACT,
    _focused_id,
    _mounted,
    _notes_tree_settled,
    _one,
    _painted,
    _settled_on,
    _visible_in,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_library_shell,
)
from tldw_chatbook.UI.Library_Modules.canvas_sync import _sync_library_canvas
from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
    LibraryNotesController,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import PostRecomposeCallback

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_at_100x30_a_landing_checked_before_layout_keeps_next_in_view(
    tmp_path, monkeypatch
) -> None:
    """The landed Next stays in view when its check runs before any layout."""
    host, _seed = _mounted(tmp_path, count=25)
    async with host.run_test(size=COMPACT) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _notes_tree_settled(host, pilot, screen, count=25)
        await pilot.click("#library-notes-trash-open")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: "Deleted 01" in _painted(host)
            and _focused_id(screen) == "library-notes-trash-restore-0",
            lambda: f"page 1 (focus: {_focused_id(screen)!r})",
        )
        await pilot.press(*(["tab"] * 20))
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: _focused_id(screen) == "library-notes-trash-next",
            lambda: f"Tab to Next (focus: {_focused_id(screen)!r})",
        )
        await pilot.press("enter")
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: "21–25 of 25 · Page 2 of 2" in _painted(host)
            and _focused_id(screen) == "library-notes-trash-previous",
            lambda: f"page 2 on Previous (focus: {_focused_id(screen)!r})",
        )

        # Review I-1: the canvas the check waits on, in the real Library. The
        # landed control's nearest ``PostRecomposeCallback`` ancestor is the
        # Notes work pane (also the canvas in Trash mode, for a check with no
        # control), and a Notes sync queues its replay on exactly that one.
        controller = screen._notes_controller
        previous = screen.query_one("#library-notes-trash-previous", Button)
        canvas = controller._library_notes_trash_canvas(previous)
        assert canvas is next(
            node for node in previous.ancestors if isinstance(node, PostRecomposeCallback)
        )
        assert canvas.id == "library-note-work-pane", canvas
        assert controller._library_notes_trash_canvas() is canvas
        role = controller._trash_focus_role
        assert role == "trash-next", role  # Next pressed; the landing is on Previous
        controller._trash_landing_settling = role  # a landing still settling
        _sync_library_canvas(screen, "notes")
        assert canvas.has_pending_recompose_callback
        # Checked now, on a control that is focused and laid out, it waits for
        # the queued sync instead of settling on the control it will replace.
        LibraryNotesController._keep_landed_trash_control_in_view(controller, role)
        assert controller._trash_landing_settling == role
        await _settled_on(
            pilot,
            host,
            screen,
            lambda: controller._trash_landing_settling == ""
            and not canvas.has_pending_recompose_callback
            and _focused_id(screen) == "library-notes-trash-previous",
            lambda: f"the queued sync's landing (focus: {_focused_id(screen)!r}, "
            f"settling: {controller._trash_landing_settling!r})",
        )

        # Class-level: the landing binds its check through the class.
        real_keep = LibraryNotesController._keep_landed_trash_control_in_view
        real_restore = controller._restore_library_notes_after_targeted_sync
        batch = host.batch_update()
        held: list[bool] = []
        synced: list[bool] = []

        def release() -> None:
            if held:
                held.clear()
                batch.__exit__(None, None, None)

        def keep_then_sync(controller, role: str, *waits: int) -> None:
            real_keep(controller, role, *waits)
            if not synced and role == "trash-previous":
                synced.append(True)
                _sync_library_canvas(screen, "notes")

        def restore_then_release(identity) -> None:
            real_restore(identity)
            if synced and held:
                # That sync's replay. Its one-shot scrolls (Textual's
                # deferred ``scroll_visible`` through the control's parent,
                # ``set_focus``'s ``call_later``) run before this release,
                # queued behind them on the same parent: still unlaid, as the
                # two traced reds' were. Only a check after the layout can
                # scroll the row in now.
                anchor = screen.focused.parent if screen.focused is not None else host
                anchor.call_after_refresh(release)

        monkeypatch.setattr(
            LibraryNotesController, "_keep_landed_trash_control_in_view", keep_then_sync
        )
        monkeypatch.setattr(
            controller, "_restore_library_notes_after_targeted_sync", restore_then_release
        )
        def next_in_view() -> bool:
            return (
                bool(synced)
                and _focused_id(screen) == "library-notes-trash-next"
                and _one(screen, "#library-notes-trash-restore-19") is not None
                and _visible_in(_one(screen, "#library-notes-trash-list"), screen.focused)
            )

        def where_next_is() -> str:
            listing = _one(screen, "#library-notes-trash-list")
            focused = screen.focused
            placed = (
                (listing.scroll_y, tuple(listing.region), tuple(focused.region))
                if listing is not None and focused is not None
                else None
            )
            return (
                f"Next on page 1, in view (focus: {_focused_id(screen)!r}, "
                f"synced: {synced}): {placed}"
            )

        batch.__enter__()
        held.append(True)
        try:
            await pilot.press("enter")
            # Waited for, not sampled (review M-4): the settle runs after the
            # release, a layout pass and a queued check.
            await _settled_on(pilot, host, screen, next_in_view, where_next_is)
        finally:
            release()
        assert "1–20 of 25 · Page 1 of 2" in _painted(host)
