"""A Trash landing settles only once its control has been laid out.

PR #3423 rebase 21 (2026-11-24). ``test_at_100x30_a_same_page_landing_
survives_a_sync_right_after_it`` went red 2 times in 40 instrumented runs
under 8-way load: focus on the last row after a purge below the fold, the
list at offset 0, the row under it. The trace named the cause. In all 48
landings traced under that load (the rebased head and the pre-rebase head
alike) the landing's keep-in-view (``LibraryNotesController._keep_landed_
trash_control_in_view``) settled -- scrolled, and dropped the marker that
withholds the Trash list's offset (ruling T8-j) -- before the landed control
had any layout: the screen ran its ``call_after_refresh`` callbacks before
the recomposed list was laid out, and the five refresh deferrals for a
parked focus (ruling T8b-e) burned out in about 6 ms. So the scroll did
nothing, and the row was in view only when a later one-shot scroll (a sync
replay's deferred ``scroll_to_widget``) happened to run after the layout.

Pinned without timing, at 100x30: Previous back to a full page 1 lands focus
on Next, the list's last row, below the fold. The app is held in
``batch_update`` from the press until the landing's first keep-in-view check
has run: the screen still runs its queued callbacks but lays nothing out --
the order those 48 landings saw under load. Right after that check, ONE
default Notes sync fires (a tree slice or a page read landing there).
Base (2879640058): the check dropped the marker over the unlaid row, the sync
read the new list's pre-layout offset (0) and replayed it, and nothing
scrolled the replacement Next in: Next at y=33 under a list ending at y=28.
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

        # Class-level: the landing binds its check through the class.
        real_keep = LibraryNotesController._keep_landed_trash_control_in_view
        controller = screen._notes_controller
        real_restore = controller._restore_library_notes_after_targeted_sync
        batch = host.batch_update()
        held: list[bool] = []
        synced: list[bool] = []

        def release() -> None:
            if held:
                held.clear()
                batch.__exit__(None, None, None)

        def keep_then_sync(controller, role: str, deferrals: int = 0) -> None:
            real_keep(controller, role, deferrals)
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
        batch.__enter__()
        held.append(True)
        try:
            await pilot.press("enter")
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: bool(synced)
                and _focused_id(screen) == "library-notes-trash-next"
                and _one(screen, "#library-notes-trash-restore-19") is not None,
                lambda: f"page 1 on Next (focus: {_focused_id(screen)!r}, "
                f"synced: {synced})",
            )
        finally:
            release()
        listing = screen.query_one("#library-notes-trash-list")
        assert _visible_in(listing, screen.focused), (
            listing.scroll_y,
            tuple(listing.region),
            tuple(screen.focused.region),
        )
        assert "1–20 of 25 · Page 1 of 2" in _painted(host)
