"""The entry reconcile's detach window reads through to the captured control.

TASK-34000.29 Task 7 fix round 1 (N-14): ``_capture_library_entry_focus``
detaches focus on purpose (``set_focus(None)``) until its restore, and a
Notes focus capture taken inside that window used to read nothing, so its
replay fell back to the filter -- which the entry restore then took for the
user's move (measured 1 in ~8: Escape in the bulk Move picker left focus in
the filter, not on Actions…). ``_capture_library_notes_focus_identity`` now
reads through to the capture's outgoing control while focus is still None,
the capture carries a Notes identity, and the route key matches.

Final review (parked T7 pin, ruled FIX): one mounted Notes boot pins the
rule and its three controls. Base (9a95797a2b): the filter fallback applies.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.library_notes_bulk_support import (
    enter_select_mode,
    mounted,
    notes_ready,
    row_for,
    until,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

WIDE = (160, 45)
ACTIONS = "library-notes-select-actions"


def _role(screen) -> str:
    return screen._capture_library_notes_focus_identity().semantic_role


async def _focus(screen, pilot, widget) -> None:
    screen.set_focus(widget)
    await until(pilot, lambda: screen.focused is widget, f"focus on #{widget.id}")


async def test_the_entry_window_reads_through_to_the_captured_control(
    tmp_path, monkeypatch
) -> None:
    """Main arm: focus Actions…, capture the entry focus (focus goes None),
    and the Notes capture names Actions…' role. Controls: nothing captured
    reads ``''``; a real move to the filter wins (``filter``); another route
    key reads ``''``. The boot ends on a real key."""
    host, _seed = mounted(tmp_path)
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        await enter_select_mode(host, pilot, screen)
        # Actions… is disabled until a row is checked: check one (a real
        # click toggles it in select mode), then focus Actions….
        await pilot.click(f"#{row_for(screen, 'Trip Oslo').id}")
        await until(
            pilot,
            lambda: not screen.query_one(f"#{ACTIONS}", Button).disabled,
            "Actions… to enable once a row is checked",
        )
        actions = screen.query_one(f"#{ACTIONS}", Button)
        await _focus(screen, pilot, actions)
        assert screen._library_notes_semantic_role(actions) == "select-actions"

        # Main arm: the detach window reads through to Actions….
        identity = screen._capture_library_entry_focus()
        assert identity is not None and identity.widget_id == ACTIONS
        assert screen.focused is None
        assert screen._library_entry_focus_capture is not None
        assert _role(screen) == "select-actions"

        # Control 1: nothing captured -> nothing read.
        screen._library_entry_focus_capture = None
        assert screen.focused is None
        assert _role(screen) == ""

        # Control 2: the user's own move wins over the capture.
        await _focus(screen, pilot, actions)
        assert screen._capture_library_entry_focus() is not None
        filter_box = screen.query_one("#library-notes-filter")
        await _focus(screen, pilot, filter_box)
        assert _role(screen) == "filter"

        # Control 3: a capture from another route is not read.
        await _focus(screen, pilot, actions)
        assert screen._capture_library_entry_focus() is not None
        assert screen.focused is None
        with pytest.MonkeyPatch.context() as other_route:
            other_route.setattr(screen, "_library_entry_route_key", lambda: ("other",))
            assert _role(screen) == ""
        assert _role(screen) == "select-actions"

        # End on a real key: Tab lands somewhere real.
        await pilot.press("tab")
        await until(
            pilot,
            lambda: screen.focused is not None,
            lambda: f"Tab to land focus (focus {screen.focused!r})",
        )
