"""Structural-wait riders from the task-32055 review (task-32102).

The two items that live outside ``library_file_notes_workspace.py``: the
patience timer nobody retained, and the cancelled skill import whose own
receipt told the user to check a list it never refreshed. (The File Notes
half of the review is owned by task-32180 / PR #2557, which is rewriting
that row.)

Kept in its own file for the same reason: ``test_library_crit8_waits.py``
is that PR's surface.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import Tests.UI._optional_module_stubs  # noqa: F401


class FakeTimer:
    """The handle ``set_timer`` hands back, which nobody used to keep."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class WaitHost:
    """The three attributes the wait registry methods actually touch."""

    def __init__(self) -> None:
        from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

        self._library_structural_waits = {}
        self._library_structural_wait_timers = {}
        self.timers: list[FakeTimer] = []
        self._screen_cls = LibraryScreen

    def set_timer(self, delay: float, callback) -> FakeTimer:
        timer = FakeTimer()
        self.timers.append(timer)
        return timer

    def begin(self, label: str, owner: str, repaint=None):
        from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

        return LibraryScreen._begin_library_structural_wait(
            self, label, owner, repaint=repaint
        )

    def end(self, owner: str) -> None:
        from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

        LibraryScreen._end_library_structural_wait(self, owner)

    def _stop_library_structural_wait_timer(self, owner: str) -> None:
        from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

        LibraryScreen._stop_library_structural_wait_timer(self, owner)


@pytest.mark.unit
def test_settled_wait_stops_its_patience_timer() -> None:
    """A wait that ends before its patience window cannot still repaint.

    The one-shot ``set_timer`` was neither retained nor stopped, so an
    export or import that finished in under three seconds still fired a
    repaint into whatever the surface had become by then.
    """
    host = WaitHost()
    repaints: list[str] = []
    host.begin("Exporting", "export", repaint=lambda: repaints.append("export"))
    assert len(host.timers) == 1
    assert host.timers[0].stopped is False

    host.end("export")

    assert host.timers[0].stopped is True
    assert host._library_structural_wait_timers == {}


@pytest.mark.unit
def test_restarting_an_owners_wait_stops_the_previous_timer() -> None:
    """A second wait for the same owner replaces the first one's timer too."""
    host = WaitHost()
    host.begin("Changing folder", "file-notes", repaint=lambda: None)
    host.begin("Changing folder", "file-notes", repaint=lambda: None)

    assert host.timers[0].stopped is True
    assert host.timers[1].stopped is False
    assert len(host._library_structural_wait_timers) == 1


@pytest.mark.unit
def test_cancelled_skill_import_refreshes_the_list_it_names() -> None:
    """The receipt says "check the skills list", so the list must be current.

    An explicit Cancel cannot interrupt the worker thread, so the import may
    well have landed -- which is exactly why the copy sends the user to the
    list. It shipped with ``refresh_sources=False``, so that list was the
    pre-import one.
    """
    from tldw_chatbook.UI.Library_Modules.library_skill_import_controller import (
        LibrarySkillImportCoordinator,
    )

    coordinator = LibrarySkillImportCoordinator(app_instance=None)
    coordinator._cancel_requested = True

    outcome = coordinator._cancelled_outcome()

    assert "check the skills list" in outcome.status
    assert outcome.refresh_sources is True

    # A plain failure (no Cancel asked for) still refreshes nothing.
    assert coordinator._cancelled_outcome().refresh_sources is False


@pytest.mark.asyncio
async def test_a_settled_wait_never_repaints_on_the_mounted_screen() -> None:
    """The same rule, through the real screen and a real Textual timer.

    (Qodo #4) The registry tests above bind the unbound methods onto a
    three-attribute host, which pins the mechanism but not the wiring. This
    one starts a wait on a MOUNTED ``LibraryScreen``, ends it inside the
    patience window, and lets that window elapse: the repaint must never
    fire.
    """
    from tldw_chatbook.UI.Screens import library_screen
    from Tests.UI.test_library_shell import (
        LIBRARY_TEST_SIZE,
        LibraryHarness,
        _active_library_screen,
        _build_test_app,
        _wait_for_library_shell,
    )

    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        repaints: list[str] = []
        with patch.object(library_screen, "STRUCTURAL_WAIT_PATIENCE_SECONDS", 0.05):
            screen._begin_library_structural_wait(
                "Exporting",
                "export",
                repaint=lambda: repaints.append("export"),
            )
            assert "export" in screen._library_structural_wait_timers
            screen._end_library_structural_wait("export")

        assert screen._library_structural_wait_timers == {}
        await asyncio.sleep(0.15)
        await pilot.pause()
        assert repaints == [], "a settled wait still repainted its surface"
