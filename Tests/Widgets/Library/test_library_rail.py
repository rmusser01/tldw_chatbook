"""Tests for the LibraryRail widget."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.Library.library_rail_state import LibraryRailPreferences
from tldw_chatbook.Library.library_shell_state import LibraryShellState
from tldw_chatbook.Widgets.Library.library_rail import LibraryRail


pytestmark = pytest.mark.asyncio


def _make_shell() -> LibraryShellState:
    """Return a minimal Library shell state for rail tests."""
    return LibraryShellState(
        header_line="Library | Test",
        sections=(),
        details_lines=(),
        selected_row_id="",
        canvas_kind="empty",
        canvas_target="",
        canvas_empty_copy="",
    )


async def test_library_rail_top_action_factory(widget_pilot):
    """The top_action_factory is stored and its widgets are rendered first."""
    factory = lambda: [Button("Ingest", id="library-top-action")]
    preferences = LibraryRailPreferences()

    async with await widget_pilot(
        LibraryRail,
        shell=_make_shell(),
        preferences=preferences,
        top_action_factory=factory,
    ) as pilot:
        rail = pilot.app.test_widget
        assert rail.top_action_factory is factory

        await pilot.pause()
        assert isinstance(pilot.app.query_one("#library-top-action", Button), Button)
        assert isinstance(pilot.app.query_one("#library-search-input", Input), Input)


# -- task-670: RecomposeCaptureGuard extended to LibraryRail ---------------
# LibraryRail.sync_state() drives `self.refresh(recompose=True)`; before this
# fix the rail carried no guard against task-637's bug class.


async def test_post_recompose_sweep_releases_a_capture_dispatched_during_the_teardown_drain(
    widget_pilot,
):
    """Residual-window regression (mirrors ``test_post_recompose_sweep_
    releases_a_capture_dispatched_during_the_teardown_drain`` in
    ``Tests/UI/test_chatbooks_screen_server_actions.py``, the task-637
    code-review finding for ``BaseAppScreen``/task-627): a capture that
    lands on the VICTIM's own message pump -- queued before the recompose's
    pre-teardown release even ran, but processed DURING
    ``super().recompose()``'s own ``remove()`` drain -- must still be swept
    once the recompose fully completes.

    Reproduced deterministically with ``call_later`` on the victim's own
    pump, mechanism-equivalent to a forwarded ``MouseDown`` whose dispatch is
    still pending on the search Input's pump when ``sync_state()`` starts
    the rail's recompose.
    """
    async with await widget_pilot(
        LibraryRail,
        shell=_make_shell(),
        preferences=LibraryRailPreferences(),
    ) as pilot:
        rail = pilot.app.test_widget
        await pilot.pause()
        victim = pilot.app.query_one("#library-search-input", Input)

        # Schedule the recompose first (the widget's own next-callback),
        # then queue a capture-inducing message on the VICTIM's own pump --
        # modelling a MouseDown forwarded to the Input but not yet
        # dispatched when the teardown starts.
        rail.sync_state(_make_shell(), LibraryRailPreferences(), query="x")
        victim.call_later(lambda: pilot.app.capture_mouse(victim))

        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        captured = pilot.app.mouse_captured
        assert captured is None, (
            f"stale capture survived the teardown drain: {captured!r} "
            f"(attached={getattr(captured, 'is_attached', None)}) -- clicks "
            "anywhere in the app are silently swallowed again (task-670)"
        )
