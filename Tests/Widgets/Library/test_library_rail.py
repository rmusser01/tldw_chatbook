"""Tests for the LibraryRail widget."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.Library.library_rail_state import LibraryRailPreferences
from tldw_chatbook.Library.library_shell_state import (
    LibraryRailRow,
    LibraryRailSectionState,
    LibraryShellState,
)
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


# -- F-014: one count policy for every rail row ----------------------------
# dim "(…)" while the source snapshot is in flight, "(N)"/"(N+)" when the
# count is known, and no suffix at all when the source is off or unknown --
# never a misleading "(0)" for an unavailable source.


def _row(row_id: str, title: str, **kwargs) -> LibraryRailRow:
    return LibraryRailRow(
        row_id=row_id,
        section_id="browse",
        title=title,
        target_kind="canvas",
        target_id="x",
        **kwargs,
    )


def _shell_with_rows(rows) -> LibraryShellState:
    return LibraryShellState(
        header_line="Library | Test",
        sections=(
            LibraryRailSectionState(
                section_id="browse", title="Browse", rows=tuple(rows)
            ),
        ),
        details_lines=(),
        selected_row_id="",
        canvas_kind="empty",
        canvas_target="",
        canvas_empty_copy="",
    )


async def test_count_policy_loading_known_estimate_and_off_rows(widget_pilot):
    """One policy: dim placeholder while loading, count when known, nothing
    when the source is off."""
    shell = _shell_with_rows(
        [
            _row("r-loading", "Loading", count=None, count_loading=True),
            _row("r-known", "Known", count=7),
            _row("r-estimate", "Estimate", count=7, count_known=False),
            _row("r-off", "Off", count=None),
        ]
    )

    async with await widget_pilot(
        LibraryRail,
        shell=shell,
        preferences=LibraryRailPreferences(),
    ) as pilot:
        await pilot.pause()

        loading = pilot.app.query_one("#library-row-r-loading", Button)
        assert loading.label.plain == "  Loading (…)"
        assert any(
            "dim" in str(span.style) for span in loading.label.spans
        ), f"loading placeholder must render dim: {loading.label.spans}"

        known = pilot.app.query_one("#library-row-r-known", Button)
        assert known.label.plain == "  Known (7)"

        estimate = pilot.app.query_one("#library-row-r-estimate", Button)
        assert estimate.label.plain == "  Estimate (7+)"

        off = pilot.app.query_one("#library-row-r-off", Button)
        assert off.label.plain == "  Off"


async def test_details_renders_db_sizes_row_only_when_provided(widget_pilot):
    """F-014: relocated DB-size telemetry lives in the Details disclosure --
    rendered when the shell carries it, omitted entirely when it does not
    (no 'N/A' triplets)."""
    with_sizes = LibraryShellState(
        header_line="Library | Test",
        sections=(),
        details_lines=(
            "Local",
            "Notes 0 · Media 0 · Conversations 0",
            "Prompts 1.0 KB · Chats/Notes 2.0 KB · Media 3.0 KB",
        ),
        selected_row_id="",
        canvas_kind="empty",
        canvas_target="",
        canvas_empty_copy="",
    )

    async with await widget_pilot(
        LibraryRail,
        shell=with_sizes,
        preferences=LibraryRailPreferences(details_open=True),
    ) as pilot:
        await pilot.pause()
        sizes = pilot.app.query_one("#library-details-db-sizes", Static)
        text = str(sizes.renderable)
        assert "Prompts 1.0 KB" in text
        assert "Chats/Notes 2.0 KB" in text
        assert "Media 3.0 KB" in text

    without_sizes = LibraryShellState(
        header_line="Library | Test",
        sections=(),
        details_lines=("Local", "Notes 0 · Media 0 · Conversations 0"),
        selected_row_id="",
        canvas_kind="empty",
        canvas_target="",
        canvas_empty_copy="",
    )

    async with await widget_pilot(
        LibraryRail,
        shell=without_sizes,
        preferences=LibraryRailPreferences(details_open=True),
    ) as pilot:
        await pilot.pause()
        assert not list(pilot.app.query("#library-details-db-sizes"))
