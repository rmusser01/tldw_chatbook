"""Retained landing publication while native child recomposition owns the DOM."""

from __future__ import annotations

import dataclasses

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_library_entry_compose_once import (
    _landing_state,
    _LandingCanvasHarness,
)
from tldw_chatbook.Library.library_rail_state import LibraryLifecycle
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
from tldw_chatbook.Widgets.Library import (
    LibraryLandingCanvas,
    LibraryLandingContinueAction,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("boundary", ["pending", "mounting"])
async def test_landing_recompose_publishes_latest_state_and_finishes_once(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """A same-shape refresh must not query old or temporarily absent children."""
    app = _LandingCanvasHarness(_landing_state(LibraryLifecycle.STARTER))
    first = _landing_state(
        LibraryLifecycle.EXPANDED,
        continue_action=LibraryLandingContinueAction(
            label="First note", row_id=LIBRARY_ROW_BROWSE_NOTES
        ),
    )
    latest = dataclasses.replace(
        first,
        purpose="Newest Library snapshot",
        counts_line="Notes (4)",
        continue_action=LibraryLandingContinueAction(
            label="Latest note", row_id=LIBRARY_ROW_BROWSE_NOTES
        ),
    )

    async with app.run_test() as pilot:
        landing = app.query_one("#library-landing-canvas", LibraryLandingCanvas)
        completed = []
        mount_boundary_reached = False

        def finish() -> None:
            completed.append(
                (
                    str(
                        landing.query_one("#library-canvas-landing", Static).renderable
                    ),
                    str(landing.query_one("#library-hub-counts", Static).renderable),
                    str(landing.query_one("#library-hub-continue", Button).label),
                )
            )

        original_mount_all = landing.mount_all

        def mount_latest(*args, **kwargs):
            nonlocal mount_boundary_reached
            if not mount_boundary_reached:
                mount_boundary_reached = True
                assert landing.is_attached and landing.lock.is_locked
                assert not landing.children
                assert not landing._recompose_required
                assert landing._post_recompose_callback is finish
                landing.sync_state(latest)
            return original_mount_all(*args, **kwargs)

        if boundary == "mounting":
            monkeypatch.setattr(landing, "mount_all", mount_latest)
        landing.queue_after_recompose(finish)
        landing.sync_state(first)
        if boundary == "pending":
            assert landing._recompose_required and not landing.lock.is_locked
            landing.sync_state(latest)
        await pilot.pause()
        await pilot.pause()

        assert mount_boundary_reached is (boundary == "mounting")
        assert landing.state is latest
        assert completed == [("Newest Library snapshot", "Notes (4)", "Latest note")]
        assert landing._post_recompose_callback is None
        assert not landing._recompose_required
        assert not landing.lock.is_locked
