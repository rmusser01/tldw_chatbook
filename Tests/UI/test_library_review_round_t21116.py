"""TASK-21116 review round: defects found in the per-click conversion.

Each test here was written RED against the first conversion pass and
names the exact mechanism it pins, so a later refactor that reintroduces
the defect fails on the mechanism rather than on a symptom.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_canvas_scoped_sync import _screen_recompose_spy
from Tests.UI.test_library_per_click_recompose_t21116 import (
    _boot_media_library,
    _media_app_host,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import LibraryMediaCanvas
from tldw_chatbook.Widgets.Library.library_media_trash_canvas import (
    LibraryMediaTrashCanvas,
)


# ---------------------------------------------------------------------------
# M1: the media-detail worker continuation must not clobber the Trash view
# ---------------------------------------------------------------------------


def test_media_active_child_builder_honors_the_trash_view() -> None:
    """The shared media-child builder returns the Trash canvas in trash view.

    RED before the fix: ``_build_library_media_active_child`` had no
    "trash" branch at all -- unlike ``compose_content`` and
    ``_build_library_entry_active_child``, which both check it first -- so
    it returned a ``LibraryMediaCanvas`` (the LIST) for a screen sitting in
    the Trash view.
    """
    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_view = "trash"
    screen._library_media_trash_records = ()
    screen._library_media_trash_total = 0

    child = screen._build_library_media_active_child()

    assert isinstance(child, LibraryMediaTrashCanvas), type(child).__name__


@pytest.mark.asyncio
async def test_late_media_detail_arrival_cannot_clobber_the_trash_view() -> None:
    """A detail worker resolving after "Trash" must leave the Trash view alone.

    RED before the fix: ``_apply_library_media_active_surface`` guarded only
    on the rail row id, so the worker continuation
    (``_recompose_library_media_detail_if_unrendered``) swapped a
    ``LibraryMediaCanvas`` in over the mounted ``LibraryMediaTrashCanvas``
    while ``_library_media_view`` stayed "trash" -- a DOM/state divergence
    that also breaks the follow-on ``_sync_library_canvas("media-trash")``.
    The route-key supersede guard cannot catch this: the view is INSIDE the
    route key, so the key stays self-consistent.
    """
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        # Open a viewer so a real detail fetch is in the picture, then
        # leave it -- the worker whose continuation we fire below is the
        # one this open started.
        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-content-search")
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")

        screen.query_one("#library-media-trash-open", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-canvas")
        trash_before = screen.query_one(
            "#library-media-trash-canvas", LibraryMediaTrashCanvas
        )
        assert screen._library_media_view == "trash"

        # The late worker continuation, fired exactly as the worker fires it.
        screen._recompose_library_media_detail_if_unrendered()
        for _ in range(10):
            await pilot.pause(0.02)

        assert screen._library_media_view == "trash"
        assert not screen.query("#library-media-canvas"), (
            "the media LIST canvas was mounted over the Trash view"
        )
        assert (
            screen.query_one("#library-media-trash-canvas", LibraryMediaTrashCanvas)
            is trash_before
        )


# ---------------------------------------------------------------------------
# M2: the viewer-scoped sub-state rebuild must refresh the footer hint set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_viewer_substate_escape_refreshes_the_footer_shortcut_set() -> None:
    """Escaping a viewer sub-state re-registers the footer shortcuts.

    RED before the fix: the retired whole-screen recompose re-ran
    ``compose_content`` -> ``_register_footer_shortcuts``; the viewer-scoped
    rebuild did not. The sets genuinely differ -- the shortcut selector
    branches on ``_library_media_viewer_substate_active()`` -- so the footer
    kept advertising the sub-state's "back a step" after Escape had already
    returned to the plain viewer, where Escape means "back to list".
    """
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-more")
        screen.query_one("#library-media-reader-more", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-edit")

        plain_viewer_shortcuts = screen._footer_shortcut_registration

        screen.query_one("#library-media-edit", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-edit-cancel")
        substate_shortcuts = screen._footer_shortcut_registration
        # Guard the probe: if these ever stop differing, this test would
        # pass vacuously and would no longer measure its subject.
        assert substate_shortcuts != plain_viewer_shortcuts, (
            "sub-state and plain-viewer shortcut sets are identical -- this "
            "test can no longer detect a stale footer"
        )

        screen.action_library_media_viewer_back()
        for _ in range(10):
            await pilot.pause(0.02)

        assert screen._library_media_editing is False
        assert screen._library_media_view == "viewer"
        assert screen._footer_shortcut_registration == plain_viewer_shortcuts


# ---------------------------------------------------------------------------
# M3: a targeted projection must not race the canvas-sync whole-screen fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_canvas_sync_suppresses_its_screen_fallback_inside_a_projection() -> None:
    """While a projection owns the canvas host, a stray sync must not recompose.

    Treatment AND control in one test, because the treatment alone would
    pass vacuously if the sync ever stopped failing: with the projection
    marker cleared, the identical call MUST still take the whole-screen
    fallback. That control is the pre-fix behaviour, measured -- one
    whole-screen recompose, the exact thing this task removed.
    """
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)

        async def detach_canvas_child() -> None:
            """Reproduce the projection window: the host's child detached.

            Re-queried every time on purpose -- the control arm below
            recomposes the whole screen, which REPLACES ``#library-canvas``,
            so a host reference captured once goes stale and silently
            detaches children of a widget that is no longer in the tree.
            """
            live_host = screen.query_one("#library-canvas")
            await live_host.remove_children(tuple(live_host.children))
            assert not screen.query("#library-media-canvas")

        # TREATMENT FIRST (it must not recompose, so it leaves the tree
        # intact for the control arm that follows).
        await detach_canvas_child()
        screen._library_canvas_projection_depth += 1
        try:
            calls, spy = _screen_recompose_spy()
            with patch.object(BaseAppScreen, "refresh", spy):
                assert (
                    library_screen_module._sync_library_canvas(screen, "media") is False
                )
                await pilot.pause()
        finally:
            screen._finish_library_canvas_projection()
        assert calls == [], (
            "the canvas-sync fallback fired a whole-screen recompose while a "
            "targeted projection owned the canvas host"
        )
        await _wait_for_selector(screen, pilot, "#library-media-canvas")

        # CONTROL: identical failure with no projection in flight must still
        # take the legacy fallback -- otherwise the treatment proves nothing.
        await detach_canvas_child()
        control_calls, control_spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", control_spy):
            assert library_screen_module._sync_library_canvas(screen, "media") is False
            await pilot.pause()
        assert len(control_calls) == 1, (
            "the control arm did not reproduce the whole-screen fallback, so "
            "the treatment arm above proves nothing"
        )


@pytest.mark.asyncio
async def test_slow_canvas_swap_is_not_raced_by_the_media_browse_sync() -> None:
    """Reader Back keeps the retained canvas mounted and avoids recomposition."""
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-content-search")

        canvas_host = screen.query_one("#library-canvas")
        removals: list[tuple[object, ...]] = []
        original_remove = canvas_host.remove_children

        def record_remove(*args, **kwargs):
            removals.append(args)
            return original_remove(*args, **kwargs)

        canvas_host.remove_children = record_remove
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            screen.query_one("#library-media-back", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-row-0")
            for _ in range(15):
                await pilot.pause(0.02)

        assert removals == [], "Reader Back tore down the retained canvas"
        assert calls == [], (
            "a whole-screen recompose fired while returning to retained Items"
        )
        # Exactly one canvas, i.e. no duplicate-id collision survived.
        assert len(screen.query("#library-media-canvas")) == 1
        assert isinstance(
            screen.query_one("#library-media-canvas"), LibraryMediaCanvas
        )


# ---------------------------------------------------------------------------
# M4: a snapshot landing mid-swap must not silently drop the follow-up
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_routine_snapshot_midswap_still_runs_the_projection_follow_up() -> None:
    """A routine snapshot cannot suppress the permanent Reader's return focus.

    Media Items and Reader now remain mounted together, so returning from the
    viewer no longer has an await window in which to intercept a canvas-child
    swap. Preserve the original regression's user-visible contract by bumping
    the snapshot generation immediately before Back and verifying that the
    already-mounted list still receives its focus/scroll follow-up.
    """
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-content-search")

        armed: list[object] = []
        original_arm = screen._arm_library_list_entry_focus

        def spy_arm(*, media_return=None):
            armed.append(media_return)
            return original_arm(media_return=media_return)

        screen._arm_library_list_entry_focus = spy_arm

        screen._library_snapshot_state_generation += 1
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")
        for _ in range(15):
            await pilot.pause(0.02)

        assert armed, (
            "the Reader return follow-up was dropped after a snapshot "
            "generation change: task-2856 AC1 focus and scroll restore never ran"
        )
