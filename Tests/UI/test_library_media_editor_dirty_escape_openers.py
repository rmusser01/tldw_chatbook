"""TASK-34000.22 fix round 2: every opener of the Reader, Use in Console and
the stage below 64 columns respect a dirty authoring form.

The scoped re-review found one more silent drop and two strandings after
fix round 1:

- R2-1: an active review set's auto-resume, reached through the newly
  admitted Media rail press (Trash, "‹ Media", Media), ran
  ``_open_library_media_viewer`` which reset both forms with no toast. Now
  every opener refuses at one guard each (the viewer opener, the ambient
  resume, the by-id door, the navigation-sources flush).
- R2-3: Use in Console wrote the workspace membership and the "staged in
  Console" receipt BEFORE the navigation veto; the stale hand-off then
  applied on the next Console visit. Now it refuses before any write.
- R2-4: after Trash and "‹ Media" below 64 columns the list filled the
  stage and the dirty form was off-screen; a refusal or the Media press
  now restores the viewer so the form, Save and Discard are on screen.

In the census (its own file: the ``_extended`` sibling is near the lane
budget). Real keys, clicks and a real ``ReviewSetService`` over a real
collections DB; one boot per test.
"""

from __future__ import annotations

import itertools
from unittest.mock import Mock

import pytest
from textual.widgets import Button, TextArea

from Tests.UI.library_quit_guard_support import _record_toasts
from Tests.UI.test_library_media_editor_dirty_escape import (
    ROW_1,
    SIZE,
    _host,
    _open_analysis_form,
    _open_row_1,
    _painted,
    _type,
    _wait_for_text,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
)
from tldw_chatbook.Library.review_set_service import ReviewSetService
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _dirty_form_is_on_screen(screen, host=None) -> TextArea:
    state = screen._media_state
    assert state.view == "viewer"
    assert state.reader_session.loaded_id == ROW_1
    assert state.selected_media_id == ROW_1
    assert state.editing_analysis is True
    assert state.analysis_draft == "Key points"
    area = screen.query_one("#library-media-analysis-edit-text", TextArea)
    assert area.text == "Key points"
    if host is not None:
        assert area.region.height > 0 and area.region.width > 0, area.region
        assert "Key points" in _painted(host, area.region)
    return area


async def _dirty_analysis(screen, pilot) -> TextArea:
    await _open_row_1(screen, pilot)
    area = await _open_analysis_form(screen, pilot)
    await _type(pilot, "Key points")
    await _wait_for_text(screen, pilot, "Key points")
    return area


async def _trash_round_trip(screen, pilot) -> None:
    """Trash, then "‹ Media": the view comes back as "list" with the form
    still mounted beside it (fix round 1's audit)."""
    screen.query_one("#library-media-trash-open", Button).press()
    await _wait_for_condition(
        pilot, lambda: screen._media_state.view == "trash", message="Trash never opened"
    )
    back = await _wait_for_selector(screen, pilot, "#library-media-trash-back")
    back.press()
    await _wait_for_condition(
        pilot, lambda: screen._media_state.view == "list", message="Trash never closed"
    )
    for _ in range(6):
        await pilot.pause()
    assert screen._media_state.analysis_draft == "Key points"


async def test_the_review_set_auto_resume_never_drops_a_dirty_form(monkeypatch, tmp_path):
    """R2-1: active set + dirty form, Trash, "‹ Media", Media rail press. The
    resume used to open the cursor item and reset the form; now the opener
    refuses and the ambient resume stays quiet: the text is intact, the form
    is mounted and (R2-4) the viewer is the view again."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    counter = itertools.count(1)
    service = ReviewSetService(
        LibraryCollectionsDB(tmp_path / "collections.db"),
        id_factory=lambda: f"set-{next(counter)}",
        now=lambda: "2026-10-21T00:00:00Z",
    )
    service.create_review_set(
        "Both items",
        origin="browse",
        items=[(1, "Interview Recording"), (2, "Product Demo Video")],
    )
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._cached_review_set_service = service
        await _dirty_analysis(screen, pilot)
        await _trash_round_trip(screen, pilot)

        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        await _wait_for_condition(
            pilot,
            lambda: not any(
                worker.is_running
                for worker in screen.workers
                if worker.group == "library_review_set_resume"
            ),
            message="the auto-resume worker never finished",
        )
        for _ in range(8):
            await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        assert toasts == [], toasts
        _dirty_form_is_on_screen(screen, host)


async def test_use_in_console_refuses_before_linking_or_staging(monkeypatch):
    """R2-3: the Reader's Use in Console under a dirty form refuses with the
    one toast, writes no workspace membership and stages nothing."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    app.open_chat_with_handoff = Mock()
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        area = await _dirty_analysis(screen, pilot)

        screen.query_one("#library-media-use-in-chat", Button).press()
        await _wait_for_condition(pilot, lambda: bool(toasts), message="no refusal")
        for _ in range(6):
            await pilot.pause()

        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY], toasts
        app.open_chat_with_handoff.assert_not_called()
        registry = app.workspace_registry_service
        assert registry.get_item_memberships(item_type="media", item_id="1") == ()
        assert screen._media_state.link_receipt is None
        assert screen.query_one("#library-media-analysis-edit-text", TextArea) is area
        assert area.text == "Key points"


async def test_below_64_columns_a_refusal_brings_the_dirty_form_back_on_stage(
    monkeypatch,
):
    """R2-4: Trash and "‹ Media" at 100x30 (where the Trash button renders),
    then the terminal shrinks below 64 columns: the list fills the stage and
    the form is off-screen. The next refusal (a rail press on Notes) restores
    the viewer, so the text, Save and Discard changes paint again."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _dirty_analysis(screen, pilot)
        await _trash_round_trip(screen, pilot)

        await pilot.resize_terminal(60, 24)
        await _wait_for_condition(
            pilot, lambda: screen.size.width == 60, message="the resize never landed"
        )
        for _ in range(8):
            await pilot.pause()
        assert screen._media_state.view == "list"
        area = screen.query_one("#library-media-analysis-edit-text", TextArea)
        assert area.region.height == 0 or "Key points" not in _painted(host, area.region)

        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_for_condition(
            pilot, lambda: screen._media_state.view == "viewer", message="no reveal"
        )
        for _ in range(8):
            await pilot.pause()

        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY], toasts
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        _dirty_form_is_on_screen(screen, host)
        assert screen.query("#library-media-analysis-cancel")


async def test_a_refusal_in_trash_leaves_trash_as_it_is(monkeypatch):
    """R3-1: a dirty form, then Trash, then a nav-bar click and a Notes rail
    press. The reveal flips the view only from "list": in Trash the view
    stays "trash", the Trash canvas and its footer agree (``esc back to
    list``), Escape-back and the r/x Trash keys stay live, each refusal says
    why once, and the text is intact. The Media rail press from Trash stays
    in Trash too -- the state Trash owns is still the live state."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _dirty_analysis(screen, pilot)
        screen.query_one("#library-media-trash-open", Button).press()
        await _wait_for_condition(
            pilot, lambda: screen._media_state.view == "trash", message="Trash never opened"
        )
        await _wait_for_selector(screen, pilot, "#library-media-trash-canvas")

        # The Trash list is empty in this fixture, so the r/x gates read
        # whatever an empty Trash reads; a refusal must leave them exactly so.
        trash_keys = tuple(
            screen.check_action(action, ())
            for action in ("library_media_trash_restore", "library_media_trash_delete")
        )

        def trash_is_consistent() -> None:
            assert screen._media_state.view == "trash"
            assert screen.query("#library-media-trash-canvas")
            assert ("esc", "back to list") in screen._footer_shortcut_registration[1], (
                screen._footer_shortcut_registration
            )
            assert screen.check_action("library_media_trash_back", ()) is True
            assert tuple(
                screen.check_action(action, ())
                for action in ("library_media_trash_restore", "library_media_trash_delete")
            ) == trash_keys
            assert screen._media_state.analysis_draft == "Key points"
            assert screen._media_state.editing_analysis is True
            # Entering Trash is a whole-screen recompose, so the TextArea is
            # a fresh instance; the kept TEXT is the fact.
            mounted = screen.query_one("#library-media-analysis-edit-text", TextArea)
            assert mounted.text == "Key points"

        trash_is_consistent()

        # The nav-bar click's path: the app's navigation handler awaits this
        # flush before switching screens (this harness hosts the Library
        # screen directly, so the bar itself is not on a screen stack).
        assert await screen.flush_pending_work(destination="Console") is False
        for _ in range(6):
            await pilot.pause()
        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY], toasts
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        trash_is_consistent()

        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        for _ in range(6):
            await pilot.pause()
        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY] * 2, toasts
        trash_is_consistent()

        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        for _ in range(6):
            await pilot.pause()
        assert len(toasts) == 2, toasts
        trash_is_consistent()
