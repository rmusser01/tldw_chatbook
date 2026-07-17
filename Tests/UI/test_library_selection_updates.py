"""TASK-252: Library targeted (non-recompose) updates for selection interactions.

Covers the two-tier design decided for the audit's staged SELECTION
interaction class (Docs/Design/2026-07-16-performance-audit.md §P1 B2):

* Tier 1 -- select-mode checkbox toggles patch the pressed row's marker,
  the "N selected" Static, and the export-selected button's disabled
  state in place, instead of ``self.refresh(recompose=True)`` (a
  whole-screen remove/remount of the nav bar, footer, ~20-row rail, and
  50-100-row canvas).
* Tier 2 -- browse-mode row selection (the ``▸`` highlight + preview
  change) and select-mode enter/exit/select-all/clear call the mounted
  canvas's own ``sync_state`` -- a canvas-scoped recompose that rebuilds
  only the canvas's own children, skipping the nav bar, footer, and rail.

Reuses the established Library mounted-test harness from
``test_library_shell.py`` (``LibraryHarness`` / ``_build_test_app`` /
``_seed_conversations`` / the ``_wait_for_*`` pollers).
"""
from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.Widgets.Library.library_conversations_canvas import (
    LibraryConversationsCanvas,
)

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_screen_navigation import _build_test_app


def _spy_screen_recomposes(monkeypatch) -> list:
    """Patch ``BaseAppScreen.refresh`` to record every ``recompose=True`` call.

    Args:
        monkeypatch: The active pytest ``monkeypatch`` fixture (restores the
            original method automatically at test teardown).

    Returns:
        A list that accumulates the ``self`` (screen instance) of every
        ``BaseAppScreen.refresh(recompose=True)`` call made after this spy
        is installed.
    """
    calls: list = []
    original = BaseAppScreen.refresh

    def spy(self, *args, **kwargs):
        if kwargs.get("recompose"):
            calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BaseAppScreen, "refresh", spy)
    return calls


async def _enter_conversations_select_mode(screen, pilot):
    """Drive the harness to the conversations canvas with select mode on."""
    screen.query_one("#library-row-browse-conversations").press()
    await _wait_for_selector(screen, pilot, "#library-conversation-row-0")

    screen.query_one("#library-conversations-select-toggle").press()
    await _wait_for_selector(screen, pilot, "#library-conversations-select-all")
    await pilot.pause()


@pytest.mark.asyncio
async def test_checkbox_toggle_does_not_recompose_screen(monkeypatch):
    """A select-mode checkbox press updates in place, not via a screen
    recompose.

    Pre-fix RED: the select-mode branch of
    ``handle_library_conversation_row`` called
    ``self.refresh(recompose=True)`` directly, so the screen-level
    recompose count rose by 1 on every checkbox press.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_conversations_select_mode(screen, pilot)

        recompose_calls = _spy_screen_recomposes(monkeypatch)

        row = screen.query_one("#library-conversation-row-0", Button)
        assert str(row.label).startswith("☐")
        count_static = screen.query_one("#library-conversations-selected-count", Static)
        assert str(count_static.renderable) == "0 selected"
        export_button = screen.query_one("#library-conversations-export-selected", Button)
        assert export_button.disabled is True

        row.press()
        await pilot.pause()

        assert recompose_calls == []  # no screen-level recompose
        assert str(row.label).startswith("☑")  # marker flipped in place
        assert (
            str(
                screen.query_one(
                    "#library-conversations-selected-count", Static
                ).renderable
            )
            == "1 selected"
        )
        assert (
            screen.query_one("#library-conversations-export-selected", Button).disabled
            is False
        )

        # Toggling back off is symmetric.
        row.press()
        await pilot.pause()
        assert recompose_calls == []
        assert str(row.label).startswith("☐")
        assert (
            str(
                screen.query_one(
                    "#library-conversations-selected-count", Static
                ).renderable
            )
            == "0 selected"
        )
        assert (
            screen.query_one("#library-conversations-export-selected", Button).disabled
            is True
        )


@pytest.mark.asyncio
async def test_checkbox_toggle_leaves_rail_untouched():
    """AC #2: selection/checkbox interactions never change rail counts, so
    the mounted rail row must survive a toggle unchanged -- proven by
    object identity (a screen recompose would tear down and rebuild a
    fresh ``LibraryRail``, minting a new row-button instance)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_conversations_select_mode(screen, pilot)

        rail_row_before = screen.query_one("#library-row-browse-conversations")
        label_before = str(rail_row_before.label)

        screen.query_one("#library-conversation-row-0", Button).press()
        await pilot.pause()

        rail_row_after = screen.query_one("#library-row-browse-conversations")
        assert rail_row_after is rail_row_before
        assert str(rail_row_after.label) == label_before


@pytest.mark.asyncio
async def test_browse_row_selection_routes_through_canvas_sync_state(monkeypatch):
    """Clicking a conversation row outside select mode (choosing which row
    is previewed -- the ``▸`` marker + preview subtree) calls the mounted
    canvas's own ``sync_state`` (a canvas-scoped recompose rebuilding only
    the canvas's own children), never the screen-level
    ``self.refresh(recompose=True)``.

    Pre-fix RED: ``LibraryConversationsCanvas.sync_state`` had zero
    callers (audit §P1 B2); this interaction went through
    ``self.refresh(recompose=True)`` instead.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-1")
        await pilot.pause()

        sync_calls: list = []
        original_sync = LibraryConversationsCanvas.sync_state

        def spy_sync(self, canvas):
            sync_calls.append(canvas)
            return original_sync(self, canvas)

        monkeypatch.setattr(LibraryConversationsCanvas, "sync_state", spy_sync)
        recompose_calls = _spy_screen_recomposes(monkeypatch)

        # Rows sort newest-first: chat-2 (06-02) is row 0, chat-1 (06-01) is
        # row 1 -- entering the canvas auto-previews row 0 (chat-2).
        preview_before = str(
            screen.query_one("#library-conversation-preview-lines").renderable
        )
        assert "Design review notes" in preview_before

        screen.query_one("#library-conversation-row-1", Button).press()
        await pilot.pause()

        assert len(sync_calls) == 1
        assert recompose_calls == []
        preview_after = str(
            screen.query_one("#library-conversation-preview-lines").renderable
        )
        assert "Quarterly planning sync" in preview_after


@pytest.mark.asyncio
async def test_tier2_canvas_sync_releases_mouse_capture_first():
    """The shared tier-2 canvas-sync helper releases ``App.mouse_captured``
    before recomposing the canvas -- mirroring ``BaseAppScreen.refresh``'s
    guard (see its docstring for the full mouse-capture war story), since
    ``canvas.sync_state`` recomposes the canvas directly and bypasses that
    screen-level protection entirely."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        await pilot.pause()

        capture_calls: list = []
        original_capture = pilot.app.capture_mouse

        def recording_capture(widget):
            capture_calls.append(widget)
            return original_capture(widget)

        pilot.app.capture_mouse = recording_capture

        # Select-mode enter/exit is a Tier-2 canvas-sync interaction too.
        screen.query_one("#library-conversations-select-toggle").press()
        await pilot.pause()

        assert None in capture_calls


@pytest.mark.asyncio
async def test_tier1_toggle_falls_back_to_recompose_on_query_one_failure(monkeypatch):
    """If the tier-1 in-place helper's ``query_one`` raises (e.g. the
    select-mode action strip isn't mounted because the mode raced), the
    checkbox toggle falls back to the old full recompose instead of
    crashing -- and the underlying selection state still reflects the
    toggle."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_conversations_select_mode(screen, pilot)

        row = screen.query_one("#library-conversation-row-0", Button)

        original_query_one = type(screen).query_one

        def raising_query_one(self, selector, *args, **kwargs):
            if isinstance(selector, str) and "selected-count" in selector:
                raise RuntimeError("forced query_one failure (fallback test)")
            return original_query_one(self, selector, *args, **kwargs)

        monkeypatch.setattr(type(screen), "query_one", raising_query_one)
        recompose_calls = _spy_screen_recomposes(monkeypatch)

        row.press()
        await pilot.pause()

        assert len(recompose_calls) == 1  # fell back to a full recompose
        # Rows sort newest-first: chat-2 (06-02) is row 0.
        assert screen._library_conversations_row_selection.is_selected("chat-2")
