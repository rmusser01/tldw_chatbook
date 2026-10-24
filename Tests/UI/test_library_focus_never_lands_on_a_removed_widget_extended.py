"""Extended arms for TASK-34000.26 (not in the PR-gate census).

The gated core (``test_library_focus_never_lands_on_a_removed_widget.py``)
runs at 80x24. This file repeats its two RED pins at 120x36 (the j5 episode
began at 120 columns), proves the sharper AC#3 across the 120-column
breakpoint with focus in the rail, and proves the list-entry retry lands on
the REPLACEMENT filter after a canvas recompose rather than on the one being
removed.
"""

from __future__ import annotations

import pytest
from textual.screen import Screen
from textual.widgets import Button

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _library,
    _library_app,
    _open_notes,
    _settings_without_splash,
    _until,
)
from Tests.UI.test_library_shell import _wait_for_selector
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: The size the j5 journey started at.
WIDE = (120, 36)

_HELP_SECONDS = 8.0

#: Key presses allowed for one focus walk (j5 needed 27 Shift+Tabs).
_MAX_TABS = 60


async def _help_opens(app, pilot) -> None:
    await pilot.press("f1")
    await _until(
        pilot,
        lambda: isinstance(app.screen, WorkbenchHelpPanel),
        "help to open on F1",
        timeout=_HELP_SECONDS,
    )
    await pilot.press("escape")
    await _until(
        pilot,
        lambda: not isinstance(app.screen, WorkbenchHelpPanel),
        "help to close on Escape",
    )


def _live(widget) -> bool:
    return widget is not None and widget.is_attached and widget.visible


async def test_a_focus_write_during_a_filters_removal_is_refused_at_120x36(
    tmp_path, monkeypatch
):
    events: list = []
    app, _profile = _library_app(tmp_path, monkeypatch, events, read=lambda profile: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=WIDE) as pilot:
            screen = await _library(app, pilot)
            await _open_notes(screen, pilot)
            filter_widget = screen.query_one("#library-notes-filter")

            removal = filter_widget.remove()
            assert filter_widget._pruning is True
            screen.set_focus(filter_widget)
            assert screen.focused is not filter_widget
            await removal

            await _help_opens(app, pilot)


async def test_a_focus_that_went_dead_by_any_route_still_delivers_the_next_key_at_120x36(
    tmp_path, monkeypatch
):
    events: list = []
    app, _profile = _library_app(tmp_path, monkeypatch, events, read=lambda profile: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=WIDE) as pilot:
            screen = await _library(app, pilot)
            await _open_notes(screen, pilot)
            filter_widget = screen.query_one("#library-notes-filter")

            removal = filter_widget.remove()
            Screen.set_focus(screen, filter_widget)
            await removal
            assert screen.focused is filter_widget and not filter_widget.is_attached

            await _help_opens(app, pilot)
            await pilot.press("tab")
            await pilot.pause()
            assert _live(screen.focused)


async def _walk_focus_to(pilot, screen, key: str, widget_id: str) -> None:
    """Press ``key`` until ``widget_id`` holds focus, bounded by ``_MAX_TABS``."""
    for _ in range(_MAX_TABS):
        if getattr(screen.focused, "id", None) == widget_id:
            return
        await pilot.press(key)
        await pilot.pause()
    pytest.fail(f"{key} never reached #{widget_id}; last focus {screen.focused!r}")


def _not_a_corpse(screen) -> bool:
    focused = screen.focused
    return focused is None or (
        not getattr(focused, "_pruning", False) and focused.is_attached
    )


async def test_resizing_across_the_breakpoint_with_focus_in_the_rail_keeps_keys_alive(
    tmp_path, monkeypatch
):
    """Sharper AC#3, j5's own sequence: Search/RAG at 120x36 with focus on a
    Sources checkbox, resize to 80x24, Shift+Tab to the rail row Notes, Enter,
    Tab, F1. At every settle ``screen.focused`` is ``None`` or a live widget --
    never a corpse -- and the keys keep working. (The Notes browse route
    replaces the rail rows with the notes tree, so the rail row is reached
    from Search/RAG, as j5 did.)"""
    events: list = []
    app, _profile = _library_app(tmp_path, monkeypatch, events, read=lambda profile: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=WIDE) as pilot:
            screen = await _library(app, pilot)
            screen.query_one("#library-row-browse-search", Button).press()
            await _wait_for_selector(screen, pilot, "#library-rag-scope-toggle-notes")
            await _walk_focus_to(pilot, screen, "tab", "library-rag-scope-toggle-notes")

            await pilot.resize_terminal(80, 24)
            await _until(
                pilot,
                lambda: _not_a_corpse(screen),
                "focus to settle on nothing or a live widget after the resize",
            )
            await pilot.pause(0.2)
            assert _not_a_corpse(screen), screen.focused

            await _walk_focus_to(pilot, screen, "shift+tab", "library-row-browse-notes")
            await pilot.press("enter")
            await _wait_for_selector(screen, pilot, ".library-notes-tree-note-row")
            await _until(pilot, lambda: _not_a_corpse(screen), "focus to settle after Enter")
            await pilot.pause(0.2)
            assert _not_a_corpse(screen), screen.focused

            await pilot.press("tab")
            await pilot.pause()
            assert _live(screen.focused), screen.focused

            await _help_opens(app, pilot)


async def test_the_entry_focus_retry_lands_on_the_replacement_filter(
    tmp_path, monkeypatch
):
    """AC#1's second half: focus goes to an attached target instead. With the
    Notes list empty the entry-focus fallback is ``#library-notes-filter``;
    when that filter is being removed, the picker waits and the bounded retry
    lands on the NEW filter once the canvas has recomposed."""
    events: list = []
    app, _profile = _library_app(
        tmp_path, monkeypatch, events, read=lambda profile: None, notes=False
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=WIDE) as pilot:
            screen = await _library(app, pilot)
            screen.query_one("#library-row-browse-notes", Button).press()
            old = await _wait_for_selector(screen, pilot, "#library-notes-filter")
            assert not screen.query(".library-notes-row"), "precondition: an empty list"

            screen._arm_library_list_entry_focus()
            removal = old.remove()
            assert old._pruning is True
            screen._focus_library_list_entry()
            assert screen.focused is not old, "the picker offered a widget being removed"
            await removal

            canvas = screen.query_one("#library-notes-canvas")
            canvas.refresh(recompose=True)

            def _landed_on_the_replacement() -> bool:
                current = list(screen.query("#library-notes-filter"))
                return (
                    bool(current)
                    and current[0] is not old
                    and screen.focused is current[0]
                    and current[0].is_attached
                )

            await _until(
                pilot,
                _landed_on_the_replacement,
                "entry focus to land on the replacement filter",
                timeout=3.0,
            )
            await _help_opens(app, pilot)


async def test_the_picker_alone_never_writes_focus_onto_a_removed_widget(
    tmp_path, monkeypatch
):
    """Fix round 1, M-2: the picker and ``_tick`` seam pinned on its own. With
    ``LibraryScreen.set_focus`` routed to the BASE ``Screen.set_focus`` (the
    screen refusal out of the way), twenty rapid rail switches still record
    zero focus writes onto a pruning or detached widget, because the picker
    never offers one. Reverting the picker changes fails this arm."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    dead_writes: list[tuple[str | None, bool, bool]] = []
    real_set_focus = Screen.set_focus

    def recording_set_focus(self, widget, *args, **kwargs):
        if widget is not None and (
            getattr(widget, "_pruning", False) or not widget.is_attached
        ):
            dead_writes.append(
                (getattr(widget, "id", None), bool(widget._pruning), widget.is_attached)
            )
        return real_set_focus(self, widget, *args, **kwargs)

    monkeypatch.setattr(Screen, "set_focus", recording_set_focus)
    # The screen's own override steps aside: whatever the picker offers goes
    # straight to Textual.
    monkeypatch.setattr(LibraryScreen, "set_focus", recording_set_focus)

    events: list = []
    app, _profile = _library_app(tmp_path, monkeypatch, events, read=lambda profile: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await _library(app, pilot)
            await _open_notes(screen, pilot)

            for _ in range(10):
                screen.query_one("#library-row-browse-media", Button).press()
                await pilot.pause(0.05)
                screen.query_one("#library-row-browse-notes", Button).press()
                await pilot.pause(0.05)

            await _until(
                pilot,
                lambda: screen.focused is None or screen.focused.is_attached,
                "focus to settle on nothing or a live widget",
            )
            assert dead_writes == [], dead_writes
