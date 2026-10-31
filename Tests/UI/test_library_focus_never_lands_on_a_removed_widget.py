"""Library focus never lands on a widget being removed (TASK-34000.26).

Review S-10 (``Docs/QA/notes-library-ux-review-2026-10-02/verify/nl-v-s-10``):
once during the keyboard-only journey Tab, F1 and Ctrl+P all stopped working
until a mouse click. The instrumented run recorded seven focus writes onto
widgets a canvas recompose was already removing (``#library-notes-filter``,
``#library-media-type-filter``, ...); Textual 8.2.8 accepts such a write and
leaves ``screen.focused`` on the detached widget, so every non-priority key
is forwarded into a closed message pump.

Three boots of the real ``TldwCli`` at 80x24 (the review's size), so F1 is
the app's own binding and ``TextualAppGuards`` is in the MRO. The 120x36
repeats, the resize arm and the retry-lands-on-the-replacement arm live in
``test_library_focus_never_lands_on_a_removed_widget_extended.py``.
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
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: The review's compact size.
SIZE = (80, 24)

#: Help must open on the FIRST key after the dead focus; a long wait would
#: only pad a RED run.
_HELP_SECONDS = 8.0


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


async def test_a_focus_write_during_a_filters_removal_is_refused_and_f1_still_opens_help(
    tmp_path, monkeypatch
):
    """AC#1 and AC#4: the write is refused, and the keyboard stays alive."""
    events: list = []
    app, _profile = _library_app(tmp_path, monkeypatch, events, read=lambda profile: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _open_notes(screen, pilot)
            filter_widget = screen.query_one("#library-notes-filter")

            removal = filter_widget.remove()
            assert filter_widget._pruning is True
            screen.set_focus(filter_widget)
            assert screen.focused is not filter_widget, (
                "a focus write onto a widget being removed must be refused"
            )
            await removal
            assert not filter_widget.is_attached

            await _help_opens(app, pilot)


async def test_a_focus_that_went_dead_by_any_route_still_delivers_the_next_key(
    tmp_path, monkeypatch
):
    """AC#2: a focus that still died (written past the screen's refusal) is
    healed before the next key is forwarded; Tab then moves onto a live
    widget (Review Focus 4)."""
    events: list = []
    app, _profile = _library_app(tmp_path, monkeypatch, events, read=lambda profile: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _open_notes(screen, pilot)
            filter_widget = screen.query_one("#library-notes-filter")

            removal = filter_widget.remove()
            assert filter_widget._pruning is True
            # The BASE method: bypasses the screen's refusal on purpose.
            Screen.set_focus(screen, filter_widget)
            await removal
            assert screen.focused is filter_widget and not filter_widget.is_attached, (
                "precondition: the corpse state the review observed"
            )

            await _help_opens(app, pilot)

            await pilot.press("tab")
            await pilot.pause()
            assert screen.focused is not None and screen.focused.is_attached


async def test_rapid_rail_switches_never_write_focus_onto_a_removed_widget(
    tmp_path, monkeypatch
):
    """AC#3's twin: twenty rail switches inside the 50 ms retry cadence
    record zero focus writes onto a pruning or detached widget, and the
    keyboard still works afterwards. A belt -- it may pass on the base by
    timing, so the two tests above are the RED pins."""
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

    events: list = []
    app, _profile = _library_app(tmp_path, monkeypatch, events, read=lambda profile: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
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

            await _help_opens(app, pilot)
