"""TASK-34000.27: a vetoed nav-bar switch says why, keeps the current
destination highlighted, and lets the retry click work.

Review S-17 (qa/notes-library-ux-review-2026-10-02, captures
`verify/nl-vb-15/16-veto-t7-160x45.ansi` and `18-retry-console-160x45.txt`):
with the note editor reporting "Title begins or ends with whitespace — remove
it to save.", clicking ⌃2 Console left Library on screen while the nav bar
framed Console and nothing said why; after the title was fixed, the second
click on ⌃2 Console did nothing, because the bar's already-active check
swallowed it (the swallowed-retry trap `MainNavigationBar.restore_active`
documents for the crash path, task-2720).

The Pilot arms boot the REAL app into Library (its real ``LibraryScreen``
with its real ``MainNavigationBar``) and drive the click path
(``_activate_navigation_button``: optimistic highlight + ``NavigateToScreen``
through the app's own navigation worker). The unit arms mirror
``test_screen_navigation_failure_recovery.py`` for the branches a Pilot
cannot reach deterministically (a flush that times out, raises, returns a
bare False, or a confirm hook that refuses).
"""

from __future__ import annotations

import asyncio
import time

import pytest
from textual.widgets import Input, TextArea

import tldw_chatbook.UI.Screens.library_screen as library_screen_module
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    _open_note_editor,
    _seed_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
)
from tldw_chatbook.UI.Navigation.main_navigation import (
    MainNavigationBar,
    NavigateToScreen,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: The review's wide repro size.
SIZE = (160, 45)

#: The status line the editor shows for the trailing-space title
#: (`Library/library_notes_state.py`), and the toast head this task adds.
_WHITESPACE_REASON = "Title begins or ends with whitespace"
_VETO_HEAD = "Can't open Console yet"


# --- shared driving ---------------------------------------------------------


def _record_toasts(monkeypatch, app) -> list[str]:
    """Record every toast the app shows, still showing it."""
    toasts: list[str] = []
    real_notify = app.notify

    def _notify(message, *args, **kwargs):
        toasts.append(str(message))
        return real_notify(message, *args, **kwargs)

    monkeypatch.setattr(app, "notify", _notify)
    return toasts


def _library_app(monkeypatch):
    """The production app booted straight into Library, autosave fast."""
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.05)
    app = _build_test_app(configured_default="library")
    _seed_conversations(app, [], notes=_two_notes())
    return app


async def _library(app, pilot) -> LibraryScreen:
    await _wait_for_condition(
        pilot,
        lambda: isinstance(app.screen, LibraryScreen),
        message="Library never mounted as the initial screen",
    )
    screen = app.screen
    await _wait_for_library_shell(screen, pilot)
    return screen


def _note_status(screen) -> str:
    return str(screen.query_one("#library-note-status").renderable)


async def _set_title_and_wait(screen, pilot, title: str, status_fragment: str):
    screen.query_one("#library-note-title", Input).value = title
    await _wait_for_condition(
        pilot,
        lambda: status_fragment in _note_status(screen),
        message=lambda: (
            f"note status never read {status_fragment!r} after setting the "
            f"title to {title!r}; status={_note_status(screen)!r}"
        ),
    )


async def _activate_destination(app, pilot, destination_id: str):
    """Drive the real click path on the visible screen's bar.

    Returns:
        ``(activated, bar)``: what ``_activate_navigation_button`` returned
        (False is the swallowed-retry symptom) and the bar it was pressed on.
        Waits for the app's screen-navigation workers to finish, so the
        veto/switch has settled when this returns.
    """
    bar = app.screen.query_one(MainNavigationBar)
    activated = bar._activate_navigation_button(
        bar.query_one(f"#nav-{destination_id}")
    )
    await pilot.pause()
    await _wait_for_condition(
        pilot,
        lambda: all(
            worker.is_finished
            for worker in getattr(app, "_screen_navigation_workers", ())
        ),
        message="the screen-navigation worker never finished",
    )
    await pilot.pause()
    return activated, bar


# --- AC#1, AC#2, AC#4: the veto says why and keeps Library highlighted ------


async def test_vetoed_switch_keeps_library_highlighted_and_says_why(monkeypatch):
    """Capture 16-veto-t7: Library stays, ⌃3 Library stays framed, the toast
    names the field and the destination, the typed text is untouched, and the
    quit flush (its own prompt) still says nothing."""
    app = _library_app(monkeypatch)
    toasts = _record_toasts(monkeypatch, app)

    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        await _open_note_editor(screen, pilot)
        body_before = screen.query_one("#library-note-body", TextArea).text
        await _set_title_and_wait(screen, pilot, "Draft ", _WHITESPACE_REASON)
        toasts.clear()

        activated, bar = await _activate_destination(app, pilot, "console")

        assert activated is True, "test premise: the first click must post"
        assert app.screen is screen, "the veto must leave Library on the stack"
        assert bar.active_destination_id == "library", (
            f"the bar still frames {bar.active_destination_id!r} after the veto"
        )
        assert bar.query_one("#nav-library").has_class("is-active")
        assert not bar.query_one("#nav-console").has_class("is-active")
        assert [t for t in toasts if _VETO_HEAD in t and _WHITESPACE_REASON in t], (
            f"no toast said why the switch was blocked; toasts={toasts!r}"
        )
        # Unsaved text is never lost: the veto keeps the user exactly where
        # they were, with their text intact.
        assert screen.query_one("#library-note-title", Input).value == "Draft "
        assert screen.query_one("#library-note-body", TextArea).text == body_before

        # Negative control (Review Focus 3): the quit flow asks its own
        # question, so its flush stays quiet.
        toasts.clear()
        assert await screen.flush_pending_work(quitting=True) is False
        assert toasts == [], f"the quit flush toasted: {toasts!r}"


# --- AC#3, AC#4: the retry click works once the cause is fixed --------------


async def test_retry_after_fixing_the_title_navigates_on_the_first_click(
    monkeypatch,
):
    """Capture 18-retry-console: after "Saved", ONE click on ⌃2 Console opens
    Console. On the base the bar still thought Console was active, so
    ``_activate_navigation_button`` returned False and nothing happened."""
    app = _library_app(monkeypatch)
    _record_toasts(monkeypatch, app)

    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        await _open_note_editor(screen, pilot)
        await _set_title_and_wait(screen, pilot, "Draft ", _WHITESPACE_REASON)
        activated, _bar = await _activate_destination(app, pilot, "console")
        assert activated is True and app.screen is screen, "premise: vetoed"

        await _set_title_and_wait(screen, pilot, "Draft", "Saved")

        activated, _bar = await _activate_destination(app, pilot, "console")

        assert activated is True, (
            "the retry click was swallowed: the bar still believed Console "
            "was active after the veto"
        )
        deadline = time.monotonic() + 10
        while app.screen is screen and time.monotonic() < deadline:
            await pilot.pause(0.02)
        assert app.screen is not screen, "the retry never left Library"
        assert getattr(app.screen, "screen_name", None) == "chat", (
            f"the retry opened {type(app.screen).__name__}, not Console"
        )
        new_bar = app.screen.query_one(MainNavigationBar)
        assert new_bar.active_destination_id == "console"


# --- AC#2 unit arms: timeout, failure, bare veto, confirm veto --------------


class _RecordingBar:
    def __init__(self):
        self.restored: list[str] = []

    def restore_active(self, route: str) -> None:
        self.restored.append(route)


class _OutgoingScreen:
    """Outgoing screen with its own nav bar, like every BaseAppScreen."""

    screen_name = "library"

    def __init__(self, bar: _RecordingBar):
        self._bar = bar

    def query_one(self, _selector):
        return self._bar


def _wire(app, monkeypatch, outgoing) -> list[tuple[str, dict]]:
    """Point the app at a fake target and ``outgoing``; record toasts."""

    class FakeTargetScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            self.app_instance = app_instance

    monkeypatch.setattr(
        app,
        "_resolve_screen_navigation_target",
        lambda target: ("chat", "chat", FakeTargetScreen),
    )

    async def fake_switch_screen(screen):
        raise AssertionError("a refused navigation must never switch screens")

    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)
    monkeypatch.setattr(type(app), "screen", property(lambda self: outgoing))
    app._initial_screen_pushed = True
    notifications: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append((str(message), kwargs)),
    )
    return notifications


async def test_flush_timeout_restores_the_highlight(monkeypatch):
    app = _build_test_app()
    bar = _RecordingBar()
    release = asyncio.Event()

    class SlowScreen(_OutgoingScreen):
        async def flush_pending_work(self):
            await release.wait()
            return True

    notifications = _wire(app, monkeypatch, SlowScreen(bar))
    monkeypatch.setattr(app, "NAVIGATION_FLUSH_TIMEOUT_SECONDS", 0.05, raising=False)

    try:
        assert await app.handle_screen_navigation(NavigateToScreen("chat")) is None
        assert bar.restored == ["library"], (
            "a flush timeout left the bar on the clicked destination"
        )
        assert [m for m, _ in notifications if "Still saving" in m], notifications
    finally:
        release.set()
        await asyncio.sleep(0)


async def test_flush_failure_restores_the_highlight(monkeypatch):
    app = _build_test_app()
    bar = _RecordingBar()

    class FailingScreen(_OutgoingScreen):
        async def flush_pending_work(self):
            raise RuntimeError("injected: TASK-34000.27")

    notifications = _wire(app, monkeypatch, FailingScreen(bar))

    await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert bar.restored == ["library"], (
        "a flush failure left the bar on the clicked destination"
    )
    assert [m for m, _ in notifications if "Couldn't save pending changes" in m], (
        notifications
    )


async def test_plain_false_veto_restores_without_a_second_toast(monkeypatch):
    """The screen owns the reason (Library, Console and Settings all say why
    before returning False); the app rolls the bar back and adds nothing. A
    silent third-party veto is logged, as before."""
    app = _build_test_app()
    bar = _RecordingBar()

    class VetoingScreen(_OutgoingScreen):
        def flush_pending_work(self):
            return False

    notifications = _wire(app, monkeypatch, VetoingScreen(bar))

    await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert bar.restored == ["library"], (
        "a flush veto left the bar on the clicked destination"
    )
    assert notifications == [], (
        f"the app toasted over the screen's own reason: {notifications!r}"
    )


async def test_confirm_navigation_veto_restores_the_highlight(monkeypatch):
    """Roleplay and Chunking Lab refuse through ``confirm_navigation`` (a
    dialog the user answered "stay"); the same rollback applies."""
    app = _build_test_app()
    bar = _RecordingBar()

    class StayingScreen(_OutgoingScreen):
        async def confirm_navigation(self):
            return False

    _wire(app, monkeypatch, StayingScreen(bar))

    await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert bar.restored == ["library"], (
        "a confirm veto left the bar on the clicked destination"
    )


async def test_the_flush_is_told_the_destination_it_refuses(monkeypatch):
    """A flush that accepts ``destination`` gets the clicked destination's
    label, so its own veto toast can name it; one that does not is called as
    before (every other arm in this file)."""
    app = _build_test_app()
    bar = _RecordingBar()
    seen: list[str] = []

    class LabelAwareScreen(_OutgoingScreen):
        async def flush_pending_work(self, *, destination: str = ""):
            seen.append(destination)
            return False

    _wire(app, monkeypatch, LabelAwareScreen(bar))

    await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert seen == ["Console"], seen
    assert bar.restored == ["library"]
