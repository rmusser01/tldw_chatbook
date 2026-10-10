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

The Pilot arm boots the REAL app into Library once (its real ``LibraryScreen``
with its real ``MainNavigationBar``) and drives the click path
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


def _painted_text_in_region(app, region) -> str:
    """What the compositor actually painted inside ``region``."""
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text.rstrip()
        for y in range(region.y, region.bottom)
    )


def _frame_color(app, button):
    """The painted colour of the button's top-left frame cell.

    Every nav button has a `border: solid`; the active one differs only in
    colour (`$primary`, `MainNavigationBar.DEFAULT_CSS`), which is what the
    user sees as "framed". Read from the composited strips, not the styles.
    """
    strips = list(app.screen._compositor.render_strips())
    region = button.region
    for segment in strips[region.y].crop(region.x, region.right):
        if segment.text.strip():
            return segment.style.color if segment.style is not None else None
    return None


# --- AC#1-AC#4 in one boot (review 1 Important 1): the veto says why, keeps
# --- Library highlighted and the text intact; the retry then works ---------


async def test_vetoed_switch_says_why_keeps_library_framed_and_the_retry_works(
    monkeypatch,
):
    """Captures 16-veto-t7 and 18-retry-console in one app boot.

    Veto: Library stays, ⌃3 Library stays framed (painted frame, not just a
    class), exactly one toast names the field and the destination, the typed
    text is untouched, and the quit flush (its own prompt) still says nothing.
    Retry: after "Saved", ONE click on ⌃2 Console opens Console. On the base
    the bar still thought Console was active, so
    ``_activate_navigation_button`` returned False and nothing happened.
    """
    app = _library_app(monkeypatch)
    toasts = _record_toasts(monkeypatch, app)

    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        await _open_note_editor(screen, pilot)
        body_before = screen.query_one("#library-note-body", TextArea).text
        await _set_title_and_wait(screen, pilot, "Draft ", _WHITESPACE_REASON)
        toasts.clear()
        bar_before = screen.query_one(MainNavigationBar)
        active_frame = _frame_color(app, bar_before.query_one("#nav-library"))
        inactive_frame = _frame_color(app, bar_before.query_one("#nav-console"))
        assert active_frame != inactive_frame, "premise: the active frame differs"

        activated, bar = await _activate_destination(app, pilot, "console")

        assert activated is True, "test premise: the first click must post"
        assert app.screen is screen, "the veto must leave Library on the stack"
        assert bar.active_destination_id == "library", (
            f"the bar still frames {bar.active_destination_id!r} after the veto"
        )
        library_button = bar.query_one("#nav-library")
        console_button = bar.query_one("#nav-console")
        assert library_button.has_class("is-active")
        assert not console_button.has_class("is-active")
        # What the user sees (review 1 Minor 4): the active-coloured frame is
        # painted around "Library" exactly as before the click, and Console
        # wears the inactive frame -- not the other way round.
        assert "Library" in _painted_text_in_region(app, library_button.region)
        assert _frame_color(app, library_button) == active_frame, (
            "Library lost its active frame on screen after the veto"
        )
        assert _frame_color(app, console_button) == inactive_frame, (
            "Console is painted with the active frame after the veto"
        )
        veto_toasts = [t for t in toasts if _VETO_HEAD in t]
        assert [t for t in veto_toasts if _WHITESPACE_REASON in t], (
            f"no toast said why the switch was blocked; toasts={toasts!r}"
        )
        # One click, one toast (review 1 Minor 6).
        assert len(veto_toasts) == 1, veto_toasts
        # Unsaved text is never lost: the veto keeps the user exactly where
        # they were, with their text intact.
        assert screen.query_one("#library-note-title", Input).value == "Draft "
        assert screen.query_one("#library-note-body", TextArea).text == body_before

        # Negative control (Review Focus 3): the quit flow asks its own
        # question, so its flush stays quiet.
        toasts.clear()
        assert await screen.flush_pending_work(quitting=True) is False
        assert toasts == [], f"the quit flush toasted: {toasts!r}"

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


async def test_admission_veto_says_the_honest_generic_and_restores(monkeypatch):
    """Review 1 Important 2: Library's Folder-files transition lease refuses
    through ``acquire_navigation_transition`` with no message of its own; the
    app says the honest generic (like the timeout arm) and rolls back."""
    app = _build_test_app()
    bar = _RecordingBar()

    class LeasedScreen(_OutgoingScreen):
        def acquire_navigation_transition(self):
            return False

    notifications = _wire(app, monkeypatch, LeasedScreen(bar))

    await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert bar.restored == ["library"], (
        "an admission veto left the bar on the clicked destination"
    )
    assert [m for m, _ in notifications if "Couldn't leave this screen yet" in m], (
        f"the admission veto was silent: {notifications!r}"
    )
    assert len(notifications) == 1, notifications


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
