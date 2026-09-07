"""`[splash_screen] skip_on_keypress` actually fires (TASK-21591).

The setting shipped default-true, is documented in the config template as
"Allow users to skip with any keypress" and has a Settings checkbox -- and
could not work. `SplashScreen` is a `Container`, Textual routes a key event
to `App.focused or App.screen` and bubbles it *upward* from there, and
nothing ever focused the splash, so `SplashScreen.on_key` was unreachable.

These tests pin the three properties that make it real and keep it real:
the key dismisses the splash, the key is still delivered to the app's own
bindings on its way past, and turning the setting off leaves the splash
running its full duration (and unfocusable, which is what keeps the
Settings splash preview from stealing focus).
"""

from __future__ import annotations

import time

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding

from tldw_chatbook.Widgets.splash_screen import SplashScreen

pytestmark = pytest.mark.asyncio


class _SplashHost(App):
    """The minimum of the real app's splash wiring: yield it, watch Closed.

    `AUTO_FOCUS = None` on purpose. With Textual's default `"*"` the splash
    is auto-focused on screen mount the moment it becomes focusable, which
    makes `SplashScreen.on_mount`'s own `focus()` call untestable -- removing
    that call left all four of these tests green until this was set. The
    feature must not depend on a screen-level default it does not own.
    """

    AUTO_FOCUS = None
    BINDINGS = [Binding("z", "record_binding", "Record", show=False)]

    def __init__(self, *, skip: bool, duration: float) -> None:
        super().__init__()
        self.closed: list[float] = []
        self.binding_fired: list[str] = []
        self.mounted_at = 0.0
        self._skip = skip
        self._duration = duration

    def compose(self) -> ComposeResult:
        yield SplashScreen(
            card_name="default",
            duration=self._duration,
            skip_on_keypress=self._skip,
            show_progress=False,
            id="app-splash-screen",
        )

    def on_mount(self) -> None:
        self.mounted_at = time.perf_counter()

    def action_record_binding(self) -> None:
        self.binding_fired.append("z")

    def on_splash_screen_closed(self, event: SplashScreen.Closed) -> None:
        self.closed.append(time.perf_counter())


async def test_a_keypress_dismisses_the_splash_when_the_setting_is_on() -> None:
    """The whole point of the setting, and the thing that never worked."""

    app = _SplashHost(skip=True, duration=30.0)
    async with app.run_test() as pilot:
        splash = app.query_one("#app-splash-screen", SplashScreen)
        # Focus is the mechanism; without it the key never reaches `on_key`.
        assert splash.can_focus is True
        assert app.focused is splash

        await pilot.press("space")
        await pilot.pause()

        assert app.closed, "a keypress did not dismiss the splash"
        assert splash._skip_requested is True
        # Dismissed by the key, not by the auto-close timer 30s away.
        assert app.closed[0] - app.mounted_at < 5.0


async def test_the_skipping_key_is_consumed_and_does_not_also_fire_a_binding() -> None:
    """The dismissing key must not ALSO run whatever it is bound to.

    Letting the event bubble was tried first. It kept `ctrl+q` working
    during the splash, but it also let a shell-destination key through, and
    since `action_shell_destination` *posts* its `NavigateToScreen`, the
    request was handled after the now-immediate splash close had pushed the
    initial screen -- so F9 mid-splash landed on Settings, which is the
    exact thing `test_navigation_keypress_during_splash_is_safely_ignored`
    (task-1339) locks against. The splash is a modal over an app that is
    not interactive yet; the key's only job while it is up is to dismiss it.
    """

    app = _SplashHost(skip=True, duration=30.0)
    async with app.run_test() as pilot:
        await pilot.press("z")
        await pilot.pause()

        assert app.closed, "a keypress did not dismiss the splash"
        assert app.binding_fired == [], (
            "the dismissing key also ran its app binding; a navigation key "
            "pressed during startup would act on the app being booted"
        )


async def test_a_key_pressed_with_the_skip_disabled_still_reaches_its_binding() -> None:
    """Consuming the key is scoped to the skip, not to the splash being up.

    With `skip_on_keypress = false` the splash takes no focus and changes
    nothing about input routing -- the same behaviour as before TASK-21591.
    """

    app = _SplashHost(skip=False, duration=30.0)
    async with app.run_test() as pilot:
        await pilot.press("z")
        await pilot.pause()

        assert app.closed == []
        assert app.binding_fired == ["z"]


async def test_a_second_keypress_does_not_close_the_splash_twice() -> None:
    """`Closed` drives an unrepeatable mount+push in the real app."""

    app = _SplashHost(skip=True, duration=30.0)
    async with app.run_test() as pilot:
        await pilot.press("space")
        await pilot.press("space")
        await pilot.pause()

        assert len(app.closed) == 1, app.closed


async def test_the_setting_off_leaves_the_splash_running_its_full_duration() -> None:
    """`skip_on_keypress = false` must still mean "no skip", not "no splash"."""

    duration = 0.4
    app = _SplashHost(skip=False, duration=duration)
    async with app.run_test() as pilot:
        splash = app.query_one("#app-splash-screen", SplashScreen)
        # Unfocusable, so it also cannot steal focus as a Settings preview.
        assert splash.can_focus is False
        assert app.focused is not splash

        await pilot.press("space")
        await pilot.pause()
        await pilot.pause()

        assert app.closed == [], "the splash was skipped with the skip disabled"
        assert splash._skip_requested is False

        # ... and it still closes on its own, no earlier than its duration.
        deadline = time.perf_counter() + 5.0
        while not app.closed and time.perf_counter() < deadline:
            await pilot.pause()
        assert app.closed, "the splash never auto-closed"
        assert app.closed[0] - app.mounted_at >= duration


async def test_a_keypress_skips_the_startup_splash_in_the_full_app() -> None:
    """End to end in the real app wiring, not a minimal host.

    The mechanism tests above pin focus routing on a bare App; this one
    boots the real TldwCli compose path, where the splash competes with
    app startup for focus and message processing. Duration is pinned to
    60s so the auto-close timer cannot account for the close: only the
    keypress can.
    """
    from unittest.mock import patch

    from tldw_chatbook import app as app_mod

    from Tests.UI.app_factory import _build_test_app

    real_get = app_mod.get_cli_setting

    def pinned_duration(section, key=None, default=None):
        if section == "splash_screen" and key == "duration":
            return 60.0
        return real_get(section, key, default)

    with patch.object(app_mod, "get_cli_setting", side_effect=pinned_duration):
        app = _build_test_app()
        async with app.run_test() as pilot:
            deadline = time.perf_counter() + 3.0
            while not (
                app.splash_screen_active and app._splash_screen_widget is not None
            ):
                assert time.perf_counter() < deadline, "splash never mounted"
                await pilot.pause()
            await pilot.press("x")

            deadline = time.perf_counter() + 10.0
            while app.splash_screen_active:
                assert (
                    time.perf_counter() < deadline
                ), "keypress did not close the startup splash"
                await pilot.pause()
            assert app._splash_screen_widget is None
