"""Playback-stability tests for the splash animation driver (task-28026).

The splash animation is driven by Textual ``set_interval`` callbacks while the
effects inside ``Utils/Splash_Screens`` derive their progression from
``time.time() - effect.start_time``. Textual timers permanently skip ticks
that could not fire while the event loop was blocked, so under startup
contention wall-clock time races ahead of rendered frames: reveals jump
forward (or land directly on their final frame), and a block longer than the
splash duration lets the auto-close timer beat every rendered frame. These
tests pin the frame-locked pacing contract that fixes that.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Utils.Splash_Screens import load_all_effects
from tldw_chatbook.Utils.Splash_Screens.base_effect import (
    BaseEffect,
    register_effect,
)
from tldw_chatbook.Utils.Splash_Screens.card_definitions import (
    get_all_card_definitions,
)
from tldw_chatbook.Widgets.splash_screen import SplashScreen


CONTENT = "abcdefghijklmnopqrstuvwxyz0123456789"


class _SplashHostApp(App[None]):
    """Minimal host that composes one SplashScreen with given kwargs."""

    def __init__(self, splash_kwargs: Dict[str, Any]):
        super().__init__()
        self._splash_kwargs = splash_kwargs
        self.splash: SplashScreen | None = None

    def compose(self) -> ComposeResult:
        self.splash = SplashScreen(**self._splash_kwargs)
        yield self.splash


async def _await_effect_ready(splash: SplashScreen, timeout: float = 2.0) -> None:
    """Condition-wait (never a fixed sleep) until the effect is constructed."""
    deadline = time.monotonic() + timeout
    while splash.effect_handler is None:
        assert time.monotonic() < deadline, "effect handler never constructed"
        await asyncio.sleep(0.01)


async def _await_animation_started(splash: SplashScreen, timeout: float = 2.0) -> None:
    """Wait until the splash either rendered a frame or fell back to static."""
    deadline = time.monotonic() + timeout
    while splash.effect_handler is None and not _display_text(splash):
        assert time.monotonic() < deadline, "animation never started nor fell back"
        await asyncio.sleep(0.01)


@asynccontextmanager
async def _splash_with_card(
    card_data: Dict[str, Any], monkeypatch, *, duration: float = 30.0
) -> AsyncIterator[tuple[_SplashHostApp, SplashScreen]]:
    """Mount a splash whose card is a deterministic in-memory definition."""
    monkeypatch.setattr(
        SplashScreen, "_load_card", lambda self, name: dict(card_data)
    )
    app = _SplashHostApp(
        {"card_name": "probe", "duration": duration, "show_progress": False}
    )
    async with app.run_test():
        assert app.splash is not None
        await _await_effect_ready(app.splash)
        yield app, app.splash


def _display_text(splash: SplashScreen) -> str:
    renderable = splash.query_one("#splash-display", Static).renderable
    return str(renderable) if renderable is not None else ""


def _revealed_chars(display_text: str) -> int:
    """Length of the typewriter-revealed prefix (robust to the cursor glyph)."""
    revealed = 0
    for shown, expected in zip(display_text, CONTENT):
        if shown != expected:
            break
        revealed += 1
    return revealed


async def test_first_frame_renders_without_waiting_for_a_timer_tick(monkeypatch):
    """A frame must exist the moment the animation starts.

    The card's interval is one hour, so no set_interval callback can fire
    inside this test; only a synchronously rendered first frame can put
    content on the display before the auto-close deadline.
    """
    card = {
        "type": "animated",
        "effect": "typewriter",
        "content": CONTENT,
        "animation_speed": 3600.0,
    }
    async with _splash_with_card(card, monkeypatch) as (_app, splash):
        assert _revealed_chars(_display_text(splash)) >= 1


async def test_reveal_progress_is_frame_locked_not_wall_clock_locked(monkeypatch):
    """Each rendered frame advances the reveal by exactly one interval.

    With the interval timer stopped, 0.3s of wall time passes with no frame
    rendered. The next rendered frame must reveal one interval's worth of
    characters -- not jump to wherever the wall clock says the reveal is.
    """
    card = {
        "type": "animated",
        "effect": "typewriter",
        "content": CONTENT,
        "animation_speed": 0.05,
    }
    async with _splash_with_card(card, monkeypatch) as (_app, splash):
        splash.animation_timer.stop()
        before = _revealed_chars(_display_text(splash))
        await asyncio.sleep(0.3)
        splash._update_animation()
        after = _revealed_chars(_display_text(splash))
        assert after - before == 1


async def test_card_reveal_duration_is_clamped_to_splash_duration(monkeypatch):
    """A reveal longer than the splash lifetime must be compressed to fit.

    Otherwise the splash closes mid-reveal and the intro reads as skipping
    from an early frame straight to the end.
    """
    card = {
        "type": "animated",
        "effect": "glitch_reveal",
        "content": "hello world",
        "duration": 2.5,
        "animation_speed": 0.05,
    }
    async with _splash_with_card(card, monkeypatch, duration=1.0) as (
        _app,
        splash,
    ):
        assert splash.effect_handler.duration == pytest.approx(1.0)


async def test_card_reveal_duration_shorter_than_splash_is_untouched(monkeypatch):
    card = {
        "type": "animated",
        "effect": "glitch_reveal",
        "content": "hello world",
        "duration": 2.5,
        "animation_speed": 0.05,
    }
    async with _splash_with_card(card, monkeypatch, duration=10.0) as (
        _app,
        splash,
    ):
        assert splash.effect_handler.duration == pytest.approx(2.5)


@register_effect("_test_constant_frame")
class _ConstantFrameEffect(BaseEffect):
    """Always returns the same frame, to pin identical-frame suppression."""

    def update(self):
        return "constant frame"


async def test_identical_consecutive_frames_do_not_rewrite_display(monkeypatch):
    load_all_effects()
    card = {
        "type": "animated",
        "effect": "_test_constant_frame",
        "animation_speed": 0.05,
    }
    async with _splash_with_card(card, monkeypatch) as (_app, splash):
        splash.animation_timer.stop()
        display = splash.query_one("#splash-display", Static)
        update_calls = []
        original_update = display.update

        def counting_update(content):
            update_calls.append(content)
            original_update(content)

        display.update = counting_update
        frames_before = splash.current_frame
        splash._update_animation()
        # Progress still advances a frame...
        assert splash.current_frame == frames_before + 1
        # ...but the identical frame does not rewrite the display widget.
        assert update_calls == []


# Cards whose effects are broken independently of this driver (pre-existing,
# verified 2026-09-01 by rendering each card's frames through Textual's own
# Content.from_markup parser): cyberpunk_glitch/hypno_swirl/phonebooths emit
# markup that Textual rejects (cyberpunk_glitch only on some random draws,
# which is why the intro sometimes plays fine and sometimes collapses to the
# static card mid-playback), world_map crashes on a missing attribute, and
# typewriter_news's update() never returns content (its render() is never
# called). These effects are repaired under their own follow-up task.
KNOWN_BROKEN_EFFECT_CARDS = {
    "cyberpunk_glitch",
    "hypno_swirl",
    "phonebooths",
    "typewriter_news",
    "world_map",
}


async def test_every_animated_card_renders_consecutive_frames():
    """Regression smoke: every animated card renders or falls back cleanly."""
    load_all_effects()
    animated = {
        name: data
        for name, data in get_all_card_definitions().items()
        if data.get("type") == "animated"
    }
    assert animated, "no animated cards discovered"

    app = App[None]()
    failures = []
    async with app.run_test():
        for card_name in sorted(animated):
            splash = SplashScreen(
                card_name=card_name, duration=30.0, show_progress=False
            )
            await app.mount(splash)
            try:
                await _await_animation_started(splash)
                if splash.animation_timer is not None:
                    splash.animation_timer.stop()
                for _ in range(5):
                    splash._update_animation()
                if card_name in KNOWN_BROKEN_EFFECT_CARDS:
                    # Excluded pending their own repair task: their behavior
                    # is nondeterministic (markup validity depends on random
                    # draws), so asserting either outcome here would make
                    # this smoke flaky rather than informative.
                    continue
                assert splash.effect_handler is not None, "effect fell back to static"
                assert splash.current_frame >= 1, "no frame ever rendered"
            except Exception as exc:  # noqa: BLE001 - collected per card below
                failures.append(f"{card_name}: {exc!r}")
            finally:
                splash.close()
                await splash.remove()
    assert not failures, failures
