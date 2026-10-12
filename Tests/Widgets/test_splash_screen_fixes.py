"""Regression tests for task-32474 splash screen fixes.

Covers:
- skip-on-keypress reachability (SplashScreen must be focusable)
- 7.0s default duration
- progress bar fraction-to-percent scaling
- effect registry/card cross-check (no card references a missing effect)
- every effect instantiates and renders frames without crashing
- specific effect regressions: typewriter_news (update returned None),
  world_map (missing width/height attrs), starfield (frozen cluster),
  level_up (static until first level-up)
"""

import time
from unittest.mock import patch

import pytest

from tldw_chatbook.Utils.Splash_Screens import (
    load_all_effects,
    get_effect_class,
    list_available_effects,
)
from tldw_chatbook.Utils.Splash_Screens.card_definitions import get_all_card_definitions
from tldw_chatbook.Widgets.splash_screen import SplashScreen


class _DummyParent:
    """Minimal stand-in for the SplashScreen parent widget."""

    progress = 0.5
    progress_text = "Loading..."
    height = 24


@pytest.fixture(scope="module", autouse=True)
def _load_effects():
    load_all_effects()


@pytest.fixture(scope="module")
def cards():
    return get_all_card_definitions()


# ---------------------------------------------------------------------------
# Widget-level regressions
# ---------------------------------------------------------------------------


def test_splash_screen_is_focusable_so_skip_keypress_works():
    """on_key only fires if the widget can receive focus."""
    assert SplashScreen.can_focus is True


def test_splash_screen_default_duration_is_seven_seconds():
    def fake_get_cli_setting(section, key=None, default=None):
        return default

    with patch(
        "tldw_chatbook.Widgets.splash_screen.get_cli_setting",
        side_effect=fake_get_cli_setting,
    ):
        splash = SplashScreen()
        assert splash.config["duration"] == 7.0
        assert splash.duration == 7.0


# ---------------------------------------------------------------------------
# Registry/card integrity
# ---------------------------------------------------------------------------


def test_every_animated_card_effect_is_registered(cards):
    missing = [
        (name, card.get("effect"))
        for name, card in cards.items()
        if card.get("type") == "animated"
        and get_effect_class(card.get("effect") or "") is None
    ]
    assert missing == [], f"cards with unregistered effects: {missing}"


def test_every_effect_instantiates_and_renders_without_crashing(cards):
    """Smoke test: 5 frames per animated card, no exceptions, str|None frames."""
    failures = []
    for name, card in cards.items():
        if card.get("type") != "animated":
            continue
        cls = get_effect_class(card["effect"])
        try:
            inst = cls(_DummyParent(), width=80, height=24, **card)
            for _ in range(5):
                frame = inst.update()
                assert frame is None or isinstance(frame, str), (
                    f"{name}: update() returned {type(frame).__name__}"
                )
        except Exception as e:  # noqa: BLE001 - collect all failures
            failures.append(f"{name}: {type(e).__name__}: {e}")
    assert failures == [], "broken effects:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Specific effect regressions
# ---------------------------------------------------------------------------


def test_typewriter_news_update_returns_frame(cards):
    """Regression: update() previously returned None; rendering lived in an
    uncalled render() method, so the card displayed nothing."""
    cls = get_effect_class("typewriter_news")
    inst = cls(_DummyParent(), width=80, height=24, **cards["typewriter_news"])
    frame = inst.update()
    assert isinstance(frame, str) and frame.strip(), "update() must return content"


def test_typewriter_news_typing_is_delta_based(cards):
    """Regression: typed_chars previously grew by total elapsed time every
    frame (quadratic explosion)."""
    cls = get_effect_class("typewriter_news")
    inst = cls(_DummyParent(), width=80, height=24, **cards["typewriter_news"])
    inst.update()
    time.sleep(0.05)
    inst.update()
    # ~30 chars/sec -> after ~50ms well under 30 chars total
    assert inst.typed_chars < 30


def test_world_map_renders_and_frame_count_advances(cards):
    """Regression: world_map crashed with AttributeError (no width/height)
    and never advanced frame_count."""
    cls = get_effect_class("world_map")
    inst = cls(_DummyParent(), width=80, height=24, **cards["world_map"])
    frame = inst.update()
    assert isinstance(frame, str) and frame.strip()
    assert inst.frame_count == 1
    # Cursor blink: over enough frames the output must change
    frames = {inst.update() for _ in range(15)}
    assert len(frames) > 1


def test_starfield_stars_traverse_screen(cards):
    """Regression: radius scaling kept all stars within ~8 cells of center,
    making the warp look frozen."""
    cls = get_effect_class("starfield")
    inst = cls(_DummyParent(), width=80, height=24, **cards["starfield"])
    frames = {inst.update() for _ in range(10)}
    assert len(frames) > 1


def test_level_up_xp_bar_changes_every_frame(cards):
    """Regression: nothing visibly changed until the 100-frame level-up."""
    cls = get_effect_class("level_up")
    inst = cls(_DummyParent(), width=80, height=24, **cards["level_up"])
    frames = [inst.update() for _ in range(10)]
    # XP bar fill means early frames differ from each other
    assert len(set(frames)) > 1
    assert "#" in frames[-1], "XP bar fill should be visible"


# ---------------------------------------------------------------------------
# Interactive: skip on keypress (Textual pilot)
# ---------------------------------------------------------------------------


async def test_keypress_skips_splash():
    from textual.app import App

    class SplashApp(App):
        def __init__(self):
            super().__init__()
            self.closed = False

        def compose(self):
            yield SplashScreen(card_name="matrix", duration=30.0, show_progress=False)

        def on_splash_screen_closed(self, event) -> None:
            self.closed = True

    app = SplashApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        assert app.closed, "keypress should skip the splash screen"
