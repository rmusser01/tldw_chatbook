"""Console background effect widget tests."""

import pytest

from textual.app import App, ComposeResult

from tldw_chatbook.Utils.console_background_effects import (
    ConsoleBackgroundEffectSettings,
    MAX_CONSOLE_BACKGROUND_FPS,
    MIN_CONSOLE_BACKGROUND_FPS,
)
from tldw_chatbook.Widgets.Console.console_background_effect import (
    ConsoleBackgroundEffect,
    ConsoleTranscriptSurface,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness


class EffectHarness(App[None]):
    def __init__(self, settings: ConsoleBackgroundEffectSettings) -> None:
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        yield ConsoleBackgroundEffect(self.settings, id="console-background-effect")


class SurfaceHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscriptSurface(
            ConsoleBackgroundEffectSettings(enabled=True, effect="rain", fps=6),
            id="console-transcript-surface",
        )


def test_console_background_effect_disabled_is_inactive():
    effect = ConsoleBackgroundEffect(
        ConsoleBackgroundEffectSettings(enabled=False, effect="matrix")
    )

    assert effect.can_focus is False
    assert effect.is_effect_active is False


def test_console_background_effect_clamps_frame_rate():
    high_fps_effect = ConsoleBackgroundEffect(
        ConsoleBackgroundEffectSettings(enabled=True, effect="rain", fps=100_000)
    )
    low_fps_effect = ConsoleBackgroundEffect(
        ConsoleBackgroundEffectSettings(enabled=True, effect="rain", fps=-50)
    )

    assert high_fps_effect.frame_rate == MAX_CONSOLE_BACKGROUND_FPS
    assert low_fps_effect.frame_rate == MIN_CONSOLE_BACKGROUND_FPS


@pytest.mark.asyncio
async def test_console_background_effect_enabled_renders_frame():
    app = EffectHarness(
        ConsoleBackgroundEffectSettings(
            enabled=True,
            effect="matrix",
            scope="transcript",
            intensity="low",
            fps=6,
        )
    )

    async with app.run_test(size=(60, 18)) as pilot:
        effect = app.query_one("#console-background-effect", ConsoleBackgroundEffect)
        await pilot.pause(0.2)

        assert effect.is_effect_active is True
        assert effect.frame_text(width=40, height=8).strip()


@pytest.mark.asyncio
async def test_console_background_effect_update_settings_stops_timer():
    app = EffectHarness(
        ConsoleBackgroundEffectSettings(enabled=True, effect="rain", fps=6)
    )

    async with app.run_test(size=(60, 18)) as pilot:
        effect = app.query_one("#console-background-effect", ConsoleBackgroundEffect)
        await pilot.pause(0.2)

        assert effect.is_effect_active is True

        effect.update_settings(
            ConsoleBackgroundEffectSettings(enabled=False, effect="rain", fps=6)
        )
        await pilot.pause(0.1)

        assert effect.is_effect_active is False
        assert effect._timer is None


def test_console_transcript_surface_preserves_transcript_identity_and_id():
    transcript = ConsoleTranscript(id="console-native-transcript")
    surface = ConsoleTranscriptSurface(
        ConsoleBackgroundEffectSettings(enabled=True, effect="snow"),
        transcript=transcript,
    )

    assert surface.transcript is transcript
    assert surface.transcript.id == "console-native-transcript"


@pytest.mark.asyncio
async def test_console_transcript_surface_keeps_effect_behind_transcript():
    app = SurfaceHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.1)
        surface = app.query_one("#console-transcript-surface", ConsoleTranscriptSurface)
        effect = app.query_one(
            "#console-transcript-background-effect",
            ConsoleBackgroundEffect,
        )
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)

        assert effect.is_mounted
        assert transcript.is_mounted
        assert effect.region == surface.region
        assert transcript.region.y == surface.region.y


@pytest.mark.asyncio
async def test_console_transcript_scope_mounts_effect_without_hiding_transcript():
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": True,
            "effect": "matrix",
            "scope": "transcript",
            "intensity": "low",
            "fps": 6,
        }
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_selector(console, pilot, "#console-transcript-background-effect")

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        effect = console.query_one(
            "#console-transcript-background-effect",
            ConsoleBackgroundEffect,
        )
        assert effect.is_effect_active is True
        assert not console.query("#console-left-rail #console-transcript-background-effect")
        assert transcript.region.y == effect.region.y


@pytest.mark.asyncio
async def test_console_background_disabled_does_not_start_active_effect():
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": False,
            "effect": "matrix",
            "scope": "transcript",
            "intensity": "low",
            "fps": 6,
        }
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_selector(console, pilot, "#console-transcript-background-effect")

        effect = console.query_one(
            "#console-transcript-background-effect",
            ConsoleBackgroundEffect,
        )
        assert effect.is_effect_active is False


@pytest.mark.asyncio
async def test_console_workbench_scope_does_not_start_transcript_effect():
    app = _build_test_app()
    app.app_config["console"] = {
        "background_effects": {
            "enabled": True,
            "effect": "matrix",
            "scope": "workbench",
            "intensity": "low",
            "fps": 6,
        }
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_selector(console, pilot, "#console-transcript-background-effect")

        effect = console.query_one(
            "#console-transcript-background-effect",
            ConsoleBackgroundEffect,
        )
        assert effect.is_effect_active is False


# --- task-261: per-frame grid cache -----------------------------------------
#
# `render_line` used to rebuild the full W×H grid via `frame_text()` once PER
# LINE (O(W·H²) per repaint). These tests prove the cache engages (one grid
# build per frame regardless of line count) and that the rendered output is
# byte-identical to the uncached `frame_text()` result.


@pytest.mark.asyncio
async def test_render_line_builds_the_frame_grid_once_per_repaint():
    app = EffectHarness(
        ConsoleBackgroundEffectSettings(enabled=True, effect="snow", fps=6)
    )

    async with app.run_test(size=(60, 18)) as pilot:
        effect = app.query_one("#console-background-effect", ConsoleBackgroundEffect)
        await pilot.pause(0.1)
        # Drive frames manually so the interval timer can't advance the frame
        # serial between render_line calls.
        effect._stop_timer()

        width, height = effect.size.width, effect.size.height
        assert width > 0 and height > 0

        real_frame_text = effect.frame_text
        calls: list[tuple[int, int]] = []

        def counting_frame_text(w: int, h: int) -> str:
            calls.append((w, h))
            return real_frame_text(w, h)

        effect.frame_text = counting_frame_text
        effect._invalidate_frame_cache()

        strips = [effect.render_line(y) for y in range(height)]

        assert calls == [(width, height)], (
            "a full repaint must compute the frame grid exactly once"
        )

        # Behavior parity: every rendered line matches the uncached grid.
        expected_lines = real_frame_text(width, height).splitlines()
        rendered_lines = [strip.text for strip in strips]
        assert rendered_lines == expected_lines


@pytest.mark.asyncio
async def test_frame_cache_invalidates_on_tick_settings_and_resize():
    app = EffectHarness(
        ConsoleBackgroundEffectSettings(enabled=True, effect="rain", fps=6)
    )

    async with app.run_test(size=(60, 18)) as pilot:
        effect = app.query_one("#console-background-effect", ConsoleBackgroundEffect)
        await pilot.pause(0.1)
        effect._stop_timer()

        width, height = effect.size.width, effect.size.height
        first = effect._frame_lines(width, height)
        assert effect._frame_lines(width, height) is first, (
            "same frame + same size must reuse the cached lines"
        )

        # A frame tick must produce a fresh grid.
        effect._advance_frame()
        after_tick = effect._frame_lines(width, height)
        assert after_tick is not first

        # A resize (different requested dimensions) must produce a fresh grid.
        smaller = effect._frame_lines(width - 5, height - 3)
        assert smaller is not after_tick
        assert len(smaller) == height - 3
        assert all(len(line) == width - 5 for line in smaller)

        # A settings change must produce a fresh grid.
        effect.update_settings(
            ConsoleBackgroundEffectSettings(enabled=True, effect="matrix", fps=6)
        )
        after_settings = effect._frame_lines(width - 5, height - 3)
        assert after_settings is not smaller
