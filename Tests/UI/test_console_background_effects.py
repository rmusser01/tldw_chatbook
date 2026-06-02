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
