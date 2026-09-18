"""Console background effect widget tests."""

from typing import ClassVar

import pytest
from textual.app import ComposeResult
from textual.containers import Container

from Tests.private_profile import private_profile_test

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Utils.console_background_effects import (
    MAX_CONSOLE_BACKGROUND_FPS,
    MIN_CONSOLE_BACKGROUND_FPS,
    ConsoleBackgroundEffectSettings,
)
from tldw_chatbook.Widgets.Console.console_background_effect import (
    ConsoleBackgroundEffect,
    ConsoleTranscriptSurface,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


class EffectHarness(ConsolidatedCSSApp):
    def __init__(self, settings: ConsoleBackgroundEffectSettings) -> None:
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        yield ConsoleBackgroundEffect(self.settings, id="console-background-effect")


class SurfaceHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscriptSurface(
            ConsoleBackgroundEffectSettings(enabled=True, effect="rain", fps=6),
            id="console-transcript-surface",
        )


class StyledSurfaceHarness(SurfaceHarness):
    CSS_PATH: ClassVar = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        with Container(id="console-shell"):
            yield from super().compose()


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
        assert effect._timer is not None


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


@pytest.mark.asyncio
async def test_deferred_mount_callback_does_not_restart_removed_effect(monkeypatch):
    pending = []
    monkeypatch.setattr(
        ConsoleBackgroundEffect,
        "call_after_refresh",
        lambda self, callback: pending.append(callback),
    )
    app = EffectHarness(
        ConsoleBackgroundEffectSettings(enabled=True, effect="matrix", fps=6)
    )
    async with app.run_test(size=(60, 18)) as pilot:
        await pilot.pause()
        effect = app.query_one(ConsoleBackgroundEffect)
        assert len(pending) == 1
        assert effect._timer is None
        await effect.remove()
        assert not effect.is_attached
        pending.pop()()
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
@private_profile_test
async def test_console_transcript_scope_mounts_effect_without_hiding_transcript(
    request,
):
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
        await _wait_for_selector(
            console, pilot, "#console-transcript-background-effect"
        )

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        effect = console.query_one(
            "#console-transcript-background-effect",
            ConsoleBackgroundEffect,
        )
        assert effect.is_effect_active is True
        assert not console.query(
            "#console-left-rail #console-transcript-background-effect"
        )
        assert transcript.region.y == effect.region.y


@pytest.mark.asyncio
@private_profile_test
async def test_console_background_disabled_does_not_start_active_effect(request):
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
        await _wait_for_selector(
            console, pilot, "#console-transcript-background-effect"
        )

        effect = console.query_one(
            "#console-transcript-background-effect",
            ConsoleBackgroundEffect,
        )
        assert effect.is_effect_active is False


@pytest.mark.asyncio
@private_profile_test
async def test_console_workbench_scope_does_not_start_transcript_effect(request):
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
        await _wait_for_selector(
            console, pilot, "#console-transcript-background-effect"
        )

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


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_effect_is_painted_in_empty_transcript_space_without_changing_messages(
    theme,
):
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )

    app = StyledSurfaceHarness()
    app.theme = theme
    async with app.run_test(size=(80, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    id="paint-user",
                    role=ConsoleMessageRole.USER,
                    content="Sample question",
                ),
                ConsoleChatMessage(
                    id="paint-assistant",
                    role=ConsoleMessageRole.ASSISTANT,
                    content="Sample answer",
                ),
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause(0.2)
        effect = app.query_one(ConsoleBackgroundEffect)
        surface = app.query_one(ConsoleTranscriptSurface)
        surface.update_settings(
            ConsoleBackgroundEffectSettings(
                enabled=True, effect="matrix", intensity="high", fps=9
            )
        )
        await pilot.pause(0.2)
        row = transcript.query_one("#console-message-paint-user")
        assert row.region.y > transcript.region.y + 2

        def empty_space():
            region = transcript.content_region
            return "\n".join(
                strip.crop(region.x, region.right).text
                for strip in app.screen._compositor.render_strips()[
                    region.y : row.region.y - 1
                ]
            )

        assert effect.is_effect_active
        painted_transcript = "\n".join(
            strip.text for strip in app.screen._compositor.render_strips()
        )
        assert "Sample question" in painted_transcript
        assert "Sample answer" in painted_transcript
        transcript.focus()
        await pilot.press("down")
        await pilot.pause(0.2)
        selected = transcript.selected_message_id
        assert selected == "paint-user"
        before_scroll = transcript.scroll_offset
        before_text = transcript.to_plain_text()
        assert any(char.isalnum() for char in empty_space()), (
            "Running effect is hidden by transcript paint"
        )
        surface.update_settings(ConsoleBackgroundEffectSettings())
        await pilot.pause(0.2)
        assert not empty_space().strip()
        assert effect._timer is None
        assert transcript.query_one("#console-message-paint-user") is row
        assert transcript.scroll_offset == before_scroll
        assert transcript.to_plain_text() == before_text
        assert transcript.selected_message_id == selected
        surface.update_settings(
            ConsoleBackgroundEffectSettings(
                enabled=True, effect="matrix", intensity="high"
            )
        )
        await transcript.recompose()
        await pilot.pause(0.2)
        row = transcript.query_one("#console-message-paint-user")
        recreated = app.query_one(ConsoleBackgroundEffect)
        assert recreated is not effect
        assert effect._timer is None
        assert recreated._timer is not None
        assert any(char.isalnum() for char in empty_space())
        assert transcript.to_plain_text() == before_text
        for enabled in (True, False):
            surface.update_settings(
                ConsoleBackgroundEffectSettings(enabled=enabled, effect="matrix")
            )
            transcript.focus()
            if transcript.selected_message_id is None:
                await pilot.press("down")
            await pilot.pause(0.2)
            assert transcript.selected_message_id == "paint-user"
            assert await pilot.click(recreated, offset=(3, 2))
            await pilot.pause(0.2)
            assert transcript.selected_message_id is None


@pytest.mark.asyncio
async def test_background_layer_does_not_change_scrollback_or_reconcile_rows():
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )

    app = StyledSurfaceHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        surface = app.query_one(ConsoleTranscriptSurface)
        surface.update_settings(ConsoleBackgroundEffectSettings())
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    id=f"scroll-{index}",
                    role=ConsoleMessageRole.USER,
                    content=f"Message {index}",
                )
                for index in range(20)
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause(0.2)
        transcript.scroll_home(animate=False)
        await pilot.pause(0.2)
        before_rows = dict(transcript._row_widgets)
        before_text = transcript.to_plain_text()
        before_max = transcript.max_scroll_y
        assert before_max > 0
        surface.update_settings(
            ConsoleBackgroundEffectSettings(enabled=True, effect="rain")
        )
        await pilot.pause(0.2)
        effect = app.query_one(ConsoleBackgroundEffect)
        top_region = effect.region
        transcript.scroll_end(animate=False)
        await pilot.pause(0.2)
        assert transcript.scroll_y == before_max
        assert effect.region == top_region
        assert transcript.max_scroll_y == before_max
        await transcript.refresh_messages()
        assert transcript._row_widgets == before_rows
        surface.update_settings(ConsoleBackgroundEffectSettings())
        await pilot.pause(0.2)
        assert transcript.scroll_y == before_max
        assert transcript.max_scroll_y == before_max
        assert transcript._row_widgets == before_rows
        assert transcript.to_plain_text() == before_text
