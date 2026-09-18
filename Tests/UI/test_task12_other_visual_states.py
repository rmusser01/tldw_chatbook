"""Mounted state transitions for Task 12's shared token-class consumers."""

from types import SimpleNamespace
from typing import ClassVar

import pytest
from textual.color import Color
from textual.widgets import ProgressBar, Static, TextArea

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Event_Handlers.ingest_status_helper import update_status
from tldw_chatbook.Widgets.audio_troubleshooting_dialog import (
    AudioTroubleshootingDialog,
)
from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor


class _StateHarness(ConsolidatedCSSApp):
    CSS_PATH: ClassVar[list[str]] = [str(BUNDLED_STYLESHEET)]

    def compose(self):
        yield TextArea(id="ingest-status")
        yield Static(id="level-text")
        yield ProgressBar(id="level-meter")
        yield Static(classes="ds-text-error", id="error-reference")
        yield Static(classes="ds-text-warning", id="warning-reference")
        yield Static(classes="ds-text-ready", id="ready-reference")
        yield Static(classes="ds-text-primary", id="primary-reference")


@pytest.mark.asyncio
async def test_ingest_status_replaces_the_previous_semantic_color():
    app = _StateHarness()
    async with app.run_test() as pilot:
        for level, reference in (
            ("error", "error"),
            ("warning", "warning"),
            ("success", "ready"),
            ("error", "error"),
        ):
            assert update_status(app, "ingest-status", level, level)
            await pilot.pause()
            status = app.query_one("#ingest-status", TextArea)
            assert (
                status.styles.color
                == app.query_one(f"#{reference}-reference").styles.color
            )
            assert (
                sum(
                    status.has_class(f"ds-text-{name}")
                    for name in ("error", "warning", "ready")
                )
                == 1
            )


@pytest.mark.asyncio
async def test_audio_level_crosses_each_color_threshold_and_returns():
    app = _StateHarness()
    async with app.run_test() as pilot:
        holder = SimpleNamespace(
            query_one=app.query_one,
            dictation_service=None,
            is_attached=True,
            screen=app.screen,
            is_testing=True,
        )
        for level, reference in (
            (0.9, "error"),
            (0.6, "ready"),
            (0.3, "warning"),
            (0.0, "primary"),
            (0.9, "error"),
        ):
            holder.dictation_service = SimpleNamespace(
                get_audio_level=lambda level=level: level
            )
            AudioTroubleshootingDialog._update_level_meter(holder)
            await pilot.pause()
            assert (
                app.query_one("#level-text").styles.color
                == app.query_one(f"#{reference}-reference").styles.color
            )


@pytest.mark.asyncio
async def test_theme_swatch_recovers_from_invalid_and_switches_contrast(tmp_path):
    app = _StateHarness()
    async with app.run_test() as pilot:
        editor = SettingsThemeEditor()
        editor.custom_themes_path = tmp_path
        await app.mount(editor)
        await pilot.pause()
        swatch = editor.color_swatches["primary"]
        for value, background, foreground in (
            ("#FFFFFF", "#FFFFFF", "#000000"),
            ("oops", "#808080", "#FFFFFF"),
            ("#000000", "#000000", "#FFFFFF"),
            ("#EEEEEE", "#EEEEEE", "#000000"),
        ):
            editor._update_color_swatch("primary", value)
            await pilot.pause()
            assert swatch.styles.background == Color.parse(background)
            assert swatch.styles.color == Color.parse(foreground)
        for palette, colors in editor.COLOR_PRESETS.items():
            for index, color in enumerate(colors):
                preset = editor.query_one(f"#settings-theme-preset-{palette}-{index}")
                assert preset.styles.background == Color.parse(color)
