"""Painted-frame checks for the Settings theme editor (TASK-31254, TASK-31260).

These tests load the PRODUCTION bundle (`tldw_cli_modular.tcss`) on a bare
harness, because the defects live in app-level CSS: a bare `App` with widget
DEFAULT_CSS only cannot see them. Frame reads go through the real compositor
(`Screen._compositor.render_strips()`), mirroring
`Tests/UI/test_checkbox_height_render.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button

from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor

_CSS_DIR = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
_BUNDLED_CSS_PATH = _CSS_DIR / "tldw_cli_modular.tcss"
# TASK-25812 split the Settings-owned rules (e.g. `.settings-compact-input`,
# `.settings-input-row`) out of the bundle into a per-screen sheet the app
# loads with it; without it every Input here keeps Textual's tall border.
_SETTINGS_SHEET_PATH = _CSS_DIR / "screen_agentic_settings.tcss"
for _sheet in (_BUNDLED_CSS_PATH, _SETTINGS_SHEET_PATH):
    assert _sheet.is_file(), f"Production CSS sheet not found at {_sheet}"


def _rendered_text(app: App) -> str:
    """Join every compositor strip's segment text into one blob (post-CSS, post-clip)."""
    strips = app.screen._compositor.render_strips()
    return "\n".join("".join(segment.text for segment in strip) for strip in strips)


class _BundleHarness(App):
    """The editor under the real app stylesheet, inside a plain container."""

    CSS_PATH = [str(_BUNDLED_CSS_PATH), str(_SETTINGS_SHEET_PATH)]
    AUTO_FOCUS = None

    def __init__(self, tmp_path: Path, *, container_id: str = "host", container_classes: str = "", width: str = "100%"):
        super().__init__()
        self._tmp_path = tmp_path
        self._container_id = container_id
        self._container_classes = container_classes
        self._width = width

    def compose(self) -> ComposeResult:
        editor = SettingsThemeEditor(id="settings-theme-editor")
        editor.custom_themes_path = self._tmp_path
        host = Vertical(editor, id=self._container_id, classes=self._container_classes)
        host.styles.width = self._width
        yield host


@pytest.mark.asyncio
async def test_swatch_paints_hex_text_and_dark_toggle_paints_state(tmp_path):
    """TASK-31254: the colour swatch shows its hex, the Dark toggle is not a
    clipped border, and the preset target is named in visible text."""
    app = _BundleHarness(tmp_path)
    async with app.run_test(size=(120, 70)) as pilot:
        await pilot.pause()
        editor = app.query_one(SettingsThemeEditor)
        editor.color_inputs["primary"].value = "#9966FF"
        for _ in range(3):
            await pilot.pause()
        painted = _rendered_text(app)
        # once in the Input, once in the swatch
        assert painted.count("#9966FF") >= 2, painted
        dark_row = next(line for line in painted.splitlines() if "Dark theme" in line)
        assert "▔" not in dark_row, dark_row
        assert "On" in dark_row or "Off" in dark_row, dark_row
        assert "Presets fill" in painted


@pytest.mark.asyncio
async def test_invalid_hex_does_not_paint_a_black_swatch(tmp_path):
    """TASK-31254: an invalid value marks the input, keeps the last colour and
    says 'invalid' in the swatch instead of silently turning black."""
    app = _BundleHarness(tmp_path)
    async with app.run_test(size=(120, 70)) as pilot:
        await pilot.pause()
        editor = app.query_one(SettingsThemeEditor)
        editor.color_inputs["primary"].value = "#GGGGGG"
        for _ in range(3):
            await pilot.pause()
        assert editor.color_inputs["primary"].has_class("settings-invalid-input")
        swatch = editor.color_swatches["primary"]
        assert str(swatch.styles.background.hex).upper() != "#000000"
        assert "invalid" in _rendered_text(app).lower()
