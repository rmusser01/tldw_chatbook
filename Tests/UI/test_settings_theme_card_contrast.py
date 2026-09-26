"""TASK-32947: Settings > Theme card controls meet contrast and focus floors.

Measured from compositor-painted segments (what the terminal actually
receives) on the real Settings destination under the production stylesheet,
at 190x55, across the two Textual built-ins and two shipped themes.
"""

from __future__ import annotations

import pytest
from textual.geometry import Region
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_overview_search_journeys import _category
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from tldw_chatbook.css.Themes.themes import ALL_THEMES

THEMES = ("textual-dark", "textual-light", "gruvbox_dark", "solarized_light")
FILLED = ("apply", "save", "reset", "delete")
PLAIN = ("new", "clone", "export", "generate", "set-default")


def _host():
    """The real Settings destination under the production stylesheet."""
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    for theme in ALL_THEMES:
        host.register_theme(theme)
    return host


def _luminance(color) -> float:
    rgb = color.get_truecolor()

    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(rgb.red)
        + 0.7152 * channel(rgb.green)
        + 0.0722 * channel(rgb.blue)
    )


def _contrast(first, second) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _cells(host, region):
    """(x, char, style) for every painted cell inside ``region``."""
    strips = host.screen._compositor.render_strips()
    for y in range(region.y, region.bottom):
        x = 0
        for segment in strips[y]:
            for char in segment.text:
                if region.x <= x < region.right:
                    yield x, char, segment.style
                x += 1


def _label_colors(host, widget):
    for _, char, style in _cells(host, widget.region):
        if char.isalnum() and style is not None and style.color and style.bgcolor:
            return style.color, style.bgcolor
    raise AssertionError(f"no painted glyph in {widget.id}")


def _card_bg(host, widget):
    """The fill of the margin cell right of ``widget`` -- bare card surface."""
    region = widget.region
    for _, _, style in _cells(host, Region(region.right, region.y, 1, 1)):
        return style.bgcolor
    raise AssertionError(f"nothing painted beside {widget.id}")


def _edges(host, widget):
    """The first and last painted cells of a one-row widget."""
    cells = list(_cells(host, widget.region))
    return cells[0], cells[-1]


async def _open_theme_card(pilot, host, theme):
    host.theme = theme
    await _category(host, pilot, "Theme")
    host.set_focus(None)
    await pilot.pause(0.2)
    return host.screen.query_one("#settings-theme-editor")


async def _show(pilot, widget):
    widget.scroll_visible(animate=False, immediate=True)
    await pilot.pause(0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", THEMES)
@private_profile_test
async def test_theme_card_button_labels_and_boundaries(theme, request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        editor = await _open_theme_card(pilot, host, theme)
        for name in FILLED + PLAIN:
            button = editor.query_one(f"#settings-theme-{name}", Button)
            await _show(pilot, button)
            fg, bg = _label_colors(host, button)
            assert _contrast(fg, bg) >= 4.5, (
                f"{theme}/{name} label {_contrast(fg, bg):.2f}:1 ({fg} on {bg})"
            )
            # A button reads as a button, not bold prose: both edge cells carry
            # a glyph that clears the 3:1 non-text floor against the card.
            card = _card_bg(host, button)
            for _, char, style in _edges(host, button):
                assert char.strip() and _contrast(style.color, card) >= 3.0, (
                    f"{theme}/{name} edge {char!r} {style.color} on card {card}"
                )


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", THEMES)
@private_profile_test
async def test_focused_theme_buttons_keep_variant_meaning(theme, request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        editor = await _open_theme_card(pilot, host, theme)
        for name in FILLED + PLAIN:
            button = editor.query_one(f"#settings-theme-{name}", Button)
            await _show(pilot, button)
            rest_fg, rest_bg = _label_colors(host, button)
            button.focus()
            await pilot.pause(0.1)
            fg, bg = _label_colors(host, button)
            assert _contrast(fg, bg) >= 4.5, (
                f"{theme}/{name} focused {_contrast(fg, bg):.2f}:1"
            )
            # Focus is a strong state change (inversion), not a 1.1:1 tint.
            assert _contrast(bg, rest_bg) >= 3.0, (
                f"{theme}/{name} focus shift {_contrast(bg, rest_bg):.2f}:1"
            )
            if name in FILLED:
                # Apply/Save/Reset/Delete keep their variant hue on focus.
                assert {rest_fg, rest_bg} & {fg, bg}, f"{theme}/{name} lost its hue"
            host.set_focus(None)
            await pilot.pause(0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", THEMES)
@private_profile_test
async def test_preset_swatches_have_edges_and_a_contrasting_focus_ring(theme, request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        editor = await _open_theme_card(pilot, host, theme)
        swatches = list(editor.query(".color-preset-swatch"))
        assert len(swatches) == 40
        for swatch in swatches:
            await _show(pilot, swatch)
            card = _card_bg(host, swatch)
            for _, char, style in _edges(host, swatch):
                assert char.strip() and _contrast(style.color, card) >= 3.0, (
                    f"{theme}/{swatch.id} edge {char!r} {style.color} on {card}"
                )
            fill = list(_cells(host, swatch.region))[1][2].bgcolor
            assert fill != card, f"{theme}/{swatch.id} shows no colour cell"

        swatch = swatches[0]
        await _show(pilot, swatch)
        rest = [(char, style.color) for _, char, style in _cells(host, swatch.region)]
        swatch.focus()
        await pilot.pause(0.1)
        focused = list(_cells(host, swatch.region))
        card = _card_bg(host, swatch)
        assert [(c, s.color) for _, c, s in focused] != rest, "focus paints nothing"
        for _, char, style in (focused[0], focused[-1]):
            # The ring is judged against the card beside it (3:1 non-text).
            assert char.strip() and _contrast(style.color, card) >= 3.0, (
                f"{theme} focus ring {style.color} on card {card}"
            )

        # The "Dark theme" checkbox's off glyph used to be painted in the
        # card's own fill (1.0:1); off must now show a visible, non-success X.
        checkbox = editor.query_one("#settings-theme-dark-mode")
        checkbox.value = False
        await _show(pilot, checkbox)
        await pilot.pause(0.1)
        glyph = next(s for _, c, s in _cells(host, checkbox.region) if c == "X")
        assert _contrast(glyph.color, glyph.bgcolor) >= 3.0, (
            f"{theme} off glyph {glyph.color} on {glyph.bgcolor}"
        )
