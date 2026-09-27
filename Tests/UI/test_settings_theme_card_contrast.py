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
from Tests.UI.theme_editor_helpers import open_theme_editor
from tldw_chatbook.css.Themes.themes import ALL_THEMES

THEMES = ("textual-dark", "textual-light", "gruvbox_dark", "solarized_light")
# TASK-32948 PR 2: New/Clone/Delete/Export moved to the picker; the editor
# card keeps Try (id `apply`), Save, Save as, Reset and Generate.
FILLED = ("apply", "save", "reset")
PLAIN = ("save-as", "generate")


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
    editor = await open_theme_editor(host, pilot)
    host.set_focus(None)
    await pilot.pause(0.2)
    return editor


async def _open_theme_picker(pilot, host, theme):
    """Land on the picker (Theme's default view; no Clone needed)."""
    host.theme = theme
    await _category(host, pilot, "Theme")
    host.set_focus(None)
    await pilot.pause(0.2)
    return host.screen.query_one("#settings-theme-picker")


async def _open_theme_picker_with_a_user_theme(pilot, host, theme):
    """Land on the picker with a saved (yours-only) theme highlighted, so
    the Edit/Rename/Delete/Export row is visible (TASK-32948 PR 2 Task 6,
    R19/R23)."""
    from textual.theme import Theme as TextualTheme

    from tldw_chatbook import config

    themes_dir = config._get_effective_config_path().parent / "themes"
    themes_dir.mkdir(exist_ok=True)
    (themes_dir / "mine.toml").write_text(
        '[theme]\nname = "mine"\ndark = true\n[colors]\nprimary = "#0099FF"\n',
        encoding="utf-8",
    )
    host.register_theme(TextualTheme(name="mine", primary="#0099FF", dark=True))
    picker = await _open_theme_picker(pilot, host, theme)
    lst = picker.query_one("#settings-theme-list")
    lst.highlighted = lst.get_option_index("mine")
    await pilot.pause(0.1)
    return picker


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
async def test_theme_picker_use_and_try_chips_meet_contrast(theme, request):
    """TASK-32948 Task 7: the picker's own chips reuse `.theme-editor-action`
    (TASK-32947's contrast rules), measured on the picker card, not the
    editor's."""
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        picker = await _open_theme_picker(pilot, host, theme)
        use_button = picker.query_one("#settings-theme-use", Button)
        await _show(pilot, use_button)
        fg, bg = _label_colors(host, use_button)
        assert _contrast(fg, bg) >= 4.5, (
            f"{theme}/use label {_contrast(fg, bg):.2f}:1 ({fg} on {bg})"
        )

        try_button = picker.query_one("#settings-theme-try", Button)
        await _show(pilot, try_button)
        card = _card_bg(host, try_button)
        for _, char, style in _edges(host, try_button):
            assert char.strip() and _contrast(style.color, card) >= 3.0, (
                f"{theme}/try edge {char!r} {style.color} on card {card}"
            )

        # The highlighted list row must not be carried by colour alone: bold
        # on its alphanumeric cells is the non-colour cue (Textual's
        # OptionList has no per-row cursor glyph slot to hold a `>`).
        #
        # DEVIATION from the brief's formula (`lst.region.y + (lst.highlighted
        # - lst.scroll_offset.y)`): `lst.highlighted` is an OPTION index, but
        # `scroll_offset.y` is a LINE offset, and group headers render a
        # divider line of their own (`option._divider`, textual's
        # OptionList._get_option_render) -- so the two only agree with zero
        # dividers above the highlighted row. Measured directly: at the
        # catalog's default highlight ("Textual Dark", scrolled into view),
        # the formula pointed at dy=21 (a plain, unhighlighted "Solarized
        # Light" row) while the real highlighted row painted at dy=22.
        # Locating the row by its own text is robust to the divider count.
        lst = picker.query_one("#settings-theme-list")
        await _show(pilot, lst)
        highlighted_text = lst.get_option_at_index(lst.highlighted).prompt.plain
        needle = highlighted_text.split("  ")[0].strip()
        row_cells = None
        for dy in range(lst.region.height):
            row_region = Region(lst.region.x, lst.region.y + dy, lst.region.width, 1)
            cells = list(_cells(host, row_region))
            if needle and needle in "".join(c for _, c, _ in cells):
                row_cells = cells
                break
        assert row_cells is not None, f"{theme}: highlighted row {needle!r} not visible"
        alnum_cells = [(char, style) for _, char, style in row_cells if char.isalnum()]
        assert alnum_cells, f"{theme}: no painted glyph on the highlighted row"
        for char, style in alnum_cells:
            assert style.bold, f"{theme}: highlighted row cell {char!r} is not bold"


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", THEMES)
@private_profile_test
async def test_picker_delete_chip_meets_contrast_at_rest_and_focus(theme, request):
    """TASK-32948 PR 2 Task 6 (R19/R23): the picker's yours-only Delete chip
    (`#settings-theme-picker-delete`, visible only once a user theme is
    highlighted) must clear the 4.5:1 label floor both at rest and focused,
    and focus must keep the error hue, not the generic neutral inversion
    `.settings-action-row Button:focus` gives every other chip (the gap
    TASK-32947 found for the editor's own Delete, before Task 4 removed it
    from the editor card and moved it to the picker)."""
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        picker = await _open_theme_picker_with_a_user_theme(pilot, host, theme)
        delete_button = picker.query_one("#settings-theme-picker-delete", Button)
        await _show(pilot, delete_button)

        rest_fg, rest_bg = _label_colors(host, delete_button)
        assert _contrast(rest_fg, rest_bg) >= 4.5, (
            f"{theme}/delete rest {_contrast(rest_fg, rest_bg):.2f}:1 ({rest_fg} on {rest_bg})"
        )

        delete_button.focus()
        await pilot.pause(0.1)
        focus_fg, focus_bg = _label_colors(host, delete_button)
        assert _contrast(focus_fg, focus_bg) >= 4.5, (
            f"{theme}/delete focused {_contrast(focus_fg, focus_bg):.2f}:1"
        )
        # Focus is a strong state change (inversion), not a neutral wash.
        assert _contrast(focus_bg, rest_bg) >= 3.0, (
            f"{theme}/delete focus shift {_contrast(focus_bg, rest_bg):.2f}:1"
        )
        # The error hue survives focus: one of the focused fg/bg pair is
        # still the rest-state error colour (fg or bg, inversion swaps them).
        assert {rest_fg, rest_bg} & {focus_fg, focus_bg}, (
            f"{theme}/delete lost its error hue on focus"
        )
        host.set_focus(None)
        await pilot.pause(0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", THEMES)
@private_profile_test
async def test_picker_use_chip_meets_contrast_at_rest_and_focus(theme, request):
    """R24 (extends R19/R23): the picker's Use chip (`#settings-theme-use`,
    variant primary) is the same defect class as the Delete chip above --
    `#settings-theme-card-column` only had a colour-keeping `-error:focus`
    rule, so a focused Use chip still fell back to the generic neutral
    `.settings-action-row Button:focus`."""
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        picker = await _open_theme_picker(pilot, host, theme)
        use_button = picker.query_one("#settings-theme-use", Button)
        await _show(pilot, use_button)

        rest_fg, rest_bg = _label_colors(host, use_button)
        assert _contrast(rest_fg, rest_bg) >= 4.5, (
            f"{theme}/use rest {_contrast(rest_fg, rest_bg):.2f}:1 ({rest_fg} on {rest_bg})"
        )

        use_button.focus()
        await pilot.pause(0.1)
        focus_fg, focus_bg = _label_colors(host, use_button)
        assert _contrast(focus_fg, focus_bg) >= 4.5, (
            f"{theme}/use focused {_contrast(focus_fg, focus_bg):.2f}:1"
        )
        # Focus is a strong state change (inversion), not a neutral wash.
        assert _contrast(focus_bg, rest_bg) >= 3.0, (
            f"{theme}/use focus shift {_contrast(focus_bg, rest_bg):.2f}:1"
        )
        # The primary hue survives focus: one of the focused fg/bg pair is
        # still the rest-state primary colour (fg or bg, inversion swaps them).
        assert {rest_fg, rest_bg} & {focus_fg, focus_bg}, (
            f"{theme}/use lost its primary hue on focus"
        )
        host.set_focus(None)
        await pilot.pause(0.05)


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
                # Try/Save/Reset keep their variant hue on focus.
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
