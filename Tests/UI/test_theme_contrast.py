# Readable-color gates for every registered theme, at the values that
# actually paint: theme `variables` dict entries win only over generated
# names no tcss source defines (see the mechanism note atop themes.py).
import re
from pathlib import Path

import pytest
from textual.color import Color
from textual.theme import BUILTIN_THEMES, Theme

from tldw_chatbook.css.Themes.themes import ALL_THEMES

CORE_VARIABLES = Path(__file__).resolve().parents[2] / (
    "tldw_chatbook/css/core/_variables.tcss"
)

AA = 4.5


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _ratio(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    lo, hi = min(la, lb), max(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def test_core_variables_do_not_freeze_readable_tokens_to_literals():
    """task-31264 root cause: `$name: value` in a tcss source shadows the
    theme's variables dict for that source (per-source variable scope), so a
    dark-tuned hex literal here freezes the token for every theme — the slate
    focused-button / unreadable light-theme error text symptom. The readable
    and focus tokens must therefore be *references* to Textual's generated,
    polarity-aware variables, never hex literals."""
    text = CORE_VARIABLES.read_text(encoding="utf-8")
    for token in (
        "ds-focus-bg",
        "ds-status-error-readable",
        "ds-text-placeholder",
        "ds-text-disabled-readable",
        # TASK-31429: Console rail grammar (active = primary hue, value =
        # accent hue) must follow the theme the same way.
        "ds-active-fg",
        "ds-value-fg",
    ):
        match = re.search(rf"^\${re.escape(token)}:\s*([^;]+);", text, re.M)
        assert match, f"{token} not defined in _variables.tcss"
        value = match.group(1).strip()
        assert value.startswith("$"), (
            f"{token} is frozen to literal {value!r}; use a $-reference "
            f"to a generated theme variable instead"
        )


def _resolved_variables(theme) -> dict:
    """Mirror runtime resolution: theme dict entries win over generated."""
    return {**theme.to_color_system().generate(), **(theme.variables or {})}


def _over(base: Color, color: Color) -> Color:
    """Composite a possibly-translucent color over a base surface."""
    return base.blend(Color(color.r, color.g, color.b), color.a)


def _resolve_color(value: str, base: Color) -> Color:
    """Parse a variable value ('#hex', '#hexAA', or 'auto NN%') over a base."""
    if value.startswith("auto"):
        percent = float(value.split()[1].rstrip("%")) / 100
        pole = Color(0, 0, 0) if base.brightness > 0.5 else Color(255, 255, 255)
        return base.blend(pole, percent)
    return _over(base, Color.parse(value))


@pytest.mark.parametrize("theme", ALL_THEMES, ids=lambda t: t.name)
def test_resolved_readable_tokens_clear_aa_on_every_theme(theme: Theme) -> None:
    """The values `$text-error` / `$text-muted` resolve to at runtime (theme
    variables dict over Textual's generated set) must clear AA on the theme's
    own surfaces — these feed the ds readable tokens since task-31264;
    task-31283 extended the gate from the Orb 12 to every registered theme.
    TASK-31429 adds `text-primary` / `text-accent`: the Console rail paints
    the active workspace/conversation and every label's value with them.
    TASK-32947 adds `text-success` / `text-warning`: the Settings > Theme
    card labels Save / Reset with them."""
    resolved = _resolved_variables(theme)
    surfaces = [Color.parse(resolved[k]) for k in ("surface", "panel")]
    for token in (
        "text-error",
        "text-muted",
        "text-primary",
        "text-accent",
        "text-success",
        "text-warning",
    ):
        for surface in surfaces:
            blended = _resolve_color(resolved[token], surface)
            ratio = _ratio(blended.hex, surface.hex)
            assert ratio >= AA, (
                f"{theme.name}: resolved {token} {blended.hex} is {ratio:.2f}:1 "
                f"against {surface.hex} (needs {AA}:1)"
            )


def test_user_saved_theme_gets_readable_text_hues(tmp_path) -> None:
    """TASK-31429: a theme saved from Settings ▸ Theme with a mid-tone
    primary/accent on a light canvas (pastel_dreams' palette) must still
    resolve readable `text-primary` / `text-accent` — the readability fix has
    to live on the load path, not only in the shipped catalog."""
    from tldw_chatbook.css.Themes.themes import load_user_themes

    (tmp_path / "pastel.toml").write_text(
        '[theme]\nname = "pastel_probe"\ndark = false\n'
        '[colors]\nprimary = "#F4C2C2"\naccent = "#B3D9E6"\n'
        'background = "#FFF8F8"\nsurface = "#FFFFFF"\npanel = "#FBEFEF"\n'
        'foreground = "#4A4A4A"\n',
        encoding="utf-8",
    )
    (theme,) = load_user_themes(tmp_path)
    resolved = _resolved_variables(theme)
    for token in ("text-primary", "text-accent"):
        for key in ("surface", "panel"):
            surface = Color.parse(resolved[key])
            blended = _resolve_color(resolved[token], surface)
            assert _ratio(blended.hex, surface.hex) >= AA, (
                f"user theme {token} {blended.hex} on {surface.hex}"
            )


# task-31284: the non-obscuring focus contract needs a *visible* background
# shift (TASK-345); primary-at-30% nullified it on themes whose primary sits
# near their surface. Floor chosen at 1.25x — well clear of the measured
# 1.02–1.08x failures, achievable with readable text on both polarities.
FOCUS_SHIFT_FLOOR = 1.25


@pytest.mark.parametrize("theme", ALL_THEMES, ids=lambda t: t.name)
def test_resolved_focus_tint_is_visible_and_readable_on_every_theme(
    theme: Theme,
) -> None:
    """The resolved focus tint must visibly shift the surface and keep text
    readable on the composite (task-31284; floors documented above)."""
    resolved = _resolved_variables(theme)
    surface = Color.parse(resolved["surface"])
    text = _resolve_color(resolved["text"], surface)
    tint = Color.parse(resolved["block-cursor-blurred-background"])
    composite = _over(surface, tint)
    shift = _ratio(composite.hex, surface.hex)
    assert shift >= FOCUS_SHIFT_FLOOR, (
        f"{theme.name}: focus tint {tint.hex} shifts the surface only "
        f"{shift:.2f}x (needs {FOCUS_SHIFT_FLOOR}x)"
    )
    readable = _ratio(text.hex, composite.hex)
    assert readable >= AA, (
        f"{theme.name}: text on the focus tint is {readable:.2f}:1 "
        f"(needs {AA}:1)"
    )


# TASK-33003.6 (WCAG 1.4.11): component boundaries and focus cues need 3:1
# against what they sit on. Resolved through the tcss chain: a `$ds-*: ...;`
# line in _variables.tcss shadows a theme dict entry of the same name, so
# agentic_terminal's "ds-grid-line" never paints (mechanism note, themes.py)
# and a dict-only fix would pass a dict-only test.
NON_TEXT = 3.0
CSS_ROOT = CORE_VARIABLES.parents[1]
LISTS_SHEET = CSS_ROOT / "components/_lists.tcss"
SETTINGS_SHEET = CSS_ROOT / "features/_settings.tcss"
_TCSS_DEFINITION = re.compile(r"^\$([a-z0-9-]+):\s*([^;]+);", re.MULTILINE)


def _measurable(theme: Theme) -> bool:
    """Whether the theme resolves to hex surfaces (ANSI palettes do not)."""
    try:
        resolved = _resolved_variables(theme)
        return all(Color.parse(resolved[k]).a == 1 for k in ("surface", "panel"))
    except Exception:  # noqa: BLE001 - ANSI palettes have no hex to parse
        return False


MEASURABLE_THEMES = [
    theme for theme in (*ALL_THEMES, *BUILTIN_THEMES.values()) if _measurable(theme)
]


def _painted_variables(theme: Theme) -> dict:
    """A tcss source's variable table: the theme's, then _variables.tcss on top."""
    table = _resolved_variables(theme)
    text = CORE_VARIABLES.read_text(encoding="utf-8")
    table.update((k, v.strip()) for k, v in _TCSS_DEFINITION.findall(text))
    return table


def _painted(table: dict, token: str) -> str:
    """Follow `$a: $b;` references to the value that paints."""
    value = table[token]
    while value.startswith("$"):
        value = table[value[1:]]
    return value


def _rule_token(sheet: Path, selector: str, prop: str) -> str:
    """The `$token` (last word) a rule in ``sheet`` sets ``prop`` to."""
    from Tests.UI.test_non_obscuring_focus_contract import css_block

    block = css_block(sheet.read_text(encoding="utf-8"), selector)
    match = re.search(rf"^\s*{prop}:\s*([^;]+);", block, re.MULTILINE)
    assert match, f"{selector} sets no {prop}"
    return match.group(1).split()[-1].lstrip("$")


def _assert_boundaries_visible(theme: Theme) -> None:
    table = _painted_variables(theme)
    for token in ("ds-grid-line", "ds-control-edge"):
        for key in ("surface", "panel"):
            surface = Color.parse(table[key])
            edge = _resolve_color(_painted(table, token), surface)
            ratio = _ratio(edge.hex, surface.hex)
            assert ratio >= NON_TEXT, (
                f"{theme.name}: {token} {edge.hex} is {ratio:.2f}:1 against "
                f"{key} {surface.hex} (needs {NON_TEXT}:1)"
            )


@pytest.mark.parametrize("theme", MEASURABLE_THEMES, ids=lambda t: t.name)
def test_boundary_tokens_clear_non_text_contrast_on_every_theme(theme: Theme) -> None:
    """TASK-33003.6 AC#1: grid lines (frames, dividers) and control edges read
    at 3:1 on both surfaces, on every shipped and Textual built-in theme --
    field borders measured 1.05:1 and the modal frame 1.01:1."""
    _assert_boundaries_visible(theme)


def test_user_saved_themes_get_visible_boundaries(tmp_path) -> None:
    """TASK-33003.6 AC#2: the floor lives on the load path, so a pastel palette
    saved from Settings ▸ Theme gets it, and so does a file that sets the
    boundary name by hand below the floor."""
    from tldw_chatbook.css.Themes.themes import BOUNDARY_VARIABLE, load_user_themes

    (tmp_path / "a_pastel.toml").write_text(
        '[theme]\nname = "pastel_probe"\ndark = false\n'
        '[colors]\nprimary = "#F4C2C2"\naccent = "#B3D9E6"\n'
        'background = "#FFF8F8"\nsurface = "#FFFFFF"\npanel = "#FBEFEF"\n'
        'foreground = "#4A4A4A"\n',
        encoding="utf-8",
    )
    (tmp_path / "b_hand_set.toml").write_text(
        '[theme]\nname = "hand_set_probe"\ndark = true\n'
        '[colors]\nprimary = "#3366CC"\nbackground = "#101010"\n'
        'surface = "#181818"\npanel = "#202020"\nforeground = "#E0E0E0"\n'
        f'[variables]\n{BOUNDARY_VARIABLE} = "#262626"\n',
        encoding="utf-8",
    )
    themes = load_user_themes(tmp_path)
    assert [t.name for t in themes] == ["pastel_probe", "hand_set_probe"]
    for theme in themes:
        _assert_boundaries_visible(theme)
        _assert_button_focus_keeps_contrast(theme)


BUTTONS_SHEET = CSS_ROOT / "components/_buttons.tcss"


def _assert_button_focus_keeps_contrast(theme: Theme) -> None:
    """A focused default button against a $panel card and a $surface pane.

    The resting fill is $surface; `Button:focus` drops Textual's 5%
    `background-tint`, so the rule's fill is what paints.
    """
    assert _rule_token(BUTTONS_SHEET, "Button:focus", "background-tint") == "transparent"
    table = _painted_variables(theme)
    fill = Color.parse(
        _painted(table, _rule_token(BUTTONS_SHEET, "Button:focus", "background"))
    )
    panel, surface = Color.parse(table["panel"]), Color.parse(table["surface"])
    rest = _ratio(surface.hex, panel.hex)
    on_panel, on_surface = _over(panel, fill), _over(surface, fill)
    assert _ratio(on_panel.hex, panel.hex) >= rest, (
        f"{theme.name}: focused {on_panel.hex} is {_ratio(on_panel.hex, panel.hex):.2f}:1 "
        f"on the card, resting $surface {rest:.2f}:1"
    )
    shift = _ratio(on_surface.hex, surface.hex)
    assert shift >= FOCUS_SHIFT_FLOOR, f"{theme.name}: {shift:.2f}x on $surface"
    for painted in (on_panel, on_surface):
        ink = _resolve_color(_painted(table, "ds-focus-fg"), painted)
        assert _ratio(ink.hex, painted.hex) >= AA, (theme.name, ink.hex, painted.hex)


@pytest.mark.parametrize("theme", MEASURABLE_THEMES, ids=lambda t: t.name)
def test_focused_default_button_never_loses_contrast(theme: Theme) -> None:
    """TASK-33003.6 AC#4, review round 1: the focus tint composited over a
    $panel card sat closer to the card than the resting $surface fill on 14
    themes (paradise_virtua 2.47 -> 1.12:1), once Settings stopped aliasing
    focus to $surface. The fill must also shift a $surface pane visibly
    (task-31284 floor) and keep its label AA."""
    _assert_button_focus_keeps_contrast(theme)


#: Every theme-generated name the app's tcss references (themes.py).
_GUARD_PROBE_CSS = (
    "\nScreen { border-left: solid $ds-grid-line;"
    " border-right: solid $ds-control-edge; background: $tldw-focus-fill; }\n"
)


@pytest.mark.parametrize(
    "theme", [*ALL_THEMES, *BUILTIN_THEMES.values()], ids=lambda t: t.name
)
def test_boundary_tokens_parse_under_every_registered_theme(theme: Theme) -> None:
    """TASK-33003.6 ruling 15: `_variables.tcss` names a variable no tcss
    defines, so every registered theme must supply it -- ANSI palettes (which
    have nothing to measure) included. A missing one is a stylesheet parse
    error the moment the user switches to that theme."""
    from textual.css.stylesheet import Stylesheet

    sheet = Stylesheet(variables=_resolved_variables(theme))
    sheet.add_source(
        CORE_VARIABLES.read_text(encoding="utf-8") + _GUARD_PROBE_CSS,
        read_from=(str(CORE_VARIABLES), ""),
    )
    sheet.parse()


@pytest.mark.asyncio
async def test_a_theme_that_skipped_the_guard_still_applies() -> None:
    """TASK-33003.6 ruling 15, review round 1: the fallback is unconditional.
    A Theme registered without ensure_readable_text_hues (a plugin, a test's
    bare `Theme()`) used to fail the stylesheet parse on apply with
    "undefined variable '$tldw-boundary'". The app's theme-variable defaults
    now supply the guard's values for whatever theme is current."""
    from textual.app import App

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.css.Themes.themes import (
        BOUNDARY_VARIABLE,
        ThemeVariableDefaultsMixin,
    )

    assert issubclass(TldwCli, ThemeVariableDefaultsMixin)

    class Host(ThemeVariableDefaultsMixin, App):
        CSS = CORE_VARIABLES.read_text(encoding="utf-8") + _GUARD_PROBE_CSS

    bare = Theme(name="bare_probe", primary="#3366CC", dark=True)
    assert not bare.variables
    app = Host()
    app.register_theme(bare)
    async with app.run_test() as pilot:
        app.theme = "bare_probe"
        await pilot.pause()
        edge = Color.parse(app.theme_variables[BOUNDARY_VARIABLE])
        for key in ("surface", "panel"):
            surface = Color.parse(app.theme_variables[key])
            assert _ratio(edge.hex, surface.hex) >= NON_TEXT, key
    assert not bare.variables, "the fallback must not mutate the registered theme"


@pytest.mark.parametrize("theme", MEASURABLE_THEMES, ids=lambda t: t.name)
def test_settings_rail_focus_edge_clears_non_text_contrast(theme: Theme) -> None:
    """TASK-33003.6 AC#3: a focused category row gains a thick left edge that
    reads at 3:1 against every background a row can have -- the resting
    panel and the focus tint composited over panel or surface. The edge
    column is blank at rest, so focus moves no label."""
    rest = "Button.settings-category-button"
    assert _rule_token(SETTINGS_SHEET, rest, "border-left") == "blank"
    edges = {
        _rule_token(SETTINGS_SHEET, f"{rest}{state}", "border-left")
        for state in (
            ":focus",
            ".settings-active-section:focus",
            ".settings-active-section:hover:focus",
        )
    }
    assert len(edges) == 1, edges
    table = _painted_variables(theme)
    tint = Color.parse(_painted(table, "ds-focus-bg"))
    for key in ("surface", "panel"):
        base = Color.parse(table[key])
        for row in (base, _over(base, tint)):
            edge = _resolve_color(_painted(table, next(iter(edges))), row)
            ratio = _ratio(edge.hex, row.hex)
            assert ratio >= NON_TEXT, (
                f"{theme.name}: rail focus edge {edge.hex} is {ratio:.2f}:1 "
                f"on {row.hex}"
            )


HIGHLIGHT_RULES = tuple(
    (LISTS_SHEET, f"{scope} > .option-list--option-highlighted")
    for scope in (
        "ConsoleSettingsModal OptionList",
        "ConsoleSettingsModal OptionList:focus",
        "#settings-providers-models-card OptionList",
    )
)


@pytest.mark.parametrize("theme", MEASURABLE_THEMES, ids=lambda t: t.name)
def test_choice_highlight_bar_clears_non_text_contrast(theme: Theme) -> None:
    """TASK-33003.6 AC#5: in Chat settings and Settings ▸ Providers & Models
    the highlighted option of a Select overlay or OptionList is a bar at 3:1
    against the unhighlighted options (surface or panel), and its label stays
    readable on the bar -- the shared contract paints $surface, 1.12:1."""
    table = _painted_variables(theme)
    for sheet, selector in HIGHLIGHT_RULES:
        bar_token = _rule_token(sheet, selector, "background")
        ink_token = _rule_token(sheet, selector, "color")
        for key in ("surface", "panel"):
            base = Color.parse(table[key])
            bar = _resolve_color(_painted(table, bar_token), base)
            ink = _resolve_color(_painted(table, ink_token), bar)
            assert _ratio(bar.hex, base.hex) >= NON_TEXT, (theme.name, selector, key)
            assert _ratio(ink.hex, bar.hex) >= AA, (theme.name, selector, key)
