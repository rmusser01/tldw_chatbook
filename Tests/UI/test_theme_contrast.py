# Readable-color gates for every registered theme, at the values that
# actually paint: theme `variables` dict entries win only over generated
# names no tcss source defines (see the mechanism note atop themes.py).
import ast
import functools
import re
from pathlib import Path
from typing import NamedTuple

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
    """Parse a variable value ('#hex', '#hexAA', '#hex NN%' or 'auto NN%') over a base."""
    if value.startswith("auto"):
        percent = float(value.split()[1].rstrip("%")) / 100
        pole = Color(0, 0, 0) if base.brightness > 0.5 else Color(255, 255, 255)
        return base.blend(pole, percent)
    suffixed = re.fullmatch(r"(.+) (\d{1,3})%", value)
    if suffixed:
        color = Color.parse(suffixed[1]).multiply_alpha(int(suffixed[2]) / 100)
        return _over(base, color)
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


#: MainNavigationBar.BUNDLED_CSS, read on use (the widget tier's nav rules).
NAV_BAR_CSS = "MainNavigationBar.BUNDLED_CSS"


def _rule_token(sheet: Path | str, selector: str, prop: str) -> str:
    """The `$token` (last word) a rule in ``sheet`` sets ``prop`` to."""
    from Tests.UI.test_non_obscuring_focus_contract import css_block

    if sheet == NAV_BAR_CSS:
        from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar

        text = MainNavigationBar.BUNDLED_CSS
    else:
        text = sheet.read_text(encoding="utf-8")
    block = css_block(text, selector)
    # `(?:^|;)`: _workflows.tcss writes a whole rule on one line.
    match = re.search(rf"(?:^|;)\s*{prop}:\s*([^;]+);", block, re.MULTILINE)
    assert match, f"{selector} sets no {prop}"
    return match.group(1).split()[-1].lstrip("$")


def _assert_edge_visible(theme: Theme, table: dict, token: str, on: str) -> None:
    """``$token`` painted as a line on the ``$on`` fill reads at 3:1."""
    fill = Color.parse(_painted(table, on))
    edge = _resolve_color(_painted(table, token), fill)
    ratio = _ratio(edge.hex, fill.hex)
    assert ratio >= NON_TEXT, (
        f"{theme.name}: {token} {edge.hex} is {ratio:.2f}:1 against "
        f"{on} {fill.hex} (needs {NON_TEXT}:1)"
    )


def _assert_boundaries_visible(theme: Theme) -> None:
    table = _painted_variables(theme)
    # TASK-33003.15 adds ds-column-line (pane frames, rail rules).
    for token in ("ds-grid-line", "ds-control-edge", "ds-column-line"):
        for key in ("surface", "panel"):
            _assert_edge_visible(theme, table, token, key)


@pytest.mark.parametrize("theme", MEASURABLE_THEMES, ids=lambda t: t.name)
def test_boundary_tokens_clear_non_text_contrast_on_every_theme(theme: Theme) -> None:
    """TASK-33003.6 AC#1: grid lines (frames, dividers) and control edges read
    at 3:1 on both surfaces, on every shipped and Textual built-in theme --
    field borders measured 1.05:1 and the modal frame 1.01:1."""
    _assert_boundaries_visible(theme)


LIBRARY_SHEET = CSS_ROOT / "features/_library.tcss"
LIBRARY_PANE_FRAMES = (
    "#library-shell-grid",
    "#library-rail",
    "#library-file-notes-rail",
    "#library-canvas",
)


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("theme", MEASURABLE_THEMES, ids=lambda t: t.name)
def test_library_pane_frames_clear_non_text_contrast(theme: Theme) -> None:
    """TASK-33003.15: Library's pane frames painted $surface-lighten-2, 1.28:1
    in textual-dark. Each frame reads at 3:1 on surface and on panel."""
    table = _painted_variables(theme)
    for selector in LIBRARY_PANE_FRAMES:
        token = _rule_token(LIBRARY_SHEET, selector, "border")
        for key in ("surface", "panel"):
            _assert_edge_visible(theme, table, token, key)


WORKFLOWS_SHEET = CSS_ROOT / "features/_workflows.tcss"
FORMS_SHEET = CSS_ROOT / "components/_forms.tcss"
WORKFLOW_PANES = {
    "WorkflowLibrary#workflows-library": "#workflows-library",
    "WorkflowNavigator#workflows-navigator": "#workflows-navigator",
}
WORKFLOW_FIELD_FOCUS = "WorkflowLibrary#workflows-library Input:focus"
WORKFLOW_PLACEHOLDER_FOCUS = "#workflow-library-search:focus > .input--placeholder"
WORKFLOW_MARKS_FOCUS = [f"#workflow-library-search:focus > .input--{mark}" for mark in ("selection", "cursor")]


def _assert_ink_readable(theme: Theme, table: dict, token: str, on: Color, what: str) -> None:
    """``$token`` painted as text on the ``on`` colour reads at 4.5:1."""
    ink = _resolve_color(_painted(table, token), on)
    ratio = _ratio(ink.hex, on.hex)
    assert ratio >= AA, f"{theme.name}: {what} {ink.hex} is {ratio:.2f}:1 on {on.hex} (needs {AA}:1)"


def _workflow_focus_fill(theme: Theme, table: dict, selector: str, pane: Color) -> Color:
    """The fill ``selector`` paints over ``pane``. It is at least as far from
    the pane's fill as the resting $surface fill of a Button or an Input is."""
    fill = _over(pane, Color.parse(_painted(table, _rule_token(WORKFLOWS_SHEET, selector, "background"))))
    rest = _ratio(Color.parse(_painted(table, "surface")).hex, pane.hex)
    shift = _ratio(fill.hex, pane.hex)
    assert shift >= rest, (
        f"{theme.name}: {selector} fill {fill.hex} is {shift:.2f}x from the pane's "
        f"{pane.hex}; the resting fill is {rest:.2f}x from it"
    )
    return fill


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("theme", MEASURABLE_THEMES, ids=lambda t: t.name)
def test_workflow_pane_edges_clear_non_text_contrast(theme: Theme) -> None:
    """TASK-33003.15 AC#3: the Workflows pane edges set no fill and sat on
    $background, where no line colour is pinned. Moving $ds-column-line to the
    boundary colour took litestep from 3.64:1 to 1.32:1 there. Each edge reads
    at 3:1 on the fill its own rule sets.

    Review round 3: the text is gated here too, on every theme. The rule's
    `color` reads at 4.5:1 on that fill. So does the ink of a focused control:
    the house focus tint is translucent, and over $panel it took that ink
    under 4.5:1 on five themes (catppuccin-frappe 4.12:1). Each focused fill
    is measured over the pane's.

    Review round 4: the fill is gated as well as the ink. The search field's
    tint on $background kept its text readable and landed on the rail's own
    luminance (textual-dark 1.00x, under 1.25x on 17 themes). A focused fill
    is at least as far from the pane's as the resting fill is; the field's
    placeholder takes an ink that reads on it; and where that fill shifts the
    pane by less than the house floor, the field's frame reads at 3:1 on it.

    Review round 5: that fill also sits under the selected query and beside
    the caret, which Textual paints in the theme's own $foreground whatever
    the fill is: selected text fell under 4.5:1 on 46 themes (atom-one-dark
    1.35:1) and the caret under 3:1 on 8. Both are the field's ink reversed,
    so each is as far from the fill as typed text is."""
    table = _painted_variables(theme)
    focus_ink = _rule_token(WORKFLOWS_SHEET, ".workflow-compact:focus", "color")
    panes = {}
    for selector in WORKFLOW_PANES:
        edge = _rule_token(WORKFLOWS_SHEET, selector, "border-right")
        fill = _rule_token(WORKFLOWS_SHEET, selector, "background")
        _assert_edge_visible(theme, table, edge, fill)
        pane = panes[selector] = Color.parse(_painted(table, fill))
        ink = _rule_token(WORKFLOWS_SHEET, selector, "color")
        _assert_ink_readable(theme, table, ink, pane, f"{selector} text")
        button = f"{selector} .workflow-compact:focus"
        focused = _workflow_focus_fill(theme, table, button, pane)
        _assert_ink_readable(theme, table, focus_ink, focused, button)
    # Textual's own 5% $foreground focus tint on an Input: left on, typed text
    # is under 4.5:1 on three themes (catppuccin-mocha 4.27:1).
    assert _rule_token(WORKFLOWS_SHEET, WORKFLOW_FIELD_FOCUS, "background-tint") == "transparent"
    library = panes[WORKFLOW_FIELD_FOCUS.split()[0]]
    field = _workflow_focus_fill(theme, table, WORKFLOW_FIELD_FOCUS, library)
    # More than the caret marks the focused field: its fill shifts the pane by
    # the house floor, or its frame reads at 3:1 on that fill (flexoki: 1.23x).
    frame = _resolve_color(_painted(table, _rule_token(WORKFLOWS_SHEET, WORKFLOW_FIELD_FOCUS, "outline")), field)
    shift, edge = _ratio(field.hex, library.hex), _ratio(frame.hex, field.hex)
    assert shift >= FOCUS_SHIFT_FLOOR or edge >= NON_TEXT, (
        f"{theme.name}: {WORKFLOW_FIELD_FOCUS} fill {field.hex} is {shift:.2f}x from the pane's "
        f"{library.hex} and its frame {frame.hex} is {edge:.2f}:1 on it"
    )
    typed = _rule_token(FORMS_SHEET, "Input:focus", "color")
    _assert_ink_readable(theme, table, typed, field, WORKFLOW_FIELD_FOCUS)
    hint = _rule_token(WORKFLOWS_SHEET, WORKFLOW_PLACEHOLDER_FOCUS, "color")
    _assert_ink_readable(theme, table, hint, field, WORKFLOW_PLACEHOLDER_FOCUS)
    for mark in WORKFLOW_MARKS_FOCUS:
        look = [_rule_token(WORKFLOWS_SHEET, mark, prop) for prop in ("background", "text-style")]
        assert look == ["transparent", "reverse"], f"{mark} is not its ink reversed: {look}"
        _assert_ink_readable(theme, table, _rule_token(WORKFLOWS_SHEET, mark, "color"), field, mark)


def _text_cells(app, region, skip=()) -> set[tuple[str, str]]:
    """{(fg hex, bg hex)} of the letters and digits painted in ``region``,
    outside the ``skip`` regions."""
    cells = set()
    for y in range(region.y, region.bottom):
        position = 0
        for segment in app.screen._compositor.render_strips()[y]:
            for glyph in segment.text:
                inside = region.x <= position < region.right and glyph.isalnum()
                if inside and not any(box.contains(position, y) for box in skip):
                    style = segment.style
                    cells.add(
                        (
                            Color.from_rich_color(style.color).hex,
                            Color.from_rich_color(style.bgcolor).hex,
                        )
                    )
                position += 1
    return cells


def _painted_text(app, widget) -> set[tuple[str, str]]:
    """The text cells painted on ``widget``'s fill. Cells inside a descendant
    with an opaque fill of its own (a list, a field, a resting button) are
    skipped: that fill is not the pane's. A translucent one (the focus tint)
    is read, because the pane's fill shows through it."""
    opaque = [child.region for child in widget.walk_children() if child.styles.background.a == 1]
    return _text_cells(app, widget.region, opaque)


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "theme_name",
    [
        # The edge on $background: the two weakest of the six themes under 3:1.
        "litestep",
        "apricot",
        # The focus tint on $panel under a focused control's text: the weakest
        # of the five themes under 4.5:1 (atom-one-dark, below, is another).
        "catppuccin-frappe",
        # Typed text in the focused search field: the lowest (4.54:1), and
        # 4.27:1 with Textual's own focus tint left on the field.
        "catppuccin-mocha",
        # The focus fill nearest its rail (1.23x): the field's frame marks it.
        "flexoki",
        # The pane headings on $panel in the theme's $foreground: under 4.5:1.
        "atom-one-dark",
        "tokyo-night",
        "solarized-dark",
        "solarized-light",
        "pastel_dreams",
        "textual-dark",
        "textual-light",
    ],
)
async def test_workflow_panes_paint_readable_edges_and_text(theme_name: str) -> None:
    """TASK-33003.15 AC#3, rendered under the production stylesheet: the edge
    cells are read from the frame, so the fill measured is the one that paints
    under them (litestep 1.32:1 and apricot 2.63:1 on $background). The panel
    fill that fixes the edge also sits under the pane headings, which
    inherited $foreground: atom-one-dark fell from 6.57:1 to 3.45:1 there, so
    text painted straight on the fill is read as well.

    Review round 3: at rest and with each control focused in turn. The fill
    also sat under the translucent focus tint, and the focused buttons and
    text typed in the search field fell under 4.5:1 on five themes.

    Review round 4: the fills are read too. Each focused control's fill is at
    least as far from the pane's as its resting fill; the search field's sat
    on the rail's own luminance (textual-dark 1.00x, catppuccin-frappe 1.01x).
    The field is read with its placeholder and then with a typed value, and
    its frame is read where its fill shifts the pane by less than the floor.

    Review round 5: and with the value selected, as a keyboard return to the
    field leaves it. The selected text reads at 4.5:1 on a fill that is told
    from the field's, and the caret reads at 3:1 against the field's fill.
    Both were skipped: atom-one-dark painted them at 1.35:1 and 1.79:1."""
    from textual.containers import Horizontal
    from textual.geometry import Region
    from textual.widgets import Button, Input, OptionList

    from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
    from tldw_chatbook.UI.Workflows_Modules.library import WorkflowLibrary
    from tldw_chatbook.UI.Workflows_Modules.navigator import WorkflowNavigator

    class Panes(ConsolidatedCSSApp):
        def compose(self):
            with Horizontal(id="workflow-panes"):
                yield WorkflowLibrary(id="workflows-library")
                yield WorkflowNavigator(id="workflows-navigator")

    app = Panes(css_path=[str(path) for path in APP_STYLESHEETS])
    async with app.run_test(size=(211, 44)) as pilot:
        for theme in ALL_THEMES:
            app.register_theme(theme)
        app.theme = theme_name
        for button in app.query(Button):
            button.disabled = False  # Previous and Next: no page to turn here
        for field in app.query(Input):
            field.cursor_blink = False  # the caret is read below
        app.set_focus(None)  # the search field takes focus on mount
        await pilot.pause(0.3)
        edges, text, fills, marks = {}, {}, {}, {}
        for selector in WORKFLOW_PANES.values():
            widget = app.query_one(selector)
            pane = widget.region
            edges[selector] = {
                cell
                for y in range(pane.y, pane.bottom)
                for cell in _painted_line(app, y, pane.right - 1, pane.right, "│")
            }
            text[selector] = _painted_text(app, widget)
            for control in [child for child in widget.walk_children() if child.focusable]:
                resting = _text_cells(app, control.region)
                app.set_focus(control)
                await pilot.pause()
                frame, focused = [], f"{selector} with #{control.id} focused"
                if isinstance(control, Input):
                    box, (x, y) = control.region, control.content_region.offset
                    frame = _painted_line(app, box.y + 1, box.x, box.x + 1, "│")
                    assert frame, f"no #{control.id} frame"
                    # The placeholder, without its first cell: the caret paints there.
                    own = _text_cells(app, box, [Region(x, y, 1, 1)])
                    assert own, f"no #{control.id} placeholder"
                    carets = _painted_line(app, y, x, x + 1, control.placeholder)
                    # Typed once the focus has landed: focusing selects a value.
                    control.value = "typed 123"
                    await pilot.pause()
                    own |= _text_cells(app, box)
                    # A keyboard return to the field selects the whole value.
                    app.set_focus(None)
                    await pilot.pause()
                    app.set_focus(control)
                    await pilot.pause()
                    end = x + len(control.value)
                    chosen = _painted_line(app, y, x, end, control.value)
                    marks[focused] = (own, carets[:1] + chosen, carets + _painted_line(app, y, end, end + 1, " "))
                else:
                    own = _text_cells(app, control.region)
                # The lists are empty here, and their fill is opaque either way.
                assert own or isinstance(control, OptionList), f"no #{control.id} text"
                text[focused] = own | _painted_text(app, widget)
                fills[focused] = (selector, resting, own, frame)
    for selector, cells in edges.items():
        assert cells, f"{theme_name}: no {selector} edge cells painted"
        for edge, fill in cells:
            ratio = _ratio(edge, fill)
            assert ratio >= NON_TEXT, f"{theme_name}: {selector} edge {edge} on {fill} is {ratio:.2f}:1"
    for selector, cells in text.items():
        assert cells, f"{theme_name}: no {selector} text painted on the pane fill"
        for ink, fill in cells:
            ratio = _ratio(ink, fill)
            assert ratio >= AA, f"{theme_name}: {selector} text {ink} on {fill} is {ratio:.2f}:1"
    for focused, (selector, resting, own, frame) in fills.items():
        (pane_fill,) = {fill for _, fill in text[selector]}
        rest = max((_ratio(fill, pane_fill) for _, fill in resting), default=0)
        for _, fill in own:
            shift = _ratio(fill, pane_fill)
            assert shift >= rest, (
                f"{theme_name}: {focused} fill {fill} is {shift:.2f}x from the pane's "
                f"{pane_fill}; at rest it is {rest:.2f}x from it"
            )
        for edge, fill in frame:
            shift, ratio = _ratio(fill, pane_fill), _ratio(edge, fill)
            assert shift >= FOCUS_SHIFT_FLOOR or ratio >= NON_TEXT, (
                f"{theme_name}: {focused} fill {fill} is {shift:.2f}x from the pane's "
                f"{pane_fill} and its frame {edge} is {ratio:.2f}:1 on it"
            )
    # Every field gate above and below runs only for a focusable Input: with
    # none, the frame, hint, selection and caret would all go unread and pass.
    assert marks, f"{theme_name}: the search field was not read"
    for focused, (own, marked, carets) in marks.items():
        (field,) = {fill for _, fill in own}
        assert len(marked) > 1 and len(carets) == 2, f"{theme_name}: {focused}, no selection or no caret"
        for ink, fill in marked:
            ratio, shift = _ratio(ink, fill), _ratio(fill, field)
            assert ratio >= AA and shift >= FOCUS_SHIFT_FLOOR, (
                f"{theme_name}: {focused}, text under the caret or selected {ink} on {fill} is "
                f"{ratio:.2f}:1, and that fill is {shift:.2f}x from the field's {field}"
            )
        for _, block in carets:
            ratio = _ratio(block, field)
            assert ratio >= NON_TEXT, f"{theme_name}: {focused}, caret {block} is {ratio:.2f}:1 on the field's {field}"


WIDGETS_SHEET = CSS_ROOT / "components/_widgets.tcss"
NAVIGATION_SHEET = CSS_ROOT / "components/_navigation.tcss"
CONSOLE_SHEET = CSS_ROOT / "features/_console.tcss"
_RAIL_HEADER = ".console-rail-section-header"
_TAB = ".nav-button.is-active"

#: TASK-33641.1: line -> (the rule and property that set its colour, the fills
#: it sits on). A fill is a rule whose `background` is read, or a surface name.
CHROME_LINES = {
    "tab-bar separator": (
        (NAV_BAR_CSS, "MainNavigationBar", "border-bottom"),
        [(NAV_BAR_CSS, "MainNavigationBar")],
    ),
    "active tab outline": ((NAVIGATION_SHEET, _TAB, "border"), [(NAVIGATION_SHEET, _TAB)]),
    # The app tier above is what paints in the app; a harness that loads the
    # widget tier alone paints these two, so both tiers name the same outline.
    "active tab outline, widget tier": ((NAV_BAR_CSS, _TAB, "border"), [(NAV_BAR_CSS, _TAB)]),
    "focused active tab outline, widget tier": (
        (NAV_BAR_CSS, _TAB + ":focus", "border"),
        [(NAV_BAR_CSS, _TAB + ":focus")],
    ),
    "Collapsible frame": (
        (WIDGETS_SHEET, "Collapsible", "border"),
        [(WIDGETS_SHEET, "Collapsible")],
    ),
    "Console rail section rule": (
        (CONSOLE_SHEET, _RAIL_HEADER, "border-top"),
        [(CONSOLE_SHEET, _RAIL_HEADER)],
    ),
    # No fill of its own: it sits on the rail.
    "Console conversations rule": (
        (CONSOLE_SHEET, ".console-workspace-conversations-header", "border-top"),
        ["surface", "panel"],
    ),
}


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("line", CHROME_LINES)
@pytest.mark.parametrize("theme", MEASURABLE_THEMES, ids=lambda t: t.name)
def test_app_chrome_lines_clear_non_text_contrast(theme: Theme, line: str) -> None:
    """TASK-33641.1 AC#1: the lines every screen shares read at 3:1 on what
    they sit on. In textual-dark the tab-bar separator measured 1.75:1, the
    active tab's outline 1.29:1, the Collapsible frame 1.21:1 and the Console
    rail's section rules 1.56:1."""
    rule, fills = CHROME_LINES[line]
    table = _painted_variables(theme)
    token = _rule_token(*rule)
    for fill in fills:
        on = fill if isinstance(fill, str) else _rule_token(*fill, "background")
        _assert_edge_visible(theme, table, token, on)


def _painted_line(app, y: int, x0: int, x1: int, glyphs: str) -> list[tuple[str, str]]:
    """[(ink hex, fill hex)] of the cells on row y, columns x0..x1, that draw
    one of ``glyphs``, as they paint: `reverse` swaps a cell's two colours."""
    cells, position = [], 0
    for segment in app.screen._compositor.render_strips()[y]:
        for glyph in segment.text:
            if x0 <= position < x1 and glyph in glyphs:
                style = segment.style
                pair = Color.from_rich_color(style.color).hex, Color.from_rich_color(style.bgcolor).hex
                cells.append(pair[::-1] if style.reverse else pair)
            position += 1
    return cells


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("theme_name", ["textual-dark", "textual-light", "pastel_dreams"])
async def test_app_chrome_lines_paint_at_non_text_contrast(theme_name: str) -> None:
    """TASK-33641.1 AC#1, rendered under the production stylesheet: the gate
    above reads rules, and the rule a sheet names is not always the one that
    paints. Each line's cells are read from the frame and measured."""
    from textual.widgets import Collapsible, Static

    from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
    from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar

    class Chrome(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="home")
            yield Collapsible(Static("body"), title="Group", id="probe-frame")
            yield Static("Section", classes=_RAIL_HEADER[1:], id="probe-rule")

    app = Chrome(css_path=[str(path) for path in APP_STYLESHEETS])
    async with app.run_test(size=(211, 44)) as pilot:
        for theme in ALL_THEMES:
            app.register_theme(theme)
        app.theme = theme_name
        await pilot.pause(0.3)
        bar = app.query_one(MainNavigationBar).region
        tab = app.query_one("#nav-home").region
        frame = app.query_one("#probe-frame").region
        rule = app.query_one("#probe-rule").region

        def outline() -> list[tuple[str, str]]:
            return [
                *_painted_line(app, tab.y, tab.x, tab.right, "╭─╮"),
                *_painted_line(app, tab.y + 1, tab.x, tab.right, "│"),
            ]

        lines = {
            "tab-bar separator": _painted_line(app, bar.bottom - 1, bar.x, bar.right, "─"),
            "Collapsible frame": _painted_line(app, frame.y, frame.x, frame.right, "╭─╮"),
            "Console rail section rule": _painted_line(app, rule.y, rule.x, rule.right, "─"),
        }
        # The shared Button:hover and Button:focus fills once replaced the
        # tab's own, and the outline is pinned against the tab's fill only.
        # The focused active tab keeps its fill and its outline, so the bold
        # underline is the only focus cue: it is read where it paints (the
        # widget tier's own text-style is overridden by this tier).
        active = app.query_one("#nav-home")
        app.set_focus(None)
        await pilot.pause()
        assert not active.rich_style.underline, f"{theme_name}: resting tab is underlined"
        lines["active tab outline"] = outline()
        await pilot.hover("#nav-home")
        await pilot.pause()
        assert active.mouse_hover
        lines["active tab outline, hovered"] = outline()
        app.set_focus(active)
        await pilot.pause()
        assert active.rich_style.underline, f"{theme_name}: hovered focus shows no underline"
        lines["active tab outline, hovered and focused"] = outline()
        await pilot.hover("#probe-rule")
        await pilot.pause()
        assert active.has_focus and not active.mouse_hover
        assert active.rich_style.underline, f"{theme_name}: focus shows no underline"
        lines["active tab outline, focused"] = outline()
    for line, cells in lines.items():
        assert cells, f"{theme_name}: no {line} cells painted"
        for edge, fill in set(cells):
            ratio = _ratio(edge, fill)
            assert ratio >= NON_TEXT, f"{theme_name}: {line} {edge} on {fill} is {ratio:.2f}:1"


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


@pytest.mark.parametrize(
    ("name", "value", "kept"),
    [
        # Textual's own form (nord: input-selection-background '#81a1c1 35%').
        ("block-cursor-blurred-background", "#81a1c1 30%", True),
        ("text", "#ffffff 87%", True),
        ("surface-lighten-2", "#2a2a2a 50%", True),
        # Kept by sanitize_theme_variables, but no colour to measure.
        ("text", "auto 60%", True),
        ("link-style", "italic", True),
        # TASK-33003.16, rewritten on purpose (all three were kept): 'auto NN%'
        # on a name painted as a background is dropped, and so is a text style
        # on a name painted as a colour (AC#4).
        ("block-cursor-blurred-background", "auto 30%", False),
        ("surface-lighten-2", "auto 50%", False),
        ("text", "bold", False),
    ],
)
def test_user_theme_with_any_kept_variable_form_still_loads(
    tmp_path, name, value, kept
) -> None:
    """Task 6 review round 2: the guard read `text`, the house focus tint and
    `surface-lighten-2` with a bare Color.parse, which raises ColorParseError
    (not a ValueError) on the '<colour> NN%' and 'auto NN%' forms
    sanitize_theme_variables keeps. load_user_themes then skipped a theme that
    loaded before, and Settings ▸ Theme ▸ Import raised from an
    Input.Submitted handler. A '<colour> NN%' entry is measured as Textual's
    CSS paints it; an unmeasurable one counts as absent."""
    from tldw_chatbook.css.Themes.themes import load_user_themes

    (tmp_path / "kept.toml").write_text(
        '[theme]\nname = "kept_probe"\ndark = true\n'
        '[colors]\nprimary = "#3366CC"\nbackground = "#101010"\n'
        'surface = "#181818"\npanel = "#202020"\nforeground = "#E0E0E0"\n'
        f'[variables]\n{name} = "{value}"\n',
        encoding="utf-8",
    )
    (theme,) = load_user_themes(tmp_path)
    assert theme.variables.get(name) == (value if kept else None)
    if not (kept and value.startswith("auto")):
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
    " border-right: solid $ds-control-edge; background: $tldw-focus-fill;"
    " border-top: round $tldw-active-edge; }\n"
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


ROLEPLAY_SHEET = CSS_ROOT / "features/_roleplay.tcss"


@pytest.mark.parametrize("theme", MEASURABLE_THEMES, ids=lambda t: t.name)
def test_roleplay_items_list_focus_edge_and_cursor_clear_contrast(
    theme: Theme,
) -> None:
    """Spec 1.5.3 (frame slice B2a): the items list keeps one edge column in
    both focus states, blank at rest, so focus moves no row (D14 census F1);
    the focus edge reads at 3:1 and the focus-only cursor row's text at
    4.5:1 on the list's $surface, on every measurable theme. The spec's
    $ds-action-focus edge measured below 3:1 on 27 of the 89 measurable
    themes (1.06:1 on sweet_sorbet), so the edge is $ds-active-fg (lowest
    4.50:1, sakura_viewing_picnic)."""
    rest = ".roleplay-items-list"
    assert _rule_token(ROLEPLAY_SHEET, rest, "border-left") == "blank"
    focused = f"{rest}:focus"
    edge_token = _rule_token(ROLEPLAY_SHEET, focused, "border-left")
    # OptionList's own focus tint is cleared, so the list paints on $surface.
    assert _rule_token(ROLEPLAY_SHEET, focused, "background-tint") == "transparent"
    cursor = f"{focused} > .option-list--option-highlighted"
    bar_token = _rule_token(ROLEPLAY_SHEET, cursor, "background")
    ink_token = _rule_token(ROLEPLAY_SHEET, cursor, "color")
    table = _painted_variables(theme)
    base = Color.parse(table["surface"])
    edge = _resolve_color(_painted(table, edge_token), base)
    assert _ratio(edge.hex, base.hex) >= NON_TEXT, (theme.name, edge.hex)
    bar = _over(base, Color.parse(_painted(table, bar_token)))
    ink = _resolve_color(_painted(table, ink_token), bar)
    assert _ratio(ink.hex, bar.hex) >= AA, (theme.name, ink.hex, bar.hex)


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_roleplay_items_search_placeholder_and_typed_text_clear_aa_on_every_theme() -> (
    None
):
    """Frame slice B2b AC#13 (RP-062), painted under the production
    stylesheets on every measurable theme: the search box's placeholder at
    rest, the placeholder with the box focused and the text typed into it
    each read at 4.5:1 on the fill they paint on. On dev the placeholder was
    Textual's $text-disabled, under 4.5:1 on all 89 measurable themes (2.12:1
    on paradise_virtua), and focused nothing painted at all (RP-009, a
    focused box with no content row)."""
    from textual.geometry import Region
    from textual.widgets import Input

    from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
    from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
        PersonasLibraryPane,
    )

    class Pane(ConsolidatedCSSApp):
        def compose(self):
            # The workbench's own id and class: the panel fill the focused
            # box's translucent focus fill composites over.
            yield PersonasLibraryPane(
                id="personas-library-pane", classes="destination-workbench-pane"
            )

    app = Pane(css_path=[str(path) for path in APP_STYLESHEETS])
    shortfalls: list[str] = []
    async with app.run_test(size=(80, 24)) as pilot:
        for theme in ALL_THEMES:
            app.register_theme(theme)
        app.query_one(PersonasLibraryPane).styles.width = 35
        search = app.query_one("#personas-library-search", Input)
        search.cursor_blink = False  # the caret is skipped, not read mid-blink
        for theme in MEASURABLE_THEMES:
            app.theme = theme.name
            app.set_focus(None)
            search.value = ""
            await pilot.pause()
            box = search.content_region
            read = {"placeholder at rest": _text_cells(app, box)}
            app.set_focus(search)
            await pilot.pause()
            # The caret paints on the placeholder's first cell.
            read["placeholder focused"] = _text_cells(
                app, box, [Region(box.x, box.y, 1, 1)]
            )
            search.value = "typed 123"
            # The caret after the text, on a blank cell: it paints Textual's
            # own cursor colours, which AC#13 does not cover (a programmatic
            # value can leave it on the first letter).
            search.cursor_position = len(search.value)
            await pilot.pause()
            read["typed text focused"] = _text_cells(app, box)
            for what, cells in read.items():
                if not cells:
                    shortfalls.append(f"{theme.name}: no {what} painted")
                for ink, fill in cells:
                    ratio = _ratio(ink, fill)
                    if ratio < AA:
                        shortfalls.append(
                            f"{theme.name}: {what} {ink} on {fill} is {ratio:.2f}:1"
                        )
    assert len(MEASURABLE_THEMES) >= 89
    assert not shortfalls, f"{len(shortfalls)} shortfalls:\n" + "\n".join(shortfalls)


# TASK-36479 (WCAG 1.4.11): every focus edge -- a border or outline set by a
# `:focus`, `:focus-within` or `.*-focused` rule -- reads at 3:1 against
# $surface, $panel and the fill its own rule paints over either. The edges
# are discovered in every CSS source the app loads (the tcss modules and the
# class-level CSS strings in Python), each resolved in its own source's
# variable scope, so a new focus edge is measured without being listed here,
# whatever token or literal colour it paints. The only edges not measured
# are listed one by one: the dev debt in focus_edge_shortfall_on_dev.tsv
# (TASK-36663) and FOCUS_EDGES_MEASURED_ELSEWHERE.
# On dev, 43 focus edges painted $ds-action-focus ($primary; $accent in
# PromptBlockEditor's own palette), under 3:1 on $surface in 27 of the 89
# measurable themes (sweet_sorbet 1.06:1); they paint $ds-active-fg now, or
# $text-accent in PromptBlockEditor. Not modelled: a fill another rule paints
# under the edge. A Button whose focus rule sets no fill takes the generic
# `Button:focus` $tldw-focus-fill, and $ds-active-fg reads 2.33-2.88:1 on that
# fill in tokyo-night, atom-one-dark, catppuccin-frappe and earthy_nature
# (painted, TASK-36479 notes; tracked by TASK-36663); against the surface
# beside it, 4.5:1 or more.
_FOCUS_STATE = re.compile(r":focus(?:-within)?(?![\w-])|\.[\w-]+-focused(?![\w-])")
_EDGE_PROPERTY = re.compile(r"(?:border|outline)(?:-(?:top|right|bottom|left))?")
_EDGE_TOKEN = re.compile(r"\$([\w-]+)(?:\s+\d{1,3}%)?")
#: An edge declaration that paints nothing.
_NO_EDGE = ("none", "hidden", "blank")
_CLASS_CSS_ATTRS = ("BUNDLED_CSS", "BUNDLED_SCREEN_CSS", "DEFAULT_CSS", "CSS")
_LOCAL_DEFINITION = re.compile(r"^\s*\$([\w-]+)\s*:\s*([^;]+);", re.MULTILINE)

#: Focus edges under 3:1 on dev today that TASK-36479 (the $ds-action-focus
#: edges) did not fix: 152 rows, one per edge -- source, selector, property,
#: painted value -- tracked by TASK-36663. Keyed per edge, so a new focus
#: edge, or a listed one whose value changes, is measured like any other.
#: test_focus_edge_discovery_and_shortfall_list_are_current fails once a row
#: no longer matches an edge or its edge clears 3:1 on every theme, so the
#: list only shrinks: never add a row.
FOCUS_EDGE_DEBT_FILE = Path(__file__).with_name("focus_edge_shortfall_on_dev.tsv")

#: Focus edges another test measures under its own, reviewed contract.
FOCUS_EDGES_MEASURED_ELSEWHERE = {
    ("css/features/_workflows.tcss", WORKFLOW_FIELD_FOCUS, "outline"): (
        "test_workflow_pane_edges_clear_non_text_contrast: its $tldw-focus-fill"
        " shifts the pane by the house floor, or its frame reads at 3:1 on it"
    ),
}

#: One edge per kind of source, so a discovery that finds nothing fails.
FOCUS_EDGE_ANCHORS = (
    ("css/features/_library.tcss", ".library-rail-row:focus"),
    ("css/features/_settings.tcss", ".settings-compact-select:focus"),
    ("css/features/_console.tcss", "#console-native-composer.console-composer-focused"),
    (
        "Widgets/Prompts/prompt_block_editor.py::PromptBlockEditor.DEFAULT_CSS",
        ".prompt-lane:focus-within",
    ),
    (
        "UI/Navigation/main_navigation.py::MainNavigationBar.BUNDLED_CSS",
        ".nav-button:focus",
    ),
    (
        "Widgets/emoji_picker.py::EmojiPickerScreen.BUNDLED_SCREEN_CSS",
        "#search-input:focus",
    ),
)


class FocusEdge(NamedTuple):
    source: str
    selector: str
    prop: str
    #: The colour it paints: `$token`, `$token NN%` or a literal; "" when the
    #: declaration draws an edge in no colour this test can read.
    value: str
    fill: str | None
    #: The source's own `$name: value;` lines; None for a tcss module, which
    #: parses in the bundle's scope (the theme, then _variables.tcss).
    local: dict | None
    #: Whether _variables.tcss reaches the source: the tcss modules, and the
    #: BUNDLED_CSS blocks the build seeds with it. BUNDLED_SCREEN_CSS blocks
    #: are rendered without it (build_css.build_screen_css), DEFAULT_CSS and
    #: CSS parse as sources of their own.
    vocabulary: bool

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.source, self.selector, self.prop, self.value)


def _focus_edge_debt() -> list[tuple[str, ...]]:
    """The rows of FOCUS_EDGE_DEBT_FILE, in file order."""
    lines = FOCUS_EDGE_DEBT_FILE.read_text(encoding="utf-8").splitlines()
    return [
        tuple(line.split("\t")) for line in lines if line and not line.startswith("#")
    ]


def _edge_colour(value: str) -> str | None:
    """The colour an edge declaration paints, "" if none is readable, or
    None when it draws no edge (`none`, `hidden`, `blank`)."""
    if token := _EDGE_TOKEN.search(value):
        return token[0]
    from textual.css.constants import VALID_BORDER

    words = value.split()
    if not words or words[0] in _NO_EDGE:
        return None
    return " ".join(words[1:] if words[0] in VALID_BORDER else words)


def _theme_reaches(theme: Theme, edge: FocusEdge) -> bool:
    """A `.-dark-mode` / `.-light-mode` rule paints only under that polarity."""
    if ".-dark-mode" in edge.selector:
        return theme.dark
    if ".-light-mode" in edge.selector:
        return not theme.dark
    return True


def _css_rules(css: str) -> list[tuple[str, str]]:
    """(selector list, declarations) per rule; a nested rule joins its parents."""
    rules: list[tuple[str, str]] = []
    stack: list[list[str]] = []
    parts = re.split(r"([{}])", re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL))
    for index, part in enumerate(parts):
        if part == "}":
            rules.append(tuple(stack.pop()))
        elif index + 1 < len(parts) and parts[index + 1] == "{":
            body, _, head = part.rpartition(";")
            if stack:
                stack[-1][1] += body
            parents = stack[-1][0].split(",") if stack else [""]
            selector = ",".join(
                s.strip().replace("&", p.strip())
                if "&" in s
                else f"{p.strip()} {s.strip()}".strip()
                for p in parents
                for s in head.split(",")
            )
            stack.append([selector, ""])
        elif part != "{" and stack:
            stack[-1][1] += part
    return rules


def _class_css_sources() -> list[tuple[str, str, str]]:
    """(label, attr, css) per class-level CSS string literal that styles focus."""
    from tldw_chatbook.css import widget_css

    package = CSS_ROOT.parent
    found = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if any(part in widget_css.EXCLUDED_DIRS for part in relative.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if "focus" not in text or not any(attr in text for attr in _CLASS_CSS_ATTRS):
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                # As widget_css.iter_blocks lifts them: `X = "..."` and
                # `X: str = "..."`.
                if isinstance(stmt, ast.Assign):
                    targets = stmt.targets
                elif isinstance(stmt, ast.AnnAssign):
                    targets = [stmt.target]
                else:
                    continue
                if not (
                    isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    continue
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in _CLASS_CSS_ATTRS:
                        label = f"{relative.as_posix()}::{node.name}.{target.id}"
                        found.append((label, target.id, stmt.value.value))
    return found


@functools.lru_cache(maxsize=1)
def _focus_edges() -> tuple[FocusEdge, ...]:
    from tldw_chatbook.css import build_css

    sources = [
        (f"css/{module}", None, (CSS_ROOT / module).read_text(encoding="utf-8"))
        for module in build_css.CSS_MODULES
    ]
    sources += _class_css_sources()
    edges = []
    for label, attr, css in sources:
        local = None
        if attr is not None:
            uncommented = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
            local = {k: v.strip() for k, v in _LOCAL_DEFINITION.findall(uncommented)}
        for selectors, body in _css_rules(css):
            declarations = [d.partition(":")[::2] for d in body.split(";") if ":" in d]
            fills = [v.strip() for p, v in declarations if p.strip() == "background"]
            for prop, value in declarations:
                if not _EDGE_PROPERTY.fullmatch(prop.strip()):
                    continue
                colour = _edge_colour(value)
                if colour is None:
                    continue
                for selector in selectors.split(","):
                    if _FOCUS_STATE.search(selector):
                        edges.append(
                            FocusEdge(
                                label,
                                selector.strip(),
                                prop.strip(),
                                colour,
                                fills[-1] if fills else None,
                                local,
                                attr in (None, "BUNDLED_CSS"),
                            )
                        )
    return tuple(edges)


def _focus_edge_scope(theme: Theme, edge: FocusEdge) -> dict:
    """The variables ``edge``'s source parses with under ``theme``."""
    table = _painted_variables(theme) if edge.vocabulary else _resolved_variables(theme)
    return {**table, **(edge.local or {})}


def _paint(table: dict, value: str) -> str:
    """Follow `$a` / `$a NN%` references to the value that paints."""
    while match := re.fullmatch(r"\$([\w-]+)(\s+\d{1,3}%)?", value.strip()):
        value = table[match[1]].strip() + (match[2] or "")
    return value


def _focus_edge_floor(theme: Theme, edge: FocusEdge) -> tuple[float, str]:
    """The edge's lowest ratio over its grounds, and the ground it was on.

    An edge in its own rule's fill colour extends that fill (the theme
    editor's chips invert into their hue on focus), so it is measured
    against the surfaces only.
    """
    table = _focus_edge_scope(theme, edge)
    lowest = (float("inf"), "")
    for key in ("surface", "panel"):
        surface = Color.parse(table[key])
        grounds = {key: surface}
        if edge.fill and _paint(table, edge.fill) != _paint(table, edge.value):
            grounds[f"{edge.fill} on {key}"] = _resolve_color(
                _paint(table, edge.fill), surface
            )
        for name, ground in grounds.items():
            line = _resolve_color(_paint(table, edge.value), ground)
            lowest = min(lowest, (_ratio(line.hex, ground.hex), f"{name} {ground.hex}"))
    return lowest


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("theme", MEASURABLE_THEMES, ids=lambda t: t.name)
def test_focus_edges_clear_non_text_contrast_on_every_theme(theme: Theme) -> None:
    """TASK-36479 AC#2: every focus edge the app's CSS draws reads at 3:1
    against $surface, $panel and its own rule's fill over either, on every
    measurable theme -- outside the edges listed above one by one (the
    TASK-36663 debt, and the edges another test measures)."""
    debt = set(_focus_edge_debt())
    short = []
    for edge in _focus_edges():
        if edge.key in debt or edge.key[:3] in FOCUS_EDGES_MEASURED_ELSEWHERE:
            continue
        if not _theme_reaches(theme, edge):
            continue
        if not edge.value:
            short.append(
                f"  {edge.source} {edge.selector} {{{edge.prop}}}: no readable colour"
            )
            continue
        ratio, ground = _focus_edge_floor(theme, edge)
        if ratio < NON_TEXT:
            short.append(
                f"  {edge.source} {edge.selector} {{{edge.prop}: {edge.value}}}"
                f" {ratio:.2f}:1 on {ground}"
            )
    assert not short, (
        f"{theme.name}: {len(short)} focus edge(s) under {NON_TEXT}:1\n"
        + "\n".join(short)
    )


def test_focus_edge_discovery_and_shortfall_list_are_current() -> None:
    """TASK-36479: the discovery reaches every kind of CSS source, and every
    row of the TASK-36663 debt list is still an edge that falls under 3:1
    somewhere (fixing one means deleting its row, never re-using it)."""
    edges = _focus_edges()
    found = {(edge.source, edge.selector) for edge in edges}
    missing = [anchor for anchor in FOCUS_EDGE_ANCHORS if anchor not in found]
    missing += [
        listed
        for listed in FOCUS_EDGES_MEASURED_ELSEWHERE
        if listed not in {edge.key[:3] for edge in edges}
    ]
    assert not missing, f"focus-edge discovery missed {missing}"
    rows = _focus_edge_debt()
    assert len(rows) == len(set(rows)), "a debt row is listed twice"
    by_key: dict[tuple[str, ...], list[FocusEdge]] = {}
    for edge in edges:
        by_key.setdefault(edge.key, []).append(edge)
    gone = [row for row in rows if row not in by_key]
    assert not gone, f"debt rows that match no focus edge any more; drop them: {gone}"
    cleared = [
        row
        for row in rows
        if not any(
            _focus_edge_floor(theme, edge)[0] < NON_TEXT
            for edge in by_key[row]
            for theme in MEASURABLE_THEMES
            if _theme_reaches(theme, edge)
        )
    ]
    assert not cleared, f"debt rows that clear {NON_TEXT}:1 now; drop them: {cleared}"


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


def _paints_a_colour(value: str) -> bool:
    """Whether a variable value paints a colour of its own: it parses, it is
    not transparent, and it is not the terminal's default (ansi_default)."""
    suffixed = re.fullmatch(r"(.+) (\d{1,3})%", value)
    try:
        color = Color.parse(suffixed[1] if suffixed else value)
    except Exception:  # noqa: BLE001 - 'auto NN%', a text style: no colour
        return False
    if suffixed:
        color = color.multiply_alpha(int(suffixed[2]) / 100)
    return color.a > 0 and color.ansi != -1


def _assert_guard_colours_visible(theme: Theme) -> None:
    from tldw_chatbook.css.Themes.themes import GUARD_VARIABLES

    resolved = _resolved_variables(theme)
    for name in GUARD_VARIABLES:
        value = resolved[name]
        assert _paints_a_colour(value), f"{theme.name}: ${name} is {value!r}"
        assert value not in (resolved["surface"], resolved["panel"]), (
            f"{theme.name}: ${name} {value!r} is a surface's own colour"
        )


@pytest.mark.parametrize(
    "theme", [*ALL_THEMES, *BUILTIN_THEMES.values()], ids=lambda t: t.name
)
def test_guard_colours_are_visible_on_every_registered_theme(theme: Theme) -> None:
    """Qodo #2937: where the surfaces cannot be measured (the ANSI palettes)
    the guard fell back to their old sources, a transparent edge and an
    ansi_default focus fill -- the terminal's own background, so a focused
    button showed no fill. The measurable themes' 3:1 and focus floors are
    the tests above; here every theme must at least paint a colour."""
    _assert_guard_colours_visible(theme)


@pytest.mark.parametrize(
    ("section", "name", "value"),
    [
        ("variables", "surface", "#181818 50%"),
        ("colors", "surface", "#18181880"),
    ],
)
def test_user_theme_with_unmeasurable_surfaces_keeps_visible_guard_colours(
    tmp_path, section, name, value
) -> None:
    """Qodo #2937: a user theme whose surface or panel has no hex to measure
    lost every generated value, so both guard names became transparent: no
    grid lines, no control edges and no focused-button fill."""
    from tldw_chatbook.css.Themes.themes import load_user_themes

    colors = {
        "primary": "#3366CC",
        "background": "#101010",
        "surface": "#181818",
        "panel": "#202020",
        "foreground": "#E0E0E0",
    }
    variables = {}
    (colors if section == "colors" else variables)[name] = value
    (tmp_path / "probe.toml").write_text(
        '[theme]\nname = "unmeasurable_probe"\ndark = true\n[colors]\n'
        + "".join(f'{key} = "{val}"\n' for key, val in colors.items())
        + "[variables]\n"
        + "".join(f'{key} = "{val}"\n' for key, val in variables.items()),
        encoding="utf-8",
    )
    (theme,) = load_user_themes(tmp_path)
    assert not _measurable(theme)
    _assert_guard_colours_visible(theme)


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(("name", "value"), [("surface", "auto 50%"), ("panel", "auto 10%")])
def test_user_theme_with_auto_on_a_surface_stays_measurable(tmp_path, name, value) -> None:
    """TASK-33003.16, rewritten on purpose: these were two rows of the test
    above, where the kept 'auto NN%' left the surface with nothing to measure
    (and nothing to paint). The sanitiser now drops it, so the surface keeps
    its [colors] value and the guard pins real boundaries."""
    from tldw_chatbook.css.Themes.themes import load_user_themes

    (tmp_path / "probe.toml").write_text(
        '[theme]\nname = "auto_surface_probe"\ndark = true\n'
        '[colors]\nprimary = "#3366CC"\nbackground = "#101010"\n'
        'surface = "#181818"\npanel = "#202020"\nforeground = "#E0E0E0"\n'
        f'[variables]\n{name} = "{value}"\n',
        encoding="utf-8",
    )
    (theme,) = load_user_themes(tmp_path)
    assert name not in theme.variables
    assert _measurable(theme)
    _assert_boundaries_visible(theme)
    _assert_guard_colours_visible(theme)
