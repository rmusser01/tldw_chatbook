"""Console modals take their frame and copy colours from the active theme.

TASK-33626.2. Seventeen Console modals hard-coded ``border: tall gray;
background: black`` in their Python CSS. In a light theme that is a black box
in a light UI, and copy with no colour of its own inherits the light theme's
dark foreground: the Library search description and the Edit Message
consequence line measured #1f1f1f on #000000, 1.27:1, live on textual-light.
Focused rows were primary-at-30% composited over the black frame, so Rewind
and the composer Menu showed dark-navy slabs (#001423) in an otherwise light
screen.

Two layers of evidence, both of which fail on the pre-fix code:

* a source ratchet over every class-level CSS string in
  ``Widgets/Console`` (named colour keywords, hex, ``rgb()``/``hsl()``),
  because that is where the literal came from and no existing gate read it
  (the hex ratchet in test_design_token_governance.py reads ``.tcss`` only);
* rendered checks that push the real modal classes onto an App loading the
  production stylesheets (``APP_STYLESHEETS`` plus the consolidated widget
  sheets) under the real registered themes, and read the colours the
  compositor paints. CSS source says nothing about what an inherited colour
  lands on, which is the whole defect.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from rich.cells import cell_len
from textual.color import COLOR_NAME_TO_RGB, Color
from textual.widget import Widget
from textual.widgets import Button

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.css.Themes.themes import ALL_THEMES, _contrast_ratio
from tldw_chatbook.Widgets.Console.console_auto_speak_consent import (
    AutoSpeakConsentModal,
)
from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    ConsoleComposerMenuModal,
)
from tldw_chatbook.Widgets.Console.console_edit_message_modal import (
    ConsoleEditMessageModal,
)
from tldw_chatbook.Widgets.Console.console_library_search_modal import (
    ConsoleLibrarySearchModal,
)
from tldw_chatbook.Widgets.Console.console_rag_settings_modal import (
    ConsoleRagSettingsModal,
)
from tldw_chatbook.Widgets.Console.console_rewind_modal import (
    ConsoleRewindModal,
    RewindPromptRow,
)

#: The modal modules import guarded config readers at collection time; the
#: per-test sandbox redirect would then fail admission at setup with
#: RecoveryRequired("raw_source_selection_changed") (Tests/conftest.py).
pytestmark = pytest.mark.bootstrap_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSOLE_WIDGETS = REPO_ROOT / "tldw_chatbook" / "Widgets" / "Console"

# ---------------------------------------------------------------------------
# AC#4: the source ratchet
# ---------------------------------------------------------------------------

#: Class attributes Textual (or build_css.py) reads as CSS.
_CSS_ATTRIBUTES = frozenset({"DEFAULT_CSS", "BUNDLED_CSS", "BUNDLED_SCREEN_CSS", "CSS"})
#: Properties that can carry a colour. A colour name in any other property
#: (`align: center middle`) is not a colour, so only these are read.
_COLOUR_PROPERTY = re.compile(
    r"color|background|border|outline|tint|scrollbar|link|hatch|keyline", re.I
)
_DECLARATION = re.compile(r"([a-z-]+)\s*:\s*([^;{}]+)")
_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_COLOUR_FUNCTION = re.compile(r"\b(?:rgba?|hsla?)\(")
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
#: `transparent` paints nothing, so it cannot clash with a theme.
_COLOUR_NAMES = frozenset(COLOR_NAME_TO_RGB) - {"transparent"}

#: Documented exceptions: ``(file name, class name, "property: value")`` ->
#: why a literal is right there. Empty on purpose. A literal colour in a
#: Console widget paints the same in every theme, which is the defect this
#: file exists to stop; an entry needs a reason a reviewer can check (for
#: example, a colour that is content, like a user-picked swatch).
ALLOWED_LITERAL_COLOURS: dict[tuple[str, str, str], str] = {}


def literal_colours_in_css(css: str) -> list[tuple[str, str]]:
    """Find colour declarations that use a literal instead of a theme variable.

    Args:
        css: One class-level CSS string.

    Returns:
        ``(property, value)`` pairs whose value names a colour keyword, a hex
        literal, or an ``rgb()``/``hsl()`` function.
    """
    found: list[tuple[str, str]] = []
    for match in _DECLARATION.finditer(_COMMENT.sub("", css)):
        prop, value = match.group(1), " ".join(match.group(2).split())
        if not _COLOUR_PROPERTY.search(prop):
            continue
        words = re.findall(r"\$?[A-Za-z_][\w-]*", value)
        keyword = any(
            not word.startswith("$") and word.lower() in _COLOUR_NAMES
            for word in words
        )
        if keyword or _HEX.search(value) or _COLOUR_FUNCTION.search(value):
            found.append((prop, value))
    return found


def console_widget_literal_colours(root: Path = CONSOLE_WIDGETS):
    """Every literal colour declaration in a Console widget's class CSS.

    Args:
        root: Directory to scan (recursively) for Python sources.

    Returns:
        ``(file name, class name, "property: value")`` triples, sorted.
    """
    hits: set[tuple[str, str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if not (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id in _CSS_ATTRIBUTES
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    continue
                for prop, value in literal_colours_in_css(statement.value.value):
                    hits.add((path.name, node.name, f"{prop}: {value}"))
    return sorted(hits)


def test_detector_flags_keywords_hex_and_functions_but_not_theme_variables() -> None:
    """The ratchet's own detector, on known input (a detector that finds
    nothing would pass the real scan below vacuously)."""
    css = """
    #a { border: tall gray; background: black; color: $text; }
    #b { color: #1f1f1f; background: rgb(0, 0, 0); outline: solid $primary; }
    #c { align: center middle; border: none; background: transparent; }
    /* #d { background: black; } is a comment */
    #e { background-tint: $foreground 5%; scrollbar-color: Red; }
    """
    assert literal_colours_in_css(css) == [
        ("border", "tall gray"),
        ("background", "black"),
        ("color", "#1f1f1f"),
        ("background", "rgb(0, 0, 0)"),
        ("scrollbar-color", "Red"),
    ]


def test_console_widget_css_uses_theme_colours_only() -> None:
    """AC#4: no named colour or hex literal in Console widget CSS outside the
    documented allow-list. Pre-fix this listed 52 declarations in 18 files."""
    unexpected = [
        hit
        for hit in console_widget_literal_colours()
        if hit not in ALLOWED_LITERAL_COLOURS
    ]
    assert unexpected == [], "\n".join(" | ".join(hit) for hit in unexpected)


def test_allow_list_entries_are_still_needed() -> None:
    """A stale allow-list entry would silently re-admit the literal later."""
    present = set(console_widget_literal_colours())
    stale = [key for key in ALLOWED_LITERAL_COLOURS if key not in present]
    assert stale == []


# ---------------------------------------------------------------------------
# AC#1, #2, #3, #5: rendered colours under real themes
# ---------------------------------------------------------------------------

#: AC#2's theme set. high_contrast_yellow_black is the shipped high-contrast
#: theme; the three light themes are AC#1's.
THEMES = (
    "textual-dark",
    "textual-light",
    "paper_light",
    "solarized_light",
    "high_contrast_yellow_black",
)
LIGHT_THEMES = frozenset({"textual-light", "paper_light", "solarized_light"})
#: AC#5: the review's two sizes (MacBook full screen, smallest supported).
SIZES = ((235, 52), (80, 24))
#: Literal ids, so the PR-gate census can name one case (scripts/check_ui_pr_gate_census.py).
SIZE_IDS = ("235x52", "80x24")
AA_TEXT = 4.5
#: task-31284's house floor for a visible fill shift (Tests/UI/
#: test_theme_contrast.py FOCUS_SHIFT_FLOOR): the focused row against the
#: rows beside it.
VISIBLE_SHIFT = 1.25
BLACK = Color(0, 0, 0)


class _ModalThemeApp(ConsolidatedCSSApp[None]):
    """The production stylesheets, with every shipped theme registered."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def on_mount(self) -> None:
        for theme in ALL_THEMES:
            if theme.name not in self.available_themes:
                self.register_theme(theme)


def _painted_cell(screen, x: int, y: int) -> tuple[str, Color, Color]:
    """(glyph, ink, fill) the compositor paints at screen cell (x, y)."""
    position = 0
    for segment in screen._compositor.render_strips()[y]:
        width = cell_len(segment.text)
        if position + width > x:
            style = segment.style
            fg = Color.from_rich_color(style.color) if style and style.color else None
            bg = (
                Color.from_rich_color(style.bgcolor)
                if style and style.bgcolor
                else None
            )
            if style is not None and style.reverse:
                fg, bg = bg, fg
            glyph = segment.text[x - position] if x - position < len(segment.text) else " "
            return glyph, fg, bg
        position += width
    raise AssertionError(f"({x}, {y}) is off screen")


class _OffScreen(Exception):
    """The widget's content lies wholly outside the terminal."""


def _first_ink(screen, widget: Widget) -> tuple[int, int, Color, Color]:
    """The first painted glyph inside ``widget``'s on-screen content region."""
    region = widget.content_region.intersection(screen.region)
    if not region.area:
        raise _OffScreen(repr(widget))
    for y in range(region.y, region.bottom):
        for x in range(region.x, region.right):
            glyph, ink, fill = _painted_cell(screen, x, y)
            if glyph.strip():
                return x, y, ink, fill
    raise AssertionError(f"{widget!r} painted no glyph in {region}")


def _theme_colour(app, name: str) -> Color:
    return Color.parse(app.get_css_variables()[name])


async def _settle(pilot) -> None:
    await pilot.pause()
    await pilot.pause()


def _frame_report(app, screen, frame: Widget, label: str, failures: list[str]) -> str:
    """AC#1: the frame paints the theme's panel and boundary, never black."""
    panel = _theme_colour(app, "panel")
    boundary = _theme_colour(app, "tldw-boundary")
    # The padding cell just inside the left border: frame fill, no ink.
    _, _, fill = _painted_cell(screen, frame.region.x + 1, frame.region.y + 1)
    edge_glyph, edge_ink, edge_fill = _painted_cell(
        screen, frame.region.x, frame.region.y + 1
    )
    edge_colours = {c.rgb for c in (edge_ink, edge_fill) if c is not None}
    if fill is None or fill.rgb != panel.rgb:
        failures.append(f"{label}: frame fill {fill} is not the theme panel {panel.hex}")
    if fill is not None and app.current_theme.name in LIGHT_THEMES and fill.rgb == BLACK.rgb:
        failures.append(f"{label}: black frame in a light theme")
    # The outer half of a `tall` edge cell shows the backdrop, which IS black
    # in a black-canvas theme (high_contrast_yellow_black); only a light
    # theme can tell a black edge from its backdrop.
    light = app.current_theme.name in LIGHT_THEMES
    if boundary.rgb not in edge_colours or (light and BLACK.rgb in edge_colours):
        failures.append(
            f"{label}: frame edge {edge_glyph!r} {sorted(edge_colours)} is not the "
            f"theme boundary {boundary.hex}"
        )
    return f"frame {fill.hex if fill else None} edge {boundary.hex}"


def _copy_report(screen, widget: Widget, label: str, failures: list[str]) -> str:
    """AC#2: explanation/consequence copy reads at AA on what it sits on.

    Copy the terminal cannot show at all is reported, not measured: that is a
    geometry defect (the modal outgrows 24 rows), not a colour one.
    """
    try:
        _, _, ink, fill = _first_ink(screen, widget)
    except _OffScreen:
        return "copy off-screen"
    ratio = _contrast_ratio(ink, fill)
    if ratio < AA_TEXT:
        failures.append(
            f"{label}: copy {ink.hex} on {fill.hex} is {ratio:.2f}:1 (needs {AA_TEXT}:1)"
        )
    return f"copy {ink.hex}/{fill.hex} {ratio:.2f}:1"


def _row_report(
    app, screen, focused: Button, other: Button, label: str, failures: list[str]
) -> str:
    """AC#3: the focused row is readable, distinct, and not a dark slab."""
    x, y, ink, bar = _first_ink(screen, focused)
    _, _, _, rest = _first_ink(screen, other)
    ink_ratio = _contrast_ratio(ink, bar)
    shift = _contrast_ratio(bar, rest)
    panel = _theme_colour(app, "panel")
    if ink_ratio < AA_TEXT:
        failures.append(f"{label}: focused row text {ink_ratio:.2f}:1")
    if shift < VISIBLE_SHIFT:
        failures.append(f"{label}: focused row shifts only {shift:.2f}x from its neighbour")
    if app.current_theme.name in LIGHT_THEMES and _contrast_ratio(bar, panel) > _contrast_ratio(
        ink, panel
    ):
        # A light frame whose highlight sits further from the frame than the
        # row's own text does is the dark-slab-in-a-light-box the review saw.
        failures.append(f"{label}: focused row fill {bar.hex} is a dark slab on {panel.hex}")
    return f"row {ink.hex}/{bar.hex} {ink_ratio:.2f}:1 shift {shift:.2f}x vs {rest.hex}"


def _rewind_prompts() -> tuple[RewindPromptRow, ...]:
    return tuple(
        RewindPromptRow(message_id=f"m{index}", index_label=f"#{index}", preview=text)
        for index, text in (
            (3, "Reply with exactly: hi there"),
            (2, "Summarise the open file"),
            (1, "What changed since yesterday?"),
        )
    )


async def _measure_all(app, pilot, failures: list[str]) -> list[str]:
    """Push each modal the review named, measure it, pop it."""
    lines: list[str] = []
    theme = app.current_theme.name
    size = f"{app.size.width}x{app.size.height}"

    async def show(modal):
        await app.push_screen(modal)
        await _settle(pilot)
        return app.screen

    async def close() -> None:
        app.pop_screen()
        await _settle(pilot)

    # Edit Message: the consequence line (live: 1.27:1 on textual-light).
    screen = await show(
        ConsoleEditMessageModal(content="Reply with exactly: hi there", can_resend=True)
    )
    label = f"{theme} {size} edit-message"
    lines.append(
        f"{label}: "
        + _frame_report(app, screen, screen.query_one("#console-edit-message-modal"), label, failures)
        + "; "
        + _copy_report(screen, screen.query_one("#console-edit-message-context"), label, failures)
    )
    await close()

    # Library search: inherits ConsoleRagSettingsModal's frame.
    screen = await show(ConsoleLibrarySearchModal(item_scope_summary="Scope: everything"))
    label = f"{theme} {size} library-search"
    lines.append(
        f"{label}: "
        + _frame_report(app, screen, screen.query_one("#console-rag-settings"), label, failures)
        + "; "
        + _copy_report(screen, screen.query_one(".console-rag-settings-status"), label, failures)
    )
    await close()

    # RAG settings, the frame Library search inherits.
    screen = await show(ConsoleRagSettingsModal(item_scope_summary="Scope: everything"))
    label = f"{theme} {size} rag-settings"
    lines.append(
        f"{label}: "
        + _frame_report(app, screen, screen.query_one("#console-rag-settings"), label, failures)
        + "; "
        + _copy_report(screen, screen.query_one(".console-rag-settings-status"), label, failures)
    )
    await close()

    # AutoSpeak consent: every line is consent copy.
    screen = await show(
        AutoSpeakConsentModal("OpenAI", "https://api.openai.com", True)
    )
    label = f"{theme} {size} auto-speak-consent"
    parts = [
        _frame_report(app, screen, screen.query_one("#console-auto-speak-consent-modal"), label, failures)
    ]
    for selector in (
        "#console-auto-speak-consent-provider",
        "#console-auto-speak-consent-destination",
        "#console-auto-speak-consent-charges",
    ):
        parts.append(_copy_report(screen, screen.query_one(selector), f"{label} {selector}", failures))
    lines.append(f"{label}: " + "; ".join(parts))
    await close()

    # Composer Menu: focused row, disabled-row reason, and the hint.
    screen = await show(ConsoleComposerMenuModal(attachment_kind="none"))
    label = f"{theme} {size} composer-menu"
    items = [
        button
        for button in screen.query(".console-composer-menu-item").results(Button)
        if not button.disabled
    ]
    items[0].focus()
    await _settle(pilot)
    parts = [
        _frame_report(app, screen, screen.query_one("#console-composer-menu"), label, failures),
        _row_report(app, screen, items[0], items[1], label, failures),
        _copy_report(screen, screen.query_one(".console-composer-menu-reason"), f"{label} reason", failures),
        _copy_report(screen, screen.query_one(".console-composer-menu-hint"), f"{label} hint", failures),
    ]
    lines.append(f"{label}: " + "; ".join(parts))
    await close()

    # Rewind: focused row, then the action summary its selection reveals.
    screen = await show(ConsoleRewindModal(prompts=_rewind_prompts()))
    label = f"{theme} {size} rewind"
    rows = list(screen.query(".console-rewind-row").results(Button))
    rows[0].focus()
    await _settle(pilot)
    parts = [
        _frame_report(app, screen, screen.query_one("#console-rewind-modal"), label, failures),
        _row_report(app, screen, rows[0], rows[1], label, failures),
    ]
    await pilot.press("enter")
    await _settle(pilot)
    summary = list(screen.query(".console-rewind-summary-copy"))
    if summary:
        parts.append(_copy_report(screen, summary[0], f"{label} summary", failures))
    else:
        failures.append(f"{label}: selecting a row showed no summary copy")
    lines.append(f"{label}: " + "; ".join(parts))
    await close()
    return lines


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("theme", THEMES)
async def test_console_modals_paint_theme_frames_and_readable_copy(theme, size) -> None:
    """AC#1-#3 and #5, measured from the compositor under each theme."""
    app = _ModalThemeApp()
    async with app.run_test(size=size) as pilot:
        app.theme = theme
        await _settle(pilot)
        assert app.current_theme.name == theme
        failures: list[str] = []
        lines = await _measure_all(app, pilot, failures)
    print("\n" + "\n".join(lines))
    assert failures == [], "\n".join(failures)
