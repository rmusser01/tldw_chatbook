"""Library list rows: keyboard focus must be visible against selection -- TASK-31983.

Critique #6 P1. `.library-media-row:focus` and `.library-media-row-selected`
(and the sibling conversation/notes/notes-folder/prompt canvases) shared the
IDENTICAL treatment in `_agentic_terminal.tcss`::

    background: $ds-focus-bg; color: $ds-focus-fg; text-style: bold underline;

so the row the keyboard was ON looked exactly like the open/selected row --
arrow-key navigation read as a dead key. The fix gives ``:focus`` a
non-colour-only cue -- a ``border-left: thick $ds-action-focus`` left-edge bar
that ``-selected`` does NOT carry -- so the two states are distinguishable even
when a row is BOTH focused and selected (the common case: arrow onto the open
row). The left border replaces the row's 1-column left padding, so the label
never shifts or is clipped (no ``outline: heavy`` regression).

Painted-frame reads go through the real compositor
(`Screen._compositor.render_strips()`) against the PRODUCTION bundle + library
split sheet, the same pattern as `test_compact_focus_outline_render.py` -- the
`:focus`/`-selected` rules live in `screen_agentic_library.tcss`, so a
bundle-only harness would see neither.
"""

from __future__ import annotations

import re

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button

from Tests.UI.consolidated_css import APP_STYLESHEETS, app_css_text

_APP_CSS_PATHS = [str(path) for path in APP_STYLESHEETS]

# `border-left: thick $ds-action-focus` paints a full-block left edge.
_THICK_LEFT_GLYPH = "█"  # '█'


class _LibraryRowsHost(App):
    """Three real media rows under the production stylesheet: one open.

    Rows are `Button`s classed `library-media-row` exactly as
    `library_media_canvas.py` mounts them; the open item additionally carries
    `library-media-row-selected` the way `button.set_class(...)` toggles it.
    `AUTO_FOCUS = None` keeps every row genuinely blurred until a test moves
    focus, so the observed cue is the one a real user's Tab/arrow produces.
    """

    AUTO_FOCUS = None
    CSS_PATH = _APP_CSS_PATHS

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Button("Media Alpha", id="row-0", classes="library-media-row")
            yield Button(
                "Media Bravo",
                id="row-1",
                classes="library-media-row library-media-row-selected",
            )
            yield Button("Media Charlie", id="row-2", classes="library-media-row")


def _strip_rows(app: App) -> list[str]:
    """Painted frame as one plain-text string per screen row."""
    strips = app.screen._compositor.render_strips()
    return ["".join(segment.text for segment in strip) for strip in strips]


def _leftmost_char(rows: list[str], widget) -> str:
    """The single painted cell at the widget's top-left (its border column)."""
    region = widget.region
    line = rows[region.y]
    return line[region.x] if region.x < len(line) else ""


@pytest.mark.asyncio
async def test_focused_row_border_distinct_from_selected_background() -> None:
    """AC#1 (computed). A focused row carries a left-edge border a selected
    row does not, and both share the same focus background -- so the cue that
    separates them is structural, never colour-alone."""
    app = _LibraryRowsHost()
    async with app.run_test(size=(235, 52)) as pilot:
        focused = app.query_one("#row-0", Button)
        selected = app.query_one("#row-1", Button)
        focused.focus()
        await pilot.pause()
        assert focused.has_focus and not selected.has_focus

        assert focused.styles.border_left[0], (
            "focused row must carry a left-edge border cue; got "
            f"{focused.styles.border_left!r}"
        )
        assert not selected.styles.border_left[0], (
            "selected (blurred) row must NOT carry the focus border; got "
            f"{selected.styles.border_left!r}"
        )
        # Same background proves the distinction is the border, not colour.
        assert focused.styles.background == selected.styles.background


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30)])
async def test_focus_cue_painted_and_moves_at_both_sizes(size) -> None:
    """AC#1/#2/#3 (painted). The focused row paints a left-edge glyph the
    blurred/selected rows do not, its label survives intact, and the glyph
    follows focus onto the next row (including onto the open row -- the
    focused+selected case) with no selection change."""
    app = _LibraryRowsHost()
    async with app.run_test(size=size) as pilot:
        row0 = app.query_one("#row-0", Button)
        row1 = app.query_one("#row-1", Button)
        row0.focus()
        await pilot.pause()

        rows = _strip_rows(app)
        assert _leftmost_char(rows, row0) == _THICK_LEFT_GLYPH, (
            f"focused row's left edge glyph missing at size {size}"
        )
        assert _leftmost_char(rows, row1) != _THICK_LEFT_GLYPH, (
            "blurred selected row must not paint the focus glyph"
        )
        # AC#3: the compact 2-cell row keeps its whole label.
        painted_row0 = rows[row0.region.y] + rows[row0.region.y + 1]
        assert "Media Alpha" in painted_row0, (
            f"focused row's label was clipped at size {size}: {painted_row0!r}"
        )

        # AC#2: move focus onto the open row -- cue follows, selection unchanged.
        await pilot.press("tab")
        await pilot.pause()
        assert row1.has_focus and row1.has_class("library-media-row-selected")

        rows = _strip_rows(app)
        assert _leftmost_char(rows, row1) == _THICK_LEFT_GLYPH, (
            "focus cue did not move onto the next (open) row"
        )
        assert _leftmost_char(rows, row0) != _THICK_LEFT_GLYPH, (
            "focus cue lingered on the row focus left"
        )
        painted_row1 = rows[row1.region.y] + rows[row1.region.y + 1]
        assert "Media Bravo" in painted_row1, (
            "focused+selected row's label was clipped"
        )


def _rule_body(text: str, selector: str) -> str:
    """First rule body whose comma-joined selector list contains ``selector``."""
    uncommented = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    for match in re.finditer(r"\{(?P<body>[^{}]*)\}", uncommented, flags=re.DOTALL):
        prefix = uncommented[: match.start()]
        start = max(prefix.rfind("}"), prefix.rfind(";")) + 1
        selectors = [item.strip() for item in prefix[start : match.start()].split(",")]
        if selector in selectors:
            return match.group("body")
    raise AssertionError(f"selector not found: {selector!r}")


@pytest.mark.parametrize(
    "focus_selector",
    [
        ".library-media-row:focus",
        ".library-conversation-row:focus",
        ".library-notes-row:focus",
        ".library-notes-folder-row:focus",
        ".library-prompt-row:focus",
    ],
)
def test_every_sibling_focus_rule_carries_the_border_cue(focus_selector) -> None:
    """AC#4. The border-left focus cue is uniform across all row canvases,
    and none of the ``-selected`` rules steal it (they stay background-only)."""
    css = app_css_text()
    assert "border-left" in _rule_body(css, focus_selector), (
        f"{focus_selector} is missing the border-left focus cue"
    )
    for selected in (
        ".library-media-row-selected",
        ".library-conversation-row-selected",
    ):
        assert "border-left" not in _rule_body(css, selected), (
            f"{selected} must NOT carry the focus border cue"
        )
