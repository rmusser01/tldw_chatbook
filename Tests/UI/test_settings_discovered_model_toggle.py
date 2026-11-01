"""TASK-368: the discovered-model SelectionList checkbox glyph must render
selected vs unselected distinctly.

The theme previously lumped the toggle component classes into the row-cursor
rule (all $surface/$text), erasing the checked/unchecked distinction so the
``X`` read as checked in both states. This loads the REAL built CSS bundle plus
the app theme (the pilot harness does not, so a component-style regression is
otherwise invisible to tests) and asserts the selected toggle differs from the
unselected one.
"""

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import SelectionList
from textual.widgets.selection_list import Selection

import tldw_chatbook
from tldw_chatbook.css.Themes.themes import agentic_terminal_theme

_BUNDLE = Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"


class _ToggleApp(App):
    CSS_PATH = str(_BUNDLE)

    def compose(self) -> ComposeResult:
        yield SelectionList[str](
            Selection("selected-model", "a", True),
            Selection("unselected-model", "b", False),
            id="settings-discovered-models-list",
        )

    def on_mount(self) -> None:
        self.register_theme(agentic_terminal_theme)
        self.theme = "agentic_terminal"


@pytest.mark.asyncio
async def test_selection_list_toggle_distinguishes_selected_from_unselected():
    app = _ToggleApp()
    async with app.run_test(size=(80, 8)) as pilot:
        await pilot.pause()
        selection_list = app.query_one(SelectionList)
        selected = selection_list.get_component_rich_style(
            "selection-list--button-selected"
        )
        unselected = selection_list.get_component_rich_style("selection-list--button")

        # The core regression: a selected checkbox must NOT render like an
        # unselected one. Selected is a bright, framed glyph; unselected is an
        # empty box, so both the inner colour and the button background differ.
        assert selected.color != unselected.color
        assert selected.bgcolor != unselected.bgcolor


def _button_cells(app, listing: SelectionList, row: int) -> list[tuple[str, str, str]]:
    """[(glyph, fg hex, bg hex)] of the ▐X▌ button the frame paints on an option row."""
    from textual.color import Color

    region = listing.content_region
    cells, position = [], 0
    for segment in app.screen._compositor.render_strips()[region.y + row]:
        for glyph in segment.text:
            if region.x <= position < region.x + 3:
                style = segment.style
                cells.append(
                    (
                        glyph,
                        Color.from_rich_color(style.color).hex,
                        Color.from_rich_color(style.bgcolor).hex,
                    )
                )
            position += 1
    return cells


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("theme", ["agentic_terminal", "textual-dark", "textual-light"])
@pytest.mark.parametrize("in_card", [True, False], ids=["discovered-models", "any-list"])
async def test_unchecked_row_under_the_cursor_shows_no_checked_glyph(theme, in_card):
    """TASK-33003.17: `selection-list--button-highlighted` is the UNCHECKED
    box under the cursor, and the shared selected-row contract painted its X
    like a checked one. Rendered under the production stylesheet (bundle and
    split sheets), with the cursor on a checked row and then on an unchecked
    one:

    - the discovered-models list (inside the Providers & Models card, so the
      card's highlight bar applies) paints no X on the unchecked row, and a
      3:1 X on the checked row;
    - any other SelectionList paints the unchecked row under the cursor as
      it paints an unchecked row at rest.
    """
    from textual.containers import Vertical

    from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
    from tldw_chatbook.css.Themes.themes import _contrast_ratio

    class Listing(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            options = (
                Selection("checked-model", "a", True),
                Selection("unchecked-model", "b", False),
                Selection("resting-model", "c", False),
            )
            if in_card:
                with Vertical(id="settings-providers-models-card"):
                    yield SelectionList[str](*options, id="settings-discovered-models-list")
            else:
                yield SelectionList[str](*options)

    app = Listing(css_path=[str(path) for path in APP_STYLESHEETS])
    async with app.run_test(size=(80, 12)) as pilot:
        app.register_theme(agentic_terminal_theme)
        app.theme = theme
        listing = app.query_one(SelectionList)
        listing.focus()
        painted = {}
        for row, name in ((0, "checked"), (1, "unchecked")):
            listing.highlighted = row
            await pilot.pause()
            painted[name] = _button_cells(app, listing, row)
        resting = _button_cells(app, listing, 2)

    def ratio(first: str, second: str) -> float:
        from textual.color import Color

        return _contrast_ratio(Color.parse(first), Color.parse(second))

    assert [cell[0] for cell in painted["checked"]] == ["▐", "X", "▌"]
    _, checked_ink, checked_box = painted["checked"][1]
    _, unchecked_ink, unchecked_box = painted["unchecked"][1]
    assert ratio(checked_ink, checked_box) >= 3.0, (theme, painted["checked"])
    if in_card:
        assert unchecked_ink == unchecked_box, (theme, painted["unchecked"])
    else:
        assert painted["unchecked"][1] == resting[1], (theme, painted["unchecked"], resting)
        assert ratio(checked_ink, unchecked_ink) >= 3.0, (theme, checked_ink, unchecked_ink)
