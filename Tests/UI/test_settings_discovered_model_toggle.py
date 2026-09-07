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
