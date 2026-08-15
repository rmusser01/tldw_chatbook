# test_consolidated_css_harness.py
# Description: Pins Tests/UI/consolidated_css.py's own CSS_PATH-merge behavior
# (TASK-15995).
#
# `ConsolidatedCSSApp.CSS_PATH` carries the two generated screen/modal sheets
# (TASK-15450) so a harness that pushes one of the seven `BUNDLED_SCREEN_CSS`
# modals gets its class-level CSS. But it is an ordinary class attribute, and
# ~27 real test harnesses declare their own `CSS_PATH` (most to also load the
# app bundle) -- which, absent a merge, shadows it wholesale via normal Python
# attribute lookup and drops the screen sheets. Textual's `App.__init__` also
# accepts a `css_path=` constructor kwarg that short-circuits the
# `self.CSS_PATH` class-attribute branch entirely (`css_path or
# self.CSS_PATH`), so the merge has to intercept both forms. This module pins
# both rather than relying on any of the 27 real harnesses, since none of them
# currently happens to push a `BUNDLED_SCREEN_CSS` modal (a vacuous-pass trap
# noted in the task).

from __future__ import annotations

import pytest
from textual.color import Color

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog import (
    NoteSelectionDialog,
)


class _StyledHarness(ConsolidatedCSSApp):
    """Mirrors the real combiners: a subclass that declares its own CSS_PATH."""

    CSS_PATH = str(BUNDLED_STYLESHEET)


@pytest.mark.asyncio
async def test_css_path_class_attr_override_still_loads_screen_sheets():
    """A subclass ``CSS_PATH`` class attribute must not drop the screen sheets.

    `NoteSelectionDialog` (one of the seven `BUNDLED_SCREEN_CSS` modals) gives
    `#note-selection-container` a fixed `width: 80` only via its screen sheet
    entry; absent that sheet the container falls back to `Container`'s own
    `width: 1fr` default and fills the whole content area instead. This is a
    computed-geometry consequence of the CSS actually applying, not merely a
    check that pushing the modal raised no exception.
    """
    app = _StyledHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(NoteSelectionDialog([]))
        await pilot.pause()

        container = app.screen.query_one("#note-selection-container")
        assert container.region.width == 80, (
            "NoteSelectionDialog's BUNDLED_SCREEN_CSS width:80 rule for "
            "#note-selection-container did not apply (region.width="
            f"{container.region.width}) -- a subclass CSS_PATH class "
            "attribute override dropped ConsolidatedCSSApp's screen sheets"
        )


@pytest.mark.asyncio
async def test_css_path_kwarg_override_loads_screen_sheets_and_keeps_own_entry(
    tmp_path,
):
    """A ``css_path=`` constructor kwarg override must survive the merge too.

    Textual's ``App.__init__`` resolves ``css_path or self.CSS_PATH`` -- a
    kwarg short-circuits the class-attribute branch entirely, so the merge
    has to intercept it independently of a subclass's ``CSS_PATH`` attribute.
    None of the real harnesses use this form today, but the mechanism must
    still compose with it. Asserts both halves: the harness's own supplied
    stylesheet still applies, and the screen sheets are still merged in
    alongside it.
    """
    custom_css = tmp_path / "harness_only.tcss"
    custom_css.write_text("#note-search-input { background: red; }\n", encoding="utf-8")

    app = ConsolidatedCSSApp(css_path=str(custom_css))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(NoteSelectionDialog([]))
        await pilot.pause()

        search_input = app.screen.query_one("#note-search-input")
        assert search_input.styles.background == Color.parse("red"), (
            "the harness's own css_path= entry did not apply -- the merge "
            "must not replace it, only bracket it with the screen sheets"
        )

        container = app.screen.query_one("#note-selection-container")
        assert container.region.width == 80, (
            "a css_path= constructor kwarg override dropped the screen "
            "sheets (region.width=" + str(container.region.width) + ")"
        )
