"""TASK-25831: Console radios convey state through colour alone.

The wizard already fixed this class -- SetupRadioButton exists because "stock
ToggleButton renders one constant BUTTON_INNER glyph and conveys on/off purely
through the glyph's color, which is invisible in a monochrome capture and fails
WCAG 1.4.1". The Console Library-access modal still used the stock widget, so
both of its options rendered the same filled glyph and differed only by dot
colour -- measured 1.42:1 for the off state, i.e. invisible in any text capture.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult


class _Harness(App):
    def compose(self) -> ComposeResult:
        from tldw_chatbook.Widgets.Console.console_library_access_modal import (
            ConsoleAccessRadioButton,
        )

        yield ConsoleAccessRadioButton("On", value=True, id="on")
        yield ConsoleAccessRadioButton("Off", value=False, id="off")


@pytest.mark.asyncio
async def test_selected_and_unselected_use_different_glyphs() -> None:
    from tldw_chatbook.Widgets.Console.console_library_access_modal import (
        ConsoleAccessRadioButton,
    )

    app = _Harness()
    async with app.run_test(size=(40, 6)):
        on = app.query_one("#on", ConsoleAccessRadioButton)
        off = app.query_one("#off", ConsoleAccessRadioButton)
        on_glyph = str(on._button)
        off_glyph = str(off._button)

    assert on_glyph != off_glyph, (
        "selected and unselected must differ by glyph, not colour alone; "
        f"both rendered {on_glyph!r}"
    )
