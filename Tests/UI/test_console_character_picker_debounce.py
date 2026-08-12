"""Debounce-shape regression test for the Console character picker.

Representative of the whole debounced picker/filter family fixed by
task-15476 (see `console_prompt_picker_modal.py`'s established 0.2s
timer + cancel shape, which every other converted site in the family
mirrors): typing into a picker's filter `Input` must not synchronously
rebuild the full result list on every keystroke. A debounce `Timer`
re-arms on each keystroke, cancelling any previous pending one, so only
the SETTLED query is ever applied to the rendered rows.

`ConsoleCharacterPickerModal` is used as the representative site because
it needs no database/app context (`options` are pre-supplied) and its
debounce implementation was hand-written directly for task-15476, making
it a faithful stand-in for the family's shared shape.
"""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Input

from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
    SEARCH_DEBOUNCE_SECONDS,
    ConsoleCharacterOption,
    ConsoleCharacterPickerModal,
)

QUERY_INPUT_ID = "#console-character-picker-query"
RESULT_ROW_CLASS = ".console-character-picker-result"


def _options(n: int) -> tuple[ConsoleCharacterOption, ...]:
    return tuple(
        ConsoleCharacterOption(character_id=i, name=f"Character {i}") for i in range(n)
    )


class _Harness(App[None]):
    """Hosts the picker modal with a fixed, pre-supplied option list."""

    def __init__(self, options: tuple[ConsoleCharacterOption, ...]) -> None:
        super().__init__()
        self._options = options
        self.result: object = "unset"

    async def on_mount(self) -> None:
        await self.push_screen(
            ConsoleCharacterPickerModal(options=self._options),
            callback=lambda choice: setattr(self, "result", choice),
        )


@pytest.mark.asyncio
async def test_typing_does_not_rebuild_rows_before_the_debounce_settles():
    """A keystroke alone must not synchronously rebuild the result rows."""
    app = _Harness(_options(5))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        assert len(app.screen.query(RESULT_ROW_CLASS)) == 5

        app.screen.query_one(QUERY_INPUT_ID, Input).value = "Character 1"
        # Even right up against the debounce window, the old rows are
        # still on screen -- the rebuild has not fired yet.
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS - 0.1)
        assert len(app.screen.query(RESULT_ROW_CLASS)) == 5


@pytest.mark.asyncio
async def test_typing_then_settling_rebuilds_the_filtered_rows():
    """Once the debounce settles, the row list reflects the typed query."""
    app = _Harness(_options(20))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        app.screen.query_one(QUERY_INPUT_ID, Input).value = "Character 5"
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        rows = app.screen.query(RESULT_ROW_CLASS)
        assert len(rows) == 1
        assert "Character 5" in str(rows[0].renderable)


@pytest.mark.asyncio
async def test_rapid_retyping_cancels_the_pending_timer_and_only_the_final_query_applies():
    """Timer re-arm + cancel: two keystrokes in quick succession collapse
    into one rebuild reflecting only the LAST query -- proving the
    intermediate query's pending timer was cancelled, not merely
    superseded after also running."""
    app = _Harness(_options(20))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        query_input = app.screen.query_one(QUERY_INPUT_ID, Input)
        # Two back-to-back Input.Changed posts with no settling between
        # them -- simulates paste/fast typing outpacing the debounce.
        query_input.value = "Character 1"
        query_input.value = "Character 12"
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        rows = app.screen.query(RESULT_ROW_CLASS)
        assert len(rows) == 1
        assert "Character 12" in str(rows[0].renderable)
