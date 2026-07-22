"""Mounted tests for the TagFilterPicker modal (P3a Task 3)."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, ListView

from tldw_chatbook.Widgets.Persona_Widgets.tag_filter_picker import TagFilterPicker

pytestmark = pytest.mark.asyncio


class _PickerHost(App):
    def __init__(self, tags, current):
        super().__init__()
        self._tags = tags
        self._current = current
        self.result = "unset"

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        # push_screen_wait requires an active worker (NoActiveWorker otherwise);
        # mirrors the pattern used by the sibling ConversationAttachPicker test
        # (Tests/UI/test_conversation_attach_picker.py).
        self.run_worker(self._drive)

    async def _drive(self) -> None:
        self.result = await self.push_screen_wait(
            TagFilterPicker(self._tags, self._current)
        )


def _select(listing: ListView, index: int) -> None:
    """Highlight ``index`` then select it, mirroring an Enter keypress."""
    listing.index = index
    listing.action_select_cursor()


async def test_all_row_clears_filter():
    app = _PickerHost(["hero", "villain"], current="villain")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        listing = app.screen.query_one("#tag-filter-list", ListView)
        _select(listing, 0)  # "All (clear filter)"
        await pilot.pause()
    assert app.result is None


async def test_tag_row_returns_exact_tag_despite_ambiguous_prefix():
    # "hero" is a text-prefix of "hero mage" - the recovered value must come
    # from the stored row-index -> tag mapping, not from re-parsing/matching
    # the rendered label, or picking one could resolve to the other.
    tags = ["hero", "hero mage"]

    app = _PickerHost(tags, current=None)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        listing = app.screen.query_one("#tag-filter-list", ListView)
        _select(listing, 1)  # row 0 = All, row 1 = "hero"
        await pilot.pause()
    assert app.result == "hero"

    app2 = _PickerHost(tags, current=None)
    async with app2.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        listing = app2.screen.query_one("#tag-filter-list", ListView)
        _select(listing, 2)  # row 2 = "hero mage"
        await pilot.pause()
    assert app2.result == "hero mage"


async def test_escape_cancels_distinct_from_clear_filter():
    app = _PickerHost(["hero"], current=None)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is TagFilterPicker.CANCEL
    assert app.result is not None  # cancel must never be mistaken for clear-filter


async def test_search_narrows_rows_then_selects_correct_tag():
    tags = ["alpha", "beta", "gamma"]
    app = _PickerHost(tags, current=None)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen.query_one("#tag-filter-search", Input).value = "beta"
        await pilot.pause()
        listing = screen.query_one("#tag-filter-list", ListView)
        # "All" + the single matching "beta" row - proves the filter rebuilt
        # the row set (unfiltered would be 4 rows: All + alpha/beta/gamma).
        assert len(listing) == 2
        _select(listing, 1)
        await pilot.pause()
    assert app.result == "beta"
