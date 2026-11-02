"""The tag picker on a real host app (frame slice B2b, spec 1.5.1; RP-095).

The picker is a combobox: focus stays in the box while Down and Up move the
list's highlight, the first matching tag is highlighted (never "All (clear
filter)"), and Enter picks the highlighted row after flushing a pending
filter. On dev Enter and Down did nothing in the box, and Enter after Tab
picked "All", clearing the filter the user was narrowing to (Survey 3 5.2).
No PersonasScreen mounts here, so this file runs in the UI Fast Lane.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, ListView, Static

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from tldw_chatbook.Widgets.Persona_Widgets import tag_filter_picker as picker_module
from tldw_chatbook.Widgets.Persona_Widgets.tag_filter_picker import TagFilterPicker

pytestmark = pytest.mark.asyncio

TAGS = ["fantasy", "sci-fi", "noir", "fanfic"]
#: Nothing has been picked yet (``None`` is a pick: "All").
NOT_YET = object()


class _Host(App):
    """Pushes the picker the way the Roleplay screen does and keeps its answer."""

    def __init__(self, tags: list[str], current: str | None = None) -> None:
        super().__init__()
        self._tags = tags
        self._current = current
        self.picked: object = NOT_YET

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(TagFilterPicker(self._tags, self._current), self._keep)

    def _keep(self, result: object) -> None:
        self.picked = result


async def _open(pilot) -> tuple[TagFilterPicker, Input, ListView]:
    await pilot.pause()
    picker = pilot.app.screen
    assert isinstance(picker, TagFilterPicker)
    search = picker.query_one("#tag-filter-search", Input)
    assert picker.focused is search
    return picker, search, picker.query_one("#tag-filter-list", ListView)


async def _settled(pilot) -> None:
    """Past the filter's debounce, with the rebuilt list on screen."""
    await pilot.pause(picker_module.SEARCH_DEBOUNCE_SECONDS + 0.1)
    await pilot.pause()


def _rows(listing: ListView) -> list[str]:
    return [str(item.query_one(Static).render()) for item in listing.children]


def _highlighted(picker: TagFilterPicker, listing: ListView) -> str | None:
    """The row Enter picks, which must also be the one painted highlighted."""
    rows = _rows(listing)
    painted = [
        row for row, item in zip(rows, listing.children) if item.has_class("-highlight")
    ]
    picked = None if listing.index is None else rows[listing.index]
    assert painted == ([] if picked is None else [picked]), (painted, picked)
    return picked


async def test_enter_in_the_box_picks_the_only_match():
    app = _Host(TAGS)
    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press(*"noir")
        await _settled(pilot)
        await pilot.press("enter")
        await pilot.pause()
        assert app.picked == "noir"


async def test_enter_picks_the_first_matching_tag_never_all():
    """The trap Survey 3 measured: after a filter the highlight sat on "All
    (clear filter)", so picking the highlighted row cleared the filter."""
    app = _Host(TAGS)
    async with app.run_test() as pilot:
        picker, _search, listing = await _open(pilot)
        await pilot.press(*"fan")
        await _settled(pilot)
        assert _rows(listing) == ["All (clear filter)", "fantasy", "fanfic"]
        assert _highlighted(picker, listing) == "fantasy"
        await pilot.press("enter")
        await pilot.pause()
        assert app.picked == "fantasy"


async def test_down_and_up_move_the_highlight_while_typing_stays_in_the_box():
    app = _Host(TAGS)
    async with app.run_test() as pilot:
        picker, search, listing = await _open(pilot)
        await pilot.press(*"f")
        await _settled(pilot)
        assert _highlighted(picker, listing) == "fantasy"
        await pilot.press("down")
        assert _highlighted(picker, listing) == "sci-fi"
        assert picker.focused is search
        await pilot.press("up", "up")
        assert _highlighted(picker, listing) == "All (clear filter)"
        await pilot.press("down")
        await pilot.press(*"an")  # typing still goes to the box
        assert search.value == "fan"
        await _settled(pilot)
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.picked == "fanfic"


async def test_enter_inside_the_debounce_picks_from_what_was_typed(monkeypatch):
    """Enter faster than the 0.2 s filter flushes it first, as TASK-33910.35
    did for the items search: the pick is from "noi", never the stale list."""
    # A debounce that never fires by itself, so only Enter's flush can apply
    # "noi" however slowly the keys arrive.
    monkeypatch.setattr(picker_module, "SEARCH_DEBOUNCE_SECONDS", 60.0)
    app = _Host(TAGS)
    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press(*"noi", "enter")
        await pilot.pause()
        assert app.picked == "noir"


@pytest.mark.parametrize("current", [None, "noir"], ids=["all", "noir"])
async def test_enter_with_an_empty_box_keeps_the_current_choice(current):
    """An empty box highlights the active choice ("All" when no tag is on), so
    Enter answers with it: the filter is unchanged."""
    app = _Host(TAGS, current)
    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("enter")
        await pilot.pause()
        assert app.picked == current


async def test_enter_with_no_match_picks_nothing_and_keeps_the_picker_open():
    app = _Host(TAGS, "noir")
    async with app.run_test() as pilot:
        picker, _search, listing = await _open(pilot)
        await pilot.press(*"zzz")
        await _settled(pilot)
        assert _rows(listing) == ["All (clear filter)"]
        assert listing.index is None
        await pilot.press("enter")
        await pilot.pause()
        assert app.picked is NOT_YET
        assert app.screen is picker


async def test_down_and_up_do_nothing_while_the_box_matches_no_tag():
    """Deferred minor T11a: with a query no tag matches, the list holds only
    "All (clear filter)" with nothing highlighted, and Textual's cursor_down
    went from no highlight to row 0, so Enter then cleared the filter the
    user was narrowing to (new with B2b's combobox). Down and Up now do
    nothing there; Enter still picks nothing."""
    app = _Host(TAGS, "noir")
    async with app.run_test() as pilot:
        picker, search, listing = await _open(pilot)
        await pilot.press(*"zzz")
        await _settled(pilot)
        assert _rows(listing) == ["All (clear filter)"]
        assert _highlighted(picker, listing) is None
        await pilot.press("down")
        assert _highlighted(picker, listing) is None
        await pilot.press("up")
        assert _highlighted(picker, listing) is None
        assert picker.focused is search
        await pilot.press("enter")
        await pilot.pause()
        assert app.picked is NOT_YET and app.screen is picker
        # Deleting back to a query with matches highlights its first match.
        await pilot.press("backspace", "backspace", "backspace", *"fan")
        await _settled(pilot)
        assert _highlighted(picker, listing) == "fantasy"


async def test_up_at_the_first_row_and_down_at_the_last_row_stay_put():
    """Deferred minor T11a pins the ends: Up on the first row ("All") and
    Down on the last row keep that row highlighted (Textual's ListView
    clamps); typing still goes to the box."""
    app = _Host(TAGS)
    async with app.run_test() as pilot:
        picker, search, listing = await _open(pilot)
        assert _highlighted(picker, listing) == "All (clear filter)"
        await pilot.press("up")
        assert _highlighted(picker, listing) == "All (clear filter)"
        for _ in range(len(TAGS) + 2):
            await pilot.press("down")
        assert _highlighted(picker, listing) == TAGS[-1]
        await pilot.press("down")
        assert _highlighted(picker, listing) == TAGS[-1]
        assert picker.focused is search
        await pilot.press("enter")
        await pilot.pause()
        assert app.picked == TAGS[-1]


async def test_a_row_picked_in_the_list_returns_its_exact_tag():
    """Rows map back by index, never by their painted label (a tag that looks
    like markup is returned as typed)."""
    app = _Host(["[b]x", "hero"])
    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press("tab", "down", "enter")
        await pilot.pause()
        assert app.picked == "[b]x"


async def test_escape_cancels():
    """Escape is CANCEL, distinct from None ("All", clear the filter)."""
    app = _Host(TAGS, "noir")
    async with app.run_test() as pilot:
        await _open(pilot)
        await pilot.press(*"no", "escape")
        await pilot.pause()
        assert app.picked is TagFilterPicker.CANCEL
        assert TagFilterPicker.CANCEL is not None
