"""TASK-2300: the Watchlists Selects must actually show their options.

The 2026-08-04 UAT (live tmux, fresh profile, 235x52) reported the Items
tab's status filter opening "an empty floating overlay -- a bare border,
nothing selectable", and the New Rule form's condition Select looking like an
empty list.

Everything here runs against the **production stylesheet in the full shell**
(`full_app_destination_context`), and every assertion reads the pixels the
compositor actually painted rather than the widget's own bookkeeping. That is
load-bearing, not fastidiousness: through the entire defect
`SelectOverlay.option_count` was **6**, the right six `Option` objects were in
the list, and `Select.value` was correct. The options were destroyed on their
way to the screen -- by the app-wide `*:focus { outline: solid ... }` fallback
painting over a compact overlay's outermost rendered lines, which reserve no
geometry because `OptionList.-textual-compact` removes the border. A test
asserting `option_count == 6` is green on a screen showing nothing.

See the TASK-2300 block in `css/components/_lists.tcss` for the mechanism and
the measured before/after.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Select
from textual.widgets._select import SelectOverlay

from Tests.UI.full_app_destination_context import (
    StaticWatchlistsScopeService,
    active_destination_screen as _active_destination_screen,
    full_app_destination_context as _visual_destination_harness,
    wait_for_selector as _wait_for_selector,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RulesPane

pytestmark = pytest.mark.asyncio

# The UAT's own terminal size, so the geometry asserted here is the geometry
# it saw.
UAT_SIZE = (235, 52)


def _watchlists_host():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    return _visual_destination_harness(app, "watchlists_collections")


def _painted_rows(screen, region) -> list[str]:
    """The characters the compositor painted inside `region`, row by row.

    The whole point of this suite: what a widget *holds* and what the
    terminal *shows* are different questions, and TASK-2300 is a defect that
    only exists in the gap between them.
    """
    compositor = screen.app.screen._compositor
    lines = [
        "".join(segment.text for segment in strip._segments)
        for strip in compositor.render_strips()
    ]
    return [
        lines[y][region.x : region.x + region.width]
        for y in range(region.y, region.y + region.height)
    ]


#: Box-drawing and block glyphs a `SelectOverlay` can legitimately paint on
#: its own perimeter. A NON-compact overlay keeps Textual's `border: tall`,
#: which the box model reserves, so its outermost cells are supposed to be
#: chrome; a COMPACT one reserves nothing, so any chrome there has eaten an
#: option. Stripping these lets one assertion cover both shapes: chrome-only
#: rows collapse to "" and are dropped, while an option whose first character
#: was overwritten survives as a MANGLED label and still fails the compare.
_CHROME = " ▊▎▔▁│┌┐└┘─━┃╭╮╰╯"


def _painted_option_labels(screen, select: Select) -> list[str]:
    """Every option label as the terminal renders it, in overlay order."""
    overlay = select.query_one(SelectOverlay)
    rows = [row.strip(_CHROME) for row in _painted_rows(screen, overlay.region)]
    return [row for row in rows if row]


async def test_items_status_filter_paints_every_status_option():
    """AC#1. The filter's overlay shows every status, intact.

    `option_count` is asserted too, but only as a control: it was already
    correct while the screen showed a bare box, so on its own it proves
    nothing.
    """
    host = _watchlists_host()
    async with host.run_test(size=UAT_SIZE) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "items"
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#items-status-select", timeout=5.0)

        select = screen.query_one("#items-status-select", Select)
        select.expanded = True
        await pilot.pause()
        await pilot.pause()

        assert select.query_one(SelectOverlay).option_count == len(
            ItemsPane._STATUS_OPTIONS
        )
        painted = _painted_option_labels(screen, select)
        expected = [label for label, _value in ItemsPane._STATUS_OPTIONS]
        assert painted == expected, (
            "every status option must reach the screen intact; the overlay "
            f"painted {painted!r}"
        )


async def test_the_status_filter_still_shows_its_value_when_focused_or_hovered():
    """The same defect one level up, on the control rather than its popup.

    Found in live verification with this file already green, which is the
    point of it being here: the overlay tests above pass while the `Select`
    they hang off paints `┌──────────────┐` over its own only row. A compact
    `Select` is ONE row tall and three app-wide rules give it a border or an
    outline on focus and on hover (see the TASK-2300 blocks in
    `css/components/_lists.tcss`). Hover matters as much as focus: it fires on
    the way TO clicking, before anything has been chosen.
    """
    host = _watchlists_host()
    async with host.run_test(size=UAT_SIZE) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "items"
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#items-status-select", timeout=5.0)
        select = screen.query_one("#items-status-select", Select)

        blurred = _painted_rows(screen, select.region)[0]
        assert "All statuses" in blurred, "precondition: the value is readable at rest"

        select.focus()
        await pilot.pause()
        await pilot.pause()
        assert "All statuses" in _painted_rows(screen, select.region)[0], (
            "a focused one-row Select must still say what it is set to"
        )

        await pilot.hover("#items-status-select")
        await pilot.pause()
        assert "All statuses" in _painted_rows(screen, select.region)[0], (
            "and so must a hovered one"
        )


async def test_items_status_filter_covers_every_status_the_backend_produces():
    """AC#1's other half: the vocabulary is the backend's, not a subset.

    Pinned against `LocalWatchlistsService.ITEM_STATUSES` so adding a status
    to the backend without adding it to the filter fails here rather than
    silently making those items unreachable (which is TASK-2301's harm).
    """
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        LocalWatchlistsService,
    )

    filter_values = {value for _label, value in ItemsPane._STATUS_OPTIONS}
    assert "all" in filter_values, "the filter must offer an unfiltered view"
    assert filter_values - {"all"} == set(LocalWatchlistsService.ITEM_STATUSES)


async def test_picking_a_status_filters_the_items_list():
    """AC#1. The filter is wired, not merely populated."""
    host = _watchlists_host()
    async with host.run_test(size=UAT_SIZE) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "items"
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#items-status-select", timeout=5.0)

        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
        pane.items = [
            {"id": "1", "title": "Fresh", "status": "new", "source_name": "F"},
            {"id": "2", "title": "Filed", "status": "ingested", "source_name": "F"},
        ]
        await pilot.pause()
        assert len(pane.displayed_items()) == 2

        select = screen.query_one("#items-status-select", Select)
        select.value = "ingested"
        await pilot.pause()
        await pilot.pause()

        displayed = screen.query_one("#watchlists-items-pane", ItemsPane).displayed_items()
        assert [row["id"] for row in displayed] == ["2"]


async def test_new_rule_condition_select_paints_the_real_vocabulary():
    """AC#2. The condition Select offers every condition, intact.

    Its closed state legitimately reads "No items" -- that is the first
    condition's own name, not an empty-list message -- so the assertion that
    matters is what the OPEN overlay paints.
    """
    host = _watchlists_host()
    async with host.run_test(size=UAT_SIZE) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "rules"
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#rules-new-button", timeout=5.0)

        screen.query_one("#rules-new-button", Button).press()
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#rules-create-condition", timeout=5.0)

        select = screen.query_one("#rules-create-condition", Select)
        select.expanded = True
        await pilot.pause()
        await pilot.pause()

        painted = _painted_option_labels(screen, select)
        expected = [label for label, _value in RulesPane._CONDITION_OPTIONS]
        assert painted == expected, (
            "every alert condition must reach the screen intact; the overlay "
            f"painted {painted!r}"
        )


async def test_the_rule_condition_select_still_shows_its_value_when_focused():
    """The non-compact half of the same defect, on the other reported Select.

    A three-row `Select` draws its border on its child `SelectCurrent`, so an
    app rule putting one on the `Select` ADDS two rows of chrome to a widget
    whose three rows are already spoken for -- and the value goes. Clicking
    the control is what focuses it, so this was its state the instant a user
    reached for it. See the TASK-2300 block in components/_forms.tcss.
    """
    host = _watchlists_host()
    async with host.run_test(size=UAT_SIZE) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "rules"
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#rules-new-button", timeout=5.0)
        screen.query_one("#rules-new-button", Button).press()
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#rules-create-condition", timeout=5.0)
        select = screen.query_one("#rules-create-condition", Select)

        painted = "".join(_painted_rows(screen, select.region))
        assert "No items" in painted, "precondition: the value is readable at rest"

        select.focus()
        await pilot.pause()
        await pilot.pause()
        assert "No items" in "".join(_painted_rows(screen, select.region)), (
            "a focused Select must still say what it is set to"
        )

        await pilot.hover("#rules-create-condition")
        await pilot.pause()
        assert "No items" in "".join(_painted_rows(screen, select.region)), (
            "and so must a hovered one"
        )


async def test_a_two_option_select_overlay_is_not_painted_away_entirely():
    """The worst case of the same mechanism, pinned separately.

    A compact overlay reserves no perimeter, so an outline over its outermost
    rendered lines costs two options. At six options that mangles the list; at
    two it leaves a bordered box with NOTHING in it -- which is what the UAT
    described. `#watchlists-backend-select` is the two-option case on this
    screen, so it is the one that goes to zero first.
    """
    host = _watchlists_host()
    async with host.run_test(size=UAT_SIZE) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(
            screen, pilot, "#watchlists-backend-select", timeout=5.0
        )

        select = screen.query_one("#watchlists-backend-select", Select)
        select.expanded = True
        await pilot.pause()
        await pilot.pause()

        painted = _painted_option_labels(screen, select)
        assert [row for row in painted if row], (
            "a two-option Select must paint its options, not an empty box"
        )
        assert len(painted) == select.query_one(SelectOverlay).option_count
