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
from textual.widgets._select import SelectCurrent, SelectOverlay

from Tests.UI.full_app_destination_context import (
    active_destination_screen as _active_destination_screen,
    full_app_destination_context as _visual_destination_harness,
    wait_for_selector as _wait_for_selector,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_configuration_hub import (
    StyledSettingsDestinationHarness,
    _open_settings_category,
)
from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RulesPane

pytestmark = pytest.mark.asyncio

# The UAT's own terminal size, so the geometry asserted here is the geometry
# it saw.
UAT_SIZE = (235, 52)


def _watchlists_host():
    app = _build_test_app()
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
            ArticleListPane._FILTER_OPTIONS
        )
        painted = _painted_option_labels(screen, select)
        expected = [label for label, _value in ArticleListPane._FILTER_OPTIONS]
        assert painted == expected, (
            "every status option must reach the screen intact; the overlay "
            f"painted {painted!r}"
        )


async def test_the_status_filter_still_shows_its_value_when_focused_or_hovered():
    """The same defect one level up, on the control rather than its popup.

    Found in live verification with this file already green, which is the
    point of it being here: the overlay tests above pass while the `Select`
    they hang off paints `┌──────────────┐` over its own only row. A compact
    `Select` is ONE row tall and the app-wide focus/hover rules gave it a
    border or an outline anyway (see the TASK-2300 blocks in
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
        assert "All" in blurred, "precondition: the value is readable at rest"

        # Hover FIRST, on a blurred control -- which is the order a user meets
        # them in, and the only order that measures hover at all. Round 2, O1
        # gave the focused state a real background, and focus deliberately
        # outranks hover, so hovering an already-focused Select correctly
        # shows the focus colour and says nothing about the hover rule.
        current, rest_background = await _hover(pilot, screen, select)
        assert "All" in _painted_rows(screen, select.region)[0], (
            "a hovered one-row Select must still say what it is set to"
        )
        assert current.styles.background != rest_background, (
            "the hover cue must actually land -- see `_hover`"
        )

        await pilot.hover("#items-search-input")
        await pilot.pause()
        select.focus()
        await pilot.pause()
        await pilot.pause()
        assert "All" in _painted_rows(screen, select.region)[0], (
            "and so must a focused one"
        )


async def _hover(pilot, screen, select: Select):
    """Put the pointer on `select`'s value row and confirm hover really landed.

    Review wave, I1. The first version of the hover assertions in this file was
    vacuous: `pilot.hover("#items-status-select")` leaves
    `select.mouse_hover == False`, so "the value is still painted" held
    trivially and could not have detected a hover regression.

    Textual gives `:hover` to the innermost widget under the pointer that has a
    hover style (`_has_hover_style`, derived from the selector NAMES a rule
    mentions -- `css/stylesheet.py:506`). For a `Select` that widget is
    `SelectCurrent`, never the `Select`, which is why a rule written as
    `Select:hover > SelectCurrent` can never match: naming the child is what
    makes the child the hover target.

    So this asserts against the widget Textual actually hovers, and returns its
    pre-hover background so the caller can prove the cue changed something.

    Returns:
        `(SelectCurrent, background_before_hover)`.
    """
    current = select.query_one(SelectCurrent)
    rest_background = current.styles.background
    await pilot.hover(f"#{select.id}")
    await pilot.pause()
    assert current.mouse_hover, (
        "the pointer must actually be hovering the widget the cue is written "
        "against, or this test proves nothing"
    )
    return current, rest_background


async def test_items_status_filter_covers_the_reader_set_the_backend_produces():
    """AC#1's other half, TASK-3072: the reader set is the backend's
    vocabulary minus the two statuses a reader deliberately hides.

    Still pinned against `LocalWatchlistsService.ITEM_STATUSES`, so adding a
    status to the backend without deciding whether a reader should see it
    fails here rather than silently making those items unreachable (which is
    TASK-2301's harm). The exclusions are named literally, not computed:
    `ignored` was hidden by the user on purpose, `error` belongs to Runs.
    """
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        LocalWatchlistsService,
    )

    filter_values = {value for _label, value in ArticleListPane._FILTER_OPTIONS}
    assert filter_values == {"unread", "all"}, (
        "the reader's filter is exactly the Unread/All pair"
    )
    assert ArticleListPane._READER_STATUSES == (
        set(LocalWatchlistsService.ITEM_STATUSES) - {"ignored", "error"}
    )


async def test_picking_a_status_filters_the_items_list():
    """AC#1. The filter is wired, not merely populated."""
    host = _watchlists_host()
    async with host.run_test(size=UAT_SIZE) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "items"
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#items-status-select", timeout=5.0)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        pane.items = [
            {"id": "1", "title": "Fresh", "status": "new", "source_name": "F"},
            {"id": "2", "title": "Filed", "status": "ingested", "source_name": "F"},
        ]
        await pilot.pause()
        assert len(pane.displayed_items()) == 2

        select = screen.query_one("#items-status-select", Select)
        select.value = "unread"
        await pilot.pause()
        await pilot.pause()

        displayed = screen.query_one("#watchlists-items-pane", ArticleListPane).displayed_items()
        assert [row["id"] for row in displayed] == ["1"]


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

        rest_rows = _painted_rows(screen, select.region)
        assert "No items" in "".join(rest_rows), (
            "precondition: the value is readable at rest"
        )
        current = select.query_one(SelectCurrent)
        rest_border = current.styles.border

        select.focus()
        await pilot.pause()
        await pilot.pause()
        focused_rows = _painted_rows(screen, select.region)
        assert "No items" in "".join(focused_rows), (
            "a focused Select must still say what it is set to"
        )
        assert len(focused_rows) == len(rest_rows), (
            "focus must not change the control's height"
        )
        assert current.styles.border != rest_border, (
            "focus must still be indicated -- on the border SelectCurrent "
            "already reserves room for, which is Textual's own cue"
        )

        _current, rest_background = await _hover(pilot, screen, select)
        assert "No items" in "".join(_painted_rows(screen, select.region)), (
            "and so must a hovered one"
        )
        assert current.styles.background != rest_background, (
            "the hover cue must actually land -- see `_hover`"
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


async def test_a_bordered_compact_select_keeps_its_frame_under_focus_and_hover():
    """Review wave, Critical 1. "Compact" is not the same as "has no frame".

    The first fix here opted every `Select.-textual-compact` out of `border`
    on focus and hover, on the premise that a compact Select has no perimeter
    to spare. `.settings-compact-select` (components/_agentic_terminal.tcss)
    disproves it: those Selects are compact AND `height: 3` with a `border` of
    their own, sized around -- ~21 of them across Settings, Speech, ImageGen
    and the splash viewer. The blanket opt-out won on specificity ((0,2,1)
    beats (0,2,0)) and stripped a frame the layout was drawn with, so the value
    jumped a row and left two blank ones; hovering strobed it, because removing
    the border moved the pointer's own target out from under it and it
    oscillated every frame.

    Measured on `#settings-provider-value` at 180x50 with the regression in:

        rest      ┌────────────────────────────┐
                  │ Manual / custom provider  ▼ │
                  └────────────────────────────┘
        focused    Manual / custom provider  ▼        <- frame gone

    Nothing in the existing suite saw it (`test_settings_provider_test_draft.py`
    stayed green), so it is pinned here, on the painted rows, in all three
    states.

    task-17664: the original exemplar died by design — 484c74af2 replaced
    the visible provider Select with the search + picker flow and kept
    `#settings-provider-value` only as a hidden manual-entry compat control
    (`settings-provider-manual-hidden`, zero-size), which made this pin
    IndexError on an empty paint. The contract outlives the exemplar
    (~21 bordered compact Selects remain), so it is pinned on the Console
    Behavior compaction-mode Select instead, scrolled into view first the
    way a user reaches it.
    """
    app = _build_test_app()
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = pilot.app.screen
        await _wait_for_selector(
            screen, pilot, "#settings-console-context-compaction-mode", timeout=5.0
        )
        select = screen.query_one("#settings-console-context-compaction-mode", Select)
        assert "-textual-compact" in select.classes, (
            "precondition: this control is the compact-AND-bordered shape"
        )
        select.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.pause()

        def _frame() -> tuple[str, str]:
            rows = _painted_rows(screen, select.region)
            return rows[0].strip(), rows[-1].strip()

        rest_top, rest_bottom = _frame()
        assert rest_top and rest_bottom, (
            "precondition: this Select draws a frame of its own at rest"
        )

        select.focus()
        await pilot.pause()
        await pilot.pause()
        assert _frame() == (rest_top, rest_bottom), (
            "a bordered compact Select must keep its own frame on focus; "
            f"painted {_painted_rows(screen, select.region)!r}"
        )

        await pilot.hover("#settings-console-context-compaction-mode")
        await pilot.pause()
        assert _frame() == (rest_top, rest_bottom), (
            "and on hover -- a frame that appears and disappears under the "
            "pointer is a self-sustaining flicker, not a cue"
        )


def _relative_luminance(color) -> float:
    """WCAG relative luminance of a Rich `Color`."""
    triplet = color.get_truecolor()

    def _channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * _channel(triplet.red)
        + 0.7152 * _channel(triplet.green)
        + 0.0722 * _channel(triplet.blue)
    )


def _contrast(first, second) -> float:
    """WCAG contrast ratio between two rendered background colours."""
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _rendered_background(screen, region, row_offset: int = 0):
    """The background colour the compositor actually painted, one cell in.

    One cell in, not at the edge: on a bordered control the edge cell is the
    frame. Reading the compositor's own segments rather than
    `widget.styles.background` is the same discipline as the rest of this file
    -- a style that is set but painted over by an opaque child is not a cue.
    """
    strips = list(screen.app.screen._compositor.render_strips())
    strip = strips[region.y + row_offset]
    x = 0
    for segment in strip._segments:
        for _character in segment.text:
            if x == region.x + 1:
                return segment.style.bgcolor
            x += 1
    raise AssertionError("no segment covers that cell")


#: The floor a focus cue has to clear to BE a cue. `core/_variables.tcss`
#: (TASK-345) records ~1.1:1 as the measured value of the nullified-focus
#: failure -- "an invisible-focus Tab+Enter activated 'Save as…'" -- and sets
#: `$ds-focus-bg` to a steel blue that measures ~3:1 against the dark control
#: surfaces. 2.0 sits between the two: comfortably above the failure, below
#: the token's own value, so this fails on a regression rather than on a
#: theme tweak.
MIN_FOCUS_CONTRAST = 2.0


@pytest.mark.parametrize(
    "select_id", ["#items-status-select", "#watchlists-backend-select"]
)
async def test_a_borderless_compact_select_has_a_visible_focus_cue(select_id):
    """Round 2, O1. Removing the outline must not leave focus invisible.

    The outline that TASK-2300 took off these controls was destroying their
    only row, so it had to go -- but the replacement was assumed rather than
    measured. `Select:focus`'s recolour paints the SELECT, and `SelectCurrent`
    covers the whole of it with an opaque `background: $surface`, so nothing
    of it reached the screen: all that survived was Textual's 5%
    `background-tint`.

        rest #1e1e1e   focused #272727   ~1.10:1

    which is the exact number `core/_variables.tcss` records as the failure
    that nullified the focus contract once already -- and, with the hover cue
    repaired in the same wave, hover had become STRONGER than focus.

    Asserted on the colour the compositor painted, not on
    `styles.background`, because "set but painted over by a child" is the
    whole defect.
    """
    host = _watchlists_host()
    async with host.run_test(size=UAT_SIZE) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = (
            "items" if select_id == "#items-status-select" else "sources"
        )
        await pilot.pause()
        await _wait_for_selector(screen, pilot, select_id, timeout=5.0)
        select = screen.query_one(select_id, Select)
        assert not select.disabled, "focus contrast must be measured on a focusable control"

        rest = _rendered_background(screen, select.region)
        select.focus()
        await pilot.pause()
        await pilot.pause()
        focused = _rendered_background(screen, select.region)

        ratio = _contrast(rest, focused)
        assert ratio >= MIN_FOCUS_CONTRAST, (
            f"focus must be visible on {select_id}: {rest} -> {focused} is "
            f"{ratio:.2f}:1, below the {MIN_FOCUS_CONTRAST}:1 floor"
        )
        # And it must still be readable while focused.
        assert _painted_rows(screen, select.region)[0].strip(), (
            "the focused control must still paint its value"
        )


async def test_focus_is_at_least_as_loud_as_hover():
    """Round 2, O1's other half: the two cues must not be inverted.

    Hover says "you could interact with this"; focus says "you ARE interacting
    with this, and Enter will act here". A hover cue louder than the focus cue
    is worse than a missing one -- it points at the wrong control.
    """
    host = _watchlists_host()
    async with host.run_test(size=UAT_SIZE) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = "items"
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#items-status-select", timeout=5.0)
        select = screen.query_one("#items-status-select", Select)

        rest = _rendered_background(screen, select.region)

        await _hover(pilot, screen, select)
        hovered = _rendered_background(screen, select.region)

        await pilot.hover("#items-search-input")
        await pilot.pause()
        select.focus()
        await pilot.pause()
        await pilot.pause()
        focused = _rendered_background(screen, select.region)

        assert _contrast(rest, focused) >= _contrast(rest, hovered), (
            f"focus ({focused}) must not be quieter than hover ({hovered})"
        )
