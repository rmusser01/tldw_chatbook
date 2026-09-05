"""Every Inspect-rail Tab stop must SHOW that it has focus, in plain text.

TASK-31663. The 2026-09-05 critique measured all five of the rail's focus
treatments at 1.03-1.79:1 against their own unfocused state and found five
button stops with no plain-text change at all. Re-measured on this branch
through the real compositor (`Screen._compositor.render_strips()`), with
focus parked on the composer between every stop so a scroll-into-view
cannot be mistaken for a focus cue:

    80x24 and 200x50, indication-free:
        console-inspector-rail-collapse
        console-project-instruction-status-button
        console-inspector-section-environment-toggle
        console-inspector-section-environment-view-all
        console-run-library-rag
        console-retrieval-scope-narrow
        console-inspector-rail-body  (80x24 only)

TASK-24702 already ruled out the obvious repair: on this near-black theme a
background tint cannot reach the 3:1 non-text floor at any alpha short of
opaque, so the carrier has to change CELLS, not colours. The house answer is
the accent edge (`outline-left: thick $ds-action-focus`), which paints a
solid block glyph. Two physics facts decide where it can be applied, both
measured rather than assumed:

* On a leaf control it works, and it is LOSSLESS here: every focusable
  Button in the rail already paints at least one leading space (Textual's
  `Button` padding, or the centring of a fixed-width one), so the block
  lands on padding and no label character is overwritten.
* On a CONTAINER it is almost entirely invisible, because the compositor
  paints children OVER the container's own strips. `#console-inspector-
  rail-body` showed its edge only in the gaps BETWEEN sections -- and at
  80x24 its three rows are fully covered, so it showed nothing at all.
  The edge therefore also has to be applied to the innermost widget with a
  free column: `.console-inspector-section-body`, whose `padding-left: 1`
  reserves exactly one.
"""

from __future__ import annotations

import pytest
from textual.widget import Widget
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import APP_STYLESHEETS, app_css_text
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


#: The two terminal sizes the rail's critique was measured at.
SUPPORTED_SIZES = ((80, 24), (200, 50))


class FocusCarrierHarness(ConsoleHarness):
    """Console harness under the real generated stylesheets.

    Focus treatment is a CSS question; a bare ``ConsoleHarness`` sets no
    ``CSS_PATH`` and would pass every assertion below vacuously.
    """

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


def _painted_rows(app, region) -> list[str]:
    """Return the plain text the compositor actually painted in ``region``."""

    strips = app.screen._compositor.render_strips()
    rows: list[str] = []
    for y in range(region.y, min(region.bottom, len(strips))):
        row = "".join(segment.text for segment in strips[y])
        rows.append(row[region.x : region.right])
    return rows


async def _open_rail(host, pilot):
    """Open the Inspect rail and return the console screen and rail."""

    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    await pilot.press("alt+i")
    await pilot.pause()
    await pilot.pause()
    return console, console.query_one("#console-right-rail")


async def _focus_carrier_rows(host, pilot, console, widget) -> tuple[list[str], list[str]]:
    """Return (unfocused, focused) painted rows for one widget's own region.

    Both captures have to describe the SAME cells, and inside a scroller
    that is the whole difficulty: focusing a stop scrolls it into view, so a
    naive "capture, focus, capture" reads the scroll as if it were the focus
    cue (a false pass), and a "focus, capture, blur, capture" reads whatever
    the settling blur scrolled into that region instead (a false failure --
    it produced one, comparing the More toggle against a Selected
    Conversation heading). Park focus outside, scroll the widget in
    deliberately, pin its settled region, then focus with
    ``scroll_visible=False`` so nothing moves.
    """

    composer = console.query_one("#console-native-composer")
    console.set_focus(composer)
    await pilot.pause()
    widget.scroll_visible(animate=False, immediate=True)
    await pilot.pause()
    await pilot.pause()
    region = widget.region
    unfocused = _painted_rows(host, region)
    console.set_focus(widget, scroll_visible=False)
    await pilot.pause()
    await pilot.pause()
    assert widget.region == region, (
        f"#{widget.id} moved between the two captures ({region} -> "
        f"{widget.region}); the comparison would read two different areas"
    )
    focused = _painted_rows(host, region)
    return unfocused, focused


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SUPPORTED_SIZES)
async def test_every_rail_tab_stop_changes_what_it_paints_on_focus(size):
    """AC#1 + AC#2: no Tab stop in the rail is indication-free.

    The whole rail focus ring, not a hand-listed subset -- the critique's
    finding was that a convention was invisible, and a subset list would let
    the next stop ship with the same defect.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=size) as pilot:
        console, rail = await _open_rail(host, pilot)
        stops = [
            widget
            for widget in console.focus_chain
            if rail in widget.ancestors_with_self
        ]
        assert len(stops) >= 8, f"rail focus ring collapsed to {len(stops)} stops"

        indication_free: list[str] = []
        for widget in stops:
            unfocused, focused = await _focus_carrier_rows(
                host, pilot, console, widget
            )
            if unfocused == focused:
                indication_free.append(widget.id or type(widget).__name__)
        assert not indication_free, (
            "these Inspect-rail Tab stops paint identical text focused and "
            f"unfocused at {size}: {indication_free}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SUPPORTED_SIZES)
async def test_rail_button_focus_edge_costs_no_label_character(size):
    """AC#1: the carrier is an accent edge on padding, not a truncation.

    A block glyph that overwrote the first label character would trade one
    a11y defect for a legibility one -- "Narrow…" reading "█arrow…". Every
    rail Button already paints a leading space, so the edge is lossless;
    this pins that, per button, rather than trusting the current labels.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=size) as pilot:
        console, rail = await _open_rail(host, pilot)
        buttons = [
            widget
            for widget in console.focus_chain
            if rail in widget.ancestors_with_self and isinstance(widget, Button)
        ]
        assert buttons, "the rail has no focusable buttons to check"
        for button in buttons:
            unfocused, focused = await _focus_carrier_rows(
                host, pilot, console, button
            )
            assert focused[0].startswith("█"), (
                f"#{button.id} has no accent edge when focused: {focused[0]!r}"
            )
            assert focused[0][1:] == unfocused[0][1:], (
                f"#{button.id}'s focus edge overwrote label text: "
                f"{unfocused[0]!r} -> {focused[0]!r}"
            )


# --- AC#3: nothing hidden may hold a Tab stop --------------------------------


def _hidden_behind_an_ancestor(widget: Widget) -> bool:
    node: object = widget
    while isinstance(node, Widget):
        if not node.display:
            return True
        node = node.parent
    return False


@pytest.mark.asyncio
async def test_no_console_tab_stop_is_hidden_behind_a_display_none_ancestor():
    """AC#3: a widget nobody can see must not hold a place in the Tab ring.

    The critique attributed "Tab never reaches the rail" to a hidden-but-
    focusable blank widget labelled "Review changes". Measured, that widget
    is `#console-prompt-improvement-review` in the COMPOSER's hidden
    prompt-improvement recovery row (not the left rail), and it is kept out
    of the ring only incidentally -- by `disabled=True` plus its parent's
    `display: none`. This is the standing guard: no focus-chain member may
    be hidden by any ancestor.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=(200, 50)) as pilot:
        console, _rail = await _open_rail(host, pilot)
        hidden = [
            widget.id or type(widget).__name__
            for widget in console.focus_chain
            if _hidden_behind_an_ancestor(widget)
        ]
        assert not hidden, f"hidden widgets hold Tab stops: {hidden}"


@pytest.mark.asyncio
async def test_hidden_composer_recovery_actions_are_not_focusable():
    """AC#3: the display/focusability coupling is explicit, not incidental.

    `#console-prompt-improvement-review` ("Review changes") and its Undo
    sibling live in a row that is `display: none` until an improvement lands.
    Today only `disabled=True` and the hidden parent keep them out of the
    ring; enabling either one without showing the row would put a blank stop
    in the composer's Tab region. `can_focus` now follows the row's display.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=(200, 50)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        row = console.query_one("#console-prompt-improvement-recovery")
        assert not row.display, "fixture assumption: the recovery row starts hidden"
        for widget_id in (
            "console-prompt-improvement-undo",
            "console-prompt-improvement-review",
        ):
            widget = console.query_one(f"#{widget_id}", Button)
            assert widget.can_focus is False, (
                f"#{widget_id} is focusable while its row is hidden"
            )


@pytest.mark.asyncio
async def test_tab_from_the_composer_stays_in_the_composer_region():
    """AC#3: Tab not reaching the rail is the DESIGN, not an absorbed route.

    `ChatScreen.action_focus_next` scopes Tab to the focused widget's
    `CONSOLE_TAB_REGIONS` entry (TASK-2154.11 AC-02) so the tour does not
    cross fifteen app-nav buttons mid-Console; F6 moves between panes and
    Alt+I opens and enters the rail. Measured here so the critique's "40
    presses and never reaches the rail" is pinned to its real cause and a
    future regression cannot re-file it as a hidden widget.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=(200, 50)) as pilot:
        console, rail = await _open_rail(host, pilot)
        composer = console.query_one("#console-native-composer")
        console.set_focus(composer)
        await pilot.pause()
        visited = []
        for _ in range(24):
            await pilot.press("tab")
            await pilot.pause()
            focused = host.focused
            assert focused is not None
            visited.append(focused.id or type(focused).__name__)
            assert rail not in focused.ancestors_with_self, (
                "Tab crossed into the Inspect rail; the region scoping in "
                "CONSOLE_TAB_REGIONS regressed"
            )
        assert len(set(visited)) > 1, f"Tab did not move at all: {visited}"

        # F6 is the documented route, and it works.
        for _ in range(5):
            await pilot.press("f6")
            await pilot.pause()
            focused = host.focused
            if focused is not None and rail in focused.ancestors_with_self:
                break
        else:  # pragma: no cover - only on regression
            pytest.fail("F6 never reached the Inspect rail from the composer")


# --- AC#4: the scrollbar thumb must be visible against its track -------------


def _relative_luminance(triplet) -> float:
    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


def _contrast(first, second) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _scrollbar_thumb_and_track(app, widget):
    """Return the painted (thumb, track) colours of a widget's vertical bar.

    Read from the compositor, not from `styles`: Textual paints the thumb
    with `Style(color=bar, reverse=True)`, so the colour a viewer SEES as
    the thumb's background is the segment's foreground. A styles-only
    assertion cannot see that, and the defect this pins (thumb and track
    rendering as the same block of colour) hides exactly there.
    """

    strips = app.screen._compositor.render_strips()
    region = widget.region
    column = region.right - 1
    thumb = None
    track = None
    for y in range(region.y, min(region.bottom, len(strips))):
        x = 0
        for segment in strips[y]:
            for _character in segment.text:
                if x == column:
                    style = segment.style
                    if style.reverse and style.color is not None:
                        thumb = style.color.triplet
                    elif style.bgcolor is not None and thumb is not None:
                        track = track or style.bgcolor.triplet
                x += 1
    return thumb, track


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SUPPORTED_SIZES)
async def test_rail_scrollbar_thumb_is_visible_against_its_track(size):
    """AC#4: measured 1.01:1 before this task -- an invisible thumb.

    `scrollbar-color: $ds-grid-line` (#2d2d2d) on `scrollbar-background:
    $ds-surface-panel` (#242f38) are two near-identical dark greys: the
    scroller advertised a fold it could not show a position for.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=size) as pilot:
        console, _rail = await _open_rail(host, pilot)
        body = console.query_one("#console-inspector-rail-body")
        assert body.show_vertical_scrollbar, "fixture assumption: the rail overflows"
        thumb, track = _scrollbar_thumb_and_track(host, body)
        assert thumb is not None and track is not None, (
            f"could not read the scrollbar's painted colours: {thumb} / {track}"
        )
        ratio = _contrast(thumb, track)
        assert ratio >= 3.0, (
            f"the rail scrollbar thumb measures {ratio:.2f}:1 against its "
            f"track at {size} (thumb={thumb} track={track}); the non-text "
            "floor is 3:1"
        )


# --- AC#5: the pinned stack must not eat the scroll body at 80x24 -----------


def test_the_compact_pinned_block_has_both_halves_of_its_geometry():
    """AC#5: the widget pins its height inline; the sheet must agree.

    `ConsoleSendAuthoritySummary` has always written its own height inline
    (a bare harness loads no bundle), and inline wins over CSS in Textual --
    so a compact rule missing from the stylesheet would never be noticed at
    runtime and would leave the sheet claiming a six-row block forever.
    """

    stylesheet = app_css_text()
    assert "#console-send-authority-summary.-authority-compact" in stylesheet, (
        "the compact pinned-summary height is only in Python; the stylesheet "
        "still says the block is six rows tall at every size"
    )


@pytest.mark.asyncio
async def test_pinned_authority_block_compacts_at_80x24_and_not_at_200x50():
    """AC#5: eight pinned rows over a three-row body is not a rail.

    Measured on this branch before the change: at 80x24 the rail's five
    children were header 1, project instruction 1, `#console-send-authority-
    summary` 6, scroll body 3, overflow hint 1. Compacting the block to two
    rows turns that body into seven -- exactly one Environment section at
    rest (TASK-31662 measured it at seven rows).
    """

    from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
        ConsoleSendAuthoritySummary,
    )

    for size, expected_height, expected_compact in (
        ((80, 24), 2, True),
        ((200, 50), 6, False),
    ):
        app = _build_test_app()
        _configure_native_ready_console(app)
        host = FocusCarrierHarness(app)
        async with host.run_test(size=size) as pilot:
            console, _rail = await _open_rail(host, pilot)
            summary = console.query_one(
                "#console-send-authority-summary", ConsoleSendAuthoritySummary
            )
            assert summary.compact is expected_compact, (
                f"at {size} the pinned block's compact state is "
                f"{summary.compact}, expected {expected_compact}"
            )
            assert summary.region.height == expected_height, (
                f"at {size} the pinned block paints "
                f"{summary.region.height} rows, expected {expected_height}"
            )
            body = console.query_one("#console-inspector-rail-body")
            if expected_compact:
                assert body.content_region.height >= 7, (
                    "the compacted pinned stack did not buy the scroll body "
                    f"a section's worth of rows: {body.content_region.height}"
                )
            # The five facts survive the compaction: the projection, not the
            # mounted rows, is what F1's contextual help reads.
            assert len(summary.contextual_help_rows()) == 5
            run_row = console.query_one("#console-send-authority-run")
            assert run_row.display, "the Run rollup must keep its line"
            if expected_compact:
                assert summary.tooltip is not None, (
                    "the compacted block must carry the hidden facts"
                )
                for widget_id in (
                    "console-send-authority-where",
                    "console-send-authority-scope",
                    "console-send-authority-sources",
                    "console-send-authority-approvals",
                ):
                    assert not console.query_one(f"#{widget_id}").display


@pytest.mark.asyncio
async def test_environment_four_rows_are_painted_at_80x24():
    """AC#5, the outcome the compaction exists for.

    TASK-31662 made the Environment section seven rows tall at rest; this is
    the rail-level sibling of its component test -- the four rows have to
    survive the trip through the REAL pinned stack at the smallest supported
    terminal, not just inside a harness sized to fit them.
    """

    from Tests.UI.test_console_environment_wiring import _snapshot
    from tldw_chatbook.Widgets.Console.console_inspector_section import (
        ConsoleInspectorSectionRow,
    )
    from tldw_chatbook.Workspaces.change_tracking import ChangedFile

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        console, _rail = await _open_rail(host, pilot)
        snapshot = _snapshot(files=(ChangedFile("M", "a.py", 3, 1),))
        console._console_environment.snapshot = snapshot
        console._land_console_environment(snapshot)
        await pilot.pause()
        await pilot.pause()

        section = console.query_one("#console-environment-section")
        rows = list(section.query(ConsoleInspectorSectionRow))
        assert len(rows) == 4, [row.row_id for row in rows]
        painted = host.screen._compositor.visible_widgets
        unpainted = [row.row_id for row in rows if row not in painted]
        assert not unpainted, (
            f"Environment rows still below the fold at 80x24: {unpainted}"
        )
        for row in rows:
            primary = row.query_one(".console-inspector-section-row-primary")
            assert primary in painted, f"{row.row_id}'s text is not painted"
            assert _painted_rows(host, primary.region)[0].strip(), (
                f"{row.row_id} paints a blank line at 80x24"
            )
