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
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import APP_STYLESHEETS, app_css_text
from Tests.UI.test_console_environment_wiring import _snapshot
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSectionRow,
)
from tldw_chatbook.Workspaces.change_tracking import ChangedFile


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
    await _land_environment_rows(console, pilot)
    return console, console.query_one("#console-right-rail")


async def _land_environment_rows(console, pilot) -> None:
    """Put the Environment section's four rows into the rail's focus ring.

    Without this the default fixture's Environment section is UNBOUND and
    projects no rows at all, so a "walk every stop in the ring" test walks a
    ring that contains no `ConsoleInspectorSectionRow` -- which is how the
    first cut of this file shipped a dead `.console-inspector-section-row:focus`
    rule with nothing to catch it. The same canned OK-git snapshot
    `test_console_environment_wiring.py` uses; nothing shells out to git.
    """

    console._stop_console_transcript_sync_timer()
    snapshot = _snapshot(files=(ChangedFile("M", "a.py", 3, 1),))
    console._console_environment.snapshot = snapshot
    console._land_console_environment(snapshot)
    await pilot.pause()
    await pilot.pause()
    # Freeze the Environment feed for the rest of the test. A focus walk over
    # the whole ring is many `pilot.pause`es long, and opening the rail arms
    # `notify_rail_opened` -> a background refresh whose landing REPLACES row
    # widgets when the section's structural key changes. That orphaned the
    # "Refresh" button mid-walk (its region went to 0x0) -- a fixture race,
    # not a product defect, but one that makes the walk's verdict luck.
    console._console_environment.request_refresh = lambda **_kwargs: None
    console._console_environment.poll_tick = lambda: None


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
        # The ring must actually CONTAIN the widget classes it claims to
        # cover. A ring with no rows in it is how a dead row rule passed.
        assert [
            widget.row_id
            for widget in stops
            if isinstance(widget, ConsoleInspectorSectionRow)
        ], f"no inspector rows in the ring at {size}; the walk proves nothing"

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
    rail Button already paints a leading space, so the edge is lossless.

    The check is on the CARRIER COLUMN's own content before focus, not on a
    focused-vs-unfocused diff of the rest of the row: the latter compares
    both sides with that column already dropped and so cannot see the loss
    it is meant to catch (it is exactly what let the collapsed rail handle,
    whose column 0 holds the "◂" of "◂ Inspect", look safe).
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
            _assert_lossless_edge(button.id, unfocused, focused, side="left")


def _assert_lossless_edge(widget_id, unfocused, focused, *, side: str) -> None:
    """Assert an accent edge appeared and cost no painted character.

    Args:
        widget_id: The widget's DOM id, for failure messages.
        unfocused: Painted rows before focus.
        focused: Painted rows while focused.
        side: ``"left"`` (column 0) or ``"right"`` (the last column).
    """

    index = 0 if side == "left" else -1
    assert any(row[index] == "█" for row in focused), (
        f"#{widget_id} has no accent edge on its {side} when focused: {focused!r}"
    )
    blocked = [row for row in unfocused if row and row[index] not in (" ", "█")]
    assert not blocked, (
        f"#{widget_id}'s {side} carrier column already paints content "
        f"({blocked!r}); the edge would overwrite it"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SUPPORTED_SIZES)
async def test_a_focused_inspector_row_paints_corner_brackets(size):
    """AC#1, and the correction of a wrong premise in this task's first cut.

    Rows are NOT tint-only and never were: the app-wide fallback
    `*:focus { outline: solid $ds-focus-accent }` in `css/core/_reset.tcss`
    reaches them -- nothing opts rows out the way `Button:focus` opts buttons
    out with `outline: none` -- so a focused row paints `┌…┐` around its own
    padding columns. That is the row's carrier, and it is deliberately NOT
    the buttons' `█` edge: brackets wrap the ONE focused row, while a `█`
    column beside every row is what the focused SCROLLER paints, and keeping
    them distinct is what tells those two states apart.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=size) as pilot:
        console, rail = await _open_rail(host, pilot)
        rows = [
            widget
            for widget in console.focus_chain
            if rail in widget.ancestors_with_self
            and isinstance(widget, ConsoleInspectorSectionRow)
        ]
        assert rows, "no focusable inspector rows to check"
        for row in rows:
            unfocused, focused = await _focus_carrier_rows(host, pilot, console, row)
            assert focused[0][0] in "┌╭", (
                f"row {row.row_id} paints no focus bracket: {focused[0]!r}"
            )
            assert unfocused[0][0] == " ", (
                f"row {row.row_id}'s bracket column already paints content: "
                f"{unfocused[0]!r}"
            )
            assert focused[0][1:-1] == unfocused[0][1:-1], (
                f"row {row.row_id}'s focus brackets ate its text: "
                f"{unfocused[0]!r} -> {focused[0]!r}"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(200, 50), (140, 40)])
async def test_the_collapsed_rail_handle_has_a_focus_carrier(size):
    """AC#1 for the rail's SHIPPING DEFAULT state (review M8).

    `#console-inspector-rail-open` is a Button, but it lives in
    `#console-inspector-rail-handle`, outside `#console-right-rail`, so the
    rail-scoped rule missed it -- and measured, focusing it changed nothing
    at all. It is the one stop a user meets before the rail is ever opened.

    A RIGHT edge, because measured its column 0 carries the "◂" of
    "◂ Inspect" at every width the handle is shown while its last column is
    blank on every row. Below ~84 columns the handle is hidden entirely and
    Alt+I is the only route, so there is no stop to indicate there.
    """

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusCarrierHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        handle = console.query_one("#console-inspector-rail-handle")
        assert handle.display, (
            f"fixture assumption: the rail starts collapsed at {size}"
        )
        opener = console.query_one("#console-inspector-rail-open", Button)
        assert opener in console.focus_chain, "the handle is not a Tab stop"
        unfocused, focused = await _focus_carrier_rows(host, pilot, console, opener)
        assert unfocused != focused, (
            "focusing the collapsed rail's handle changes nothing a keyboard "
            f"user can see at {size}"
        )
        _assert_lossless_edge(opener.id, unfocused, focused, side="right")


# --- AC#3: nothing hidden may hold a Tab stop --------------------------------
#
# There is deliberately NO "no focus-chain member is hidden by an ancestor"
# test here. Textual builds `Screen.focus_chain` by walking `displayed_children`,
# so such a test cannot fail whatever this app does -- it would assert the
# framework, not the code. The real guard is the coupling test below.


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
    assert (
        "#console-send-authority-summary.console-authority-compact" in stylesheet
    ), (
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
            heading = console.query_one("#console-send-authority-heading")
            painted_heading = _painted_rows(host, heading.region)[0]
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
                # Review M7: the compression must ANNOUNCE itself. Four facts
                # vanishing silently leaves two rows that still read like a
                # complete answer.
                assert "+4" in painted_heading, (
                    "the compacted block hides four facts with no cue: "
                    f"{painted_heading!r}"
                )
                assert "…" not in painted_heading, (
                    "the compact heading is being ellipsized, so its own "
                    f"marker is what gets cut: {painted_heading!r}"
                )
            else:
                assert "+4" not in painted_heading, (
                    f"the full-height block claims hidden facts: {painted_heading!r}"
                )


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
