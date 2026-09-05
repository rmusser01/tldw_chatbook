"""Supervisor fleet PR 2b, Task 3: the reusable Inspector-style section.

Rendered-geometry assertions throughout, not DOM presence -- per the Library
UAT lesson this repo has been burned by before: an unbounded-width Static
can be "present" in a headless query while rendering invisible on a real
terminal. `_assert_painted_at_own_region`/`_assert_widget_and_ancestors_
displayed` (imported from `test_console_parallel_runs.py`, the precedent
this whole PR cites) use the compositor's own hit-test rather than a raw
`region.y` bound, since `Widget.region` is reported UNCLIPPED -- a widget
positioned below the fold of a scrollable ancestor still has a non-empty
`region`, just one nothing actually paints.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual import on

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll

from Tests.UI.test_console_parallel_runs import (
    _assert_painted_at_own_region,
    _assert_widget_and_ancestors_displayed,
)
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionRow,
    ConsoleInspectorSectionState,
    InspectorSectionRow,
)


def _rows(
    n: int, *, clickable: bool = False, status: str = "", cancellable: bool = False
) -> tuple[InspectorSectionRow, ...]:
    return tuple(
        InspectorSectionRow(
            row_id=f"row-{i}",
            primary_text=f"Agent {i} - running",
            secondary_text=f"last step {i}",
            status=status,
            clickable=clickable,
            cancellable=cancellable,
        )
        for i in range(n)
    )


def test_state_construction_requires_both_rows_and_summary():
    """task-3 review round 3, HIGH: round 1's fix kept per-field defaults
    on `ConsoleInspectorSectionState` (`rows: ... = ()`, `summary: str =
    ""`), so `ConsoleInspectorSectionState(rows=updated_rows)` --
    omitting `summary`, the exact "just refresh the rows" shape this test
    file itself used to write -- silently reproduced the original
    "sync_state wipes the other dimension" bug one call-frame later.
    Removing the defaults makes partial construction a `TypeError` instead
    of a silent data-loss bug."""
    with pytest.raises(TypeError):
        ConsoleInspectorSectionState(rows=_rows(1))  # summary omitted
    with pytest.raises(TypeError):
        ConsoleInspectorSectionState(summary="2 working")  # rows omitted
    with pytest.raises(TypeError):
        ConsoleInspectorSectionState()  # both omitted


def test_state_construction_with_an_explicit_empty_summary_still_works():
    """The deliberate "no summary" case must stay expressible -- it just
    has to be SAID (`summary=""`), not left to a default."""
    state = ConsoleInspectorSectionState(rows=_rows(1), summary="")
    assert state.rows == _rows(1)
    assert state.summary == ""


class _SectionHarness(ConsolidatedCSSApp):
    """Minimal host mounting one Inspector section directly (no ChatScreen
    involved -- this is a standalone-component test, mirroring the
    `_HandleHarness`/`_SectionHeaderHarness` pattern in
    `test_destination_rail.py`)."""

    def __init__(self, section: ConsoleInspectorSection) -> None:
        super().__init__()
        self._section = section
        self.activated: list[tuple[str, str]] = []
        self.cancel_requested: list[tuple[str, str]] = []
        self.view_all_events: list[str] = []
        self.collapse_events: list[tuple[str, bool]] = []

    def compose(self) -> ComposeResult:
        yield self._section

    @on(ConsoleInspectorSection.RowActivated)
    def _on_row_activated(self, event: ConsoleInspectorSection.RowActivated) -> None:
        self.activated.append((event.section_id, event.row_id))

    @on(ConsoleInspectorSection.RowCancelRequested)
    def _on_row_cancel_requested(
        self, event: ConsoleInspectorSection.RowCancelRequested
    ) -> None:
        self.cancel_requested.append((event.section_id, event.row_id))

    @on(ConsoleInspectorSection.ViewAllRequested)
    def _on_view_all(self, event: ConsoleInspectorSection.ViewAllRequested) -> None:
        self.view_all_events.append(event.section_id)

    @on(ConsoleInspectorSection.CollapseToggled)
    def _on_collapse_toggled(
        self, event: ConsoleInspectorSection.CollapseToggled
    ) -> None:
        self.collapse_events.append((event.section_id, event.open))


class _ScrolledSectionHarness(ConsolidatedCSSApp):
    """Hosts a section inside a height-constrained `VerticalScroll`, the
    same shape as the real Console rail body (`#console-left-rail-body`) --
    needed to create a genuine below-the-fold row for the hit-test guard."""

    CSS = "#scroll { height: 8; }"

    def __init__(self, section: ConsoleInspectorSection) -> None:
        super().__init__()
        self._section = section

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="scroll"):
            yield self._section


@pytest.mark.asyncio
async def test_header_and_each_row_have_positive_rendered_regions():
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        summary="2 working, 1 done",
        rows=_rows(3),
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(70, 20)) as pilot:
        await pilot.pause()
        header = app.query_one("#console-inspector-section-agents-header")
        assert header.region.width > 0 and header.region.height > 0
        _assert_widget_and_ancestors_displayed(header)

        for index in range(3):
            row = app.query_one(
                f"#console-inspector-section-agents-row-{index}",
                ConsoleInspectorSectionRow,
            )
            assert row.region.width > 0 and row.region.height > 0
            _assert_widget_and_ancestors_displayed(row)
            # The hit-test asserts IDENTITY at a pixel, and a compositor hit
            # always resolves to the deepest (leaf) widget painted there --
            # a container row is never itself "the thing painted", its
            # primary-line Static is. Applied to the leaf, matching how the
            # precedent in `test_console_parallel_runs.py` always targets a
            # Static, never a container.
            primary = app.query_one(
                f"#console-inspector-section-agents-row-{index}-primary"
            )
            assert primary.region.width > 0 and primary.region.height > 0
            _assert_painted_at_own_region(app, primary)


@pytest.mark.asyncio
async def test_summary_is_right_aligned_within_the_header_region_no_chevron():
    """`collapsible=False` -- the summary is the header's last element, so
    its right edge must land exactly on the header's own right edge."""
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        summary="3 working, 1 done",
        collapsible=False,
        rows=_rows(1),
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        header = app.query_one("#console-inspector-section-agents-header")
        title = app.query_one("#console-inspector-section-agents-title")
        summary = app.query_one("#console-inspector-section-agents-summary")
        assert summary.region.width > 0 and summary.region.height > 0
        # Title is `1fr` -- it grows to consume all room not claimed by the
        # summary, so the two must be exactly adjacent...
        assert title.region.right == summary.region.x
        # ...and the summary, being the last header child, ends flush with
        # the header's own right edge.
        assert summary.region.right == header.region.right


@pytest.mark.asyncio
async def test_summary_sits_between_title_and_chevron_when_collapsible():
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        summary="2 working",
        collapsible=True,
        open=True,
        rows=_rows(1),
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        header = app.query_one("#console-inspector-section-agents-header")
        title = app.query_one("#console-inspector-section-agents-title")
        summary = app.query_one("#console-inspector-section-agents-summary")
        toggle = app.query_one("#console-inspector-section-agents-toggle")
        assert title.region.right == summary.region.x
        assert summary.region.right == toggle.region.x
        assert toggle.region.right == header.region.right


@pytest.mark.asyncio
async def test_collapsing_hides_the_body_but_keeps_the_header_painted():
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        summary="1 working",
        rows=_rows(3),
        open=True,
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        header = app.query_one("#console-inspector-section-agents-header")
        # The compositor hit-test resolves to the deepest (leaf) widget
        # painted at a pixel, never a container -- title is the header's
        # leftmost leaf, so it stands in for "the header is really
        # painted" the same way the precedent always hit-tests a Static.
        title = app.query_one("#console-inspector-section-agents-title")
        body = app.query_one("#console-inspector-section-agents-body")
        row0_primary = app.query_one(
            "#console-inspector-section-agents-row-0-primary"
        )

        _assert_painted_at_own_region(app, title)
        _assert_painted_at_own_region(app, row0_primary)

        section.set_open(False)
        await pilot.pause()

        assert app.collapse_events == [("agents", False)]
        assert not body.display
        # A collapsed body's children lose their laid-out region entirely
        # (Textual excludes `display: none` subtrees from layout) -- the
        # meaningful assertion is that the HEADER is still there and still
        # painted, not that the (now off-layout) row still reports a
        # positive region.
        assert row0_primary.region.width == 0
        _assert_widget_and_ancestors_displayed(header)
        _assert_painted_at_own_region(app, title)

        section.set_open(True)
        await pilot.pause()
        assert app.collapse_events == [("agents", False), ("agents", True)]
        assert body.display
        _assert_painted_at_own_region(app, row0_primary)


@pytest.mark.asyncio
async def test_collapsible_false_forces_permanently_open_and_hides_the_chevron():
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        collapsible=False,
        open=False,  # ignored -- collapsible=False always wins
        rows=_rows(1),
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        assert section.open is True
        assert len(app.query("#console-inspector-section-agents-toggle")) == 0
        body = app.query_one("#console-inspector-section-agents-body")
        assert body.display
        # set_open is a no-op when the section isn't collapsible.
        section.set_open(False)
        await pilot.pause()
        assert section.open is True
        assert app.collapse_events == []


@pytest.mark.asyncio
async def test_row_past_the_fold_is_caught_by_the_compositor_hit_test():
    """Mirrors the exact shape the brief warns about: rows compose fine and
    are individually "displayed", but most of them sit below the visible
    fold of a height-constrained scrollable ancestor. Only the compositor
    hit-test -- not `region` alone -- tells the two cases apart."""
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        rows=_rows(15),
        id="section",
    )
    app = _ScrolledSectionHarness(section)
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        # Leaf Statics, not the row containers -- the compositor hit-test
        # asserts pixel IDENTITY, and a hit always resolves to the deepest
        # widget painted at that pixel (the primary-line Static), never its
        # container.
        first_row_primary = app.query_one(
            "#console-inspector-section-agents-row-0-primary"
        )
        last_row_primary = app.query_one(
            "#console-inspector-section-agents-row-14-primary"
        )
        # Within the scroll box's 8-row fold: really painted.
        _assert_painted_at_own_region(app, first_row_primary)
        # Below the fold: `region` is still non-empty (unclipped)...
        assert (
            last_row_primary.region.width > 0
            and last_row_primary.region.height > 0
        )
        # ...but nothing is actually painted there.
        hit_widget, _hit_region = app.get_widget_at(
            last_row_primary.region.x + 1, last_row_primary.region.y
        )
        assert hit_widget is not last_row_primary
        with pytest.raises(AssertionError):
            _assert_painted_at_own_region(app, last_row_primary)


@pytest.mark.asyncio
async def test_clicking_a_row_after_an_in_place_patch_still_routes_to_the_right_row():
    """The load-bearing test the brief demands: prove click targeting
    survives the in-place (non-recompose) update path, rather than assuming
    it the way `console_workspace_context.py`'s history warns against."""
    rows = (
        InspectorSectionRow(
            row_id="alpha",
            primary_text="Agent alpha - running",
            secondary_text="step 1",
            status="running",
            clickable=True,
        ),
        InspectorSectionRow(
            row_id="beta",
            primary_text="Agent beta - running",
            secondary_text="step 1",
            status="running",
            clickable=True,
        ),
    )
    section = ConsoleInspectorSection(
        title="Agents", section_id="agents", rows=rows, id="section"
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        row1_widget = app.query_one(
            "#console-inspector-section-agents-row-1", ConsoleInspectorSectionRow
        )

        await pilot.click("#console-inspector-section-agents-row-1")
        await pilot.pause()
        assert app.activated == [("agents", "beta")]

        # Same row_id set, same order -- structurally compatible, so this
        # must take the IN-PLACE path (recompose_count stays 0), not a
        # hidden recompose that would trivially "fix" click targeting by
        # rebuilding the row.
        updated_rows = (
            rows[0],
            InspectorSectionRow(
                row_id="beta",
                primary_text="Agent beta - done",
                secondary_text="step 2",
                status="done",
                clickable=True,
            ),
        )
        section.sync_state(
            ConsoleInspectorSectionState(rows=updated_rows, summary="")
        )
        await pilot.pause()

        assert section.recompose_count == 0
        # Literally the same mounted widget instance -- the strongest proof
        # this went through the patch path, not a recompose in disguise.
        assert (
            app.query_one(
                "#console-inspector-section-agents-row-1", ConsoleInspectorSectionRow
            )
            is row1_widget
        )
        assert row1_widget.row_id == "beta"
        # The in-place patch must actually have RENDERED the new content,
        # not merely left recompose_count at 0 (task-3 review round 2,
        # MEDIUM finding -- stripping the `.update()` calls left every
        # structural/identity assertion above green).
        assert (
            str(
                app.query_one(
                    "#console-inspector-section-agents-row-1-primary"
                ).renderable
            )
            == "Agent beta - done"
        )
        assert (
            str(
                app.query_one(
                    "#console-inspector-section-agents-row-1-secondary"
                ).renderable
            )
            == "step 2"
        )

        await pilot.click("#console-inspector-section-agents-row-1")
        await pilot.pause()
        assert app.activated == [("agents", "beta"), ("agents", "beta")]


@pytest.mark.asyncio
async def test_clicking_a_non_clickable_row_posts_nothing():
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        rows=_rows(1, clickable=False),
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        await pilot.click("#console-inspector-section-agents-row-0")
        await pilot.pause()
        assert app.activated == []


@pytest.mark.asyncio
async def test_sync_state_updating_rows_while_carrying_the_summary_forward_does_not_wipe_it():
    """task-3 review round 1, HIGH: reproduced against the pre-fix code,
    `sync_state(rows=<updated>)` treated the omitted `summary` kwarg as
    "clear this" -- the summary Static disappeared from the DOM and
    `recompose_count` spuriously incremented, even though the caller never
    intended to touch the summary. The atomic `ConsoleInspectorSectionState`
    makes that call shape impossible: the caller states the whole section
    state, so the natural "just refresh the rows" call carries the summary
    it already had forward, unchanged."""
    initial_rows = _rows(2)
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        rows=initial_rows,
        summary="2 working",
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert (
            str(app.query_one("#console-inspector-section-agents-summary").renderable)
            == "2 working"
        )

        updated_rows = tuple(
            InspectorSectionRow(
                row_id=row.row_id,
                primary_text=f"{row.primary_text} (updated)",
                secondary_text=row.secondary_text,
                status=row.status,
                clickable=row.clickable,
            )
            for row in initial_rows
        )
        section.sync_state(
            ConsoleInspectorSectionState(rows=updated_rows, summary=section.summary)
        )
        await pilot.pause()

        # Same row_id sequence + summary still present -> in-place, not a
        # recompose (and definitely not the spurious recompose the bug
        # caused every time).
        assert section.recompose_count == 0
        assert (
            str(app.query_one("#console-inspector-section-agents-summary").renderable)
            == "2 working"
        )
        assert (
            str(
                app.query_one(
                    "#console-inspector-section-agents-row-0-primary"
                ).renderable
            )
            == "Agent 0 - running (updated)"
        )


@pytest.mark.asyncio
async def test_sync_state_updating_only_the_summary_does_not_wipe_the_rows():
    """Mirror of the finding above: a summary-only-intended update must not
    wipe the rows either."""
    initial_rows = _rows(2)
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        rows=initial_rows,
        summary="2 working",
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()

        section.sync_state(
            ConsoleInspectorSectionState(
                rows=section.rows, summary="3 working, 1 done"
            )
        )
        await pilot.pause()

        assert section.recompose_count == 0
        assert (
            str(app.query_one("#console-inspector-section-agents-summary").renderable)
            == "3 working, 1 done"
        )
        for index, row in enumerate(initial_rows):
            assert (
                str(
                    app.query_one(
                        f"#console-inspector-section-agents-row-{index}-primary"
                    ).renderable
                )
                == row.primary_text
            )


@pytest.mark.asyncio
async def test_row_becoming_clickable_via_sync_state_does_not_recompose():
    """task-3 review round 2, LOW: `clickable` is deliberately excluded
    from the structural key -- a row transitioning non-clickable ->
    clickable (a real fleet transition, e.g. queued -> running) must patch
    in place, not force a whole-section recompose, and the row's actual
    click behavior must reflect the new state afterward."""
    rows = (
        InspectorSectionRow(
            row_id="alpha", primary_text="Agent alpha - queued", clickable=False
        ),
    )
    section = ConsoleInspectorSection(
        title="Agents", section_id="agents", rows=rows, id="section"
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        row_widget = app.query_one(
            "#console-inspector-section-agents-row-0", ConsoleInspectorSectionRow
        )
        assert row_widget.can_focus is False

        await pilot.click("#console-inspector-section-agents-row-0")
        await pilot.pause()
        assert app.activated == []  # not clickable yet

        updated_rows = (
            InspectorSectionRow(
                row_id="alpha", primary_text="Agent alpha - running", clickable=True
            ),
        )
        section.sync_state(
            ConsoleInspectorSectionState(rows=updated_rows, summary="")
        )
        await pilot.pause()

        # Same row_id sequence -> in-place, even though clickability flipped.
        assert section.recompose_count == 0
        assert (
            app.query_one(
                "#console-inspector-section-agents-row-0", ConsoleInspectorSectionRow
            )
            is row_widget
        )
        assert row_widget.clickable is True
        assert row_widget.can_focus is True

        await pilot.click("#console-inspector-section-agents-row-0")
        await pilot.pause()
        assert app.activated == [("agents", "alpha")]


@pytest.mark.asyncio
async def test_pressing_delete_on_a_cancellable_row_posts_row_cancel_requested():
    """PR2b Task 5 (per-row cancel): Delete -- not Enter/Space -- is the
    cancel gesture, so it can coexist with a clickable row's own
    drill-in gesture without contention (see the "both at once" test
    below)."""
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        rows=_rows(1, cancellable=True),
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        row_widget = app.query_one(
            "#console-inspector-section-agents-row-0", ConsoleInspectorSectionRow
        )
        assert row_widget.can_focus is True
        row_widget.focus()
        await pilot.pause()
        await pilot.press("delete")
        await pilot.pause()
        assert app.cancel_requested == [("agents", "row-0")]
        assert app.activated == []


@pytest.mark.asyncio
async def test_pressing_delete_on_a_non_cancellable_row_posts_nothing():
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        rows=_rows(1, cancellable=False),
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        row_widget = app.query_one(
            "#console-inspector-section-agents-row-0", ConsoleInspectorSectionRow
        )
        # Not focusable at all: neither clickable nor cancellable.
        assert row_widget.can_focus is False
        row_widget.focus()
        await pilot.pause()
        await pilot.press("delete")
        await pilot.pause()
        assert app.cancel_requested == []


@pytest.mark.asyncio
async def test_a_row_can_be_both_clickable_and_cancellable_independently():
    """Enter drills in; Delete cancels -- the same row answers to both
    gestures without either interfering with the other, proving `clickable`
    and `cancellable` are genuinely independent dimensions."""
    row = InspectorSectionRow(
        row_id="alpha",
        primary_text="Agent alpha - running",
        clickable=True,
        cancellable=True,
    )
    section = ConsoleInspectorSection(
        title="Agents", section_id="agents", rows=(row,), id="section"
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        row_widget = app.query_one(
            "#console-inspector-section-agents-row-0", ConsoleInspectorSectionRow
        )
        row_widget.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.activated == [("agents", "alpha")]
        assert app.cancel_requested == []

        await pilot.press("delete")
        await pilot.pause()
        assert app.cancel_requested == [("agents", "alpha")]
        assert app.activated == [("agents", "alpha")]  # unchanged


@pytest.mark.asyncio
async def test_cancellable_re_syncs_on_an_in_place_patch():
    """Mirrors `test_row_becoming_clickable_via_sync_state_does_not_
    recompose` for the new `cancellable` field -- a running -> done
    transition (cancellable -> not) must patch in place, not recompose,
    and the row must stop answering to Delete afterward."""
    rows = (
        InspectorSectionRow(
            row_id="alpha",
            primary_text="Agent alpha - running",
            status="running",
            cancellable=True,
        ),
    )
    section = ConsoleInspectorSection(
        title="Agents", section_id="agents", rows=rows, id="section"
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        row_widget = app.query_one(
            "#console-inspector-section-agents-row-0", ConsoleInspectorSectionRow
        )
        assert row_widget.can_focus is True

        updated_rows = (
            InspectorSectionRow(
                row_id="alpha",
                primary_text="Agent alpha - done",
                status="done",
                cancellable=False,
            ),
        )
        section.sync_state(
            ConsoleInspectorSectionState(rows=updated_rows, summary="")
        )
        await pilot.pause()

        # Same row_id sequence -> in-place, even though cancellability
        # flipped (mirrors clickable's own exclusion from the structural
        # key).
        assert section.recompose_count == 0
        assert (
            app.query_one(
                "#console-inspector-section-agents-row-0", ConsoleInspectorSectionRow
            )
            is row_widget
        )
        assert row_widget.cancellable is False
        assert row_widget.can_focus is False

        row_widget.focus()
        await pilot.pause()
        await pilot.press("delete")
        await pilot.pause()
        assert app.cancel_requested == []


@pytest.mark.asyncio
async def test_sync_state_recomposes_on_a_structural_row_change():
    section = ConsoleInspectorSection(
        title="Agents", section_id="agents", rows=_rows(2), id="section"
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert section.recompose_count == 0

        # A third row added is a structural change (different row_id
        # sequence) -- must recompose, not silently drop the new row.
        section.sync_state(ConsoleInspectorSectionState(rows=_rows(3), summary=""))
        await pilot.pause()

        assert section.recompose_count == 1
        new_row = app.query_one(
            "#console-inspector-section-agents-row-2", ConsoleInspectorSectionRow
        )
        assert new_row.region.width > 0 and new_row.region.height > 0


@pytest.mark.asyncio
async def test_view_all_tail_posts_view_all_requested_for_this_section():
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        rows=_rows(1),
        view_all_label="View all",
        id="section",
    )
    app = _SectionHarness(section)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        view_all = app.query_one("#console-inspector-section-agents-view-all")
        assert view_all.region.width > 0 and view_all.region.height > 0

        await pilot.click("#console-inspector-section-agents-view-all")
        await pilot.pause()
        assert app.view_all_events == ["agents"]


def test_inspector_section_css_is_styled_in_source_and_bundle():
    """Regression guard against a hand-edit-only-the-bundle desync
    (TASK-395's failure mode) -- both the source module and the generated
    bundle must carry the new grammar's rules.

    Round-1 review M7 (TASK-31661): the CSS build's screen-owned split
    (`build_css.py`'s `split_agentic_terminal`/`split_owned_module`) moved
    every `.console-inspector-section*` rule OUT of the monolithic
    `tldw_cli_modular.tcss` and into the Console screen's own generated
    sheet, `screen_agentic_console.tcss` (loaded directly by `app.py` and
    `chat_screen.py`) -- these selectors are owned by that screen, not
    shared, so the split moves them wholesale rather than duplicating
    them. Checking the old monolithic bundle here was baselined as a
    pre-existing red for that reason: it was asserting against a file
    that no longer carries these rules at all, not detecting a real
    desync. Pointing this guard at the bundle that actually ships them
    restores its purpose.
    """
    for path in (
        Path("tldw_chatbook/css/components/_agentic_terminal.tcss"),
        Path("tldw_chatbook/css/screen_agentic_console.tcss"),
    ):
        text = path.read_text(encoding="utf-8")
        for class_name in (
            ".console-inspector-section",
            ".console-inspector-section-header",
            ".console-inspector-section-title",
            ".console-inspector-section-summary",
            ".console-inspector-section-toggle",
            ".console-inspector-section-body",
            ".console-inspector-section-row",
            ".console-inspector-section-row-primary",
            ".console-inspector-section-row-secondary",
            ".console-inspector-section-view-all",
        ):
            assert class_name in text, f"{class_name} missing from {path}"
