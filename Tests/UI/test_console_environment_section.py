"""Widget-level tests for the Environment/Tasks sections (ConsolidatedCSSApp harness).

Mirrors ``Tests/UI/test_console_inspector_section.py``'s ``_SectionHarness``
pattern (task-9 brief): a minimal host mounting one ``ConsoleInspectorSection``
directly, no ``ChatScreen`` involved -- this is a standalone-component test
proving the Task 3 projection (``project_environment_section``) renders
correctly through the reusable section grammar. The rail-mounting itself is
covered by ``Tests/UI/test_console_right_rail.py``'s census tests.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_console_parallel_runs import (
    _assert_painted_at_own_region,
    _assert_widget_and_ancestors_displayed,
)
from tldw_chatbook.Chat.console_environment_state import (
    ENV_ROW_BRANCH,
    ENV_ROW_CHANGES,
    ENV_ROW_COMMIT_PUSH,
    ENV_ROW_LOCAL,
    ENV_SUMMARY_BUDGET,
    EnvironmentSnapshot,
    EnvSourceAvailability,
    GitEnvState,
    project_environment_section,
)
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    RAIL_CONTENT_WIDTH_MIN,
    ConsoleInspectorSection,
    ConsoleInspectorSectionRow,
    ConsoleInspectorSectionState,
    InspectorSectionRow,
)
from tldw_chatbook.Workspaces.change_tracking import ChangedFile

_NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _section(
    branch: str = "feat/task-1-x", *, open: bool = True, files=()
) -> ConsoleInspectorSection:
    snapshot = EnvironmentSnapshot(git=GitEnvState(
        availability=EnvSourceAvailability.OK, root="/w", branch=branch,
        adds=sum(f.adds for f in files) or 10,
        dels=sum(f.dels for f in files) or 2,
        files=tuple(files),
    ))
    state = project_environment_section(snapshot, frozenset(), now=_NOW)
    # Same kwargs the rail composes with (`right_rail.py::compose`) -- the
    # production half of the suppression pair is asserted in
    # `Tests/UI/test_console_environment_wiring.py`.
    return ConsoleInspectorSection(
        title="Environment", section_id="environment",
        rows=state.rows, summary=state.summary,
        collapsible=True, open=open, view_all_label="Refresh",
        suppress_summary_when_open=True,
        id="console-environment-section",
    )


class _EnvironmentSectionHarness(ConsolidatedCSSApp):
    """Minimal host mounting the Environment section directly."""

    BRANCH = "feat/task-1-x"

    def __init__(self) -> None:
        super().__init__()
        self.view_all_events: list[str] = []

    def compose(self) -> ComposeResult:
        yield _section(self.BRANCH)

    @on(ConsoleInspectorSection.ViewAllRequested)
    def _on_view_all(self, event: ConsoleInspectorSection.ViewAllRequested) -> None:
        self.view_all_events.append(event.section_id)


@pytest.mark.asyncio
async def test_environment_section_renders_rows_and_refresh_slot():
    app = _EnvironmentSectionHarness()
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        section = app.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        primaries = [
            str(static.renderable)
            for static in section.query(".console-inspector-section-row-primary")
        ]
        assert any("Changes" in primary for primary in primaries)

        view_all = app.query_one("#console-inspector-section-environment-view-all")
        view_all.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.view_all_events == ["environment"]


@pytest.mark.asyncio
async def test_view_all_busy_flips_the_tail_label_and_restores_it():
    """TASK-31664 AC#3, widget seam: the "Refresh" tail can show a
    transient acknowledgment independent of ``sync_state`` -- which is the
    point, since ``sync_state`` treats a content-identical landing as a
    no-op by design and never touches the tail button itself."""
    from textual.widgets import Button

    app = _EnvironmentSectionHarness()
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        section = app.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        button = app.query_one(
            "#console-inspector-section-environment-view-all", Button
        )
        assert str(button.label) == "Refresh"

        section.set_view_all_busy(True)
        await pilot.pause()
        assert str(button.label) == "Refreshing…"

        section.set_view_all_busy(False)
        await pilot.pause()
        assert str(button.label) == "Refresh"


@pytest.mark.asyncio
async def test_view_all_busy_survives_a_mid_ack_recompose():
    """TASK-31664 round-1 review follow-on (surfaced by I1's own test).

    ``set_view_all_busy`` used to be a pure live-widget mutation with no
    backing state on the section itself -- so a structural row-set change
    (the PENDING -> real-rows transition is the production case, but ANY
    row added/removed/reordered while an explicit Refresh is still in
    flight has the same shape) triggers ``sync_state``'s full recompose,
    which rebuilds the tail Button from ``compose()`` and silently
    reverted "Refreshing…" to "Refresh" while the press it was
    acknowledging was still outstanding. The busy flag must now be
    consulted by ``compose()`` itself, so a fresh rebuild reproduces it.
    """
    from textual.widgets import Button

    app = _EnvironmentSectionHarness()
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        section = app.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        section.set_view_all_busy(True)
        await pilot.pause()
        button = app.query_one(
            "#console-inspector-section-environment-view-all", Button
        )
        assert str(button.label) == "Refreshing…"

        # A structurally DIFFERENT row set (different row_ids) forces
        # `sync_state` down the recompose path, not the in-place patch one.
        section.sync_state(
            ConsoleInspectorSectionState(
                rows=(InspectorSectionRow(row_id="a-brand-new-row", primary_text="new"),),
                summary="",
            )
        )
        await pilot.pause()

        # Re-queried, deliberately: the recompose means the ORIGINAL
        # `button` reference is now a detached, orphaned widget -- reading
        # it again would be exactly the false-green this test exists to
        # rule out.
        rebuilt_button = app.query_one(
            "#console-inspector-section-environment-view-all", Button
        )
        assert str(rebuilt_button.label) == "Refreshing…"

        section.set_view_all_busy(False)
        await pilot.pause()
        assert str(
            app.query_one(
                "#console-inspector-section-environment-view-all", Button
            ).label
        ) == "Refresh"


@pytest.mark.asyncio
async def test_row_markers_resolve_through_ascii_glyph_fallback():
    """TASK-31664 round-1 review I3.

    The section header's own chevron already resolves through
    ``glyph_fallback`` (``appearance.ascii_glyphs``); the row-level "▸"/"▾"
    affordance markers TASK-31664 added did not, so a narrow-font terminal
    that correctly showed ">"/"v" on the header would print the raw ▸/▾ on
    every row -- exactly the terminals the fallback exists for. Resolution
    lives at this render seam (``ConsoleInspectorSectionRow``), not in the
    (pure) projection that built the row's ``primary_text``.
    """
    from tldw_chatbook.Widgets.glyph_fallback import set_ascii_glyph_mode

    app = _EnvironmentSectionHarness()
    set_ascii_glyph_mode(True)
    try:
        async with app.run_test(size=(70, 24)) as pilot:
            await pilot.pause()
            primaries = [
                str(static.renderable)
                for static in app.query(".console-inspector-section-row-primary")
            ]
            changes_row = next(t for t in primaries if t.startswith("Changes"))
            assert changes_row == "Changes >"  # not the raw "Changes ▸"
            assert "▸" not in changes_row and "▾" not in changes_row
    finally:
        set_ascii_glyph_mode(False)


@pytest.mark.asyncio
async def test_view_all_busy_is_a_no_op_without_a_tail():
    """A section with no ``view_all_label`` has no button to flip -- must
    not raise."""
    class _NoTailHarness(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield ConsoleInspectorSection(
                title="Environment", section_id="environment",
                id="console-environment-section",
            )

    app = _NoTailHarness()
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        section = app.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        section.set_view_all_busy(True)  # must not raise
        section.set_view_all_busy(False)  # must not raise


class _RailWidthHarness(ConsolidatedCSSApp):
    """Host that pins the section to the Inspect rail's real content width.

    TASK-31662 / TASK-31629 #12: that width is 30 columns, not the 34 this
    harness used to assume. Probed on this branch 2026-09-05 against the
    real Console (`#console-environment-section` inside
    `#console-inspector-rail-body`): Size(width=30) at 80x24 and
    Size(width=36) at 200x50 -- so 30 is what the SMALLEST supported
    terminal actually produces, and pinning 34 tested a width no user has.
    At 34 the old 18-column budget passed; at the real 30 it painted
    "Environm…" (measured, same probe).

    Loads the app sheets rather than the widget-defaults pair: the row
    padding and body indent that take a row's own text width to 27 live in
    the console-owned split sheet.
    """

    BRANCH = "feat/task-1-x"
    OPEN = True
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        section = _section(self.BRANCH, open=self.OPEN)
        section.styles.width = RAIL_CONTENT_WIDTH_MIN
        section.styles.max_width = RAIL_CONTENT_WIDTH_MIN
        yield section


@pytest.mark.parametrize(
    "branch",
    [
        "feat/task-1-x",
        "feat/console-inspector-environment-redesign-and-then-some",
    ],
    ids=["short-branch", "long-branch"],
)
@pytest.mark.asyncio
async def test_section_title_still_paints_at_rail_width(branch):
    """F1: the header summary must never squeeze out the title or chevron.

    Asserts what the title Static actually PAINTS (its own rendered line),
    not the whole frame -- a whole-frame assertion would pass on a title
    that had been reduced to a single visible character. Collapsed, which
    is the state the summary is FOR (see the open case below).
    """

    class _Harness(_RailWidthHarness):
        BRANCH = branch
        OPEN = False

    app = _Harness()
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        title = app.query_one(
            "#console-inspector-section-environment-title", Static
        )
        toggle = app.query_one("#console-inspector-section-environment-toggle")
        summary = app.query_one(
            "#console-inspector-section-environment-summary", Static
        )

        # The title's own painted line first -- that is the reported symptom.
        assert title.size.width >= len("Environment")
        painted = title.render_line(0).text.strip()
        assert painted == "Environment", painted
        assert toggle.size.width == 3  # the collapse chevron survives too
        assert summary.size.width <= ENV_SUMMARY_BUDGET
        assert summary.display


@pytest.mark.parametrize(
    "branch",
    [
        "feat/task-1-x",
        "feat/console-inspector-environment-redesign-and-then-some",
    ],
    ids=["short-branch", "long-branch"],
)
@pytest.mark.asyncio
async def test_open_section_hides_its_summary_and_paints_the_full_title(branch):
    """AC#3 + AC#5 at the real width: while the section is OPEN its rows
    already carry the branch and the counts, so the summary stands down and
    the title gets the whole header."""

    class _Harness(_RailWidthHarness):
        BRANCH = branch

    app = _Harness()
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        title = app.query_one(
            "#console-inspector-section-environment-title", Static
        )
        summary = app.query_one(
            "#console-inspector-section-environment-summary", Static
        )
        toggle = app.query_one("#console-inspector-section-environment-toggle")
        assert not summary.display
        assert title.render_line(0).text.strip() == "Environment"
        assert toggle.size.width == 3


class _RestingSectionHarness(ConsolidatedCSSApp):
    """The Environment section at rest, at the rail's real width, inside a
    scrollable viewport as tall as the section's own at-rest content."""

    # 8 = the section's own seven at-rest lines (header + four rows +
    # the Refresh tail's margin and button) plus the one-line bottom
    # margin `.console-inspector-section` carries between sections.
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]
    CSS = f"#scroll {{ width: {RAIL_CONTENT_WIDTH_MIN + 1}; height: 8; }}"

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="scroll"):
            section = _section(files=(ChangedFile("M", "a.py", 3, 1),))
            section.styles.width = RAIL_CONTENT_WIDTH_MIN
            section.styles.max_width = RAIL_CONTENT_WIDTH_MIN
            yield section


@pytest.mark.asyncio
async def test_environment_at_rest_shows_its_four_top_level_rows_unscrolled():
    """AC#2: the four rows the panel opens with -- Changes, Local, the
    branch, and the commit/push offer -- each take ONE line, so all four
    (plus the header) paint inside seven rows with nothing scrolled off.

    Before TASK-31662 the same four rows took eight lines and the section
    needed eleven (measured at 80x24 on this branch), which is what made a
    3-line rail viewport show a single row that restated its own header.
    """
    app = _RestingSectionHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        scroll = app.query_one("#scroll", VerticalScroll)
        section = app.query_one(
            "#console-environment-section", ConsoleInspectorSection
        )
        rows = list(section.query(ConsoleInspectorSectionRow))
        assert [row.row_id for row in rows] == [
            ENV_ROW_CHANGES, ENV_ROW_LOCAL, ENV_ROW_BRANCH, ENV_ROW_COMMIT_PUSH
        ]
        for row in rows:
            assert row.size.height == 1, row.row_id
            primary = row.query_one(".console-inspector-section-row-primary", Static)
            _assert_widget_and_ancestors_displayed(primary)
            _assert_painted_at_own_region(app, primary)
        # Nothing is hidden below a fold: the viewport holds the whole
        # header + four rows without a scroll offset to reach them.
        # 6 since TASK-31665 AC#5 attached the Refresh tail to the section
        # (it used to reserve a blank line above itself, making this 7).
        assert section.size.height == 6
        assert scroll.max_scroll_y == 0
