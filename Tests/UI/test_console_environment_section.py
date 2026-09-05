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
from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_environment_state import (
    ENV_SUMMARY_BUDGET,
    EnvironmentSnapshot,
    EnvSourceAvailability,
    GitEnvState,
    project_environment_section,
)
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
)

_NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _section(branch: str = "feat/task-1-x") -> ConsoleInspectorSection:
    snapshot = EnvironmentSnapshot(git=GitEnvState(
        availability=EnvSourceAvailability.OK, root="/w", branch=branch,
        adds=10, dels=2,
        files=(),
    ))
    state = project_environment_section(snapshot, frozenset(), now=_NOW)
    return ConsoleInspectorSection(
        title="Environment", section_id="environment",
        rows=state.rows, summary=state.summary,
        collapsible=True, open=True, view_all_label="Refresh",
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


class _RailWidthHarness(ConsolidatedCSSApp):
    """Host that pins the section to the Inspect rail's real body width (34).

    The rail is a fixed-width column regardless of terminal size (verified
    live at 80x24 AND 200x50), which is why an unbudgeted summary starved
    the title at every size rather than only on small terminals.
    """

    BRANCH = "feat/task-1-x"

    def compose(self) -> ComposeResult:
        section = _section(self.BRANCH)
        section.styles.width = 34
        section.styles.max_width = 34
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
    that had been reduced to a single visible character.
    """

    class _Harness(_RailWidthHarness):
        BRANCH = branch

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
