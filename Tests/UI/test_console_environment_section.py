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

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_environment_state import (
    EnvironmentSnapshot,
    EnvSourceAvailability,
    GitEnvState,
    project_environment_section,
)
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
)

_NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _section() -> ConsoleInspectorSection:
    snapshot = EnvironmentSnapshot(git=GitEnvState(
        availability=EnvSourceAvailability.OK, root="/w", branch="feat/task-1-x",
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

    def __init__(self) -> None:
        super().__init__()
        self.view_all_events: list[str] = []

    def compose(self) -> ComposeResult:
        yield _section()

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
