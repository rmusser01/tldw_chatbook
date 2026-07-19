"""TASK-259: ConsoleRunInspector updates changed rows only.

Row-level state changes (same rendered structure, different text/status)
must update the mounted row Statics in place; structural changes (rows or
actions added/removed, dictionary section shape changes) still recompose
the widget -- and only the widget, never the owning screen.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import (
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleInspectorState,
)
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector


def _base_state(**overrides) -> ConsoleInspectorState:
    values = {
        "rows": (
            ConsoleDisplayRow("Run recipe", "Chat with provider"),
            ConsoleDisplayRow("Provider", "OpenAI / gpt-4o", status="ready"),
            ConsoleDisplayRow("Sources", "1 staged", status="ready"),
        ),
        "actions": (),
        "dictionary_rows": (),
        "dictionary_actions": (),
    }
    values.update(overrides)
    return ConsoleInspectorState(**values)


class InspectorHarness(App):
    def __init__(self, state: ConsoleInspectorState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleRunInspector(self._state, id="inspector")


@pytest.mark.asyncio
async def test_inspector_row_text_change_updates_rows_in_place():
    app = InspectorHarness(_base_state())

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        provider_row_before = inspector.query_one("#console-inspector-provider", Static)
        recipe_row_before = inspector.query_one("#console-inspector-run-recipe", Static)

        new_state = _base_state(
            rows=(
                ConsoleDisplayRow("Run recipe", "Chat with provider"),
                ConsoleDisplayRow("Provider", "Anthropic / claude", status="ready"),
                ConsoleDisplayRow("Sources", "1 staged", status="ready"),
            )
        )
        inspector.sync_state(new_state)
        await pilot.pause()

        assert inspector.recompose_count == 0
        provider_row_after = inspector.query_one("#console-inspector-provider", Static)
        assert provider_row_after is provider_row_before
        assert str(provider_row_after.renderable) == "Provider: Anthropic / claude"
        # Unchanged rows keep both identity and content.
        assert inspector.query_one("#console-inspector-run-recipe", Static) is recipe_row_before


@pytest.mark.asyncio
async def test_inspector_row_status_change_swaps_class_and_summary_in_place():
    app = InspectorHarness(_base_state())

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        summary_before = inspector.query_one(
            "#console-inspector-run-status-summary", Static
        )
        assert str(summary_before.renderable) == "Status: Ready"

        new_state = _base_state(
            rows=(
                ConsoleDisplayRow("Run recipe", "Chat with provider"),
                ConsoleDisplayRow(
                    "Provider", "Missing API key", status="blocked", recovery="Add a key"
                ),
                ConsoleDisplayRow("Sources", "1 staged", status="ready"),
            )
        )
        inspector.sync_state(new_state)
        await pilot.pause()

        assert inspector.recompose_count == 0
        provider_row = inspector.query_one("#console-inspector-provider", Static)
        assert provider_row.has_class("console-inspector-row-blocked")
        assert not provider_row.has_class("console-inspector-row-ready")
        assert str(provider_row.renderable) == "Provider: Missing API key - Add a key"
        summary_after = inspector.query_one(
            "#console-inspector-run-status-summary", Static
        )
        assert summary_after is summary_before
        assert str(summary_after.renderable) == "Status: Blocked"


@pytest.mark.asyncio
async def test_inspector_structural_row_change_recomposes_widget():
    app = InspectorHarness(_base_state())

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)

        new_state = _base_state(
            rows=_base_state().rows
            + (ConsoleDisplayRow("Artifacts", "Chatbook available", status="ready"),)
        )
        inspector.sync_state(new_state)
        await pilot.pause()

        assert inspector.recompose_count == 1
        artifacts_row = inspector.query_one("#console-inspector-artifacts", Static)
        assert str(artifacts_row.renderable) == "Artifacts: Chatbook available"


@pytest.mark.asyncio
async def test_inspector_action_change_recomposes_widget():
    app = InspectorHarness(_base_state())

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)

        new_state = _base_state(
            actions=(
                ConsoleInspectorAction(
                    widget_id="console-inspector-save-chatbook",
                    label="Save Chatbook",
                    enabled=True,
                ),
            )
        )
        inspector.sync_state(new_state)
        await pilot.pause()

        assert inspector.recompose_count == 1
        assert app.query("#console-inspector-save-chatbook")


@pytest.mark.asyncio
async def test_inspector_dictionary_row_text_change_updates_in_place():
    dict_state = _base_state(
        dictionary_rows=(ConsoleDisplayRow("Dictionaries", "2 active"),)
    )
    app = InspectorHarness(dict_state)

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        row_before = inspector.query_one(
            "#console-inspector-dictionaries-row-0", Static
        )

        inspector.sync_state(
            _base_state(
                dictionary_rows=(ConsoleDisplayRow("Dictionaries", "3 active"),)
            )
        )
        await pilot.pause()

        assert inspector.recompose_count == 0
        row_after = inspector.query_one("#console-inspector-dictionaries-row-0", Static)
        assert row_after is row_before
        assert str(row_after.renderable) == "Dictionaries: 3 active"

        # Changing the dictionary section shape is structural.
        inspector.sync_state(_base_state(dictionary_rows=()))
        await pilot.pause()
        assert inspector.recompose_count == 1
        assert len(app.query("#console-inspector-dictionaries-row-0")) == 0


@pytest.mark.asyncio
async def test_inspector_equal_state_sync_is_noop():
    app = InspectorHarness(_base_state())

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        provider_row_before = inspector.query_one("#console-inspector-provider", Static)

        inspector.sync_state(_base_state())
        await pilot.pause()

        assert inspector.recompose_count == 0
        assert (
            inspector.query_one("#console-inspector-provider", Static)
            is provider_row_before
        )
