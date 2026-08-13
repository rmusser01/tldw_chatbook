"""Focused tests for the Console status row's mounted presentations."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_display_state import (
    ConsoleControlState,
    ConsoleRetrievalScopeState,
)
from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusChips


def _state(**overrides) -> ConsoleControlState:
    values = {
        "provider_label": "Provider: Anthropic",
        "model_label": "Model: claude-3-haiku",
        "assistant_label": "Assistant: General",
        "rag_label": "Library search: off",
        "sources_label": "Sources: 0 staged",
        "tools_label": "Tools: —",
        "approvals_label": "Approvals: 0 pending",
        "sources_active": False,
        "tools_active": False,
        "approvals_active": False,
    }
    values.update(overrides)
    return ConsoleControlState(**values)


def _cost_state() -> ConsoleCostState:
    return ConsoleCostState(
        label="$0.42 ● ~+$0.03",
        compact_label="$0.42 ●",
        tooltip="Total: $0.42",
        alert=False,
        cold=False,
    )


def _is_effectively_displayed(widget) -> bool:
    """Return whether the widget and every ancestor are displayed."""
    current = widget
    while current is not None:
        if current.display is False or current.styles.display == "none":
            return False
        current = current.parent
    return True


class _StatusRowApp(App):
    CSS = ".console-control-chip { width: auto; }"

    def __init__(self, *, collapsed: bool = False) -> None:
        super().__init__()
        self._collapsed = collapsed

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(
            _state(),
            collapsed=self._collapsed,
            id="console-status-chips",
        )


@pytest.mark.asyncio
async def test_widget_toggles_mounted_presentations_without_replacing_chips() -> None:
    app = _StatusRowApp()
    async with app.run_test(size=(180, 8)) as pilot:
        await pilot.pause()
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        expanded = app.query_one("#console-status-expanded")
        collapsed = app.query_one("#console-status-collapsed")
        model_chip = app.query_one("#console-model-chip")

        assert strip.collapsed is False
        assert _is_effectively_displayed(expanded)
        assert not _is_effectively_displayed(collapsed)
        assert expanded.query_one("#console-status-collapse", Button)
        assert expanded.query_one("#console-model-chip") is model_chip

        strip.set_collapsed(True)
        strip.set_collapsed(True)
        await pilot.pause()

        assert strip.collapsed is True
        assert not _is_effectively_displayed(expanded)
        assert _is_effectively_displayed(collapsed)
        assert collapsed.query_one("#console-status-expand", Button)
        assert "Status hidden" in str(
            collapsed.query_one("#console-status-collapsed-copy", Static).render()
        )
        assert app.query_one("#console-model-chip") is model_chip


@pytest.mark.asyncio
async def test_widget_constructor_honors_initial_collapsed_state() -> None:
    app = _StatusRowApp(collapsed=True)
    async with app.run_test(size=(180, 8)) as pilot:
        await pilot.pause()
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)

        assert strip.collapsed is True
        assert not _is_effectively_displayed(
            app.query_one("#console-status-expanded")
        )
        assert _is_effectively_displayed(app.query_one("#console-status-collapsed"))


@pytest.mark.asyncio
async def test_widget_preserves_conditional_chip_updates_while_collapsed() -> None:
    app = _StatusRowApp()
    async with app.run_test(size=(200, 8)) as pilot:
        await pilot.pause()
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        model_chip = app.query_one("#console-model-chip")
        strip.set_collapsed(True)
        strip.sync_state(
            replace(
                _state(),
                model_label="Model: updated",
                tools_label="Tools: 4 ready",
                tools_active=True,
            )
        )
        strip.sync_run_chip(True, "Streaming updated response.")
        strip.sync_temporary_chip(True)
        strip.sync_scope_chip(
            ConsoleRetrievalScopeState(
                is_scoped=True,
                item_count=3,
                conv_item_count=3,
            )
        )
        strip.sync_cost_state(_cost_state())
        await pilot.pause()

        assert not _is_effectively_displayed(
            app.query_one("#console-status-expanded")
        )
        assert _is_effectively_displayed(app.query_one("#console-status-collapsed"))
        assert all(
            not _is_effectively_displayed(chip)
            for chip in app.query(".console-control-chip")
        )

        strip.set_collapsed(False)
        await pilot.pause()

        assert app.query_one("#console-model-chip") is model_chip
        expected_copy = {
            "#console-model-chip": "Model: updated",
            "#console-tools-chip": "Tools: 4 ready",
            "#console-run-chip": "Run: Streaming updated response.",
            "#console-temporary-chip": "Temporary",
            "#console-scope-chip": "Scope: 3",
            "#console-cost-chip": "$0.42",
        }
        for selector, copy in expected_copy.items():
            chip = app.query_one(selector)
            assert chip.display is True
            assert _is_effectively_displayed(chip)
            assert copy in str(chip.render())


@pytest.mark.asyncio
async def test_widget_status_toggle_buttons_are_focusable_and_described() -> None:
    app = _StatusRowApp()
    async with app.run_test(size=(180, 8)) as pilot:
        await pilot.pause()
        collapse = app.query_one("#console-status-collapse", Button)
        expand = app.query_one("#console-status-expand", Button)

        assert collapse.can_focus is True
        assert expand.can_focus is True
        assert "Collapse status" in str(collapse.tooltip)
        assert "Expand status" in str(expand.tooltip)
