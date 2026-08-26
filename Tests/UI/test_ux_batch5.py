"""Regression tests for the round-5 UX batch (UX-045/046/054, 077, 043)."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench

CSS_BUNDLE = Path("tldw_chatbook/css/tldw_cli_modular.tcss")


# UX-077 -----------------------------------------------------------------
def test_queue_keyboard_ops_bound_and_implemented() -> None:
    keys = {binding.key for binding in SchedulesWorkbench.BINDINGS}
    assert {"c", "e", "space", "d", "s"} <= keys
    for binding in SchedulesWorkbench.BINDINGS:
        assert hasattr(SchedulesWorkbench, f"action_{binding.action}")


def test_footer_hints_cover_queue_keyboard_ops() -> None:
    hint_keys = {key for key, _label in SchedulesWorkbench.SCHEDULES_SHORTCUTS}
    binding_keys = {
        binding.key
        for binding in SchedulesWorkbench.BINDINGS
        if binding.key != "escape"
    }
    assert hint_keys == binding_keys


# UX-043 -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_recovery_callout_shows_when_service_unavailable() -> None:
    from tldw_chatbook.UI.Workbench.workbench_widgets import RecoveryCallout

    class Harness(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield Static()

    from textual.widgets import Static

    app = Harness()
    async with app.run_test(size=(120, 36)) as pilot:
        # No scheduling_service attribute on this app instance.
        await app.push_screen(SchedulesWorkbench(app_instance=app))
        await pilot.pause()
        callout = app.screen.query_one("#scheduling-recovery", RecoveryCallout)
        assert callout.display is True
        text = callout.renderable.plain
        assert "Scheduling service unavailable" in text


# UX-046 -----------------------------------------------------------------
def test_writing_window_has_vertical_override_in_bundle() -> None:
    bundle = CSS_BUNDLE.read_text()
    assert "#writing-window" in bundle
    assert "layout: vertical" in bundle.split("#writing-window", 1)[1][:200]


# UX-054 -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_llamacpp_actions_above_the_fold() -> None:
    """Start/Stop appear before the path inputs in the llama.cpp view."""
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

    class Harness(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LLMManagementWindow(None)

    app = Harness()
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        view = app.query_one("#llm-view-llama-cpp")
        children = list(view.children)
        button_row = next(
            i for i, w in enumerate(children) if w.has_class("button_container")
        )
        first_input_row = next(
            i for i, w in enumerate(children) if w.has_class("input_container")
        )
        assert button_row < first_input_row, (
            "Start/Stop must precede the path fields so they render above the fold"
        )
