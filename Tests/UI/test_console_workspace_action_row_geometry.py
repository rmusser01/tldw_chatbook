"""Geometry regression for the Console workspace action row (TASK-712).

The Session action row once clipped the New button entirely out of view while
leaving a blank clickable strip: the row's left margin plus Textual's default
16-column Button min-width overflowed the ~37-column rail, so the New label
rendered outside the clip. These tests mount the real ChatScreen with the real
app stylesheet (the defect lives in app-tier CSS, not widget DEFAULT_CSS) and
assert every workspace action button fits inside its row's content region.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"


class StyledConsoleHarness(ConsoleHarness):
    """ConsoleHarness with the shipped stylesheet so app-tier rules apply."""

    CSS_PATH = str(BUNDLE)


@pytest.mark.asyncio
async def test_workspace_action_buttons_fit_inside_their_rows() -> None:
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        await pilot.pause()

        tray = console.query_one("#console-workspace-context")
        checks = [
            ("#console-workspace-action-row", "#console-change-workspace"),
            ("#console-workspace-action-row", "#console-new-workspace"),
            ("#console-workspace-rag-scope-row", "#console-workspace-rag-scope-open"),
        ]
        for row_selector, button_selector in checks:
            row = console.query_one(row_selector)
            button = console.query_one(button_selector)
            assert button.display, f"{button_selector} must be displayed"
            assert button.region.width > 0, f"{button_selector} has no width"
            assert button.region.right <= row.content_region.right, (
                f"{button_selector} overflows its row: button right edge "
                f"{button.region.right} > row content right "
                f"{row.content_region.right} - the overflowed part renders "
                "outside the rail clip (invisible but clickable)."
            )
            assert row.region.right <= tray.content_region.right, (
                f"{row_selector} overflows the tray content region"
            )


@pytest.mark.asyncio
async def test_workspace_action_row_holds_switch_and_new_side_by_side() -> None:
    """Both actions must coexist on the shared row without either collapsing."""
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        await pilot.pause()

        switch = console.query_one("#console-change-workspace")
        new = console.query_one("#console-new-workspace")
        # Side by side on one line, no overlap, both wide enough for their labels.
        assert switch.region.y == new.region.y
        assert switch.region.right <= new.region.x
        assert switch.region.width >= len("Switch")
        assert new.region.width >= len("New")
