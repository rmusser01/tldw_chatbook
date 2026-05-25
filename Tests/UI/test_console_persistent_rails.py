"""Mounted Console persistent rail first-start contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)


def _is_displayed(widget) -> bool:
    current = widget
    while current is not None:
        if current.display is False or current.styles.display == "none":
            return False
        current = getattr(current, "parent", None)
    return True


def _assert_selector_hidden_or_absent(screen, selector: str) -> None:
    matches = list(screen.query(selector))
    assert not matches or all(not _is_displayed(widget) for widget in matches)


def test_generated_console_stylesheet_includes_rail_rules():
    stylesheet = Path("tldw_chatbook/css/tldw_cli_modular.tcss")
    css = stylesheet.read_text(encoding="utf-8")

    for selector in (
        "#console-right-rail",
        ".console-rail-handle",
        ".console-rail-header",
        ".console-rail-collapse-button",
    ):
        assert selector in css


@pytest.mark.asyncio
async def test_console_first_start_renders_left_rail_and_right_handle():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

        assert _is_displayed(console.query_one("#console-left-rail"))
        assert _is_displayed(console.query_one("#console-staged-context-tray"))
        assert _is_displayed(console.query_one("#console-workspace-context"))
        _assert_selector_hidden_or_absent(console, "#console-context-rail-handle")
        _assert_selector_hidden_or_absent(console, "#console-right-rail")
        _assert_selector_hidden_or_absent(console, "#console-run-inspector-state")
        _assert_selector_hidden_or_absent(
            console,
            "#console-live-work-source-readiness",
        )
        assert _is_displayed(console.query_one("#console-inspector-rail-handle"))
        assert "Inspector" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_first_start_does_not_create_rail_state_config_on_read():
    app = _build_test_app()
    console_config = app.app_config.setdefault("console", {})
    console_config.pop("rail_state", None)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")

    assert "rail_state" not in console_config


@pytest.mark.asyncio
async def test_console_first_start_right_handle_is_focusable():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-open")

        for _ in range(80):
            focused = console.focused
            if getattr(focused, "id", None) == "console-inspector-rail-open":
                assert isinstance(focused, Button)
                return
            await pilot.press("tab")

        focused_id = getattr(console.focused, "id", None)
        raise AssertionError(
            "console-inspector-rail-open was not reachable by tab; "
            f"focused={focused_id!r}"
        )
