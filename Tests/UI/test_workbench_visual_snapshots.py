"""Visual smoke gates for the Console Workbench shell."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from html import unescape
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

if TYPE_CHECKING:
    from textual.pilot import Pilot


BROKEN_TEXT_PATTERNS = (
    "Traceback",
    "Unhandled exception",
    "Unable to mount",
    "Internal Error",
)
RAW_OBJECT_REPR = re.compile(r"<[\w.]+ object at 0x[0-9a-fA-F]+>")


def _test_cli_setting(section: str, key: str | None = None, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


async def _wait_until(
    pilot: "Pilot",
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 5.0,
    interval_seconds: float = 0.05,
    context: str = "condition",
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"Timed out waiting for {context}")


async def _open_console(app, pilot: "Pilot") -> None:
    if app.current_tab != "chat" or app.screen.__class__.__name__ != "ChatScreen":
        await app.handle_screen_navigation(NavigateToScreen("chat"))
    await _wait_until(
        pilot,
        lambda: app.current_tab == "chat"
        and app.screen.__class__.__name__ == "ChatScreen",
        context="Console screen",
    )
    await _wait_for_selector(app.screen, pilot, "#console-shell")


def _assert_svg_healthy(svg: str) -> None:
    assert "<svg" in svg
    assert "</svg>" in svg
    assert len(svg) > 1_000
    for broken in BROKEN_TEXT_PATTERNS:
        assert broken not in svg
    assert RAW_OBJECT_REPR.search(svg) is None


def _assert_console_density_evidence(svg: str) -> None:
    normalized_svg = unescape(svg).replace("\xa0", " ")
    assert "Provider:" in normalized_svg
    assert "Model:" in normalized_svg
    assert "Assistant:" in normalized_svg or "Persona:" in normalized_svg
    assert "RAG:" in normalized_svg
    assert "Sources:" in normalized_svg
    assert "Tools:" in normalized_svg
    assert "Approvals:" in normalized_svg
    assert "Settings" in normalized_svg
    assert "Attach" in normalized_svg
    assert "Library RAG" in normalized_svg


@pytest.mark.parametrize("density", ("normal", "compact"))
@pytest.mark.asyncio
async def test_console_workbench_normal_and_compact_snapshots(density: str) -> None:
    app = _build_test_app()
    app.app_config = getattr(app, "app_config", {}) or {}
    app.app_config.setdefault("appearance", {})["ui_density"] = density

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 42)) as pilot:
            await _open_console(app, pilot)

            shell = app.screen.query_one("#console-shell")
            assert shell.has_class(f"density-{density}")
            svg = app.export_screenshot(
                title=f"Console Workbench {density}",
                simplify=True,
            )
            _assert_svg_healthy(svg)
            _assert_console_density_evidence(svg)


@pytest.mark.asyncio
async def test_console_workbench_command_palette_snapshot() -> None:
    app = _build_test_app()

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 42)) as pilot:
            await _open_console(app, pilot)
            await pilot.press("ctrl+p")
            await pilot.pause()

            stack_names = {
                screen.__class__.__name__.lower() for screen in app.screen_stack
            }
            assert any("command" in name and "palette" in name for name in stack_names)
            _assert_svg_healthy(
                app.export_screenshot(
                    title="Console Workbench Command Palette",
                    simplify=True,
                )
            )


@pytest.mark.asyncio
async def test_console_workbench_focus_state_snapshot() -> None:
    app = _build_test_app()

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 42)) as pilot:
            await _open_console(app, pilot)
            settings_action = app.screen.query_one("#workbench-action-settings", Button)
            settings_action.focus()
            await pilot.pause()

            assert app.focused is settings_action
            _assert_svg_healthy(
                app.export_screenshot(
                    title="Console Workbench Focus State",
                    simplify=True,
                )
            )
