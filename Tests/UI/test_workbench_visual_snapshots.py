"""Visual smoke gates for the Console Workbench shell."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from html import unescape
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from textual.widgets import Button, OptionList, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
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


def _mark_console_onboarding_complete(app) -> None:
    app.app_config = getattr(app, "app_config", {}) or {}
    console_config = app.app_config.setdefault("console", {})
    onboarding = console_config.setdefault("onboarding", {})
    onboarding["first_send_completed"] = True


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
        lambda: (
            app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen"
        ),
        context="Console screen",
    )
    await _wait_until(
        pilot,
        lambda: bool(app.screen.query("#console-shell")),
        context="Console shell",
    )


def _assert_svg_healthy(svg: str) -> None:
    assert "<svg" in svg
    assert "</svg>" in svg
    assert len(svg) > 1_000
    for broken in BROKEN_TEXT_PATTERNS:
        assert broken not in svg
    assert RAW_OBJECT_REPR.search(svg) is None


def _assert_console_density_evidence(svg: str) -> None:
    normalized_svg = unescape(svg).replace("\xa0", " ")
    assert normalized_svg.count("Provider:") == 1
    assert normalized_svg.count("Model:") == 1
    assert normalized_svg.count("Assistant:") == 1
    assert "Library search:" in normalized_svg
    assert "Sources:" in normalized_svg
    assert "Approvals:" in normalized_svg
    assert "Settings" in normalized_svg
    assert "Attach" in normalized_svg
    assert "Search Library" in normalized_svg
    assert "Model: not selected" in normalized_svg
    assert "Send disabled" not in normalized_svg
    assert "Setup required" not in normalized_svg


def _assert_console_inspector_evidence(svg: str) -> None:
    normalized_svg = unescape(svg).replace("\xa0", " ")
    assert "Inspector" in normalized_svg
    assert "Status: Blocked" in normalized_svg
    assert "Run recipe" in normalized_svg
    assert "Blocked impact" in normalized_svg
    assert "Next action" in normalized_svg
    assert "Choose provider" in normalized_svg
    assert "Provider: blocked" in normalized_svg
    assert "Send disabled" not in normalized_svg
    assert "Setup required" not in normalized_svg
    assert (
        normalized_svg.index("Status: Blocked")
        < normalized_svg.index("Run recipe")
        < normalized_svg.index("Blocked impact")
        < normalized_svg.index("Next action")
        < normalized_svg.index("Provider: blocked")
    )


def _assert_command_palette_evidence(svg: str) -> None:
    # The palette's match highlighting splits matched characters into separate
    # <text> elements, so a command name is never contiguous in the raw SVG.
    # Rejoin the text elements in document order before searching.
    joined = "".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    normalized_svg = unescape(joined).replace("\xa0", " ")
    assert "Console Workbench Command Palette" in normalized_svg
    assert any(
        command in normalized_svg
        for command in (
            "New Chat Conversation",
            "Search All Content",
            "Import Media File",
            "Switch to Console",
            "Switch to Home",
        )
    )


def _assert_console_focus_evidence(svg: str) -> None:
    normalized_svg = unescape(svg).replace("\xa0", " ")
    assert "Console Workbench Focus State" in normalized_svg
    assert "Settings" in normalized_svg
    assert "Provider:" in normalized_svg
    assert "Model: not selected" in normalized_svg


def _assert_visible_ancestors(widget) -> None:
    current = widget
    while current is not None:
        assert current.display is not False
        assert current.styles.display != "none"
        current = getattr(current, "parent", None)


def _assert_solid_border(widget) -> None:
    border = widget.styles.border
    assert border.top[0] == "solid"
    assert border.right[0] == "solid"
    assert border.bottom[0] == "solid"
    assert border.left[0] == "solid"


def _painted_region_rows(screen, region) -> list[str]:
    """Return non-empty text rows painted inside an exact screen region."""
    strips = list(screen._compositor.render_strips())
    return [
        text
        for y in range(region.y, region.bottom)
        if (text := strips[y].crop(region.x, region.right).text.strip())
    ]


@pytest.mark.parametrize("density", ("normal", "compact"))
@pytest.mark.asyncio
async def test_console_workbench_normal_and_compact_snapshots(density: str) -> None:
    app = _build_test_app()
    app.app_config = getattr(app, "app_config", {}) or {}
    app.app_config.setdefault("appearance", {})["ui_density"] = density
    _mark_console_onboarding_complete(app)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(160, 42)) as pilot:
            await _open_console(app, pilot)

            shell = app.screen.query_one("#console-shell")
            assert shell.has_class(f"density-{density}")
            svg = app.export_screenshot(
                title=f"Console Workbench {density}",
                simplify=True,
            )
            _assert_svg_healthy(svg)
            _assert_console_density_evidence(svg)


@pytest.mark.parametrize("size", ((130, 30), (140, 42), (160, 45)))
@pytest.mark.parametrize("approval_count", (0, 3))
@pytest.mark.asyncio
async def test_task_15783_console_collapsed_inspector_rail_visual_parity_sweep(
    size: tuple[int, int], approval_count: int
) -> None:
    app = _build_test_app(configured_default="home")
    app.console_pending_approval_count = approval_count
    _mark_console_onboarding_complete(app)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=size) as pilot:
            _configure_native_ready_console(app)
            await _open_console(app, pilot)
            await _wait_until(
                pilot,
                lambda: (
                    app.screen.query_one("#console-workspace-grid").region.height > 0
                    and app.screen.query_one(
                        "#console-inspector-rail-handle"
                    ).region.width
                    > 0
                ),
                context=f"collapsed Inspector handle at {size}",
            )
            await pilot.pause(0.5)

            screen = app.screen
            workspace = screen.query_one("#console-workspace-grid")
            context_handle = screen.query_one("#console-context-rail-handle")
            inspector_handle = screen.query_one("#console-inspector-rail-handle")
            inspector_button = screen.query_one("#console-inspector-rail-open", Button)
            transcript = screen.query_one("#console-transcript-region")

            assert inspector_handle.display is True
            assert _painted_region_rows(screen, inspector_button.region) == ["Inspect-->"]
            assert inspector_button.label == "Inspect-->"
            assert inspector_button.tooltip == "Open Inspector rail"
            assert workspace.content_region.contains_region(inspector_handle.region), (
                f"Inspector handle escapes workspace at {size}: "
                f"handle={inspector_handle.region}, "
                f"workspace={workspace.content_region}"
            )
            assert inspector_handle.region.height == workspace.content_region.height
            assert inspector_handle.region.width == 11
            assert inspector_handle.content_region.width == 9
            assert inspector_handle.styles.background.a > 0
            _assert_solid_border(inspector_handle)
            assert transcript.region.width > 0

            assert (
                inspector_handle.styles.background == context_handle.styles.background
            )
            inspector_border = inspector_handle.styles.border
            context_border = context_handle.styles.border
            assert inspector_border.top == context_border.top
            assert inspector_border.right == context_border.right
            assert inspector_border.bottom == context_border.bottom
            assert inspector_border.left == context_border.left

            badge_rows = list(screen.query("#console-inspector-rail-badge"))
            available_bottom = inspector_handle.content_region.bottom
            if approval_count:
                assert len(badge_rows) == 1
                badge = screen.query_one("#console-inspector-rail-badge", Static)
                assert str(badge.renderable) == "3 appr"
                assert inspector_button.region.bottom <= badge.region.y
                assert (
                    inspector_button.region.height
                    == inspector_handle.content_region.height - badge.region.height
                )
                assert badge.region.right <= inspector_handle.content_region.right
                assert badge.region.bottom <= inspector_handle.content_region.bottom
                available_bottom = badge.region.y
            else:
                assert badge_rows == []

            assert inspector_button.region.x >= inspector_handle.content_region.x
            assert (
                inspector_button.region.right <= inspector_handle.content_region.right
            )
            assert inspector_button.region.y >= inspector_handle.content_region.y
            assert inspector_button.region.bottom <= available_bottom
            assert (
                inspector_button.region.x + inspector_button.region.right
                == inspector_handle.content_region.x
                + inspector_handle.content_region.right
            )
            assert (
                inspector_button.region.y + inspector_button.region.bottom
                == inspector_handle.content_region.y + available_bottom
            )

            svg = app.export_screenshot(
                title=(
                    f"TASK-15783 Inspector Rail Parity {size[0]}x{size[1]} "
                    f"approvals={approval_count}"
                ),
                simplify=True,
            )
            _assert_svg_healthy(svg)


@pytest.mark.asyncio
async def test_console_workbench_standard_width_inspector_snapshot() -> None:
    app = _build_test_app()
    _mark_console_onboarding_complete(app)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(128, 40)) as pilot:
            await _open_console(app, pilot)

            right_rail = app.screen.query_one("#console-right-rail")
            assert right_rail.display is True
            assert right_rail.region.width > 0
            svg = app.export_screenshot(
                title="Console Workbench Standard Width Inspector",
                simplify=True,
            )
            _assert_svg_healthy(svg)
            _assert_console_inspector_evidence(svg)


@pytest.mark.asyncio
async def test_console_workbench_command_palette_snapshot() -> None:
    app = _build_test_app()
    _mark_console_onboarding_complete(app)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 42)) as pilot:
            await _open_console(app, pilot)
            await pilot.press("ctrl+p")
            await pilot.pause()
            # The palette's discover() hit order and its match-highlight text
            # splitting are both nondeterministic in the exported SVG, so assert
            # command evidence on the option list itself (deterministic) and use
            # the SVG only for the visual health check.
            for character in "New Chat":
                await pilot.press(character)
            palette = app.screen_stack[-1]
            option_list = palette.query_one(OptionList)
            for _ in range(60):
                await pilot.pause()
                prompts = [
                    str(option_list.get_option_at_index(index).prompt)
                    for index in range(option_list.option_count)
                ]
                if any("New Chat Conversation" in prompt for prompt in prompts):
                    break
            assert any("New Chat Conversation" in prompt for prompt in prompts), prompts

            stack_names = {
                screen.__class__.__name__.lower() for screen in app.screen_stack
            }
            assert any("command" in name and "palette" in name for name in stack_names)
            svg = app.export_screenshot(
                title="Console Workbench Command Palette",
                simplify=True,
            )
            _assert_svg_healthy(svg)


@pytest.mark.asyncio
async def test_console_workbench_focus_state_snapshot() -> None:
    app = _build_test_app()
    _mark_console_onboarding_complete(app)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 42)) as pilot:
            await _open_console(app, pilot)
            # Let the first 0.2-second Console state-sync replace the initial
            # control bar before choosing a focus target. Focusing the
            # pre-sync instance tests a widget that production immediately
            # retires and correctly sends focus back to the composer.
            await pilot.pause(0.5)
            settings_action = app.screen.query_one("#console-control-settings", Button)
            settings_action.focus()
            await pilot.pause(0.05)

            assert app.focused is settings_action
            assert settings_action.region.width > 0
            assert settings_action.region.height > 0
            _assert_visible_ancestors(settings_action)
            svg = app.export_screenshot(
                title="Console Workbench Focus State",
                simplify=True,
            )
            _assert_svg_healthy(svg)
            _assert_console_focus_evidence(svg)
