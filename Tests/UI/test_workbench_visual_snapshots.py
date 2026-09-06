"""Visual smoke gates for the Console Workbench shell."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from html import unescape
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from textual.containers import Horizontal
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
    rendered_text = _rendered_svg_text(svg)
    assert normalized_svg.count("Provider:") == 1
    assert normalized_svg.count("Model:") == 1
    assert normalized_svg.count("Assistant:") == 1
    assert "Library search:" in normalized_svg
    assert "Sources:" in normalized_svg
    assert "Approvals:" in normalized_svg
    assert "Settings" in normalized_svg
    assert "Attach" in normalized_svg
    assert "Search Library" in normalized_svg
    assert "Model: gpt-5.6-terra" in rendered_text
    assert "Send disabled" not in normalized_svg
    assert "Setup required" not in normalized_svg


def _assert_console_inspector_evidence(svg: str) -> None:
    normalized_svg = unescape(svg).replace("\xa0", " ")
    rendered_text = _rendered_svg_text(svg)
    assert "Inspector" in normalized_svg
    assert "Run recipe" in normalized_svg
    assert "Blocked impact" in normalized_svg
    assert "Send disabled" not in normalized_svg
    assert "Setup required" not in normalized_svg
    assert rendered_text.index("Run recipe") < rendered_text.index("Blocked impact")


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
    rendered_text = _rendered_svg_text(svg)
    assert "Console Workbench Focus State" in normalized_svg
    assert "Settings" in normalized_svg
    assert "Provider:" in normalized_svg
    assert "Model: gpt-5.6-terra" in rendered_text


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
    # task-17651: grid children suppress their bottom edge — the grid's
    # own bottom border closes the workbench frame.
    assert border.bottom[0] in {"", "none"}
    assert border.left[0] == "solid"


def _assert_interior_handle_border(widget, edge: str) -> None:
    """Collapsed rail handles own only their interior divider edge."""
    border = widget.styles.border
    for candidate in ("top", "right", "bottom", "left"):
        style = getattr(border, candidate)[0]
        assert style == ("solid" if candidate == edge else "")


def _painted_region_rows(screen, region) -> list[str]:
    """Return non-empty text rows painted inside an exact screen region."""
    strips = list(screen._compositor.render_strips())
    return [
        text
        for y in range(region.y, region.bottom)
        if (text := strips[y].crop(region.x, region.right).text.strip())
    ]


def _painted_center_row(screen, region) -> str:
    """Return the direct compositor row through the center of a control."""
    strip = list(screen._compositor.render_strips())[
        region.y + (region.height - 1) // 2
    ]
    return strip.crop(region.x, region.right).text.strip()


def _rendered_svg_text(svg: str) -> str:
    """Rejoin adjacent rendered text nodes without crossing rows or gaps."""
    rows: dict[float, list[tuple[float, float, str]]] = {}
    for attributes, raw_text in re.findall(
        r"<text\b([^>]*)>([^<]*)</text>", svg, flags=re.DOTALL
    ):
        parsed_attributes = dict(re.findall(r'(\w+)="([^"]*)"', attributes))
        if "clip-path" not in attributes:
            continue
        x = float(parsed_attributes["x"])
        y = float(parsed_attributes["y"])
        text_length = float(parsed_attributes["textLength"])
        text = unescape(raw_text).replace("\xa0", " ")
        rows.setdefault(y, []).append((x, text_length, text))

    rendered_rows = []
    for y in sorted(rows):
        rendered_row = ""
        previous_right: float | None = None
        for x, text_length, text in sorted(rows[y]):
            if previous_right is not None and x > previous_right + 0.01:
                rendered_row += " "
            rendered_row += text
            previous_right = max(previous_right or x, x + text_length)
        rendered_rows.append(rendered_row)
    return "\n".join(rendered_rows)


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
            send = app.screen.query_one("#console-send-message", Button)
            assert send.render_line(0).text.strip() == "Send"
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
            assert _painted_region_rows(screen, inspector_button.region) == [
                "◂ Inspect"
            ]
            assert inspector_button.label == "◂ Inspect"
            assert inspector_button.tooltip == "Open Inspector rail"
            assert workspace.content_region.contains_region(inspector_handle.region), (
                f"Inspector handle escapes workspace at {size}: "
                f"handle={inspector_handle.region}, "
                f"workspace={workspace.content_region}"
            )
            assert inspector_handle.region.height == workspace.content_region.height
            assert inspector_handle.region.width == 11
            assert inspector_handle.content_region.width == 10
            assert inspector_handle.styles.background.a > 0
            _assert_interior_handle_border(inspector_handle, "left")
            assert transcript.region.width > 0

            assert (
                inspector_handle.styles.background == context_handle.styles.background
            )
            inspector_border = inspector_handle.styles.border
            context_border = context_handle.styles.border
            assert inspector_border.top == context_border.top
            assert inspector_border.bottom == context_border.bottom
            assert inspector_border.right == context_border.left
            assert inspector_border.left == context_border.right

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


@pytest.mark.parametrize(
    (
        "size",
        "target_state",
        "context_open",
        "inspector_open",
        "expected_rail_widths",
    ),
    (
        ((140, 42), "context-open-inspector-collapsed", True, False, (30, 0)),
        ((140, 42), "inspector-priority-after-both-open", True, True, (0, 34)),
        ((140, 42), "context-collapsed-inspector-open", False, True, (0, 34)),
        ((140, 42), "both-collapsed", False, False, (0, 0)),
        ((160, 45), "context-open-inspector-collapsed", True, False, (30, 0)),
        ((160, 45), "both-open", True, True, (30, 34)),
        ((160, 45), "context-collapsed-inspector-open", False, True, (0, 35)),
        ((160, 45), "both-collapsed", False, False, (0, 0)),
    ),
)
@pytest.mark.asyncio
async def test_task_16001_console_directional_rail_buttons_visual_sweep(
    size: tuple[int, int],
    target_state: str,
    context_open: bool,
    inspector_open: bool,
    expected_rail_widths: tuple[int, int],
) -> None:
    """Verify directional rail controls across viewport and visibility states.

    Args:
        size: Terminal viewport dimensions.
        target_state: Human-readable rail-state identifier.
        context_open: Whether the Context rail should be open.
        inspector_open: Whether the Inspector rail should be open.
        expected_rail_widths: Expected Context and Inspector rail widths.
    """
    app = _build_test_app(configured_default="home")
    _mark_console_onboarding_complete(app)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=size) as pilot:
            _configure_native_ready_console(app)
            await _open_console(app, pilot)

            async def drive_rail(
                *,
                rail_selector: str,
                open_selector: str,
                collapse_selector: str,
                target_open: bool,
            ) -> None:
                if bool(app.screen.query_one(rail_selector).display) == target_open:
                    return
                selector = open_selector if target_open else collapse_selector
                button = app.screen.query_one(selector, Button)
                assert button.display is True
                assert await pilot.click(button)
                await _wait_until(
                    pilot,
                    lambda: (
                        bool(app.screen.query_one(rail_selector).display) == target_open
                    ),
                    context=f"{rail_selector} target state {target_open}",
                )
                await pilot.pause(0.2)

            def assert_control_preconditions(
                *,
                rail_selector: str,
                handle_selector: str,
                button_selector: str,
                open_state: bool,
                handle_width: int,
                content_width: int,
            ) -> tuple[Button, Horizontal | None]:
                screen = app.screen
                workspace = screen.query_one("#console-workspace-grid")
                rail = screen.query_one(rail_selector)
                handle = screen.query_one(handle_selector)
                button = screen.query_one(button_selector, Button)

                assert rail.display is open_state
                assert handle.display is (not open_state)
                owner = rail if open_state else handle
                assert workspace.content_region.contains_region(owner.region)
                assert owner.region.width > 0
                assert button.region.width > 0

                header = button.parent if open_state else None
                if open_state:
                    assert isinstance(header, Horizontal)
                    assert rail.content_region.contains_region(header.region)
                    assert header.region.height == 1
                    assert header.content_region.contains_region(button.region)
                    assert button.region.height == 1
                else:
                    assert handle.region.width == handle_width
                    assert handle.content_region.width == content_width
                    assert handle.content_region.contains_region(button.region)
                assert len(_painted_region_rows(screen, button.region)) == 1
                return button, header

            await drive_rail(
                rail_selector="#console-left-rail",
                open_selector="#console-context-rail-open",
                collapse_selector="#console-context-rail-collapse",
                target_open=context_open,
            )
            await drive_rail(
                rail_selector="#console-right-rail",
                open_selector="#console-inspector-rail-open",
                collapse_selector="#console-inspector-rail-collapse",
                target_open=inspector_open,
            )
            await pilot.pause(0.5)

            effective_context_open = expected_rail_widths[0] > 0
            effective_inspector_open = expected_rail_widths[1] > 0

            assert (
                app.screen.query_one("#console-left-rail").region.width,
                app.screen.query_one("#console-right-rail").region.width,
            ) == expected_rail_widths

            context_selector = (
                "#console-context-rail-collapse"
                if effective_context_open
                else "#console-context-rail-open"
            )
            inspector_selector = (
                "#console-inspector-rail-collapse"
                if effective_inspector_open
                else "#console-inspector-rail-open"
            )
            context_button, context_header = assert_control_preconditions(
                rail_selector="#console-left-rail",
                handle_selector="#console-context-rail-handle",
                button_selector=context_selector,
                open_state=effective_context_open,
                handle_width=13,
                content_width=12,
            )
            inspector_button, inspector_header = assert_control_preconditions(
                rail_selector="#console-right-rail",
                handle_selector="#console-inspector-rail-handle",
                button_selector=inspector_selector,
                open_state=effective_inspector_open,
                handle_width=11,
                content_width=10,
            )
            transcript = app.screen.query_one("#console-transcript-region")
            assert transcript.region.width > 0

            svg = app.export_screenshot(
                title=(
                    f"TASK-16001 Directional Rails {target_state} {size[0]}x{size[1]}"
                ),
                simplify=True,
            )
            _assert_svg_healthy(svg)
            rendered_text = _rendered_svg_text(svg)
            assert rendered_text.strip()
            assert "Console" in rendered_text

            # TASK-23195 and its follow-up replaced the two ASCII-art header
            # labels with a name plus one resolved glyph, mirrored across the
            # rails: the glyph sits on the edge adjacent to the transcript,
            # pointing the way that rail leaves. The COLLAPSED handles keep
            # their compact ASCII forms, which are unchanged.
            context_label = "Context ◂" if effective_context_open else "Context ▸"
            inspector_label = (
                "▸ Inspect" if effective_inspector_open else "◂ Inspect"
            )
            context_tooltip = (
                "Collapse Console context rail"
                if effective_context_open
                else "Open Context rail"
            )
            inspector_tooltip = (
                "Collapse Inspector rail"
                if effective_inspector_open
                else "Open Inspector rail"
            )

            assert (
                str(context_button.label),
                str(inspector_button.label),
            ) == (context_label, inspector_label)
            assert (
                _painted_center_row(app.screen, context_button.region),
                _painted_center_row(app.screen, inspector_button.region),
            ) == (context_label, inspector_label)
            assert context_button.tooltip == context_tooltip
            assert inspector_button.tooltip == inspector_tooltip

            if effective_context_open:
                assert context_header is not None
                assert not app.screen.query("#console-context-rail-title")
                assert list(context_header.children) == [context_button]
                assert (
                    context_button.region.width == context_header.content_region.width
                )
            if effective_inspector_open:
                assert inspector_header is not None
                assert not app.screen.query("#console-inspector-rail-title")
                assert list(inspector_header.children) == [inspector_button]
                assert (
                    inspector_button.region.width
                    == inspector_header.content_region.width
                )
            assert context_label in rendered_text
            assert inspector_label in rendered_text


@pytest.mark.asyncio
async def test_console_workbench_standard_width_inspector_snapshot() -> None:
    app = _build_test_app()
    _mark_console_onboarding_complete(app)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(128, 40)) as pilot:
            await _open_console(app, pilot)

            # TASK-23197: the Inspector no longer opens ITSELF at 118-128
            # columns. That automatic open tripped priority resolution and
            # evicted the Context rail, so a one-column resize swapped which
            # sidebar the user had. Open it the way a user now does; the
            # evidence this test captures -- the Inspector's next-action row
            # at standard width -- is unchanged.
            assert await pilot.click("#console-inspector-rail-open")
            await pilot.pause(0.3)

            right_rail = app.screen.query_one("#console-right-rail")
            assert right_rail.display is True
            assert right_rail.region.width > 0
            next_action = app.screen.query_one("#console-inspector-next-action", Static)
            assert next_action.render_line(0).text.strip() == (
                "Next action: Set up provider"
            )
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
