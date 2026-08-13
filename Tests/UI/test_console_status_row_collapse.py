"""Focused tests for the Console status row's mounted presentations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import pytest
from textual.app import App, ComposeResult
from textual.containers import HorizontalScroll
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_display_state import (
    ConsoleControlState,
    ConsoleRetrievalScopeState,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusChips


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CSS_ROOT = _REPO_ROOT / "tldw_chatbook/css"
_AGENTIC_SOURCE = _CSS_ROOT / "components/_agentic_terminal.tcss"
_BUNDLED_STYLESHEET = _CSS_ROOT / "tldw_cli_modular.tcss"


def _rule_body(css: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>[^{{}}]*)\}}", css)
    assert match is not None, f"Missing CSS rule for {selector}"
    return match.group("body")


def _assert_declarations(body: str, *declarations: str) -> None:
    for declaration in declarations:
        assert declaration in body


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
    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, *, collapsed: bool = False) -> None:
        super().__init__()
        self._collapsed = collapsed

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(
            _state(),
            collapsed=self._collapsed,
            id="console-status-chips",
        )


def _ready_console_host() -> tuple[ConsoleHarness, object]:
    app = _build_test_app()
    _configure_native_ready_console(app)
    return ConsoleHarness(app), app


async def _mounted_console(host: ConsoleHarness, pilot) -> ChatScreen:
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-status-chips")
    return console


def _assert_inside(widget, viewport) -> None:
    region = widget.region
    assert region.width > 0
    assert region.height == 1
    assert region.x >= viewport.x
    assert region.right <= viewport.right


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 32), (140, 42)])
async def test_collapsed_status_row_stays_one_line_and_left_anchored(
    size: tuple[int, int],
) -> None:
    app = _StatusRowApp(collapsed=True)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        collapsed = app.query_one("#console-status-collapsed")
        expand = app.query_one("#console-status-expand", Button)
        copy = app.query_one("#console-status-collapsed-copy", Static)

        viewport = strip.content_region
        assert strip.region.height == 1
        assert collapsed.region.height == 1
        assert collapsed.region.x == viewport.x
        assert expand.region.x == collapsed.content_region.x
        _assert_inside(expand, viewport)
        _assert_inside(copy, viewport)
        assert expand.region.right <= copy.region.x


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 32), (140, 42)])
async def test_expanded_status_row_keeps_toggle_left_and_scroller_in_viewport(
    size: tuple[int, int],
) -> None:
    app = _StatusRowApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        expanded = app.query_one("#console-status-expanded")
        collapse = app.query_one("#console-status-collapse", Button)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)

        viewport = strip.content_region
        assert strip.region.height == 1
        assert expanded.region.height == 1
        assert expanded.region.x == viewport.x
        assert collapse.region.x == expanded.content_region.x
        _assert_inside(collapse, viewport)
        _assert_inside(scroller, viewport)
        assert collapse.region.right <= scroller.region.x
        assert scroller.region.right == viewport.right


def test_status_row_stylesheet_contract_is_in_source_and_bundle() -> None:
    for stylesheet in (_AGENTIC_SOURCE, _BUNDLED_STYLESHEET):
        css = stylesheet.read_text(encoding="utf-8")

        host = _rule_body(css, "#console-status-chips")
        _assert_declarations(
            host,
            "width: 100%;",
            "min-width: 0;",
            "height: 1;",
            "min-height: 1;",
            "max-height: 1;",
            "layout: horizontal;",
            "border: none;",
            "padding: 0 1;",
        )

        presentations = _rule_body(
            css, "#console-status-expanded,\n#console-status-collapsed"
        )
        _assert_declarations(
            presentations,
            "width: 100%;",
            "min-width: 0;",
            "height: 1;",
            "min-height: 1;",
            "max-height: 1;",
            "layout: horizontal;",
            "align: left middle;",
        )

        scroller = _rule_body(css, "#console-status-chip-scroll")
        _assert_declarations(
            scroller,
            "width: 1fr;",
            "min-width: 0;",
            "height: 1;",
            "min-height: 1;",
            "max-height: 1;",
            "layout: horizontal;",
            "overflow-x: auto;",
            "overflow-y: hidden;",
            "border: none;",
            "padding: 0;",
            "scrollbar-size-horizontal: 0;",
        )

        toggles = _rule_body(
            css, "#console-status-collapse,\n#console-status-expand"
        )
        _assert_declarations(
            toggles,
            "width: 9;",
            "min-width: 9;",
            "max-width: 9;",
            "height: 1;",
            "min-height: 1;",
            "max-height: 1;",
            "border: none;",
            "padding: 0;",
        )

        collapsed_copy = _rule_body(css, "#console-status-collapsed-copy")
        _assert_declarations(
            collapsed_copy,
            "width: 1fr;",
            "min-width: 0;",
            "height: 1;",
            "min-height: 1;",
            "color: $ds-text-muted;",
            "text-wrap: nowrap;",
            "text-overflow: ellipsis;",
        )

        assert "#console-status-chips {\n    scrollbar-size-horizontal: 0;" not in css


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


@pytest.mark.asyncio
async def test_screen_fresh_status_row_starts_expanded() -> None:
    host, _ = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        strip = console.query_one("#console-status-chips", ConsoleStatusChips)

        assert console._console_status_chips_collapsed is False
        assert strip.collapsed is False
        assert _is_effectively_displayed(
            strip.query_one("#console-status-expanded")
        )
        assert not _is_effectively_displayed(
            strip.query_one("#console-status-collapsed")
        )


@pytest.mark.asyncio
async def test_screen_status_collapse_updates_state_and_focuses_expand() -> None:
    host, _ = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        strip = console.query_one("#console-status-chips", ConsoleStatusChips)

        await pilot.click("#console-status-collapse")
        await pilot.pause()

        expand = strip.query_one("#console-status-expand", Button)
        assert console._console_status_chips_collapsed is True
        assert strip.collapsed is True
        assert _is_effectively_displayed(expand)
        assert host.focused is expand


@pytest.mark.asyncio
async def test_screen_status_expand_updates_state_and_focuses_collapse() -> None:
    host, _ = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        strip = console.query_one("#console-status-chips", ConsoleStatusChips)
        await pilot.click("#console-status-collapse")
        await pilot.pause()

        await pilot.click("#console-status-expand")
        await pilot.pause()

        collapse = strip.query_one("#console-status-collapse", Button)
        assert console._console_status_chips_collapsed is False
        assert strip.collapsed is False
        assert _is_effectively_displayed(collapse)
        assert host.focused is collapse


@pytest.mark.asyncio
async def test_screen_status_collapse_state_resets_on_new_screen() -> None:
    host, app = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        await pilot.click("#console-status-collapse")
        await pilot.pause()
        assert console._console_status_chips_collapsed is True

        replacement = ChatScreen(app)

        assert replacement._console_status_chips_collapsed is False
