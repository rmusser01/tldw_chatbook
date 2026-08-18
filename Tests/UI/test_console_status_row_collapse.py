"""Focused tests for the Console status row's mounted presentations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
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
    selector_pattern = r"\s*,\s*".join(
        re.escape(part.strip()) for part in selector.split(",")
    )
    match = re.search(rf"{selector_pattern}\s*\{{(?P<body>[^{{}}]*)\}}", css)
    assert match is not None, f"Missing CSS rule for {selector}"
    return match.group("body")


def test_rule_body_tolerates_equivalent_selector_whitespace() -> None:
    """Match selector lists independently of their whitespace formatting."""
    css = "#first,\n    #second { width: 1; }"

    assert "width: 1;" in _rule_body(css, "#first, #second")


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


class StatusRowApp(ConsolidatedCSSApp):
    """Mount the status row in isolation for focused layout tests."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, *, collapsed: bool = False) -> None:
        super().__init__()
        self._collapsed = collapsed

    def compose(self) -> ComposeResult:
        """Compose the isolated status row."""
        yield ConsoleStatusChips(
            _state(),
            collapsed=self._collapsed,
            id="console-status-chips",
        )


class PanelStatusRowApp(StatusRowApp):
    """Mount the status row exactly as the Console screen does.

    The real screen composes the chips with ``classes="ds-panel"``
    (chat_screen.py), which carries a bottom margin the id rule must
    cancel; a stand-in neighbor below makes the resulting gap, if any,
    observable as real geometry.
    """

    def compose(self) -> ComposeResult:
        """Compose the ds-panel status row above a stand-in neighbor."""
        yield ConsoleStatusChips(
            _state(),
            collapsed=self._collapsed,
            id="console-status-chips",
            classes="ds-panel",
        )
        yield Static("footer stand-in", id="footer-stand-in")


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


def _assert_full_button_label_fits(button: Button, expected_label: str) -> None:
    """Assert the mounted button renders its complete label inside its chrome."""
    rendered_line = button.render_line(0)
    rendered_text = rendered_line.text
    internal_chrome_cells = len(rendered_text) - len(rendered_text.strip())
    rendered_label_capacity = button.content_region.width - internal_chrome_cells

    assert str(button.label) == expected_label
    assert rendered_label_capacity >= button.label.cell_length, (
        f"{button.id} region={button.region.width}, "
        f"content={button.content_region.width}, "
        f"label_capacity={rendered_label_capacity}, "
        f"label_cells={button.label.cell_length}, "
        f"rendered={rendered_text!r}"
    )
    assert rendered_text.strip() == expected_label


def _assert_full_static_copy_fits(copy: Static, expected_copy: str) -> None:
    """Assert mounted one-line copy is neither clipped nor ellipsized."""
    rendered_text = copy.render_line(0).text.rstrip()

    assert str(copy.render()) == expected_copy
    assert copy.content_region.width >= len(expected_copy)
    assert rendered_text == expected_copy


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 32), (140, 42)])
async def test_collapsed_status_row_stays_one_line_and_left_anchored(
    size: tuple[int, int],
) -> None:
    """Keep the collapsed restore control visible at the left edge.

    Args:
        size: Terminal dimensions used for the mounted layout check.
    """
    app = StatusRowApp(collapsed=True)
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
        _assert_full_button_label_fits(expand, "Status ▴")
        _assert_full_static_copy_fits(copy, "Status hidden")
        assert expand.region.right <= copy.region.x


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 32), (140, 42)])
async def test_expanded_status_row_keeps_toggle_left_and_scroller_in_viewport(
    size: tuple[int, int],
) -> None:
    """Keep the expanded toggle left of its in-viewport chip scroller.

    Args:
        size: Terminal dimensions used for the mounted layout check.
    """
    app = StatusRowApp()
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
        _assert_full_button_label_fits(collapse, "Status ▾")
        assert collapse.region.right <= scroller.region.x
        assert scroller.region.right == viewport.right


@pytest.mark.asyncio
async def test_status_row_panel_class_reserves_no_margin_row() -> None:
    """ds-panel's bottom margin must not open a blank row under the chips.

    ``.ds-panel`` declares ``margin: 0 0 1 0``; the ``#console-status-chips``
    rule overrides the class's other box styles and must cancel the margin
    too, or a zero-information blank row renders between the chips and the
    footer on the real screen (TASK-17650).
    """
    app = PanelStatusRowApp()
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        neighbor = app.query_one("#footer-stand-in", Static)

        assert chips.styles.margin.bottom == 0
        assert neighbor.region.y == chips.region.y + chips.region.height


def test_status_row_stylesheet_contract_is_in_source_and_bundle() -> None:
    """Keep source and bundled status-row geometry contracts equivalent."""
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
            "margin: 0;",
        )

        presentations = _rule_body(
            css, "#console-status-expanded, #console-status-collapsed"
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

        toggles = _rule_body(css, "#console-status-collapse, #console-status-expand")
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

        assert "scrollbar-size-horizontal" not in host


@pytest.mark.asyncio
async def test_widget_toggles_mounted_presentations_without_replacing_chips() -> None:
    """Swap mounted presentations without replacing status-chip instances."""
    app = StatusRowApp()
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
    """Honor an initially collapsed presentation at widget construction."""
    app = StatusRowApp(collapsed=True)
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
    """Preserve hidden chip updates until the expanded row is restored."""
    app = StatusRowApp()
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
    """Expose focusable status toggles with descriptive tooltips."""
    app = StatusRowApp()
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
    """Start each new Console screen with the status row expanded."""
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
    """Collapse through screen state and focus the inverse expand control."""
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
    """Expand through screen state and focus the inverse collapse control."""
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
async def test_screen_status_collapse_persists_to_config_and_new_screen() -> None:
    """Collapse survives Console recreation via the live config (task-17652).

    Inverts the former reset-on-recreation contract: collapsing pokes
    ``[console] status_chips_collapsed`` synchronously, and a freshly
    constructed screen seeds its state from that value.
    """
    host, app = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        await pilot.click("#console-status-collapse")
        await pilot.pause()
        assert console._console_status_chips_collapsed is True
        assert app.app_config["console"]["status_chips_collapsed"] is True

        replacement = ChatScreen(app)

        assert replacement._console_status_chips_collapsed is True

        await pilot.click("#console-status-expand")
        await pilot.pause()
        assert app.app_config["console"]["status_chips_collapsed"] is False
        assert ChatScreen(app)._console_status_chips_collapsed is False


@pytest.mark.asyncio
async def test_status_collapse_restores_from_persisted_config() -> None:
    """A stored collapsed=True composes the strip collapsed from first paint."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.app_config.setdefault("console", {})["status_chips_collapsed"] = True
    host = ConsoleHarness(app)
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        strip = console.query_one("#console-status-chips", ConsoleStatusChips)

        assert console._console_status_chips_collapsed is True
        assert strip.collapsed is True
        expand = strip.query_one("#console-status-expand", Button)
        assert _is_effectively_displayed(expand)
