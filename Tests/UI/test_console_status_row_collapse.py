"""Focused tests for the Console status row's mounted layout.

TASK-35160 follow-up: the Status collapsible (Status ▾/▴ toggle pair, the
collapsed "Status hidden" presentation, and its persisted
``[console] status_chips_collapsed`` state) is removed -- the strip is one
always-on text line, so this file now pins the row's geometry and its
stylesheet contract only.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from textual.containers import HorizontalScroll
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleStatusChips,
    ConsoleStatusText,
)

# The strip mount imports real config getters (consolidated_css -> app
# import chain); per-test env redirect trips the config-participant
# admission (sibling suites' opt-in).
pytestmark = pytest.mark.bootstrap_profile


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CSS_ROOT = _REPO_ROOT / "tldw_chatbook/css"
# TASK-35160: the status-row rules moved to the console source sheet
# (ADR-161's console-sources home); the old components/_agentic_terminal
# arm predates the rule's current home and never carried it.
_CONSOLE_SOURCE = _CSS_ROOT / "features/_console.tcss"
_BUNDLED_STYLESHEET = _CSS_ROOT / "tldw_cli_modular.tcss"


def _detok(value: str) -> str:
    """Resolve ADR-161 sizing tokens to literals (the CSS is token-legal per
    ADR-150; the contract's literal expectations are checked post-resolution,
    same approach as test_console_chip_focus_contract.py)."""
    resolved = value
    for pattern, repl in (
        (r"\$ds-space-(\d+)", r"\1"),
        (r"\$ds-size-(\d+)", r"\1"),
        (r"\$ds-percent-(\d+)", r"\1%"),
        (r"\$ds-fr-(\d+)", r"\1fr"),
        (r"\$ds-width-full", "100%"),
        (r"\$ds-height-full", "100%"),
        (r"\$ds-width-fill", "1fr"),
        (r"\$ds-height-fill", "1fr"),
    ):
        resolved = re.sub(pattern, repl, resolved)
    return resolved


def _rule_body(css: str, selector: str) -> str:
    css = _detok(css)
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


class StatusRowApp(ConsolidatedCSSApp):
    """Mount the status row in isolation for focused layout tests."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        """Compose the isolated status row."""
        yield ConsoleStatusChips(_state(), id="console-status-chips")


class PanelStatusRowApp(StatusRowApp):
    """Mount the status row exactly as the Console screen does.

    The real screen composes the strip with ``classes="ds-panel"``
    (chat_screen.py), which carries a bottom margin the id rule must
    cancel; a stand-in neighbor below makes the resulting gap, if any,
    observable as real geometry.
    """

    def compose(self) -> ComposeResult:
        """Compose the ds-panel status row above a stand-in neighbor."""
        yield ConsoleStatusChips(
            _state(),
            id="console-status-chips",
            classes="ds-panel",
        )
        yield Static("footer stand-in", id="footer-stand-in")


@pytest.mark.asyncio
async def test_status_row_panel_class_reserves_no_margin_row() -> None:
    """ds-panel's bottom margin must not open a blank row under the strip.

    ``.ds-panel`` declares ``margin: 0 0 1 0``; the ``#console-status-chips``
    rule overrides the class's other box styles and must cancel the margin
    too, or a zero-information blank row renders between the strip and the
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
    for stylesheet in (_CONSOLE_SOURCE, _BUNDLED_STYLESHEET):
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

        assert "scrollbar-size-horizontal" not in host


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 32), (140, 42)])
async def test_status_row_is_one_scroller_with_the_line_inside(
    size: tuple[int, int],
) -> None:
    """No collapsible, no second presentation: the strip composes the
    scroll viewport directly, the line fills it, and both stay one row."""
    app = StatusRowApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        line = app.query_one("#console-status-text", ConsoleStatusText)

        from textual.widgets import Button

        left = app.query_one("#console-status-scroll-left", Button)
        right = app.query_one("#console-status-scroll-right", Button)

        viewport = strip.content_region
        assert strip.region.height == 1
        assert scroller.region.height == 1
        assert line.region.height == 1
        assert line.region.x >= viewport.x
        # Arrow visibility tracks actual overflow, whichever way this
        # width lands: hidden when the line fits, flanking it when not.
        overflows = scroller.max_scroll_x > 0
        assert left.display is overflows
        assert right.display is overflows
        # The removed collapsible leaves no orphaned widgets behind.
        assert not app.query("#console-status-collapse")
        assert not app.query("#console-status-expand")
        assert not app.query("#console-status-collapsed")
