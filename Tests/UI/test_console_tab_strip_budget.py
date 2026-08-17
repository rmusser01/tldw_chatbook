"""Console session tab strip space budget (TASK-2154.4).

Regression coverage for UX-review finding LY-02
(Docs/superpowers/qa/console-ux-review-2026-08/console-ux-review.md): the
tab strip rendered the ``Temporary`` button as ``Temporar`` at 140 AND 160
cols. Measured root cause: not the strip's column budget (91 cells at 140
with the Inspector collapsed, 111 at 160, vs the 48 the four strip controls
need) but per-button label budget -- Button's ``line-pad: 1`` (one blank
cell each side, not zero-able via CSS) stacked on the strip rule's
``padding: 0 1`` left plain Buttons only ``width - 4`` label cells, so
``Temporary`` (9 chars) got 8 in its 12-wide button. (Session tabs escaped:
``ConsoleSessionTabButton``'s nowrap+clip BUNDLED_CSS paints past the
line-pad area up to the widget edge, so their 19-char labels rendered whole
all along.) The rule now carries ``padding: 0``; line-pad alone keeps the
identical one-cell visual gap and leaves the label budget at ``width - 2``.

Multi-tab overflow keeps the strip's HorizontalScroll semantics: fixed-width
tabs, the active tab auto-scrolled into view, and the trailing New
tab/Temporary controls reachable by focus (Textual scrolls a focused widget
into its scrollable ancestor's viewport).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.containers import HorizontalScroll
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Widgets.Console.console_session_surface import (
    CONSOLE_SESSION_TAB_DISPLAY_CHARS,
    ConsoleSessionSurface,
)

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"


class StyledConsoleHarness(ConsoleHarness):
    """ConsoleHarness with the shipped stylesheet so app-tier rules apply."""

    CSS_PATH = str(BUNDLE)


def _rendered_button_line(button: Button) -> str:
    """Return the button's single rendered row exactly as painted."""
    return button.render_line(0).text


def _strip_children_inside_viewport(console) -> None:
    """Assert every tab-strip child renders fully inside the strip viewport."""
    strip = console.query_one("#console-native-tab-strip", HorizontalScroll)
    viewport = strip.content_region
    for child in strip.children:
        region = child.region
        assert region.width > 0, f"#{child.id} has zero width"
        assert region.x >= viewport.x, (
            f"#{child.id} starts left of the strip viewport: {region.x} < {viewport.x}"
        )
        assert region.right <= viewport.right, (
            f"#{child.id} is clipped by the strip viewport: right {region.right} "
            f"> viewport right {viewport.right} (the LY-02 clip)"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(140, 42), (160, 48)])
async def test_temporary_button_fully_visible_with_inspector_handle(
    size: tuple[int, int],
) -> None:
    """LY-02 AC#1/#2: at 140 and 160 cols with the Inspector handle present,
    every strip control renders inside the strip viewport and the Temporary
    button paints its full label (was 'Temporar')."""
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-temporary-tab")
        await pilot.pause(0.2)

        assert console.query_one("#console-inspector-rail-handle").display is True
        _strip_children_inside_viewport(console)

        temporary = console.query_one("#console-new-temporary-tab", Button)
        assert "Temporary" in _rendered_button_line(temporary), (
            f"Temporary button renders {_rendered_button_line(temporary)!r} at "
            f"{size} -- the label must paint whole"
        )
        # New tab and the active tab's close control stay fully rendered too.
        new_tab = console.query_one("#console-new-chat-tab", Button)
        assert "New tab" in _rendered_button_line(new_tab)
        close = console.query_one(".console-session-close-button", Button)
        assert "✕" in _rendered_button_line(close)


@pytest.mark.asyncio
async def test_temporary_button_fully_visible_with_inspector_open_at_140() -> None:
    """LY-02 AC#1: at 140 cols with the Inspector OPEN (the state after
    TASK-2154.2 made it reachable below 150), the strip shrinks to ~68 cells
    but the four controls (48) still fit without scrolling."""
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-temporary-tab")
        await pilot.pause(0.2)

        # press() rather than pilot.click: in the bundle-styled harness the
        # (transparent, soon-to-hide) setup backdrop overlay still covers the
        # handle's cell at this point and swallows the raw click; the click
        # path itself is covered by test_console_inspector_compact_access.py
        # and UAT p7. What THIS test pins is the strip budget once the rail
        # is open, and press() reaches the same handler.
        console.query_one("#console-inspector-rail-open", Button).press()
        right_rail = console.query_one("#console-right-rail")
        for _ in range(30):
            if right_rail.display is True:
                break
            await pilot.pause(0.1)
        assert right_rail.display is True

        _strip_children_inside_viewport(console)
        temporary = console.query_one("#console-new-temporary-tab", Button)
        assert "Temporary" in _rendered_button_line(temporary)


async def _wait_for_strip_child_count(pilot, strip, count: int) -> None:
    for _ in range(50):
        if len(strip.children) == count:
            return
        await pilot.pause(0.1)
    raise AssertionError(
        f"tab strip has {len(strip.children)} children, expected {count}"
    )


@pytest.mark.asyncio
async def test_many_tabs_scroll_and_keep_all_controls_reachable() -> None:
    """LY-02 AC#3: with four sessions the strip content (4*24 + 24 = 120
    cells) overflows the 140-col viewport (~91), so the strip scrolls -- and
    every control stays reachable: the active tab is auto-scrolled fully into
    view, focusing New tab/Temporary scrolls them into view, and each close
    button rides immediately right of its own tab."""
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        await pilot.pause(0.2)

        strip = console.query_one("#console-native-tab-strip", HorizontalScroll)
        # Seeded app starts with one session (tab + close + 2 controls = 4
        # children); three New-tab activations take it to four sessions.
        # Re-query the button each round: sync_sessions re-mounts the strip's
        # children, so a held reference goes stale after the first press.
        for expected in (6, 8, 10):
            console.query_one("#console-new-chat-tab", Button).press()
            await pilot.pause(0.1)
            await _wait_for_strip_child_count(pilot, strip, expected)
        # Let the auto-scroll retry land (it defers while the freshly
        # mounted active tab still has a zero region).
        await pilot.pause(0.5)

        # The strip content now overflows its viewport: scrolling is the
        # chosen degradation (fixed-width tab contract preserved).
        assert strip.virtual_size.width > strip.content_region.width
        assert strip.is_scrollable and strip.allow_horizontal_scroll

        # The newest (active) tab was auto-scrolled fully into view.
        tabs = list(console.query(".console-session-tab"))
        active = console.query_one(".console-session-tab-active", Button)
        viewport = strip.content_region
        assert active.region.x >= viewport.x
        assert active.region.right <= viewport.right

        # Focusing the trailing controls scrolls them into the viewport --
        # the keyboard/focus reachability path for the scrolled-out buttons.
        temporary = console.query_one("#console-new-temporary-tab", Button)
        temporary.focus()
        await pilot.pause(0.2)
        assert temporary.region.x >= viewport.x
        assert temporary.region.right <= viewport.right

        # Every close button rides immediately right of its own tab and
        # scrolls with it.
        close_buttons = list(console.query(".console-session-close-button"))
        assert len(tabs) == 4
        assert len(close_buttons) == 4
        for tab, close in zip(tabs, close_buttons, strict=True):
            assert close.region.x == tab.region.right, (
                f"close button for {tab.id} detached from its tab"
            )


class StyledTabStripHost(ConsolidatedCSSApp):
    """Bare session surface with the shipped stylesheet (app-tier rules)."""

    CSS_PATH = str(BUNDLE)

    def compose(self):
        yield ConsoleSessionSurface(SimpleNamespace(notify=MagicMock()))


@pytest.mark.asyncio
async def test_max_length_session_label_keeps_its_ellipsis() -> None:
    """Guard the session-tab label budget: a title truncated to the 19-char
    tab budget must keep rendering whole, ellipsis included, under the
    shipped stylesheet (the tab's nowrap+clip DEFAULT_CSS is what lets the
    label paint past the button's line-pad area up to the widget edge)."""
    app = StyledTabStripHost()
    async with app.run_test(size=(120, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        session = ConsoleChatSession(
            title="summarize the UX review plan", id="s1"
        )
        await surface.sync_sessions(sessions=[session], active_session_id="s1")
        await pilot.pause(0.2)

        tab = app.query_one("#console-session-tab-s1", Button)
        label = str(tab.label)
        assert len(label) == CONSOLE_SESSION_TAB_DISPLAY_CHARS
        rendered = _rendered_button_line(tab)
        assert label in rendered, (
            f"tab renders {rendered!r}; full truncated label {label!r} "
            f"(ellipsis included) must paint"
        )

        temporary = app.query_one("#console-new-temporary-tab", Button)
        assert "Temporary" in _rendered_button_line(temporary)
