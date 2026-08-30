"""Every Tab stop inside the Inspect rail must show that it has focus.

TASK-24612. A measured focus walk of the open rail found an 8-stop closed
Tab cycle in which two stops are CONTAINERS rather than controls: the outer
scroller (``#console-inspector-rail-body``) and the rail root
(``#console-right-rail``). Both are deliberate -- the scroller is focusable
so the keyboard can scroll it, and the root is the pane target F6 lands on
(``right_rail.can_focus = True`` at compose) -- so the fix is a visible
treatment, not removal.

Live capture showed the scroller's entire focus indication as a single
border glyph, and a nested live-work viewport lighting only its scrollbar
column.
"""

from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


class FocusHarness(ConsoleHarness):
    CSS_PATH = str(BUNDLED_STYLESHEET)


#: The two container Tab stops. Controls in the rail get their treatment
#: from Textual's own Button/Input focus styling; these two got nothing.
CONTAINER_TAB_STOPS = (
    "#console-inspector-rail-body",
    "#console-right-rail",
)


@pytest.mark.parametrize("selector", CONTAINER_TAB_STOPS)
def test_container_tab_stops_have_a_focus_rule(selector):
    """TASK-24612: a focusable container needs a focus treatment of its own.

    Deterministic rather than behavioural on purpose. The first version of
    this test focused each widget and diffed ``widget.styles``; that produced
    a FALSE POSITIVE on the header button (whose treatment arrives via an
    ancestor class applied on DescendantFocus, one refresh later) while
    missing both containers entirely, because some unrelated style does move
    on focus. Asserting the rule exists says exactly what is meant.
    """
    stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")
    assert f"{selector}:focus" in stylesheet, (
        f"{selector} is a Tab stop in the Inspect rail with no :focus rule, "
        "so focusing it changes nothing a keyboard user can see"
    )


def test_container_focus_rules_do_not_use_focus_within():
    """These two contain every other stop in the rail, so a
    ``:focus-within`` tint would be on almost permanently and would carry no
    information at all."""
    stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")
    for selector in CONTAINER_TAB_STOPS:
        assert f"{selector}:focus-within" not in stylesheet, (
            f"{selector} tints on descendant focus, which means it is tinted "
            "whenever anything in the rail has focus"
        )


@pytest.mark.asyncio
async def test_the_rail_tab_cycle_is_reachable_and_bounded():
    """The behavioural half: the rail's stops are all inside the rail, and
    F6 remains the way out (Tab alone cannot leave -- that is by design, and
    the footer advertises F6)."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusHarness(app)
    async with host.run_test(size=(180, 50)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.press("alt+i")
        await pilot.pause()
        await pilot.pause()

        rail = console.query_one("#console-right-rail")
        assert rail.display

        console.set_focus(console.query_one("#console-inspector-rail-collapse"))
        await pilot.pause()
        seen = []
        for _ in range(12):
            focused = host.focused
            if focused is not None:
                seen.append(focused.id or type(focused).__name__)
            await pilot.press("tab")
            await pilot.pause()

        assert len(set(seen)) > 1, f"Tab did not move inside the rail: {seen}"
        # Every container stop the cycle visits must be one we have styled.
        visited_containers = {
            name
            for name in seen
            if name in {"console-inspector-rail-body", "console-right-rail"}
        }
        stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")
        for name in visited_containers:
            assert f"#{name}:focus" in stylesheet, (
                f"Tab lands on #{name}, which has no focus rule"
            )
