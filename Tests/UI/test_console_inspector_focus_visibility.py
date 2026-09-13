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

import re

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import APP_STYLESHEETS, app_css_text
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


class FocusHarness(ConsoleHarness):
    """Console harness that loads the real generated stylesheet.

    Focus treatment is a CSS question, so a bare ``ConsoleHarness`` --
    which sets no ``CSS_PATH`` -- would apply none of the rules under
    test and pass vacuously no matter what the bundle says. Pointing at
    ``APP_STYLESHEETS`` makes these assertions read the same files the
    app ships.
    """

    # TASK-25812: the bundle alone no longer carries the console rules --
    # the app loads the split screen sheets too, so the harness must.
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


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
    stylesheet = app_css_text()
    assert f"{selector}:focus" in stylesheet, (
        f"{selector} is a Tab stop in the Inspect rail with no :focus rule, "
        "so focusing it changes nothing a keyboard user can see"
    )


def test_container_focus_rules_do_not_use_focus_within():
    """These two contain every other stop in the rail, so a
    ``:focus-within`` tint would be on almost permanently and would carry no
    information at all."""
    stylesheet = app_css_text()
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
        stylesheet = app_css_text()
        for name in visited_containers:
            assert f"#{name}:focus" in stylesheet, (
                f"Tab lands on #{name}, which has no focus rule"
            )


# --- TASK-24700 / TASK-24701 / TASK-24702 -----------------------------------

#: Every class the Inspect rail attaches in Python. TASK-24608 fixed the four
#: `console-inspector-row-*` classes and pinned only those; a second wave then
#: shipped with the same defect, so this list is the generalisation that task
#: should have made. `console-library-activity-error` is the worst of them: it
#: carries a Library operation's failure summary and rendered in body colour.
RAIL_CLASSES_REQUIRING_A_RULE = (
    "console-inspector-outer-scroll-hint",
    "console-library-activity-error",
    "console-library-activity-action",
    "console-library-activity-source-ref",
    "console-selected-turn-subsection",
)


@pytest.mark.parametrize("class_name", RAIL_CLASSES_REQUIRING_A_RULE)
def test_rail_class_attached_in_python_has_a_stylesheet_rule(class_name):
    """A class the rail attaches must paint something."""
    stylesheet = app_css_text()
    assert f".{class_name}" in stylesheet, (
        f"{class_name} is attached in Python but has no rule in the bundled "
        "stylesheet, so whatever it encodes renders as plain body text"
    )


def test_the_collapsed_handle_does_not_demand_more_rows_than_the_rail():
    """TASK-24700: the fold-height fix landed on half of a widget pair.

    TASK-24605 lowered `#console-right-rail` to `min-height: 12` because a
    24-row terminal allots it 13. The COLLAPSED form is a different widget,
    and it kept `min-height: 20` -- so the rail's shipping default state
    still over-claimed rows on exactly the terminals the fix was written for.
    """
    stylesheet = app_css_text()
    start = stylesheet.index(".console-inspector-rail-handle {")
    block = stylesheet[start : stylesheet.index("}", start)]
    match = re.search(r"min-height:\s*(\d+)", block)
    assert match, f"no min-height in the handle rule: {block!r}"
    assert int(match.group(1)) <= 12, (
        f"the collapsed handle demands {match.group(1)} rows; the open rail's "
        "floor is 12 because a 24-row terminal allots the rail 13"
    )


def test_the_container_focus_cue_is_an_edge_not_a_tint():
    """TASK-24702: the rail's container focus cue must be an accent EDGE.

    Three measurements ruled out the alternatives. A tint cannot carry the
    cue on this near-black theme: `$ds-action-focus 12%` measured 1.35:1
    against the rail ground and 1.11:1 against the pinned card, 45% reaches
    only ~1.74:1, and even a fully opaque accent is 3.77:1 -- so a tint has
    to be ~85-90% opaque, i.e. a solid block behind the text, to clear the
    3:1 non-text floor. A full `outline` clears it and is what DESIGN.md
    prescribes, but it paints over the widget's own edge cells, and at 80x24
    the rail body is THREE rows -- a top and bottom border would take two of
    them. A one-column left edge costs a column instead of two rows, is the
    same accent (so the same 3.77:1), and is already the house dense-form
    convention.
    """
    stylesheet = app_css_text()
    start = stylesheet.index("#console-inspector-rail-body:focus")
    block = stylesheet[start : stylesheet.index("}", start)]
    assert "outline-left" in block, (
        f"the container focus cue is not an edge: {block!r}"
    )
    assert "$ds-action-focus" in block, (
        f"the focus edge does not use the accent token: {block!r}"
    )
    assert not re.search(r"\$ds-action-focus\s+\d+%", block), (
        "an alpha-blended tint is back; it cannot reach 3:1 on this theme"
    )


@pytest.mark.asyncio
async def test_focusing_a_rail_container_actually_changes_its_painted_edge():
    """The behavioural half of TASK-24702: the rule must resolve, not just
    exist in the stylesheet."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = FocusHarness(app)
    async with host.run_test(size=(180, 50)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.press("alt+i")
        await pilot.pause()
        await pilot.pause()

        body = console.query_one("#console-inspector-rail-body")
        before = str(body.styles.outline_left)
        body.focus()
        await pilot.pause()
        after = str(body.styles.outline_left)

        assert before != after, (
            "focusing the rail body changed nothing about its left edge; the "
            f"rule did not resolve (before={before!r} after={after!r})"
        )
        assert "thick" in after, f"expected a thick focus edge, got {after!r}"
