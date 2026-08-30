"""Keyboard reachability and recoverability of the Console Inspect rail.

TASK-24604: the rail ships closed (``CONSOLE_RAIL_RIGHT_DEFAULT_OPEN`` is
False while the LEFT rail defaults open), ``action_focus_next_workbench_pane``
filters out non-displayed panes, and the collapsed handle is not in
``CONSOLE_FOCUS_REGISTRY.pane_order`` -- so F6 could not reach the rail in
its shipping state. No ``Binding`` referenced it, ``CONSOLE_WORKBENCH_
SHORTCUTS`` had no entry for it, and the command palette had none. The only
route in was a mouse click, on a product whose own PRODUCT.md commits to
keyboard speed and discoverable essential actions.

TASK-24600: below ``CONSOLE_SINGLE_PANE_COLUMNS`` (84) both rail handles
hide, while a budget-eligible explicit rail may still render. Collapsing at
that width therefore removed every reference to the Inspector from the
screen -- ``grep -c Inspect`` on a live 80x24 capture returned 0 -- and the
only observed recovery was resizing the terminal.
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
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_FOCUS_PANE_FOR_WIDGET,
    CONSOLE_FOCUS_REGISTRY,
    CONSOLE_PANE_COLLAPSED_STAND_IN,
    CONSOLE_WORKBENCH_SHORTCUTS,
    ChatScreen,
)


class KeyboardHarness(ConsoleHarness):
    """Console harness that loads the real generated stylesheet.

    The keyboard route is asserted against what the rail actually
    renders -- widths, visibility, and the collapsed stand-in -- all of
    which are CSS-driven. A bare ``ConsoleHarness`` sets no ``CSS_PATH``
    and would let these checks pass against an unstyled tree.
    """

    CSS_PATH = str(BUNDLED_STYLESHEET)


def _binding_keys() -> set[str]:
    return {str(binding.key) for binding in ChatScreen.BINDINGS}


def test_a_binding_exists_for_the_inspect_rail():
    """TASK-24604: the rail has a key of its own, like Model and Workspace."""
    assert "alt+i" in _binding_keys(), (
        "no Binding opens or focuses the Inspect rail; Console binds alt+m "
        "for the model popover and alt+w for the workspace switcher"
    )


def test_the_inspect_shortcut_is_advertised():
    """TASK-24604: an accelerator nothing announces is not discoverable."""
    labels = " ".join(f"{key} {label}" for key, label in CONSOLE_WORKBENCH_SHORTCUTS)
    assert "Alt+I" in labels, (
        f"Inspect rail shortcut missing from the footer vocabulary: {labels}"
    )


def test_the_collapsed_rail_keeps_its_f6_pane_stop():
    """TASK-24604: F6 skips hidden panes, so a collapsed rail lost its stop.

    The task text proposed adding the handle to ``pane_order``. That is the
    wrong mechanism here: ``_console_workbench_focus_id_for_widget`` checks
    ``pane_order`` BEFORE ``CONSOLE_FOCUS_PANE_FOR_WIDGET``, so promoting the
    handle to a pane would make focus inside the collapsed rail report the
    handle instead of its logical pane and change TASK-2154.11's documented
    between-panes behaviour. The pane stays one entry and gains a collapsed
    stand-in instead, which is what this asserts.
    """
    assert "console-inspector-rail-handle" not in CONSOLE_FOCUS_REGISTRY.pane_order
    assert (
        CONSOLE_PANE_COLLAPSED_STAND_IN["console-right-rail"]
        == "console-inspector-rail-handle"
    )
    assert (
        CONSOLE_FOCUS_PANE_FOR_WIDGET["console-inspector-rail-handle"]
        == "console-right-rail"
    ), "the handle must still resolve to its logical pane (TASK-2154.11)"


@pytest.mark.asyncio
async def test_f6_reaches_the_inspector_pane_while_the_rail_is_collapsed():
    """The behavioural half of the above: the pane cycle must still contain
    the Inspector in the rail's DEFAULT (closed) state."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = KeyboardHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        assert not console.query_one("#console-right-rail").display

        assert console._console_pane_is_reachable("console-right-rail"), (
            "the Inspector pane is unreachable while collapsed, so F6 drops "
            "it from the cycle in the rail's shipping state"
        )

        reached = set()
        for _ in range(8):
            await console.action_focus_next_workbench_pane()
            await pilot.pause()
            current = console._console_workbench_focus_id_for_widget(host.focused)
            if current:
                reached.add(current)
        assert "console-right-rail" in reached, (
            f"F6 never reached the Inspector pane; visited {sorted(reached)}"
        )


@pytest.mark.asyncio
async def test_alt_i_opens_and_focuses_the_rail_then_closes_it():
    """TASK-24604: Alt+I is a round trip, not just an open.

    The rail previously had no keyboard route at all -- the only way in was
    clicking its edge handle, which is hidden at narrow widths. This asserts
    the binding both opens the rail AND lands focus inside it: an open that
    leaves focus in the composer is not a route. A second press must close
    it again, so the same key is also the way back out.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = KeyboardHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        rail = console.query_one("#console-right-rail")
        assert not rail.display, "precondition: the rail ships collapsed"

        await pilot.press("alt+i")
        await pilot.pause()
        await pilot.pause()
        rail = console.query_one("#console-right-rail")
        assert rail.display, "alt+i did not open the Inspect rail"
        focused = host.focused
        assert focused is not None and rail in focused.ancestors_with_self, (
            f"alt+i opened the rail but left focus on {focused!r}"
        )

        await pilot.press("alt+i")
        await pilot.pause()
        await pilot.pause()
        assert not console.query_one("#console-right-rail").display, (
            "alt+i did not close the Inspect rail again"
        )


@pytest.mark.asyncio
async def test_collapsing_below_84_columns_leaves_a_way_back():
    """TASK-24600: the P0. At 80x24 a collapsed rail left nothing on screen
    referencing the Inspector, and only a terminal resize brought it back."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = KeyboardHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        await pilot.press("alt+i")
        await pilot.pause()
        await pilot.pause()
        assert console.query_one("#console-right-rail").display, (
            "alt+i must open the rail even below the single-pane threshold"
        )

        await pilot.press("alt+i")
        await pilot.pause()
        await pilot.pause()
        assert not console.query_one("#console-right-rail").display

        # The whole point: it must come back without resizing the terminal.
        await pilot.press("alt+i")
        await pilot.pause()
        await pilot.pause()
        assert console.query_one("#console-right-rail").display, (
            "the Inspect rail could not be reopened at 80x24 -- this is the "
            "one-way trip TASK-24600 exists to remove"
        )


# --- TASK-24703 -------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_shortcut_does_not_land_focus_on_the_close_control():
    """TASK-24703: opening a pane must not put the caret on its own closer.

    TASK-24604's action correctly moved focus INTO the rail, but the pane's
    default target list starts with `console-inspector-rail-collapse`, so the
    caret arrived on the button that closes the pane the user just opened --
    one stray Enter and they are back where they started.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = KeyboardHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        await pilot.press("alt+i")
        await pilot.pause()
        await pilot.pause()

        rail = console.query_one("#console-right-rail")
        assert rail.display
        focused = host.focused
        assert focused is not None
        assert rail in focused.ancestors_with_self, (
            f"alt+i left focus outside the rail, on {focused!r}"
        )
        assert focused.id != "console-inspector-rail-collapse", (
            "alt+i landed focus on the rail's own collapse button"
        )
        assert focused.id == "console-send-authority-summary", (
            f"expected the pinned authority summary, got {focused.id!r}"
        )


def test_the_inspect_hint_is_promoted_when_the_shell_is_single_pane():
    """TASK-24703: below the single-pane threshold the rail's edge handle is
    hidden, making Alt+I the only route in -- and that is exactly the width
    where `AppFooterStatus`, which degrades by keeping a PREFIX, was dropping
    the hint. Promotion, not mere presence, is what survives truncation."""
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_WORKBENCH_SHORTCUTS_SINGLE_PANE,
    )

    assert CONSOLE_WORKBENCH_SHORTCUTS_SINGLE_PANE[0] == ("Alt+I", "inspect")
    # Same content, only reordered -- nothing is lost from the normal list.
    assert sorted(CONSOLE_WORKBENCH_SHORTCUTS_SINGLE_PANE) == sorted(
        CONSOLE_WORKBENCH_SHORTCUTS
    )


def test_the_inspect_route_is_in_the_f1_reference_not_only_the_footer():
    """TASK-24704 (Qodo #11): the footer and F1 are separate data sources.

    TASK-24604 added Alt+I to `CONSOLE_WORKBENCH_SHORTCUTS` (the footer hint
    strip) and stopped there. `action_show_workbench_help` renders
    `CONSOLE_WORKBENCH_SHORTCUT_GROUPS` instead, so the full keyboard
    reference never mentioned the route.

    That is the wrong half to land. The footer degrades by keeping a PREFIX
    of its hints as width falls, so the surface that DID name Alt+I is the
    one that drops entries -- and it drops them at exactly the widths where
    the rail's edge handle is hidden and Alt+I is the only way in. F1 never
    truncates, which makes it the surface that has to carry the route.
    """
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_WORKBENCH_SHORTCUTS,
        CONSOLE_WORKBENCH_SHORTCUT_GROUPS,
    )

    assert any(key == "Alt+I" for key, _ in CONSOLE_WORKBENCH_SHORTCUTS)

    panes = next(
        entries for title, entries in CONSOLE_WORKBENCH_SHORTCUT_GROUPS
        if title == "Panes"
    )
    inspect = [entry for entry in panes if entry[0] == "Alt+I"]
    assert inspect, (
        "Alt+I is in the footer hints but missing from the F1 reference's "
        f"Panes group: {panes}"
    )
    # The help panel is read, not scanned -- it gets the full phrase rather
    # than the footer's one-word "inspect".
    assert "Inspect rail" in inspect[0][1], inspect
