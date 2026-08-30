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
