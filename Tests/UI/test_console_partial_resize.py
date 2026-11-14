"""Keep incomplete Console composition from consuming a real resize."""

import asyncio
from copy import deepcopy

import pytest
from textual.events import Resize
from textual.geometry import Size

from Tests.UI.app_factory import _build_test_app
from Tests.UI.pump_probe import console_attach_ended, until
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_runtime import dispose_console_runtime

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize(
    "missing_selector",
    [
        "#console-workspace-grid",
        "#console-main-column",
        "#console-left-rail",
        "#console-right-rail",
        "#console-context-rail-handle",
        "#console-inspector-rail-handle",
    ],
)
async def test_partial_console_resize_preserves_transition_until_complete(
    missing_selector: str,
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            await console.query_one(missing_selector).remove()
            previous_settings = deepcopy(app.app_config.get("console", {}))
            console._last_console_workspace_width_band = None

            # Exercise the missing-widget guard before config retries can outlive it.
            console._adapt_console_workspace_to_width.__wrapped__(
                console, Resize(Size(80, 40), Size(80, 40))
            )

            assert console._last_console_workspace_width_band is None
            assert app.app_config.get("console", {}) == previous_settings
            await console.recompose()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 2.0
            await _wait_for_selector(console, pilot, "#console-native-composer")
            await pilot.resize_terminal(80, 40)
            await pilot.pause()
            # The real Resize can still own a pending config retry after Pilot drains.
            assert await until(
                lambda: (
                    console.size.width == 80
                    and console._last_console_workspace_width_band == "single-pane"
                    and "_adapt_console_workspace_to_width"
                    not in getattr(console, "_console_config_retries", {})
                ),
                max(0.0, deadline - loop.time()),
            ), "the real 80-column resize did not finish after recomposition"
            assert console._last_console_workspace_width_band == "single-pane"
            assert not console.query_one("#console-left-rail").display
            assert not console.query_one("#console-right-rail").display
            assert not console.query("#screen-content-error")
    finally:
        await dispose_console_runtime(app)


async def test_recomposed_console_does_not_reuse_a_previous_tree_resize(tmp_path):
    """A queued narrow resize must update the newly composed wide rail owners."""
    import inspect
    import json
    import sys

    trace = []
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            await pilot.pause()

            def observe(phase, state=None):
                left = list(console.query("#console-left-rail"))
                cached = console._last_console_rail_state
                trace.append(
                    {
                        "phase": phase,
                        "screen_width": console.size.width,
                        "band": console._last_console_workspace_width_band,
                        "left_id": id(left[0]) if left else None,
                        "left_display": left[0].display if left else None,
                        "cached_left": cached.left_open if cached is not None else None,
                        "cached_right": cached.right_open
                        if cached is not None
                        else None,
                        "requested_left": state.left_open
                        if state is not None
                        else None,
                        "requested_right": state.right_open
                        if state is not None
                        else None,
                    }
                )

            observe("initial")
            narrow = Resize(Size(80, 40), Size(80, 40))
            console._adapt_console_workspace_to_width(narrow)
            assert console._last_console_workspace_width_band == "single-pane"
            old_left = console.query_one("#console-left-rail")
            assert not old_left.display
            observe("old_tree_narrow_applied")
            await console.recompose()
            new_left = console.query_one("#console-left-rail")
            assert new_left is not old_left and new_left.display
            observe("fresh_wide_tree")
            code = inspect.unwrap(
                type(console)._adapt_console_workspace_to_width
            ).__code__
            codes = {
                code: "resize_body_enter",
                type(
                    console
                )._sync_console_rail_visibility_if_changed.__code__: "dedup_enter",
                type(console)._sync_console_rail_visibility.__code__: "apply_enter",
            }
            entered = []

            def entry(observed, _offset):
                frame = sys._getframe(1)
                if frame.f_locals.get("self") is console:
                    observe(codes[observed], frame.f_locals.get("rail_state"))
                    if observed is code:
                        entered.append(True)

            tool = None
            for candidate in range(5, -1, -1):
                try:
                    sys.monitoring.use_tool_id(candidate, "console-resize-generation")
                except ValueError:
                    continue
                tool = candidate
                break
            assert tool is not None
            try:
                sys.monitoring.register_callback(
                    tool,
                    sys.monitoring.events.PY_START,
                    entry,
                )
                for observed in codes:
                    sys.monitoring.set_local_events(
                        tool, observed, sys.monitoring.events.PY_START
                    )
                console._adapt_console_workspace_to_width(narrow)
            finally:
                for observed in codes:
                    sys.monitoring.set_local_events(tool, observed, 0)
                sys.monitoring.register_callback(
                    tool, sys.monitoring.events.PY_START, None
                )
                sys.monitoring.free_tool_id(tool)
            observe("resize_body_return")
            assert entered == [True], (
                "the real resize body was deferred, so this setup is invalid"
            )
            (tmp_path / "native-resize-generation-trace.json").write_text(
                json.dumps(trace, indent=2)
            )
            assert not new_left.display, (
                "the previous tree's resize suppressed the new rail owner"
            )
            assert not console.query_one("#console-right-rail").display
    finally:
        await dispose_console_runtime(app)


@pytest.mark.parametrize(
    "missing_selector",
    [
        "#console-left-rail",
        "#console-right-rail",
        "#console-context-rail-handle",
        "#console-inspector-rail-handle",
        "#console-main-column",
    ],
)
async def test_incomplete_rail_sync_does_not_publish_a_partial_state(missing_selector):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            await pilot.pause()
            wide = console._current_console_rail_state(available_columns=160)
            console._sync_console_rail_visibility_if_changed(wide)
            assert console._last_console_rail_state == wide
            narrow = console._current_console_rail_state(available_columns=80)
            assert narrow != wide
            await console.query_one(missing_selector).remove()
            before = {
                w.id: w.display for w in console.query("#console-workspace-grid > *")
            }
            preferences = deepcopy(app.app_config.get("console", {}))
            console._sync_console_rail_visibility_if_changed(narrow)
            assert console._last_console_rail_state == wide, (
                "incomplete paint consumed the state"
            )
            assert {
                w.id: w.display for w in console.query("#console-workspace-grid > *")
            } == before
            assert app.app_config.get("console", {}) == preferences
            await console.recompose()
            await pilot.resize_terminal(80, 40)
            await pilot.pause()
            assert not console.query_one("#console-left-rail").display
            assert not console.query_one("#console-right-rail").display
    finally:
        await dispose_console_runtime(app)


async def test_real_terminal_resize_after_partial_recompose_updates_new_rails():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=(80, 40)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            await console_attach_ended(console)
            await pilot.resize_terminal(160, 48)
            await pilot.resize_terminal(80, 40)
            assert console._last_console_workspace_width_band == "single-pane"
            assert not console.query_one("#console-left-rail").display
            await console.query_one("#console-left-rail").remove()
            await pilot.resize_terminal(160, 48)
            assert console._last_console_workspace_width_band == "single-pane"
            await console.recompose()
            assert console.query_one("#console-left-rail").display
            await pilot.resize_terminal(80, 40)
            await pilot.pause()
            assert not console.query_one("#console-left-rail").display
            assert not console.query_one("#console-right-rail").display
    finally:
        await dispose_console_runtime(app)
