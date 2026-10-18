"""Keep incomplete Console composition from consuming a real resize."""

from copy import deepcopy

import pytest
from textual.events import Resize
from textual.geometry import Size

from Tests.UI.app_factory import _build_test_app
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

            console._adapt_console_workspace_to_width(
                Resize(Size(80, 40), Size(80, 40))
            )

            assert console._last_console_workspace_width_band is None
            assert app.app_config.get("console", {}) == previous_settings
            await console.recompose()
            await _wait_for_selector(console, pilot, "#console-native-composer")
            await pilot.resize_terminal(80, 40)
            await pilot.pause()
            assert console._last_console_workspace_width_band == "single-pane"
            assert not console.query_one("#console-left-rail").display
            assert not console.query_one("#console-right-rail").display
            assert not console.query("#screen-content-error")
    finally:
        await dispose_console_runtime(app)
