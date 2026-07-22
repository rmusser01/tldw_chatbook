"""Console feedback toasts must not sit over the composer action cluster (task-352)."""

import pytest
from textual.widgets._toast import ToastRack

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


@pytest.mark.asyncio
async def test_console_toast_rack_docks_top_not_over_composer():
    """Textual docks notification toasts bottom-right by default — directly over
    the Console composer's Send/Attach/Save cluster and staged-chip strip — and
    toasts capture clicks, so a click aimed at those controls during a toast
    hits the toast instead. The Console screen must dock its toast rack to the
    TOP so feedback never obscures, or swallows clicks aimed at, the composer
    controls (task-352).
    """
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        # Toasts are not rendered by the headless notification system, so mount a
        # ToastRack directly under the Console screen and assert the screen's CSS
        # docks it to the top (Textual's own default is bottom-right).
        rack = ToastRack()
        await console.mount(rack)
        await pilot.pause()

        assert rack.styles.align_vertical == "top", (
            "Console toast rack aligns "
            f"{rack.styles.align_vertical!r} — it would sit over the composer"
        )
        assert str(rack.styles.dock) == "top", (
            f"Console toast rack docks {rack.styles.dock!r}, expected 'top'"
        )
