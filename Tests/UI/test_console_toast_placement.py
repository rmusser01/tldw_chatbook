"""Console feedback toasts must not sit over the composer action cluster (task-352)."""

import pytest

# Textual does not export ToastRack publicly (it is not in `textual.widgets`),
# and the headless notification system does not mount one, so the test reaches
# into the private module to construct one directly and assert the Console
# screen's CSS docks it. Mirrors the existing `textual.widgets._markdown`
# private import already used under Tests/.
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static
from textual.widgets._toast import Toast, ToastRack

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


@pytest.mark.asyncio
async def test_console_toast_rack_docks_top_not_over_composer() -> None:
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


class _Dialog(ModalScreen[None]):
    """Any dialog over the Console, as Switch model is."""

    def compose(self) -> ComposeResult:
        yield Static("dialog")


def _shown_toasts(screen) -> list[str]:
    """The notices a screen's own rack draws (none while the rack is hidden)."""
    return [
        str(toast._notification.message)
        for rack in screen.query(ToastRack)
        if rack.display
        for toast in rack.query(Toast)
    ]


@pytest.mark.asyncio
# TASK-33008 review round 2: _build_test_app reads guarded config, so under the
# per-test sandbox it raises RecoveryRequired("raw_source_selection_changed");
# the node keeps the bootstrap profile and runs in the admission-sensitive step.
@pytest.mark.bootstrap_profile
async def test_a_dialog_over_the_console_draws_each_notice_once() -> None:
    """Capture finding 7: a notice posted just before Switch model reopened
    drew twice, in full at the dialog's bottom right and clipped behind the
    dialog at the Console's top right. It was posted once: the dialog's own
    rack shows the app's notices, and the Console's rack under it kept its
    copy. The Console hides its rack while it is covered and shows the live
    notices again when it is back on top."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(211, 44), notifications=True) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        host.notify("Console provider changed", timeout=30)
        await pilot.pause(0.2)
        assert _shown_toasts(console) == ["Console provider changed"]

        dialog = _Dialog()
        await host.push_screen(dialog)
        await pilot.pause(0.2)
        assert _shown_toasts(dialog) == ["Console provider changed"]
        assert _shown_toasts(console) == []

        host.pop_screen()
        await pilot.pause(0.2)
        assert _shown_toasts(console) == ["Console provider changed"]
