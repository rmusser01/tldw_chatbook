"""TASK-34100.5 AC#10 (new-entry-exit-handoff-02).

Console's toast rack docks to the top (task-352: never over the composer),
but with a one-cell margin it painted over the nav tab bar, the header's
speech controls and readiness chip, and the control bar for ~5 s on the
first Console frame after setup. A toast must start below the header and
control rows. Measured on the real app with its real stylesheet at 120x40.
"""

from __future__ import annotations

import pytest
from textual.widgets._toast import Toast

from Tests.UI.test_console_visible_turn_attention import _console_app, _reach_console

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_console_toast_paints_below_the_nav_header_and_status_rows(
    tmp_path,
) -> None:
    app, _gateway = _console_app(tmp_path)
    async with app.run_test(size=(120, 40), notifications=True) as pilot:
        chat = await _reach_console(app, pilot, "explore_home")
        protected = [
            chat.query_one("MainNavigationBar"),
            chat.query_one("#console-workbench-header"),
            chat.query_one("#console-speech-controls"),
            chat.query_one("#workbench-header-status"),
            chat.query_one("#console-control-bar"),
            # Review round 2 (V2-F10, live g5-v2-oai 06): a toast over the
            # transcript tab strip swallowed clicks on New tab / Temporary.
            chat.query_one("#console-native-tab-strip"),
        ]
        app.notify("A short Console notice that wraps onto a second line here.")
        for _ in range(40):
            if list(chat.query(Toast)):
                break
            await pilot.pause(0.05)
        toasts = list(chat.query(Toast))
        assert toasts, "no toast was mounted"
        await pilot.pause(0.2)

        header_bottom = max(widget.region.bottom for widget in protected)
        for toast in toasts:
            region = toast.region
            assert region.height > 0
            assert region.y >= header_bottom, (
                f"toast {region} paints over the Console top chrome "
                f"(ends at row {header_bottom})"
            )
            for widget in protected:
                assert not region.overlaps(widget.region), (
                    f"toast {region} covers {widget!r} at {widget.region}"
                )
