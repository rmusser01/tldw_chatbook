"""task-3010: the Console control-bar sync coalesces its mount-window burst.

cProfile (post-task-3011): `_sync_console_control_bar` executed 14 times
during one screen push at ~47ms each — 0.65s of a ~1.2s settled push, every
caller individually justified, nothing deduplicating them.

KNOWN FLAKE (task-15766, documented 2026-08-15): this file failed 2/3 on dev
at 48a54ed9c/762596846 (task-3070 notes, 2026-08-10) with no code change
involved, and passed 18/18 file-runs at 48ad9e7de across unloaded, 10-process
CPU-loaded (2.5x slowdown), and default-plugin conditions — the mechanism was
NOT changed in between (`git log -S` on `_run_coalesced_control_bar_sync`,
`_console_sync_requested`, and the mount-time `call_after_refresh` scheduling
is empty over that range). Identified trigger condition, still live in the
code: both tests settle on a fixed pause-count heuristic, not a quiescence
fence. If a mount-driven `_sync_native_console_chat_ui` is still in flight
when the window closes, (a) an overlapping request sets
`_console_sync_requested`, whose finally-block re-dispatch
(chat_screen.py `_sync_native_console_chat_ui`) runs one EXTRA
`_sync_console_control_bar`, which can breach test 1's `<= 6` bound, and
(b) a trailing execution can land after test 2 captures `settled`, breaking
its exactly-one assertion. If this file goes red intermittently with counts
one or two above the bound, suspect scheduler pressure widening the mount
window past the pause heuristic before suspecting the coalescer itself.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def sync_spy(monkeypatch):
    """Count real executions of the underlying control-bar sync."""
    real_sync = ChatScreen._sync_console_control_bar
    calls: list[object] = []

    def counting_sync(self, rail_state=None):
        calls.append(rail_state)
        return real_sync(self, rail_state)

    monkeypatch.setattr(ChatScreen, "_sync_console_control_bar", counting_sync)
    return calls


async def test_screen_push_runs_a_bounded_number_of_control_bar_syncs(sync_spy):
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = ChatScreen(app)
        await app.push_screen(screen)
        for _ in range(8):
            await pilot.pause()

    # 8, not fewer: the coalescer covers the sync-pipeline burst sites, but
    # immediacy-bearing callers stay direct — the scope-refresh pair and the
    # native-sync inline call (its precomputed rail_state anchors the rail
    # cascade ordering, TASK-251/task-3010 round 2), plus the provider/model
    # Select initializers that legitimately fire once each at mount. Was 14.
    # The visible hands-free Switch added two legitimate initialization
    # synchronizations; this still keeps the pre-coalescer count of 14 nearly
    # halved while rejecting an actual mount-window burst.
    assert len(sync_spy) <= 8, (
        f"control-bar sync ran {len(sync_spy)} times during one push — "
        "the mount-window burst is back"
    )


async def test_requested_sync_still_executes(sync_spy):
    """Coalescing must not swallow the sync: a request after the mount
    storm settles still produces exactly one additional execution."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = ChatScreen(app)
        await app.push_screen(screen)
        for _ in range(8):
            await pilot.pause()

        settled = len(sync_spy)
        screen._request_console_control_bar_sync()
        screen._request_console_control_bar_sync()
        screen._request_console_control_bar_sync()
        for _ in range(3):
            await pilot.pause()

    assert len(sync_spy) == settled + 1, (
        f"three coalesced requests produced {len(sync_spy) - settled} runs "
        "(expected exactly 1)"
    )
