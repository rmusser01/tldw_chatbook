"""PR2b Task 5: the Console Agent fleet section's sync coalesces a burst.

Mirrors `test_console_control_bar_coalescing.py` (task-3010) exactly -- same
spy-on-the-real-method shape, same "settle, then fire N requests, expect
exactly 1 execution" structure -- applied to `_sync_console_agent_section`
via its own coalescer, `_request_console_agent_fleet_sync`
(`_run_coalesced_console_agent_fleet_sync` is the `call_after_refresh`
trailing-run target, matching `_run_coalesced_control_bar_sync`'s shape).
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def sync_spy(monkeypatch):
    """Count real executions of the underlying Agent-fleet-section sync."""
    real_sync = ChatScreen._sync_console_agent_section
    calls: list[object] = []

    def counting_sync(self):
        calls.append(self)
        return real_sync(self)

    monkeypatch.setattr(ChatScreen, "_sync_console_agent_section", counting_sync)
    return calls


async def test_a_burst_of_fleet_sync_requests_produces_exactly_one_run(sync_spy):
    """The core coalescing proof the task brief asks for: N requests ("a
    burst of N fleet events") in a row, one trailing sync."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = ChatScreen(app)
        await app.push_screen(screen)
        for _ in range(8):
            await pilot.pause()

        settled = len(sync_spy)
        # N = 5: a burst standing in for e.g. five near-simultaneous
        # per-row Cancel presses, or any other UI-thread caller reacting
        # to a burst of fleet events in the same tick.
        for _ in range(5):
            screen._request_console_agent_fleet_sync()
        for _ in range(3):
            await pilot.pause()

    assert len(sync_spy) == settled + 1, (
        f"5 coalesced fleet-sync requests produced "
        f"{len(sync_spy) - settled} runs (expected exactly 1)"
    )


async def test_requests_landing_before_the_trailing_run_still_all_fold_in(sync_spy):
    """Requests made across several `call_after_refresh` cycles, as long as
    each new request lands before the PREVIOUS one's trailing run has
    fired, keep folding into a single pending run -- not one per
    `pilot.pause()`."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = ChatScreen(app)
        await app.push_screen(screen)
        for _ in range(8):
            await pilot.pause()

        settled = len(sync_spy)
        screen._request_console_agent_fleet_sync()
        # A second request while one is already scheduled must not queue a
        # SECOND trailing run -- the scheduled-flag guard exists precisely
        # to make this a no-op. Checked BEFORE any `pilot.pause()` (or the
        # `async with` block's own exit, which drains pending refresh
        # callbacks as part of teardown) has a chance to run the trailing
        # sync at all.
        screen._request_console_agent_fleet_sync()
        assert screen._console_agent_fleet_sync_scheduled is True
        assert len(sync_spy) == settled  # not yet run


async def test_the_coalesced_request_still_actually_runs_the_sync(sync_spy):
    """Coalescing must not swallow the sync entirely -- a single request
    still produces exactly one execution once the event loop catches up."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = ChatScreen(app)
        await app.push_screen(screen)
        for _ in range(8):
            await pilot.pause()

        settled = len(sync_spy)
        screen._request_console_agent_fleet_sync()
        assert screen._console_agent_fleet_sync_scheduled is True
        await pilot.pause()

    assert len(sync_spy) == settled + 1
    assert screen._console_agent_fleet_sync_scheduled is False
