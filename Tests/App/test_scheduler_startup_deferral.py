"""A slow mount must not let the scheduler spend the first-frame budget."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext

import pytest

from Tests.Performance import test_ui_ready_module_census as census

_SLOW_MOUNT = """
    # Give the scheduler a whole tick before post-mount setup, if it was
    # incorrectly launched in on_mount. A normal fast boot hides this race.
    tick_finished = asyncio.Event()
    tick_ready_states = []
    owner_loop = asyncio.get_running_loop()
    real_setup = tldw_chatbook.app.TldwCli._post_mount_setup
    real_tick = tldw_chatbook.app.SchedulerLoop.tick

    async def slow_setup(self):
        try:
            await asyncio.wait_for(tick_finished.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        await real_setup(self)

    async def observed_tick(self):
        assert asyncio.get_running_loop() is owner_loop
        tick_ready_states.append(app._ui_ready)
        await real_tick(self)
        tick_finished.set()

    tldw_chatbook.app.TldwCli._post_mount_setup = slow_setup
    tldw_chatbook.app.SchedulerLoop.tick = observed_tick
"""

_VERIFY_TICK = """
        await asyncio.wait_for(tick_finished.wait(), timeout=20.0)
        scheduler_modules = (
            "tldw_chatbook.emergency_stop",
            "tldw_chatbook.Scheduling.scheduler_heartbeat",
        )
        assert all(name in sys.modules for name in scheduler_modules)
        assert tick_ready_states and all(tick_ready_states), (
            "scheduler tick ran before readiness", tick_ready_states,
            [name for name in scheduler_modules if name in FLAG_TIME_MODULES],
        )
        assert app.scheduler_loop.running
        assert not app.scheduler_worker.is_finished
"""


@pytest.mark.integration
def test_real_scheduler_waits_for_readiness_on_a_slow_mount(tmp_path, monkeypatch):
    """Boot twice in an isolated profile; the real first tick still completes.

    Keep the canonical flag-time snapshot and its environment isolation. The
    delay yields the event loop, so off-thread queue/ledger reads can finish
    even when UI setup is slow. No scheduler methods are stubbed out.
    """
    script = census._CENSUS_SCRIPT.replace(
        "    app = tldw_chatbook.app.TldwCli()",
        _SLOW_MOUNT + "    app = tldw_chatbook.app.TldwCli()",
    ).replace("        # The copy taken", _VERIFY_TICK + "        # The copy taken")
    monkeypatch.setattr(census, "_CENSUS_SCRIPT", script)

    modules = census._boot_and_census(tmp_path)

    assert "tldw_chatbook.emergency_stop" not in modules
    assert "tldw_chatbook.Scheduling.scheduler_heartbeat" not in modules
    assert set(census.EXPECTED_AT_READY) <= set(modules)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["quit", "shutdown", "setup_failure"])
async def test_pre_ready_exit_does_not_start_the_scheduler(monkeypatch, stop_reason):
    """A late or failed setup cannot start work after quit or shutdown begins."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    real_setup = type(app)._post_mount_setup
    setup_returned = asyncio.Event()

    async def exit_before_setup(self):
        if stop_reason == "setup_failure":
            setup_returned.set()
            raise RuntimeError("injected pre-ready setup failure")
        if stop_reason == "shutdown":
            self._shutting_down = True
        else:
            self.exit()
        await real_setup(self)
        setup_returned.set()
        self.exit()

    monkeypatch.setattr(type(app), "_post_mount_setup", exit_before_setup)
    expected_error = (
        pytest.raises(RuntimeError, match="injected pre-ready setup failure")
        if stop_reason == "setup_failure"
        else nullcontext()
    )
    with expected_error:
        async with app.run_test(size=(120, 40)):
            await asyncio.wait_for(setup_returned.wait(), timeout=20.0)

    assert not app._ui_ready
    assert getattr(app, "scheduler_worker", None) is None
    assert not app.scheduler_loop.running
