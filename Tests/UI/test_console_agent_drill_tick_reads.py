"""A Console poll tick reads the Agent rail's drilled sub-agent run once (TASK-33620.15.1.1).

TASK-33620.15.1 made the config-sync cadence read the Agent rail's section
lines and fleet rows as projection inputs on every 0.2 s poll tick. A tick
that projects then computes the same lines again to render them. While the
rail is drilled into a sub-agent, each computation reads that run's durable
record (``ConsoleAgentBridge.subagent_run``, an ``AgentRunsDB`` read on the
UI loop), so a projecting tick read it twice where one read serves both.

These tests drive the real Console screen's tick projection
(``config_sync.project_console_tick``) on a mounted Console drilled into a
real ``AgentRunsDB`` sub-agent run, and count the run reads and the Agent
section renders the tick makes.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest

from Tests.UI.test_console_fleet_budget_history import _bridge, _seed
from Tests.UI.test_console_fleet_panel import _AGENT_SECTION_SIZE, _setup_console
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from Tests.UI.test_console_send_acknowledgement import until
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_runtime import ensure_console_runtime
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Console_Modules import config_sync
from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController

pytestmark = pytest.mark.bootstrap_profile


@contextmanager
def _tick_reads(bridge):
    """Count, on this thread, drilled-run reads and Agent section renders.

    The tick under test runs synchronously in the caller, so everything it
    does is counted and nothing the app does elsewhere is.
    """
    read = ConsoleAgentBridge.subagent_run.__code__
    render = ConsoleAgentController._console_agent_section_payload.__code__
    stats = {"reads": 0, "renders": 0}

    def observe(frame, event, _argument):
        if event != "call":
            return
        if frame.f_code is read and frame.f_locals.get("self") is bridge:
            stats["reads"] += 1
        elif frame.f_code is render:
            stats["renders"] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        yield stats
    finally:
        sys.setprofile(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("resumed", [False, True], ids=["run", "resumed-run"])
async def test_drilled_run_is_read_once_per_poll_tick(tmp_path, monkeypatch, resumed):
    """A projecting tick reads the drilled run once; a skipped tick at most once.

    The cadence reads the rail's lines as an input, and the projection renders
    them: both in the same synchronous tick, so one read serves both. A
    resumed run's record also carries its continuation budget (a second query
    inside the same read).
    """
    now = [1000.0]
    monkeypatch.setattr(config_sync, "_clock", lambda: now[0])
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    try:
        parent = db.create_run(conversation_id="conv-A", agent_kind="primary")
        ancestor = _seed(db, 100, parent=parent) if resumed else None
        child = _seed(db, 40, parent=parent, ancestor=ancestor)
        db.set_status(parent, "done")
        bridge = _bridge(db)
        app = _build_test_app()
        _configure_native_ready_console(app)
        ensure_console_runtime(app).set_agent_bridge(bridge)
        host = ConsoleHarness(app)
        async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
            console = await _setup_console(pilot, host, bridge)
            console._console_agent_drilldown_run_id = child
            console._sync_console_agent_section()
            await pilot.pause()
            status = console.query_one("#console-agent-section-status")
            await until(lambda: "Sub-agent" in str(status.content), timeout=10)

            # A projecting tick: forced, so it projects whatever its inputs
            # read. A cold checked display defers its first projection to the
            # replay (no render); tick again once it has warmed.
            projected = None
            for _ in range(20):
                with _tick_reads(bridge) as tick:
                    config_sync.project_console_tick(console, force=True)
                if tick["renders"]:
                    projected = tick
                    break
                await pilot.pause(0.05)
            assert projected is not None, "no forced tick rendered the Agent section"
            assert projected["renders"] == 1, projected
            assert projected["reads"] == 1, (
                f"a projecting tick read the drilled run {projected['reads']} times"
                " (the input read and the render each read it)"
            )

            # The next tick sees nothing changed within the backstop second
            # and skips the projection: the input read alone.
            with _tick_reads(bridge) as skipped:
                config_sync.project_console_tick(console)
            assert skipped["renders"] == 0, skipped
            assert skipped["reads"] <= 1, skipped

            # The drill-in still renders the run's own detail.
            assert "Sub-agent · done" in str(status.content)
    finally:
        db.close()
