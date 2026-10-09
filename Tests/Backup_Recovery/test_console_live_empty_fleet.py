"""Established live empty fleet does not reopen unrelated durable history."""

import os
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from Tests.UI.test_console_refresh_read_batching import (
    _agent,
    _finish,
    _historical_bridge,
)
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Chat.console_agent_bridge import (
    AgentLiveSnapshot,
    ConsoleAgentBridge,
    SubAgentSummary,
)
from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


@contextmanager
def _native_history_calls():
    code = ConsoleAgentBridge._derive_historical_snapshot.__code__
    helper = private_sqlite.prepare_in_helper.__code__
    native = None
    if os.name == "nt":
        from tldw_chatbook.Utils import windows_files

        native = windows_files._native().open_handle.__func__.__code__
    calls = {"history": 0, "helpers": 0, "opens": 0, "stats": 0}
    before_thread, before_main = threading.getprofile(), sys.getprofile()

    def observe(frame, event, _arg):
        if event == "c_call" and os.name != "nt" and _arg is os.stat:
            calls["stats"] += 1
        if event != "call":
            return
        if frame.f_code is code:
            calls["history"] += 1
        elif frame.f_code is helper:
            calls["helpers"] += 1
        elif frame.f_code is native:
            calls["opens"] += 1

    threading.setprofile_all_threads(observe)
    try:
        yield calls
    finally:
        threading.setprofile_all_threads(before_thread)
        sys.setprofile(before_main)


def _bind_running(bridge, database, tmp_path):
    run_id = database.create_run(conversation_id="conv", agent_kind="primary")
    bridge._remember_run_log_authority(
        run_id,
        tmp_path,
        session_id="session",
        access_scope=lambda: nullcontext(tmp_path),
        conversation_id="conv",
    )
    bridge._publish_live(
        "conv", "current-turn", AgentLiveSnapshot(status="running"), primary=True
    )
    return run_id


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["setup", "running"])
async def test_real_live_empty_fleet_avoids_native_history(
    tmp_path, record_property, phase
):
    database = AgentRunsDB(tmp_path / "live-empty.db", "live-empty")
    bridge, old_child = _historical_bridge(database)
    agent, tasks = _agent(bridge)
    try:
        assert bridge.historical_snapshot("conv").subagents[0].run_id == old_child
        if phase == "setup":
            bridge.begin_setup_phase("conv")
        else:
            _bind_running(bridge, database, tmp_path)
        with _native_history_calls() as calls:
            started = time.monotonic()
            for _ in range(4):
                agent._console_agent_fleet_rows()
                agent._console_agent_section_lines()
            await _finish(tasks)
            rows = agent._console_agent_fleet_rows()
            elapsed = time.monotonic() - started
        record_property("established_live_receipt", {**calls, "seconds": elapsed})
        assert (
            rows == ()
        ), "previous durable children leaked into established current live empty fleet"
        assert (
            calls["history"] == 0
        ), "known live empty children reopened durable history"
        assert calls["opens"] == calls["helpers"] == 0
        assert not tasks and not worker_leases(database)
    finally:
        await _finish(tasks)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["restored", "unknown_running", "terminal", "custom"])
async def test_unestablished_or_terminal_fleet_preserves_actual_durable_fallback(
    tmp_path, state
):
    database = AgentRunsDB(tmp_path / "durable-fallback.db", "fallback")
    real, child = _historical_bridge(database)
    bridge = real
    if state == "unknown_running":
        real._publish_live(
            "conv", "unbound", AgentLiveSnapshot(status="running"), primary=True
        )
        assert real.live_primary_run_id("conv") is None
    elif state == "terminal":
        primary = database.latest_primary_run_metadata("conv")["id"]
        real._remember_run_log_authority(
            primary,
            tmp_path,
            session_id="session",
            access_scope=lambda: nullcontext(tmp_path),
            conversation_id="conv",
        )
        real._publish_live(
            "conv", "finished", AgentLiveSnapshot(status="done"), primary=True
        )
    elif state == "custom":
        bridge = SimpleNamespace(
            fleet_snapshot=lambda _: [],
            live_snapshot=lambda _: AgentLiveSnapshot(status="running"),
            historical_snapshot=real.historical_snapshot,
        )
    agent, tasks = _agent(bridge)
    try:
        with _native_history_calls() as calls:
            first = agent._console_agent_fleet_rows()
            await _finish(tasks)
            rows = agent._console_agent_fleet_rows()
        assert [row.row_id for row in rows] == [child]
        assert calls["history"] == 1, "native fallback positive control stopped reading"
        if os.name == "nt":
            assert calls["opens"] > 0
        else:
            assert calls["stats"] > 0
            if state != "custom":
                assert calls["helpers"] > 0
        if state != "custom":
            assert first == () and len(tasks) >= 1
        assert not worker_leases(database)
    finally:
        await _finish(tasks)
        database.close()


@pytest.mark.asyncio
async def test_actual_fleet_handles_precede_live_empty_primary(tmp_path):
    database = AgentRunsDB(tmp_path / "survivor.db", "survivor")
    bridge, _old = _historical_bridge(database)
    _bind_running(bridge, database, tmp_path)
    fleet = FleetCoordinator(2, time.monotonic)
    handle = fleet.reserve("surviving live child", None)
    assert handle is not None
    fleet.attach_run(handle.handle_id, "survivor")
    bridge._fleet_coordinators["conv"] = fleet
    bridge._fleet_services["conv"] = SimpleNamespace(fleet_snapshot=fleet.snapshot)
    agent, tasks = _agent(bridge)
    try:
        with _native_history_calls() as calls:
            rows = agent._console_agent_fleet_rows()
        assert len(rows) == 1 and rows[0].row_id == handle.handle_id
        assert "surviving live child" in rows[0].secondary_text
        assert calls["history"] == 0 and not tasks
    finally:
        await _finish(tasks)
        database.close()


@pytest.mark.asyncio
async def test_bound_live_inline_children_precede_durable_history(tmp_path):
    database = AgentRunsDB(tmp_path / "inline.db", "inline")
    bridge, _old = _historical_bridge(database)
    _bind_running(bridge, database, tmp_path)
    bridge._publish_live(
        "conv",
        "current-turn",
        AgentLiveSnapshot(
            status="running",
            subagents=(
                SubAgentSummary("current inline child", run_id="current-child"),
            ),
        ),
        primary=True,
    )
    agent, tasks = _agent(bridge)
    try:
        with _native_history_calls() as calls:
            rows = agent._console_agent_fleet_rows()
        assert [row.row_id for row in rows] == ["current-child"]
        assert calls["history"] == 0 and not tasks
    finally:
        await _finish(tasks)
        database.close()
