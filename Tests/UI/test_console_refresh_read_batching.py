"""Finite refresh reads avoid repeated UI-thread database admission."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController

pytestmark = pytest.mark.bootstrap_profile


def _agent(bridge):
    tasks = []
    runtime = SimpleNamespace(agent_bridge=bridge)

    def run_worker(operation, **_kwargs):
        task = asyncio.create_task(operation)
        tasks.append(task)
        return task

    async def sync():
        return None

    controller = ConsoleAgentController(
        SimpleNamespace(run_worker=run_worker, _console_runtime=lambda: runtime),
        app_instance=SimpleNamespace(),
        chat_store_accessor=lambda: None,
        provider_gateway_accessor=lambda: None,
        native_tool_calls_enabled_accessor=lambda: None,
        current_rail_conversation_id=lambda: "conv",
        current_rail_state_accessor=lambda: None,
        chat_controller_accessor=lambda: SimpleNamespace(
            run_state=SimpleNamespace(status=ConsoleRunStatus.STREAMING)
        ),
        sync_native_console_chat_ui_accessor=lambda: sync,
        reveal_agent_detail=lambda: None,
    )
    return controller, tasks


class _Bridge:
    def __init__(self, db):
        self._db = db
        self.calls = []
        self.children = ()
        query = db.count_subagents_by_conversation

        def counted(ids):
            self.calls.append(threading.get_ident())
            return query(ids)

        db.count_subagents_by_conversation = counted

    def live_snapshot(self, _cid):
        return SimpleNamespace(subagents=self.children)

    def run_log_target_token(self, _cid):
        return ("turn", "run")

    def subagent_counts(self, ids):
        return self._db.count_subagents_by_conversation(ids)


async def _finish(tasks):
    while any(not task.done() for task in tasks):
        await asyncio.gather(*tuple(tasks))


def _historical_bridge(database):
    primary = database.create_run(conversation_id="conv", agent_kind="primary")
    database.set_status(primary, "done")
    child = database.create_run(
        conversation_id="conv",
        agent_kind="subagent",
        parent_run_id=primary,
        task="persisted research",
    )
    database.set_status(child, "done", result="saved result")
    return ConsoleAgentBridge(
        agent_runs_db=database, store=None, provider_gateway=None
    ), child


@pytest.mark.asyncio
async def test_cold_fleet_and_overview_share_one_counted_historical_read_off_ui(
    tmp_path,
    monkeypatch,
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    database = AgentRunsDB(tmp_path / "fleet-history.sqlite", "fleet-history")
    try:
        bridge, child = _historical_bridge(database)
        agent, tasks = _agent(bridge)
        reads, admissions, sql_owners = [], [], []
        list_runs = database.list_runs
        original_operation = storage._repository_operation

        def read(*args, **kwargs):
            reads.append(threading.get_ident())
            sql_owners.append(getattr(storage._operation_local, "operation", None))
            return list_runs(*args, **kwargs)

        def operation(*args, **kwargs):
            admissions.append(threading.get_ident())
            return original_operation(*args, **kwargs)

        monkeypatch.setattr(database, "list_runs", read)
        monkeypatch.setattr(storage, "_repository_operation", operation)
        for _ in range(6):
            agent._console_agent_fleet_rows()
            agent._console_agent_section_lines()
        assert reads == [] and admissions == [], "cold presentation admitted DB on UI"
        assert len(tasks) == 1, "overview/fleet scheduled independent historical reads"
        await _finish(tasks)
        assert len(reads) == 2 and all(
            thread != threading.get_ident() for thread in reads
        )
        assert sql_owners[0] is not None and sql_owners[0] is sql_owners[1]
        # One counted callback plus the database's existing checked close
        # operations. Newly opened worker connections still retire normally.
        assert len(admissions) == 3 and all(
            thread != threading.get_ident() for thread in admissions
        )
        rows = agent._console_agent_fleet_rows()
        assert len(rows) == 1 and rows[0].row_id == child
        assert "persisted research" in rows[0].primary_text
        assert agent._console_agent_section_lines()[0] == "Agent: done"
        for _ in range(6):
            agent._console_agent_fleet_rows()
            agent._console_agent_section_lines()
        assert len(reads) == 2 and len(admissions) == 3
        # Direct bridge reads retain their original behavior and prove the
        # real database/admission counters are live, rather than empty spies.
        reads.clear()
        admissions.clear()
        direct = bridge.historical_snapshot("conv")
        assert direct.subagents[0].run_id == child
        assert reads == [threading.get_ident()] * 2
        assert admissions == [threading.get_ident()] * 2
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["bridge", "database", "conversation", "run", "profile"]
)
async def test_historical_projection_rejects_changed_owner_and_captures_database(
    tmp_path,
    monkeypatch,
    mutation,
):
    from tldw_chatbook.Chat.console_agent_bridge import AgentLiveSnapshot

    old_db = AgentRunsDB(tmp_path / "old-history.sqlite", "old-history")
    new_db = AgentRunsDB(tmp_path / "new-history.sqlite", "new-history")
    entered, release = threading.Event(), threading.Event()
    try:
        bridge, _child = _historical_bridge(old_db)
        agent, tasks = _agent(bridge)
        list_runs = old_db.list_runs
        receivers = []

        def blocked(*args, **kwargs):
            receivers.append(old_db)
            entered.set()
            assert release.wait(5)
            return list_runs(*args, **kwargs)

        monkeypatch.setattr(old_db, "list_runs", blocked)
        assert agent._console_agent_fleet_rows() == ()
        assert await asyncio.to_thread(entered.wait, 5)
        if mutation == "bridge":
            agent._screen._console_runtime().agent_bridge = ConsoleAgentBridge(
                agent_runs_db=new_db, store=None, provider_gateway=None
            )
        elif mutation == "database":
            bridge._db = new_db
        elif mutation == "conversation":
            agent._current_rail_conversation_id = lambda: "different-conversation"
        elif mutation == "run":
            bridge._publish_live(
                "conv", "later-turn", AgentLiveSnapshot(status="running"), primary=True
            )
        else:
            agent.app_instance.chachanotes_db = object()
        release.set()
        await _finish(tasks)
        assert agent._console_historical_read is None
        assert receivers == [
            old_db,
            old_db,
        ], "captured historical read redirected to new DB"
        assert bridge._historical_cache == {}, "UI worker polluted authoritative cache"
    finally:
        release.set()
        old_db.close()
        new_db.close()


@pytest.mark.asyncio
async def test_cancelled_historical_projection_retires_and_retries(
    tmp_path, monkeypatch
):
    database = AgentRunsDB(tmp_path / "cancel-history.sqlite", "cancel-history")
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    try:
        bridge, child = _historical_bridge(database)
        agent, tasks = _agent(bridge)
        original = bridge._derive_historical_snapshot

        def held(*args, **kwargs):
            try:
                entered.set()
                assert release.wait(5)
                return original(*args, **kwargs)
            finally:
                retired.set()

        monkeypatch.setattr(bridge, "_derive_historical_snapshot", held)
        assert agent._console_agent_fleet_rows() == ()
        assert await asyncio.to_thread(entered.wait, 5)
        tasks[0].cancel()
        await asyncio.sleep(0)
        assert not tasks[0].done() and not retired.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        assert retired.is_set() and agent._console_historical_read is None
        monkeypatch.setattr(bridge, "_derive_historical_snapshot", original)
        assert agent._console_agent_fleet_rows() == ()
        await asyncio.gather(*tasks[1:])
        assert agent._console_agent_fleet_rows()[0].row_id == child
    finally:
        release.set()
        database.close()


@pytest.mark.asyncio
async def test_active_refresh_batches_count_reads_off_ui_thread(tmp_path):
    database = AgentRunsDB(tmp_path / "agent-counts.sqlite", "refresh-counts")
    try:
        primary = database.create_run(conversation_id="conv", agent_kind="primary")
        child = database.create_run(
            conversation_id="conv", agent_kind="subagent", parent_run_id=primary
        )
        bridge = _Bridge(database)
        bridge.children = (SimpleNamespace(run_id=child, handle_id="child"),)
        agent, tasks = _agent(bridge)
        rows = (SimpleNamespace(conversation_id="conv"),)
        for _ in range(6):
            agent._console_subagent_counts_for_rows(bridge, rows)
        assert bridge.calls == [], "UI refresh performed synchronous DB reads"
        await _finish(tasks)
        assert len(bridge.calls) == 1
        assert bridge.calls[0] != threading.get_ident()
        assert agent._console_subagent_counts_for_rows(bridge, rows) == {"conv": 1}
        # A newly spawned child invalidates within the same active run/TTL.
        child2 = database.create_run(
            conversation_id="conv", agent_kind="subagent", parent_run_id=primary
        )
        bridge.children += (SimpleNamespace(run_id=child2, handle_id="child2"),)
        agent._console_subagent_counts_for_rows(bridge, rows)
        await _finish(tasks)
        assert agent._console_subagent_counts_for_rows(bridge, rows) == {"conv": 2}
    finally:
        database.close()


@pytest.mark.asyncio
async def test_old_profile_count_worker_cannot_publish_into_new_bridge(tmp_path):
    old_db = AgentRunsDB(tmp_path / "old.sqlite", "old-counts")
    new_db = AgentRunsDB(tmp_path / "new.sqlite", "new-counts")
    release = threading.Event()
    entered = threading.Event()
    try:
        primary = old_db.create_run(conversation_id="conv", agent_kind="primary")
        old_db.create_run(
            conversation_id="conv", agent_kind="subagent", parent_run_id=primary
        )
        old, new = _Bridge(old_db), _Bridge(new_db)
        original = old_db.count_subagents_by_conversation

        def held(ids):
            entered.set()
            assert release.wait(5)
            return original(ids)

        old_db.count_subagents_by_conversation = held
        agent, tasks = _agent(old)
        rows = (SimpleNamespace(conversation_id="conv"),)
        agent._console_subagent_counts_for_rows(old, rows)
        assert await asyncio.to_thread(entered.wait, 5)
        agent._screen._console_runtime().agent_bridge = new
        assert agent._console_subagent_counts_for_rows(new, rows) == {}
        release.set()
        await _finish(tasks)
        assert agent._console_subagent_counts_for_rows(new, rows) == {}
    finally:
        release.set()
        await _finish(tasks if "tasks" in locals() else [])
        old_db.close()
        new_db.close()


@pytest.mark.asyncio
async def test_alternating_browser_row_sets_do_not_restart_pending_counts(tmp_path):
    database = AgentRunsDB(tmp_path / "alternating.sqlite", "alternating-counts")
    try:
        primary = database.create_run(conversation_id="conv", agent_kind="primary")
        database.create_run(
            conversation_id="conv", agent_kind="subagent", parent_run_id=primary
        )
        bridge = _Bridge(database)
        agent, tasks = _agent(bridge)
        rows = (SimpleNamespace(conversation_id="conv"),)
        larger = rows + (SimpleNamespace(conversation_id="other"),)
        for _ in range(6):
            agent._console_subagent_counts_for_rows(bridge, rows)
            agent._console_subagent_counts_for_rows(bridge, larger)
        await _finish(tasks)
        assert len(bridge.calls) == 2
        assert agent._console_subagent_counts_for_rows(bridge, rows) == {"conv": 1}
        assert agent._console_subagent_counts_for_rows(bridge, larger) == {"conv": 1}
        await _finish(tasks)
        assert len(bridge.calls) == 2
    finally:
        database.close()


@pytest.mark.asyncio
async def test_cancelled_count_refresh_remains_retryable(tmp_path, monkeypatch):
    from tldw_chatbook.DB import base_db

    database = AgentRunsDB(tmp_path / "cancelled.sqlite", "cancelled-counts")
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        bridge = _Bridge(database)
        agent, tasks = _agent(bridge)
        original = base_db.run_owned_db_call

        async def held(*args):
            entered.set()
            await release.wait()
            return await original(*args)

        monkeypatch.setattr(base_db, "run_owned_db_call", held)
        rows = (SimpleNamespace(conversation_id="conv"),)
        agent._console_subagent_counts_for_rows(bridge, rows)
        await entered.wait()
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        monkeypatch.setattr(base_db, "run_owned_db_call", original)
        agent._console_subagent_counts_for_rows(bridge, rows)
        await _finish(tasks[1:])
        assert len(bridge.calls) == 1
    finally:
        release.set()
        database.close()
