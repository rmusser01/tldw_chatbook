"""Declared count callbacks keep pending presentation input ownership."""

import asyncio
import threading
from types import MethodType, SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_console_presentation_cadence import _actual_calls
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root  # noqa: PLC0414
from Tests.UI.test_console_refresh_read_batching import _agent, _finish
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


@pytest.mark.asyncio
@pytest.mark.parametrize("definition", ["instance", "class"])
async def test_pending_declared_count_refuses_primary_drift(
    tmp_path, monkeypatch, definition
):
    database = AgentRunsDB(tmp_path / "custom-primary.db", "custom-primary")
    primary = database.create_run(conversation_id="conv", agent_kind="primary")
    database.create_run(
        conversation_id="conv", agent_kind="subagent", parent_run_id=primary
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=database, store=None, provider_gateway=None
    )
    bridge._live_primary_runs["conv"] = "old-primary"
    agent, tasks = _agent(bridge)
    rows = (SimpleNamespace(conversation_id="conv"),)
    entered, release = threading.Event(), threading.Event()
    original = AgentRunsDB.count_subagents_by_conversation
    observed = []
    held = False

    def declared_count(receiver, conversation_ids):
        token = bridge._live_primary_runs.get("conv")
        observed.append(
            (receiver, list(conversation_ids), token, threading.get_ident())
        )
        values = original(receiver, conversation_ids)
        return {
            key: value + (10 if token == "old-primary" else 20)
            for key, value in values.items()
        }

    def barrier(_frame, name):
        nonlocal held
        if name == "counts_return" and not held:
            held = True
            entered.set()
            assert release.wait(10)

    try:
        # Construct the actual-code observer before installing the declared
        # semantic override; its SQL oracle stays the original DB method.
        with _actual_calls(barrier=barrier) as calls:
            if definition == "instance":
                monkeypatch.setattr(
                    database,
                    "count_subagents_by_conversation",
                    MethodType(declared_count, database),
                )
            else:
                monkeypatch.setattr(
                    AgentRunsDB, "count_subagents_by_conversation", declared_count
                )
            assert agent._console_subagent_counts_for_rows(bridge, rows) == {}
            state = agent._console_subagent_counts_read[frozenset({"conv"})]
            assert await asyncio.to_thread(entered.wait, 10)
            assert held and worker_leases(
                database
            ), "original SQL callback did not hold its real worker receiver"
            bridge._live_primary_runs["conv"] = "new-primary"
            release.set()
            await _finish(tasks)
            assert (
                state["values"] == {} and state["at"] == 0
            ), "custom primary-dependent result published under changed inputs"
            assert calls["counts"] == 1 and not worker_leases(database)
            assert agent._console_subagent_counts_for_rows(bridge, rows) == {}
            await _finish(tasks)
            assert agent._console_subagent_counts_for_rows(bridge, rows) == {"conv": 21}
            assert calls["counts"] == 2 and not worker_leases(database)
            assert [
                (owner is database, ids, token) for owner, ids, token, _ in observed
            ] == [(True, ["conv"], "old-primary"), (True, ["conv"], "new-primary")]
            assert all(thread != threading.get_ident() for _, _, _, thread in observed)
    finally:
        release.set()
        await _finish(tasks)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("definition", ["instance", "class"])
async def test_pending_declared_count_refuses_query_replacement(
    tmp_path, monkeypatch, definition
):
    database = AgentRunsDB(tmp_path / "custom-replacement.db", "custom-replacement")
    primary = database.create_run(conversation_id="conv", agent_kind="primary")
    database.create_run(
        conversation_id="conv", agent_kind="subagent", parent_run_id=primary
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=database, store=None, provider_gateway=None
    )
    agent, tasks = _agent(bridge)
    rows = (SimpleNamespace(conversation_id="conv"),)
    entered, release = threading.Event(), threading.Event()
    original = AgentRunsDB.count_subagents_by_conversation
    observed = []
    held = False

    def original_query(receiver, conversation_ids):
        observed.append(
            ("original", receiver, list(conversation_ids), threading.get_ident())
        )
        return {
            key: value + 10
            for key, value in original(receiver, conversation_ids).items()
        }

    def replacement_query(receiver, conversation_ids):
        observed.append(
            ("replacement", receiver, list(conversation_ids), threading.get_ident())
        )
        return {
            key: value + 30
            for key, value in original(receiver, conversation_ids).items()
        }

    def install(function):
        if definition == "instance":
            monkeypatch.setattr(
                database,
                "count_subagents_by_conversation",
                MethodType(function, database),
            )
        else:
            monkeypatch.setattr(
                AgentRunsDB, "count_subagents_by_conversation", function
            )

    def barrier(_frame, name):
        nonlocal held
        if name == "counts_return" and not held:
            held = True
            entered.set()
            assert release.wait(10)

    try:
        with _actual_calls(barrier=barrier) as calls:
            install(original_query)
            assert agent._console_subagent_counts_for_rows(bridge, rows) == {}
            state = agent._console_subagent_counts_read[frozenset({"conv"})]
            assert await asyncio.to_thread(entered.wait, 10)
            assert held and worker_leases(
                database
            ), "original SQL callback did not hold its real worker receiver"
            install(replacement_query)
            release.set()
            await _finish(tasks)
            assert (
                state["values"] == {} and state["at"] == 0
            ), "old query published under a replacement callback"
            assert calls["counts"] == 1 and not worker_leases(database)
            assert agent._console_subagent_counts_for_rows(bridge, rows) == {}
            await _finish(tasks)
            assert agent._console_subagent_counts_for_rows(bridge, rows) == {"conv": 31}
            assert calls["counts"] == 2 and not worker_leases(database)
            assert (
                [(name, owner is database, ids) for name, owner, ids, _ in observed]
                == [("original", True, ["conv"]), ("replacement", True, ["conv"])]
            ), "pending callback was redirected instead of retired on its original receiver"
            assert all(thread != threading.get_ident() for _, _, _, thread in observed)
    finally:
        release.set()
        await _finish(tasks)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("definition", ["instance", "class", "descriptor"])
async def test_declared_same_class_bridge_counter_preserves_selected_callable_abi(
    tmp_path, monkeypatch, definition
):
    database = AgentRunsDB(tmp_path / "custom-bridge-counter.db", "custom-bridge")
    primary = database.create_run(conversation_id="conv", agent_kind="primary")
    database.create_run(
        conversation_id="conv", agent_kind="subagent", parent_run_id=primary
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=database, store=None, provider_gateway=None
    )
    agent, tasks = _agent(bridge)
    rows = (SimpleNamespace(conversation_id="conv"),)
    original = ConsoleAgentBridge.subagent_counts
    invoked, getters = [], []

    def declared_count(receiver, conversation_ids):
        invoked.append((receiver, list(conversation_ids), threading.get_ident()))
        return {
            key: value + 40
            for key, value in original(receiver, conversation_ids).items()
        }

    class DeclaredCounter:
        def __get__(self, receiver, _owner):
            getters.append(threading.get_ident())
            return MethodType(declared_count, receiver)

    try:
        with _actual_calls() as calls:
            if definition == "instance":
                monkeypatch.setattr(
                    bridge, "subagent_counts", MethodType(declared_count, bridge)
                )
            elif definition == "class":
                monkeypatch.setattr(
                    ConsoleAgentBridge, "subagent_counts", declared_count
                )
            else:
                monkeypatch.setattr(
                    ConsoleAgentBridge, "subagent_counts", DeclaredCounter()
                )
            assert agent._console_subagent_counts_for_rows(bridge, rows) == {}
            await _finish(tasks)
            assert agent._console_subagent_counts_for_rows(bridge, rows) == {"conv": 41}
            await _finish(tasks)
            assert calls["counts"] == 1 and not worker_leases(database)
            assert len(invoked) == 1
            assert invoked[0][:2] == (bridge, ["conv"])
            assert invoked[0][2] != threading.get_ident()
            if definition == "descriptor":
                # One selected callable read at admission, publication and the
                # next display boundary, with no extra getter for stock probing.
                assert getters == [threading.get_ident()] * 3
    finally:
        await _finish(tasks)
        database.close()
