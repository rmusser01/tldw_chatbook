"""Evidence-only focused real DB cases; root owns native execution/install."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import inspect
import os
import sqlite3
import sys
import time
from types import CodeType, MethodType, SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_agent_bridge import (
    AgentLiveSnapshot,
    ConsoleAgentBridge,
    SubAgentSummary,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB import base_db, private_sqlite
from tldw_chatbook.DB.base_db import BaseDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.private_sqlite import HelperLease
from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS,
)

pytestmark = pytest.mark.bootstrap_profile


@contextmanager
def _original_count_work(database):
    """Observe actual original count/helper/connection bodies without wrapping them."""
    query_outer = inspect.getattr_static(AgentRunsDB, "count_subagents_by_conversation")
    query = inspect.unwrap(query_outer)
    bound_query = database.count_subagents_by_conversation
    assert type(bound_query) is MethodType
    assert bound_query.__self__ is database and bound_query.__func__ is query_outer
    invoke_code = next(
        value
        for value in base_db.run_owned_db_call.__code__.co_consts
        if type(value) is CodeType and value.co_name == "invoke"
    )
    connector_outer = private_sqlite._connect_registered_sqlite
    connector = inspect.unwrap(connector_outer)
    helper = inspect.getattr_static(HelperLease, "start").__func__
    getter = inspect.unwrap(BaseDB._get_connection)
    codes = {
        query.__code__: "query",
        helper.__code__: "helper",
        getter.__code__: "get",
        connector_outer.__code__: "admission",
        connector.__code__: "connector",
    }
    tool = next(index for index in (3, 4) if sys.monitoring.get_tool(index) is None)
    name = "subagent-badge-count-work-" + str(id(database))
    sys.monitoring.use_tool_id(tool, name)
    assert sys.monitoring.get_events(tool) == 0
    work = {
        "queries": 0,
        "query_returns": 0,
        "helpers": 0,
        "admissions": 0,
        "connectors": 0,
        "connections": [],
        "issues": [],
    }

    def inside(frame):
        while frame is not None:
            values = frame.f_locals
            if frame.f_code is query.__code__ and values.get("self") is database:
                return True
            if (
                frame.f_code is query_outer.__code__
                and values.get("function") is query
                and values.get("repository", values.get("self")) is database
            ):
                return True
            if frame.f_code is invoke_code and values.get("database") is database:
                callback = values.get("operation")
                if (
                    type(callback) is MethodType
                    and callback.__self__ is database
                    and callback.__func__ is query_outer
                ):
                    return True
            frame = frame.f_back
        return False

    def actual_connector(frame, *, wrapper):
        return (
            inside(frame.f_back)
            and frame.f_locals.get("owner_id") == "db.base"
            and str(frame.f_locals.get("database")) == database.db_path_str
            and (not wrapper or frame.f_locals.get("function") is connector)
        )

    def start(code, offset):
        frame = sys._getframe(1)
        if frame.f_code is not code:
            work["issues"].append("wrong_start_frame")
            return
        if codes[code] == "query" and frame.f_locals.get("self") is database:
            work["queries"] += 1
        elif codes[code] == "helper" and inside(frame.f_back):
            work["helpers"] += 1
        elif codes[code] == "admission" and actual_connector(frame, wrapper=True):
            work["admissions"] += 1
        elif codes[code] == "connector" and actual_connector(frame, wrapper=False):
            work["connectors"] += 1

    def returned(code, offset, value):
        frame = sys._getframe(1)
        if frame.f_code is not code:
            work["issues"].append("wrong_return_frame")
            return
        if codes[code] == "query" and frame.f_locals.get("self") is database:
            work["query_returns"] += 1
        elif (
            codes[code] == "get"
            and frame.f_locals.get("self") is database
            and inside(frame.f_back)
            and isinstance(value, sqlite3.Connection)
        ):
            if all(value is not existing for existing in work["connections"]):
                work["connections"].append(value)

    mask = sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN
    assert (
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, start)
        is None
    )
    assert (
        sys.monitoring.register_callback(
            tool, sys.monitoring.events.PY_RETURN, returned
        )
        is None
    )
    for code in codes:
        sys.monitoring.set_local_events(tool, code, mask)
    try:
        yield work
    finally:
        assert (
            sys.monitoring.get_tool(tool) == name
            and sys.monitoring.get_events(tool) == 0
        )
        for code in codes:
            assert sys.monitoring.get_local_events(tool, code) == mask
            sys.monitoring.set_local_events(tool, code, 0)
        assert (
            sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, None)
            is start
        )
        assert (
            sys.monitoring.register_callback(
                tool, sys.monitoring.events.PY_RETURN, None
            )
            is returned
        )
        assert (
            inspect.getattr_static(AgentRunsDB, "count_subagents_by_conversation")
            is query_outer
        )
        assert inspect.unwrap(query_outer) is query
        assert private_sqlite._connect_registered_sqlite is connector_outer
        assert inspect.unwrap(connector_outer) is connector
        assert inspect.getattr_static(HelperLease, "start").__func__ is helper
        assert inspect.unwrap(BaseDB._get_connection) is getter
        sys.monitoring.free_tool_id(tool)
        assert all(sys.monitoring.get_local_events(tool, code) == 0 for code in codes)
        assert sys.monitoring.get_tool(tool) is None


def _controller(notes, bridge):
    """Use real constructor/DBs and only declared view/runtime scheduler seams."""
    tasks = []
    runtime = SimpleNamespace(agent_bridge=bridge)

    def schedule(awaitable, **options):
        task = asyncio.create_task(awaitable)
        tasks.append(task)
        return task

    async def repaint():
        pass

    screen = SimpleNamespace(run_worker=schedule, _console_runtime=lambda: runtime)
    app = SimpleNamespace(chachanotes_db=notes)
    controller = ConsoleAgentController(
        screen,
        app_instance=app,
        chat_store_accessor=lambda: None,
        provider_gateway_accessor=lambda: None,
        native_tool_calls_enabled_accessor=lambda: (lambda: False),
        current_rail_conversation_id=lambda: "conv-A",
        current_rail_state_accessor=lambda: None,
        chat_controller_accessor=lambda: None,
        sync_native_console_chat_ui_accessor=lambda: repaint,
        reveal_agent_detail=lambda: None,
    )
    return controller, tasks


async def _settle(tasks):
    consumed = 0
    while consumed < len(tasks):
        current = tasks[consumed:]
        consumed = len(tasks)
        await asyncio.gather(*current)


def _closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True
    return False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transition", ["primary_only", "child", "ttl", "database_owner"]
)
async def test_real_badge_count_work_tracks_subagents_not_primary_publication(
    tmp_path, transition, record_property
):
    notes = CharactersRAGDB(tmp_path / "notes.db", client_id="badge-primary-proof")
    runs = AgentRunsDB(
        tmp_path / "runs.db", client_id="badge-primary-proof", reconcile_on_init=False
    )
    other = None
    tasks = []
    try:
        # Persist the real primary before warming. The tested post-warm
        # transition is the original process-local publisher, so no DB write
        # latency can accidentally consume the unchanged two-second TTL.
        primary = runs.create_run(conversation_id="conv-A", agent_kind="primary")
        runs.close()
        notes.close_connection()
        bridge = ConsoleAgentBridge(
            agent_runs_db=runs, store=None, provider_gateway=None
        )
        controller, tasks = _controller(notes, bridge)
        rows = [SimpleNamespace(conversation_id="conv-A")]
        row_ids = frozenset({"conv-A"})
        with _original_count_work(runs) as work:
            assert controller._console_subagent_counts_for_rows(bridge, rows) == {}
            await _settle(tasks)
            state = controller._console_subagent_counts_read[row_ids]
            assert not state["pending"] and state["at"] > 0 and state["values"] == {}
            assert work["queries"] == work["query_returns"] == 1
            assert work["admissions"] == work["connectors"] == 1
            assert work["helpers"] == (0 if os.name == "nt" else 1)
            assert work["connections"]
            assert all(_closed(connection) for connection in work["connections"])
            before = (
                work["queries"],
                work["helpers"],
                work["admissions"],
                work["connectors"],
            )
            if transition == "primary_only":
                bridge._publish_live(
                    "conv-A", primary, AgentLiveSnapshot(status="running"), primary=True
                )
                assert bridge.run_log_target_token("conv-A")[0] == primary
                assert bridge.live_snapshot("conv-A").subagents == ()
                assert (
                    time.monotonic() - state["at"]
                    < CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS
                )
            elif transition == "child":
                child = runs.create_run(
                    conversation_id="conv-A",
                    agent_kind="subagent",
                    parent_run_id=primary,
                    task="real child",
                )
                runs.close()
                bridge._publish_live(
                    "conv-A",
                    primary,
                    AgentLiveSnapshot(
                        status="running",
                        subagents=(
                            SubAgentSummary(
                                text="real child", status="running", run_id=child
                            ),
                        ),
                    ),
                    primary=True,
                )
            elif transition == "ttl":
                await asyncio.sleep(CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS + 0.02)
            else:
                other = AgentRunsDB(
                    tmp_path / "other-runs.db",
                    client_id="badge-owner-proof",
                    reconcile_on_init=False,
                )
                other.close()
                bridge._db = other
            controller._console_subagent_counts_for_rows(bridge, rows)
            await _settle(tasks)
            record_property("actual_original_count_queries_before", before[0])
            record_property("actual_original_count_queries_after", work["queries"])
            record_property("actual_helper_starts_before", before[1])
            record_property("actual_helper_starts_after", work["helpers"])
            record_property("actual_connector_admissions_before", before[2])
            record_property("actual_connector_admissions_after", work["admissions"])
            record_property("actual_connector_bodies_before", before[3])
            record_property("actual_connector_bodies_after", work["connectors"])
            assert not work["issues"]
            assert all(_closed(connection) for connection in work["connections"])
            if transition == "primary_only":
                assert (
                    work["queries"] == before[0]
                ), "primary-only publication reissued sub-agent count work"
                assert work["helpers"] == before[1]
                assert work["admissions"] == before[2]
                assert work["connectors"] == before[3]
                assert controller._console_subagent_counts_read[row_ids] is state
            elif transition in ("child", "ttl"):
                assert work["queries"] == work["query_returns"] == before[0] + 1
                assert work["helpers"] == before[1] + (0 if os.name == "nt" else 1)
                assert work["admissions"] == before[2] + 1
                assert work["connectors"] == before[3] + 1
                if transition == "child":
                    assert controller._console_subagent_counts_read[row_ids][
                        "values"
                    ] == {"conv-A": 1}
            else:
                assert controller._console_subagent_counts_read[row_ids] is not state
                assert (
                    controller._console_subagent_counts_read[row_ids]["key"][-1]
                    is other
                )
                assert work["queries"] == before[0]
    finally:
        await _settle(tasks)
        if other is not None:
            other.close()
        runs.close()
        notes.close()


@pytest.mark.parametrize(
    "custom", ["instance", "class", "borrowed", "body", "subclass"]
)
def test_custom_primary_token_contract_is_preserved(tmp_path, monkeypatch, custom):
    runs = AgentRunsDB(
        tmp_path / "custom-runs.db",
        client_id="badge-custom-proof",
        reconcile_on_init=False,
    )
    try:

        class CustomBridge(ConsoleAgentBridge):
            pass

        bridge_type = CustomBridge if custom == "subclass" else ConsoleAgentBridge
        bridge = bridge_type(agent_runs_db=runs, store=None, provider_gateway=None)
        original = ConsoleAgentBridge.run_log_target_token

        def custom_target(self, conversation_id):
            return "custom-turn", "custom-run"

        if custom == "instance":
            monkeypatch.setattr(
                bridge, "run_log_target_token", MethodType(custom_target, bridge)
            )
        elif custom == "class":
            monkeypatch.setattr(
                ConsoleAgentBridge, "run_log_target_token", custom_target
            )
        elif custom == "borrowed":
            borrowed = ConsoleAgentBridge(
                agent_runs_db=runs, store=None, provider_gateway=None
            )
            borrowed._publish_live(
                "conv-A", "borrowed-turn", AgentLiveSnapshot(), primary=True
            )
            monkeypatch.setattr(
                bridge, "run_log_target_token", borrowed.run_log_target_token
            )
        elif custom == "body":
            monkeypatch.setattr(original, "__code__", custom_target.__code__)
        else:
            bridge._publish_live(
                "conv-A", "subclass-turn", AgentLiveSnapshot(), primary=True
            )
        token = ConsoleAgentController._subagent_count_live_token(
            bridge, frozenset({"conv-A"})
        )
        expected = {
            "borrowed": ("borrowed-turn", None),
            "subclass": ("subclass-turn", None),
        }.get(custom, ("custom-turn", "custom-run"))
        assert token[0][1] == expected
    finally:
        runs.close()
