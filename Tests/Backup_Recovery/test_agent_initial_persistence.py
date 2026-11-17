"""Initial Agent run persistence uses one finite Windows admission."""

import inspect
import json
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root
from tldw_chatbook.Agents import activation as agents
from tldw_chatbook.Agents.agent_models import AgentConfig, AgentStep, RunBudget
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Utils import windows_files

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows admission"
)
pytestmark = [pytestmark, pytest.mark.usefixtures("local_root")]


@pytest.mark.parametrize(
    "route",
    ["stock", "custom", "wrapped", "changed", "late_wrapped", "allocator", "path"],
)
def test_initial_run_persistence_is_one_finite_operation(
    tmp_path, monkeypatch, request, route
):
    db = AgentRunsDB(tmp_path / "initial-runs.db")
    request.addfinalizer(db.close)
    run_id = "initial-run-proof"
    observer = AgentRunsDB(db.db_path, reconcile_on_init=False)
    request.addfinalizer(observer.close)
    original_path = db.db_path
    replacement_path = original_path.with_name("replacement-runs.db")
    counts, statements, step_rows = Counter(), [], []
    provider = []
    phase = False
    reads = []
    custom_calls = []
    ctor_scopes = []

    def chat_call(**_kwargs):
        assert getattr(storage._operation_local, "operation", None) is None
        provider.append(True)
        return {"choices": [{"message": {"content": "done"}}]}

    service = AgentService(db, ToolCatalogRegistry(), chat_call=chat_call)
    constructor = AgentStep.__init__

    def init_step(step, *args, **kwargs):
        constructor(step, *args, **kwargs)
        if step.kind == "agent_run_created":
            ctor_scopes.append(getattr(storage._operation_local, "operation", None))

    # The unchanged acceptance trace observes the same constructor seam.
    monkeypatch.setattr(AgentStep, "__init__", init_step)
    original = db.insert_steps_at_indices

    def custom_insert(run_id, rows):
        custom_calls.append(rows[0][1]["kind"])
        return original(run_id, rows)

    if route == "custom":
        monkeypatch.setattr(db, "insert_steps_at_indices", custom_insert)

    body = inspect.unwrap(AgentRunsDB.connection)
    original_body = body.__code__
    body_scopes = []

    def install_wrapped_body():
        def replaced(self, *, operation_deadline=None):
            yield self._held_connection(operation_deadline=operation_deadline)

        # Replace the executable body while preserving the outer decorator
        # identities. The substitute really serves the normal SQLite read.
        monkeypatch.setattr(body, "__code__", replaced.__code__)

    if route == "wrapped":
        install_wrapped_body()

    def read_durably():
        try:
            row = observer.get_run(run_id)
            return None if row is None else (row["status"], row["steps"])
        finally:
            observer.close()

    def trace_sql(sql):
        if phase and sql in ("BEGIN", "BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"):
            statements.append(sql)

    counts_codes = {
        windows_files._Native.open_handle.__code__: "native_open",
        windows_files._Native.security.__code__: "native_security",
        storage._OperationPath.observe.__code__: "path_observation",
    }
    starts = inspect.unwrap(AgentService._run_one).__code__
    acquire = storage.acquire_storage.__code__
    create = AgentRunsDB.create_run.__code__
    get = AgentRunsDB.get_run.__code__
    append = AgentService._append_run_lifecycle.__code__
    insert = AgentRunsDB.insert_steps_at_indices.__code__
    persist = getattr(AgentService, "_persist_run_lifecycle", None)
    initial_ends = {append} if persist is None else {append, persist.__code__}

    with ThreadPoolExecutor(max_workers=1) as executor:

        def observe(frame, event, value):
            nonlocal phase
            code = frame.f_code
            if event == "call" and code is starts and not reads:
                phase = True
            if not phase:
                return
            if event == "call" and code in counts_codes:
                counts[counts_codes[code]] += 1
            if event == "call" and code is body.__code__ and code is not original_body:
                body_scopes.append(getattr(storage._operation_local, "ancestry", ()))
            if (
                event == "call"
                and code is acquire
                and frame.f_locals.get("path") == db.db_path
            ):
                counts["acquisition"] += 1
            if event == "return" and code is create and value == run_id:
                reads.append(executor.submit(read_durably).result(timeout=10))
                if route == "changed":

                    def changed_read(run_id):
                        custom_calls.append("get_run")
                        return type(db).get_run(db, run_id)

                    monkeypatch.setattr(db, "get_run", changed_read)
                elif route == "late_wrapped":
                    install_wrapped_body()
                elif route == "allocator":

                    class ChangedAllocator:
                        def allocate(self, minimum):
                            custom_calls.append("allocate")
                            return minimum

                    service._owner_seq_allocators[run_id] = ChangedAllocator()
                elif route == "path":
                    monkeypatch.setattr(db, "db_path", replacement_path)
                    monkeypatch.setattr(db, "db_path_str", str(replacement_path))
            if event == "call" and code is get and frame.f_locals["self"] is db:
                counts["read"] += 1
            if event == "call" and code is insert and frame.f_locals["self"] is db:
                step_rows.extend(frame.f_locals["steps"])
            if (
                event == "return"
                and code in initial_ends
                and frame.f_locals.get("kind") == "agent_run_created"
            ):
                phase = False

        def invoke():
            borrowed = db._held_connection()
            borrowed.set_trace_callback(trace_sql)
            try:
                old = sys.getprofile()
                sys.setprofile(observe)
                try:
                    result = service.run_turn(
                        conversation_id="conversation",
                        messages=[{"role": "user", "content": "hello"}],
                        config=AgentConfig(
                            model="fixture", system_prompt="test", budget=RunBudget()
                        ),
                        api_endpoint="openai",
                        requested_run_id=run_id,
                    )
                finally:
                    sys.setprofile(old)
                assert borrowed.execute("SELECT 1").fetchone()[0] == 1
                assert getattr(storage._operation_local, "operation", None) is None
                return borrowed, result
            finally:
                if route == "path":
                    db.db_path = original_path
                    db.db_path_str = str(original_path)
                borrowed.set_trace_callback(None)

        with ThreadPoolExecutor(max_workers=1) as worker:
            result = worker.submit(agents.worker_guard(service)(invoke))
            if route in {"changed", "late_wrapped", "allocator", "path"}:
                with pytest.raises(RecoveryRequired):
                    result.result(timeout=60)
                assert reads == [("running", [])]
                assert custom_calls == ctor_scopes == provider == []
                assert body_scopes == []
                assert statements.count("COMMIT") == 1
                assert not replacement_path.exists()
                assert executor.submit(read_durably).result(timeout=10) == (
                    "running",
                    [],
                )
                return
            borrowed, result = result.result(timeout=60)
    assert result[0] == run_id and provider == [True]
    assert ctor_scopes == [None]
    assert reads == [("running", [])]
    assert counts["read"] == 2
    assert [sql for sql in statements if sql.startswith("BEGIN")] == [
        "BEGIN IMMEDIATE",
        "BEGIN IMMEDIATE",
    ]
    assert statements.count("COMMIT") == 2
    assert [(row["kind"], row["owner_seq"]) for _, row in step_rows] == [
        ("agent_run_created", 0)
    ]
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        borrowed.execute("SELECT 1")
    request.node.user_properties.append(("initial_native_counts", json.dumps(counts)))
    assert counts["acquisition"] == (4 if route in {"custom", "wrapped"} else 2), counts
    assert counts["native_open"] and counts["native_security"]
    if route == "wrapped":
        assert body_scopes and all(len(scope) == 1 for scope in body_scopes)
    if route == "custom":
        assert custom_calls[0] == "agent_run_created"
