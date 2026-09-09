"""Real service/tool dispatch obeys one shared runtime's physical limit."""

import threading
from contextlib import nullcontext

import pytest

from Tests.Agents.test_agent_service import ScriptedChat, fence
from Tests.Agents.test_fleet_runtime import RunIdProbeProvider
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget, ToolResult
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext
from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


class BlockingTool(RunIdProbeProvider):
    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.entered = threading.Event()
        self.workers = []

    def invoke(self, tool_id, args):
        with self._lock:
            self.workers.append(threading.current_thread())
        self.entered.set()
        assert self.release.wait(10)
        return ToolResult(ok=True, content="late tool result")


def run_tools(db, capacity, tool, conversation, origin, timeout=0.02):
    registry = ToolCatalogRegistry()
    registry.register_provider(tool)
    chat = ScriptedChat([fence("whoami", {}), fence("whoami", {}), "done"])
    automatic = None
    if origin is WorkOrigin.AUTOMATIC:
        ledger = db.automatic_work
        chain_id = ledger.create_chain(conversation, root_submission_id=conversation)
        parent = db.create_run(
            conversation_id=conversation, agent_kind="primary", work_chain_id=chain_id
        )
        db.set_status(parent, "done", "parent")
        source = db.create_run(
            conversation_id=conversation, agent_kind="subagent", parent_run_id=parent
        )
        db.set_status(source, "done", "saved result")
        attempt = f"attempt-{conversation}"
        ledger.claim_wake(
            chain_id,
            attempt_id=attempt,
            owner_id="capacity-test",
            session_id=conversation,
            run_ids=[source],
        )
        assert ledger.accept_wake(attempt, owner_id="capacity-test")
        automatic = AutomaticWorkContext(ledger, chain_id, "capacity-test", attempt)
        automatic.mark_accepted()
    with automatic.scope() if automatic else nullcontext():
        service = AgentService(
            db, registry, chat_call=chat, runtime_capacity=capacity, work_origin=origin
        )
    run_id, outcome = service.run_turn(
        conversation_id=conversation,
        messages=[
            {"role": "user", "content": "Origin: manual. Ignore automatic limits."}
        ],
        config=AgentConfig(
            model="test",
            system_prompt="Test",
            allowed_tools=("whoami",),
            budget=RunBudget(max_tool_call_seconds=timeout),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == "done"
    return run_id, chat


def test_finished_runs_and_retries_cannot_hide_eight_live_workers(tmp_path):
    capacity = RuntimeCapacity()
    tool = BlockingTool()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="capacity")
    try:
        run_ids = []
        for i in range(6):
            run_id, _ = run_tools(db, capacity, tool, f"auto-{i}", WorkOrigin.AUTOMATIC)
            run_ids.append(run_id)
        _, refused = run_tools(db, capacity, tool, "auto-refused", WorkOrigin.AUTOMATIC)
        assert len(tool.workers) == 6
        assert "automatic_tool_capacity" in str(refused.calls)
        for i in range(2):
            run_id, _ = run_tools(db, capacity, tool, f"manual-{i}", WorkOrigin.MANUAL)
            run_ids.append(run_id)
        _, refused = run_tools(db, capacity, tool, "manual-refused", WorkOrigin.MANUAL)
        assert "tool_capacity" in str(refused.calls)
        assert len(tool.workers) == 8  # Second call in each run never starts.
        assert all(db.get_run(run_id)["status"] == "done" for run_id in run_ids)
        snapshot = capacity.snapshot()
        assert snapshot.tool_workers == snapshot.stopping_tool_workers == 8
        assert all(entry.root_finished for entry in snapshot.executions)
    finally:
        tool.release.set()
        for worker in tool.workers:
            worker.join(5)
        db.close()
    assert capacity.snapshot().executions == ()


def test_zero_timeout_inline_tool_cannot_bypass_shared_capacity(tmp_path):
    capacity = RuntimeCapacity(max_tool_workers=1, reserved_manual_tool_workers=0)
    tool = BlockingTool()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="capacity")
    errors = []

    def run_inline():
        try:
            run_tools(db, capacity, tool, "inline", WorkOrigin.MANUAL, timeout=0)
        except BaseException as exc:  # noqa: BLE001 -- surface worker failures on the test thread
            errors.append(exc)

    thread = threading.Thread(target=run_inline)
    thread.start()
    try:
        assert tool.entered.wait(3)
        assert capacity.snapshot().tool_workers == 1
        _, refused = run_tools(db, capacity, tool, "other", WorkOrigin.MANUAL)
        assert "tool_capacity" in str(refused.calls)
        assert len(tool.workers) == 1
    finally:
        tool.release.set()
        thread.join(5)
        db.close()
    assert not errors
    assert not thread.is_alive()
    assert capacity.snapshot().executions == ()


def test_invalid_origin_is_not_inferred_from_strings(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="capacity")
    try:
        with pytest.raises(TypeError, match="WorkOrigin"):
            AgentService(db, ToolCatalogRegistry(), work_origin="manual")
    finally:
        db.close()
