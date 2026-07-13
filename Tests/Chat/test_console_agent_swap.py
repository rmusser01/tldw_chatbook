"""The controller send path runs the agent loop when the bridge is wired."""
import json

import pytest

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


class _Gateway:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = 0

    async def resolve_for_send(self, selection):
        class _R:
            ready = True
            provider = "llama_cpp"
            visible_copy = ""
        return _R()

    async def stream_chat(self, resolution, messages):
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


def _controller(tmp_path, scripts, *, enabled=True):
    gateway = _Gateway(scripts)
    store = ConsoleChatStore()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, provider="llama_cpp", model="test-model",
        agent_bridge=bridge, agent_runtime_enabled=enabled)
    return controller, store, db


def _all_runs(db):
    """Read every persisted run record directly (AgentRunsDB has no list-all)."""
    with db.connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM agent_runs").fetchall()]


@pytest.mark.asyncio
async def test_agent_send_no_tools_streams_like_today(tmp_path):
    controller, store, _db = _controller(tmp_path, [["Tok", "yo."]])
    result = await controller.submit_draft("capital of Japan?")
    assert result.accepted is True
    messages = store.messages_for_session(store.active_session_id)
    assert messages[-1].role is ConsoleMessageRole.ASSISTANT
    assert messages[-1].content == "Tokyo."


@pytest.mark.asyncio
async def test_agent_tool_turn_renders_marker_not_prose(tmp_path):
    controller, store, _db = _controller(
        tmp_path, [[_fence("calculator", {"expression": "6*7"})], ["It is ", "42."]])
    await controller.submit_draft("what is 6*7?")
    messages = store.messages_for_session(store.active_session_id)
    assert any(m.role is ConsoleMessageRole.TOOL for m in messages)
    assert all(FENCE_OPEN not in m.content for m in messages
               if m.role is ConsoleMessageRole.ASSISTANT)


@pytest.mark.asyncio
async def test_stop_cancels_tree_and_persists_cancelled(tmp_path):
    controller, store, db = _controller(
        tmp_path, [[_fence("calculator", {"expression": "1"})], ["late"]])

    original = controller._agent_bridge.run_reply

    def cancel_after_first(*args, **kwargs):
        controller._stop_requested = True         # simulate Stop during the run
        return original(*args, **kwargs)

    controller._agent_bridge.run_reply = cancel_after_first
    await controller.submit_draft("loop please")

    primary = [r for r in _all_runs(db) if r["agent_kind"] == "primary"]
    assert primary and primary[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_config_gate_off_uses_legacy_path(tmp_path):
    controller, store, db = _controller(tmp_path, [["legacy answer."]], enabled=False)
    await controller.submit_draft("hi")
    messages = store.messages_for_session(store.active_session_id)
    assert messages[-1].content == "legacy answer."
    # Legacy path never touches AgentRunsDB.
    assert _all_runs(db) == []
