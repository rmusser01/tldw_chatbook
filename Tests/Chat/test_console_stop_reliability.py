"""TASK-337: Stop must freeze the message at the stop point.

The review's run A: Stop during a streaming agent reply left the message
rendering as streaming, and content the surviving bridge thread kept
generating was persisted — the saved message diverged from what the user
saw (UX review finding j4-stop-feedback-unreliable). These tests park the
REAL bridge thread mid-stream (task-227 rig), stop, then release the
thread and assert the message never grows past the stop point.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


class _ChunkThenParkGateway:
    """Stream real chunks, then park the bridge thread until released."""

    def __init__(self):
        self.first_chunks_fed = threading.Event()
        self.release = threading.Event()

    async def resolve_for_send(self, _selection):
        return SimpleNamespace(ready=True, provider="llama_cpp", visible_copy="")

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        yield "Once upon a "
        yield "time"
        self.first_chunks_fed.set()
        # Blocking wait on the bridge's own worker thread (see task-227 rig).
        self.release.wait(timeout=60)
        yield " — and two more"
        yield " paragraphs the user never saw."


def _controller(tmp_path, gateway):
    store = ConsoleChatStore()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=gateway
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    return controller, store


@pytest.mark.asyncio
async def test_stop_freezes_message_content_at_stop_point(tmp_path):
    gateway = _ChunkThenParkGateway()
    controller, store = _controller(tmp_path, gateway)

    send_task = asyncio.ensure_future(controller.submit_draft("hello"))
    for _ in range(3000):
        if gateway.first_chunks_fed.is_set():
            break
        await asyncio.sleep(0.01)
    assert gateway.first_chunks_fed.is_set()
    # Let the fed chunks materialize into the visible message.
    await asyncio.sleep(0.05)

    assert controller.stop_active_run() is True

    session_id = store.active_session_id
    assistant = next(
        m
        for m in store.messages_for_session(session_id)
        if m.role is ConsoleMessageRole.ASSISTANT
    )
    # Immediately after Stop: terminal status, run state stopped — this is
    # the synchronous acknowledgment the UI renders.
    assert assistant.status == "stopped"
    assert controller.run_state.status is ConsoleRunStatus.STOPPED
    content_at_stop = assistant.content

    result = await send_task
    assert result.accepted is True

    # The surviving bridge thread finishes its generation…
    gateway.release.set()
    for _ in range(200):
        await asyncio.sleep(0.02)
        if controller._active_stream_task is None:
            break
    await asyncio.sleep(0.3)

    # …but the stopped message must not grow past the stop point: what is
    # persisted must match what was displayed.
    final = store.get_message(assistant.id)
    assert final.status == "stopped"
    assert final.content == content_at_stop, (
        "post-stop generation leaked into the stopped message"
    )
    assert "never saw" not in final.content
