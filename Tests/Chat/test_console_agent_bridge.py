"""Console agent bridge: streaming, markers, spawn, supersede (fakes only)."""
import json

import pytest

from tldw_chatbook.Chat.console_agent_bridge import (
    CONSOLE_AGENT_OPERATING_PROMPT, ConsoleAgentBridge, compose_agent_system_prompt,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


class _ChunkGateway:
    """A gateway whose stream_chat replays a script keyed by call index."""

    def __init__(self, scripts):
        self._scripts = list(scripts)   # each entry: list[str] chunks for that turn
        self.calls = 0

    async def stream_chat(self, resolution, messages):
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


def _bridge(tmp_path, scripts):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts))
    return bridge, db, store, session, assistant.id


def _run(bridge, store, session, assistant_id, **over):
    kwargs = dict(
        conversation_id="conv-1", session_id=session.id, resolution=object(),
        assistant_message_id=assistant_id, model="test-model",
        session_system_prompt="", agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False)
    kwargs.update(over)
    return bridge.run_reply(**kwargs)


def test_compose_prepends_session_prompt_then_agent_prompt():
    composed = compose_agent_system_prompt("You are Ada.")
    assert composed.startswith("You are Ada.")
    assert CONSOLE_AGENT_OPERATING_PROMPT in composed
    assert compose_agent_system_prompt("") == CONSOLE_AGENT_OPERATING_PROMPT


def test_no_tool_message_streams_final_answer_like_today(tmp_path):
    bridge, _db, store, session, aid = _bridge(tmp_path, [["Tok", "yo."]])
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done" and outcome.final_text == "Tokyo."
    assert store.get_message(aid).content == "Tokyo."
    # No tool markers were appended.
    roles = [m.role for m in store.messages_for_session(session.id)]
    assert ConsoleMessageRole.TOOL not in roles


def test_tool_turn_renders_a_tool_marker_not_prose(tmp_path):
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],   # turn 1: leading fence
        ["It is ", "42."],                                # turn 2: final answer
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    tool_rows = [m for m in store.messages_for_session(session.id)
                 if m.role is ConsoleMessageRole.TOOL]
    assert tool_rows, "a tool turn must drop a TOOL marker"
    assert "calculator" in tool_rows[0].content
    # The fenced tool JSON never streamed into the assistant answer.
    assert FENCE_OPEN not in store.get_message(aid).content
    assert store.get_message(aid).content == "It is 42."


def test_spawn_renders_marker_and_persists_linked_subagent(tmp_path):
    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        ["2"],                                                 # sub-agent turn
        ["Done: ", "2."],                                     # primary final
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    assert db.count_subagent_runs("conv-1") == 1
    spawn_markers = [m for m in store.messages_for_session(session.id)
                     if m.role is ConsoleMessageRole.TOOL and "sub-agent" in m.content.lower()]
    assert spawn_markers
    snap = bridge.live_snapshot("conv-1")
    assert any(s.text for s in snap.subagents)


def test_supersede_marks_previous_primary_and_tree(tmp_path):
    bridge, db, store, session, aid = _bridge(tmp_path, [["one."], ["two."]])
    _run(bridge, store, session, aid)                        # first run
    first = db.list_runs("conv-1")[0]
    assert first["status"] == "done"
    # Second run supersedes the previous primary.
    aid2 = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT,
                                content="").id
    _run(bridge, store, session, aid2, supersede_previous=True)
    prior = db.get_run(first["id"])
    assert prior["status"] == "superseded"


def test_stop_persists_cancelled(tmp_path):
    # A long tool loop; cancel flips after the first step.
    scripts = [[_fence("calculator", {"expression": "1"})], ["never reached"]]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    flags = iter([False, True])
    outcome = _run(bridge, store, session, aid, should_cancel=lambda: next(flags, True))
    assert outcome.status == "cancelled"
    assert db.list_runs("conv-1")[0]["status"] == "cancelled"
