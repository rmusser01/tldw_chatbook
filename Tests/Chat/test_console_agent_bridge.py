"""Console agent bridge: streaming, markers, spawn, supersede (fakes only)."""
import asyncio
import contextlib
import json

import pytest

from tldw_chatbook.Chat.console_agent_bridge import (
    CONSOLE_AGENT_OPERATING_PROMPT, ConsoleAgentBridge, compose_agent_system_prompt,
    format_agent_step_marker, inject_resume_agent_markers, _compose_run_allowed_tools,
    _compose_run_registry_and_allowed, _non_colliding_mcp_names,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ProviderToolCalls
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_models import (
    LOAD_TOOLS_NAME, SPAWN_TOOL_NAME, STEP_ERROR, STEP_MODEL, STEP_SPAWN,
    STEP_TOOL_RESULT, ToolCatalogEntry, ToolResult, ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


class _FakeMCPProvider:
    """Minimal ``ToolProvider`` double standing in for a composed
    ``MCPToolProvider`` (T3) -- these bridge-level tests only need the
    catalog/invoke seam, not the real service/approval plumbing."""

    def __init__(self, entries):
        # entries: iterable of (name, description) pairs
        self._entries = list(entries)
        self.invoke_calls: list[tuple[str, dict]] = []
        self.stamp_scope_calls = 0

    def list_catalog(self):
        return [
            ToolCatalogEntry(id=name, name=name, one_line_description=desc, source="mcp")
            for name, desc in self._entries
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id, name=tool_id, description="",
            parameters={"type": "object", "properties": {}})

    def invoke(self, tool_id, args):
        self.invoke_calls.append((tool_id, dict(args or {})))
        return ToolResult(ok=True, content=f"mcp-result:{tool_id}")

    @contextlib.contextmanager
    def stamp_scope(self):
        # C1 (probe-verified security regression): stands in for
        # MCPToolProvider.stamp_scope -- a no-op snapshot/restore here since
        # this fake carries no per-turn stamp state of its own, just a call
        # counter so bridge-level wiring tests can assert `run_reply` threads
        # it through to AgentService(review_state_scope=...).
        self.stamp_scope_calls += 1
        yield


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


class _ChunkGateway:
    """A gateway whose stream_chat replays a script keyed by call index.

    Each scripted entry is a list of chunks, where a chunk is either a
    plain ``str`` (streamed text, as before) or a ``ProviderToolCalls``
    sentinel (native tool-calls, yielded as the final item of that turn).
    ``tools_seen`` records the ``tools=`` kwarg passed on each call, in
    call order, so tests can assert whether/what was forwarded.
    """

    def __init__(self, scripts):
        self._scripts = list(scripts)   # each entry: list of str and/or ProviderToolCalls
        self.calls = 0
        self.tools_seen = []

    async def stream_chat(self, resolution, messages, tools=None):
        self.tools_seen.append(tools)
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


class _NativeResolution:
    """A fake resolution whose execution_key resolves to a native-capable provider."""

    provider = "Groq"
    execution_key = "groq"


def _native_calls(name, args, call_id="c1"):
    return ProviderToolCalls(tool_calls=(
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}},))


def _bridge(tmp_path, scripts, native_tools_enabled=None):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        native_tools_enabled=native_tools_enabled)
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


def test_leaked_prose_before_disobedient_fence_is_reset_not_garbled(tmp_path):
    # Finding A repro: a disobedient turn streams prose live, THEN a tool
    # fence, in the same response. The gate has already forwarded the prose
    # to the store by the time the loop classifies the turn as a tool call.
    # That leaked prose must not survive to garble the real final answer
    # that streams onto the same assistant message afterward.
    scripts = [
        ["Let me check that ", "for you.\n```tool_call\n",
         '{"name": "calculator", "arguments": {"expression": "6*7"}}\n```'],
        ["42."],
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    assert outcome.final_text == "42."
    assert store.get_message(aid).content == "42."


# -- Task 5: native provider tool-calls through the streaming adapter,
# plus the [console] native_tool_calls kill-switch. --


def test_native_tool_call_round_trip_streams_final_answer(tmp_path):
    bridge, db, store, session, aid = _bridge(tmp_path, [
        [_native_calls("get_current_datetime", {})],
        ["It is ", "now."]])
    outcome = _run(bridge, store, session, aid, resolution=_NativeResolution())
    assert outcome.status == "done"
    assert store.get_message(aid).content == "It is now."
    gateway = bridge._gateway
    assert gateway.tools_seen[0] is not None          # tools= sent on turn 1
    names = [t["function"]["name"] for t in gateway.tools_seen[0]]
    assert "get_current_datetime" in names
    kinds = [step["kind"] for step in db.list_runs("conv-1")[0]["steps"]]
    assert "tool_call" in kinds and "tool_result" in kinds
    tool_rows = [m for m in store.messages_for_session(session.id)
                 if m.role is ConsoleMessageRole.TOOL]
    assert tool_rows, "a native tool turn must drop a TOOL marker too"
    assert "get_current_datetime" in tool_rows[0].content


def test_native_leaked_prose_is_reset_before_final_answer(tmp_path):
    """Prose streamed before the ProviderToolCalls arrives must not survive
    (Finding-A parity with the fence path)."""
    bridge, db, store, session, aid = _bridge(tmp_path, [
        ["Let me check. ", _native_calls("get_current_datetime", {})],
        ["Done."]])
    outcome = _run(bridge, store, session, aid, resolution=_NativeResolution())
    assert outcome.status == "done"
    assert store.get_message(aid).content == "Done."


def test_native_kill_switch_off_stays_on_fence_path(tmp_path):
    bridge, db, store, session, aid = _bridge(
        tmp_path,
        [[_fence("get_current_datetime", {})], ["Done."]],
        native_tools_enabled=lambda: False)
    outcome = _run(bridge, store, session, aid, resolution=_NativeResolution())
    assert outcome.status == "done"
    assert bridge._gateway.tools_seen[0] is None       # no tools= despite groq


def test_multi_turn_run_reuses_one_event_loop_across_chat_call_turns(tmp_path):
    """PR #629 Fix 1(c) (Gemini HIGH x2 + Qodo-8): ``_StreamingModelAdapter.
    chat_call`` used to bridge every turn via its own ``asyncio.run()`` --
    a fresh loop per turn, and therefore (per the gateway's per-loop
    ``_active_http_client`` swap) a client swap/churn on every single turn
    of a run. ``run_reply`` must create ONE event loop per invocation and
    reuse it for every turn -- the tool-call turn and the final-answer turn
    here -- so at most one swap happens per run."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],   # turn 1: tool call
        ["It is ", "42."],                                # turn 2: final answer
    ]
    seen_loops = []

    class _LoopSpyGateway(_ChunkGateway):
        async def stream_chat(self, resolution, messages, tools=None):
            seen_loops.append(asyncio.get_running_loop())
            async for chunk in super().stream_chat(resolution, messages, tools=tools):
                yield chunk

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_LoopSpyGateway(scripts))
    outcome = _run(bridge, store, session, assistant.id)

    assert outcome.status == "done"
    assert len(seen_loops) == 2, "both turns of this run must reach the gateway"
    assert seen_loops[0] is seen_loops[1], (
        "every turn of one run_reply invocation must share the same event "
        "loop -- a fresh loop per turn is exactly the per-turn churn Fix "
        "1(c) removes"
    )
    assert seen_loops[0].is_closed(), (
        "the run's shared loop must be closed once run_reply returns"
    )


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


def test_tool_marker_with_brackets_renders_literally_not_escaped(tmp_path):
    # Both TOOL-marker consumers (console_transcript.py's Content.assemble
    # and chat_screen.py's legacy Text(...) fallback) render markup-off, so
    # a bracketed task/result must survive as literal text -- not as a
    # backslash-escaped sequence (`fetch \[docs]`) that a markup parser
    # would need to consume but never runs.
    scripts = [
        [_fence("spawn_subagent", {"task": "fetch [docs]"})],  # primary turn 1
        ["ok"],                                                 # sub-agent turn
        ["Done."],                                             # primary final
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    spawn_markers = [m for m in store.messages_for_session(session.id)
                     if m.role is ConsoleMessageRole.TOOL and "sub-agent" in m.content.lower()]
    assert spawn_markers
    assert "[docs]" in spawn_markers[0].content
    assert "\\[docs]" not in spawn_markers[0].content


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


def test_stop_before_first_chunk_persists_cancelled_not_error(tmp_path):
    """Plan-B agent-runtime gate Finding 1: reproduces the live-repro'd race
    where Stop is clicked before the (slow) provider has streamed a single
    chunk. The controller's ``stop_active_run`` marks the assistant message
    "stopped" and flips ``should_cancel`` *before* the first chunk arrives;
    when it finally does, ``append_stream_chunk`` must drop it silently
    (store-level fix) instead of raising, so the run settles as
    "cancelled" (AgentRunsDB) rather than "error" with a step-log message
    of "Cannot append stream chunks to a stopped message.\""""
    scripts = [["late", " chunk", " arrives", " anyway."]]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)

    # Mirror ConsoleChatController.stop_active_run(): the message is
    # finalized to "stopped" up front, before any chunk streamed. The
    # first should_cancel() poll (at the top of the loop, before the model
    # call) still returns False -- the run genuinely starts -- then flips
    # True from the second poll onward, mirroring how ``_stop_requested``
    # was already True by the time the slow provider's first chunk
    # finally arrived inside ``_consume()``.
    store.mark_message_stopped(aid)
    flags = iter([False])
    outcome = _run(bridge, store, session, aid, should_cancel=lambda: next(flags, True))

    assert outcome.status == "cancelled"
    assert db.list_runs("conv-1")[0]["status"] == "cancelled"
    from tldw_chatbook.Agents.agent_models import STEP_ERROR
    assert not any(s.kind == STEP_ERROR for s in outcome.steps)
    # The message stays exactly as Stop left it -- no late content leaked in.
    stored = store.get_message(aid)
    assert stored.status == "stopped"
    assert stored.content == ""


def test_stop_mid_final_answer_persists_cancelled_and_store_agrees(tmp_path):
    # Finding B: a Stop that lands mid a plain final-answer stream (no tool
    # call to dispatch) must not be downgraded to "done" -- the outcome
    # status, the persisted AgentRunsDB row, and the store's own streamed
    # content must all agree that the run was cancelled.
    scripts = [["Par", "tial", " answer."]]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    flags = iter([False, True])
    outcome = _run(bridge, store, session, aid,
                   should_cancel=lambda: next(flags, True))
    assert outcome.status == "cancelled"
    assert db.list_runs("conv-1")[0]["status"] == "cancelled"
    assert store.get_message(aid).content == outcome.final_text


# -- Plan-B agent-runtime gate Finding 2: rail summary re-derived from
# AgentRunsDB after a restart, when this bridge instance has no in-process
# live-run record for the conversation. --


def test_historical_snapshot_idle_when_conversation_never_ran(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    snap = bridge.historical_snapshot("conv-never-seen")
    assert snap.status == "idle"
    assert snap.steps == ()
    assert snap.subagents == ()


def test_historical_snapshot_derives_status_steps_and_subagents_from_db(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(primary_id, [
        {"index": 0, "kind": "model", "summary": "The capital of France is Paris.",
         "tool_name": "", "args": None, "result": "", "created_at": ""},
    ])
    db.set_status(primary_id, "done", result="The capital of France is Paris.")
    sub_id = db.create_run(
        conversation_id="conv-1", agent_kind="subagent",
        task="research pricing", parent_run_id=primary_id)
    db.set_status(sub_id, "done", result="done researching")

    # Fresh bridge instance -- simulates an app restart: `_live` starts empty.
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    assert bridge.live_snapshot("conv-1").status == "idle"

    snap = bridge.historical_snapshot("conv-1")
    assert snap.status == "done"
    assert len(snap.steps) == 1
    assert "Paris" in snap.steps[0].text
    assert snap.steps[0].agent_kind == "primary"
    assert len(snap.subagents) == 1
    assert snap.subagents[0].text == "research pricing"
    assert snap.subagents[0].status == "done"


def test_historical_snapshot_ignores_subagents_of_other_runs(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.set_status(primary_id, "done", result="ok")
    other_primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.set_status(other_primary_id, "superseded")
    db.create_run(
        conversation_id="conv-1", agent_kind="subagent",
        task="orphaned", parent_run_id=other_primary_id)

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    snap = bridge.historical_snapshot("conv-1")
    assert snap.status == "done"
    assert snap.subagents == ()


def test_historical_snapshot_caches_per_conversation_not_hit_every_call(tmp_path, monkeypatch):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.set_status(primary_id, "done", result="ok")

    calls = []
    original = db.list_runs

    def spy(conversation_id, *args, **kwargs):
        calls.append(conversation_id)
        return original(conversation_id, *args, **kwargs)

    monkeypatch.setattr(db, "list_runs", spy)
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)

    first = bridge.historical_snapshot("conv-1")
    second = bridge.historical_snapshot("conv-1")
    assert first == second
    assert len(calls) == 1   # the 0.2s rail poll must not re-hit the DB

    # A different conversation is a separate cache entry.
    bridge.historical_snapshot("conv-2")
    assert len(calls) == 2


# -- Plan-B final-review Medium-1: inline transcript TOOL markers re-derive
# from AgentRunsDB on resume, the same way the rail already does. --


def test_format_agent_step_marker_matches_each_live_marker_shape():
    assert format_agent_step_marker(
        STEP_SPAWN, summary="research pricing") == "⤷ spawned sub-agent: research pricing"
    assert format_agent_step_marker(
        STEP_TOOL_RESULT, tool_name="calculator", result="42") == "⚙ calculator → 42"
    assert format_agent_step_marker(STEP_ERROR, summary="boom") == "⚠ boom"
    # Quiet tool-catalog steps and plain model steps never produce a marker.
    assert format_agent_step_marker(STEP_TOOL_RESULT, tool_name="find_tools", result="[]") is None
    assert format_agent_step_marker(STEP_TOOL_RESULT, tool_name="load_tools", result="[]") is None
    assert format_agent_step_marker(STEP_MODEL, summary="The answer is 42.") is None


def test_resume_marker_messages_reproduces_live_markers_after_simulated_restart(tmp_path):
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],   # turn 1: leading fence
        ["It is ", "42."],                                # turn 2: final answer
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    _run(bridge, store, session, aid)
    live_tool_contents = [
        m.content for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert live_tool_contents  # sanity: the live run actually left a marker

    # A fresh bridge instance -- simulates an app restart -- must re-derive
    # byte-identical marker text purely from AgentRunsDB.
    fresh_bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = fresh_bridge.resume_marker_messages("conv-1")
    resumed_tool_contents = [m.content for block in blocks for m in block]
    assert resumed_tool_contents == live_tool_contents


def test_resume_marker_messages_orders_blocks_chronologically_oldest_first(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    first = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(first, [
        {"index": 0, "kind": STEP_TOOL_RESULT, "tool_name": "calculator",
         "result": "4", "summary": "", "args": None, "created_at": ""},
    ])
    db.set_status(first, "done", result="4")
    second = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(second, [
        {"index": 0, "kind": STEP_ERROR, "summary": "timed out",
         "tool_name": "", "result": "", "args": None, "created_at": ""},
    ])
    db.set_status(second, "done", result="ok")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")
    assert len(blocks) == 2
    assert "calculator" in blocks[0][0].content
    assert "timed out" in blocks[1][0].content


def test_resume_marker_messages_skips_superseded_runs(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    superseded = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(superseded, [
        {"index": 0, "kind": STEP_ERROR, "summary": "old attempt",
         "tool_name": "", "result": "", "args": None, "created_at": ""},
    ])
    db.set_status(superseded, "superseded")
    kept = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(kept, [
        {"index": 0, "kind": STEP_ERROR, "summary": "final attempt",
         "tool_name": "", "result": "", "args": None, "created_at": ""},
    ])
    db.set_status(kept, "done", result="ok")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")
    assert len(blocks) == 1
    assert "final attempt" in blocks[0][0].content


def _tool_marker(text: str) -> ConsoleChatMessage:
    return ConsoleChatMessage(role=ConsoleMessageRole.TOOL, content=text, status="complete")


def test_inject_resume_agent_markers_places_block_after_matching_assistant_message():
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="42.", status="complete"),
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="again"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="ok.", status="complete"),
    ]
    blocks = [[_tool_marker("⚙ calculator → 42")], [_tool_marker("⚠ retry")]]

    result = inject_resume_agent_markers(messages, blocks)

    roles = [(m.role, m.content) for m in result]
    assert roles == [
        (ConsoleMessageRole.USER, "hi"),
        (ConsoleMessageRole.ASSISTANT, "42."),
        (ConsoleMessageRole.TOOL, "⚙ calculator → 42"),
        (ConsoleMessageRole.USER, "again"),
        (ConsoleMessageRole.ASSISTANT, "ok."),
        (ConsoleMessageRole.TOOL, "⚠ retry"),
    ]


def test_inject_resume_agent_markers_appends_leftover_block_when_more_runs_than_replies():
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="42.", status="complete"),
    ]
    blocks = [[_tool_marker("⚙ calculator → 42")], [_tool_marker("⚠ orphan run")]]

    result = inject_resume_agent_markers(messages, blocks)

    assert [m.content for m in result] == ["hi", "42.", "⚙ calculator → 42", "⚠ orphan run"]


def test_inject_resume_agent_markers_skips_empty_blocks():
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="ok.", status="complete"),
    ]
    result = inject_resume_agent_markers(messages, [[], []])
    assert [m.content for m in result] == ["ok."]


def test_inject_resume_agent_markers_is_idempotent_no_duplicates_on_second_call():
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="42.", status="complete"),
    ]
    blocks = [[_tool_marker("⚙ calculator → 42")]]

    once = inject_resume_agent_markers(messages, blocks)
    twice = inject_resume_agent_markers(once, blocks)

    tool_rows = [m.content for m in twice if m.role is ConsoleMessageRole.TOOL]
    assert tool_rows == ["⚙ calculator → 42"]
    assert len(once) == len(twice)


def test_inject_resume_agent_markers_leaves_live_session_with_markers_untouched():
    """A session that already carries live markers (this bridge ran the
    turn in-process rather than resuming) must be left byte-for-byte
    unchanged if this function is (defensively) called on it again."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="42.", status="complete"),
        _tool_marker("⚙ calculator → 42"),
    ]
    blocks = [[_tool_marker("⚙ calculator → 42")]]

    result = inject_resume_agent_markers(messages, blocks)

    assert result == messages


def test_resume_injects_markers_matching_live_format_end_to_end(tmp_path):
    """Fresh store + populated AgentRunsDB -> resuming reconstructs a
    transcript whose TOOL markers match live-format byte-for-byte, placed
    after the answer they belong to, and a second resume onto the
    already-injected transcript adds nothing more."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["It is ", "42."],
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    _run(bridge, store, session, aid)
    live_messages = store.messages_for_session(session.id)
    live_tool_contents = [
        m.content for m in live_messages if m.role is ConsoleMessageRole.TOOL
    ]

    # Simulate resume after a restart: the "ChaChaNotes-only" transcript
    # never carries markers (they persist=False), then inject markers
    # derived fresh from the DB via a brand-new bridge instance.
    chachanotes_only = [
        ConsoleChatMessage(role=m.role, content=m.content, status="complete")
        for m in live_messages if m.role is not ConsoleMessageRole.TOOL
    ]
    fresh_bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    resumed = inject_resume_agent_markers(
        chachanotes_only, fresh_bridge.resume_marker_messages("conv-1"))

    resumed_tool_contents = [m.content for m in resumed if m.role is ConsoleMessageRole.TOOL]
    assert resumed_tool_contents == live_tool_contents
    assistant_index = next(
        i for i, m in enumerate(resumed) if m.role is ConsoleMessageRole.ASSISTANT)
    assert resumed[assistant_index + 1].role is ConsoleMessageRole.TOOL

    resumed_again = inject_resume_agent_markers(
        resumed, fresh_bridge.resume_marker_messages("conv-1"))
    assert len(resumed_again) == len(resumed)


# -- Task 12: per-run spawn-wired skill executor + run allow-list composition --


class _FakeSkillsService:
    """Minimal async skills service: one trusted, model-invocable skill."""

    def __init__(self, *, skill_name="code-review", allowed_tools=None,
                 blocked=False):
        self.skill_name = skill_name
        self.allowed_tools = allowed_tools
        self.blocked = blocked
        self.execute_calls = []
        self.get_context_calls = 0

    async def get_context(self, *, mode="local"):
        self.get_context_calls += 1
        return {
            "available_skills": [
                {"name": self.skill_name, "description": "Review a diff",
                 "argument_hint": "[diff]", "trust_blocked": False,
                 "disable_model_invocation": False},
            ],
            "blocked_skills": [],
        }

    async def execute_skill(self, name, *, mode="local", args=None):
        self.execute_calls.append(args)
        if self.blocked:
            raise SkillTrustBlockedError(
                skill_name=name, reason_code="quarantined_modified",
                trust_status="quarantined_modified")
        return {
            "skill_name": name,
            "rendered_prompt": f"Review this: {args}",
            "allowed_tools": self.allowed_tools,
            "execution_mode": "inline",
        }


def test_skill_tool_call_routes_through_run_scoped_spawn(tmp_path):
    """A model-invoked skill tool runs as a budget-counted sub-agent of THIS
    run -- not SkillToolProvider.invoke (which raises by design), and not an
    unbounded/uncancellable bespoke path. The rendered skill prompt becomes
    the sub-agent's task, the sub-agent turn goes through the same scripted
    gateway as any other spawned sub-agent, and a TOOL marker records the
    call in the transcript exactly like any other tool call."""
    scripts = [
        [_fence("code-review", {"args": "the diff"})],  # primary calls the skill
        ["Looks fine to me."],                            # sub-agent turn
        ["All done."],                                    # primary final
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    skills_service = _FakeSkillsService()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service)

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-skill")

    assert outcome.status == "done"
    assert skills_service.execute_calls == ["the diff"]
    assert db.count_subagent_runs("conv-skill") == 1
    tool_rows = [m for m in store.messages_for_session(session.id)
                 if m.role is ConsoleMessageRole.TOOL]
    assert any("code-review" in row.content for row in tool_rows)


def test_skill_trust_blocked_refuses_without_spawning(tmp_path):
    """A skill whose trust was revoked between catalog build and model call
    refuses (re-verified at render time by execute_skill) -- no sub-agent is
    ever spawned, so the run tree never grows for a blocked call."""
    scripts = [
        [_fence("code-review", {"args": "x"})],
        ["I could not review that."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    skills_service = _FakeSkillsService(blocked=True)
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service)

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-blocked")

    assert outcome.status == "done"
    assert db.count_subagent_runs("conv-blocked") == 0


def test_no_skills_service_leaves_shared_registry_path_untouched(tmp_path):
    """The no-skills-service path (skills_service=None, the default) must
    stay byte-identical to the pre-Task-12 behavior: no get_context call, no
    skill_runner wiring, the bridge's own shared registry/allow-list used."""
    bridge, db, store, session, aid = _bridge(tmp_path, [["Tokyo."]])
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done" and outcome.final_text == "Tokyo."


def test_compose_run_allowed_tools_includes_eligible_skill_names():
    """Pure per-run allow-list: builtins, then eligible skill names, then
    spawn -- a trust-blocked or model-invocation-disabled skill is excluded."""
    context = {
        "available_skills": [
            {"name": "code-review", "trust_blocked": False,
             "disable_model_invocation": False},
            {"name": "needs-review", "trust_blocked": True,
             "disable_model_invocation": False},
            {"name": "user-only", "trust_blocked": False,
             "disable_model_invocation": True},
        ],
    }
    allowed = _compose_run_allowed_tools(context, ("calculator", "get_current_datetime"))
    assert allowed == (
        "calculator", "get_current_datetime", "code-review", SPAWN_TOOL_NAME)


def test_compose_run_allowed_tools_empty_context_is_builtins_plus_spawn():
    allowed = _compose_run_allowed_tools({}, ("calculator",))
    assert allowed == ("calculator", SPAWN_TOOL_NAME)


def test_compose_run_allowed_tools_builtin_shadows_same_named_skill():
    """Task 11 review note 2: a skill named the same as a builtin must never
    become a distinct, skill-routable tool -- the builtin always wins. The
    allow-list carries the name exactly once (from builtins), never twice."""
    context = {
        "available_skills": [
            {"name": "calculator", "trust_blocked": False,
             "disable_model_invocation": False},
        ],
    }
    allowed = _compose_run_allowed_tools(context, ("calculator", "get_current_datetime"))
    assert allowed == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)


def test_compose_run_allowed_tools_runtime_tool_name_shadows_same_named_skill():
    """Qodo finding 4 (PR #636 bot review): `_non_colliding_skill_entries`
    used to filter a skill's name only against `BuiltinToolProvider` names,
    not the loop's own in-loop runtime handler names (`find_tools`/
    `load_tools`/`spawn_subagent` -- `agent_models.RUNTIME_TOOL_NAMES`). A
    skill front-matter'd with one of those names would be advertised in the
    run's catalog/allow-list, then get hijacked by the loop's own
    name-based dispatch (`agent_runtime.run_agent_loop` checks
    `call.name == FIND_TOOLS_NAME` etc. before any registry/skill routing),
    making the skill permanently unreachable while still occupying a
    catalog slot with a misleading schema. The allow-list must exclude it
    exactly like a builtin-name collision does."""
    context = {
        "available_skills": [
            {"name": "find_tools", "trust_blocked": False,
             "disable_model_invocation": False},
        ],
    }
    allowed = _compose_run_allowed_tools(context, ("calculator", "get_current_datetime"))
    assert allowed == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)


def test_compose_run_registry_excludes_skill_named_like_a_runtime_tool():
    """Same collision, verified against the actual registry/allow-list this
    run would use: the skill must not appear as a distinct catalog entry
    (under either FIND_TOOLS_NAME or LOAD_TOOLS_NAME or SPAWN_TOOL_NAME)."""
    context = {
        "available_skills": [
            {"name": LOAD_TOOLS_NAME, "description": "d", "argument_hint": "",
             "trust_blocked": False, "disable_model_invocation": False},
        ],
    }
    registry, allowed_tools, builtin_names = _compose_run_registry_and_allowed(context)
    assert LOAD_TOOLS_NAME not in allowed_tools[len(builtin_names):]
    catalog_entries = [(entry.name, entry.source) for entry in registry.list_catalog()]
    assert (LOAD_TOOLS_NAME, "skill") not in catalog_entries


# -- P5-T6: MCPToolProvider registration + collision precedence --


def test_compose_run_registry_and_allowed_includes_mcp_entries_when_eligible():
    mcp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    registry, allowed_tools, _builtin_names = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp_provider)
    assert "mcp__srv_a__search" in allowed_tools
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert ("mcp__srv_a__search", "mcp") in catalog_entries
    result = registry.invoke_by_name("mcp__srv_a__search", {"query": "weather"})
    assert result.ok is True
    assert mcp_provider.invoke_calls == [("mcp__srv_a__search", {"query": "weather"})]


def test_compose_run_registry_and_allowed_absent_mcp_provider_is_unchanged():
    """`mcp_provider=None` (the default) must not add anything -- the
    pre-P5-T6 no-MCP behavior stays byte-identical."""
    registry, allowed_tools, _builtin_names = _compose_run_registry_and_allowed({})
    assert allowed_tools == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    assert len(registry.list_catalog()) == 2


def test_compose_run_registry_and_allowed_excludes_mcp_name_colliding_with_builtin():
    """Task 11 review note 2's shadowing precedent, extended to MCP: a
    builtin always wins a same-named MCP tool -- the name is carried
    exactly once (from the builtin), and invoking it never reaches the
    MCP fake."""
    mcp_provider = _FakeMCPProvider(
        [("calculator", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")])
    registry, allowed_tools, _builtin_names = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp_provider)
    assert allowed_tools.count("calculator") == 1
    assert "mcp__srv_a__search" in allowed_tools
    result = registry.invoke_by_name("calculator", {"expression": "1+1"})
    assert result.ok is True
    assert mcp_provider.invoke_calls == []  # the builtin handled it, not the MCP fake


def test_compose_run_registry_and_allowed_excludes_mcp_name_colliding_with_runtime_tool():
    """Qodo finding 4 (PR #636)'s shadowing precedent, extended to MCP: a
    tool named like one of the loop's own in-loop runtime handlers must
    never become a distinct, MCP-routable catalog entry."""
    mcp_provider = _FakeMCPProvider([(LOAD_TOOLS_NAME, "shadowing MCP tool")])
    registry, allowed_tools, builtin_names = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp_provider)
    assert LOAD_TOOLS_NAME not in allowed_tools[len(builtin_names):]
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert (LOAD_TOOLS_NAME, "mcp") not in catalog_entries


def test_compose_run_registry_and_allowed_excludes_mcp_name_colliding_with_skill():
    """A skill (registered before MCP) also wins a same-named MCP tool."""
    context = {
        "available_skills": [
            {"name": "code-review", "description": "Review a diff",
             "argument_hint": "", "trust_blocked": False,
             "disable_model_invocation": False},
        ],
    }
    mcp_provider = _FakeMCPProvider(
        [("code-review", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")])
    registry, allowed_tools, _builtin_names = _compose_run_registry_and_allowed(
        context, mcp_provider=mcp_provider)
    assert allowed_tools.count("code-review") == 1
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert ("code-review", "skill") in catalog_entries
    assert ("code-review", "mcp") not in catalog_entries
    assert "mcp__srv_a__search" in allowed_tools


def test_compose_run_registry_and_allowed_all_mcp_names_colliding_skips_registration():
    """When every MCP entry collides, the provider is not registered at
    all -- no dangling catalog entries the model could never reach."""
    mcp_provider = _FakeMCPProvider([("calculator", "shadowing MCP tool")])
    registry, allowed_tools, _builtin_names = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp_provider)
    assert allowed_tools == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert ("calculator", "mcp") not in catalog_entries


def test_non_colliding_mcp_names_pure_helper():
    mcp_provider = _FakeMCPProvider([("calculator", "x"), ("mcp__srv__y", "y")])
    assert _non_colliding_mcp_names(mcp_provider, {"calculator"}) == ("mcp__srv__y",)


def test_run_reply_routes_fence_call_to_mcp_provider(tmp_path):
    """End-to-end: a run with no skills service still registers an eligible
    MCP provider fresh (not the shared, construction-time registry) and
    dispatches a matching fence call to it."""
    scripts = [
        [_fence("mcp__srv_a__search", {"query": "weather"})],
        ["The weather is nice."],
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    mcp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])

    outcome = _run(bridge, store, session, aid, mcp_provider=mcp_provider)

    assert outcome.status == "done"
    assert outcome.final_text == "The weather is nice."
    assert mcp_provider.invoke_calls == [("mcp__srv_a__search", {"query": "weather"})]
    tool_rows = [m for m in store.messages_for_session(session.id)
                 if m.role is ConsoleMessageRole.TOOL]
    assert any("mcp__srv_a__search" in row.content for row in tool_rows)


def test_run_reply_forwards_review_tool_calls_hook_to_agent_service(tmp_path):
    """`review_tool_calls=` must reach AgentService/the loop -- a batch
    verdict other than "proceed" skips dispatch and becomes the tool
    result, exactly like the T4 hook contract documents."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["done."],
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    captured_batches = []

    def hook(calls):
        captured_batches.append(list(calls))
        return {"calculator": "blocked by test hook"}

    outcome = _run(bridge, store, session, aid, review_tool_calls=hook)

    assert outcome.status == "done"
    assert captured_batches and captured_batches[0][0].name == "calculator"
    tool_rows = [m for m in store.messages_for_session(session.id)
                 if m.role is ConsoleMessageRole.TOOL]
    assert any("blocked by test hook" in row.content for row in tool_rows)


def test_run_reply_wires_mcp_provider_stamp_scope_around_a_spawned_child(tmp_path):
    """C1 (probe-verified security regression): `run_reply` must thread
    `mcp_provider.stamp_scope` through to `AgentService(review_state_scope=
    ...)` whenever an MCP provider is composed for this run -- that seam is
    what protects a parent turn's MCP approval stamps from being clobbered by
    a sub-agent's own inline nested run (see `MCPToolProvider.stamp_scope`'s
    own docstring and `Tests/Agents/test_agent_service_review_state_scope.py`
    for the full adversarial reproduction). This is the bridge-level wiring
    check: a run that spawns exactly one sub-agent must enter/exit the
    composed MCP provider's `stamp_scope()` exactly once around that spawn."""
    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        ["2"],                                                 # sub-agent turn
        ["Done: ", "2."],                                     # primary final
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    mcp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])

    outcome = _run(bridge, store, session, aid, mcp_provider=mcp_provider)

    assert outcome.status == "done"
    assert mcp_provider.stamp_scope_calls == 1


def test_skill_named_like_a_runtime_tool_never_shadows_it_at_invocation(tmp_path):
    """End-to-end: a skill front-matter'd as "find_tools" must not hijack
    the runtime's own find_tools meta-tool -- the real runtime dispatch
    still answers, and the skill is never invoked (it's excluded from the
    run's catalog/allow-list entirely, so no sub-agent is ever spawned for
    what looks like a skill call)."""
    scripts = [
        [_fence("find_tools", {"query": "anything"})],
        ["done."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    skills_service = _FakeSkillsService(skill_name="find_tools")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service)

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-runtime-collide")

    assert outcome.status == "done"
    assert skills_service.execute_calls == []          # the skill was never invoked
    assert db.count_subagent_runs("conv-runtime-collide") == 0


def test_skill_named_like_a_builtin_never_shadows_it_at_invocation(tmp_path):
    """End-to-end: a skill front-matter'd as "calculator" must not hijack
    calculator calls -- the real builtin still answers, and no sub-agent
    is spawned for what looks like a skill call."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["It is 42."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    skills_service = _FakeSkillsService(skill_name="calculator")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service)

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-collide")

    assert outcome.status == "done"
    assert skills_service.execute_calls == []          # the skill was never invoked
    assert db.count_subagent_runs("conv-collide") == 0  # no sub-agent spawned
    tool_rows = [m for m in store.messages_for_session(session.id)
                 if m.role is ConsoleMessageRole.TOOL]
    assert any("42" in row.content for row in tool_rows)


# -- Skills Phase-2 gate finding 1: discovery-heavy runs must not exhaust the
# bare engine step budget right after a successful skill call, before the
# final wrap-up reply (Task-14 gate scenario 5: "Find a skill that can
# shout, load it, and use it on: hello"). --


class _ManySkillsService:
    """9 real skills (> DIRECT_DISCLOSE_THRESHOLD == 8), so the catalog
    defers everything to find_tools/load_tools -- the exact >8-skill shape
    that engaged progressive disclosure in the live gate capture."""

    def __init__(self):
        self.execute_calls = []

    async def get_context(self, *, mode="local"):
        names = ["shout"] + [f"filler{i}" for i in range(8)]
        return {
            "available_skills": [
                {"name": n, "description": f"{n} skill", "argument_hint": "[args]",
                 "trust_blocked": False, "disable_model_invocation": False}
                for n in names
            ],
            "blocked_skills": [],
        }

    async def execute_skill(self, name, *, mode="local", args=None):
        self.execute_calls.append((name, args))
        return {
            "skill_name": name,
            "rendered_prompt": f"SHOUT[{args}]",
            "allowed_tools": None,
            "execution_mode": "inline",
        }


def _discovery_heavy_shout_scripts():
    # Mirrors the gate's live raw step log: find_tools({"query": "shout"})
    # -> load_tools({"ids": ["skill:shout"]}) -> shout({"args": "hello"})
    # -> the sub-agent's own turn -> the primary's final wrap-up reply.
    return [
        [_fence("find_tools", {"query": "shout"})],
        [_fence("load_tools", {"ids": ["skill:shout"]})],
        [_fence("shout", {"args": "hello"})],
        ["HELLO"],              # sub-agent turn (never streamed to the store)
        ["Shouted: HELLO"],     # primary final answer
    ]


def test_discovery_heavy_skill_run_completes_done_not_stuck(tmp_path):
    """Task-14 gate finding 1 repro: find_tools -> load_tools -> a skill
    call -> final answer needs exactly 10 primary-loop steps at minimum (3
    steps per tool round x 3 rounds, plus 1 final model turn -- see
    agent_runtime.run_agent_loop's per-round STEP_MODEL/STEP_TOOL_CALL/
    STEP_TOOL_RESULT accounting). The bare engine default
    (agent_models.RunBudget.max_steps == 8, pinned by
    test_agent_models.test_budget_defaults) is ONE ROUND short of that --
    it exhausts right after the skill's tool_result, one step before the
    wrap-up reply, even though every tool call already succeeded. The
    Console bridge must give this exact shape enough headroom to actually
    reach the final answer and persist `done`."""
    scripts = _discovery_heavy_shout_scripts()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    skills_service = _ManySkillsService()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service)

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-discover")

    assert outcome.status == "done"
    assert outcome.final_text == "Shouted: HELLO"
    assert skills_service.execute_calls == [("shout", "hello")]
    assert db.count_subagent_runs("conv-discover") == 1
    tool_rows = [m for m in store.messages_for_session(session.id)
                 if m.role is ConsoleMessageRole.TOOL]
    assert any("shout" in row.content for row in tool_rows)


def test_console_run_budget_is_raised_above_the_bare_engine_default(tmp_path):
    """Pins the config-assembly override directly: a primary Console run's
    PERSISTED budget must sit strictly above the engine's own pure default
    (RunBudget().max_steps == 8 -- see test_agent_models.test_budget_defaults,
    which stays unchanged) -- with enough headroom (>= 16, per the counted
    10-step discovery-heavy floor above) to survive a real disclosure run,
    and a proportionally raised wall-clock allowance for the extra turns."""
    bridge, db, store, session, aid = _bridge(tmp_path, [["hi there"]])
    _run(bridge, store, session, aid, conversation_id="conv-budget")
    run = db.list_runs("conv-budget")[0]
    assert run["agent_kind"] == "primary"
    assert run["budget"]["max_steps"] > 8
    assert run["budget"]["max_steps"] >= 16
    assert run["budget"]["max_wall_seconds"] > 240.0
