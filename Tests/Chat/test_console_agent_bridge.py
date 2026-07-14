"""Console agent bridge: streaming, markers, spawn, supersede (fakes only)."""
import asyncio
import json

import pytest

from tldw_chatbook.Chat.console_agent_bridge import (
    CONSOLE_AGENT_OPERATING_PROMPT, ConsoleAgentBridge, compose_agent_system_prompt,
    format_agent_step_marker, inject_resume_agent_markers,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_models import STEP_ERROR, STEP_MODEL, STEP_SPAWN, STEP_TOOL_RESULT


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
        async def stream_chat(self, resolution, messages):
            seen_loops.append(asyncio.get_running_loop())
            async for chunk in super().stream_chat(resolution, messages):
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
