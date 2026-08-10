"""Console agent bridge: streaming, markers, spawn, supersede (fakes only)."""

import asyncio
import contextlib
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tldw_chatbook.Chat.console_agent_bridge import (
    CONSOLE_AGENT_OPERATING_PROMPT,
    FIND_LOAD_DISCOVERY_HINT,
    ConsoleAgentBridge,
    compose_agent_system_prompt,
    format_agent_step_marker,
    format_todo_marker,
    inject_resume_agent_markers,
    _BridgeSkillRunner,
    _compose_run_allowed_tools,
    _compose_run_registry_and_allowed,
    _non_colliding_mcp_names,
    _WARNED_SHADOWED_MCP_NAMES,
    shadowed_mcp_names,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderStreamSignals,
    ProviderToolCalls,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_models import (
    DIRECT_DISCLOSE_THRESHOLD,
    LOAD_TOOLS_NAME,
    RUN_DONE,
    RUN_ERROR,
    SPAWN_TOOL_NAME,
    STEP_ERROR,
    STEP_MODEL,
    STEP_SPAWN,
    STEP_TOOL_RESULT,
    RunOutcome,
    SkillFileBindings,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.run_context import current_run_id
from tldw_chatbook.Agents.tool_catalog import (
    SkillToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError

from Tests.Agents.test_agent_service import SUBAGENT_PROMPT_PREFIX


@pytest.fixture(autouse=True)
def _reset_shadowed_mcp_warning_dedup():
    """Reset the shadowed-MCP-name log-once set so tests are order-independent.

    Mirrors ``Tests/Internal_Prompts/conftest.py``'s
    ``resolver._warned_ids.clear()`` idiom for the identical reason:
    ``_compose_run_registry_and_allowed``'s shadowed-name warning is now
    deduped per name for the life of the process (finding 8, substrate
    review). Without this reset, whichever test in this file runs FIRST
    for a given shadowed name would win, and every later test asserting
    the same warning would silently observe nothing logged.
    """
    _WARNED_SHADOWED_MCP_NAMES.clear()
    yield
    _WARNED_SHADOWED_MCP_NAMES.clear()


class _FakeMCPProvider:
    """Minimal ``ToolProvider`` double standing in for a composed
    ``MCPToolProvider`` (T3) -- these bridge-level tests only need the
    catalog/invoke seam, not the real service/approval plumbing."""

    def __init__(self, entries):
        # entries: iterable of (name, description) pairs
        self._entries = list(entries)
        self.invoke_calls: list[tuple[str, dict]] = []
        self.stamp_scope_calls = 0
        self.list_catalog_calls = 0

    def list_catalog(self):
        self.list_catalog_calls += 1
        return [
            ToolCatalogEntry(
                id=name, name=name, one_line_description=desc, source="mcp"
            )
            for name, desc in self._entries
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id,
            description="",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        self.invoke_calls.append((tool_id, dict(args or {})))
        return ToolResult(ok=True, content=f"mcp-result:{tool_id}")

    @contextlib.contextmanager
    def stamp_scope(self, run_id):
        # C1 (probe-verified security regression): stands in for
        # MCPToolProvider.stamp_scope -- a no-op snapshot/restore here since
        # this fake carries no per-turn stamp state of its own, just a call
        # counter so bridge-level wiring tests can assert `run_reply` threads
        # it through to AgentService(review_state_scope=...). PR2a Task 5:
        # the scope takes the run id whose slice it guards.
        self.stamp_scope_calls += 1
        yield


def _fence(name, args):
    return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"


class _ChunkGateway:
    """A gateway whose stream_chat replays a script keyed by call index.

    Each scripted entry is a list of chunks, where a chunk is either a
    plain ``str`` (streamed text, as before) or a ``ProviderToolCalls``
    sentinel (native tool-calls, yielded as the final item of that turn).
    ``tools_seen`` records the ``tools=`` kwarg passed on each call, in
    call order, so tests can assert whether/what was forwarded.
    """

    def __init__(self, scripts):
        self._scripts = list(
            scripts
        )  # each entry: list of str and/or ProviderToolCalls
        self.calls = 0
        self.tools_seen = []

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        self.tools_seen.append(tools)
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk


class _FleetChunkGateway:
    """A gateway that addresses scripts by AGENT, not by arrival order.

    PR2a Task 6.5: `_ChunkGateway` indexes its scripts by call count
    (`self._scripts[self.calls]`), which stops being deterministic once the
    fleet is ON by default -- a spawned child runs on its own thread and
    its turn can land before, after, or between the parent's. This keeps
    the same chunk-list-per-turn shape but keeps ONE queue for the primary
    agent and one per child, so each turn's script reaches the agent it was
    written for.

    Children are identified exactly as ``_StreamingModelAdapter.
    _is_subagent`` identifies them -- a system prompt starting with the
    sub-agent prompt -- so the addressing follows the production contract
    rather than a test-only convention.

    Args:
        parent_script: turns for the primary agent, in order.
        child_script: turns for THE child, in order. These suites spawn
            exactly one, and its task text is an implementation detail of
            whatever spawned it (a skill's rendered prompt, say), so
            addressing on "is a child" is both sufficient and stabler than
            addressing on that text.
    """

    def __init__(self, parent_script, child_script=()):
        self._parent = list(parent_script)
        self._child = list(child_script)
        self.calls = 0
        self.tools_seen = []
        self.child_calls = 0
        self._lock = threading.Lock()

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        is_child = system.startswith(SUBAGENT_PROMPT_PREFIX)
        with self._lock:
            self.tools_seen.append(tools)
            self.calls += 1
            if is_child:
                self.child_calls += 1
                assert self._child, "child script exhausted"
                chunks = self._child.pop(0)
            else:
                assert self._parent, "parent script exhausted"
                chunks = self._parent.pop(0)
        # Resolved OUTSIDE the lock: a turn may be a zero-arg callable that
        # BLOCKS on an Event, which is how a test pins an interleaving; the
        # lock would serialize the very concurrency under test.
        if callable(chunks):
            chunks = chunks()
        for chunk in chunks:
            yield chunk


def _join_fleet_threads(timeout=5.0):
    """Block until every live fleet child thread has fully finished.

    `AgentService` names them ``fleet-<handle>``. "Fully finished" is the
    point: a child's own run row goes terminal slightly BEFORE its thread
    unwinds, so joining the thread -- not polling the DB -- is what
    guarantees any context manager wrapping that run has already exited.
    """
    for thread in list(threading.enumerate()):
        if thread.name.startswith("fleet-"):
            thread.join(timeout)


class _SignalChunkGateway(_ChunkGateway):
    """Scripted gateway that records the out-of-band signal by identity."""

    def __init__(self, scripts):
        super().__init__(scripts)
        self.signals_seen = []
        self.signal_states_seen = []

    async def stream_chat(self, resolution, messages, tools=None, signals=None):
        self.signals_seen.append(signals)
        self.signal_states_seen.append(signals.synthetic_fallback_emitted)
        async for chunk in super().stream_chat(
            resolution,
            messages,
            tools=tools,
        ):
            yield chunk


class _NativeResolution:
    """A fake resolution whose execution_key resolves to a native-capable provider."""

    provider = "Groq"
    execution_key = "groq"


def _native_calls(name, args, call_id="c1"):
    return ProviderToolCalls(
        tool_calls=(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            },
        )
    )


def _bridge(tmp_path, scripts, native_tools_enabled=None):
    return _bridge_with_gateway(
        tmp_path,
        _ChunkGateway(scripts),
        native_tools_enabled=native_tools_enabled,
    )


def _bridge_with_gateway(tmp_path, gateway, native_tools_enabled=None):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=native_tools_enabled,
    )
    return bridge, db, store, session, assistant.id


def _run(bridge, store, session, assistant_id, **over):
    kwargs = dict(
        conversation_id="conv-1",
        session_id=session.id,
        resolution=object(),
        assistant_message_id=assistant_id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
    )
    kwargs.update(over)
    # run_reply returns (run_id, outcome); these tests assert on the outcome.
    _run_id, outcome = bridge.run_reply(**kwargs)
    return outcome


def test_compose_prepends_session_prompt_then_agent_prompt():
    composed = compose_agent_system_prompt("You are Ada.")
    assert composed.startswith("You are Ada.")
    assert CONSOLE_AGENT_OPERATING_PROMPT in composed
    assert compose_agent_system_prompt("") == CONSOLE_AGENT_OPERATING_PROMPT


def test_compose_appends_discovery_hint_only_when_find_load_offered():
    # Default (direct disclosure): no hint — find/load is not the live mode.
    assert FIND_LOAD_DISCOVERY_HINT not in compose_agent_system_prompt("You are Ada.")
    assert FIND_LOAD_DISCOVERY_HINT not in compose_agent_system_prompt("")
    # Past the threshold the caller flags find/load mode: the hint is
    # appended after the operating prompt, session prompt still first.
    composed = compose_agent_system_prompt("You are Ada.", offer_find_load=True)
    assert composed.startswith("You are Ada.")
    assert CONSOLE_AGENT_OPERATING_PROMPT in composed
    assert composed.endswith(FIND_LOAD_DISCOVERY_HINT)
    blank = compose_agent_system_prompt("", offer_find_load=True)
    assert blank.startswith(CONSOLE_AGENT_OPERATING_PROMPT)
    assert blank.endswith(FIND_LOAD_DISCOVERY_HINT)


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
        [_fence("calculator", {"expression": "6*7"})],  # turn 1: leading fence
        ["It is ", "42."],  # turn 2: final answer
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows, "a tool turn must drop a TOOL marker"
    assert "calculator" in tool_rows[0].content
    # The fenced tool JSON never streamed into the assistant answer.
    assert FENCE_OPEN not in store.get_message(aid).content
    assert store.get_message(aid).content == "It is 42."


class _RefusingBuiltinGate:
    """A `BuiltinToolGate` double that refuses every call -- proves
    `run_reply`'s `builtin_gate=` argument is the SAME object
    `BuiltinToolProvider.invoke()` ends up checking, end to end."""

    def __init__(self) -> None:
        self.checked: list[str] = []

    def check(self, tool, run_id):
        self.checked.append(tool.name)
        return f"disabled for test: {tool.name}"


def test_run_reply_threads_builtin_gate_end_to_end(tmp_path):
    """task-545/T6: a `builtin_gate` handed to `run_reply` must be the
    exact instance the run's own `BuiltinToolProvider.invoke()` consults --
    a second, independently-built gate would silently desync from
    whatever the caller's review hook already decided (the core risk this
    task's wiring exists to avoid)."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["it was refused."],
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    gate = _RefusingBuiltinGate()
    outcome = _run(bridge, store, session, aid, builtin_gate=gate)
    assert outcome.status == "done"
    assert gate.checked == ["calculator"]
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows, "a refused tool call still drops a TOOL marker"
    assert "disabled for test: calculator" in tool_rows[0].content
    assert store.get_message(aid).content == "it was refused."


def test_run_reply_threads_session_workspace_id_end_to_end(tmp_path, monkeypatch):
    """task-6 (settings-workspaces-folder-roots spec Sec3): the RUNNING
    session's own ``workspace_id`` must reach the run's
    ``BuiltinToolProvider``, so a builtin tool observes it (via
    ``workspace_file_roots.run_workspace``) while it executes -- not
    whatever workspace happens to be active in the UI by the time the
    call actually fires. Verified by monkeypatching ``CalculatorTool.
    execute`` (the real built-in tool the scripted fence calls) to record
    the ContextVar it sees, since the bridge builds its own
    ``BuiltinToolProvider`` internally with no seam for a test double."""
    from tldw_chatbook.Tools import workspace_file_roots as wfr
    from tldw_chatbook.Tools.tool_executor import CalculatorTool

    observed: list[str | None] = []
    real_execute = CalculatorTool.execute

    async def _recording_execute(self, expression):
        observed.append(wfr.current_run_workspace_id())
        return await real_execute(self, expression)

    monkeypatch.setattr(CalculatorTool, "execute", _recording_execute)

    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["It is 42."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.create_session(workspace_id="ws-session-42")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts)
    )
    # Any non-None `builtin_gate` forces the fresh-build branch that
    # actually looks up and threads the session's workspace_id (see
    # `run_reply`'s own docstring) -- the shared/no-gate fast path's
    # provider is built once at bridge-construction time, before any
    # session exists, so it is out of scope for a per-session binding.
    outcome = _run(
        bridge, store, session, assistant.id,
        builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
    )
    assert outcome.status == "done"
    assert observed == ["ws-session-42"]
    assert wfr.current_run_workspace_id() is None  # cleared after the run


def test_run_reply_refuses_write_file_in_an_ephemeral_session_end_to_end(
    tmp_path, monkeypatch
):
    """F4 (final-review): agent tool calls are a 9th, ungated local-write
    sink -- an ordinary reply in a temporary Console session can compose
    and dispatch `write_file` (a gateable built-in) exactly like any other
    session, independently of the Console UI action-id registry in
    `Chat/console_ephemeral.py`. Verified end-to-end through `run_reply`,
    mirroring `test_run_reply_threads_session_workspace_id_end_to_end`'s
    own pattern (the bridge builds its own `BuiltinToolProvider` internally
    with no seam for a test double) -- the RUNNING session's `ephemeral`
    flag must reach it via `_store.session_is_ephemeral`.
    """
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Tools.file_operation_tools import WriteFileTool

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True
            if section == "tools" and key == "write_file_enabled"
            else default
        ),
    )

    called = {"n": 0}

    async def _recording_execute(self, **kwargs):
        called["n"] += 1
        return {"success": True}

    monkeypatch.setattr(WriteFileTool, "execute", _recording_execute)

    scripts = [
        [_fence("write_file", {"file_path": "note.txt", "content": "hi"})],
        ["I could not save that."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts)
    )
    outcome = _run(
        bridge, store, session, assistant.id,
        builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
    )
    assert outcome.status == "done"
    assert called["n"] == 0, "write_file must never execute in a temporary chat"
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows, "a refused tool call still drops a TOOL marker"
    assert "temporary chat" in tool_rows[0].content

    # CONTROL: the identical scripted call executes normally outside a
    # temporary chat.
    called["n"] = 0
    db2 = AgentRunsDB(tmp_path / "runs2.db", client_id="t")
    store2 = ConsoleChatStore()
    normal_session = store2.create_session()
    store2.append_message(
        normal_session.id, role=ConsoleMessageRole.USER, content="hi"
    )
    normal_assistant = store2.append_message(
        normal_session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    scripts2 = [
        [_fence("write_file", {"file_path": "note.txt", "content": "hi"})],
        ["Saved."],
    ]
    bridge2 = ConsoleAgentBridge(
        agent_runs_db=db2, store=store2, provider_gateway=_ChunkGateway(scripts2)
    )
    outcome2 = _run(
        bridge2, store2, normal_session, normal_assistant.id,
        builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
    )
    assert outcome2.status == "done"
    assert called["n"] == 1, "write_file must execute normally outside a temporary chat"


def test_leaked_prose_before_disobedient_fence_is_reset_not_garbled(tmp_path):
    # Finding A repro: a disobedient turn streams prose live, THEN a tool
    # fence, in the same response. The gate has already forwarded the prose
    # to the store by the time the loop classifies the turn as a tool call.
    # That leaked prose must not survive to garble the real final answer
    # that streams onto the same assistant message afterward.
    scripts = [
        [
            "Let me check that ",
            "for you.\n```tool_call\n",
            '{"name": "calculator", "arguments": {"expression": "6*7"}}\n```',
        ],
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
    bridge, db, store, session, aid = _bridge(
        tmp_path, [[_native_calls("get_current_datetime", {})], ["It is ", "now."]]
    )
    outcome = _run(bridge, store, session, aid, resolution=_NativeResolution())
    assert outcome.status == "done"
    assert store.get_message(aid).content == "It is now."
    gateway = bridge._gateway
    assert gateway.tools_seen[0] is not None  # tools= sent on turn 1
    names = [t["function"]["name"] for t in gateway.tools_seen[0]]
    assert "get_current_datetime" in names
    kinds = [step["kind"] for step in db.list_runs("conv-1")[0]["steps"]]
    assert "tool_call" in kinds and "tool_result" in kinds
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert tool_rows, "a native tool turn must drop a TOOL marker too"
    assert "get_current_datetime" in tool_rows[0].content


def test_native_leaked_prose_is_reset_before_final_answer(tmp_path):
    """Prose streamed before the ProviderToolCalls arrives must not survive
    (Finding-A parity with the fence path)."""
    bridge, db, store, session, aid = _bridge(
        tmp_path,
        [["Let me check. ", _native_calls("get_current_datetime", {})], ["Done."]],
    )
    outcome = _run(bridge, store, session, aid, resolution=_NativeResolution())
    assert outcome.status == "done"
    assert store.get_message(aid).content == "Done."


def test_native_kill_switch_off_stays_on_fence_path(tmp_path):
    bridge, db, store, session, aid = _bridge(
        tmp_path,
        [[_fence("get_current_datetime", {})], ["Done."]],
        native_tools_enabled=lambda: False,
    )
    outcome = _run(bridge, store, session, aid, resolution=_NativeResolution())
    assert outcome.status == "done"
    assert bridge._gateway.tools_seen[0] is None  # no tools= despite groq


def test_multi_turn_run_reuses_one_event_loop_across_chat_call_turns(tmp_path):
    """PR #629 Fix 1(c) (Gemini HIGH x2 + Qodo-8): ``_StreamingModelAdapter.
    chat_call`` used to bridge every turn via its own ``asyncio.run()`` --
    a fresh loop per turn, and therefore (per the gateway's per-loop
    ``_active_http_client`` swap) a client swap/churn on every single turn
    of a run. ``run_reply`` must create ONE event loop per invocation and
    reuse it for every turn -- the tool-call turn and the final-answer turn
    here -- so at most one swap happens per run."""
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],  # turn 1: tool call
        ["It is ", "42."],  # turn 2: final answer
    ]
    seen_loops = []

    class _LoopSpyGateway(_ChunkGateway):
        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            seen_loops.append(asyncio.get_running_loop())
            async for chunk in super().stream_chat(resolution, messages, tools=tools):
                yield chunk

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_LoopSpyGateway(scripts)
    )
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


def test_a_concurrent_child_can_use_the_runs_shared_event_loop(tmp_path):
    """PR2a Task 6.5: a fleet child's turn must survive overlapping the
    parent's on the run's ONE shared event loop.

    Found by probe when the fleet default was flipped, and it failed every
    overlapping child: `chat_call` drove the shared loop with
    `run_until_complete` from whichever thread was calling, which is only
    sound while the whole run tree is single-threaded. A loop may be driven
    by exactly one thread, so the second concurrent caller got
    ``RuntimeError: This event loop is already running``, its coroutine was
    dropped un-awaited, and the child's run row persisted `error` -- silent
    from the parent's side, which just saw a failed sub-agent. `chat_call`
    now submits with `run_coroutine_threadsafe` to a loop running on its
    own thread.

    The gateway awaits inside every turn so the parent and the child really
    are in flight together (a turn that yields without awaiting can slip
    through the window); the child additionally blocks until the parent has
    entered its own next turn, so the overlap is pinned rather than raced
    for. PR #629 Fix 1(c) is re-asserted here too: it is still ONE loop.
    """
    parent_in_flight = threading.Event()
    seen_loops = []

    class _OverlappingGateway(_FleetChunkGateway):
        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            system = str(messages[0].get("content", "")) if messages else ""
            if not system.startswith(SUBAGENT_PROMPT_PREFIX):
                parent_in_flight.set()
            seen_loops.append(asyncio.get_running_loop())
            # Yield control so the other agent's coroutine can be scheduled
            # onto this same loop while this one is still open.
            await asyncio.sleep(0.01)
            async for chunk in super().stream_chat(resolution, messages, tools=tools):
                yield chunk

    def gated_child_turn():
        assert parent_in_flight.wait(5), "the parent never started a turn"
        return ["child answer"]

    gateway = _OverlappingGateway(
        [
            [_fence("spawn_subagent", {"task": "child work"})],
            [_fence("wait_agents", {})],
            ["parent final"],
        ],
        [gated_child_turn],
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=gateway
    )

    outcome = _run(
        bridge, store, session, assistant.id, conversation_id="conv-overlap"
    )

    assert outcome.status == "done"
    assert outcome.final_text == "parent final"
    # The child completed -- it did NOT die on the loop.
    child_runs = [
        r for r in db.list_runs("conv-overlap") if r["agent_kind"] == "subagent"
    ]
    assert len(child_runs) == 1
    assert child_runs[0]["status"] == RUN_DONE
    assert child_runs[0]["result"] == "child answer"
    # ... and its answer really reached the parent, through wait_agents.
    assert "child answer" in str(outcome.steps)
    # Still ONE loop for the whole run tree, children included (Fix 1(c)).
    assert len(seen_loops) == 4  # 3 parent turns + 1 child turn
    assert all(loop is seen_loops[0] for loop in seen_loops)
    assert seen_loops[0].is_closed()


def test_provider_stream_signal_survives_primary_tool_and_final_turns(tmp_path):
    signals = ConsoleProviderStreamSignals()
    gateway = _SignalChunkGateway(
        [
            [_fence("calculator", {"expression": "6*7"})],
            ["It is ", "42."],
        ]
    )
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        provider_stream_signals=signals,
    )

    assert outcome.status == "done"
    assert gateway.signals_seen == [signals, signals]
    assert all(item is signals for item in gateway.signals_seen)


def test_provider_stream_signal_survives_subagent_turns(tmp_path):
    signals = ConsoleProviderStreamSignals()
    gateway = _SignalChunkGateway(
        [
            [_fence("spawn_subagent", {"task": "compute 1+1"})],
            ["2"],
            ["Done: 2."],
        ]
    )
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        provider_stream_signals=signals,
    )

    assert outcome.status == "done"
    assert gateway.signals_seen == [signals, signals, signals]
    assert all(item is signals for item in gateway.signals_seen)


def test_provider_stream_signal_is_never_reset_by_bridge(tmp_path):
    signals = ConsoleProviderStreamSignals()
    signals.mark_synthetic_fallback()
    gateway = _SignalChunkGateway([["Already marked."]])
    bridge, _db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)

    outcome = _run(
        bridge,
        store,
        session,
        aid,
        provider_stream_signals=signals,
    )

    assert outcome.status == "done"
    assert gateway.signals_seen == [signals]
    assert gateway.signal_states_seen == [True]
    assert signals.synthetic_fallback_emitted is True


def test_provider_stream_signal_omission_preserves_legacy_gateway_signature(tmp_path):
    bridge, _db, store, session, aid = _bridge(tmp_path, [["Unchanged."]])

    outcome = _run(bridge, store, session, aid)

    assert outcome.status == "done"
    assert bridge._gateway.calls == 1
    assert store.get_message(aid).content == "Unchanged."


def test_spawn_renders_marker_and_persists_linked_subagent(tmp_path):
    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        ["2"],  # sub-agent turn
        ["Done: ", "2."],  # primary final
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    assert db.count_subagent_runs("conv-1") == 1
    spawn_markers = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL and "sub-agent" in m.content.lower()
    ]
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
        ["ok"],  # sub-agent turn
        ["Done."],  # primary final
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    outcome = _run(bridge, store, session, aid)
    assert outcome.status == "done"
    spawn_markers = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL and "sub-agent" in m.content.lower()
    ]
    assert spawn_markers
    assert "[docs]" in spawn_markers[0].content
    assert "\\[docs]" not in spawn_markers[0].content


def test_supersede_marks_previous_primary_and_tree(tmp_path):
    bridge, db, store, session, aid = _bridge(tmp_path, [["one."], ["two."]])
    _run(bridge, store, session, aid)  # first run
    first = db.list_runs("conv-1")[0]
    assert first["status"] == "done"
    # Second run supersedes the previous primary.
    aid2 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    ).id
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
    outcome = _run(bridge, store, session, aid, should_cancel=lambda: next(flags, True))
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
    db.append_steps(
        primary_id,
        [
            {
                "index": 0,
                "kind": "model",
                "summary": "The capital of France is Paris.",
                "tool_name": "",
                "args": None,
                "result": "",
                "created_at": "",
            },
        ],
    )
    db.set_status(primary_id, "done", result="The capital of France is Paris.")
    sub_id = db.create_run(
        conversation_id="conv-1",
        agent_kind="subagent",
        task="research pricing",
        parent_run_id=primary_id,
    )
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
        conversation_id="conv-1",
        agent_kind="subagent",
        task="orphaned",
        parent_run_id=other_primary_id,
    )

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    snap = bridge.historical_snapshot("conv-1")
    assert snap.status == "done"
    assert snap.subagents == ()


def test_historical_snapshot_caches_per_conversation_not_hit_every_call(
    tmp_path, monkeypatch
):
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
    assert len(calls) == 1  # the 0.2s rail poll must not re-hit the DB

    # A different conversation is a separate cache entry.
    bridge.historical_snapshot("conv-2")
    assert len(calls) == 2


# -- Plan-B final-review Medium-1: inline transcript TOOL markers re-derive
# from AgentRunsDB on resume, the same way the rail already does. --


def test_format_agent_step_marker_matches_each_live_marker_shape():
    assert (
        format_agent_step_marker(STEP_SPAWN, summary="research pricing")
        == "⤷ spawned sub-agent: research pricing"
    )
    assert (
        format_agent_step_marker(STEP_TOOL_RESULT, tool_name="calculator", result="42")
        == "⚙ calculator → 42"
    )
    assert format_agent_step_marker(STEP_ERROR, summary="boom") == "⚠ boom"
    # Quiet tool-catalog steps and plain model steps never produce a marker.
    assert (
        format_agent_step_marker(STEP_TOOL_RESULT, tool_name="find_tools", result="[]")
        is None
    )
    assert (
        format_agent_step_marker(STEP_TOOL_RESULT, tool_name="load_tools", result="[]")
        is None
    )
    assert format_agent_step_marker(STEP_MODEL, summary="The answer is 42.") is None


def test_long_tool_result_marker_is_collapsed_with_truncation_affordance():
    """TASK-350: a spawn_subagent whose result IS the full answer must not be
    dumped verbatim into the TOOL marker — it duplicated the assistant bubble
    word-for-word. Collapse it to a preview and mark the truncation plus how much
    is hidden, so the marker reads as provenance, not a second copy."""
    answer = "### Understanding SQLite WAL. " + "word " * 300  # long answer
    marker = format_agent_step_marker(
        STEP_TOOL_RESULT, tool_name="spawn_subagent", result=answer
    )
    assert marker.startswith("\u2699 spawn_subagent \u2192 ### Understanding SQLite")
    assert "\u2026" in marker  # ellipsis marks the cut
    assert "chars)" in marker  # explicit "how much more" affordance
    assert len(marker) < len(answer) // 2  # collapsed, not a full duplicate


def test_short_tool_result_marker_is_unchanged():
    # Short results stay verbatim — no ellipsis, no affordance.
    marker = format_agent_step_marker(
        STEP_TOOL_RESULT, tool_name="calculator", result="42"
    )
    assert marker == "\u2699 calculator \u2192 42"
    assert "\u2026" not in marker


def test_step_truncation_cuts_on_word_boundary_with_ellipsis():
    from tldw_chatbook.Chat.console_agent_bridge import _truncate_step_text

    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    out = _truncate_step_text(text, limit=24)
    assert "\u2026" in out and "(+" in out and "chars)" in out
    preview = out.split("\u2026", 1)[0].strip()
    # every token shown is a whole word from the source — never a mid-word clip
    assert preview
    assert all(tok in text.split() for tok in preview.split())


def test_step_truncation_cuts_on_newline_and_tab_boundaries():
    """Qodo #3: markdown/structured results split on newlines/tabs, not just
    spaces — the boundary search must treat any whitespace as a token break so a
    fenced/heading result is not clipped mid-token."""
    from tldw_chatbook.Chat.console_agent_bridge import _truncate_step_text

    text = "### Heading\n\nsome body text that keeps going well past the limit here"
    out = _truncate_step_text(text, limit=15)
    assert out.split("\u2026", 1)[0] == "### Heading"  # cut at the newline, not "so"
def test_format_todo_marker_renders_statuses_and_active_form():
    text = format_todo_marker(
        [
            {"content": "write tests", "status": "completed"},
            {"content": "implement", "status": "in_progress", "activeForm": "implementing"},
            {"content": "commit", "status": "pending"},
        ]
    )
    assert text == (
        "☰ Todos (1 in progress):\n"
        "  [x] write tests\n"
        "  [~] implementing\n"
        "  [ ] commit"
    )


def test_format_todo_marker_empty_list_reads_as_cleared():
    assert format_todo_marker([]) == "☰ Todos cleared"


def test_format_todo_marker_truncates_long_item_text():
    # Same 200-char convention as step-marker summaries (_summarize).
    long_content = "y" * 300
    text = format_todo_marker([{"content": long_content, "status": "pending"}])
    assert text == f"☰ Todos (0 in progress):\n  [ ] {'y' * 200}"


def test_format_todo_marker_flattens_newlines_in_item_text():
    # Markers stay one line per item; embedded newlines become spaces.
    text = format_todo_marker(
        [{"content": "first\nsecond\r\nthird", "status": "pending"}]
    )
    assert text == "☰ Todos (0 in progress):\n  [ ] first second third"


def test_append_todo_marker_appends_tool_message_to_store(tmp_path):
    bridge, _db, store, session, _aid = _bridge(tmp_path, [])
    bridge.append_todo_marker(
        session.id, [{"content": "ship it", "status": "in_progress"}]
    )
    tool_messages = [
        m for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert [m.content for m in tool_messages] == [
        "☰ Todos (1 in progress):\n  [~] ship it"
    ]


def test_resume_marker_messages_reproduces_live_markers_after_simulated_restart(
    tmp_path,
):
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],  # turn 1: leading fence
        ["It is ", "42."],  # turn 2: final answer
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    _run(bridge, store, session, aid)
    live_tool_contents = [
        m.content
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert live_tool_contents  # sanity: the live run actually left a marker

    # A fresh bridge instance -- simulates an app restart -- must re-derive
    # byte-identical marker text purely from AgentRunsDB.
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    blocks = fresh_bridge.resume_marker_messages("conv-1")
    resumed_tool_contents = [m.content for _anchor, block in blocks for m in block]
    assert resumed_tool_contents == live_tool_contents


def test_resume_marker_messages_surfaces_assistant_message_id_anchor(tmp_path):
    """Task 3: each block is paired with the run's ``assistant_message_id``
    (``None`` for a legacy/pre-Phase-C run that never recorded one) so the
    placement layer can anchor by id instead of guessing ordinally."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    anchored = db.create_run(
        conversation_id="conv-1",
        agent_kind="primary",
        assistant_message_id="asst-anchor",
    )
    db.append_steps(
        anchored,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "boom",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(anchored, "done", result="ok")
    legacy = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        legacy,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "legacy",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(legacy, "done", result="ok")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")

    assert len(blocks) == 2
    assert blocks[0][0] == "asst-anchor"
    assert "boom" in blocks[0][1][0].content
    assert blocks[1][0] is None
    assert "legacy" in blocks[1][1][0].content


def test_resume_marker_messages_orders_blocks_chronologically_oldest_first(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    first = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        first,
        [
            {
                "index": 0,
                "kind": STEP_TOOL_RESULT,
                "tool_name": "calculator",
                "result": "4",
                "summary": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(first, "done", result="4")
    second = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        second,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "timed out",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(second, "done", result="ok")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")
    assert len(blocks) == 2
    assert "calculator" in blocks[0][1][0].content
    assert "timed out" in blocks[1][1][0].content


def test_resume_marker_messages_skips_superseded_runs(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    superseded = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        superseded,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "old attempt",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(superseded, "superseded")
    kept = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(
        kept,
        [
            {
                "index": 0,
                "kind": STEP_ERROR,
                "summary": "final attempt",
                "tool_name": "",
                "result": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    db.set_status(kept, "done", result="ok")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    blocks = bridge.resume_marker_messages("conv-1")
    assert len(blocks) == 1
    assert "final attempt" in blocks[0][1][0].content


def _tool_marker(text: str) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL, content=text, status="complete"
    )


def test_inject_resume_agent_markers_places_block_after_matching_assistant_message():
    """Task 3: placement follows the block's anchor id (matched against
    each message's ``persisted_message_id``), not block order/ordinal
    position. The blocks below are listed in the OPPOSITE order of their
    anchors -- the block for the SECOND assistant reply ("asst-2") comes
    first in the list -- so an ordinal placement would put it after "42."
    (wrong); id-anchoring must still put it after "ok." (right)."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="42.",
            status="complete",
            persisted_message_id="asst-1",
        ),
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="again"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="ok.",
            status="complete",
            persisted_message_id="asst-2",
        ),
    ]
    blocks = [
        ("asst-2", [_tool_marker("⚠ retry")]),
        ("asst-1", [_tool_marker("⚙ calculator → 42")]),
    ]

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


def test_inject_resume_agent_markers_drops_block_whose_anchor_matches_no_active_path_message():
    """A block anchored to an assistant_message_id that isn't in the
    active-path ``messages`` (the run's reply lives on another branch) must
    be hidden entirely -- never appended anywhere -- rather than leak an
    off-branch tool trace onto the visible transcript."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="42.",
            status="complete",
            persisted_message_id="asst-1",
        ),
    ]
    blocks = [("asst-OFF-PATH", [_tool_marker("⚙ calculator → 999")])]

    result = inject_resume_agent_markers(messages, blocks)

    assert [m.content for m in result] == ["hi", "42."]
    assert not any(m.role is ConsoleMessageRole.TOOL for m in result)


def test_inject_resume_agent_markers_null_anchor_blocks_place_ordinally_when_no_id_claims():
    """Legacy (pre-Phase-C) runs never recorded an assistant_message_id --
    their blocks carry a ``None`` anchor and keep the old ordinal
    placement: Nth null block <-> Nth assistant reply."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="42.", status="complete"
        ),
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="again"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="ok.", status="complete"
        ),
    ]
    blocks = [
        (None, [_tool_marker("⚙ calculator → 42")]),
        (None, [_tool_marker("⚠ retry")]),
    ]

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


def test_inject_resume_agent_markers_mixed_id_and_null_anchors_null_skips_claimed_assistant():
    """Mixed case: one block is id-anchored to the FIRST assistant reply: the
    remaining null (legacy) block's ordinal fallback must skip that already
    id-claimed reply and land on the next unclaimed one, not double up on
    "42."."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="42.",
            status="complete",
            persisted_message_id="asst-1",
        ),
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="again"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="ok.", status="complete"
        ),
    ]
    blocks = [
        ("asst-1", [_tool_marker("⚙ calculator → 42")]),
        (None, [_tool_marker("⚠ legacy retry")]),
    ]

    result = inject_resume_agent_markers(messages, blocks)

    roles = [(m.role, m.content) for m in result]
    assert roles == [
        (ConsoleMessageRole.USER, "hi"),
        (ConsoleMessageRole.ASSISTANT, "42."),
        (ConsoleMessageRole.TOOL, "⚙ calculator → 42"),
        (ConsoleMessageRole.USER, "again"),
        (ConsoleMessageRole.ASSISTANT, "ok."),
        (ConsoleMessageRole.TOOL, "⚠ legacy retry"),
    ]


def test_inject_resume_agent_markers_appends_leftover_block_when_more_runs_than_replies():
    """Preserves the old leftover-append behavior, now for NULL-anchored
    blocks: a legacy run with no corresponding assistant reply left in the
    active path is appended at the end rather than dropped."""
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="42.", status="complete"
        ),
    ]
    blocks = [
        (None, [_tool_marker("⚙ calculator → 42")]),
        (None, [_tool_marker("⚠ orphan run")]),
    ]

    result = inject_resume_agent_markers(messages, blocks)

    assert [m.content for m in result] == [
        "hi",
        "42.",
        "⚙ calculator → 42",
        "⚠ orphan run",
    ]


def test_inject_resume_agent_markers_skips_empty_blocks():
    messages = [
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="ok.", status="complete"
        ),
    ]
    result = inject_resume_agent_markers(messages, [(None, []), ("some-id", [])])
    assert [m.content for m in result] == ["ok."]


def test_inject_resume_agent_markers_is_idempotent_no_duplicates_on_second_call():
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="42.",
            status="complete",
            persisted_message_id="asst-1",
        ),
    ]
    blocks = [("asst-1", [_tool_marker("⚙ calculator → 42")])]

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
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="42.",
            status="complete",
            persisted_message_id="asst-1",
        ),
        _tool_marker("⚙ calculator → 42"),
    ]
    blocks = [("asst-1", [_tool_marker("⚙ calculator → 42")])]

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
        for m in live_messages
        if m.role is not ConsoleMessageRole.TOOL
    ]
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    resumed = inject_resume_agent_markers(
        chachanotes_only, fresh_bridge.resume_marker_messages("conv-1")
    )

    resumed_tool_contents = [
        m.content for m in resumed if m.role is ConsoleMessageRole.TOOL
    ]
    assert resumed_tool_contents == live_tool_contents
    assistant_index = next(
        i for i, m in enumerate(resumed) if m.role is ConsoleMessageRole.ASSISTANT
    )
    assert resumed[assistant_index + 1].role is ConsoleMessageRole.TOOL

    resumed_again = inject_resume_agent_markers(
        resumed, fresh_bridge.resume_marker_messages("conv-1")
    )
    assert len(resumed_again) == len(resumed)


# -- Task 12: per-run spawn-wired skill executor + run allow-list composition --


class _FakeSkillsService:
    """Minimal async skills service: one trusted, model-invocable skill."""

    def __init__(self, *, skill_name="code-review", allowed_tools=None, blocked=False):
        self.skill_name = skill_name
        self.allowed_tools = allowed_tools
        self.blocked = blocked
        self.execute_calls = []
        self.get_context_calls = 0

    async def get_context(self, *, mode="local"):
        self.get_context_calls += 1
        return {
            "available_skills": [
                {
                    "name": self.skill_name,
                    "description": "Review a diff",
                    "argument_hint": "[diff]",
                    "trust_blocked": False,
                    "disable_model_invocation": False,
                },
            ],
            "blocked_skills": [],
        }

    async def execute_skill(self, name, *, mode="local", args=None):
        self.execute_calls.append(args)
        if self.blocked:
            raise SkillTrustBlockedError(
                skill_name=name,
                reason_code="quarantined_modified",
                trust_status="quarantined_modified",
            )
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
        ["Looks fine to me."],  # sub-agent turn
        ["All done."],  # primary final
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service,
    )

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-skill")

    assert outcome.status == "done"
    assert skills_service.execute_calls == ["the diff"]
    assert db.count_subagent_runs("conv-skill") == 1
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
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
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService(blocked=True)
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service,
    )

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-blocked")

    assert outcome.status == "done"
    assert db.count_subagent_runs("conv-blocked") == 0


# -- task-4 (skills-fork-reachability): _BridgeSkillRunner grants its own
# name skill_file authorization pre-spawn and appends a "Bundled files"
# pointer block to the rendered task text whenever execute_skill reports
# reference_files -- unit-level, directly against _BridgeSkillRunner, so
# these don't need a full run_reply/model round trip. --


class _FakeSkillsServiceWithRefs(_FakeSkillsService):
    """Same fake as above, but execute_skill also reports a bundle manifest."""

    async def execute_skill(self, name, *, mode="local", args=None):
        self.execute_calls.append(args)
        return {
            "skill_name": name,
            "rendered_prompt": f"Review this: {args}",
            "allowed_tools": self.allowed_tools,
            "execution_mode": "inline",
            "reference_files": [
                {"path": "references/api.md", "size": 120, "is_text": True},
                {"path": "assets/logo.png", "size": 2048, "is_text": False},
            ],
        }


def test_bridge_skill_runner_grants_own_name_and_appends_bundle_block_before_spawn():
    skills_service = _FakeSkillsServiceWithRefs()
    bindings = SkillFileBindings(authorized=set())
    runner = _BridgeSkillRunner(
        skills_service=skills_service,
        skill_names=frozenset({"code-review"}),
        builtin_names=(),
        skill_file_bindings=bindings,
    )
    spawn_calls = []

    def spawn(rendered, *, allowed_tools):
        # The name must already be authorized by the time spawn actually
        # runs -- so the spawned child's very first turn can read its own
        # bundle -- not merely by the time .run() returns.
        assert "code-review" in bindings.authorized
        spawn_calls.append(rendered)
        return ToolResult(ok=True, content="sub-agent result")

    result = runner.run("code-review", "the diff", spawn)

    assert result.ok
    assert spawn_calls == [
        "Review this: the diff\n\nBundled files (readable via skill_file): "
        "references/api.md (120 bytes), assets/logo.png (2048 bytes, binary)"
    ]
    assert "code-review" in bindings.authorized


def test_bridge_skill_runner_no_reference_files_body_unchanged_still_authorizes():
    skills_service = _FakeSkillsService()  # no reference_files key at all
    bindings = SkillFileBindings(authorized=set())
    runner = _BridgeSkillRunner(
        skills_service=skills_service,
        skill_names=frozenset({"code-review"}),
        builtin_names=(),
        skill_file_bindings=bindings,
    )
    spawn_calls = []

    def spawn(rendered, *, allowed_tools):
        spawn_calls.append(rendered)
        return ToolResult(ok=True, content="sub-agent result")

    runner.run("code-review", "the diff", spawn)

    assert spawn_calls == ["Review this: the diff"]
    assert "code-review" in bindings.authorized


def test_bridge_skill_runner_bindings_none_is_byte_identical_legacy_behavior():
    # reference_files IS present in the execute_skill result, but with no
    # skill_file_bindings wired at all (legacy/non-bridge construction) the
    # block must never be appended and nothing must crash.
    skills_service = _FakeSkillsServiceWithRefs()
    runner = _BridgeSkillRunner(
        skills_service=skills_service,
        skill_names=frozenset({"code-review"}),
        builtin_names=(),
    )
    spawn_calls = []

    def spawn(rendered, *, allowed_tools):
        spawn_calls.append(rendered)
        return ToolResult(ok=True, content="sub-agent result")

    runner.run("code-review", "the diff", spawn)

    assert spawn_calls == ["Review this: the diff"]


def test_run_reply_wires_one_skill_file_bindings_to_both_service_and_runner(
    tmp_path,
):
    """run_reply must construct exactly ONE SkillFileBindings per run and
    hand the SAME object to both AgentService and the _BridgeSkillRunner --
    never two independently-seeded copies (which would let the runner's
    pre-spawn grant silently fail to reach the loop's authorization check)."""
    captured = {}
    real_runner_init = _BridgeSkillRunner.__init__
    real_service_init = AgentService.__init__

    def spy_runner_init(self, **kwargs):
        captured["runner_bindings"] = kwargs.get("skill_file_bindings")
        real_runner_init(self, **kwargs)

    def spy_service_init(self, *args, **kwargs):
        captured["service_bindings"] = kwargs.get("skill_file_bindings")
        real_service_init(self, *args, **kwargs)

    scripts = [
        [_fence("code-review", {"args": "the diff"})],
        ["Looks fine to me."],
        ["All done."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service,
    )

    with patch.object(_BridgeSkillRunner, "__init__", spy_runner_init), patch.object(
        AgentService, "__init__", spy_service_init
    ):
        outcome = _run(
            bridge, store, session, assistant.id, conversation_id="conv-bindings"
        )

    assert outcome.status == "done"
    assert captured["runner_bindings"] is not None
    assert captured["runner_bindings"] is captured["service_bindings"]


def test_run_reply_seeds_turn_bindings_into_shared_object(tmp_path):
    """Task 5: the turn's already-resolved `$skill` binding names (splice
    output the CONTROLLER computed for the triggering turn) must be seeded
    into THIS run's SkillFileBindings.authorized -- the SAME shared object
    handed to both AgentService and the _BridgeSkillRunner (Task 4) -- so
    the primary agent's very first turn can already read that skill's own
    bundle via skill_file. Reuses the sibling wiring test's spy idiom to
    prove the seed lands on the one shared object, not a second copy."""
    captured = {}
    real_runner_init = _BridgeSkillRunner.__init__
    real_service_init = AgentService.__init__

    def spy_runner_init(self, **kwargs):
        captured["runner_bindings"] = kwargs.get("skill_file_bindings")
        real_runner_init(self, **kwargs)

    def spy_service_init(self, *args, **kwargs):
        captured["service_bindings"] = kwargs.get("skill_file_bindings")
        real_service_init(self, *args, **kwargs)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway([["Tokyo."]]),
        skills_service=skills_service,
    )

    with patch.object(_BridgeSkillRunner, "__init__", spy_runner_init), patch.object(
        AgentService, "__init__", spy_service_init
    ):
        outcome = _run(
            bridge,
            store,
            session,
            assistant.id,
            conversation_id="conv-turn-bindings",
            turn_skill_bindings=("code-review",),
        )

    assert outcome.status == "done"
    assert captured["runner_bindings"] is not None
    assert "code-review" in captured["runner_bindings"].authorized
    # Seeded onto the ONE shared object -- never two independently-seeded
    # copies (Task 4's invariant, re-verified here under a non-empty seed).
    assert captured["runner_bindings"] is captured["service_bindings"]


def test_run_reply_appends_bundle_block_copy_safely(tmp_path):
    """Task 5: the turn's pre-rendered "Bundled files" block is appended to
    a NEW list + NEW dict for the last role=="user" entry -- the caller's
    own `agent_messages` list and its message dict are never mutated. A
    future refactor that switches to in-place `message["content"] += ...`
    must fail assertions (b) below."""
    real_run_turn = AgentService.run_turn

    def _spy(captured):
        def spy_run_turn(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return real_run_turn(self, **kwargs)

        return spy_run_turn

    block = "Bundled files (readable via skill_file): notes.md (1 bytes)"

    # -- (a) + (b): a non-empty block is appended copy-safely -------------
    bridge, _db, store, session, aid = _bridge(tmp_path, [["Tokyo."]])
    original_user_message = {"role": "user", "content": "hi"}
    agent_messages = [original_user_message]
    captured = {}

    with patch.object(AgentService, "run_turn", _spy(captured)):
        outcome = _run(
            bridge,
            store,
            session,
            aid,
            conversation_id="conv-bundle-block",
            agent_messages=agent_messages,
            turn_bundle_block=block,
        )

    assert outcome.status == "done"
    run_messages = captured["messages"]
    assert run_messages is not None
    # (a) the run actually received the block appended after "\n\n" on the
    # last user entry.
    assert run_messages[-1]["content"] == f"hi\n\n{block}"
    assert run_messages[-1]["role"] == "user"
    # (b) non-mutation contract: the caller's own list length and the dict
    # identity at that index are unchanged, and the original dict's content
    # carries no trace of the block. This is exactly what would break if
    # the append were switched to in-place `message["content"] += ...`.
    assert len(agent_messages) == 1
    assert agent_messages[0] is original_user_message
    assert original_user_message["content"] == "hi"
    assert "Bundled files" not in original_user_message["content"]
    # A new list AND a new dict were built for the run -- not the caller's.
    assert run_messages is not agent_messages
    assert run_messages[-1] is not original_user_message

    # -- (c) sibling case: an empty block appends nothing, no-op path -----
    bridge2, _db2, store2, session2, aid2 = _bridge(tmp_path, [["Tokyo."]])
    agent_messages2 = [{"role": "user", "content": "hi"}]
    captured2 = {}

    with patch.object(AgentService, "run_turn", _spy(captured2)):
        outcome2 = _run(
            bridge2,
            store2,
            session2,
            aid2,
            conversation_id="conv-bundle-block-empty",
            agent_messages=agent_messages2,
            turn_bundle_block="",
        )

    assert outcome2.status == "done"
    assert captured2["messages"][-1]["content"] == "hi"
    # No block to append: the original list is used unchanged (not merely
    # equal -- the very same object), matching the documented no-op path.
    assert captured2["messages"] is agent_messages2


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
            {
                "name": "code-review",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
            {
                "name": "needs-review",
                "trust_blocked": True,
                "disable_model_invocation": False,
            },
            {
                "name": "user-only",
                "trust_blocked": False,
                "disable_model_invocation": True,
            },
        ],
    }
    allowed = _compose_run_allowed_tools(
        context, ("calculator", "get_current_datetime")
    )
    assert allowed == (
        "calculator",
        "get_current_datetime",
        "code-review",
        SPAWN_TOOL_NAME,
    )


def test_compose_run_allowed_tools_empty_context_is_builtins_plus_spawn():
    allowed = _compose_run_allowed_tools({}, ("calculator",))
    assert allowed == ("calculator", SPAWN_TOOL_NAME)


def test_compose_run_allowed_tools_builtin_shadows_same_named_skill():
    """Task 11 review note 2: a skill named the same as a builtin must never
    become a distinct, skill-routable tool -- the builtin always wins. The
    allow-list carries the name exactly once (from builtins), never twice."""
    context = {
        "available_skills": [
            {
                "name": "calculator",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    allowed = _compose_run_allowed_tools(
        context, ("calculator", "get_current_datetime")
    )
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
            {
                "name": "find_tools",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    allowed = _compose_run_allowed_tools(
        context, ("calculator", "get_current_datetime")
    )
    assert allowed == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)


def test_compose_run_registry_excludes_skill_named_like_a_runtime_tool():
    """Same collision, verified against the actual registry/allow-list this
    run would use: the skill must not appear as a distinct catalog entry
    (under either FIND_TOOLS_NAME or LOAD_TOOLS_NAME or SPAWN_TOOL_NAME)."""
    context = {
        "available_skills": [
            {
                "name": LOAD_TOOLS_NAME,
                "description": "d",
                "argument_hint": "",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    registry, allowed_tools, builtin_names, _local_names = _compose_run_registry_and_allowed(context)
    assert LOAD_TOOLS_NAME not in allowed_tools[len(builtin_names) :]
    catalog_entries = [(entry.name, entry.source) for entry in registry.list_catalog()]
    assert (LOAD_TOOLS_NAME, "skill") not in catalog_entries


# -- P5-T6: MCPToolProvider registration + collision precedence --


def test_compose_run_registry_and_allowed_includes_mcp_entries_when_eligible():
    mcp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    registry, allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp_provider
    )
    assert "mcp__srv_a__search" in allowed_tools
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert ("mcp__srv_a__search", "mcp") in catalog_entries
    result = registry.invoke_by_name("mcp__srv_a__search", {"query": "weather"})
    assert result.ok is True
    assert mcp_provider.invoke_calls == [("mcp__srv_a__search", {"query": "weather"})]


def test_compose_run_registry_and_allowed_absent_mcp_provider_is_unchanged():
    """`mcp_provider=None` (the default) must not add anything -- the
    pre-P5-T6 no-MCP behavior stays byte-identical."""
    registry, allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed({})
    assert allowed_tools == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    assert len(registry.list_catalog()) == 2


class _FakeBuiltinGateForRegistry:
    """Minimal `BuiltinToolGate` double -- only `.check()` is exercised by
    `BuiltinToolProvider.invoke()`."""

    def __init__(self, refuse: bool) -> None:
        self.refuse = refuse
        self.checked: list[str] = []

    def check(self, tool, run_id):
        self.checked.append(tool.name)
        return f"disabled for test: {tool.name}" if self.refuse else None


def test_compose_run_registry_and_allowed_threads_builtin_gate_into_the_provider():
    """task-545/T6: `builtin_gate=` must reach the freshly-built
    `BuiltinToolProvider` -- NOT a second, independently-built gate --
    else a decision the caller's review hook stamped on that gate would
    never be visible to `invoke()`."""
    gate = _FakeBuiltinGateForRegistry(refuse=True)
    registry, _allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        {}, builtin_gate=gate
    )
    result = registry.invoke_by_name("calculator", {"expression": "6*7"})
    assert result.ok is False
    assert result.error == "disabled for test: calculator"
    assert gate.checked == ["calculator"]


def test_compose_run_registry_and_allowed_no_builtin_gate_is_unchanged():
    """`builtin_gate=None` (the default) must not alter the pre-task-545
    no-skills/no-MCP behavior -- the provider builds its own lazy gate."""
    registry, allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed({})
    assert allowed_tools == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    result = registry.invoke_by_name("calculator", {"expression": "6*7"})
    assert result.ok is True


class _WorkspaceProbeTool:
    """A minimal Tool double that reports the run workspace bound around it."""

    name = "probe_workspace"
    description = "records the bound run workspace"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        from tldw_chatbook.Tools import workspace_file_roots as wfr

        return {"workspace": wfr.current_run_workspace_id()}


def test_compose_run_registry_and_allowed_threads_workspace_id_into_the_provider():
    """task-6 (settings-workspaces-folder-roots spec Sec3): `workspace_id=`
    must reach the freshly-built `BuiltinToolProvider` so its `invoke()`
    binds the run's workspace around every tool call."""
    registry, _allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        {},
        builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
        workspace_id="ws-compose",
    )
    # registry._providers[0] is the BuiltinToolProvider this call just built
    # (see _compose_run_registry_and_allowed's own body) -- poke a probe
    # tool onto it exactly as the constructor-level test does, since the
    # real calculator/datetime tools have no way to report the ContextVar.
    registry._providers[0]._tools["probe_workspace"] = _WorkspaceProbeTool()
    result = registry.invoke_by_name("probe_workspace", {})
    assert result.ok, result.error
    assert '"workspace": "ws-compose"' in result.content


def test_compose_run_registry_and_allowed_no_workspace_id_is_unchanged():
    """`workspace_id=None` (the default) must not alter the pre-task-6
    behavior -- the provider leaves the run workspace unbound."""
    registry, _allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        {}, builtin_gate=_FakeBuiltinGateForRegistry(refuse=False)
    )
    registry._providers[0]._tools["probe_workspace"] = _WorkspaceProbeTool()
    result = registry.invoke_by_name("probe_workspace", {})
    assert result.ok, result.error
    assert '"workspace": null' in result.content


class _StubWriteFileTool:
    """Minimal Tool double named ``write_file`` -- ``_WorkspaceProbeTool``
    above hardcodes its own name (``probe_workspace``), which would make
    `registry.invoke_by_name("write_file", ...)` fail to resolve since the
    catalog entry it produces is keyed by ``.name``, not by the provider's
    internal dict key it was poked under."""

    name = "write_file"
    description = "stub"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return {"ok": True}


def test_compose_run_registry_and_allowed_threads_ephemeral_into_the_provider():
    """F4 (final-review): `ephemeral=` must reach the freshly-built
    `BuiltinToolProvider` so its `invoke()` refuses the write-shaped
    built-ins for a temporary session. Mirrors ``..._threads_workspace_id_
    into_the_provider`` exactly."""
    registry, _allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        {},
        builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
        ephemeral=True,
    )
    registry._providers[0]._tools["write_file"] = _StubWriteFileTool()
    result = registry.invoke_by_name("write_file", {})
    assert result.ok is False
    assert "temporary chat" in result.error


def test_compose_run_registry_and_allowed_no_ephemeral_is_unchanged():
    """`ephemeral=False` (the default) must not alter pre-F4 behavior --
    the provider dispatches the tool normally."""
    registry, _allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        {}, builtin_gate=_FakeBuiltinGateForRegistry(refuse=False)
    )
    registry._providers[0]._tools["write_file"] = _StubWriteFileTool()
    result = registry.invoke_by_name("write_file", {})
    assert result.ok, result.error


# -- the temporary-session tool gate at the shared choke point --
#
# Every refusal assertion below carries its allowed-normally control in the
# SAME test. That is the dangerous direction here: a guard that fired
# unconditionally would break MCP and skill tools for every user of the app,
# and each "assert refused" half would still be perfectly green.


def _stub_tool(tool_name: str):
    """A minimal always-succeeding ``Tool`` double registered under a name.

    ``_StubWriteFileTool`` above hardcodes one name; the gate tests need a
    handful (the three write-shaped built-ins and the four gateable
    read-only ones, none of which are registered by default because their
    ``[tools]`` gates ship off).
    """

    class _Stub:
        name = tool_name
        description = "stub"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            return {"ok": True}

    return _Stub()


class _FakeSourcedProvider:
    """``ToolProvider`` double that reports an arbitrary catalog ``source``.

    Lets the source-based policy be exercised for skills (whose real
    provider deliberately raises from ``invoke``) and for a source no one
    has whitelisted, without inventing a whole provider per case.
    """

    def __init__(self, source, entries):
        self._source = source
        self._entries = list(entries)
        self.invoke_calls: list[tuple[str, dict]] = []

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=name, name=name, one_line_description="d", source=self._source
            )
            for name in self._entries
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id,
            description="",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        self.invoke_calls.append((tool_id, dict(args or {})))
        return ToolResult(ok=True, content=f"result:{tool_id}")


_SKILL_CONTEXT = {
    "available_skills": [
        {
            "name": "tidy_repo",
            "description": "Tidy the repository",
            "argument_hint": "",
            "trust_blocked": False,
            "disable_model_invocation": False,
        },
    ],
}


def test_invoke_by_name_refuses_an_mcp_tool_only_in_a_temporary_run():
    """Requirements 1 and 5, together: the CHOKE POINT is load-bearing on
    its own, not merely a backstop to the allow-list.

    Both registries here are built by hand and the MCP provider registered
    straight onto them -- no allow-list, no ``_compose_run_registry_and_
    allowed`` filtering anywhere in the picture. So the refusal can only be
    coming from ``invoke_by_name`` itself, and the control proves the same
    call succeeds when the session is not temporary.
    """
    saved_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    saved = ToolCatalogRegistry()
    saved.register_provider(saved_provider)
    control = saved.invoke_by_name("mcp__srv_a__search", {"query": "weather"})
    assert control.ok is True, control.error
    assert saved_provider.invoke_calls == [("mcp__srv_a__search", {"query": "weather"})]

    temp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    temporary = ToolCatalogRegistry(ephemeral=True)
    temporary.register_provider(temp_provider)
    refused = temporary.invoke_by_name("mcp__srv_a__search", {"query": "weather"})
    assert refused.ok is False
    assert "temporary chat" in refused.error
    assert "mcp__srv_a__search" in refused.error
    # The provider was never reached -- refused before dispatch, not after.
    assert temp_provider.invoke_calls == []


def test_invoke_by_name_refuses_a_skill_tool_only_in_a_temporary_run():
    """Requirement 2, at the choke point, with the real ``SkillToolProvider``.

    That class's ``invoke`` raises by design (skill calls route through the
    run-scoped spawn executor), which makes the control unusually sharp: in
    a saved chat the call must still REACH the provider, and the only
    observable proof of that is the RuntimeError escaping. In a temporary
    chat the gate must intercept first and hand back a ToolResult.
    """
    entries = [
        {"name": "tidy_repo", "description": "Tidy the repository", "argument_hint": ""}
    ]

    saved = ToolCatalogRegistry()
    saved.register_provider(SkillToolProvider(entries))
    with pytest.raises(RuntimeError):
        saved.invoke_by_name("tidy_repo", {"args": ""})

    temporary = ToolCatalogRegistry(ephemeral=True)
    temporary.register_provider(SkillToolProvider(entries))
    refused = temporary.invoke_by_name("tidy_repo", {"args": ""})
    assert refused.ok is False
    assert "temporary chat" in refused.error
    assert "tidy_repo" in refused.error


def test_invoke_by_name_refuses_a_skill_sourced_tool_that_would_otherwise_succeed():
    """Requirement 2 again, with a skill-sourced provider that DOES return
    a successful result -- so the control half is a real, observed success
    rather than the real provider's by-design raise."""
    saved_provider = _FakeSourcedProvider("skill", ["tidy_repo"])
    saved = ToolCatalogRegistry()
    saved.register_provider(saved_provider)
    control = saved.invoke_by_name("tidy_repo", {"args": "x"})
    assert control.ok is True, control.error
    assert saved_provider.invoke_calls == [("tidy_repo", {"args": "x"})]

    temp_provider = _FakeSourcedProvider("skill", ["tidy_repo"])
    temporary = ToolCatalogRegistry(ephemeral=True)
    temporary.register_provider(temp_provider)
    refused = temporary.invoke_by_name("tidy_repo", {"args": "x"})
    assert refused.ok is False
    assert "temporary chat" in refused.error
    assert temp_provider.invoke_calls == []


def test_invoke_by_name_refuses_a_provider_source_nobody_whitelisted():
    """Unknown capability fails toward not-writing: a provider added after
    this was written is gated on the day it is added, and the control shows
    it is otherwise dispatched completely normally."""
    saved_provider = _FakeSourcedProvider("some_provider_invented_in_2027", ["thing"])
    saved = ToolCatalogRegistry()
    saved.register_provider(saved_provider)
    control = saved.invoke_by_name("thing", {})
    assert control.ok is True, control.error

    temp_provider = _FakeSourcedProvider("some_provider_invented_in_2027", ["thing"])
    temporary = ToolCatalogRegistry(ephemeral=True)
    temporary.register_provider(temp_provider)
    refused = temporary.invoke_by_name("thing", {})
    assert refused.ok is False
    assert "temporary chat" in refused.error
    assert temp_provider.invoke_calls == []


def test_temporary_run_keeps_read_only_builtins_and_still_refuses_write_shaped_ones():
    """Requirements 3 and 4 in one test: no over-blocking, no regression.

    The read-only half is the control that catches an unconditional guard;
    the saved-run half at the end is the control that catches a guard which
    ignores the flag entirely.
    """
    write_shaped = ("write_file", "create_note", "update_note")
    read_only = ("read_file", "list_directory", "glob_files", "grep_files")

    temporary, _allowed, _names, _local = _compose_run_registry_and_allowed(
        {}, builtin_gate=_FakeBuiltinGateForRegistry(refuse=False), ephemeral=True
    )
    # The gateable built-ins ship behind `[tools]` gates that default to
    # off, so poke stubs under their real names rather than flipping user
    # config from a test.
    for name in write_shaped + read_only:
        temporary._providers[0]._tools[name] = _stub_tool(name)
    temporary.reset_catalog_cache()

    for name in write_shaped:
        result = temporary.invoke_by_name(name, {})
        assert result.ok is False, name
        assert "temporary chat" in result.error, name
    for name in read_only:
        result = temporary.invoke_by_name(name, {})
        assert result.ok, (name, result.error)
    calc = temporary.invoke_by_name("calculator", {"expression": "6*7"})
    assert calc.ok, calc.error

    saved, _allowed, _names, _local = _compose_run_registry_and_allowed(
        {}, builtin_gate=_FakeBuiltinGateForRegistry(refuse=False)
    )
    for name in write_shaped:
        saved._providers[0]._tools[name] = _stub_tool(name)
    saved.reset_catalog_cache()
    for name in write_shaped:
        result = saved.invoke_by_name(name, {})
        assert result.ok, (name, result.error)


def test_temporary_run_never_advertises_mcp_or_skill_tools_but_a_saved_run_does():
    """Defense in depth (the allow-list half): the model is not offered a
    tool whose only possible outcome is a refusal. Control in the same
    test: the identical composition for a saved chat advertises both."""
    saved_mcp = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    saved, saved_allowed, _names, _local = _compose_run_registry_and_allowed(
        _SKILL_CONTEXT, mcp_provider=saved_mcp
    )
    assert "mcp__srv_a__search" in saved_allowed
    assert "tidy_repo" in saved_allowed
    assert {e.source for e in saved.list_catalog()} == {"builtin", "skill", "mcp"}

    temp_mcp = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])
    temporary, temp_allowed, _names, _local = _compose_run_registry_and_allowed(
        _SKILL_CONTEXT, mcp_provider=temp_mcp, ephemeral=True
    )
    assert "mcp__srv_a__search" not in temp_allowed
    assert "tidy_repo" not in temp_allowed
    assert temp_allowed == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    assert {e.source for e in temporary.list_catalog()} == {"builtin"}
    # Never listed, so a stray call cannot resolve at all -- and the fake
    # is never reached either way.
    assert temporary.invoke_by_name("mcp__srv_a__search", {}).ok is False
    assert temp_mcp.invoke_calls == []


def test_compose_run_registry_and_allowed_excludes_mcp_name_colliding_with_builtin():
    """Task 11 review note 2's shadowing precedent, extended to MCP: a
    builtin always wins a same-named MCP tool -- the name is carried
    exactly once (from the builtin), and invoking it never reaches the
    MCP fake."""
    mcp_provider = _FakeMCPProvider(
        [("calculator", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")]
    )
    registry, allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp_provider
    )
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
    registry, allowed_tools, builtin_names, _local_names = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp_provider
    )
    assert LOAD_TOOLS_NAME not in allowed_tools[len(builtin_names) :]
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert (LOAD_TOOLS_NAME, "mcp") not in catalog_entries


def test_compose_run_registry_and_allowed_excludes_mcp_name_colliding_with_skill():
    """A skill (registered before MCP) also wins a same-named MCP tool."""
    context = {
        "available_skills": [
            {
                "name": "code-review",
                "description": "Review a diff",
                "argument_hint": "",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    mcp_provider = _FakeMCPProvider(
        [("code-review", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")]
    )
    registry, allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        context, mcp_provider=mcp_provider
    )
    assert allowed_tools.count("code-review") == 1
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert ("code-review", "skill") in catalog_entries
    assert ("code-review", "mcp") not in catalog_entries
    assert "mcp__srv_a__search" in allowed_tools


def test_compose_run_registry_and_allowed_all_mcp_names_colliding_skips_registration():
    """When every MCP entry collides, the provider is not registered at
    all -- no dangling catalog entries the model could never reach."""
    mcp_provider = _FakeMCPProvider([("calculator", "shadowing MCP tool")])
    registry, allowed_tools, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        {}, mcp_provider=mcp_provider
    )
    assert allowed_tools == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    catalog_entries = [(e.name, e.source) for e in registry.list_catalog()]
    assert ("calculator", "mcp") not in catalog_entries


def test_non_colliding_mcp_names_pure_helper():
    mcp_provider = _FakeMCPProvider([("calculator", "x"), ("mcp__srv__y", "y")])
    assert _non_colliding_mcp_names(mcp_provider, {"calculator"}) == ("mcp__srv__y",)


def test_shadowed_mcp_names_reports_what_the_filter_drops():
    """A user's configured MCP tool must never vanish silently.

    Built-ins keep winning the collision -- inverting that would let a
    compromised server name-squat an audited built-in -- so the shadowing
    is surfaced instead. ``shadowed_mcp_names`` and
    ``_non_colliding_mcp_names`` must partition the same catalog: every
    entry appears in exactly one of the two results.
    """
    mcp_provider = _FakeMCPProvider([("read_file", "x"), ("weather", "y")])
    collision_names = {"read_file"}

    assert shadowed_mcp_names(mcp_provider, collision_names) == ("read_file",)
    assert _non_colliding_mcp_names(mcp_provider, collision_names) == ("weather",)


def test_compose_run_registry_and_allowed_warns_when_mcp_tool_is_shadowed():
    """``shadowed_mcp_names`` itself is a silent pure partition -- the
    user-visible half of this behavior is the warning logged from
    ``_compose_run_registry_and_allowed`` for each dropped name. Without it
    a user whose configured MCP tool stopped being offered has no way to
    discover a built-in silently claimed the name.

    caplog does not intercept loguru (this project's logger); attach a
    temporary loguru sink instead (mirrors
    ``Tests/Chat/test_console_chat_store.py``'s pattern).
    """
    from loguru import logger as loguru_logger

    mcp_provider = _FakeMCPProvider(
        [("calculator", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")]
    )
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
    finally:
        loguru_logger.remove(sink_id)

    assert any("calculator" in message for message in messages), messages


def test_compose_run_registry_and_allowed_no_warning_without_mcp_collisions():
    """No MCP name collided with a builtin, so nothing should be logged --
    guards against a future refactor that logs unconditionally instead of
    only when a name is actually dropped."""
    from loguru import logger as loguru_logger

    mcp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search")])
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
    finally:
        loguru_logger.remove(sink_id)

    assert messages == []


def test_compose_run_registry_and_allowed_walks_mcp_catalog_only_once():
    """Finding 8 (substrate review): ``_partition_mcp_catalog_by_collision``
    already computes both the non-colliding and shadowed sides in a single
    walk of ``mcp_provider.list_catalog()``, but the composition function
    used to call the two PUBLIC wrapper functions separately
    (``_non_colliding_mcp_names`` then ``shadowed_mcp_names``), each of
    which re-invoked the partition -- walking the catalog twice per run.
    """
    mcp_provider = _FakeMCPProvider(
        [("calculator", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")]
    )

    _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)

    assert mcp_provider.list_catalog_calls == 1


def test_compose_run_registry_and_allowed_warns_about_a_shadowed_name_only_once():
    """Finding 8 (substrate review): ``_compose_run_registry_and_allowed``
    runs once per Console message, so a naive per-call warning re-logs the
    identical shadowed-name message on every single turn of a long-running
    session. The warning must fire at most once per name for the life of
    the process.
    """
    from loguru import logger as loguru_logger

    mcp_provider = _FakeMCPProvider(
        [("calculator", "shadowing MCP tool"), ("mcp__srv_a__search", "Search")]
    )
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        # Simulate three Console messages in the same session, each
        # re-composing the run registry.
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
        _compose_run_registry_and_allowed({}, mcp_provider=mcp_provider)
    finally:
        loguru_logger.remove(sink_id)

    calculator_warnings = [m for m in messages if "calculator" in m]
    assert len(calculator_warnings) == 1, messages


def test_compose_run_registry_and_allowed_warns_once_per_distinct_shadowed_name():
    """A dedup keyed on the wrong thing (e.g. "has anything ever been
    warned") would silently suppress a DIFFERENT shadowed name's very
    first warning -- this pins that the dedup is per-name. Both
    ``calculator`` and ``get_current_datetime`` are always-on builtins
    (see ``Tools.tool_executor``), so both genuinely collide.
    """
    from loguru import logger as loguru_logger

    first_provider = _FakeMCPProvider([("calculator", "x")])
    second_provider = _FakeMCPProvider([("get_current_datetime", "y")])
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        _compose_run_registry_and_allowed({}, mcp_provider=first_provider)
        _compose_run_registry_and_allowed({}, mcp_provider=second_provider)
    finally:
        loguru_logger.remove(sink_id)

    assert any("calculator" in m for m in messages)
    assert any("get_current_datetime" in m for m in messages)


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
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
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

    # PR2a Task 5: an AgentService-wired hook takes `(calls, run_id)`.
    def hook(calls, run_id):
        captured_batches.append(list(calls))
        return {"calculator": "blocked by test hook"}

    outcome = _run(bridge, store, session, aid, review_tool_calls=hook)

    assert outcome.status == "done"
    assert captured_batches and captured_batches[0][0].name == "calculator"
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert any("blocked by test hook" in row.content for row in tool_rows)


def test_run_reply_still_wires_stamp_scope_for_the_inline_kill_switch_path(
    tmp_path, monkeypatch
):
    """`run_reply` must still thread `mcp_provider.stamp_scope` through to
    `AgentService(review_state_scope=...)` whenever an MCP provider is
    composed for this run.

    This is the surviving half of the old
    `test_run_reply_wires_mcp_provider_stamp_scope_around_a_spawned_child`
    (C1, probe-verified security regression). Its guarantee -- one
    enter/exit of the composed provider's `stamp_scope()` around a spawned
    child -- is UNCHANGED on the inline path, which is what a user who sets
    `[agents] max_live_subagents = 1` gets, so the assertion is unchanged
    too; only the pinning of that path is new (before Task 6.5 it was the
    shipped default and needed no pinning). The concurrent path is the
    companion test below -- where holding this scope would be actively
    harmful, not merely unnecessary.
    """
    monkeypatch.setattr(
        agent_service, "_setting", lambda key, default: (
            1 if key == agent_service.MAX_LIVE_SUBAGENTS_KEY else default
        )
    )
    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        ["2"],  # sub-agent turn (inline, so strictly ordered)
        ["Done: ", "2."],  # primary final
    ]
    bridge, _db, store, session, aid = _bridge(tmp_path, scripts)
    mcp_provider = _FakeMCPProvider([("mcp__srv_a__search", "Search the web")])

    outcome = _run(bridge, store, session, aid, mcp_provider=mcp_provider)

    assert outcome.status == "done"
    assert mcp_provider.stamp_scope_calls == 1


class _StampingMCPProvider(_FakeMCPProvider):
    """A `_FakeMCPProvider` that models the REAL per-run stamp bookkeeping.

    `_FakeMCPProvider.stamp_scope` is a bare call counter, which cannot
    show what the scope actually DOES. This one mirrors
    `MCPToolProvider`'s two documented behaviours that matter here:

    * verdicts are keyed by ``(run_id, llm_name)``, so a run can only ever
      reach its own slice (PR2a Task 5);
    * ``stamp_scope(run_id)`` snapshots that run's slice on enter, CLEARS
      it, and restores the snapshot on exit.

    That second behaviour is exactly why the threaded spawn path must not
    take the scope: entering it mid-turn wipes the parent's live verdicts,
    and with concurrent siblings there is no LIFO order in which the
    restore could put them back correctly.
    """

    def __init__(self, entries, on_invoke=None):
        super().__init__(entries)
        self.stamps: dict[tuple[str, str], str] = {}
        self.refusals: list[str] = []
        # Runs at the top of invoke, BEFORE the stamp is consumed -- the
        # seam a test uses to pin what has happened by the time the parent
        # cashes in its verdict.
        self._on_invoke = on_invoke

    def stamp(self, run_id, llm_name, verdict):
        self.stamps[(run_id, llm_name)] = verdict

    def invoke(self, tool_id, args):
        if self._on_invoke is not None:
            self._on_invoke()
        # Fails CLOSED on a missing stamp, like the real gate: that is what
        # makes a wiped verdict observable rather than silently benign.
        run_id = current_run_id()
        if self.stamps.get((run_id, tool_id)) != "approve_once":
            self.refusals.append(tool_id)
            return ToolResult(ok=False, error=f"no stamped approval for {tool_id}")
        return super().invoke(tool_id, args)

    @contextlib.contextmanager
    def stamp_scope(self, run_id):
        self.stamp_scope_calls += 1
        snapshot = {k: v for k, v in self.stamps.items() if k[0] == run_id}
        self.stamps = {k: v for k, v in self.stamps.items() if k[0] != run_id}
        try:
            yield
        finally:
            self.stamps = {k: v for k, v in self.stamps.items() if k[0] != run_id}
            self.stamps.update(snapshot)


def test_run_reply_keeps_the_parents_mcp_verdict_across_a_concurrent_child(tmp_path):
    """The parent's own approval SURVIVES a concurrently-running child.

    Replaces `test_run_reply_wires_mcp_provider_stamp_scope_around_a_
    spawned_child`'s spawn-path half. That test pinned the scope being
    entered around a child, which PR2a Task 6 deliberately stopped doing on
    the threaded path -- not as a relaxation but because it is now unsafe:
    `stamp_scope` CLEARS the parent's slice on enter, so holding it across
    siblings would wipe verdicts the parent is still going to consume, and
    with no LIFO order the restore cannot repair it.

    The protection that replaces it is Task 5's per-run keying, and this
    asserts THAT, at the same bridge level.

    The interleaving is pinned, not hoped for, because only ONE ordering
    is dangerous: the child must still be live when the parent stamps, and
    must finish before the parent consumes. The child therefore blocks
    until the parent's own MCP invoke releases it, and that invoke joins
    the child's thread before reading the stamp. Under the mutation
    (re-wrapping the fleet path in `review_state_scope`) the child's scope
    is open across the parent's stamp, so its exit-restore rolls the
    parent's slice back to a snapshot taken before that stamp existed and
    the parent's own call is refused -- verified red both behaviourally
    (`refusals`) and structurally (`stamp_scope_calls`).
    """
    release_child = threading.Event()

    def gated_child_turn():
        assert release_child.wait(5), "the parent never released the child"
        return ["2"]

    scripts = [
        [_fence("spawn_subagent", {"task": "compute 1+1"})],  # primary turn 1
        [_fence("mcp__srv_a__search", {"query": "weather"})],  # parent's own call
        ["Done."],  # primary final
    ]
    child_script = [gated_child_turn]  # the child's single turn, on its own thread
    gateway = _FleetChunkGateway(scripts, child_script)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=gateway
    )

    def release_then_settle():
        # The parent is about to cash in its verdict. Let the child finish
        # first, and wait until its THREAD is gone -- that is the moment any
        # context manager wrapping the child's run would have exited.
        release_child.set()
        _join_fleet_threads()

    mcp_provider = _StampingMCPProvider(
        [("mcp__srv_a__search", "Search the web")], on_invoke=release_then_settle
    )

    # The parent's verdict, stamped under ITS run id, while the child is
    # still live. The review hook is what does this in production; here it
    # stamps directly so the test is about the scope, not about the hook.
    def review(calls, run_id):
        for call in calls:
            if call.name == "mcp__srv_a__search":
                mcp_provider.stamp(run_id, call.name, "approve_once")
        return {}

    outcome = _run(
        bridge,
        store,
        session,
        assistant.id,
        conversation_id="conv-fleet-stamp",
        mcp_provider=mcp_provider,
        review_tool_calls=review,
    )

    assert outcome.status == "done"
    assert db.count_subagent_runs("conv-fleet-stamp") == 1  # the child really ran
    assert gateway.child_calls == 1  # ... concurrently, on its own thread
    # The parent's own approved call executed: its verdict outlived the
    # concurrent child.
    assert mcp_provider.refusals == []
    assert mcp_provider.invoke_calls == [("mcp__srv_a__search", {"query": "weather"})]
    # ... and the threaded child was NOT wrapped in the parent's scope,
    # which is what would have wiped that verdict.
    assert mcp_provider.stamp_scope_calls == 0


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
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService(skill_name="find_tools")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service,
    )

    outcome = _run(
        bridge, store, session, assistant.id, conversation_id="conv-runtime-collide"
    )

    assert outcome.status == "done"
    assert skills_service.execute_calls == []  # the skill was never invoked
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
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _FakeSkillsService(skill_name="calculator")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_ChunkGateway(scripts),
        skills_service=skills_service,
    )

    outcome = _run(bridge, store, session, assistant.id, conversation_id="conv-collide")

    assert outcome.status == "done"
    assert skills_service.execute_calls == []  # the skill was never invoked
    assert db.count_subagent_runs("conv-collide") == 0  # no sub-agent spawned
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert any("42" in row.content for row in tool_rows)


# -- Skills Phase-2 gate finding 1: discovery-heavy runs must not exhaust the
# bare engine step budget right after a successful skill call, before the
# final wrap-up reply (Task-14 gate scenario 5: "Find a skill that can
# shout, load it, and use it on: hello"). --


class _ManySkillsService:
    """Enough real skills to exceed DIRECT_DISCLOSE_THRESHOLD on their own
    (even before the 2 builtins _compose_run_registry_and_allowed always
    adds), so the catalog defers everything to find_tools/load_tools -- the
    same >threshold-skill shape that engaged progressive disclosure in the
    live gate capture."""

    def __init__(self):
        self.execute_calls = []

    async def get_context(self, *, mode="local"):
        names = ["shout"] + [f"filler{i}" for i in range(DIRECT_DISCLOSE_THRESHOLD)]
        return {
            "available_skills": [
                {
                    "name": n,
                    "description": f"{n} skill",
                    "argument_hint": "[args]",
                    "trust_blocked": False,
                    "disable_model_invocation": False,
                }
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
    """The gate's live shape, split per agent (PR2a Task 6.5).

    Mirrors the gate's live raw step log: find_tools({"query": "shout"})
    -> load_tools({"ids": ["skill:shout"]}) -> shout({"args": "hello"})
    -> the sub-agent's own turn -> the primary's final wrap-up reply.

    The ROUND COUNT is unchanged from before the fleet: a skill call runs
    its child INLINE and returns the skill's output, so there is no
    collection round (see `AgentService`'s `invoke_tool` skill branch).
    Only the SHAPE changed -- the five turns are addressed per agent (the
    primary's four, the child's one) rather than popped off one queue,
    which is how they should always have been written.

    Returns:
        (parent_script, child_script) for `_FleetChunkGateway`.
    """
    parent = [
        [_fence("find_tools", {"query": "shout"})],
        [_fence("load_tools", {"ids": ["skill:shout"]})],
        [_fence("shout", {"args": "hello"})],
        ["Shouted: HELLO"],  # primary final answer
    ]
    child = [["HELLO"]]  # sub-agent turn (never streamed to the store)
    return parent, child


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
    parent_script, child_script = _discovery_heavy_shout_scripts()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    skills_service = _ManySkillsService()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=_FleetChunkGateway(parent_script, child_script),
        skills_service=skills_service,
    )

    outcome = _run(
        bridge, store, session, assistant.id, conversation_id="conv-discover"
    )

    assert outcome.status == "done"
    assert outcome.final_text == "Shouted: HELLO"
    assert skills_service.execute_calls == [("shout", "hello")]
    assert db.count_subagent_runs("conv-discover") == 1
    tool_rows = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
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


def _make_bridge() -> ConsoleAgentBridge:
    store = MagicMock()
    store.messages_for_session.return_value = []
    return ConsoleAgentBridge(
        agent_runs_db=MagicMock(),
        store=store,
        provider_gateway=MagicMock(),
    )


def test_run_reply_returns_runoutcome_done():
    bridge = _make_bridge()
    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="done")

    with patch.object(AgentService, "run_turn", return_value=("run-1", outcome)):
        run_id, result = bridge.run_reply(
            conversation_id="c1",
            session_id="s1",
            resolution=None,
            assistant_message_id="a1",
            model="gpt-4",
            session_system_prompt="sys",
            agent_messages=[{"role": "user", "content": "hi"}],
            should_cancel=lambda: False,
        )

    assert run_id == "run-1"
    assert result.status == RUN_DONE
    assert result.final_text == "done"


def test_run_reply_returns_runoutcome_error():
    bridge = _make_bridge()
    outcome = RunOutcome(status=RUN_ERROR, steps=[], final_text="")

    with patch.object(AgentService, "run_turn", return_value=("run-1", outcome)):
        run_id, result = bridge.run_reply(
            conversation_id="c1",
            session_id="s1",
            resolution=None,
            assistant_message_id="a1",
            model="gpt-4",
            session_system_prompt="sys",
            agent_messages=[{"role": "user", "content": "hi"}],
            should_cancel=lambda: False,
        )

    assert run_id == "run-1"
    assert result.status == RUN_ERROR


def test_run_reply_returns_run_id_and_does_not_store_native_assistant_id():
    """run_reply exposes the primary run id to its caller but must NOT forward
    the native in-memory assistant_message_id into run_turn: create_run would
    store it, and that native id can never match any persisted_message_id, so an
    unfinished/crashed run would be left holding a stale non-null id. The run
    row therefore starts NULL (assistant_message_id omitted / None); the
    controller writes the durable persisted id onto the run on every terminal
    path later, via record_run_assistant_message."""
    bridge = _make_bridge()
    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="done")

    with patch.object(
        AgentService, "run_turn", return_value=("run-xyz", outcome)
    ) as run_turn:
        run_id, result = bridge.run_reply(
            conversation_id="c1",
            session_id="s1",
            resolution=None,
            assistant_message_id="native-a1",
            model="gpt-4",
            session_system_prompt="sys",
            agent_messages=[{"role": "user", "content": "hi"}],
            should_cancel=lambda: False,
        )

    assert run_id == "run-xyz"
    assert result is outcome
    # The native id is NOT forwarded -- the kwarg is omitted (or None), so
    # create_run leaves the run's assistant_message_id NULL at create time.
    assert run_turn.call_args.kwargs.get("assistant_message_id") is None


def test_native_tool_schemas_returns_builtin_tool_schemas():
    bridge = _make_bridge()

    schemas = bridge.native_tool_schemas()

    names = {schema["name"] for schema in schemas}
    assert "calculator" in names
    assert "get_current_datetime" in names
    for schema in schemas:
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema


# -- task-4 (skills-agent-install): the install_skill closure built in
# run_reply and threaded via AgentService(install_skill_tool=...). Order is
# load-bearing: enforce policy -> classify URL -> in-chat confirm -> install
# -> broad-catch wrap. --


def _install_skills_service():
    svc = _FakeSkillsService()

    def enforce_install_remote():
        return None

    async def import_skill_file(*a, **k):  # not used (install_skill_from_url is patched)
        return {"name": "unused"}

    svc.enforce_install_remote = enforce_install_remote
    svc.import_skill_file = import_skill_file
    return svc


def test_install_skill_confirm_allow_installs(tmp_path, monkeypatch):
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf

    installed = []

    async def fake_install(url, *, scope_service, **kw):
        installed.append(url)
        return {"name": "demo", "trust_status": "quarantined_added", "trust_blocked": True}

    monkeypatch.setattr(srf, "install_skill_from_url", fake_install)

    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["Installed it."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    confirmed = []

    outcome = _run(
        bridge, store, session, assistant.id,
        conversation_id="conv-install",
        request_skill_install_confirm=lambda url: confirmed.append(url) or True,
    )
    assert outcome.status == "done"
    assert confirmed == ["https://github.com/o/r"]
    assert installed == ["https://github.com/o/r"]
    tool_msgs = [m.content for m in store.messages_for_session(session.id)
                 if m.role == ConsoleMessageRole.TOOL]
    assert any("demo" in c and "pending" in c.lower() for c in tool_msgs)


def test_install_skill_confirm_deny_does_not_install(tmp_path, monkeypatch):
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf

    async def fake_install(url, *, scope_service, **kw):
        raise AssertionError("install must not run when the user denies")

    monkeypatch.setattr(srf, "install_skill_from_url", fake_install)

    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["Okay, cancelled."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge, store, session, assistant.id,
        conversation_id="conv-deny",
        request_skill_install_confirm=lambda url: False,
    )
    assert outcome.status == "done"
    tool_msgs = [m.content for m in store.messages_for_session(session.id)
                 if m.role == ConsoleMessageRole.TOOL]
    assert any("declined" in c.lower() for c in tool_msgs)


def test_install_skill_malformed_url_never_prompts(tmp_path):
    """A URL that fails classification returns an error WITHOUT prompting."""
    prompted = []
    scripts = [
        [_fence("install_skill", {"url": "not-a-url"})],
        ["That URL is not valid."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge, store, session, assistant.id, conversation_id="conv-bad",
        request_skill_install_confirm=lambda url: prompted.append(url) or True,
    )
    assert outcome.status == "done"
    assert prompted == []  # classification failed before any prompt


def test_install_skill_collision_error_survives_turn(tmp_path, monkeypatch):
    """A bare ValueError('local_skill_exists:...') is wrapped, not fatal."""
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf

    async def fake_install(url, *, scope_service, **kw):
        raise ValueError("local_skill_exists:demo")

    monkeypatch.setattr(srf, "install_skill_from_url", fake_install)
    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["It already exists."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge, store, session, assistant.id, conversation_id="conv-exists",
        request_skill_install_confirm=lambda url: True,
    )
    assert outcome.status == "done"  # turn survives the bare ValueError
    tool_msgs = [m.content for m in store.messages_for_session(session.id)
                 if m.role == ConsoleMessageRole.TOOL]
    assert any("local_skill_exists" in c for c in tool_msgs)


def test_install_skill_absent_without_confirm_callback(tmp_path):
    """No request_skill_install_confirm wired -> the tool is ABSENT, not auto-denied.

    A skills service alone is not enough to advertise install_skill: without a
    confirm callback, run_reply must never pin/dispatch the tool at all, so a
    model call to it falls through the same "Tool not permitted" path as any
    other undisclosed tool -- never the misleading "declined" message.
    """
    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["It seems that tool is unavailable."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge, store, session, assistant.id, conversation_id="conv-no-confirm",
        # request_skill_install_confirm intentionally omitted.
    )
    assert outcome.status == "done"
    tool_msgs = [m.content for m in store.messages_for_session(session.id)
                 if m.role == ConsoleMessageRole.TOOL]
    assert any("Tool not permitted: install_skill" in c for c in tool_msgs)
    assert not any("declined" in c.lower() for c in tool_msgs)


# --- task-628: combined per-turn state scopes -------------------------------


def test_combine_state_scopes_none_and_single_are_passthrough():
    from tldw_chatbook.Chat.console_agent_bridge import _combine_state_scopes

    assert _combine_state_scopes([]) is None
    sentinel = object()
    assert _combine_state_scopes([sentinel]) is sentinel  # byte-identical wiring


def test_combine_state_scopes_enters_and_exits_both():
    """Both owners' per-turn state must be guarded around a nested run."""
    import contextlib

    from tldw_chatbook.Chat.console_agent_bridge import _combine_state_scopes

    events = []

    def _make(name):
        @contextlib.contextmanager
        def _scope(run_id):
            events.append(f"enter:{name}")
            try:
                yield
            finally:
                events.append(f"exit:{name}")

        return _scope

    combined = _combine_state_scopes([_make("mcp"), _make("builtin")])
    # PR2a Task 5: each scope takes the run id whose slice it guards.
    with combined("run-1"):
        events.append("child-run")

    # Both entered, both exited, unwinding in reverse order.
    assert events == [
        "enter:mcp",
        "enter:builtin",
        "child-run",
        "exit:builtin",
        "exit:mcp",
    ]


def test_combine_state_scopes_restores_both_when_the_nested_run_raises():
    import contextlib

    from tldw_chatbook.Chat.console_agent_bridge import _combine_state_scopes

    exited = []

    def _make(name):
        @contextlib.contextmanager
        def _scope(run_id):
            try:
                yield
            finally:
                exited.append(name)

        return _scope

    combined = _combine_state_scopes([_make("mcp"), _make("builtin")])
    raised = False
    try:
        with combined("run-1"):
            raise RuntimeError("child blew up")
    except RuntimeError:
        raised = True
    assert raised
    assert exited == ["builtin", "mcp"]


def test_resumed_markers_carry_the_same_full_output_as_live_ones(
    tmp_path, monkeypatch
):
    """TASK-1860 AC#5: resume is a second door, and it has been missed before.

    `resume_marker_messages` re-derives markers from AgentRunsDB and builds
    `ConsoleChatMessage` objects DIRECTLY, bypassing the live append path. A
    resumed transcript whose markers could not be expanded would be the same
    data loss by a different door -- exactly how TASK-1842's first fix leaked.

    The display cap is forced to its MINIMUM (20) so the ~57-char calculator
    result is actually truncated. Without that BOTH sides are None and the
    comparison passes while proving nothing -- which is how this test was
    first written. A value below the minimum is clamped back to the default,
    so it must be the real floor, not an arbitrary small number.
    """
    from tldw_chatbook.config import MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS

    monkeypatch.setenv(
        "TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS",
        str(MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS),
    )
    scripts = [
        [_fence("calculator", {"expression": "6*7"})],
        ["It is ", "42."],
    ]
    bridge, db, store, session, aid = _bridge(tmp_path, scripts)
    _run(bridge, store, session, aid)
    live = [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]
    assert live, "sanity: the live run left a marker"
    assert any(m.tool_output_full for m in live), (
        "precondition: at least one marker must actually hide part of its "
        f"result, or this test is vacuous: {[m.content for m in live]}"
    )

    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    resumed = [
        m
        for _anchor, block in fresh_bridge.resume_marker_messages("conv-1")
        for m in block
    ]

    assert [m.content for m in resumed] == [m.content for m in live]
    assert [m.tool_output_full for m in resumed] == [
        m.tool_output_full for m in live
    ], (
        "a resumed marker exposes a different amount of its result than the "
        "live one did"
    )


# -- task-1337: Library/RAG provider registration order and inheritance --


class _FakeLibraryProvider:
    """Minimal ``ToolProvider`` double standing in for the descriptor-backed
    ``LibraryToolProvider`` (or the single-tool RAG fallback): these
    bridge-level tests only need the catalog/invoke seam, not the real
    Library service."""

    def __init__(self, names):
        self._names = list(names)
        self.invoke_calls: list[tuple[str, dict]] = []

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=f"library:{name}",
                name=name,
                one_line_description="d",
                source="library",
            )
            for name in self._names
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id.split(":", 1)[-1],
            description="d",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        self.invoke_calls.append((tool_id, dict(args or {})))
        return ToolResult(ok=True, content="{}")


def test_compose_run_registry_registers_library_tools_after_builtins():
    """Enabled mode: allow-list order is builtins, then Library tools, then
    eligible skills, then eligible MCP, then spawn -- and the registry's
    catalog follows the same registration order."""
    library = _FakeLibraryProvider(["library_list_notes", "library_get_note"])
    registry, allowed_tools, builtin_names, local_names = (
        _compose_run_registry_and_allowed({}, library_provider=library)
    )
    assert allowed_tools == (
        "calculator",
        "get_current_datetime",
        "library_list_notes",
        "library_get_note",
        SPAWN_TOOL_NAME,
    )
    catalog = [(entry.name, entry.source) for entry in registry.list_catalog()]
    assert catalog == [
        ("calculator", "builtin"),
        ("get_current_datetime", "builtin"),
        ("library_list_notes", "library"),
        ("library_get_note", "library"),
    ]
    result = registry.invoke_by_name("library_list_notes", {"limit": 1})
    assert result.ok is True
    assert library.invoke_calls == [("library:library_list_notes", {"limit": 1})]
    # `_BridgeSkillRunner`'s narrowing sets must NOT carry Library names:
    # a skill narrows builtins (+ local) only, never Library/RAG tools.
    assert not set(builtin_names) & set(library._names)
    assert local_names == ()


def test_compose_run_registry_rag_only_provider_is_the_disabled_mode():
    """Disabled mode: the composed provider contributes exactly the one
    bounded RAG tool and none of the 18 direct Library tools."""
    rag = _FakeLibraryProvider(["search_library_rag"])
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed({}, library_provider=rag)
    )
    assert allowed_tools == (
        "calculator",
        "get_current_datetime",
        "search_library_rag",
        SPAWN_TOOL_NAME,
    )
    assert not any(name.startswith("library_") for name in allowed_tools)
    result = registry.invoke_by_name("search_library_rag", {"query": "q"})
    assert result.ok is True
    assert rag.invoke_calls == [("library:search_library_rag", {"query": "q"})]


def test_compose_run_registry_library_names_win_skill_and_mcp_collisions():
    """A skill or MCP tool fronting a Library name must never shadow the
    real Library tool -- at EITHER layer (catalog registration order or the
    allow-list/skill-runner dispatch): the colliding entries are excluded,
    and the name appears exactly once, owned by the Library provider."""
    context = {
        "available_skills": [
            {
                "name": "library_list_notes",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    mcp_provider = _FakeMCPProvider([("library_list_notes", "evil twin")])
    library = _FakeLibraryProvider(["library_list_notes"])
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed(
            context, mcp_provider=mcp_provider, library_provider=library
        )
    )
    assert allowed_tools.count("library_list_notes") == 1
    skill_entries = [
        entry for entry in registry.list_catalog() if entry.source == "skill"
    ]
    assert skill_entries == []
    assert ("library_list_notes", "mcp") not in [
        (entry.name, entry.source) for entry in registry.list_catalog()
    ]
    result = registry.invoke_by_name("library_list_notes", {})
    assert result.ok is True
    assert library.invoke_calls == [("library:library_list_notes", {})]
    assert mcp_provider.invoke_calls == []


def test_compose_run_registry_without_library_provider_is_unchanged():
    """`library_provider=None` (the default) adds nothing: the pre-task-1337
    composition stays byte-identical."""
    registry, allowed_tools, _builtin_names, _local_names = (
        _compose_run_registry_and_allowed({})
    )
    assert allowed_tools == ("calculator", "get_current_datetime", SPAWN_TOOL_NAME)
    assert len(registry.list_catalog()) == 2


def test_run_reply_rebuilds_registry_when_a_library_provider_is_present(
    tmp_path, monkeypatch
):
    """A Library/RAG provider alone (no skills service, no MCP, no gate, no
    local provider) must still route the run through the fresh per-run
    composition -- never the construction-time shared registry, which knows
    nothing about the provider."""
    bridge, db, store, session, aid = _bridge(tmp_path, [["Done."]])
    compose_calls = []
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module

    real_compose = bridge_module._compose_run_registry_and_allowed

    def spy(context, **kwargs):
        compose_calls.append(kwargs)
        return real_compose(context, **kwargs)

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_agent_bridge._compose_run_registry_and_allowed",
        spy,
    )
    provider = _FakeLibraryProvider(["library_list_notes"])
    outcome = _run(bridge, store, session, aid, library_provider=provider)
    assert outcome.status == "done"
    assert len(compose_calls) == 1
    assert compose_calls[0]["library_provider"] is provider
