# Tests/Agents/test_agent_service_review_state_scope.py
"""C1 (SECURITY, probe-verified): sub-agent turns must not clobber
parent-turn MCP approval stamps -- a DENIED call could execute.

Mechanism (reviewer-verified): `MCPToolProvider._stamped_decisions` lives in
ONE dict shared by the whole run tree. `apply_batch_decisions` REPLACES it
every turn (Finding F1), never merges. `spawn_subagent` runs the child's
ENTIRE agent loop INLINE, synchronously, mid-parent-dispatch
(`AgentService._run_one`'s `spawn` closure, called directly from
`agent_runtime.run_agent_loop`'s own per-call dispatch loop -- see
`agent_runtime.py`'s `for call in calls:`). So: a parent turn batches
[spawn_subagent, mcp_X], the review hook stamps mcp_X "deny" for THIS turn,
spawn_subagent dispatches inline and the child's OWN turn re-stamps the SAME
provider's dict for its own (different) decision on mcp_X -- by the time
control returns to the parent's own dispatch loop for its own mcp_X call, the
stamp it peeks is the CHILD's, not its own. A parent-denied call executes.

Fix: `AgentService(review_state_scope=...)` -- an optional, generic seam that
wraps every nested `_run_one` call in a caller-supplied context manager.
`MCPToolProvider.stamp_scope()` is that context manager for the MCP case:
snapshot `_stamped_decisions` on enter, restore (not merge) it on exit, so
the child's mutations to the shared dict are fully undone the instant it
returns control to the parent.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    SPAWN_TOOL_NAME,
    AgentConfig,
    ToolCall,
)
from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT, AgentService
from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.Agents.mcp_tool_provider import DENY_REFUSAL, MCPToolProvider
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_controller import (
    build_mcp_review_hook,
    build_tool_review_hook,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.tool_executor import Tool


def _catalog_record(profile_id: str, tools: list[dict]) -> dict:
    return {
        "profile_id": profile_id,
        "is_connected": True,
        "discovery_snapshot": {"tools": tools},
    }


def _tool_dict(name: str, description: str = "") -> dict:
    return {"name": name, "description": description}


class FakeMCPService:
    """Mirrors the real service seams `MCPToolProvider` consumes (a smaller,
    file-local twin of `Tests/Agents/test_mcp_tool_provider.py`'s fake of the
    same name -- kept self-contained here since this scenario only needs a
    single always-"ask" tool)."""

    def __init__(self) -> None:
        self.default_state = EffectiveToolState(state="ask", origin="global_default")
        self.session_approvals: set[tuple[str, str]] = set()
        self.record_tool_decision_calls: list[tuple] = []
        self.execute_calls: list[tuple] = []
        self.local_service = SimpleNamespace(get_inventory=lambda: {"tools": []})

    def get_kill_switch(self) -> bool:
        return False

    async def local_external_catalog(self) -> list[dict]:
        return [_catalog_record("srv", [_tool_dict("run")])]

    def effective_tool_states(self, tools):
        return {(t.server_key, t.name): self.default_state for t in tools}

    def gate_tool_test(self, tool):
        return self.default_state

    def is_session_approved(self, server_key: str, tool_name: str) -> bool:
        return (server_key, tool_name) in self.session_approvals

    def approve_for_session(self, server_key: str, tool_name: str) -> None:
        self.session_approvals.add((server_key, tool_name))

    def set_tool_state(
        self, server_key: str, tool_name: str, ui_state, *, tool=None
    ) -> None:
        pass

    def record_tool_decision(
        self,
        server_key: str,
        tool_name: str,
        *,
        decision: str,
        initiator: str = "agent",
        error: str | None = None,
    ) -> None:
        self.record_tool_decision_calls.append(
            (server_key, tool_name, decision, initiator, error)
        )

    def _tool_call_timeout(self) -> float:
        return 5.0

    async def execute_hub_tool(
        self,
        server_key: str,
        tool_name: str,
        arguments: dict | None = None,
        *,
        initiator: str = "test",
        decision: str = "allowed",
        timeout_seconds: float | None = None,
    ) -> dict:
        self.execute_calls.append(
            (server_key, tool_name, dict(arguments or {}), initiator, decision)
        )
        return {"content": [{"type": "text", "text": "executed"}]}


@pytest.fixture
def running_loop():
    """A real event loop running forever in a background thread (the "main loop" stand-in)."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert ready.wait(timeout=2), "background loop failed to start"
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


@pytest.fixture
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="t")


def _native_turn(calls: list[ToolCall], text: str = "") -> dict:
    raw = [
        {
            "id": c.call_id,
            "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.args)},
        }
        for c in calls
    ]
    return {"content": text, "tool_calls": raw}


class ScriptedChat:
    """Distinguishes parent vs. child turns by the leading system message --
    child turns always start with `SUBAGENT_SYSTEM_PROMPT`, mirroring
    `console_agent_bridge._StreamingModelAdapter._is_subagent`'s own
    production discriminator for the exact same distinction."""

    def __init__(self, parent_replies: list, child_replies: list) -> None:
        self.parent_replies = list(parent_replies)
        self.child_replies = list(child_replies)
        self.calls: list[list[dict]] = []

    def __call__(self, *, messages_payload, **_kwargs) -> dict:
        self.calls.append(messages_payload)
        is_child = bool(messages_payload) and (
            messages_payload[0].get("role") == "system"
            and str(messages_payload[0].get("content", "")).startswith(
                SUBAGENT_SYSTEM_PROMPT
            )
        )
        queue = self.child_replies if is_child else self.parent_replies
        item = queue.pop(0)
        message = item if isinstance(item, dict) else {"content": item}
        return {"choices": [{"message": message}]}


def _registry_with_mcp(provider: MCPToolProvider) -> ToolCatalogRegistry:
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(provider)
    return registry


# ---------------------------------------------------------------------------
# C1: the reviewer's exact adversarial interleave
# ---------------------------------------------------------------------------


def test_parent_deny_is_not_overridden_by_a_same_turn_spawned_childs_approval(
    db, running_loop
):
    """Parent batch = [spawn_subagent, mcp_X]. Parent denies mcp_X. The
    spawned child (dispatched INLINE, mid-parent-dispatch) approves its OWN
    call to the SAME mcp_X. The parent's remaining (second) call to mcp_X
    must still be denied -- never silently executed on the child's leftover
    stamp."""
    service = FakeMCPService()
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    asyncio.run(provider.compose_catalog())
    tool_id = provider.list_catalog()[0].id
    registry = _registry_with_mcp(provider)

    approval_calls: list[list[str]] = []

    def request_mcp_approvals(pending):
        approval_calls.append([p.llm_name for p in pending])
        if len(approval_calls) == 1:
            return {tool_id: "deny"}  # the PARENT's own decision
        return {tool_id: "approve_once"}  # the CHILD's own (different) decision

    review_hook = build_mcp_review_hook(provider, request_mcp_approvals)

    spawn_call = ToolCall(
        name=SPAWN_TOOL_NAME, args={"task": "call the tool"}, call_id="p-spawn"
    )
    parent_mcp_call = ToolCall(name=tool_id, args={"q": "parent"}, call_id="p-mcp")
    child_mcp_call = ToolCall(name=tool_id, args={"q": "child"}, call_id="c-mcp")

    chat = ScriptedChat(
        parent_replies=[_native_turn([spawn_call, parent_mcp_call]), "parent done"],
        child_replies=[_native_turn([child_mcp_call]), "child done"],
    )

    agent_service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=review_hook,
        review_state_scope=provider.stamp_scope,
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(tool_id, SPAWN_TOOL_NAME),
        native_tools=True,
    )

    _run_id, outcome = agent_service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="openai",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert len(approval_calls) == 2  # one round trip for the parent, one for the child

    # Exactly ONE execution happened for the whole run tree: the child's own
    # approved call. The parent's own (denied) call must never reach
    # execute_hub_tool at all.
    assert service.execute_calls == [
        ("local:srv", "run", {"q": "child"}, "agent", "approved")
    ]

    # The parent's own remaining call was refused, not executed.
    tool_result_steps = [
        s for s in outcome.steps if s.kind == "tool_result" and s.tool_name == tool_id
    ]
    assert len(tool_result_steps) == 1
    assert tool_result_steps[0].result == f"ERROR: {DENY_REFUSAL}"
    assert (
        "local:srv",
        "run",
        "denied",
        "agent",
        None,
    ) in service.record_tool_decision_calls


# ---------------------------------------------------------------------------
# C1 benign direction: a child's routine empty apply must not wipe a
# genuine parent approval
# ---------------------------------------------------------------------------


def test_child_run_does_not_wipe_a_parent_approval_via_its_own_empty_apply(
    db, running_loop
):
    """The child spawned this turn calls a NON-MCP tool (calculator) -- its
    OWN review-hook invocation resolves `pending=[]` and (routinely, every
    turn with tool calls) calls `apply_batch_decisions({})`. That routine
    clear must not wipe out the PARENT's own earlier approval for its OWN
    remaining same-turn mcp call: an approved call must still execute after
    the child returns."""
    service = FakeMCPService()
    provider = MCPToolProvider(service=service, main_loop=running_loop)
    asyncio.run(provider.compose_catalog())
    tool_id = provider.list_catalog()[0].id
    registry = _registry_with_mcp(provider)

    approval_calls: list[list[str]] = []

    def request_mcp_approvals(pending):
        approval_calls.append([p.llm_name for p in pending])
        return {tool_id: "approve_once"}

    review_hook = build_mcp_review_hook(provider, request_mcp_approvals)

    spawn_call = ToolCall(
        name=SPAWN_TOOL_NAME, args={"task": "use the calculator"}, call_id="p-spawn"
    )
    parent_mcp_call = ToolCall(name=tool_id, args={"q": "parent"}, call_id="p-mcp")
    calc_call = ToolCall(
        name="calculator", args={"expression": "1+1"}, call_id="c-calc"
    )

    chat = ScriptedChat(
        parent_replies=[_native_turn([spawn_call, parent_mcp_call]), "parent done"],
        child_replies=[_native_turn([calc_call]), "child done"],
    )

    agent_service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=review_hook,
        review_state_scope=provider.stamp_scope,
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(tool_id, "calculator", SPAWN_TOOL_NAME),
        native_tools=True,
    )

    _run_id, outcome = agent_service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="openai",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    # Only ONE request_mcp_approvals round trip ever happens (the parent's) --
    # the child's own calculator call never needed MCP gating at all.
    assert len(approval_calls) == 1

    # The parent's own approved call still executed after the child (whose
    # own turn cleared the provider's live stamp dict) returned.
    assert service.execute_calls == [
        ("local:srv", "run", {"q": "parent"}, "agent", "approved")
    ]

    tool_result_steps = [
        s for s in outcome.steps if s.kind == "tool_result" and s.tool_name == tool_id
    ]
    assert len(tool_result_steps) == 1
    assert not tool_result_steps[0].result.startswith("ERROR")


# ---------------------------------------------------------------------------
# C1: absent review_state_scope stays byte-identical (no MCP, or no wiring)
# ---------------------------------------------------------------------------


def test_review_state_scope_defaults_to_none_and_spawn_still_works(db):
    """Omitting `review_state_scope` (every caller before this task, and
    every non-MCP run today) must not change existing spawn behavior --
    `spawn` falls back to a no-op `contextlib.nullcontext()`."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())

    spawn_call = ToolCall(
        name=SPAWN_TOOL_NAME, args={"task": "say hi"}, call_id="p-spawn"
    )
    chat = ScriptedChat(
        parent_replies=[_native_turn([spawn_call]), "parent done"],
        child_replies=["child done"],
    )
    agent_service = AgentService(db=db, registry=registry, chat_call=chat)
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(SPAWN_TOOL_NAME,),
        native_tools=True,
    )

    _run_id, outcome = agent_service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="openai",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.subagents_spawned == 1


# ---------------------------------------------------------------------------
# task-628: the built-in gate's own nested-clobber (BuiltinToolGate.stamp_
# scope mirrors MCPToolProvider.stamp_scope for a shared BuiltinToolGate).
# Unlike MCP, `BuiltinToolGate.check()` has no per-call approval fallback --
# a lost stamp fails CLOSED outright rather than re-prompting -- so an
# unguarded shared gate would make an approved mutating built-in tool
# unusable from inside any sub-agent the instant P2 tags one.
# ---------------------------------------------------------------------------


class _MutatingBuiltinTool(Tool):
    """A real ``Tool`` tagged ``risk_tags=("mutates",)``.

    P2 has not shipped an actual mutating built-in tool yet (task-628's own
    AC#3 blocks it on this task), so this mirrors
    ``test_builtin_tool_gate.py``'s own ``_Mutating`` -- a minimal stand-in
    good enough to reach ``resolve_builtin_state``'s high-risk floor
    (inherited ``allow`` -> ``ask`` for a tag in ``HIGH_RISK_TAGS``).
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "d"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def risk_tags(self) -> tuple[str, ...]:
        return ("mutates",)

    async def execute(self, **kwargs) -> dict:
        return {"wrote": self._name}


def _tool_result_collector():
    """An ``on_step`` sink indexing every ``"tool_result"`` step -- parent OR
    any nested child -- by tool name.

    ``RunOutcome.steps`` (what ``run_turn`` returns) is scoped to the ONE
    ``_run_one`` call it came from: a spawned child's own steps live only
    in the child's own ``RunOutcome``, which ``spawn()`` discards after
    reading ``final_text`` (see ``agent_service.py``'s ``spawn`` closure) --
    they are never merged into the parent's returned ``outcome.steps``.
    Every step of every run in the tree DOES reach ``AgentService``'s
    ``on_step`` callback, tagged with its own ``agent_kind`` -- the same
    seam ``console_agent_bridge.py`` itself uses in production to observe
    a spawned child's tool calls -- so a test that needs to see a CHILD's
    own tool result reads it from here, not from ``outcome.steps``.

    Returns:
        A ``(results, on_step)`` pair: ``results`` maps ``tool_name`` to
        its ``AgentStep`` (last-write-wins if a name recurs), populated as
        ``on_step`` is driven by the run.
    """
    results: dict = {}

    def on_step(step, _agent_kind: str) -> None:
        if step.kind == "tool_result" and step.tool_name:
            results[step.tool_name] = step

    return results, on_step


def _registry_with_mutating_builtins(
    gate: BuiltinToolGate, *names: str
) -> tuple[ToolCatalogRegistry, BuiltinToolProvider]:
    """A real ``BuiltinToolProvider`` sharing ``gate``, plus one mutating tool
    per name in ``names``.

    ``BuiltinToolProvider.__init__`` only ever wraps ``CalculatorTool``/
    ``DateTimeTool`` -- there is no constructor seam for extra tools -- so
    this reaches into its ``_tools`` dict directly after construction, the
    same precedent ``Tests/Agents/test_tool_catalog.py`` already uses
    (``provider._tools["calculator"]``). ``list_catalog()``/``tool_for()``/
    ``invoke()`` all read that same dict, so an injected tool is visible to
    every path a registered provider needs to be.
    """
    provider = BuiltinToolProvider(gate=gate)
    for name in names:
        provider._tools[name] = _MutatingBuiltinTool(name)
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    return registry, provider


def test_child_spawned_mid_turn_resolves_a_mutating_builtin_tool_and_the_parents_own_stamp_survives(
    db,
):
    """AC#2 (task-628): a parent batch = [spawn_subagent, write_thing]. The
    parent's own review round trip approves ``write_thing`` for this turn.
    ``spawn_subagent`` dispatches first and runs the child's ENTIRE loop
    INLINE, mid-parent-dispatch (same mechanism the MCP C1 test above
    exercises for ``MCPToolProvider``); the child calls a DIFFERENT
    mutating tool, ``write_other_thing``, gated by the SAME shared
    ``BuiltinToolGate`` instance.

    Two things must both hold:

    1. The child's own call resolves via its own (separate) approval round
       trip and SUCCEEDS -- proving a nested run is not automatically
       fail-closed merely for being nested; the shared review hook and
       gate serve it exactly as they would a top-level run.
    2. After the child returns, the PARENT's own remaining call to
       ``write_thing`` -- approved BEFORE the child ever ran -- must still
       succeed. This is the actual regression: the review hook's first act
       is ``builtin_gate.begin_turn()``, which unconditionally clears the
       WHOLE ``_stamps`` dict (not just the child's own key). Without
       ``stamp_scope`` wrapping the nested run, the child's own
       ``begin_turn()`` wipes the parent's ``write_thing`` stamp before the
       parent's own dispatch loop ever gets to consume it, and
       ``BuiltinToolGate.check()`` has no per-call fallback to fall back
       to -- the parent's already-approved call would fail closed for no
       reason a user could see.
    """
    gate = BuiltinToolGate(service=None)  # Constraint 7: still gates.
    registry, builtin_provider = _registry_with_mutating_builtins(
        gate, "write_thing", "write_other_thing"
    )

    approval_calls: list[list[str]] = []

    def request_approvals(pending):
        approval_calls.append([p.llm_name for p in pending])
        # Both are approved -- the point of this test is NOT that a denial
        # propagates (test_child_run_with_no_approval_route_still_fails_
        # closed... below covers fail-closed), it is that the PARENT's
        # own approval, granted before the child ever ran, is still the
        # one honored for the parent's own remaining call afterward.
        return {p.llm_name: "approve_once" for p in pending}

    review_hook = build_tool_review_hook(gate, builtin_provider, None, request_approvals)

    spawn_call = ToolCall(
        name=SPAWN_TOOL_NAME, args={"task": "write the other thing"}, call_id="p-spawn"
    )
    parent_call = ToolCall(name="write_thing", args={"q": "parent"}, call_id="p-write")
    child_call = ToolCall(
        name="write_other_thing", args={"q": "child"}, call_id="c-write"
    )

    chat = ScriptedChat(
        parent_replies=[_native_turn([spawn_call, parent_call]), "parent done"],
        child_replies=[_native_turn([child_call]), "child done"],
    )

    results, on_step = _tool_result_collector()
    agent_service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=review_hook,
        review_state_scope=gate.stamp_scope,
        on_step=on_step,
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("write_thing", "write_other_thing", SPAWN_TOOL_NAME),
        native_tools=True,
    )

    _run_id, outcome = agent_service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="openai",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    # One round trip for the parent's own batch, one for the child's --
    # nested approval genuinely round-trips, it never deadlocks.
    assert len(approval_calls) == 2
    assert approval_calls[0] == ["write_thing"]
    assert approval_calls[1] == ["write_other_thing"]

    assert {"write_thing", "write_other_thing"} <= set(results)

    # (1) The child's own call, gated by the SAME shared instance, resolved
    # via its own approval round trip -- not fail-closed for being nested.
    assert not results["write_other_thing"].result.startswith("ERROR")

    # (2) The actual regression: the parent's own pre-child approval
    # survived the child's nested begin_turn() wipe and still resolves the
    # parent's own remaining call.
    assert not results["write_thing"].result.startswith("ERROR")


def test_child_spawned_mid_turn_without_stamp_scope_clobbers_the_parents_approval(
    db,
):
    """Same interleave as the test above, with ``review_state_scope`` left
    unwired -- proving the previous test is not passing trivially. Without
    the scope the child's ``begin_turn()`` clears the WHOLE ``_stamps``
    dict and is never restored, so the parent's own pre-child
    ``write_thing`` approval is gone by the time the parent's own dispatch
    loop reaches it -- it fails closed, even though a human already
    approved it this very turn."""
    gate = BuiltinToolGate(service=None)
    registry, builtin_provider = _registry_with_mutating_builtins(
        gate, "write_thing", "write_other_thing"
    )

    def request_approvals(pending):
        return {p.llm_name: "approve_once" for p in pending}

    review_hook = build_tool_review_hook(gate, builtin_provider, None, request_approvals)

    spawn_call = ToolCall(
        name=SPAWN_TOOL_NAME, args={"task": "write the other thing"}, call_id="p-spawn"
    )
    parent_call = ToolCall(name="write_thing", args={"q": "parent"}, call_id="p-write")
    child_call = ToolCall(
        name="write_other_thing", args={"q": "child"}, call_id="c-write"
    )

    chat = ScriptedChat(
        parent_replies=[_native_turn([spawn_call, parent_call]), "parent done"],
        child_replies=[_native_turn([child_call]), "child done"],
    )

    results, on_step = _tool_result_collector()
    agent_service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=review_hook,
        on_step=on_step,
        # No review_state_scope -- the pre-task-628 wiring.
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("write_thing", "write_other_thing", SPAWN_TOOL_NAME),
        native_tools=True,
    )

    _run_id, outcome = agent_service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="openai",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    # The child's own call is unaffected either way.
    assert not results["write_other_thing"].result.startswith("ERROR")
    # This is the clobber: the parent's own pre-child approval was wiped by
    # the child's begin_turn() and never restored, so its remaining call
    # fails closed despite having been approved this turn.
    assert results["write_thing"].result.startswith("ERROR")
    assert "approval" in results["write_thing"].result.lower()


def test_child_run_with_no_approval_route_still_fails_closed_for_a_mutating_builtin_tool(
    db,
):
    """AC#4 (task-628): this task fixes ROUTING for a run that has an
    approval path; it must not weaken ``BuiltinToolGate``'s own fail-closed
    guarantee for a run that has none at all (e.g. headless -- no
    ``review_tool_calls`` hook wired for anyone to ask). ``check()`` has no
    per-call approval fallback the way MCP's ``invoke()`` does, so an
    "ask" tool with no stamp and no session approval must refuse --
    spawned child or not, and regardless of whether a ``review_state_
    scope`` happens to be wired."""
    gate = BuiltinToolGate(service=None)
    registry, _builtin_provider = _registry_with_mutating_builtins(gate, "write_thing")

    spawn_call = ToolCall(
        name=SPAWN_TOOL_NAME, args={"task": "write the thing"}, call_id="p-spawn"
    )
    child_call = ToolCall(name="write_thing", args={"q": "child"}, call_id="c-write")

    chat = ScriptedChat(
        parent_replies=[_native_turn([spawn_call]), "parent done"],
        child_replies=[_native_turn([child_call]), "child done"],
    )

    results, on_step = _tool_result_collector()
    agent_service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        # No review_tool_calls at all: `deps.review_tool_calls is None`
        # short-circuits to an empty verdicts map (agent_runtime.py), so
        # every call -- including the child's gated one -- dispatches
        # straight to BuiltinToolProvider.invoke()'s own defense-in-depth
        # gate.check() with no batched approval card ever offered.
        review_state_scope=gate.stamp_scope,
        on_step=on_step,
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("write_thing", SPAWN_TOOL_NAME),
        native_tools=True,
    )

    _run_id, outcome = agent_service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="openai",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert "write_thing" in results
    assert results["write_thing"].result.startswith("ERROR")
    assert "approval" in results["write_thing"].result.lower()
