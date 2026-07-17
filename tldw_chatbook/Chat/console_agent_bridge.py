# tldw_chatbook/Chat/console_agent_bridge.py
"""Impure Console glue between the synchronous agent engine and the store.

Builds the AgentConfig, drives a streaming model adapter (StreamGate +
provider_gateway.stream_chat), appends TOOL markers for the primary run's
tool/spawn steps, keeps an in-memory live snapshot for the rail poll, and
runs AgentService.run_turn synchronously (the controller wraps it in
asyncio.to_thread). No widget mutation.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY, AGENT_KIND_SUBAGENT, FIND_TOOLS_NAME, LOAD_TOOLS_NAME,
    RunBudget, RUNTIME_TOOL_NAMES, SPAWN_TOOL_NAME, STEP_ERROR, STEP_SPAWN,
    STEP_TOOL_RESULT, AgentConfig, AgentStep, RunOutcome, ToolCall, ToolCatalogEntry,
    ToolResult, ToolSchema,
)
from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT, AgentService
from tldw_chatbook.Agents.agent_stream import StreamGate
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider, SkillToolProvider, ToolCatalogRegistry,
    intersect_skill_tools,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_provider_gateway import ProviderToolCalls
from tldw_chatbook.Chat.console_skill_resolver import SKILL_UNTRUSTED_REFUSE
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError

CONSOLE_AGENT_OPERATING_PROMPT = (
    "You are a capable assistant with optional tools. Answer directly when no "
    "tool is needed. When a tool would help, call exactly one tool per reply "
    "using the fenced protocol described below, then continue once you have the "
    "result. Use spawn_subagent to delegate a self-contained sub-task to an "
    "isolated helper. Keep replies concise.")

# Skills Phase-2 gate finding 1 (Task-14 report, scenario 5: "Find a skill
# that can shout, load it, and use it on: hello"): a discovery-heavy run --
# find_tools -> load_tools -> a tool/skill call -> the final wrap-up reply --
# needs, at the floor, 3 tool rounds + 1 wrap-up = 4 model turns / 10 steps
# (3 steps per tool round: STEP_MODEL + STEP_TOOL_CALL + STEP_TOOL_RESULT,
# times 3 rounds, plus 1 final STEP_MODEL with no tool call -- see
# agent_runtime.run_agent_loop). That 10-step floor already sat ABOVE the
# engine's own pure step default (agent_models.RunBudget.max_steps == 8),
# so any >DIRECT_DISCLOSE_THRESHOLD skill catalog -- which forces the
# find/load path -- used to exhaust the bare step default right after the
# skill's successful tool_result, one step short of the wrap-up reply: the
# run persisted `stuck` even though every tool call already succeeded
# (live-gate confirmed, pre-task-244).
#
# task-244 adds a model-turn budget tier (agent_models.RunBudget.
# max_model_turns) and makes IT, not the raw step count, this run's PRIMARY
# limiter. Two additional real tool rounds beyond the 4-turn/10-step floor
# cost 2 more turns / 6 more steps (6 turns / 16 steps total); under
# max_model_turns=8 the floor plus those two extra rounds fits with 2 turns
# to spare. max_steps=32 is a backstop, not a derived worst case: fence
# turns cost at most 3 steps (one tool call per reply), but a NATIVE
# multi-call batch (task-243) costs 1 + 2N steps per turn, so a run of
# heavy parallel batches can legitimately hit the step backstop before the
# 8-turn cap — that is the backstop doing its job. Either way it can no
# longer starve a discovery run one step short of its wrap-up the way the
# bare step default once did.
# max_wall_seconds stays at the prior 480s (25-50s/turn x up to 8 model
# turns at the slow local-model pace this gate exercises). The engine's own
# RunBudget defaults (agent_models.RunBudget) are left UNCHANGED -- this
# override applies only at the Console bridge's own config-assembly site
# (run_reply below); other callers of RunBudget()/AgentConfig keep the bare
# engine default.
CONSOLE_RUN_BUDGET = RunBudget(
    max_steps=32, max_wall_seconds=480.0, max_model_turns=8)

_QUIET_STEP_TOOLS = {FIND_TOOLS_NAME, LOAD_TOOLS_NAME}


def compose_agent_system_prompt(session_prompt: str) -> str:
    """Compose the primary system prompt: session prompt first, agent prompt appended.

    Args:
        session_prompt: The Console session's own system prompt, if any.

    Returns:
        ``session_prompt`` followed by ``CONSOLE_AGENT_OPERATING_PROMPT``
        (blank-line separated), or just the operating prompt when
        ``session_prompt`` is blank.
    """
    base = (session_prompt or "").strip()
    if not base:
        return CONSOLE_AGENT_OPERATING_PROMPT
    return f"{session_prompt}\n\n{CONSOLE_AGENT_OPERATING_PROMPT}"


def format_agent_step_marker(
    kind: str,
    *,
    tool_name: str | None = None,
    result: Any = None,
    summary: str | None = None,
) -> str | None:
    """Return the transcript TOOL marker text for one primary-agent step.

    Shared by the live bridge (``ConsoleAgentBridge.run_reply``'s ``on_step``
    hook, called per in-flight ``AgentStep``) and resume re-derivation
    (``ConsoleAgentBridge.resume_marker_messages``, called per persisted
    ``AgentRunsDB`` step dict), so a resumed transcript's markers render
    byte-identical to what the live run produced (Plan-B final-review
    Medium-1). Returns ``None`` for step kinds that never produce a
    transcript marker: ``STEP_MODEL`` and the quiet tool-catalog steps
    (``find_tools``/``load_tools``, see ``_QUIET_STEP_TOOLS``).

    Kept raw (no escaping): both consumers render markup-off --
    ``console_transcript.py``'s ``_message_render_text`` builds a
    ``Content`` via ``Content.assemble`` (never markup-parsed) and
    ``chat_screen.py``'s legacy fallback wraps the string in a bare
    ``rich.text.Text(...)`` (also never markup-parsed). Escaping here for a
    parser that never runs would leave literal backslashes in the rendered
    marker (``fetch [docs]`` -> ``fetch \\[docs]``).
    """
    if kind == STEP_SPAWN:
        return f"⤷ spawned sub-agent: {summary}"
    if kind == STEP_TOOL_RESULT and tool_name not in _QUIET_STEP_TOOLS:
        return f"⚙ {tool_name} → {result}"
    if kind == STEP_ERROR:
        return f"⚠ {summary}"
    return None


def inject_resume_agent_markers(
    messages: list[ConsoleChatMessage],
    marker_blocks: list[list[ConsoleChatMessage]],
) -> list[ConsoleChatMessage]:
    """Interleave AgentRunsDB-derived TOOL marker blocks into a resumed transcript.

    Placement (Plan-B final-review Medium-1): each run's marker block is
    matched ordinally to the Nth ASSISTANT message in ``messages`` --
    oldest run <-> oldest assistant reply -- so in the common case (every
    assistant reply in the conversation came from the agent path) each
    run's markers land directly after the answer they belong to, exactly
    mirroring where they rendered live. This is the "simplest correct"
    placement given persisted messages carry no per-step timestamp to
    interleave by more precisely. A run left over with no corresponding
    assistant message -- only possible when ``agent_runtime`` was toggled
    off mid-conversation after some replies already used the agent path --
    has its block appended at the end of the transcript instead of being
    silently dropped.

    Idempotent: a block whose marker texts are already present as TOOL
    messages anywhere in ``messages`` is skipped, so calling this twice (or
    resuming into a transcript that already carries live markers) never
    duplicates a block.

    Args:
        messages: The rebuilt transcript (ChaChaNotes-derived; never
            contains TOOL rows on its own, since markers are appended
            live with ``persist=False``).
        marker_blocks: Per-run marker-message blocks, oldest run first
            (see ``ConsoleAgentBridge.resume_marker_messages``).

    Returns:
        A new list with marker blocks interleaved; ``messages`` itself is
        not mutated.
    """
    non_empty_blocks = [block for block in marker_blocks if block]
    if not non_empty_blocks:
        return list(messages)

    existing_tool_contents = {
        message.content for message in messages
        if message.role is ConsoleMessageRole.TOOL
    }
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.role is ConsoleMessageRole.ASSISTANT
    ]
    matched = dict(zip(assistant_indexes, non_empty_blocks))
    leftover_blocks = non_empty_blocks[len(assistant_indexes):]

    def _already_present(block: list[ConsoleChatMessage]) -> bool:
        return all(marker.content in existing_tool_contents for marker in block)

    result: list[ConsoleChatMessage] = []
    for index, message in enumerate(messages):
        result.append(message)
        block = matched.get(index)
        if block is not None and not _already_present(block):
            result.extend(block)
    for block in leftover_blocks:
        if not _already_present(block):
            result.extend(block)
    return result


@dataclass(frozen=True)
class AgentLiveStep:
    """One in-flight agent step, as rendered on the rail's live poll.

    Attributes:
        kind: The step kind (one of ``agent_models``'s ``STEP_*``
            constants, e.g. ``STEP_TOOL_RESULT``/``STEP_SPAWN``/
            ``STEP_ERROR``).
        text: Rendered summary text for this step (see
            ``ConsoleAgentBridge._summarize``); already truncated and left
            raw/unescaped for the rail's markup-off ``Static``.
        agent_kind: Which agent produced the step -- ``AGENT_KIND_PRIMARY``
            or ``AGENT_KIND_SUBAGENT``.
    """

    kind: str
    text: str
    agent_kind: str


@dataclass(frozen=True)
class SubAgentSummary:
    """A spawned sub-agent's rail summary, as of the last observed step.

    Attributes:
        text: Rendered summary of the sub-agent's task (live) or its
            recorded ``task`` (historical, resume-derived).
        status: The sub-agent run's status -- ``"running"`` while the
            primary's step log has not yet recorded its outcome.
    """

    text: str
    status: str = "running"


@dataclass(frozen=True)
class AgentLiveSnapshot:
    """Rail-facing snapshot of one conversation's primary agent run.

    Returned by both ``ConsoleAgentBridge.live_snapshot`` (this process's
    own in-flight/just-finished run) and ``historical_snapshot`` (re-derived
    from ``AgentRunsDB`` after a restart) -- callers read whichever is not
    idle, so both must expose the same shape.

    Attributes:
        status: Run status -- ``"idle"``, ``"running"``, or a terminal
            ``RunOutcome.status`` value (``"done"``/``"error"``/
            ``"cancelled"``/``"stuck"``).
        step: Total number of steps observed so far for this run.
        steps: The most recent steps (bounded to the last 5), oldest first.
        subagents: Summaries of this run's spawned sub-agents, in the order
            they were spawned/recorded.
    """

    status: str = "idle"
    step: int = 0
    steps: tuple[AgentLiveStep, ...] = ()
    subagents: tuple[SubAgentSummary, ...] = ()


class _StreamingModelAdapter:
    """chat_call-compatible adapter that streams every PRIMARY turn live.

    AgentService calls it as ``chat_call(api_endpoint=…, messages_payload=…,
    streaming=False, model=…)`` and expects a
    ``{"choices":[{"message":{"content": <full text>}}]}`` response. Sub-agent
    turns (leading system content == SUBAGENT_SYSTEM_PROMPT) are streamed to a
    throwaway gate and never touch the transcript.

    Every non-sealed primary turn streams live to the store as it arrives —
    not just the final answer — since the gate cannot know in advance
    whether a given turn will end up being a tool call or the final answer:
    a well-behaved fence-first tool call never streams anything (the gate
    seals from the first token), but a disobedient turn that emits prose
    before a mid-stream fence has already forwarded that prose to the store
    by the time the completed turn is classified as a tool call. When that
    happens, this adapter resets the message's streamed content back to
    empty once the turn is confirmed to carry a tool call (see
    ``ConsoleChatStore.reset_stream_content``), so the leaked prose — already
    preserved in that turn's ``STEP_MODEL`` step log — does not survive to
    garble the next turn's chunks on the same message (Plan-B Task 5
    Finding A).

    ``should_cancel`` is polled once per received chunk, AFTER it has been fed
    to the gate and (for the primary) flushed to the store — never before.
    Checking before-feed would let a single-chunk turn's content vanish
    entirely (e.g. a whole leading tool-call fence dropped mid-flight),
    which the pure loop would then misreport as a normal empty "done" turn
    rather than "cancelled", since it only re-polls cancellation at its own
    step/tool-call boundaries. Checking after-feed guarantees every received
    chunk is always accounted for, while still stopping the stream promptly
    (no more chunks are pulled once cancellation is observed) and letting the
    loop's own boundary check catch the cancellation on the next poll.

    ``chat_call`` bridges into async gateway code via the single event
    ``loop`` passed in at construction — created once by
    ``ConsoleAgentBridge.run_reply`` and reused for every turn of that run
    (the tool-call turn(s), any sub-agent turns, and the final-answer turn),
    rather than a fresh ``asyncio.run()`` per turn (PR #629 Fix 1(c)). A
    fresh loop per turn meant a fresh loop identity on every single
    ``chat_call``, which forced the gateway's owned ``httpx.AsyncClient`` to
    swap (see ``ConsoleProviderGateway._active_http_client``) once per turn
    instead of at most once per run.
    """

    def __init__(self, *, store, provider_gateway, resolution, assistant_message_id,
                 should_cancel, loop):
        self._store = store
        self._gateway = provider_gateway
        self._resolution = resolution
        self._assistant_message_id = assistant_message_id
        self._should_cancel = should_cancel
        self._loop = loop

    def chat_call(self, *, messages_payload, model=None, api_endpoint=None,
                  streaming=False, tools=None, **_ignored) -> dict:
        is_subagent = self._is_subagent(messages_payload)
        gate = StreamGate()
        any_streamed = False
        native_calls: list[dict] = []

        async def _consume() -> None:
            nonlocal any_streamed
            # Forwarding `tools=` only when it is non-None (rather than
            # always passing the keyword, even as None) keeps every
            # pre-Task-5 gateway fake elsewhere in the test suite — whose
            # `stream_chat(resolution, messages)` signature predates this
            # parameter and has no matching call-site to update under this
            # task's own touched-files constraint — working unchanged for
            # the (still overwhelmingly common) fence path, where `tools`
            # is always None. The real gateway and any fake built against
            # this task's own `tools=None` contract see identical behavior
            # either way, since the callee-side default is also None.
            stream_kwargs = {"tools": tools} if tools is not None else {}
            async for chunk in self._gateway.stream_chat(
                    self._resolution, messages_payload, **stream_kwargs):
                if isinstance(chunk, ProviderToolCalls):
                    # Plan-B contract: structured deltas never hit the
                    # transcript — captured here, surfaced only through the
                    # returned message dict's `tool_calls`.
                    native_calls.extend(chunk.tool_calls)
                    continue
                visible = gate.feed(chunk)
                if visible and not is_subagent:
                    self._store.append_stream_chunk(self._assistant_message_id, visible)
                    any_streamed = True
                if self._should_cancel():
                    break
            tail = gate.flush_tail()
            if tail and not is_subagent:
                self._store.append_stream_chunk(self._assistant_message_id, tail)
                any_streamed = True

        # The service runs on a worker thread with no running loop of its
        # own, so `run_until_complete` on this run's shared loop is safe
        # here (the loop is never touched concurrently — every chat_call
        # for this run_reply happens synchronously, one at a time, on this
        # same thread; see ConsoleAgentBridge.run_reply).
        self._loop.run_until_complete(_consume())
        if any_streamed and not is_subagent:
            # Finding A: this turn leaked prose to the store before it was
            # known to be a tool call (a well-behaved fence-first tool call
            # never streams anything, so any_streamed stays False and this
            # never fires for the common case). Now that the full buffer is
            # in and the authoritative parse is available, discard that
            # leaked prose so it doesn't survive to garble the next turn's
            # chunks on the same message. Extended for native tool-calls
            # (Task 5): a native turn that streamed leaked prose before its
            # ProviderToolCalls sentinel arrived must be reset the same way.
            _visible, tool_call = gate.result()
            if tool_call is not None or native_calls:
                self._store.reset_stream_content(self._assistant_message_id)
        message: dict = {"content": gate.full_text}
        if native_calls:
            message["tool_calls"] = native_calls
        return {"choices": [{"message": message}]}

    @staticmethod
    def _is_subagent(messages_payload) -> bool:
        if not messages_payload:
            return False
        first = messages_payload[0]
        return (first.get("role") == "system"
                and str(first.get("content", "")).startswith(SUBAGENT_SYSTEM_PROMPT))


def _eligible_skill_entries(context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Trusted, model-invocable skill summaries from a ``get_context`` snapshot.

    Mirrors ``ChatScreen._console_skill_trusted_candidates_from_context``'s
    defensive filter shape, but scoped to model-invocation eligibility
    rather than user (``/skill-name``) invocation: a skill is eligible here
    when it is not ``trust_blocked`` (the local-skill-trust-integrity gate)
    and does not opt out of model calls via ``disable_model_invocation``
    (the skill author's own front-matter flag). Both fields default to
    "eligible" when absent so a caller-supplied fake ``get_context`` (as in
    tests) does not need to set every field.

    Args:
        context: A ``get_context(mode="local")`` payload (or a plain dict
            shaped like one); anything else yields no entries.

    Returns:
        The raw ``available_skills`` entries (plain mappings, unmodified)
        that pass the eligibility filter, in the order ``get_context``
        returned them.
    """
    available = context.get("available_skills") if isinstance(context, Mapping) else None
    return [
        item for item in (available or [])
        if isinstance(item, Mapping)
        and item.get("name")
        and not item.get("trust_blocked", False)
        and not item.get("disable_model_invocation", False)
    ]


def _non_colliding_skill_entries(
    context: Mapping[str, Any], builtin_names: tuple[str, ...],
) -> list[Mapping[str, Any]]:
    """Eligible skill entries, excluding any name that collides with a
    builtin OR one of the loop's own in-loop runtime tool names.

    Shadowing (Task 11 review note 2 + this task's own allow-list
    ordering): a builtin tool name must always win over a same-named
    skill -- for BOTH the registry's own first-match resolution (builtins
    registered before skills, so ``resolve_name``/``_owner_and_id`` find
    the builtin first) AND the actual invocation dispatch in
    ``AgentService.invoke_tool`` (which checks
    ``skill_runner.is_skill_tool(name)`` BEFORE falling back to the
    registry -- a check the registry's own registration order can't
    influence). Excluding the collision here, at composition time, keeps
    both paths in agreement: a skill literally named e.g. ``"calculator"``
    is simply never treated as a distinct, skill-routable tool, and the
    real builtin still works exactly as before.

    Qodo finding 4 (PR #636 bot review): the same reasoning applies to
    ``RUNTIME_TOOL_NAMES`` (``spawn_subagent``/``find_tools``/
    ``load_tools``) -- these are dispatched by a direct name comparison
    inside ``agent_runtime.run_agent_loop`` itself, BEFORE the loop ever
    reaches the registry or ``skill_runner``. A skill front-matter'd with
    one of those names would previously still be advertised in the run's
    catalog/allow-list (a distinct, misleadingly-schema'd entry), yet could
    never actually be invoked -- the loop's own name-based dispatch always
    wins that comparison first. Excluding these names too means such a
    skill is simply never registered as a catalog entry at all, matching
    what would happen at invocation time anyway.
    """
    collision_names = set(builtin_names) | RUNTIME_TOOL_NAMES
    return [
        item for item in _eligible_skill_entries(context)
        if str(item["name"]) not in collision_names
    ]


def _compose_run_allowed_tools(
    context: Mapping[str, Any], builtin_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Pure per-run allow-list: builtins, then eligible skill names, then spawn.

    Kept as its own tiny pure helper (no registry construction) so the
    allow-list composition itself -- the part that actually gates what a
    run may call -- is directly unit-testable without also standing up a
    ``ToolCatalogRegistry``/``SkillToolProvider``.

    Args:
        context: A ``get_context(mode="local")`` payload.
        builtin_names: The run's builtin tool names, in registry order.

    Returns:
        ``builtin_names + eligible non-colliding skill names +
        (SPAWN_TOOL_NAME,)``.
    """
    skill_names = tuple(
        str(item["name"])
        for item in _non_colliding_skill_entries(context, builtin_names))
    return tuple(builtin_names) + skill_names + (SPAWN_TOOL_NAME,)


class _CollisionFilteredMCPProvider:
    """View over a composed MCP provider that hides collision-excluded entries.

    Registered into the run's registry INSTEAD OF the raw provider, so
    ``ToolCatalogRegistry.list_catalog()`` (a raw sweep across every
    registered provider) never surfaces a shadowed MCP entry either --
    mirrors ``_non_colliding_skill_entries``'s own raw-catalog exclusion
    invariant (a colliding skill is filtered out before ``SkillToolProvider``
    is even constructed, so its catalog never contains it). MCP tools are
    already fully composed by the time this run sees them (T3: composed
    once, by the caller, on the main loop), so this wraps rather than
    reconstructs -- a read-time filter only. The underlying provider's own
    state (its composed catalog, per-turn decision stamps) is never
    touched, and ``invoke``/``load_schema`` are simple pass-throughs: a
    filtered-out tool id can never reach them anyway, since the registry's
    own owner/name caches are built exclusively from this wrapper's
    (already-filtered) ``list_catalog()``.
    """

    def __init__(self, provider: Any, allowed_names: frozenset[str]) -> None:
        self._provider = provider
        self._allowed_names = allowed_names

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            entry for entry in self._provider.list_catalog()
            if entry.name in self._allowed_names
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        return self._provider.load_schema(tool_id)

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        return self._provider.invoke(tool_id, args)


def _non_colliding_mcp_names(
    mcp_provider: Any, collision_names: frozenset[str] | set[str],
) -> tuple[str, ...]:
    """Eligible MCP tool names, excluding any collision with a builtin, a
    runtime tool, or an already-included skill name.

    Mirrors ``_non_colliding_skill_entries``'s shadowing precedent (Task 11
    review note 2 / Qodo finding 4, PR #636 -- see that function's own
    docstring): MCP is the LAST provider registered for a run (see
    ``_compose_run_registry_and_allowed``), so excluding a collision here
    at composition time keeps this the single place a cross-provider name
    conflict is resolved. A colliding MCP tool is simply never advertised
    in the run's allow-list/registry -- the underlying
    ``MCPToolProvider``'s own internal catalog (already deduplicated
    within itself by T1's ``dedupe_names``) is left untouched; only what
    THIS run offers the model is filtered.

    Args:
        mcp_provider: A composed ``MCPToolProvider`` (or any
            ``ToolProvider``-shaped double in tests) whose ``list_catalog()``
            has already been built (T3: composed once, on the main loop,
            before this run's worker thread starts).
        collision_names: Names that must never be treated as a distinct
            MCP tool -- builtins, ``RUNTIME_TOOL_NAMES``, and this run's
            own eligible skill names.

    Returns:
        The subset of ``mcp_provider.list_catalog()`` names not present in
        ``collision_names``, in catalog order.
    """
    return tuple(
        entry.name for entry in mcp_provider.list_catalog()
        if entry.name not in collision_names
    )


def _compose_run_registry_and_allowed(
    context: Mapping[str, Any],
    *,
    mcp_provider: Any | None = None,
) -> tuple[ToolCatalogRegistry, tuple[str, ...], tuple[str, ...]]:
    """Build a fresh per-run tool registry + allow-list from a skills snapshot.

    Called once per ``run_reply`` invocation (never cached across runs --
    the per-run freshness doctrine: a skill approved/edited/revoked since
    the last run must take effect on the very next one). Registers
    ``BuiltinToolProvider`` first, then (only when there is at least one
    non-colliding eligible entry) a ``SkillToolProvider`` snapshot, then
    (P5-T6, only when there is at least one non-colliding eligible entry)
    an already-composed MCP provider -- shadowing order: builtins beat
    skills beat MCP, matching the allow-list's own
    ``builtins ∪ skills ∪ mcp`` ordering.

    Args:
        context: A fresh ``get_context(mode="local")`` payload.
        mcp_provider: This run's already-composed MCP tool provider (see
            ``MCPToolProvider.compose_catalog`` -- composed by the caller
            on the main loop BEFORE this function runs), or ``None`` when
            no MCP tools should be offered this run (no service, kill
            switch on, or composition yielded nothing).

    Returns:
        ``(registry, allowed_tools, builtin_names)`` -- the per-run
        registry, its full allow-list (builtins + eligible skills +
        eligible MCP tools + spawn), and just the builtin names (needed
        separately by ``_BridgeSkillRunner`` to intersect a skill's own
        declared ``allowed_tools`` against -- never against skill names,
        so a skill's sub-agent can never call another skill).
    """
    registry = ToolCatalogRegistry()
    builtin_provider = BuiltinToolProvider()
    registry.register_provider(builtin_provider)
    builtin_names = tuple(entry.name for entry in builtin_provider.list_catalog())
    eligible = _non_colliding_skill_entries(context, builtin_names)
    if eligible:
        registry.register_provider(SkillToolProvider(eligible))
    skill_names = tuple(str(item["name"]) for item in eligible)
    allowed_tools = tuple(builtin_names) + skill_names
    if mcp_provider is not None:
        collision_names = set(builtin_names) | set(skill_names) | RUNTIME_TOOL_NAMES
        mcp_names = _non_colliding_mcp_names(mcp_provider, collision_names)
        if mcp_names:
            registry.register_provider(
                _CollisionFilteredMCPProvider(mcp_provider, frozenset(mcp_names)))
            allowed_tools += mcp_names
    allowed_tools += (SPAWN_TOOL_NAME,)
    return registry, allowed_tools, builtin_names


class _BridgeSkillRunner:
    """``SkillRunner``: renders a skill, then routes it through THIS run's spawn.

    Built fresh per ``run_reply`` invocation from that run's own eligible
    skill-name set and builtin names (see ``_compose_run_registry_and_allowed``).
    ``run`` re-verifies trust at render time via ``execute_skill`` -- never
    a cached snapshot -- so a skill approved when the catalog was built but
    revoked before the model actually calls it still refuses (mirrors
    ``ConsoleChatController._apply_skill_substitution``'s own re-verification
    discipline for the ``/skill-name`` user-invocation path).
    """

    def __init__(self, *, skills_service: Any, skill_names: frozenset[str],
                 builtin_names: tuple[str, ...]) -> None:
        self._skills_service = skills_service
        self._skill_names = skill_names
        self._builtin_names = builtin_names

    def is_skill_tool(self, name: str) -> bool:
        return name in self._skill_names

    def run(self, name: str, args: str,
            spawn: Callable[..., ToolResult]) -> ToolResult:
        try:
            result = asyncio.run(
                self._skills_service.execute_skill(name, mode="local", args=args))
        except SkillTrustBlockedError as exc:
            return ToolResult(
                ok=False,
                error=SKILL_UNTRUSTED_REFUSE.format(name=name, reason=exc.reason_code))
        rendered = result.get("rendered_prompt", "") if isinstance(result, Mapping) else ""
        declared_allowed_tools = (
            result.get("allowed_tools") if isinstance(result, Mapping) else None)
        allowed_tools = intersect_skill_tools(declared_allowed_tools, self._builtin_names)
        return spawn(rendered, allowed_tools=allowed_tools)


class ConsoleAgentBridge:
    """Owns the tool registry + run store and runs one primary agent reply."""

    def __init__(self, *, agent_runs_db: AgentRunsDB, store,
                 provider_gateway, registry: ToolCatalogRegistry | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 skills_service: Any | None = None,
                 native_tools_enabled: Callable[[], bool] | None = None) -> None:
        self._db = agent_runs_db
        self._store = store
        self._gateway = provider_gateway
        self._clock = clock
        self._skills_service = skills_service
        self._native_tools_enabled = native_tools_enabled
        if registry is None:
            registry = ToolCatalogRegistry()
            registry.register_provider(BuiltinToolProvider())
        self._registry = registry
        self._allowed_tools = tuple(
            e.name for e in registry.list_catalog()) + (SPAWN_TOOL_NAME,)
        self._live: dict[str, AgentLiveSnapshot] = {}
        self._historical_cache: dict[str, AgentLiveSnapshot] = {}

    # -- run ------------------------------------------------------------

    def run_reply(self, *, conversation_id: str, session_id: str, resolution: Any,
                  assistant_message_id: str, model: str, session_system_prompt: str,
                  agent_messages: list[dict], should_cancel: Callable[[], bool],
                  supersede_previous: bool = False,
                  mcp_provider: Any | None = None,
                  review_tool_calls: Callable[[list[ToolCall]], dict[str, str]] | None = None,
                  ) -> RunOutcome:
        # Per-run tool registry + allow-list (Task 12, extended by P5-T6 for
        # MCP): rebuilt FRESH for this run whenever there is a skills
        # service OR an already-composed MCP provider for this run (never
        # cached across runs, and never the shared self._registry/
        # self._allowed_tools built at construction) -- so a skill or MCP
        # tool approved/edited/revoked since the last run always takes
        # effect on the very next one. `mcp_provider` is built and
        # composed by the CALLER (ConsoleChatController._compose_mcp_
        # provider, on the running Textual main loop, BEFORE this method
        # is dispatched onto asyncio.to_thread) -- see MCPToolProvider's
        # own module docstring for why `compose_catalog()`'s async I/O can
        # never run from inside this worker-thread method. Neither a
        # skills service nor an MCP provider: the shipped shared
        # registry/allow-list is used unchanged -- the no-skills, no-MCP
        # path stays byte-identical to before this task.
        registry = self._registry
        allowed_tools = self._allowed_tools
        skill_runner = None
        if self._skills_service is not None or mcp_provider is not None:
            context: Mapping[str, Any] = {}
            if self._skills_service is not None:
                context = asyncio.run(self._skills_service.get_context(mode="local"))
            registry, allowed_tools, builtin_names = (
                _compose_run_registry_and_allowed(context, mcp_provider=mcp_provider))
            if self._skills_service is not None:
                skill_names = frozenset(
                    str(item["name"])
                    for item in _non_colliding_skill_entries(context, builtin_names))
                skill_runner = _BridgeSkillRunner(
                    skills_service=self._skills_service, skill_names=skill_names,
                    builtin_names=builtin_names)
        # [console] native_tool_calls kill-switch (Task 5): a caller-supplied
        # predicate (chat_screen.py's _console_native_tool_calls_enabled)
        # gates whether this run may use native provider tool-calls at all;
        # no predicate (fakes/tests that never pass one) defaults to
        # always-on, matching the pre-kill-switch behavior.
        native_tools = (True if self._native_tools_enabled is None
                        else bool(self._native_tools_enabled()))
        config = AgentConfig(
            model=model,
            system_prompt=compose_agent_system_prompt(session_system_prompt),
            allowed_tools=allowed_tools,
            budget=CONSOLE_RUN_BUDGET,
            native_tools=native_tools)
        # One event loop for the whole run (PR #629 Fix 1(c)): every turn
        # this run makes -- primary tool-call turns, any sub-agent turns,
        # and the final-answer turn -- bridges through this same loop via
        # `_StreamingModelAdapter.chat_call`'s `run_until_complete`, instead
        # of each turn spinning up (and tearing down) its own loop via
        # `asyncio.run()`. That per-turn churn forced a client swap on the
        # gateway's owned httpx client every single turn (see
        # `ConsoleProviderGateway._active_http_client`); reusing one loop
        # for the whole run means at most one swap per run.
        run_loop = asyncio.new_event_loop()
        adapter = _StreamingModelAdapter(
            store=self._store, provider_gateway=self._gateway, resolution=resolution,
            assistant_message_id=assistant_message_id, should_cancel=should_cancel,
            loop=run_loop)

        live_steps: list[AgentLiveStep] = []
        subagents: list[SubAgentSummary] = []
        self._live[conversation_id] = AgentLiveSnapshot(status="running")
        # A live run is starting -- live_snapshot takes over as the rail's
        # source of truth for this conversation from here on, so any
        # previously cached historical (DB-derived) summary is stale.
        self._historical_cache.pop(conversation_id, None)

        def on_step(step: AgentStep, agent_kind: str) -> None:
            live_steps.append(AgentLiveStep(step.kind, self._summarize(step), agent_kind))
            if agent_kind == AGENT_KIND_PRIMARY:
                if step.kind == STEP_SPAWN:
                    subagents.append(SubAgentSummary(step.summary or ""))
                # format_agent_step_marker is the single source of truth for
                # marker text -- shared with resume_marker_messages below --
                # so live and resume-rebuilt transcripts render identically
                # (Plan-B final-review Medium-1). See its docstring for why
                # the text must stay raw/unescaped.
                marker_text = format_agent_step_marker(
                    step.kind, tool_name=step.tool_name, result=step.result,
                    summary=step.summary)
                if marker_text is not None:
                    self._append_marker(session_id, marker_text)
            self._live[conversation_id] = AgentLiveSnapshot(
                status="running", step=len(live_steps),
                steps=tuple(live_steps[-5:]), subagents=tuple(subagents))

        service = AgentService(
            self._db, registry, chat_call=adapter.chat_call,
            clock=self._clock, on_step=on_step, skill_runner=skill_runner,
            review_tool_calls=review_tool_calls)

        supersede_run_id = (
            self._previous_primary_run_id(conversation_id) if supersede_previous else None)
        try:
            _run_id, outcome = service.run_turn(
                conversation_id=conversation_id, messages=agent_messages, config=config,
                # execution_key-first (Task 5): the service's capability
                # check keys off api_endpoint, and execution_key is by
                # definition "Provider key passed to chat_api_call" — the
                # PROVIDER_PARAM_MAP key space provider_supports_native_tools
                # matches against. `provider` (the display key) and then
                # "agent" remain fallbacks for fakes lacking either
                # attribute (e.g. resolution=object() in existing tests),
                # which keeps them on the fence path unchanged.
                api_endpoint=str(
                    getattr(resolution, "execution_key", "")
                    or getattr(resolution, "provider", "") or "agent"),
                should_cancel=should_cancel, supersede_run_id=supersede_run_id)
        finally:
            run_loop.close()
        self._live[conversation_id] = AgentLiveSnapshot(
            status=outcome.status, step=len(live_steps),
            steps=tuple(live_steps[-5:]), subagents=tuple(subagents))
        # The run just finished -- drop any stale historical cache entry so
        # a *later* resume (in a future process) always re-derives fresh
        # rather than reading this run's now-superseded snapshot (belt and
        # braces on top of the pop at run start above).
        self._historical_cache.pop(conversation_id, None)
        return outcome

    # -- rail reads -----------------------------------------------------

    def live_snapshot(self, conversation_id: str) -> AgentLiveSnapshot:
        return self._live.get(conversation_id, AgentLiveSnapshot())

    def historical_snapshot(self, conversation_id: str) -> AgentLiveSnapshot:
        """Rail summary derived from ``AgentRunsDB`` for a conversation this
        bridge instance has never run in-process (Plan-B agent-runtime gate
        Finding 2): after an app restart, ``live_snapshot`` stays ``idle``
        forever for a resumed conversation, since its ``_live`` dict starts
        empty every new process. The drill-in (``subagent_run``/
        ``subagent_runs``) and the ``[N Sub-Agents]`` badge already read
        ``AgentRunsDB`` directly and correctly survive a restart; this gives
        the rail's top-level summary line the same durability, by deriving
        it from the most recent non-superseded primary run for the
        conversation and that primary's own sub-agent runs.

        Returns the idle default when the conversation has no primary run
        at all -- callers should prefer ``live_snapshot`` and only fall
        back to this when it reports ``idle`` (see
        ``ChatScreen._console_agent_section_lines``), so a truly-idle
        conversation (never run, ever) renders identically either way.

        Cached per ``conversation_id`` (Task-7 discipline: the rail poll
        ticks every 0.2s and must not hit the DB on every tick) --
        invalidated whenever this bridge instance itself starts or
        finishes a run for that conversation, at which point
        ``live_snapshot`` takes over as the source of truth anyway.
        """
        cached = self._historical_cache.get(conversation_id)
        if cached is not None:
            return cached
        snapshot = self._derive_historical_snapshot(conversation_id)
        self._historical_cache[conversation_id] = snapshot
        return snapshot

    def subagent_runs(self, conversation_id: str) -> list[dict]:
        return [r for r in self._db.list_runs(conversation_id)
                if r["agent_kind"] == AGENT_KIND_SUBAGENT]

    def subagent_run(self, run_id: str) -> dict | None:
        return self._db.get_run(run_id)

    def subagent_count(self, conversation_id: str) -> int:
        return self._db.count_subagent_runs(conversation_id)

    def subagent_counts(self, conversation_ids: list[str]) -> dict[str, int]:
        """Batched per-conversation sub-agent counts (Finding A).

        One call replaces one ``subagent_count(cid)`` call per row -- see
        ``AgentRunsDB.count_subagents_by_conversation`` for the query.
        """
        return self._db.count_subagents_by_conversation(conversation_ids)

    def resume_marker_messages(self, conversation_id: str) -> list[list[ConsoleChatMessage]]:
        """Re-derive transcript TOOL marker messages from ``AgentRunsDB`` for resume.

        Plan-B final-review Medium-1: the rail (``historical_snapshot``) and
        the ``[N Sub-Agents]`` badge already re-derive from ``AgentRunsDB``
        on resume; the inline transcript TOOL markers did not -- they are
        only ever appended live via ``_append_marker`` with
        ``persist=False``, so a session rebuilt fresh from ChaChaNotes never
        sees them.

        Returns one marker-message block per non-superseded PRIMARY run for
        the conversation, oldest run first (``list_runs`` itself returns
        newest-first, so the order is reversed here). Each block holds that
        run's own TOOL marker messages, in the run's recorded step order,
        built with ``format_agent_step_marker`` -- the same formatter the
        live bridge uses -- so a resumed transcript's markers are
        byte-identical to what the live run produced. A run with no
        marker-worthy steps (e.g. a plain answer, no tool/spawn/error step)
        yields an empty block; callers should skip those rather than inject
        nothing.

        Placement of the returned blocks into a transcript is the caller's
        job -- see ``inject_resume_agent_markers``.
        """
        records = [
            record for record in self._db.list_runs(conversation_id, include_superseded=False)
            if record["agent_kind"] == AGENT_KIND_PRIMARY
        ]
        records.reverse()  # list_runs is newest-first; markers must read chronologically
        blocks: list[list[ConsoleChatMessage]] = []
        for record in records:
            block: list[ConsoleChatMessage] = []
            for step in record.get("steps") or []:
                text = format_agent_step_marker(
                    str(step.get("kind") or ""),
                    tool_name=step.get("tool_name"),
                    result=step.get("result"),
                    summary=step.get("summary"),
                )
                if text is not None:
                    block.append(ConsoleChatMessage(
                        role=ConsoleMessageRole.TOOL, content=text, status="complete"))
            blocks.append(block)
        return blocks

    # -- internals ------------------------------------------------------

    def _append_marker(self, session_id: str, text: str) -> None:
        # Kept raw (no escaping): both consumers render markup-off --
        # console_transcript.py's _message_render_text builds a Content via
        # Content.assemble (never markup-parsed) and chat_screen.py's legacy
        # fallback wraps the string in a bare rich.text.Text(...) (also never
        # markup-parsed). Escaping here for a parser that never runs used to
        # leave literal backslashes in the rendered marker (`fetch [docs]` ->
        # `fetch \[docs]`).
        try:
            self._store.append_message(
                session_id, role=ConsoleMessageRole.TOOL, content=text)
        except KeyError:
            pass   # session vanished mid-run; the rail still has the live snapshot

    @staticmethod
    def _summarize(step: AgentStep) -> str:
        # Finding B: feeds only AgentLiveStep.text, which
        # _console_agent_section_lines renders into a markup=False Static --
        # escaping here (a second guard on top of markup=False) produced
        # literal backslashes for bracketed text. Left raw; the transcript
        # TOOL marker path (_append_marker) is also raw, since its consumers
        # never parse the text as markup either.
        raw = step.summary or step.result or step.tool_name or step.kind
        return str(raw)[:200]

    def _previous_primary_run_id(self, conversation_id: str) -> str | None:
        for record in self._db.list_runs(conversation_id, include_superseded=False):
            if record["agent_kind"] == AGENT_KIND_PRIMARY:
                return record["id"]
        return None

    def _derive_historical_snapshot(self, conversation_id: str) -> AgentLiveSnapshot:
        # One query covers both the primary lookup and its sub-agents --
        # AgentRunsDB has no separate "get one conversation's tree" call,
        # and issuing two queries here would double the DB hit this cache
        # exists to avoid.
        records = self._db.list_runs(conversation_id, include_superseded=False)
        primary = next(
            (r for r in records if r["agent_kind"] == AGENT_KIND_PRIMARY), None)
        if primary is None:
            return AgentLiveSnapshot()
        steps = tuple(
            AgentLiveStep(
                kind=str(step.get("kind") or ""),
                text=self._summarize_persisted_step(step),
                agent_kind=AGENT_KIND_PRIMARY,
            )
            for step in (primary.get("steps") or [])[-5:]
        )
        subagents = tuple(
            SubAgentSummary(
                text=str(record.get("task") or ""),
                status=str(record.get("status") or "running"),
            )
            for record in records
            if record["agent_kind"] == AGENT_KIND_SUBAGENT
            and record.get("parent_run_id") == primary["id"]
        )
        return AgentLiveSnapshot(
            status=str(primary.get("status") or "idle"),
            step=len(primary.get("steps") or []),
            steps=steps,
            subagents=subagents,
        )

    @staticmethod
    def _summarize_persisted_step(step: dict) -> str:
        # Mirrors _summarize's precedence for a live AgentStep, but reads a
        # persisted (JSON-decoded) step dict instead -- also left raw (no
        # escaping) for the same Finding-B reason: this text only ever
        # renders into a markup=False Static.
        raw = (step.get("summary") or step.get("result")
               or step.get("tool_name") or step.get("kind") or "")
        return str(raw)[:200]
