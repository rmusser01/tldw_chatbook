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
import contextlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    FIND_TOOLS_NAME,
    LOAD_TOOLS_NAME,
    RunBudget,
    RUNTIME_TOOL_NAMES,
    SPAWN_TOOL_NAME,
    STEP_ERROR,
    STEP_SPAWN,
    STEP_TOOL_RESULT,
    AgentConfig,
    AgentStep,
    RunOutcome,
    SkillFileBindings,
    ToolCall,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT, AgentService
from tldw_chatbook.Agents.agent_stream import StreamGate
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    SkillToolProvider,
    ToolCatalogRegistry,
    intersect_skill_tools,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_provider_gateway import ProviderToolCalls
from tldw_chatbook.Chat.console_skill_resolver import SKILL_UNTRUSTED_REFUSE
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Internal_Prompts import get_internal_prompt
from tldw_chatbook.Internal_Prompts.catalog import CATALOG
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError

# Catalog-default re-export: keeps existing imports/tests valid and pins
# the "shipped default" text. compose_agent_system_prompt below resolves
# the live (possibly overridden) value at call time via get_internal_prompt.
CONSOLE_AGENT_OPERATING_PROMPT = CATALOG["agents.console_agent_operating"].default

# Every `agents.subagent_system` value ever resolved by `_is_subagent`
# (below) this process, seeded with the shipped default. Grows by at most
# one entry per distinct override text a user configures -- see
# `_StreamingModelAdapter._is_subagent` for why a single current-value
# check is not enough.
_KNOWN_SUBAGENT_PREFIXES: set[str] = {SUBAGENT_SYSTEM_PROMPT}

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
# cost 2 more turns / 6 more steps (6 turns / 16 steps total), so even the
# old 8-turn cap cleared the floor with room to spare.
#
# The three numbers below are sized TOGETHER so that max_model_turns stays
# the primary limiter -- raising it alone would just move the wall to
# whichever of the other two binds first:
#   * max_model_turns=20 gives ~20 tool-calling rounds per user message
#     (raised from 8).
#   * max_steps=64: a fence tool round costs 3 steps (STEP_MODEL +
#     STEP_TOOL_CALL + STEP_TOOL_RESULT), so 19 rounds + 1 wrap-up
#     STEP_MODEL = 3*19 + 1 = 58 steps. At the old 32 the step check would
#     have fired around round 10, never letting the 20-turn cap be reached.
#     64 clears 58 while staying a real backstop: a NATIVE multi-call batch
#     (task-243) costs 1 + 2N steps per turn, so a run of heavy parallel
#     batches can still legitimately hit the step backstop before the turn
#     cap -- that is the backstop doing its job.
#   * max_wall_seconds=1200: the prior 480s was derived as 25-50s/turn x up
#     to 8 model turns at the slow local-model pace this gate exercises; at
#     20 turns that same pace needs ~500-1000s, so 480 would have become
#     the new binding limit around turn 10. 1200s covers the 20-turn worst
#     case. This is a backstop, not a target -- fast cloud models finish 20
#     turns in a fraction of it, and the user can Stop at any point (the
#     tool-call wrapper polls cancellation every 0.5s, task-327).
# The engine's own RunBudget defaults (agent_models.RunBudget) keep the
# bare max_steps=8, so this override applies only at the Console bridge's
# own config-assembly site (run_reply below); other callers of
# RunBudget()/AgentConfig keep the conservative engine default.
#: Tool-calling rounds the Console agent gets per user message. THE primary
#: limiter -- the two constants below exist to keep it reachable.
CONSOLE_MAX_MODEL_TURNS = 20

#: Step backstop. A fence round costs 3 steps (STEP_MODEL + STEP_TOOL_CALL +
#: STEP_TOOL_RESULT) and the wrap-up reply costs 1, so N turns need
#: 3*(N-1)+1 steps -- 58 at N=20. 64 clears that while staying a real
#: backstop for native multi-call batches (1 + 2N steps per turn).
#: `test_console_budget_step_cap_admits_a_full_model_turn_run` fails if this
#: ever drops below the derived minimum.
CONSOLE_MAX_STEPS = 64

#: Wall-clock backstop for the whole run, at the slow local-model pace this
#: gate exercises (25-50s per turn x CONSOLE_MAX_MODEL_TURNS).
CONSOLE_MAX_WALL_SECONDS = 1200.0

CONSOLE_RUN_BUDGET = RunBudget(
    max_steps=CONSOLE_MAX_STEPS,
    max_wall_seconds=CONSOLE_MAX_WALL_SECONDS,
    max_model_turns=CONSOLE_MAX_MODEL_TURNS,
)

_QUIET_STEP_TOOLS = {FIND_TOOLS_NAME, LOAD_TOOLS_NAME}


def _combine_state_scopes(scopes: list) -> "Any | None":
    """Combine per-turn state scopes into the one ``review_state_scope`` seam.

    ``AgentService.review_state_scope`` holds a single
    ``Callable[[], AbstractContextManager]``, but more than one component
    can own per-turn stamp state that a nested sub-agent run would clobber
    (task-628): the MCP provider's ``_stamped_decisions`` and the built-in
    gate's ``_stamps``. Entering them together keeps the seam's shape while
    guarding both.

    Args:
        scopes: Zero or more zero-argument callables, each returning a
            context manager that snapshots and restores its owner's
            per-turn state.

    Returns:
        ``None`` when ``scopes`` is empty (the service then uses a
        ``nullcontext``), the single callable when there is exactly one
        (byte-identical to the pre-task-628 wiring), else a callable that
        enters every scope on an ``ExitStack`` so all are restored in
        reverse order even if the nested run raises.
    """
    if not scopes:
        return None
    if len(scopes) == 1:
        return scopes[0]

    @contextlib.contextmanager
    def _combined():
        with contextlib.ExitStack() as stack:
            for scope in scopes:
                stack.enter_context(scope())
            yield

    return _combined


def compose_agent_system_prompt(session_prompt: str) -> str:
    """Compose the primary system prompt: session prompt first, agent prompt appended.

    Args:
        session_prompt: The Console session's own system prompt, if any.

    Returns:
        ``session_prompt`` followed by the (registry-resolved) console agent
        operating prompt (blank-line separated), or just the operating
        prompt when ``session_prompt`` is blank.
    """
    operating = get_internal_prompt("agents.console_agent_operating")
    base = (session_prompt or "").strip()
    if not base:
        return operating
    return f"{session_prompt}\n\n{operating}"


_STEP_MARKER_RESULT_LIMIT = 160
_STEP_SUMMARY_LIMIT = 200


def _truncate_step_text(text: str, *, limit: int) -> str:
    """Collapse long step text to a preview with an explicit truncation affordance.

    TASK-350: a tool result that IS the full answer must not be dumped verbatim
    into a transcript marker (it duplicated the assistant bubble word-for-word),
    and a truncated summary must never be a silent mid-word clip ("the traditional
    rollba"). Cuts on a word boundary when one sits reasonably close to ``limit``,
    then appends an ellipsis and a ``(+N chars)`` hint so the reader can see it was
    collapsed and by how much. Deterministic, so the shared live/resume marker
    formatter stays byte-identical.
    """
    text = str(text if text is not None else "")
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    # Cut on any whitespace boundary (space/newline/tab/CR), not just a literal
    # space — markdown and structured tool output split on newlines/tabs, so a
    # space-only search would still clip those mid-token (Qodo #3).
    boundary = max(cut.rfind(ws) for ws in (" ", "\n", "\t", "\r"))
    if boundary >= limit // 2:
        cut = cut[:boundary].rstrip()
    hidden = len(text) - len(cut)
    return f"{cut}… (+{hidden} chars)"


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
        # Collapse the result to a preview: a spawn_subagent result IS the full
        # answer, and dumping it verbatim duplicated the assistant bubble (task-350).
        preview = _truncate_step_text(
            str(result if result is not None else ""),
            limit=_STEP_MARKER_RESULT_LIMIT,
        )
        return f"⚙ {tool_name} → {preview}"
    if kind == STEP_ERROR:
        return f"⚠ {summary}"
    return None


def inject_resume_agent_markers(
    messages: list[ConsoleChatMessage],
    anchored_blocks: list[tuple[str | None, list[ConsoleChatMessage]]],
) -> list[ConsoleChatMessage]:
    """Interleave AgentRunsDB-derived TOOL marker blocks into a resumed transcript.

    Task 3 placement, anchored by the run's ``assistant_message_id`` (the
    persisted id of the reply it produced -- see
    ``ConsoleAgentBridge.record_run_assistant_message``, written on every
    terminal path since Task 2):

    - **Anchor id set and it matches** a message's ``persisted_message_id``
      in ``messages`` -- the block is inserted immediately after that
      message, wherever it sits (this is exact, not ordinal: a resumed
      transcript may have been edited/branched since the run happened, so
      the Nth run no longer need be the Nth reply).
    - **Anchor id set but it matches no message in ``messages``** -- the
      reply that run produced lives on a different branch than the one
      currently active (an edit/regenerate moved the active path off of
      it). The block is **dropped**: showing that run's tool trace next to
      a DIFFERENT reply would misattribute it, so hiding it is correct.
    - **Anchor id is ``None``** -- a legacy (pre-Phase-C) run, a sub-agent
      run, or one whose terminal path never got to record the id (crash /
      never-persisted reply). Falls back to the prior ordinal placement:
      the Nth null-anchored block is matched to the Nth ASSISTANT message
      in ``messages`` that isn't already claimed by an id-anchored block,
      oldest first. A null block left over with no unclaimed assistant
      message to pair with is appended at the end of the transcript
      instead of being silently dropped, preserving the pre-Task-3
      leftover behavior for this fallback path.

    Idempotent: a block whose marker texts are already present as TOOL
    messages anywhere in ``messages`` is skipped, so calling this twice (or
    resuming into a transcript that already carries live markers) never
    duplicates a block.

    Args:
        messages: The rebuilt transcript (ChaChaNotes-derived; never
            contains TOOL rows on its own, since markers are appended
            live with ``persist=False``).
        anchored_blocks: Per-run ``(assistant_message_id, marker_block)``
            pairs, oldest run first (see
            ``ConsoleAgentBridge.resume_marker_messages``).

    Returns:
        A new list with marker blocks interleaved; ``messages`` itself is
        not mutated.
    """
    non_empty = [(anchor, block) for anchor, block in anchored_blocks if block]
    if not non_empty:
        return list(messages)

    existing_tool_contents = {
        message.content
        for message in messages
        if message.role is ConsoleMessageRole.TOOL
    }

    def _already_present(block: list[ConsoleChatMessage]) -> bool:
        return all(marker.content in existing_tool_contents for marker in block)

    by_persisted = {
        message.persisted_message_id: index
        for index, message in enumerate(messages)
        if message.persisted_message_id
    }

    matched: dict[int, list[list[ConsoleChatMessage]]] = {}
    null_blocks: list[list[ConsoleChatMessage]] = []
    used_indexes: set[int] = set()
    for anchor_id, block in non_empty:
        if anchor_id is None:
            null_blocks.append(block)
            continue
        index = by_persisted.get(anchor_id)
        if index is None:
            continue  # off-path: this run's reply lives on another branch
        matched.setdefault(index, []).append(block)
        used_indexes.add(index)

    unclaimed_assistant_indexes = [
        index
        for index, message in enumerate(messages)
        if message.role is ConsoleMessageRole.ASSISTANT and index not in used_indexes
    ]
    for index, block in zip(unclaimed_assistant_indexes, null_blocks):
        matched.setdefault(index, []).append(block)
    leftover_blocks = null_blocks[len(unclaimed_assistant_indexes) :]

    result: list[ConsoleChatMessage] = []
    for index, message in enumerate(messages):
        result.append(message)
        for block in matched.get(index, ()):
            if not _already_present(block):
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
    turns (leading system content prefixed by the registry-resolved or
    shipped-default ``agents.subagent_system`` prompt — see ``_is_subagent``)
    are streamed to a throwaway gate and never touch the transcript.

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

    def __init__(
        self,
        *,
        store,
        provider_gateway,
        resolution,
        assistant_message_id,
        should_cancel,
        loop,
    ):
        self._store = store
        self._gateway = provider_gateway
        self._resolution = resolution
        self._assistant_message_id = assistant_message_id
        self._should_cancel = should_cancel
        self._loop = loop

    def chat_call(
        self,
        *,
        messages_payload,
        model=None,
        api_endpoint=None,
        streaming=False,
        tools=None,
        **_ignored,
    ) -> dict:
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
                self._resolution, messages_payload, **stream_kwargs
            ):
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
        if first.get("role") != "system":
            return False
        content = str(first.get("content", ""))
        # Multi-prefix match: a sub-agent's system prompt is baked into its
        # messages_payload once, at spawn time (agent_service.py:411's own
        # get_internal_prompt call), then stays fixed for the rest of that
        # sub-agent's multi-step tool loop. Comparing only against the
        # CURRENTLY resolved override (plus the shipped default) can flip
        # false if `agents.subagent_system` is edited live -- e.g. from
        # Settings, on the UI thread, mid-run -- to a *different* override
        # rather than reverted to the default: an already-spawned
        # sub-agent's later turns would then match neither and leak into
        # the primary transcript. Accumulating every value resolved so far
        # this process (starting from the shipped default) keeps detection
        # stable no matter when the override changed.
        resolved = get_internal_prompt("agents.subagent_system")
        _KNOWN_SUBAGENT_PREFIXES.add(resolved)
        return any(
            content.startswith(prefix) for prefix in _KNOWN_SUBAGENT_PREFIXES
        )


def _eligible_skill_entries(context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Trusted, model-invocable skill summaries from a ``get_context`` snapshot.

    Mirrors ``ChatScreen._console_skill_trusted_candidates_from_context``'s
    defensive filter shape, but scoped to model-invocation eligibility
    rather than user (``$skill-name``) invocation: a skill is eligible here
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
    available = (
        context.get("available_skills") if isinstance(context, Mapping) else None
    )
    return [
        item
        for item in (available or [])
        if isinstance(item, Mapping)
        and item.get("name")
        and not item.get("trust_blocked", False)
        and not item.get("disable_model_invocation", False)
    ]


def _non_colliding_skill_entries(
    context: Mapping[str, Any],
    builtin_names: tuple[str, ...],
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
        item
        for item in _eligible_skill_entries(context)
        if str(item["name"]) not in collision_names
    ]


def _compose_run_allowed_tools(
    context: Mapping[str, Any],
    builtin_names: tuple[str, ...],
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
        for item in _non_colliding_skill_entries(context, builtin_names)
    )
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
            entry
            for entry in self._provider.list_catalog()
            if entry.name in self._allowed_names
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        return self._provider.load_schema(tool_id)

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        return self._provider.invoke(tool_id, args)


def _truncate_log_value(value: Any, *, max_len: int = 200) -> str:
    """Return a safe, bounded string representation for logging.

    Tool arguments and results may contain secrets or very large payloads;
    this helper truncates the string form so log lines stay readable and do
    not dump sensitive data into logs.
    """
    text = str(value)
    if len(text) > max_len:
        return f"{text[: max_len - 3]}..."
    return text


def _non_colliding_mcp_names(
    mcp_provider: Any,
    collision_names: frozenset[str] | set[str],
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
    non_colliding, _shadowed = _partition_mcp_catalog_by_collision(
        mcp_provider, collision_names
    )
    return non_colliding


def shadowed_mcp_names(
    mcp_provider: Any,
    collision_names: frozenset[str] | set[str],
) -> tuple[str, ...]:
    """MCP tool names this run drops because a built-in owns the name.

    The exact complement of ``_non_colliding_mcp_names``. Built-ins win
    collisions deliberately -- letting the MCP side win would let a
    compromised server name-squat an audited built-in like ``write_file``
    and intercept calls the user believes are gated -- but a user whose
    configured tool silently stops working has no way to discover why.
    TASK-656's permissions view renders these rows.

    Both this function and ``_non_colliding_mcp_names`` delegate to
    ``_partition_mcp_catalog_by_collision``, which walks the catalog once
    and buckets every entry into exactly one side. That keeps the two
    public results an exact partition by construction -- there is no
    second copy of the ``entry.name in collision_names`` test to drift out
    of sync as the collision rule evolves.

    Args:
        mcp_provider: A composed ``MCPToolProvider`` (or test double).
        collision_names: Names owned by builtins, runtime tools, or skills.

    Returns:
        The dropped names, in catalog order.
    """
    _non_colliding, shadowed = _partition_mcp_catalog_by_collision(
        mcp_provider, collision_names
    )
    return shadowed


def _partition_mcp_catalog_by_collision(
    mcp_provider: Any,
    collision_names: frozenset[str] | set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split an MCP provider's catalog into (non-colliding, shadowed) names.

    The single place the collision predicate is evaluated. Both
    ``_non_colliding_mcp_names`` and ``shadowed_mcp_names`` are thin views
    onto this partition, so they can never disagree about which side a
    given name falls on.

    Args:
        mcp_provider: A composed ``MCPToolProvider`` (or test double) whose
            ``list_catalog()`` has already been built.
        collision_names: Names that must never be treated as a distinct
            MCP tool -- builtins, ``RUNTIME_TOOL_NAMES``, and this run's
            own eligible skill names.

    Returns:
        A ``(non_colliding, shadowed)`` pair, each in catalog order, whose
        union (in either order) reproduces the full catalog's names with
        no overlap and no omission.
    """
    non_colliding: list[str] = []
    shadowed: list[str] = []
    for entry in mcp_provider.list_catalog():
        bucket = shadowed if entry.name in collision_names else non_colliding
        bucket.append(entry.name)
    return tuple(non_colliding), tuple(shadowed)


def _compose_run_registry_and_allowed(
    context: Mapping[str, Any],
    *,
    mcp_provider: Any | None = None,
    builtin_gate: Any | None = None,
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
        builtin_gate: task-545/T6 -- THIS run's ``BuiltinToolGate``,
            threaded into the freshly-constructed ``BuiltinToolProvider``
            so its ``invoke()`` enforces the SAME gate instance the run's
            review hook stamps (``console_chat_controller.
            build_tool_review_hook``). ``None`` leaves the provider to
            build its own lazy gate on first use (``BuiltinToolProvider``'s
            own fail-closed default) -- callers that care about the hook
            and ``invoke()`` agreeing on stamps must pass the same object
            to both.

    Returns:
        ``(registry, allowed_tools, builtin_names)`` -- the per-run
        registry, its full allow-list (builtins + eligible skills +
        eligible MCP tools + spawn), and just the builtin names (needed
        separately by ``_BridgeSkillRunner`` to intersect a skill's own
        declared ``allowed_tools`` against -- never against skill names,
        so a skill's sub-agent can never call another skill).
    """
    registry = ToolCatalogRegistry()
    builtin_provider = BuiltinToolProvider(gate=builtin_gate)
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
        for shadowed in shadowed_mcp_names(mcp_provider, collision_names):
            logger.warning(
                "MCP tool {name} is shadowed by a built-in of the same name "
                "and is not offered this run",
                name=shadowed,
            )
        if mcp_names:
            registry.register_provider(
                _CollisionFilteredMCPProvider(mcp_provider, frozenset(mcp_names))
            )
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
    discipline for the ``$skill-name`` user-invocation path).
    """

    def __init__(
        self,
        *,
        skills_service: Any,
        skill_names: frozenset[str],
        builtin_names: tuple[str, ...],
        skill_file_bindings: SkillFileBindings | None = None,
    ) -> None:
        self._skills_service = skills_service
        self._skill_names = skill_names
        self._builtin_names = builtin_names
        self._skill_file_bindings = skill_file_bindings

    def is_skill_tool(self, name: str) -> bool:
        return name in self._skill_names

    def run(self, name: str, args: str, spawn: Callable[..., ToolResult]) -> ToolResult:
        try:
            result = asyncio.run(
                self._skills_service.execute_skill(name, mode="local", args=args)
            )
        except SkillTrustBlockedError as exc:
            return ToolResult(
                ok=False,
                error=SKILL_UNTRUSTED_REFUSE.format(name=name, reason=exc.reason_code),
            )
        rendered = (
            result.get("rendered_prompt", "") if isinstance(result, Mapping) else ""
        )
        declared_allowed_tools = (
            result.get("allowed_tools") if isinstance(result, Mapping) else None
        )
        allowed_tools = intersect_skill_tools(
            declared_allowed_tools, self._builtin_names
        )
        # task-4 (skills-fork-reachability): grant the spawned skill's own
        # name skill_file authorization BEFORE spawn -- so the child's very
        # first turn can already read its own bundled reference files (see
        # SkillFileBindings' own docstring: authorization lives here, never
        # in config.allowed_tools) -- then append a "Bundled files" pointer
        # block to the rendered task text whenever execute_skill reported
        # any (absent when the skill has no bundle beyond SKILL.md).
        if self._skill_file_bindings is not None:
            self._skill_file_bindings.authorized.add(name)
        refs = result.get("reference_files") if isinstance(result, Mapping) else None
        if refs and self._skill_file_bindings is not None:
            rows = ", ".join(
                f"{r['path']} ({r['size']} bytes"
                f"{'' if r.get('is_text', True) else ', binary'})"
                for r in refs
            )
            rendered = f"{rendered}\n\nBundled files (readable via skill_file): {rows}"
        return spawn(rendered, allowed_tools=allowed_tools)


class ConsoleAgentBridge:
    """Owns the tool registry + run store and runs one primary agent reply."""

    def __init__(
        self,
        *,
        agent_runs_db: AgentRunsDB,
        store,
        provider_gateway,
        registry: ToolCatalogRegistry | None = None,
        clock: Callable[[], float] = time.monotonic,
        skills_service: Any | None = None,
        native_tools_enabled: Callable[[], bool] | None = None,
    ) -> None:
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
        self._allowed_tools = tuple(e.name for e in registry.list_catalog()) + (
            SPAWN_TOOL_NAME,
        )
        self._live: dict[str, AgentLiveSnapshot] = {}
        self._historical_cache: dict[str, AgentLiveSnapshot] = {}

    def native_tool_schemas(self) -> list[dict[str, Any]]:
        """Return the native tool schemas available to this bridge.

        Iterates the bridge's tool registry and returns one schema dict per
        catalog entry.  Failures to load an individual schema are logged and
        skipped so the preview remains useful even when a provider is slow or
        misconfigured.
        """
        schemas: list[dict[str, Any]] = []
        for entry in self._registry.list_catalog():
            try:
                schema = self._registry.load_schema(entry.id)
            except Exception as exc:  # pragma: no cover - defensive only
                logger.warning(
                    "Failed to load schema for {tool_id}: {exc}",
                    tool_id=entry.id, exc=exc,
                )
                continue
            schemas.append({
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.parameters,
            })
        return schemas

    # -- run ------------------------------------------------------------

    def run_reply(
        self,
        *,
        conversation_id: str,
        session_id: str,
        resolution: Any,
        assistant_message_id: str,
        model: str,
        session_system_prompt: str,
        agent_messages: list[dict],
        should_cancel: Callable[[], bool],
        supersede_previous: bool = False,
        mcp_provider: Any | None = None,
        builtin_gate: Any | None = None,
        review_tool_calls: Callable[[list[ToolCall]], dict[str, str]] | None = None,
        turn_skill_bindings: tuple[str, ...] = (),
        turn_bundle_block: str = "",
        request_skill_install_confirm: Callable[[str], bool] | None = None,
        request_skill_script_confirm: Callable[[dict], dict] | None = None,
    ) -> tuple[str, RunOutcome]:
        """Run the agent loop as the Console reply engine.

        The primary run row is created with a NULL ``assistant_message_id``
        (the native ``assistant_message_id`` argument is used only for
        streaming into the placeholder, never forwarded to ``run_turn`` --
        see the ``run_turn`` call below for why a native id must never be
        stored on the run). The caller records the reply's durable persisted
        id onto the run on every terminal path via
        ``record_run_assistant_message`` once the reply is persisted; an
        unfinished/crashed run stays NULL for resume's null->ordinal fallback.

        Concurrency: this bridge does NOT serialize runs. The
        ``_live``/``_historical_cache`` dicts are per-conversation DISPLAY
        snapshots, not a mutual-exclusion guard. Serialization is enforced
        upstream by ``ConsoleChatController`` (its ``_active_run_rejection``
        / ``run_state.is_send_allowed`` gate -- covered by
        ``Tests/UI/test_console_run_gate.py``), and that gate is actually
        CONTROLLER-WIDE (only one run active across the whole controller at
        a time), which trivially bounds it per conversation too: a second
        send -- whether to the same conversation or a different, otherwise-
        idle one -- while any run is live is rejected there before
        ``run_reply`` is ever called. Do not add a competing guard here.

        task-545/T6: ``builtin_gate`` (when passed) is threaded into this
        run's freshly-built ``BuiltinToolProvider`` so its ``invoke()``
        checks the SAME gate instance the caller's review hook
        (``console_chat_controller.build_tool_review_hook``) already
        stamped -- see ``_compose_run_registry_and_allowed``'s own
        docstring for why two independently-built gates would silently
        desynchronize. Passing ``None`` (the default -- existing callers
        that don't care about built-in gating are unaffected) leaves a
        skills/MCP-free run on the shared, construction-time
        ``self._registry``/``self._allowed_tools`` fast path unchanged.

        Returns:
            A ``(run_id, outcome)`` tuple: the primary run's id (so the
            caller can record the produced reply's persisted id onto the run
            via ``record_run_assistant_message`` after the reply is
            persisted) and its terminal ``RunOutcome``.
        """
        # Per-run tool registry + allow-list (Task 12, extended by P5-T6 for
        # MCP, and by task-545/T6 for a per-run builtin_gate): rebuilt FRESH
        # for this run whenever there is a skills service, an already-
        # composed MCP provider, OR a builtin_gate for this run (never
        # cached across runs, and never the shared self._registry/
        # self._allowed_tools built at construction) -- so a skill or MCP
        # tool approved/edited/revoked since the last run always takes
        # effect on the very next one. `mcp_provider` is built and
        # composed by the CALLER (ConsoleChatController._compose_mcp_
        # provider, on the running Textual main loop, BEFORE this method
        # is dispatched onto asyncio.to_thread) -- see MCPToolProvider's
        # own module docstring for why `compose_catalog()`'s async I/O can
        # never run from inside this worker-thread method. `builtin_gate`
        # MUST route through this fresh-build branch rather than the
        # shared fast path below: the shared path's own `BuiltinToolProvider`
        # is built once at bridge-construction time with `gate=None` (its
        # own lazy default), which would be a SECOND, independently-built
        # gate the run's review hook never stamps -- see
        # `_compose_run_registry_and_allowed`'s own docstring for the
        # desync this would cause. None of skills service, MCP provider,
        # or builtin_gate: the shipped shared registry/allow-list is used
        # unchanged -- the no-skills, no-MCP, no-gate path stays
        # byte-identical to before this task (existing callers that never
        # pass `builtin_gate` see no behavior change at all).
        registry = self._registry
        allowed_tools = self._allowed_tools
        skill_runner = None
        # task-4 (skills-fork-reachability): one SkillFileBindings per run,
        # handed to BOTH AgentService (the loop's authorization + reader
        # closure -- Task 3) and this run's _BridgeSkillRunner (which grants
        # a spawned skill's own name pre-spawn) -- never two independently-
        # seeded copies, or the runner's grant would never reach the loop's
        # check. `authorized` starts empty here (Task 5 seeds the turn's
        # $skill names); the reader is a SYNC adapter over the async scope-
        # service read, matching _BridgeSkillRunner.run's own
        # asyncio.run-in-worker-thread pattern just below.
        skill_file_bindings = None
        if (
            self._skills_service is not None
            or mcp_provider is not None
            or builtin_gate is not None
        ):
            context: Mapping[str, Any] = {}
            if self._skills_service is not None:
                context = asyncio.run(self._skills_service.get_context(mode="local"))
            registry, allowed_tools, builtin_names = _compose_run_registry_and_allowed(
                context, mcp_provider=mcp_provider, builtin_gate=builtin_gate
            )
            if self._skills_service is not None:
                skill_names = frozenset(
                    str(item["name"])
                    for item in _non_colliding_skill_entries(context, builtin_names)
                )
                skill_file_bindings = SkillFileBindings(
                    authorized=set(),
                    reader=lambda skill_name, path: asyncio.run(
                        self._skills_service.read_skill_file(
                            skill_name, path, mode="local"
                        )
                    ),
                )
                skill_runner = _BridgeSkillRunner(
                    skills_service=self._skills_service,
                    skill_names=skill_names,
                    builtin_names=builtin_names,
                    skill_file_bindings=skill_file_bindings,
                )
        # task-5 (skills-fork-reachability): seed this run's own bindings
        # with the names the CONTROLLER already resolved/spliced for the
        # triggering turn (a leading `$skill` mention, or embedded mentions
        # that actually spliced) -- so the primary agent's very first turn
        # can already read that skill's bundle via skill_file, matching
        # what a spawned skill child gets for its OWN bundle (Task 4).
        # `skill_file_bindings` is None whenever there is no skills service
        # for this run, in which case a non-empty `turn_skill_bindings`
        # (which can only happen when the controller's own skills-service-
        # gated substitution ran) has nothing to seed.
        if skill_file_bindings is not None:
            skill_file_bindings.authorized.update(turn_skill_bindings)
        # Agent-callable skill install (5th runtime tool). Built only when
        # BOTH a skills service AND a confirm callback exist -- without a
        # callback the tool is simply absent (never advertised) rather than
        # auto-denying every call; wired to AgentService, which pins/
        # dispatches it for the top-level agent only. Order (load-bearing):
        # enforce policy (no prompt on denial) -> classify URL (no prompt on
        # a bad URL) -> in-chat confirm (plain blocking call, OUTSIDE asyncio.run)
        # -> asyncio.run(install) -> broad-catch wrap. import_skill_file
        # raises a bare ValueError("local_skill_exists:...") on collision, so
        # the install catch is broad.
        install_skill_tool = None
        if self._skills_service is not None and request_skill_install_confirm is not None:
            scope = self._skills_service

            def install_skill_tool(url: str) -> ToolResult:
                from tldw_chatbook.Skills_Interop.skill_remote_fetch import (
                    classify_skill_source_url,
                    install_skill_from_url,
                )
                from tldw_chatbook.runtime_policy.types import PolicyDeniedError

                try:
                    scope.enforce_install_remote()
                except PolicyDeniedError as exc:
                    return ToolResult(ok=False, error=exc.user_message)
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(ok=False, error=str(exc))
                try:
                    classify_skill_source_url(url)
                except Exception as exc:  # noqa: BLE001 (RemoteSkillError etc.)
                    return ToolResult(ok=False, error=str(exc))
                try:
                    allowed = bool(
                        request_skill_install_confirm(url)
                        if request_skill_install_confirm is not None
                        else False
                    )
                except Exception:  # noqa: BLE001 — a UI error fails closed
                    allowed = False
                if not allowed:
                    return ToolResult(
                        ok=False, error="The user declined to install this skill."
                    )
                try:
                    result = asyncio.run(
                        install_skill_from_url(url, scope_service=scope)
                    )
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(ok=False, error=str(exc))
                name = result.get("name", "") if isinstance(result, dict) else ""
                return ToolResult(
                    ok=True,
                    content=(
                        f'Installed "{name}" — it is pending your review and '
                        "cannot run until you approve it in Library > Skills."
                    ),
                )

        # Trust-gated skill script execution (6th runtime tool). Built only
        # when a skills service AND a confirm callback exist AND this
        # platform's sandbox is actually usable -- without any one of the
        # three the tool is absent (never advertised) rather than
        # auto-denying every call. The platform check matters because the
        # sandbox (skill_script_runner.run_script_subprocess) depends on
        # POSIX-only primitives (process-group teardown, RLIMIT_* via the
        # `resource` module) that do not exist on Windows: "advertised must
        # equal usable" -- a tool the model can call but that always raises
        # is a defect, not a graceful degradation, so it must simply not be
        # wired on an unsupported platform. Order (load-bearing): enforce
        # policy (no prompt on denial) -> describe/resolve (no prompt on a
        # bad path or an unrunnable type) -> grant check (no prompt when the
        # user already granted this skill) -> confirm (plain blocking call,
        # OUTSIDE any asyncio.run) -> run -> broad-catch wrap.
        # run_skill_script re-verifies policy/trust/path authoritatively, so
        # a stale plan can never widen what actually executes.
        from tldw_chatbook.Skills_Interop.skill_script_runner import (
            sandbox_supported,
        )

        run_skill_script_tool = None
        if (
            self._skills_service is not None
            and request_skill_script_confirm is not None
            and sandbox_supported()
        ):
            scope = self._skills_service
            trust_service = getattr(
                getattr(scope, "local_service", None), "trust_service", None
            )

            def run_skill_script_tool(
                skill_name: str, script_path: str, args: list[str]
            ) -> ToolResult:
                from tldw_chatbook.runtime_policy.types import PolicyDeniedError

                try:
                    scope.enforce_run_script()
                except PolicyDeniedError as exc:
                    return ToolResult(ok=False, error=exc.user_message)
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(ok=False, error=str(exc))
                try:
                    plan = asyncio.run(
                        scope.describe_skill_script(skill_name, script_path)
                    )
                except Exception as exc:  # noqa: BLE001 (trust/path/type)
                    return ToolResult(ok=False, error=f"run_skill_script: {exc}")

                granted = False
                if trust_service is not None:
                    try:
                        granted = bool(
                            trust_service.script_execution_granted(skill_name)
                        )
                    except Exception:  # noqa: BLE001 — doubt ⇒ prompt
                        granted = False
                if not granted:
                    try:
                        decision = request_skill_script_confirm(
                            {
                                # plan.skill_name, not the agent's raw
                                # spelling: the service normalizes the name
                                # it acts on, and a consent card must show
                                # the value that will actually be used.
                                "skill_name": str(
                                    getattr(plan, "skill_name", None) or skill_name
                                ),
                                "script_path": script_path,
                                "mechanism": plan.mechanism,
                                "interpreter": plan.interpreter_display,
                                "is_binary": plan.is_binary,
                                "args": [str(a) for a in args],
                            }
                        )
                    except Exception:  # noqa: BLE001 — a UI error fails closed
                        decision = {"allow": False, "remember": False}
                    if not isinstance(decision, Mapping):
                        decision = {"allow": False, "remember": False}
                    if not decision.get("allow", False):
                        return ToolResult(
                            ok=False, error="The user declined to run this script."
                        )
                    if decision.get("remember", False) and trust_service is not None:
                        # Deliberate ordering: this persists the standing
                        # grant BEFORE run_skill_script below actually runs,
                        # so "remember my choice" sticks even if this
                        # particular run then fails (e.g. trust revoked
                        # mid-flight). That is fine -- run_skill_script
                        # re-verifies policy/trust/path authoritatively on
                        # every call regardless of this grant, so recording
                        # it early never widens what is allowed to execute.
                        try:
                            trust_service.grant_script_execution(skill_name)
                        except Exception:  # noqa: BLE001 — grant is best-effort
                            logger.opt(exception=True).debug(
                                "Failed to persist skill script grant"
                            )
                try:
                    outcome = asyncio.run(
                        scope.run_skill_script(skill_name, script_path, list(args))
                    )
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(ok=False, error=f"run_skill_script: {exc}")

                lines = [f"exit_code: {outcome.exit_code}"]
                if outcome.timed_out:
                    lines.append("timed out — the script was killed")
                if outcome.output_capped:
                    lines.append("output was truncated at the size cap")
                for warning in outcome.sandbox_warnings:
                    lines.append(f"note: {warning}")
                if outcome.stdout:
                    lines.append(f"stdout:\n{outcome.stdout}")
                if outcome.stderr:
                    lines.append(f"stderr:\n{outcome.stderr}")
                # task-584: report WHAT was produced, never its contents. This
                # string enters the model's context, and a script's artifact is
                # not trust-reviewed material -- the listing is name + size, and
                # the path lets the user (or a file tool, where enabled) open it.
                if outcome.output_files:
                    listed = ", ".join(
                        f"{item['name']} ({item['size']} bytes)"
                        for item in outcome.output_files
                    )
                    lines.append(f"produced {len(outcome.output_files)} file(s): {listed}")
                    lines.append(f"output directory: {outcome.output_dir}")
                return ToolResult(ok=True, content="\n".join(lines))

        # [console] native_tool_calls kill-switch (Task 5): a caller-supplied
        # predicate (chat_screen.py's _console_native_tool_calls_enabled)
        # gates whether this run may use native provider tool-calls at all;
        # no predicate (fakes/tests that never pass one) defaults to
        # always-on, matching the pre-kill-switch behavior.
        native_tools = (
            True
            if self._native_tools_enabled is None
            else bool(self._native_tools_enabled())
        )
        config = AgentConfig(
            model=model,
            system_prompt=compose_agent_system_prompt(session_system_prompt),
            allowed_tools=allowed_tools,
            budget=CONSOLE_RUN_BUDGET,
            native_tools=native_tools,
        )
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
            store=self._store,
            provider_gateway=self._gateway,
            resolution=resolution,
            assistant_message_id=assistant_message_id,
            should_cancel=should_cancel,
            loop=run_loop,
        )

        live_steps: list[AgentLiveStep] = []
        subagents: list[SubAgentSummary] = []
        self._live[conversation_id] = AgentLiveSnapshot(status="running")
        # A live run is starting -- live_snapshot takes over as the rail's
        # source of truth for this conversation from here on, so any
        # previously cached historical (DB-derived) summary is stale.
        self._historical_cache.pop(conversation_id, None)

        def on_step(step: AgentStep, agent_kind: str) -> None:
            live_steps.append(
                AgentLiveStep(step.kind, self._summarize(step), agent_kind)
            )
            if agent_kind == AGENT_KIND_PRIMARY:
                if step.kind == STEP_SPAWN:
                    subagents.append(SubAgentSummary(step.summary or ""))
                # format_agent_step_marker is the single source of truth for
                # marker text -- shared with resume_marker_messages below --
                # so live and resume-rebuilt transcripts render identically
                # (Plan-B final-review Medium-1). See its docstring for why
                # the text must stay raw/unescaped.
                marker_text = format_agent_step_marker(
                    step.kind,
                    tool_name=step.tool_name,
                    result=step.result,
                    summary=step.summary,
                )
                if marker_text is not None:
                    self._append_marker(session_id, marker_text)
            # Diagnostic logging for every tool call and result. The actual
            # tool invocation lives inside AgentService, so we observe it
            # through the step stream it emits.
            if step.kind == STEP_TOOL_RESULT:
                logger.debug(
                    "agent tool call: agent_kind={agent_kind} tool={tool_name} "
                    "args={args} result={result} step={step_index}",
                    agent_kind=agent_kind,
                    tool_name=step.tool_name,
                    args=_truncate_log_value(step.args),
                    result=_truncate_log_value(step.result),
                    step_index=step.index,
                )
            elif step.kind == STEP_ERROR:
                logger.warning(
                    "agent step error: agent_kind={agent_kind} tool={tool_name} "
                    "summary={summary} step={step_index}",
                    agent_kind=agent_kind,
                    tool_name=step.tool_name,
                    summary=step.summary,
                    step_index=step.index,
                )
            self._live[conversation_id] = AgentLiveSnapshot(
                status="running",
                step=len(live_steps),
                steps=tuple(live_steps[-5:]),
                subagents=tuple(subagents),
            )

        # C1 (probe-verified security regression): thread the composed MCP
        # provider's stamp_scope() through as AgentService's generic
        # review_state_scope seam whenever one was composed for this run --
        # see AgentService.__init__'s own comment and
        # MCPToolProvider.stamp_scope's docstring for the exact adversarial
        # interleave this protects against (a spawned child's own turn(s)
        # clobbering the parent turn's already-decided MCP approval stamps).
        # getattr(..., None) rather than a bare attribute access: mcp_provider
        # is typed Any (a ToolProvider-shaped double in tests may not define
        # stamp_scope at all, and MUST NOT be forced to); production always
        # hands in a real, fully-composed MCPToolProvider here, which always
        # has it.
        # task-628: the seam holds ONE context manager, but two components
        # now own per-turn stamp state a nested sub-agent run would clobber
        # -- the MCP provider's `_stamped_decisions` and the built-in gate's
        # `_stamps`. Compose whichever exist rather than leaving the gate's
        # state unguarded (it was, before this task, and unlike MCP it has
        # no per-call approval fallback to degrade to: a lost stamp fails
        # closed outright).
        _scopes = [
            scope
            for scope in (
                getattr(mcp_provider, "stamp_scope", None)
                if mcp_provider is not None
                else None,
                getattr(builtin_gate, "stamp_scope", None)
                if builtin_gate is not None
                else None,
            )
            if scope is not None
        ]
        review_state_scope = _combine_state_scopes(_scopes)
        service = AgentService(
            self._db,
            registry,
            chat_call=adapter.chat_call,
            clock=self._clock,
            on_step=on_step,
            skill_runner=skill_runner,
            skill_file_bindings=skill_file_bindings,
            review_tool_calls=review_tool_calls,
            review_state_scope=review_state_scope,
            install_skill_tool=install_skill_tool,
            run_skill_script_tool=run_skill_script_tool,
        )

        supersede_run_id = (
            self._previous_primary_run_id(conversation_id)
            if supersede_previous
            else None
        )
        # task-5 (skills-fork-reachability): append the turn's pre-rendered
        # "Bundled files" block (built controller-side as pure string work
        # over `execute_skill` results already in hand -- Task 4's
        # byte-identical row format) to the LAST role=="user" entry of THIS
        # run's OWN copy of `agent_messages` -- the caller's list and
        # message dict are never mutated. This is the only place the block
        # is ever inserted into a payload: substitution built it but never
        # wrote it into messages, and plain (non-agent) sends never call
        # run_reply at all, so they drop it unused. No-op (the original
        # `agent_messages` list is used unchanged) when there is no block
        # to append or no user message to append it to.
        run_messages = agent_messages
        if turn_bundle_block:
            for index in range(len(agent_messages) - 1, -1, -1):
                message = agent_messages[index]
                content = message.get("content")
                if (
                    message.get("role") == ConsoleMessageRole.USER.value
                    and isinstance(content, str)
                ):
                    run_messages = list(agent_messages)
                    run_messages[index] = {
                        **message,
                        "content": f"{content}\n\n{turn_bundle_block}",
                    }
                    break
        try:
            run_id, outcome = service.run_turn(
                conversation_id=conversation_id,
                messages=run_messages,
                config=config,
                # Intentionally NOT forwarding the native in-memory id here.
                # create_run would store it, but the native id can never match
                # any persisted_message_id -- so a run left unfinished (stopped
                # mid-run, cancelled, failed, or crashed) would hold a stale,
                # non-null id that resume anchoring can never match (and Task 3
                # would drop as "off-path", silently hiding its markers). By
                # omitting it the run row starts NULL; the controller writes the
                # durable PERSISTED id onto the run on EVERY terminal path once
                # the reply is persisted (see record_run_assistant_message), and
                # any still-unfinished run stays NULL for resume's null->ordinal
                # fallback. run_turn's assistant_message_id threading stays a
                # generic service capability (its own tests call it directly).
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
                    or getattr(resolution, "provider", "")
                    or "agent"
                ),
                should_cancel=should_cancel,
                supersede_run_id=supersede_run_id,
            )
        finally:
            run_loop.close()
        for step in outcome.steps:
            logger.info(
                "agent run step",
                agent_kind=AGENT_KIND_PRIMARY,
                step_kind=step.kind,
                tool_name=step.tool_name,
                summary=step.summary,
                step_index=step.index,
            )
        logger.info(
            "console agent bridge run_reply end",
            conversation_id=conversation_id,
            session_id=session_id,
            outcome_status=outcome.status,
            final_text_len=len(outcome.final_text),
            step_count=len(outcome.steps),
        )
        self._live[conversation_id] = AgentLiveSnapshot(
            status=outcome.status,
            step=len(live_steps),
            steps=tuple(live_steps[-5:]),
            subagents=tuple(subagents),
        )
        # The run just finished -- drop any stale historical cache entry so
        # a *later* resume (in a future process) always re-derives fresh
        # rather than reading this run's now-superseded snapshot (belt and
        # braces on top of the pop at run start above).
        self._historical_cache.pop(conversation_id, None)
        return run_id, outcome

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
        return [
            r
            for r in self._db.list_runs(conversation_id)
            if r["agent_kind"] == AGENT_KIND_SUBAGENT
        ]

    def subagent_run(self, run_id: str) -> dict | None:
        return self._db.get_run(run_id)

    def record_run_assistant_message(
        self, run_id: str, persisted_message_id: str
    ) -> None:
        """Record the persisted id of the assistant reply ``run_id`` produced.

        Delegates to ``AgentRunsDB.set_run_assistant_message_id``. Called by
        the controller AFTER the reply is persisted (its native create-time
        id is thereby corrected to the durable persisted id), so a later
        resume can anchor markers by ``persisted_message_id``.
        """
        self._db.set_run_assistant_message_id(run_id, persisted_message_id)

    def latest_unanchored_primary_run_id(self, conversation_id: str) -> str | None:
        """Return the newest non-superseded PRIMARY run's id while unanchored.

        task-543 seam for the stopped-via-cancel path: ``stop_active_run``'s
        ``task.cancel()`` raises ``CancelledError`` in the controller before
        ``run_reply``'s ``(run_id, outcome)`` ever binds, so the controller
        cannot learn the run id from the return value -- but the run ROW
        already exists (``create_run`` runs at loop start, long before the
        first chunk can stream), and the newest non-superseded primary is by
        construction the active run. The ``assistant_message_id IS NULL``
        guard covers the one exception: a Stop delivered before
        ``create_run`` committed would surface the PREVIOUS run here, and a
        finished run always has its anchor recorded by a finalizer terminal
        path -- so an already-anchored newest row means "record nothing"
        (row stays NULL -> ordinal fallback, the pre-fix behavior), never
        "overwrite a good anchor".

        Args:
            conversation_id: Durable conversation id whose runs to inspect.

        Returns:
            The newest non-superseded primary run's id when its
            ``assistant_message_id`` is still NULL, else ``None``.
        """
        record = self._db.latest_primary_run(conversation_id)
        if record is None or record.get("assistant_message_id") is not None:
            return None
        return record["id"]

    def subagent_count(self, conversation_id: str) -> int:
        return self._db.count_subagent_runs(conversation_id)

    def subagent_counts(self, conversation_ids: list[str]) -> dict[str, int]:
        """Batched per-conversation sub-agent counts (Finding A).

        One call replaces one ``subagent_count(cid)`` call per row -- see
        ``AgentRunsDB.count_subagents_by_conversation`` for the query.
        """
        return self._db.count_subagents_by_conversation(conversation_ids)

    def resume_marker_messages(
        self, conversation_id: str
    ) -> list[tuple[str | None, list[ConsoleChatMessage]]]:
        """Re-derive transcript TOOL marker messages from ``AgentRunsDB`` for resume.

        Plan-B final-review Medium-1: the rail (``historical_snapshot``) and
        the ``[N Sub-Agents]`` badge already re-derive from ``AgentRunsDB``
        on resume; the inline transcript TOOL markers did not -- they are
        only ever appended live via ``_append_marker`` with
        ``persist=False``, so a session rebuilt fresh from ChaChaNotes never
        sees them.

        Returns one ``(assistant_message_id, marker_block)`` pair per
        non-superseded PRIMARY run for the conversation, oldest run first
        (``list_runs`` itself returns newest-first, so the order is
        reversed here). ``assistant_message_id`` is the run's own
        ``record["assistant_message_id"]`` -- the persisted id of the
        reply it produced (set on every terminal path since Task 2), or
        ``None`` for a legacy/pre-Phase-C run, a sub-agent run, or one
        whose reply was never persisted -- Task 3's
        ``inject_resume_agent_markers`` is what turns this id into an
        anchored (or ordinal-fallback, or dropped) placement.

        Each block holds that run's own TOOL marker messages, in the run's
        recorded step order, built with ``format_agent_step_marker`` -- the
        same formatter the live bridge uses -- so a resumed transcript's
        markers are byte-identical to what the live run produced. A run
        with no marker-worthy steps (e.g. a plain answer, no
        tool/spawn/error step) yields an empty block; callers should skip
        those rather than inject nothing.

        Placement of the returned blocks into a transcript is the caller's
        job -- see ``inject_resume_agent_markers``.
        """
        records = [
            record
            for record in self._db.list_runs(conversation_id, include_superseded=False)
            if record["agent_kind"] == AGENT_KIND_PRIMARY
        ]
        records.reverse()  # list_runs is newest-first; markers must read chronologically
        blocks: list[tuple[str | None, list[ConsoleChatMessage]]] = []
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
                    block.append(
                        ConsoleChatMessage(
                            role=ConsoleMessageRole.TOOL,
                            content=text,
                            status="complete",
                        )
                    )
            blocks.append((record.get("assistant_message_id"), block))
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
                session_id, role=ConsoleMessageRole.TOOL, content=text
            )
        except KeyError:
            pass  # session vanished mid-run; the rail still has the live snapshot

    @staticmethod
    def _summarize(step: AgentStep) -> str:
        # Finding B: feeds only AgentLiveStep.text, which
        # _console_agent_section_lines renders into a markup=False Static --
        # escaping here (a second guard on top of markup=False) produced
        # literal backslashes for bracketed text. Left raw; the transcript
        # TOOL marker path (_append_marker) is also raw, since its consumers
        # never parse the text as markup either.
        raw = step.summary or step.result or step.tool_name or step.kind
        # task-350: mark truncation with an ellipsis + affordance instead of a
        # silent mid-word clip for the run inspector's live-step lines.
        return _truncate_step_text(str(raw), limit=_STEP_SUMMARY_LIMIT)

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
            (r for r in records if r["agent_kind"] == AGENT_KIND_PRIMARY), None
        )
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
        raw = (
            step.get("summary")
            or step.get("result")
            or step.get("tool_name")
            or step.get("kind")
            or ""
        )
        return str(raw)[:200]
