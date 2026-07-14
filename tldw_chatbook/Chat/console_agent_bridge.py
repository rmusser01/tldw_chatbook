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
from dataclasses import dataclass
from typing import Any, Callable

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY, AGENT_KIND_SUBAGENT, FIND_TOOLS_NAME, LOAD_TOOLS_NAME,
    RunBudget, SPAWN_TOOL_NAME, STEP_ERROR, STEP_SPAWN, STEP_TOOL_RESULT,
    AgentConfig, AgentStep, RunOutcome,
)
from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT, AgentService
from tldw_chatbook.Agents.agent_stream import StreamGate
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

CONSOLE_AGENT_OPERATING_PROMPT = (
    "You are a capable assistant with optional tools. Answer directly when no "
    "tool is needed. When a tool would help, call exactly one tool per reply "
    "using the fenced protocol described below, then continue once you have the "
    "result. Use spawn_subagent to delegate a self-contained sub-task to an "
    "isolated helper. Keep replies concise.")

_QUIET_STEP_TOOLS = {FIND_TOOLS_NAME, LOAD_TOOLS_NAME}


def compose_agent_system_prompt(session_prompt: str) -> str:
    """Compose the primary system prompt: session prompt first, agent prompt appended."""
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
    kind: str
    text: str
    agent_kind: str


@dataclass(frozen=True)
class SubAgentSummary:
    text: str
    status: str = "running"


@dataclass(frozen=True)
class AgentLiveSnapshot:
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
                  streaming=False, **_ignored) -> dict:
        is_subagent = self._is_subagent(messages_payload)
        gate = StreamGate()
        any_streamed = False

        async def _consume() -> None:
            nonlocal any_streamed
            async for chunk in self._gateway.stream_chat(self._resolution, messages_payload):
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
            # chunks on the same message.
            _visible, tool_call = gate.result()
            if tool_call is not None:
                self._store.reset_stream_content(self._assistant_message_id)
        return {"choices": [{"message": {"content": gate.full_text}}]}

    @staticmethod
    def _is_subagent(messages_payload) -> bool:
        if not messages_payload:
            return False
        first = messages_payload[0]
        return (first.get("role") == "system"
                and str(first.get("content", "")).startswith(SUBAGENT_SYSTEM_PROMPT))


class ConsoleAgentBridge:
    """Owns the tool registry + run store and runs one primary agent reply."""

    def __init__(self, *, agent_runs_db: AgentRunsDB, store,
                 provider_gateway, registry: ToolCatalogRegistry | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._db = agent_runs_db
        self._store = store
        self._gateway = provider_gateway
        self._clock = clock
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
                  supersede_previous: bool = False) -> RunOutcome:
        config = AgentConfig(
            model=model,
            system_prompt=compose_agent_system_prompt(session_system_prompt),
            allowed_tools=self._allowed_tools,
            budget=RunBudget())
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
            self._db, self._registry, chat_call=adapter.chat_call,
            clock=self._clock, on_step=on_step)

        supersede_run_id = (
            self._previous_primary_run_id(conversation_id) if supersede_previous else None)
        try:
            _run_id, outcome = service.run_turn(
                conversation_id=conversation_id, messages=agent_messages, config=config,
                api_endpoint=str(getattr(resolution, "provider", "") or "agent"),
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
