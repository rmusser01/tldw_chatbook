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
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
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
    """

    def __init__(self, *, store, provider_gateway, resolution, assistant_message_id,
                 should_cancel):
        self._store = store
        self._gateway = provider_gateway
        self._resolution = resolution
        self._assistant_message_id = assistant_message_id
        self._should_cancel = should_cancel

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

        # The service runs on a worker thread with no running loop → asyncio.run
        # is safe (same pattern as BuiltinToolProvider bridging async tools).
        asyncio.run(_consume())
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
        adapter = _StreamingModelAdapter(
            store=self._store, provider_gateway=self._gateway, resolution=resolution,
            assistant_message_id=assistant_message_id, should_cancel=should_cancel)

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
                    # Finding B: this text is only ever rendered into the
                    # rail's markup=False Statics (_console_agent_section_lines
                    # in chat_screen.py) -- never into a markup-enabled
                    # widget -- so it must stay raw. Escaping here produced
                    # literal backslashes once rendered (`fetch [docs]` ->
                    # `fetch \[docs]`), since markup=False never interprets
                    # (and so never "consumes") the escape sequence. The same
                    # is true of the transcript TOOL marker path below
                    # (_append_marker): both of its consumers
                    # (console_transcript.py's Content.assemble and
                    # chat_screen.py's Text(...)) render the text as-is
                    # rather than parsing it as markup, so _append_marker
                    # must stay raw too.
                    subagents.append(SubAgentSummary(step.summary or ""))
                    self._append_marker(
                        session_id, f"⤷ spawned sub-agent: {step.summary}")
                elif (step.kind == STEP_TOOL_RESULT
                      and step.tool_name not in _QUIET_STEP_TOOLS):
                    self._append_marker(
                        session_id, f"⚙ {step.tool_name} → {step.result}")
                elif step.kind == STEP_ERROR:
                    self._append_marker(session_id, f"⚠ {step.summary}")
            self._live[conversation_id] = AgentLiveSnapshot(
                status="running", step=len(live_steps),
                steps=tuple(live_steps[-5:]), subagents=tuple(subagents))

        service = AgentService(
            self._db, self._registry, chat_call=adapter.chat_call,
            clock=self._clock, on_step=on_step)

        supersede_run_id = (
            self._previous_primary_run_id(conversation_id) if supersede_previous else None)
        _run_id, outcome = service.run_turn(
            conversation_id=conversation_id, messages=agent_messages, config=config,
            api_endpoint=str(getattr(resolution, "provider", "") or "agent"),
            should_cancel=should_cancel, supersede_run_id=supersede_run_id)
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
